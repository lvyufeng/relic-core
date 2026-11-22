// Scale-aware paired-byte LUT decode with explicit AIV -> bounded GM ring -> L1 -> AIC ownership.
//
// This is an independent, opt-in Ascend910B prototype. It is not a production default and it does
// not change the checkpoint: the original FP4 code bytes, the original per-K32 E8M0 scale grid and
// the FP4 byte budget are all unchanged. No INT4 recode, no expanded BF16 weight bank.
//
// What differs from `online_tile_cube.cpp` is only how a decode tile is produced. That kernel
// reconstructs each original value with floating weight arithmetic: widen the code, select a scale
// factor, multiply in FP32, then narrow to BF16. Here the multiply has already happened offline:
//
//   code_lut[slot][packed_byte] -> uint32 holding TWO BF16 values
//
// where `slot = raw E8M0 scale - 117` for the checkpoint's measured global scale range 117..127,
// and `slot = 11` is a NaN pair. Both halves of the pair are the exact same BF16 bit pattern that
// `OriginalFp4Bits` produces in the accepted kernel, so the Cube still consumes identical weight
// bits in the same K order with the same M16 accumulation and the same FP32 output.
//
// Per K128 tile the AIV therefore does: widen the packed byte to a LUT byte offset, widen the scale
// byte to a slot byte offset, fan the slot offset out to one offset per pair, add, and gather. There
// is no FP32 weight value, no multiply and no BF16 narrowing on this path.
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

namespace {
constexpr uint32_t M = 16;
#ifndef RELIC_LUT_OUTPUT_ROWS
#define RELIC_LUT_OUTPUT_ROWS 32
#endif
constexpr uint32_t N = RELIC_LUT_OUTPUT_ROWS;
static_assert(N == 32 || N == 64,
              "bounded complete output tiles only; N32 is the first candidate, N64 the future one");
constexpr uint32_t SOURCE_N = 16;
constexpr uint32_t N_TILES = N / SOURCE_N;

// Fixed at four original K32 blocks: the LUT table geometry and the pair-repeat order are defined
// for K128 only, so a different grain would be a different table format, not a build option.
constexpr uint32_t GROUP_K = 4;
constexpr uint32_t K_BLOCK = 32;
constexpr uint32_t K = K_BLOCK * GROUP_K;

constexpr uint32_t SCALE_COUNT = N * GROUP_K;
constexpr uint32_t BYTES = N * K / 2;      // one byte holds two original E2M1 codes
constexpr uint32_t ELEMENTS = N * K;

// The checkpoint measurement this format is built for: every one of the 40 x 384 x 3 = 46,080
// matrices holds E8M0 bytes in 117..127 only. A byte outside that range maps to the poison slot,
// which is a NaN pair -- safe for any input, and never taken by this checkpoint.
constexpr uint32_t SCALE_MIN = 117;
constexpr uint32_t SCALE_MAX = 127;
constexpr uint32_t LUT_SLOTS = 12;         // 11 real scales plus one poison slot
constexpr uint32_t LUT_WORDS_PER_SLOT = 256;   // one word per packed byte
constexpr uint32_t LUT_BYTES_PER_SLOT = LUT_WORDS_PER_SLOT * sizeof(uint32_t);
constexpr uint32_t POISON_SLOT = 11;
constexpr uint32_t POISON_PAIR = 0x7fc07fc0u;
constexpr uint32_t LUT_WORDS = LUT_SLOTS * LUT_WORDS_PER_SLOT;

// Prepared immutable table word order. This order is the ABI the Python binding writes, so it is
// declared once here and queried through `..._table_words()` rather than assumed by the caller.
constexpr uint32_t TABLE_CODE = 0;                              // LUT_WORDS words
constexpr uint32_t TABLE_SMAP = TABLE_CODE + LUT_WORDS;          // 256 words
constexpr uint32_t TABLE_REPEAT = TABLE_SMAP + 256;              // BYTES words
constexpr uint32_t TABLE_WORDS = TABLE_REPEAT + BYTES;

// Geometry identity, so a binding can tell an N32 library from an N64 one without trusting a name.
constexpr uint32_t LAYOUT_ID = (N == 32) ? 1 : 2;

// Unified Buffer byte addresses. Every one of these is a byte address, not an element index; the
// vector operands below are constructed from them explicitly.
constexpr uint32_t CODE_OFF = 0;
constexpr uint32_t SMAP_OFF = CODE_OFF + LUT_WORDS * sizeof(uint32_t);
constexpr uint32_t REPT_OFF = SMAP_OFF + 256 * sizeof(uint32_t);
constexpr uint32_t RAW_OFF = REPT_OFF + BYTES * sizeof(uint32_t);
constexpr uint32_t SCALE_PAD = (SCALE_COUNT + 31) / 32 * 32;
constexpr uint32_t SCL_OFF = RAW_OFF + BYTES;
constexpr uint32_t CHALF_OFF = SCL_OFF + SCALE_PAD;
constexpr uint32_t CIDX_OFF = CHALF_OFF + BYTES * sizeof(half);
constexpr uint32_t SHALF_OFF = CIDX_OFF + BYTES * sizeof(int32_t);
constexpr uint32_t SIDX_OFF = SHALF_OFF + SCALE_COUNT * sizeof(half);
constexpr uint32_t SOFF_OFF = SIDX_OFF + SCALE_COUNT * sizeof(int32_t);
constexpr uint32_t ESOFF_OFF = SOFF_OFF + SCALE_COUNT * sizeof(uint32_t);
constexpr uint32_t OUT_OFF = ESOFF_OFF + BYTES * sizeof(int32_t);
constexpr uint32_t INVALID_OFF = OUT_OFF + 2 * ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t UB_BYTES = INVALID_OFF + N * sizeof(float);
static_assert(UB_BYTES <= 128 * 1024,
              "the complete Vector decode scratch must fit the Ascend910B Unified Buffer");

constexpr uint32_t L1_B_SLOT_BYTES = ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t L1_A_OFFSET = 2 * L1_B_SLOT_BYTES;
constexpr uint16_t READY_FLAG = 8;
constexpr uint16_t FREE_FLAG = 10;

// Reference form of one original value, kept here as the definition the prepared code LUT must
// reproduce bit for bit. It is the accepted kernel's helper (online_tile_cube.cpp:92-99), including
// its zero fold: the accepted INT8 x2 widening has a single integer zero, so both E2M1 zero codes
// decode to +0. The scale is applied by the LUT slot, so `scale` here is only used for the exponent
// delta; the table generator must call this with the real scale byte.
[[maybe_unused]] __aicore__ inline uint16_t OriginalFp4Bits(uint8_t code, uint8_t scale)
{
    if (scale < 2 || scale > 252) return 0x7fc0;
    const uint32_t magnitude = code & 7;
    if (magnitude == 0) return 0; // The accepted I8x2 widening folds -0.
    const int32_t delta = magnitude >= 6 ? 2 : (magnitude >= 4 ? 1 : (magnitude >= 2 ? 0 : -1));
    const uint32_t mantissa = (magnitude == 3 || magnitude == 5 || magnitude == 7) ? 0x40 : 0;
    return static_cast<uint16_t>(((code & 8) << 12) | ((scale + delta) << 7) | mantissa);
}

class TileVector {
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR s, GM_ADDR selected, GM_ADDR stage, GM_ADDR out,
                                GM_ADDR tables, uint32_t tiles, uint32_t blocks, uint32_t world,
                                uint32_t n)
    {
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(b));
        scales_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(s));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        n_ = n;
        // `tiles` is the number of source N16 banks (n / 16), the same value the accepted ABI
        // receives. One output tile of N rows consumes N_TILES of them, so the tile count this
        // kernel iterates is the derived one; storing the raw value would walk 2x or 4x too many
        // tiles and read past the bank.
        sourceTiles_ = tiles;
        tiles_ = tiles / N_TILES;
        blocks_ = blocks / GROUP_K;
        world_ = world;
        raw_ = LocalTensor<uint8_t>(TPosition::VECCALC, RAW_OFF, BYTES);
        scaleBytes_ = LocalTensor<uint8_t>(TPosition::VECCALC, SCL_OFF, SCALE_PAD);
        codeHalf_ = LocalTensor<half>(TPosition::VECCALC, CHALF_OFF, BYTES);
        codeIdx_ = LocalTensor<int32_t>(TPosition::VECCALC, CIDX_OFF, BYTES);
        scaleHalf_ = LocalTensor<half>(TPosition::VECCALC, SHALF_OFF, SCALE_COUNT);
        scaleIdx_ = LocalTensor<int32_t>(TPosition::VECCALC, SIDX_OFF, SCALE_COUNT);
        slotOff_ = LocalTensor<uint32_t>(TPosition::VECCALC, SOFF_OFF, SCALE_COUNT);
        expandedSlot_ = LocalTensor<uint32_t>(TPosition::VECCALC, ESOFF_OFF, BYTES);
        scaled_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, OUT_OFF, 2 * ELEMENTS);
        invalid_ = LocalTensor<float>(TPosition::VECCALC, INVALID_OFF, N);
        codeLut_ = LocalTensor<uint32_t>(TPosition::VECCALC, CODE_OFF, LUT_WORDS);
        scaleMap_ = LocalTensor<uint32_t>(TPosition::VECCALC, SMAP_OFF, 256);
        pairRepeat_ = LocalTensor<uint32_t>(TPosition::VECCALC, REPT_OFF, BYTES);

        if (tables != nullptr) {
            // Immutable decode metadata only: no expanded checkpoint weights and no routing values.
            tables_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tables));
            DataCopy(codeLut_, tables_[TABLE_CODE], LUT_WORDS);
            DataCopy(scaleMap_, tables_[TABLE_SMAP], 256);
            DataCopy(pairRepeat_, tables_[TABLE_REPEAT], BYTES);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        } else {
            // Defensive path only. Without the prepared table nothing can decode exactly, so every
            // entry becomes the NaN pair and every scale maps to the poison slot. The result is a
            // NaN tile rather than undefined Unified Buffer contents.
            union { uint32_t bits; float value; } poison;
            poison.bits = POISON_PAIR;
            Duplicate(codeLut_.ReinterpretCast<float>(), poison.value, LUT_WORDS);
            Duplicate(scaleMap_.ReinterpretCast<int32_t>(),
                      static_cast<int32_t>(POISON_SLOT * LUT_BYTES_PER_SLOT), 256);
            Duplicate(pairRepeat_.ReinterpretCast<int32_t>(), 0, BYTES);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stage));
        stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void Process(uint32_t groups, uint32_t owned)
    {
        const uint32_t owner = GetBlockIdx();
        for (uint32_t task = owner; task < groups * tiles_; task += GetBlockNum()) {
            const uint32_t group = task / tiles_;
            const uint32_t tile = task % tiles_;
            const int32_t expert = selected_.GetValue(group);
            if (expert < 0 || static_cast<uint32_t>(expert / world_) >= owned) {
                union { uint32_t bits; float value; } fill;
                fill.bits = expert == -1 ? 0 : 0x7fc00000;
                Duplicate(invalid_, fill.value, N);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(output_[static_cast<uint64_t>(group) * n_ + tile * N], invalid_, N);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                continue;
            }
            const uint32_t local = expert / world_;
            const uint64_t base = (static_cast<uint64_t>(local) * tiles_ + tile) * blocks_;
            for (uint32_t block = 0; block < blocks_; ++block) {
                const uint32_t slot = block & 1;
                if (block >= 2) CrossCoreWaitFlag<0x2>(FREE_FLAG + slot);
                // The offline compact layout already matches Mmad K16-major/N16-minor B, and the
                // scale bank stays one original byte per N row and K32 block.
                const uint64_t source = base + block;
                DataCopy(raw_, packed_[source * BYTES], BYTES);
                DataCopyPad(scaleBytes_, scales_[source * SCALE_COUNT],
                            DataCopyExtParams{1, SCALE_COUNT, 0, 0, 0},
                            DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                // Packed byte -> byte offset of that byte's word inside a LUT slot. The cast is the
                // proven A2 unsigned widening (uint8 -> half -> int32); it only ever moves an index,
                // never a weight value, so no rounding can appear in the decoded result.
                Cast(codeHalf_, raw_, RoundMode::CAST_NONE, BYTES);
                PipeBarrier<PIPE_V>();
                Cast(codeIdx_, codeHalf_, RoundMode::CAST_RINT, BYTES);
                PipeBarrier<PIPE_V>();
                ShiftLeft(codeIdx_.ReinterpretCast<uint32_t>(), codeIdx_.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(2), BYTES);   // one word per packed byte
                PipeBarrier<PIPE_V>();

                // Original scale byte -> byte offset of its LUT slot. The prepared map already folds
                // the measured 117..127 window and the poison slot, so an out-of-range byte can only
                // ever reach NaN pairs.
                Cast(scaleHalf_, scaleBytes_, RoundMode::CAST_NONE, SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                Cast(scaleIdx_, scaleHalf_, RoundMode::CAST_RINT, SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                ShiftLeft(scaleIdx_.ReinterpretCast<uint32_t>(), scaleIdx_.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(2), SCALE_COUNT);   // one word per scale byte
                PipeBarrier<PIPE_V>();
                Gather(slotOff_, scaleMap_, scaleIdx_.ReinterpretCast<uint32_t>(),
                       static_cast<uint32_t>(0), SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                // One scale covers a whole K32 row, i.e. sixteen packed bytes. The prepared repeat
                // table expands that to one offset per pair in Cube K16-major/N16-minor order, so
                // the decoded tile needs no separate N32-to-panel rearrangement.
                Gather(expandedSlot_, slotOff_, pairRepeat_, static_cast<uint32_t>(0), BYTES);
                PipeBarrier<PIPE_V>();

                Add(codeIdx_, codeIdx_, expandedSlot_.ReinterpretCast<int32_t>(), BYTES);
                PipeBarrier<PIPE_V>();
                // The gather writes the two exact scaled BF16 values of each byte straight into one
                // of the two BF16 output slots; there is no intermediate value tensor to copy.
                Gather(scaled_[slot * ELEMENTS].ReinterpretCast<uint32_t>(), codeLut_,
                       codeIdx_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), BYTES);
                PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(stage_[owner * 2 * ELEMENTS + slot * ELEMENTS], scaled_[slot * ELEMENTS],
                         ELEMENTS);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(READY_FLAG + slot);
                // The decoded BF16 occupies two UB slots, and FREE fences them and the GM ring for
                // reuse after two blocks. The single raw/scale UB slot is a different lifetime: the
                // next iteration's MTE2 would overwrite it while this iteration's vector decode may
                // still be reading it, so the fence here must be Vector-before-MTE2. Fencing the
                // other direction (MTE2_V) would leave that hazard open.
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            const uint32_t drains = blocks_ < 2 ? blocks_ : 2;
            for (uint32_t i = 0; i < drains; ++i) {
                const uint32_t block = blocks_ - drains + i;
                CrossCoreWaitFlag<0x2>(FREE_FLAG + (block & 1));
            }
        }
    }

private:
    LocalTensor<uint8_t> raw_, scaleBytes_;
    LocalTensor<half> codeHalf_, scaleHalf_;
    LocalTensor<int32_t> codeIdx_, scaleIdx_;
    LocalTensor<uint32_t> slotOff_, expandedSlot_, codeLut_, scaleMap_, pairRepeat_;
    LocalTensor<bfloat16_t> scaled_;
    LocalTensor<float> invalid_;
    GlobalTensor<bfloat16_t> stage_;
    GlobalTensor<uint8_t> packed_, scales_;
    GlobalTensor<int32_t> selected_;
    GlobalTensor<float> output_;
    GlobalTensor<uint32_t> tables_;
    uint32_t tiles_, sourceTiles_, blocks_, world_, n_;
};

class TileCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR selected, GM_ADDR out, GM_ADDR stage,
                                uint32_t tiles, uint32_t blocks, uint32_t world, uint32_t n,
                                bool shared, uint32_t groups)
    {
        activation_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stage));
        stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        // Same derivation as the Vector side: the ABI hands over the source N16 bank count.
        sourceTiles_ = tiles;
        tiles_ = tiles / N_TILES;
        blocks_ = blocks / GROUP_K; world_ = world; n_ = n; shared_ = shared;
        sharedB_ = LocalTensor<bfloat16_t>(TPosition::B1, 0, 2 * ELEMENTS);
        sharedA_ = LocalTensor<bfloat16_t>(TPosition::A1, L1_A_OFFSET, M * K);
        a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, M * K);
        b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, ELEMENTS);
        c_ = LocalTensor<float>(TPosition::CO1, 0, M * N);
    }

    __aicore__ inline void Process(uint32_t groups, uint32_t owned)
    {
        const uint32_t owner = GetBlockIdx();
        for (uint32_t task = owner; task < groups * tiles_; task += GetBlockNum()) {
            const uint32_t group = task / tiles_;
            const uint32_t tile = task % tiles_;
            const int32_t expert = selected_.GetValue(group);
            if (expert < 0 || static_cast<uint32_t>(expert / world_) >= owned) continue;
            const uint64_t xBase = shared_ ? 0 : static_cast<uint64_t>(group) * blocks_ * K;
            for (uint32_t block = 0; block < blocks_; ++block) {
                CrossCoreWaitFlag<0x2>(READY_FLAG + (block & 1));
                DataCopy(sharedB_[(block & 1) * ELEMENTS],
                         stage_[owner * 2 * ELEMENTS + (block & 1) * ELEMENTS], ELEMENTS);
                CrossCoreSetFlag<0x2, PIPE_MTE2>(FREE_FLAG + (block & 1));
                Nd2NzParams convert;
                convert.ndNum = 1;
                convert.nValue = 1;
                convert.dValue = K;
                convert.srcNdMatrixStride = 0;
                convert.srcDValue = K;
                convert.dstNzC0Stride = M;
                convert.dstNzNStride = 1;
                convert.dstNzMatrixStride = 0;
                DataCopy(sharedA_, activation_[xBase + block * K], convert);
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LoadData2DParams load;
                load.startIndex = 0;
                load.repeatTimes = K / 16;
                load.srcStride = 1;
                load.dstGap = 0;
                load.ifTranspose = false;
                // Exactly one A1 slot exists (L1_A_OFFSET, M*K) and the DataCopy above always
                // writes it at base 0, so the L0A load is always at base 0 too. There is no
                // per-group activation base here: `shared` selects which GM activation row is
                // read (row 0 for a shared launch, otherwise this group's row), not a second L1
                // slot. Indexing this by `group` would read past the one allocation whenever
                // shared is false and group is non-zero.
                LoadData(a_, sharedA_, load);
                // The B1 ring still contains only the current K tile, not the complete K dimension.
                // Mmad consumes the complete K16-major/N16-minor B tile, in native block order.
                load.startIndex = 0;
                load.repeatTimes = (K / 16) * N_TILES;
                LoadData(b_, sharedB_[(block & 1) * ELEMENTS], load);
                // MTE2 must not replace the single activation L1 slot before MTE1 has loaded its
                // previous value into L0A.
                SetFlag<HardEvent::MTE1_MTE2>(0);
                WaitFlag<HardEvent::MTE1_MTE2>(0);
                SetFlag<HardEvent::MTE1_M>(0);
                WaitFlag<HardEvent::MTE1_M>(0);
                MmadParams params;
                params.m = M; params.n = N; params.k = K;
                params.cmatrixInitVal = block == 0;
                params.cmatrixSource = false;
                Mmad(c_, a_, b_, params);
                SetFlag<HardEvent::M_MTE1>(0);
                WaitFlag<HardEvent::M_MTE1>(0);
            }
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 fix;
            fix.nSize = N; fix.mSize = 1; fix.srcStride = M;
            fix.dstStride = n_; fix.quantPre = QuantMode_t::NoQuant;
            Fixpipe<float, float, CFG_ROW_MAJOR>(
                output_[static_cast<uint64_t>(group) * n_ + tile * N], c_, fix);
            SetFlag<HardEvent::FIX_M>(0);
            WaitFlag<HardEvent::FIX_M>(0);
        }
    }

private:
    LocalTensor<bfloat16_t> sharedA_, sharedB_, a_, b_;
    LocalTensor<float> c_;
    GlobalTensor<bfloat16_t> activation_, stage_;
    GlobalTensor<int32_t> selected_;
    GlobalTensor<float> output_;
    uint32_t tiles_, sourceTiles_, blocks_, world_, n_;
    bool shared_;
};
} // namespace

extern "C" __global__ __aicore__ void relic_scale_lut_tile_cube(
    GM_ADDR x, GM_ADDR b, GM_ADDR s, GM_ADDR selected, GM_ADDR out, GM_ADDR stage, GM_ADDR tables,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIV {
        TileVector vector;
        vector.Init(b, s, selected, stage, out, tables, tiles, blocks, world, n);
        vector.Process(groups, owned);
    } else {
        TileCube cube;
        cube.Init(x, selected, out, stage, tiles, blocks, world, n, shared != 0, groups);
        cube.Process(groups, owned);
    }
}

// Queried geometry and ABI. A binding must read these instead of assuming a tile shape.
extern "C" uint32_t relic_scale_lut_tile_cube_output_rows()
{
    return N;
}

extern "C" uint32_t relic_scale_lut_tile_cube_layout_id()
{
    return LAYOUT_ID;
}

extern "C" uint32_t relic_scale_lut_tile_cube_k_blocks_per_tile()
{
    return GROUP_K;
}

// The same value under the second spelling the binding may use. Both are literal geometry.
extern "C" uint32_t relic_scale_lut_tile_cube_group_k()
{
    return GROUP_K;
}

extern "C" uint32_t relic_scale_lut_tile_cube_table_words()
{
    return TABLE_WORDS;
}

extern "C" uint32_t relic_scale_lut_tile_cube_scale_min()
{
    return SCALE_MIN;
}

extern "C" uint32_t relic_scale_lut_tile_cube_scale_max()
{
    return SCALE_MAX;
}

extern "C" uint32_t relic_scale_lut_tile_cube_prepared_tables()
{
    return 1;
}

// Established repo spelling, matching the accepted compact and paired consumers. Exactly one
// unsigned query is exported so a binder that requires a single match cannot see two.
extern "C" uint32_t relic_scale_lut_tile_cube_unsigned_byte_cast()
{
    return 1;
}

extern "C" uint32_t relic_scale_lut_tile_cube_lut_slots()
{
    return LUT_SLOTS;
}

extern "C" uint32_t relic_scale_lut_tile_cube_poison_slot()
{
    return POISON_SLOT;
}

// BF16 elements per ring slot, i.e. rows * 128. The ring is [block_dim, 2, this].
extern "C" uint32_t relic_scale_lut_tile_cube_stage_elements()
{
    return ELEMENTS;
}

extern "C" void relic_scale_lut_tile_cube_launch_tables(uint32_t blockDim, void *stream,
    void *x, void *b, void *s, void *selected, void *out, void *stage, void *tables,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    relic_scale_lut_tile_cube<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(b), reinterpret_cast<GM_ADDR>(s),
        reinterpret_cast<GM_ADDR>(selected), reinterpret_cast<GM_ADDR>(out),
        reinterpret_cast<GM_ADDR>(stage), reinterpret_cast<GM_ADDR>(tables),
        tiles, blocks, groups, owned, world, n, shared);
}
