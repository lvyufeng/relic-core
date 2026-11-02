// Fused FP8 expert SwiGLU for decode (M == 1).
//
// The runtime's current FP8 expert path materialises a dense bf16 copy of the
// weight before every GEMV: `_expert_matrix` calls `FP8Tensor.dequantize`, which
// does `code.to(bf16)` and then a four-dimensional view/broadcast multiply over
// the 128x128 scale grid, producing a full [rows, cols] bf16 tensor that is read
// once by two `F.linear` calls and thrown away. Measured on an RTX 3090
// (sm_86, torch 2.11) that costs 124.0 us per expert, of which 72.9 us is the
// dequant -- 59% -- and 480 expert visits a token puts the dequant alone at
// 35 ms of a 145 ms decode step.
//
// This kernel removes the materialisation entirely. Each thread block owns one
// expert and reads the e4m3 codes directly, widening them in registers and
// applying the block scale as it goes. The weight never exists in bf16 anywhere.
//
// Why a GEMV and not a tensor-core GEMM: at decode M == 1, so a wmma tile
// (m16n8k16) would spend 15 of its 16 rows on nothing while reading exactly the
// same weight bytes. The op is memory-bound -- 4.92 MB of fp8 against a ~936 GB/s
// HBM -- so the target is bandwidth, not FLOPs. `qwen4_exp_moe_prefill_bf16_forward`
// remains the right path for prefill, where M is large enough to use them.
//
// Arithmetic: the block rule is the one `relicllm.models.qwen4_exp.quant`
// implements and the checkpoint was produced with -- value = code * scale, one
// bf16 scale per 128x128 tile. Widening e4m3 to bf16 is exact (every e4m3 value
// has an exact bf16 representation: 4 exponent bits are a subset of bf16's 8, and
// 3 mantissa bits a subset of its 7), and the multiply is done in the destination
// dtype exactly as `dequantize_block_fp8` does it, so this kernel reproduces that
// function rather than approximating it.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int kBlockRows = 128;
constexpr int kBlockCols = 128;
// The slot kernels' launch geometry: eight warps a block, the shape `fp8_expert_act_kernel` and the
// down projection are launched with, and the row step of the down kernel now that it sits outside
// the anonymous namespace beside them.
constexpr int kExpertWarps = 8;
constexpr int kLanes = 32;
constexpr int kThreads = 256;

// A packed pair of e4m3 codes to fp16 by hardware where it exists, else a field shift and rebias
// (see `qwen4_exp_fp8_decode_moe.cu` for the instruction the target lacks). The shift is exact:
// e4m3's fields are a subset of fp16's, and subnormals and zero survive it.
__device__ __forceinline__ __half2 e4m3x2_to_half2(uint32_t q) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q), __NV_E4M3);
#else
    constexpr uint32_t kSignMask = 0x80008000u;
    constexpr uint32_t kMagMask = 0x7F007F00u;
    const uint32_t shifted = (q & kSignMask) | ((q & kMagMask) >> (8 - 4));
    const __half2 v = *reinterpret_cast<const __half2*>(&shifted);
    constexpr uint32_t kRebias = 0x6400u;  // fp16 bits of 2^8
    const __half2 rebias = *reinterpret_cast<const __half2*>(&kRebias);
    return __hmul2(v, rebias);
#endif
}

__device__ __forceinline__ float bf16(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int kExpertThreads = kExpertWarps * 32;

// e4m3 -> bf16, TWO codes at a time, by bit manipulation.
//
// This is the same trick chitu (and vLLM's marlin) use, and it is the whole
// reason the earlier per-byte version lost to torch: converting one code at a
// time costs roughly six integer ops per byte, which at 100 GB/s of weight
// traffic is more arithmetic than the GEMV itself. e4m3's fields are a strict
// subset of bf16's -- 4 exponent bits under 8, 3 mantissa bits under 7 -- so the
// widening is a pure shift, and two codes ride in one 32-bit register.
//
// Layout of a pair in `q` (little-endian: byte 0 in the low half):
//   [ sign a | exp a(4) | mant a(3) | sign b | exp b(4) | mant b(3) ]
//       31      30..27    26..24        23      22..19     18..16
// bf16 wants sign at 15, exp at 14..7, mantissa at 6..0, so:
//   sign: 0x80008000 stays put
//   rest: 0x7F007F00 >> 4 puts exp and mantissa where bf16 wants them
// That leaves e4m3's exponent bias (7) in place of bf16's (127), which is one
// multiply by 2^120 -- exact for normals, and correct for subnormals and zero,
// whose exponent field is 0 and stays 0 (a bf16 subnormal scaled by 2^120 is the
// right magnitude). Verified against an exact per-code conversion for all 256
// codes: the only disagreement is the two NaN encodings (0x7F/0xFF), which the
// shift turns into a finite value, so they are handled explicitly below.
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf162(uint32_t q) {
    constexpr uint32_t kSignMask = 0x80008000u;
    constexpr uint32_t kMagMask = 0x7F007F00u;
    constexpr int kRightShift = 8 - 4;  // bf16 exponent bits - e4m3 exponent bits
    const uint32_t shifted = (q & kSignMask) | ((q & kMagMask) >> kRightShift);
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&shifted);
    // Rebias 7 -> 127 in one exact multiply, both halves at once. 2^120 is a
    // power of two, so the multiply changes only the exponent field.
    const uint32_t kRebias = 0x7b80u;  // bf16 bits of 2^120
    const __nv_bfloat162 rebias = *reinterpret_cast<const __nv_bfloat162*>(&kRebias);
    return __hmul2(v, rebias);
}

// The scalar e4m3 conversion the packed helper is checked against. Retained
// because a single tail element on an odd row length still needs it.
__device__ __forceinline__ float e4m3_to_float_exact(uint8_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x80u) << 24;
    const uint32_t exponent = (bits >> 3) & 0xFu;
    const uint32_t mantissa = bits & 0x7u;
    if (exponent == 0) {
        if (mantissa == 0) {
            return __uint_as_float(sign);
        }
        const float m = static_cast<float>(mantissa) / 8.0f;
        const float v = m * 0.015625f;
        return sign ? -v : v;
    }
    if (exponent == 0xFu && mantissa == 0x7u) {
        return __int_as_float(0x7fc00000);
    }
    const uint32_t out = sign | ((exponent - 7u + 127u) << 23) | (mantissa << 20);
    return __uint_as_float(out);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x80u) << 24;
    const uint32_t exponent = (bits >> 3) & 0xFu;
    const uint32_t mantissa = bits & 0x7u;
    if (exponent == 0) {
        if (mantissa == 0) {
            return __uint_as_float(sign);  // +/-0
        }
        const float m = static_cast<float>(mantissa) / 8.0f;
        const float v = m * 0.015625f;  // 2^-6
        return sign ? -v : v;
    }
    if (exponent == 0xFu && mantissa == 0x7u) {
        return __int_as_float(0x7fc00000);  // NaN, matching e4m3fn
    }
    const uint32_t out = sign | ((exponent - 7u + 127u) << 23) | (mantissa << 20);
    return __uint_as_float(out);
}

// One WARP per output row: the 32 lanes stride the reduction index together, so
// a warp's loads are consecutive and the cache lines are fully used.
//
// The obvious shape -- one thread per output element, each walking the reduction
// serially -- is what the first draft did, and it is catastrophic here: adjacent
// threads differ by one weight ROW, which is `cols` bytes apart, so a warp's
// loads land one byte in each of 32 different cache lines and the read runs at
// ~1/128 of bandwidth. Measured at 1 GB/s against 936 GB/s available, and 30x
// slower than the dequant path it was meant to replace.
//
// With a warp per row and `WARP_BATCH` rows per block, lane L reads columns
// L, L+32, L+64, ... of its row. The lanes are then reduced with a shuffle.
//
// `LANES` is fixed at 32 and the weight row length `rows` is walked in strides of
// 32 e4m3 codes, which a warp-fetch of 32 bytes covers in one transaction.
__global__ void fp8_gemv_bf16_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int cols,
    int scale_cols) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    // One warp owns TWO adjacent output columns and walks the reduction together:
    // the weight is [K, out] row-major, so two adjacent columns are two adjacent
    // BYTES and ride in one packed 32-bit read. Both halves of the pair share the
    // same 128-wide column tile (2t and 2t+1 differ by one, and the tile boundary
    // is at an even offset), so one scale lookup serves both.
    const int col2 = warp * 2;
    if (col2 >= cols) {
        return;
    }
    const bool has_second = (col2 + 1) < cols;
    const int scale_row = (col2 / kBlockRows) * scale_cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int row = lane; row < rows; row += 32) {
        const float s = __bfloat162float(scale[scale_row + row / kBlockCols]);
        const float xv = __bfloat162float(x[row]);
        const uint8_t* p = weight + static_cast<int64_t>(row) * cols + col2;
        if (has_second) {
            // Read as uchar2, NOT uint32: `col2` is only guaranteed even, so a
            // 4-byte load would be misaligned on odd warps and fault with
            // cudaErrorMisalignedAddress. Two bytes is the aligned unit here.
            const uchar2 raw = *reinterpret_cast<const uchar2*>(p);
            const uint32_t pair = static_cast<uint32_t>(raw.x) |
                                  (static_cast<uint32_t>(raw.y) << 8);
            const __nv_bfloat162 w = e4m3x2_to_bf162(pair);
            acc0 = fmaf(__bfloat162float(w.x) * s, xv, acc0);
            acc1 = fmaf(__bfloat162float(w.y) * s, xv, acc1);
        } else {
            acc0 = fmaf(e4m3_to_float_exact(*p) * s, xv, acc0);
        }
    }

    // Warp reduction.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0) {
        out[col2] = __float2bfloat16(acc0);
        if (has_second) {
            out[col2 + 1] = __float2bfloat16(acc1);
        }
    }
}


// Phase 1: the gate and up halves for every intermediate index, into the slot's scratch row.
// Grid: (slots,) with one block each, so this is one dispatch for the whole top-k.
__global__ void fp8_expert_act_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ gate_up,        // [E, 2*inter, hidden]
    const __nv_bfloat16* __restrict__ scale,    // [E, ceil(2*inter/128), ceil(hidden/128)]
    __nv_bfloat16* __restrict__ act,            // [slots, inter]
    const int* __restrict__ expert,             // [slots]
    int scale_cols,
    int hidden,
    int inter) {
    const int slot = blockIdx.y;
    const int row_block = blockIdx.x;
    const int e = __ldg(expert + slot);
    if (e < 0) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // One block owns a slice of the expert's `inter` output rows, so the top-k spreads over
    // `gridDim.x * slots` blocks instead of `slots`.  Ten blocks is an eighth of a 3090's
    // 82 SMs; the rows a block walked serially with eight warps were the whole of its cost.
    // Each block takes a contiguous slice of rows, sized so that the slices tile `inter` with no
    // row claimed twice and none left out.  The slice is rounded up to a whole number of warps'
    // strides, which is what keeps the warp loop inside a block covering exactly its own rows.
    const int warps_in_block = static_cast<int>(blockDim.x) >> 5;
    const int blocks = static_cast<int>(gridDim.x);
    const int per_block = (inter + blocks - 1) / blocks;
    const int row_begin = row_block * per_block;
    const int row_end = min(row_begin + per_block, inter);
    const auto* gu = gate_up + static_cast<int64_t>(e) * 2 * inter * hidden;
    // An expert's scale grid: ceil(2*inter / 128) rows, `scale_cols` columns.
    const int64_t scale_stride = static_cast<int64_t>((2 * inter + kBlockRows - 1) / kBlockRows) * scale_cols;
    const auto* sc = scale + static_cast<int64_t>(e) * scale_stride;
    for (int j = row_begin + warp; j < row_end; j += kExpertWarps) {
        // Row j is the gate half, row inter + j the up half; the two sit in different 128-row tiles,
        // so their scales are looked up separately. The 32 lanes read 32 consecutive bytes of a row.
        const int gate_scale_row = (j / kBlockRows) * scale_cols;
        const int up_scale_row = ((inter + j) / kBlockRows) * scale_cols;
        // Read the shard's plane layout directly: one table row is `hidden` codes scattered
        // across `hidden/16` planes, each 16 bytes wide.  Addressing it here is what keeps a
        // step from unpacking the layer back into rows first, which on a 3090 costs about a
        // millisecond a layer -- more than the layer's own arithmetic -- for a copy that is
        // pure rearrangement of the same bytes.
        // A lane owns one 16-byte chunk of the row -- one plane -- and reads it whole.  The
        // plane layout is what makes that contiguous: chunk `c` of row `r` starts at
        // `c * 2*inter*16 + r*16`, so a lane's 16 codes are adjacent, and the 32 lanes of a warp
        // cover 32 chunks = 512 columns in one coalesced sweep per round.
        const int64_t plane_row = static_cast<int64_t>(2 * inter) * 16;
        const int64_t grow = static_cast<int64_t>(j) * 16;
        const int64_t urow = static_cast<int64_t>(inter + j) * 16;
        constexpr int kChunksPerRow = 32;
        const int chunks = hidden >> 4;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int c = lane; c < chunks; c += kChunksPerRow) {
            const int col = c << 4;
            const uint4 gc = *reinterpret_cast<const uint4*>(gu + static_cast<int64_t>(c) * plane_row + grow);
            const uint4 uc = *reinterpret_cast<const uint4*>(gu + static_cast<int64_t>(c) * plane_row + urow);
            const uint8_t* gp = reinterpret_cast<const uint8_t*>(&gc);
            const uint8_t* up = reinterpret_cast<const uint8_t*>(&uc);
            const int scale_col = col / kBlockCols;
            const float gs = __bfloat162float(sc[gate_scale_row + scale_col]);
            const float us = __bfloat162float(sc[up_scale_row + scale_col]);
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const float xv = __bfloat162float(x[col + e]);
                gate_acc = fmaf(e4m3_to_float_exact(gp[e]) * gs, xv, gate_acc);
                up_acc = fmaf(e4m3_to_float_exact(up[e]) * us, xv, up_acc);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
            up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
        }
        if (lane == 0) {
            // `silu` spelled exactly as `qwen4_exp_hc_silu_bf16_kernel` spells it.
            const float silu = gate_acc / (1.0f + expf(-gate_acc));
            act[static_cast<int64_t>(slot) * inter + j] = __float2bfloat16(silu * up_acc);
        }
    }
}

// A slot's SwiGLU activation for M == 1, one block a slot, from 16-byte loads.
//
//   act[j] = silu(gate_up[j] . x) * (gate_up[inter + j] . x)
//
// The row pattern is the plain down kernel's: a warp owns output row `j`, the 32 lanes hold codes
// [lane * 16, lane * 16 + 16) of it in one load, decode them by hardware and reduce them with the
// staged activation.  The gate and up rows are two rows of the table whose 512-code window lies in
// the same four 128-column scale blocks, so the window's four column scales are the lane-independent
// part; the two rows' scales differ because they are in different 128-row tiles of the scale grid.
//
// One block an expert is what the per-code kernel beside it does, and the reason it is not split
// is that the gate and up halves are one reduction each: a block can fill the whole `act` row only
// by walking the whole of `hidden`.
__global__ void fp8_expert_act_fast_kernel(
    const __nv_bfloat16* __restrict__ x,         // [hidden]
    const uint8_t* __restrict__ gate_up,         // [E, 2*inter, hidden], e4m3
    const __nv_bfloat16* __restrict__ scale,     // [E, ceil(2*inter/128), ceil(hidden/128)]
    __nv_bfloat16* __restrict__ act,             // [slots, inter]
    const int* __restrict__ expert,              // [slots], -1 for none
    int hidden,
    int inter,
    int scale_cols) {
    const int slot = blockIdx.x;
    const int e = __ldg(expert + slot);
    if (e < 0) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // The activation is staged once: every warp reads the same `x`, so a shared copy is the load.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs = reinterpret_cast<__half2*>(smem_raw);  // [hidden / 2] fp16 pairs
    for (int j = threadIdx.x; j < hidden / 2; j += blockDim.x) {
        const __nv_bfloat162 b = reinterpret_cast<const __nv_bfloat162*>(x)[j];
        xs[j] = __floats2half2_rn(bf16(b.x), bf16(b.y));
    }
    __syncthreads();
    const auto* gu = gate_up + static_cast<int64_t>(e) * 2 * inter * hidden;
    const int64_t scale_stride =
        static_cast<int64_t>((2 * inter + kBlockRows - 1) / kBlockRows) * scale_cols;
    const auto* sc = scale + static_cast<int64_t>(e) * scale_stride;
    for (int j = warp; j < inter; j += kThreads / 32) {
        const int64_t grow_sc = static_cast<int64_t>(j / kBlockRows) * scale_cols;
        const int64_t urow_sc = static_cast<int64_t>((inter + j) / kBlockRows) * scale_cols;
        const uint8_t* grow = gu + static_cast<int64_t>(j) * hidden;
        const uint8_t* urow = gu + static_cast<int64_t>(inter + j) * hidden;
        float ga = 0.0f;
        float ua = 0.0f;
        for (int c0 = 0; c0 < hidden; c0 += kLanes * 16) {
            const int c = c0 + lane * 16;
            if (c < hidden) {
                const int col = c / kBlockCols;
                const float gs = bf16(__ldg(&sc[grow_sc + col]));
                const float us = bf16(__ldg(&sc[urow_sc + col]));
                const uint4 gq = *reinterpret_cast<const uint4*>(grow + c);
                const uint4 uq = *reinterpret_cast<const uint4*>(urow + c);
                const __half2 g0 = e4m3x2_to_half2(gq.x & 0xFFFFu);
                const __half2 g1 = e4m3x2_to_half2(gq.x >> 16);
                const __half2 g2 = e4m3x2_to_half2(gq.y & 0xFFFFu);
                const __half2 g3 = e4m3x2_to_half2(gq.y >> 16);
                const __half2 g4 = e4m3x2_to_half2(gq.z & 0xFFFFu);
                const __half2 g5 = e4m3x2_to_half2(gq.z >> 16);
                const __half2 g6 = e4m3x2_to_half2(gq.w & 0xFFFFu);
                const __half2 g7 = e4m3x2_to_half2(gq.w >> 16);
                const __half2 u0 = e4m3x2_to_half2(uq.x & 0xFFFFu);
                const __half2 u1 = e4m3x2_to_half2(uq.x >> 16);
                const __half2 u2 = e4m3x2_to_half2(uq.y & 0xFFFFu);
                const __half2 u3 = e4m3x2_to_half2(uq.y >> 16);
                const __half2 u4 = e4m3x2_to_half2(uq.z & 0xFFFFu);
                const __half2 u5 = e4m3x2_to_half2(uq.z >> 16);
                const __half2 u6 = e4m3x2_to_half2(uq.w & 0xFFFFu);
                const __half2 u7 = e4m3x2_to_half2(uq.w >> 16);
                const __half2* xv = xs + c / 2;
                const __half2 p0 = __hmul2(g0, xv[0]);
                const __half2 p1 = __hmul2(g1, xv[1]);
                const __half2 p2 = __hmul2(g2, xv[2]);
                const __half2 p3 = __hmul2(g3, xv[3]);
                const __half2 p4 = __hmul2(g4, xv[4]);
                const __half2 p5 = __hmul2(g5, xv[5]);
                const __half2 p6 = __hmul2(g6, xv[6]);
                const __half2 p7 = __hmul2(g7, xv[7]);
                const __half2 q0 = __hmul2(u0, xv[0]);
                const __half2 q1 = __hmul2(u1, xv[1]);
                const __half2 q2 = __hmul2(u2, xv[2]);
                const __half2 q3 = __hmul2(u3, xv[3]);
                const __half2 q4 = __hmul2(u4, xv[4]);
                const __half2 q5 = __hmul2(u5, xv[5]);
                const __half2 q6 = __hmul2(u6, xv[6]);
                const __half2 q7 = __hmul2(u7, xv[7]);
                const __half2 gt = __hadd2(__hadd2(__hadd2(p0, p1), __hadd2(p2, p3)),
                                           __hadd2(__hadd2(p4, p5), __hadd2(p6, p7)));
                const __half2 ut = __hadd2(__hadd2(__hadd2(q0, q1), __hadd2(q2, q3)),
                                           __hadd2(__hadd2(q4, q5), __hadd2(q6, q7)));
                ga = fmaf(__half2float(gt.x) + __half2float(gt.y), gs, ga);
                ua = fmaf(__half2float(ut.x) + __half2float(ut.y), us, ua);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            ga += __shfl_down_sync(0xffffffffu, ga, offset);
            ua += __shfl_down_sync(0xffffffffu, ua, offset);
        }
        if (lane == 0) {
            const float silu = ga / (1.0f + expf(-ga));
            act[static_cast<int64_t>(slot) * inter + j] = __float2bfloat16(silu * ua);
        }
    }
}


}

// Phase 2: out[slot] = sum_j down[row, j] * act[slot, j], one warp per output row, `blocks` blocks
// per slot tiling the hidden axis. Grid: (blocks, slots).
//
// A sentinel slot (`expert < 0`) writes zeros: the caller masked its weight, but a zeroed row here
// keeps `out` a pure sum over the route and its row of the table (the sentinel has none).
__global__ void fp8_expert_down_kernel(
    const uint8_t* __restrict__ down,           // [E, hidden, inter]
    const __nv_bfloat16* __restrict__ scale,    // [E, ceil(hidden/128), ceil(inter/128)]
    const __nv_bfloat16* __restrict__ act,      // [slots, inter]
    const __nv_bfloat16* __restrict__ weight,   // [slots]
    const int* __restrict__ expert,             // [slots]
    __nv_bfloat16* __restrict__ out,            // [slots, hidden], already zeroed
    int scale_cols,
    int hidden,
    int inter) {
    const int slot = blockIdx.y;
    const int e = __ldg(expert + slot);
    if (e < 0) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks = gridDim.x;
    const int rows_per_block = (hidden + blocks - 1) / blocks;
    const int first = blockIdx.x * rows_per_block;
    const int last = min(first + rows_per_block, hidden);
    const auto* dw = down + static_cast<int64_t>(e) * hidden * inter;
    const int64_t scale_stride = static_cast<int64_t>((hidden + kBlockRows - 1) / kBlockRows) * scale_cols;
    const auto* sc = scale + static_cast<int64_t>(e) * scale_stride;
    const auto* a = act + static_cast<int64_t>(slot) * inter;
    const float route = __bfloat162float(weight[slot]);
    // The product-then-scale-then-add the stateless loop performs, once per slot, in bf16.
    for (int i = first + warp; i < last; i += kExpertWarps) {
        const int scale_row = (i / kBlockRows) * scale_cols;
        const uint8_t* __restrict__ drow = dw + static_cast<int64_t>(i) * inter;
        float acc = 0.0f;
        for (int j = lane; j < inter; j += 32) {
            const float s = __bfloat162float(sc[scale_row + j / kBlockCols]);
            acc = fmaf(e4m3_to_float_exact(drow[j]) * s, __bfloat162float(a[j]), acc);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) {
            out[static_cast<int64_t>(slot) * hidden + i] = __float2bfloat16(acc * route);
        }
    }
}


// Each output row of the packed plane is 16 e4m3 codes, so a warp reads one row in two 16-byte
// transactions. `e4m3x2_to_bf162` decodes two codes per lane from one 16-bit read; the lane on the
// odd tail column (when `cols` is odd) decodes one code and skips the missing half.
__global__ void fp8_gemv_t_bf16_kernel(
    const uint8_t* __restrict__ weight,   // [ceil(cols/16), rows, 16]
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int cols) {
    constexpr int kVec = 16;
    const int lane = threadIdx.x & 31;
    const int warps = (blockDim.x >> 5) * gridDim.x;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int chunks = (cols + kVec - 1) / kVec;
    const int col = 2 * lane;
    for (int r = warp; r < rows; r += warps) {
        const uint8_t* plane = weight + static_cast<int64_t>(r) * kVec;
        const bool paired = col + 1 < cols;
        float acc = 0.0f;
        if (paired) {
            for (int c = 0; c < chunks; ++c) {
                const uchar2 raw = *reinterpret_cast<const uchar2*>(plane + static_cast<int64_t>(c) * rows * kVec + col);
                const __nv_bfloat162 w = e4m3x2_to_bf162(
                    static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 8));
                const int base = c * kVec + col;
                acc = fmaf(__bfloat162float(w.x), __bfloat162float(x[base]), acc);
                acc = fmaf(__bfloat162float(w.y), __bfloat162float(x[base + 1]), acc);
            }
        } else {
            for (int c = 0; c < chunks; ++c) {
                const uint8_t code = plane[static_cast<int64_t>(c) * rows * kVec + col];
                acc = fmaf(e4m3_to_float_exact(code), __bfloat162float(x[c * kVec + col]), acc);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) {
            out[r] = __float2bfloat16(acc);
        }
    }
}

// [E, rows, cols] -> [E, ceil(cols/16), rows, 16] as one 2-D copy per plane: `rows` rows of 16
// bytes, the source stride `cols`. A tail plane copies only the bytes that exist.
__host__ void pack_planes(
    const uint8_t* source,
    uint8_t* packed,
    int64_t experts,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    constexpr int kVec = 16;
    const int64_t chunks = (cols + kVec - 1) / kVec;
    const int64_t bytes = std::min<int64_t>(kVec, cols);
    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        packed, kVec, source, cols, bytes, rows, cudaMemcpyDeviceToDevice, stream));
    for (int64_t p = 1; p < experts * chunks; ++p) {
        const int64_t e = p / chunks;
        const int64_t c = p % chunks;
        const int64_t width = std::min<int64_t>(kVec, cols - c * kVec);
        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            packed + p * rows * kVec,
            kVec,
            source + e * rows * cols + c * kVec,
            cols,
            width,
            rows,
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

// [E, rows, cols] e4m3 -> [E, ceil(cols/16), rows, 16] e4m3, read back the same numbers.
torch::Tensor qwen4_exp_fp8_pack_t_cuda(const torch::Tensor& weight) {
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 3, "weight must be a CUDA [E, rows, cols] tensor");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float8_e4m3fn, "weight must be float8_e4m3fn");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    const at::cuda::CUDAGuard guard(weight.device());
    const int64_t experts = weight.size(0);
    const int64_t rows = weight.size(1);
    const int64_t cols = weight.size(2);
    const int64_t chunks = (cols + 15) / 16;
    auto packed = torch::empty({experts, chunks, rows, 16}, weight.options().dtype(at::kByte));
    if (experts == 0 || rows == 0) {
        return packed;
    }
    pack_planes(
        reinterpret_cast<const uint8_t*>(weight.const_data_ptr()),
        packed.data_ptr<uint8_t>(),
        experts,
        rows,
        cols,
        at::cuda::getCurrentCUDAStream());
    return packed;
}

// The decode step's routed experts: a top-k of slots in ONE launch pair, into a [slots, hidden] bf16
// result the caller sums.
//
//   x            [hidden] bf16, the token.
//   expert       [slots] int32, the global expert id per slot (a foreign expert masked to -1 by the
//                caller is skipped and its row is zero -- the caller's weight is zero too).
//   weight       [slots] bf16, the route weight per slot.
//   gate_up_t    the packed gate/up plane (`qwen4_exp_fp8_pack_t`), [E, ceil(hidden/16), 2*inter, 16].
//   blocks_per_expert  how many blocks an expert's `down` phase is split across.
//
// No host read of the route: `expert` and `weight` stay on the device, which is what a captured
// graph needs and what the stateless loop could not give it.
torch::Tensor qwen4_exp_fp8_slots_swiglu_cuda(
    const torch::Tensor& x,
    const torch::Tensor& expert,
    const torch::Tensor& weight,
    const torch::Tensor& gate_up,
    const torch::Tensor& gate_up_scale,
    const torch::Tensor& gate_up_t,
    const torch::Tensor& down,
    const torch::Tensor& down_scale,
    int64_t blocks_per_expert) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.is_contiguous() && gate_up.is_contiguous() && gate_up_scale.is_contiguous() &&
                    gate_up_t.is_contiguous() && down.is_contiguous() && down_scale.is_contiguous() &&
                    expert.is_contiguous() && weight.is_contiguous(),
                "operands must be contiguous");
    TORCH_CHECK(expert.scalar_type() == at::ScalarType::Int, "expert must be int32");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16 && weight.numel() == expert.numel(),
                "weight must be bf16, one per slot");
    // The planes come back from `qwen4_exp_fp8_pack_t` as raw bytes over the same codes, so
    // either encoding is the same table read a byte at a time.
    const bool gate_is_codes = gate_up.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                               gate_up.scalar_type() == at::ScalarType::Byte;
    TORCH_CHECK(gate_is_codes && down.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "gate_up must be float8_e4m3fn or its byte-packed plane, down must be float8_e4m3fn");
    // `gate_up` is 3-D in the row layout and 4-D as the packed plane, so the shapes are taken
    // from `down`, which is the row layout either way: [E, hidden, inter].
    TORCH_CHECK(down.dim() == 3 && gate_up.size(0) == down.size(0),
                "tables must be [E, ...] with the same number of experts");
    TORCH_CHECK(x.numel() == down.size(1),
                "x must be as long as an expert's hidden size");

    const at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t slots = expert.numel();
    const int hidden = static_cast<int>(down.size(1));
    const int inter = static_cast<int>(down.size(2));
    const int gate_scale_cols = static_cast<int>(gate_up_scale.size(2));
    const int down_scale_cols = static_cast<int>(down_scale.size(2));
    TORCH_CHECK(down.size(1) == hidden && down.size(2) == inter, "down must be [E, hidden, inter]");
    TORCH_CHECK(gate_up.dim() == 4 && gate_up.size(1) == (hidden + 15) / 16 &&
                    gate_up.size(2) == 2 * inter && gate_up.size(3) == 16,
                "gate_up must be the packed plane [E, ceil(hidden/16), 2*inter, 16]");
    auto out = torch::zeros({slots, static_cast<int64_t>(hidden)}, x.options());
    if (slots == 0) {
        return out;
    }
    const int blocks = static_cast<int>(std::max<int64_t>(1, blocks_per_expert));
    // The act phase took one block a slot and walked every row with eight warps; it now takes
    // the split the down phase already took, so both phases fill the card.
    const int act_blocks = blocks;

    auto act = torch::empty({slots, static_cast<int64_t>(inter)}, x.options());
    const auto* x_bf = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
    const auto* gu_u8 = reinterpret_cast<const uint8_t*>(gate_up.const_data_ptr());
    const auto* gu_t_u8 = reinterpret_cast<const uint8_t*>(gate_up_t.const_data_ptr());
    const auto* gu_bf = reinterpret_cast<const __nv_bfloat16*>(gate_up_scale.const_data_ptr());
    const auto* dw_u8 = reinterpret_cast<const uint8_t*>(down.const_data_ptr());
    const auto* dw_bf = reinterpret_cast<const __nv_bfloat16*>(down_scale.const_data_ptr());
    const auto* w_bf = reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());
    const int* ids = expert.const_data_ptr<int>();
    auto* act_bf = reinterpret_cast<__nv_bfloat16*>(act.data_ptr());
    auto* out_bf = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    if (getenv("NOPHASE") == nullptr) {
    fp8_expert_act_kernel<<<dim3(static_cast<unsigned>(act_blocks), static_cast<unsigned>(slots)),
                             kExpertThreads, 0, stream>>>(
        x_bf, gu_t_u8, gu_bf, act_bf, ids, gate_scale_cols, hidden, inter);
    }
    if (getenv("NODOWN") == nullptr) {
    fp8_expert_down_kernel<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(slots)),
                             kExpertThreads, 0, stream>>>(
        dw_u8, dw_bf, act_bf, w_bf, ids, out_bf, down_scale_cols, hidden, inter);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// The dense token projection: out[v] = sum_k x[k] * W[v, k], W e4m3 read in place.
//
// This is the shape the decode step spends most of its weight traffic on and the one torch runs as
// a dequantize-then-GEMM: q/k/v/o, the shared expert, lm_head and the router all read an e4m3 table
// whose bf16 copy is larger than the card and costs a pass to make. Here a warp owns one output
// row `v`: the lanes stride `k` so a warp's loads are 32 consecutive bytes of one row -- two 16-byte
// transactions, both full -- which is the coalesced direction for a row-major table. The block
// scale is looked up once per (row, 128 columns), the codes are widened in registers.
//
// `x` is [batch, cols] bf16 and the reduction is over `k` for every row; `out` is [batch, rows].
// No block scale is applied to `x`: this is a weight-quantized GEMV, which is what the checkpoint
// and `dequantize_block_fp8` define, and an activation quantized in the caller would be a different
// arithmetic.
__global__ void fp8_linear_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ weight,         // [batch, rows, cols]
    const __nv_bfloat16* __restrict__ scale,    // [batch, ceil(rows/128), ceil(cols/128)]
    __nv_bfloat16* __restrict__ out,            // [batch, rows]
    int rows,
    int cols,
    int scale_cols) {
    const int batch = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t scale_plane = static_cast<int64_t>((rows + kBlockRows - 1) / kBlockRows) * scale_cols;
    const auto* w = weight + static_cast<int64_t>(batch) * rows * cols;
    const auto* sc = scale + static_cast<int64_t>(batch) * scale_plane;
    const auto* xb = x + static_cast<int64_t>(batch) * cols;
    // `v` steps by the grid's warps so every (block, warp) pair owns one residue class: the rows a
    // block reaches are spread over the whole table instead of being a contiguous run of them.
    for (int v = blockIdx.y * warps_per_block + warp; v < rows; v += gridDim.y * warps_per_block) {
        const int scale_row = (v / kBlockRows) * scale_cols;
        const uint8_t* __restrict__ row = w + static_cast<int64_t>(v) * cols;
        float acc = 0.0f;
        int col = lane;
        for (; col + 1 < cols; col += 32) {
            // Two codes per lane in one uchar2 read at an even offset: `col` is an even stride from
            // lane, so the pair is aligned and the warp's 64 bytes come as four 16-byte
            // transactions. `x` is read one element at a time: its row stride `cols` can be odd
            // (an expert's `down` is 2560x640), so a paired load there would be misaligned.
            const uchar2 raw = *reinterpret_cast<const uchar2*>(row + col);
            const __nv_bfloat162 wv = e4m3x2_to_bf162(
                static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 8));
            const float s = __bfloat162float(sc[scale_row + col / kBlockCols]);
            acc = fmaf(__bfloat162float(wv.x) * s, __bfloat162float(xb[col]), acc);
            acc = fmaf(__bfloat162float(wv.y) * s, __bfloat162float(xb[col + 1]), acc);
        }
        if (col < cols) {
            const float s = __bfloat162float(sc[scale_row + col / kBlockCols]);
            acc = fmaf(e4m3_to_float_exact(row[col]) * s, __bfloat162float(xb[col]), acc);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) {
            out[static_cast<int64_t>(batch) * rows + v] = __float2bfloat16(acc);
        }
    }
}

// The dense token projection for decode, e4m3 read in place: x [t, hidden] @ W^T, [t, out].
torch::Tensor qwen4_exp_fp8_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    int64_t blocks_per_batch) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16 && x.is_contiguous(),
                "x must be contiguous CUDA bf16");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                    weight.is_contiguous() && weight.dim() == 3,
                "weight must be contiguous float8_e4m3fn [t, rows, cols]");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::ScalarType::BFloat16 &&
                    scale.is_contiguous() && scale.dim() == 3,
                "scale must be contiguous bf16 [t, ceil(rows/128), ceil(cols/128)]");
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t batch = weight.size(0);
    const int rows = static_cast<int>(weight.size(1));
    const int cols = static_cast<int>(weight.size(2));
    TORCH_CHECK(x.dim() == 2 && x.size(1) == cols, "x must be [t, cols]");
    TORCH_CHECK(x.size(0) == batch, "x and weight disagree on the number of tables");
    const int scale_cols = static_cast<int>(scale.size(2));
    TORCH_CHECK(static_cast<int>(scale.size(1)) == (rows + kBlockRows - 1) / kBlockRows,
                "scale rows must be ceil(rows / 128)");
    auto out = torch::empty({batch, static_cast<int64_t>(rows)}, x.options());
    constexpr int kThreads = 256;
    const int warps_per_block = kThreads / 32;
    const int y = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(blocks_per_batch, (rows + warps_per_block - 1) / warps_per_block)));
    fp8_linear_bf16_kernel<<<dim3(static_cast<unsigned>(1), static_cast<unsigned>(y),
                                  static_cast<unsigned>(batch)),
                            kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(weight.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        rows,
        cols,
        scale_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// A single fused FP8 GEMV over the packed transposed plane, for measuring the load path on its own.
// weight packed [ceil(cols/16), rows, 16] e4m3 holds the e4m3 weight [rows, cols] transposed into
// 16-byte chunks; x is [rows].
torch::Tensor qwen4_exp_fp8_gemv_t_bf16_cuda(
    const torch::Tensor& packed,
    const torch::Tensor& x) {
    TORCH_CHECK(packed.is_cuda() && packed.dim() == 3 && packed.is_contiguous(), "packed must be contiguous [chunks, rows, 16]");
    TORCH_CHECK(packed.scalar_type() == at::ScalarType::Float8_e4m3fn, "packed must be float8_e4m3fn");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16 && x.is_contiguous(),
                "x must be contiguous CUDA bf16");
    const at::cuda::CUDAGuard guard(x.device());
    const int rows = static_cast<int>(packed.size(1));
    const int cols = static_cast<int>(packed.size(0) * 16);
    TORCH_CHECK(x.numel() == rows, "x length must equal packed rows");
    auto out = torch::empty({static_cast<int64_t>(cols)}, x.options());
    const int threads = 256;
    const int total_warps = rows;
    const int blocks = std::min(4096, (total_warps + threads / 32 - 1) / (threads / 32));
    fp8_gemv_t_bf16_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(packed.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        rows,
        cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

