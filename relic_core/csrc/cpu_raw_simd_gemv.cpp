// Raw-byte NEON host GEMV over original block-32 FP8 weights.
//
// The accepted rule is the one `cpu_block_gemv.cpp` states: 32 byte-coded
// products per K block, two 16-term halves truncated onto a 2**(anchor - 13)
// grid, the running sum becoming a peer of the second half through
// max(its anchor, floor(log2|first half|)), then one renormalisation of the
// block result, the activation scale and then the weight scale in FP32, and a
// sequential K-order FP32 accumulation of the blocks.
//
// This translation unit differs from that one in what it does per byte and how
// many times it reads the weights, not in what it computes:
//
//   * **No prepack.** The weights stay exactly the checkpoint's `[N, K]` u8
//     codes, so one call reads N*K bytes -- the minimum the reduction can read.
//     A 4-byte-per-code prepack reads four times that and was measured to gain
//     nothing, because the kernel is bandwidth-bound rather than decode-bound
//     at that point.
//   * **SIMD across K lanes, not across output columns.** One 4-byte load holds
//     four consecutive codes, so the inner vector of four K positions needs one
//     u32 load and four L1-resident table gathers instead of four separate reads
//     and four separate scalar decodes. The 32 products of a K block are eight
//     such vectors, held in registers through *both* halves, so every byte is
//     decoded once per call.
//   * The truncated 4-lane sums are integers. Each term is below 2**15 in
//     magnitude -- the extreme is `240 * 240` with a nominal sum of 14, giving
//     28800, which an exhaustive sweep of all finite code pairs confirms -- so
//     sixteen of them sum below 2**19 -- no int32 overflow, and
//     integer addition is associative, which is what makes a 4-lane horizontal
//     add the same number as the scalar lane-order sum. The FP32 additions that
//     follow (`float(whole)`, the carry, the renormalisation, the per-block
//     scales and the K accumulation) stay scalar and in the original order.
//
// So the arithmetic contract is unchanged, and `-ffp-contract=off` is still
// load-bearing: no fused multiply-add may replace the two separate FP32
// multiplies of a term or the scale multiplies of a block.
//
// Portability: the `__ARM_NEON` path is the target; a plain scalar path is
// compiled otherwise (x86) and can be forced with `-DRELIC_RAW_SIMD_FORCE_SCALAR`
// so a test can prove the two agree bit for bit. Both paths implement the same
// expressions, so the artifact's ABI and semantics do not depend on which one a
// build picked.
//
// Non-finite input and scale bytes follow `cpu_block_gemv.cpp` exactly: a NaN
// e4m3 code makes its output column NaN, and the Python binding refuses the raw
// scale byte 255 (which would mean 2**128) before the call. There is no BLAS, no
// float64, no native FP8 cast, and no BF16 expansion of the weights.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) && !defined(RELIC_RAW_SIMD_FORCE_SCALAR)
#define RELIC_RAW_SIMD_NEON 1
#include <arm_neon.h>
#endif

namespace {

constexpr int32_t kBlock = 32;
constexpr int32_t kHalf = 16;
constexpr int32_t kWindow = 13;
constexpr int32_t kAbiVersion = 1;

//: Nominal exponents, encoded in the low bits of the decode word the same way
//: `cpu_block_gemv.cpp` encodes them. Ten bits are enough: a live nominal is
//: -6..8 and the empty sentinel is -1000, so biased by 1000 the widest value is
//: 1008. The 32-bit e4m3 value always has its low twenty bits clear (its
//: significand starts at bit 20 at the lowest), so a decode word can carry both.
constexpr int32_t kNominalBias = 1000;
constexpr uint32_t kNominalMask = 0x3FFu;
constexpr uint32_t kValueMask = 0xFFFFFC00u;

//: The nominal of a signed-zero code: it contributes nothing to a half's anchor.
constexpr int32_t kEmpty = -1000;
constexpr int32_t kEmptyGuard = -500;
//: "No exponent" for a first half that summed to exactly zero.
constexpr int32_t kNoExponent = -1000000;

constexpr int32_t kOk = 0;
constexpr int32_t kBadArguments = 1;
constexpr int32_t kOutOfMemory = 2;

alignas(64) uint32_t g_packed[256];
//: `2**(byte - 127)` for every finite E8M0 byte, so a block's scale is one L1
//: load instead of a per-block `Pow2` call, and no float copy of the weight scale
//: grid is built: the scalar kernel materialises one per call (~0.6 MiB for the
//: released square grid), this kernel expands nothing.
alignas(64) float g_scale[256];
std::once_flag g_init_once;

void InitTables() {
  for (int32_t byte = 0; byte < 256; ++byte) {
    const int32_t field = (byte >> 3) & 15;
    const int32_t mantissa = byte & 7;
    float value;
    if (field == 15 && mantissa == 7) {
      value = std::numeric_limits<float>::quiet_NaN();
    } else if (field == 0) {
      value = std::ldexp(static_cast<float>(mantissa), -9);
    } else {
      value = std::ldexp(static_cast<float>(8 + mantissa), field - 10);
    }
    if ((byte >> 7) != 0) {
      value = -value;
    }
    const int32_t nominal = field >= 1 ? field - 7 : -6;
    const int32_t carried = ((byte & 0x7F) != 0) ? nominal : kEmpty;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    // The encoding relies on the low twenty bits of an e4m3 value being clear,
    // and on the biased nominal fitting the field. Both are properties of the
    // format; a violation means the table would silently corrupt a lane.
    if ((bits & 0x000FFFFFu) != 0) {
      std::abort();
    }
    if (carried + kNominalBias < 0 ||
        carried + kNominalBias >= static_cast<int32_t>(kNominalMask) + 1) {
      std::abort();
    }
    g_packed[byte] = (bits & kValueMask) |
                     (static_cast<uint32_t>(carried + kNominalBias) & kNominalMask);
    g_scale[byte] = (byte == 255) ? std::numeric_limits<float>::infinity()
                                  : std::ldexp(1.0f, byte - 127);
  }
}

inline void EnsureTables() { std::call_once(g_init_once, InitTables); }

//: 2**exponent as the nearest binary32, built from its IEEE field so no `exp2`
//: approximation can move a grid by an octave. Same function as the scalar
//: kernel's, including its saturating behaviour outside the format.
inline float Pow2(int32_t exponent) {
  if (exponent > 127) {
    return std::numeric_limits<float>::infinity();
  }
  if (exponent >= -126) {
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
  }
  if (exponent >= -149) {
    const uint32_t bits = 1u << (exponent + 149);
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
  }
  return 0.0f;
}

inline int32_t FloorExponent(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return static_cast<int32_t>((bits >> 23) & 255) - 127;
}

//: `(inverse, step)` = (2**(13 - anchor), 2**(anchor - 13)) for a live anchor,
//: and the identity for an anchorless half. Same rule as the scalar kernel.
inline void CutPowers(int32_t anchor, float *inverse, float *step) {
  if (anchor < kEmptyGuard) {
    *inverse = 1.0f;
    *step = 1.0f;
    return;
  }
  *inverse = Pow2(kWindow - anchor);
  *step = Pow2(anchor - kWindow);
}

//: Whether any of 32 consecutive e4m3 bytes is a NaN code, i.e. (b & 0x7F) ==
//: 0x7F. Tested eight bytes at a time; a false positive only sends the column
//: down the exact NaN path, so it may be conservative but never may miss.
inline bool HasNanCode(const uint8_t *codes) {
  constexpr uint64_t kLow = 0x7F7F7F7F7F7F7F7FULL;
  constexpr uint64_t kOnes = 0x0101010101010101ULL;
  constexpr uint64_t kHigh = 0x8080808080808080ULL;
  for (int32_t word = 0; word < kBlock / 8; ++word) {
    uint64_t chunk;
    std::memcpy(&chunk, codes + word * 8, sizeof chunk);
    const uint64_t masked = (chunk & kLow) ^ kLow;
    if (((masked - kOnes) & ~masked & kHigh) != 0) {
      return true;
    }
  }
  return false;
}

//: The activation side, decoded once per call and shared by every column.
struct Activation {
  const float *value;       // [k] exact e4m3 values
  const int32_t *nominal;   // [k] nominal exponents, biased by kNominalBias
  const float *scale;       // [k / 32] 2**(byte - 127)
};

//: The scalar form of the block reduction. Identical expressions, lane by lane,
//: to the vector form below; this is what a non-NEON target compiles.
inline float BlockFragmentScalar(const Activation &a, int64_t base,
                                 const uint8_t *codes) {
  float values[kBlock];
  int32_t nominals[kBlock];
  for (int32_t i = 0; i < kBlock; ++i) {
    const uint32_t packed = g_packed[codes[i]];
    const uint32_t bits = packed & kValueMask;
    std::memcpy(&values[i], &bits, sizeof(float));
    nominals[i] = static_cast<int32_t>(packed & kNominalMask) + a.nominal[base + i];
  }
  int32_t anchor_first = std::numeric_limits<int32_t>::min();
  for (int32_t i = 0; i < kHalf; ++i) {
    anchor_first = std::max(anchor_first, nominals[i]);
  }
  anchor_first -= 2 * kNominalBias;
  float inverse_first;
  float step_first;
  CutPowers(anchor_first, &inverse_first, &step_first);
  int32_t whole_first = 0;
  for (int32_t i = 0; i < kHalf; ++i) {
    whole_first += static_cast<int32_t>(
        (values[i] * a.value[base + i]) * inverse_first);
  }
  const float total_first = static_cast<float>(whole_first) * step_first;
  const int32_t running =
      total_first != 0.0f ? FloorExponent(total_first) : kNoExponent;

  int32_t anchor_second = std::numeric_limits<int32_t>::min();
  for (int32_t i = kHalf; i < kBlock; ++i) {
    anchor_second = std::max(anchor_second, nominals[i]);
  }
  anchor_second -= 2 * kNominalBias;
  const int32_t window = std::max(anchor_second, running);
  float inverse_second;
  float step_second;
  CutPowers(window, &inverse_second, &step_second);
  int32_t whole_second = 0;
  for (int32_t i = kHalf; i < kBlock; ++i) {
    whole_second += static_cast<int32_t>(
        (values[i] * a.value[base + i]) * inverse_second);
  }
  const float total =
      (static_cast<float>(whole_second) +
       std::trunc(total_first * inverse_second)) *
      step_second;
  if (total == 0.0f) {
    return total;
  }
  const int32_t exponent = FloorExponent(total);
  const float step = Pow2(exponent - kWindow);
  return std::trunc(total / step) * step;
}

#ifdef RELIC_RAW_SIMD_NEON

//: Largest of the four lanes spread over `q` vectors.
inline int32_t HorizontalMax(int32x4_t first, int32x4_t second,
                             int32x4_t third, int32x4_t fourth) {
  const int32x4_t all = vmaxq_s32(vmaxq_s32(first, second),
                                  vmaxq_s32(third, fourth));
  const int32x2_t pairs = vmax_s32(vget_low_s32(all), vget_high_s32(all));
  return vget_lane_s32(vpmax_s32(pairs, pairs), 0);
}

//: The vector form: one u32 load and four table gathers per four K lanes, the
//: eight product vectors kept in registers through both halves.
inline float BlockFragmentNeon(const Activation &a, int64_t base,
                               const uint8_t *codes) {
  float32x4_t values[8];
  float32x4_t products[8];
  int32x4_t nominals[8];
  for (int32_t group = 0; group < 8; ++group) {
    const int32_t at = 4 * group;
    uint32_t word;
    std::memcpy(&word, codes + at, sizeof word);
    // The four gathers are the only scalar loads left in the decode; the table
    // is 1 KiB, so they are L1 hits and issue alongside the vector work.
    const uint32_t lanes[4] = {
        g_packed[word & 0xFFu], g_packed[(word >> 8) & 0xFFu],
        g_packed[(word >> 16) & 0xFFu], g_packed[word >> 24]};
    const uint32x4_t packed = vld1q_u32(lanes);
    values[group] =
        vreinterpretq_f32_u32(vandq_u32(packed, vdupq_n_u32(kValueMask)));
    const int32x4_t carried = vreinterpretq_s32_u32(
        vandq_u32(packed, vdupq_n_u32(kNominalMask)));
    nominals[group] = vaddq_s32(carried, vld1q_s32(a.nominal + base + at));
    // Two separate FP32 multiplies, as the scalar form has: the scale must not
    // be contracted into the first product.
    products[group] =
        vmulq_f32(values[group], vld1q_f32(a.value + base + at));
  }

  const int32_t anchor_first =
      HorizontalMax(nominals[0], nominals[1], nominals[2], nominals[3]) -
      2 * kNominalBias;
  float inverse_first;
  float step_first;
  CutPowers(anchor_first, &inverse_first, &step_first);
  int32_t whole_first = 0;
  const float32x4_t inverse_first_vector = vdupq_n_f32(inverse_first);
  for (int32_t group = 0; group < 4; ++group) {
    // `vcvtq_s32_f32` rounds toward zero, which is the reference's cut.
    whole_first += vaddvq_s32(
        vcvtq_s32_f32(vmulq_f32(products[group], inverse_first_vector)));
  }
  const float total_first = static_cast<float>(whole_first) * step_first;
  const int32_t running =
      total_first != 0.0f ? FloorExponent(total_first) : kNoExponent;

  const int32_t anchor_second =
      HorizontalMax(nominals[4], nominals[5], nominals[6], nominals[7]) -
      2 * kNominalBias;
  const int32_t window = std::max(anchor_second, running);
  float inverse_second;
  float step_second;
  CutPowers(window, &inverse_second, &step_second);
  int32_t whole_second = 0;
  const float32x4_t inverse_second_vector = vdupq_n_f32(inverse_second);
  for (int32_t group = 4; group < 8; ++group) {
    whole_second += vaddvq_s32(
        vcvtq_s32_f32(vmulq_f32(products[group], inverse_second_vector)));
  }
  const float total =
      (static_cast<float>(whole_second) +
       std::trunc(total_first * inverse_second)) *
      step_second;
  if (total == 0.0f) {
    return total;
  }
  const int32_t exponent = FloorExponent(total);
  const float step = Pow2(exponent - kWindow);
  return std::trunc(total / step) * step;
}

#endif  // RELIC_RAW_SIMD_NEON

inline float BlockFragment(const Activation &a, int64_t base,
                           const uint8_t *codes) {
#ifdef RELIC_RAW_SIMD_NEON
  return BlockFragmentNeon(a, base, codes);
#else
  return BlockFragmentScalar(a, base, codes);
#endif
}

struct Plan {
  Activation activation;
  const uint8_t *codes;          // [n, k]
  const uint8_t *weights_scale;  // raw E8M0 bytes
  int64_t n;
  int64_t k;
  int64_t blocks;
  int32_t block_rows;
  float *out;                    // [n]
};

void RunRange(const Plan &plan, int64_t begin, int64_t end) {
  for (int64_t column = begin; column < end; ++column) {
    const int64_t scale_row =
        plan.block_rows == 1 ? column : column / plan.block_rows;
    const uint8_t *row_codes = plan.codes + column * plan.k;
    const uint8_t *row_scale = plan.weights_scale + scale_row * plan.blocks;
    float accumulator = 0.0f;
    bool finite = true;
    for (int64_t block = 0; block < plan.blocks; ++block) {
      const uint8_t *codes = row_codes + block * kBlock;
      if (HasNanCode(codes)) {
        finite = false;
        break;
      }
      const float fragment =
          BlockFragment(plan.activation, block * kBlock, codes);
      // Activation scale first, weight scale second, then the K-block add.
      accumulator += (fragment * plan.activation.scale[block]) *
                     g_scale[row_scale[block]];
    }
    plan.out[column] =
        finite ? accumulator : std::numeric_limits<float>::quiet_NaN();
  }
}

int32_t Validate(int64_t n, int64_t k, int32_t block_rows, int32_t threads) {
  if (n <= 0 || k <= 0 || k % kBlock != 0) {
    return kBadArguments;
  }
  if (block_rows != 1 && block_rows != kBlock) {
    return kBadArguments;
  }
  if (block_rows == kBlock && n % kBlock != 0) {
    return kBadArguments;
  }
  if (threads < 1) {
    return kBadArguments;
  }
  return kOk;
}

}  // namespace

extern "C" {

//: ABI version of the compiled kernel; the loader refuses an unknown one.
int32_t relic_cpu_raw_simd_gemv_abi_version(void) { return kAbiVersion; }

//: Which decode path this artifact was compiled with, for the record:
//: 4 = NEON across K lanes, 1 = the portable scalar form. It selects no
//: behaviour at run time -- the build decides -- so a caller can report what it
//: is measuring without a second entry point.
int32_t relic_cpu_raw_simd_gemv_vector_width(void) {
#ifdef RELIC_RAW_SIMD_NEON
  return 4;
#else
  return 1;
#endif
}

//: Exact M = 1 block-32 FP8 GEMV over the checkpoint's own `[N, K]` codes.
//:
//:   a_codes   [k]                        e4m3 bytes
//:   a_scale   [k / 32]                   raw E8M0 bytes
//:   b_codes   [n, k]                     e4m3 bytes, row-major, no prepack
//:   b_scale   [n / 32, k / 32]           raw E8M0 bytes, block_rows == 32
//:             [n, k / 32]                raw E8M0 bytes, block_rows == 1
//:   out       [n]                        binary32 accumulators
//:
//: `threads` splits the output columns; every column is computed identically
//: whatever the thread count, so the result does not depend on it.
int32_t relic_cpu_raw_simd_gemv_fp8(const uint8_t *a_codes,
                                    const uint8_t *a_scale,
                                    const uint8_t *b_codes,
                                    const uint8_t *b_scale, float *out,
                                    int64_t n, int64_t k, int32_t block_rows,
                                    int32_t threads, int32_t *threads_used) {
  if (threads_used != nullptr) {
    *threads_used = 0;
  }
  if (a_codes == nullptr || a_scale == nullptr || b_codes == nullptr ||
      b_scale == nullptr || out == nullptr) {
    return kBadArguments;
  }
  const int32_t status = Validate(n, k, block_rows, threads);
  if (status != kOk) {
    return status;
  }
  try {
    EnsureTables();
    const int64_t blocks = k / kBlock;

    std::vector<float> a_value(static_cast<size_t>(k));
    std::vector<int32_t> a_nominal(static_cast<size_t>(k));
    // The activation row is decoded once; every column reads these arrays, so
    // they stay in the shared last-level cache rather than being re-derived.
    for (int64_t i = 0; i < k; ++i) {
      const uint32_t packed = g_packed[a_codes[i]];
      const uint32_t bits = packed & kValueMask;
      float value;
      std::memcpy(&value, &bits, sizeof value);
      if (value != value) {
        // A NaN in the activation row poisons every column, and no block loop
        // has to run: the integer conversions below have no NaN result.
        for (int64_t column = 0; column < n; ++column) {
          out[column] = std::numeric_limits<float>::quiet_NaN();
        }
        if (threads_used != nullptr) {
          *threads_used = 1;
        }
        return kOk;
      }
      a_value[static_cast<size_t>(i)] = value;
      a_nominal[static_cast<size_t>(i)] =
          static_cast<int32_t>(packed & kNominalMask);
    }
    std::vector<float> a_scale_power(static_cast<size_t>(blocks));
    for (int64_t block = 0; block < blocks; ++block) {
      a_scale_power[static_cast<size_t>(block)] = g_scale[a_scale[block]];
    }

    Plan plan{};
    plan.activation.value = a_value.data();
    plan.activation.nominal = a_nominal.data();
    plan.activation.scale = a_scale_power.data();
    plan.codes = b_codes;
    plan.weights_scale = b_scale;
    plan.n = n;
    plan.k = k;
    plan.blocks = blocks;
    plan.block_rows = block_rows;
    plan.out = out;

    const int32_t workers = static_cast<int32_t>(std::min<int64_t>(threads, n));
    if (threads_used != nullptr) {
      *threads_used = workers;
    }
    if (workers <= 1) {
      RunRange(plan, 0, n);
      return kOk;
    }

    // Columns are handed out in chunks so a worker that draws a cheap column
    // (an early NaN break) does not idle, and so the per-column loop stays long
    // enough for the fetch to run ahead.
    const int64_t chunk =
        std::max<int64_t>(1, std::min<int64_t>(64, std::max<int64_t>(1, n / workers)));
    std::atomic<int64_t> next{0};
    auto work = [&]() {
      for (;;) {
        const int64_t begin = next.fetch_add(chunk);
        if (begin >= n) {
          return;
        }
        RunRange(plan, begin, std::min(begin + chunk, n));
      }
    };
    // The calling thread takes part, so the result is produced even if the
    // platform refuses to start a worker; a creation failure cannot escape with
    // joinable threads alive, which would terminate the process.
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    try {
      for (int32_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back(work);
      }
    } catch (...) {
      // Fall through: this thread plus the workers already running finish the
      // remaining chunks through the same counter.
    }
    work();
    for (std::thread &worker : pool) {
      worker.join();
    }
    return kOk;
  } catch (const std::bad_alloc &) {
    return kOutOfMemory;
  } catch (...) {
    return kBadArguments;
  }
}

}  // extern "C"