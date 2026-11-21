// Exact host-CPU block-32 FP8 GEMV for the original Engram `wkv` projection.
//
// This is a compiled, output-column-parallel implementation of the reduction the
// accepted CPU runtime already performs, restricted to M = 1. It exists because
// the host Engram is deliberately kept on the CPU for Ascend inference: the
// projection's activation is one row, so the general block GEMM spends most of
// its time in per-K-tile tensor bookkeeping rather than in arithmetic.
//
// The rule reproduced here, per output column and per 32-wide K block:
//
//   1. Both operands stay byte-coded. A code's value and its "nominal exponent"
//      F(v) = max(true exponent, -6) are read from a 256-entry table; a signed
//      zero code has no nominal exponent at all and contributes a sentinel.
//   2. The 32 products of one K block are split into two 16-term halves. Each
//      half is truncated onto the grid 2**(anchor - 13), where the anchor is the
//      largest summed nominal exponent in that half. The first half establishes
//      its own anchor; the second takes max(its own anchor, floor(log2|first
//      half sum|)), so the running sum is a peer of the second half's products
//      rather than a prefix.
//   3. The block result is renormalised once more onto 2**(floor(log2|total|) -
//      13). That step is separate from the two cuts: the second half's sum can
//      carry bits past its window.
//   4. The two per-block scales are applied to the finished block result in
//      FP32, activation scale first and weight scale second, and the blocks are
//      then accumulated one at a time in increasing K order.
//
// Why this is bit-exact rather than merely close. Every intermediate above is
// exactly representable in binary32:
//
//   * a product of two e4m3 codes has at most eight significant bits and lives
//     in [2**-18, 2**18], so it is exact and never subnormal;
//   * the anchor is the largest summed nominal exponent, which is at least the
//     true exponent of every product in the half, so each truncated term is
//     below 2**14 in magnitude and the 16-term sum stays below 2**18 -- whole
//     numbers, and therefore exact in binary32;
//   * the running sum's rescale onto the second window is bounded the same way
//     by the window rule itself, and the final renormalisation keeps a 14-bit
//     window, so both are exact too.
//
// The only rounding in the whole reduction is the FP32 add that accumulates the
// K blocks, which is the one place the reference rounds as well. `trunc` is
// toward zero, matching the reference's own cut. Because of that, no fused
// multiply-add may be introduced: the build passes `-ffp-contract=off`, and the
// two per-block scale multiplies are deliberately kept separate so that an
// intermediate overflow or underflow behaves exactly as the reference's does.
//
// Two notes on scope, so the claim is neither wider nor narrower than it should be:
//
//   * the final renormalisation divides by its step, and the cut of the running
//     sum multiplies by a reciprocal step. For a non-zero total both are exact
//     powers of two and the two forms agree exactly, because a non-zero product of
//     two e4m3 codes is at least 2**-18 in magnitude and the renormalised total
//     never leaves the range where a power-of-two multiply and divide coincide.
//     The reference builds its steps by shifting an integer exponent field, which
//     wraps if it is ever handed an exponent outside that domain; this kernel
//     builds them from `Pow2`, which saturates instead. Reproducing the wrap is
//     deliberately not done -- it is unreachable from finite e4m3 codes, and
//     copying an unreachable defect would only make the two disagree about
//     inputs neither can produce.
//   * the arithmetic is therefore asserted for finite e4m3 codes. NaN codes are
//     handled by the explicit propagation above rather than by the float path.
//
// Non-finite input. An e4m3 byte is a NaN code exactly when (byte & 0x7F) ==
// 0x7F. A NaN in the activation row makes every output column NaN; a NaN in one
// weight row makes that column NaN, because a NaN product poisons its half's
// sum and the block result, and the K accumulation can never recover from it.
// Both cases are detected before any float-to-integer conversion, since that
// conversion has no defined result for a NaN. The weight-side check runs per
// 32-byte block on data that has just been read, so it costs a few integer
// operations per eight bytes and adds no memory traffic.
//
// Scale bytes. A raw E8M0 byte b means 2**(b - 127), which is exact for every
// b in [0, 254] because the byte is the exponent field. Byte 255 would mean
// 2**128, which no binary32 holds; the arithmetic here maps it to infinity, and
// the Python binding refuses that byte before it ever arrives, so the public
// contract is finite grids only. That keeps this translation unit a plain
// primitive -- it does what the exponent arithmetic says and does not carry a
// policy argument -- while the interface the model actually calls is the strict
// one. (The Torch oracle in `block_gemm.py` maps the same byte to NaN, which is
// the disagreement the refusal exists to keep out of the public surface.)
//
// No BLAS, no float64, no native FP8 casting, no BF16 expansion of the weights:
// the stored codes are what this reads, and the caller owns the final narrowing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int32_t kBlock = 32;
constexpr int32_t kHalf = 16;
constexpr int32_t kWindow = 13;

//: The nominal exponent of a signed-zero code. A zero operand contributes
//: nothing to its half's anchor, so it is excluded instead of being given the
//: smallest live exponent. Every live anchor is at least -12 and every sentinel
//: pair at most -992, so `kEmptyGuard` separates the two.
constexpr int32_t kEmpty = -1000;
constexpr int32_t kEmptyGuard = -500;

//: "This running sum has no exponent." Used only for a first half that summed to
//: exactly zero; it never wins the `max` against a live anchor, and any window it
//: does produce is below `kEmptyGuard`. The magnitude is irrelevant beyond that
//: ordering, which is why any large negative value is equivalent here.
constexpr int32_t kNoExponent = -1000000;

constexpr int32_t kAbiVersion = 1;

//: Return codes. Any non-zero code means the call did no work.
constexpr int32_t kOk = 0;
constexpr int32_t kBadArguments = 1;
constexpr int32_t kOutOfMemory = 2;

//: One decode word per e4m3 byte: the value's IEEE bits in the high bits and the
//: nominal exponent, biased by ``kNominalBias``, in the low eleven.
//:
//: The packing is possible because an e4m3 value has at most four significant
//: bits, so the low twenty bits of its binary32 representation are always zero --
//: for the normal encodings the significand starts at bit 20, for the seven
//: subnormal ones it starts at bit 21 or above, and the two zeros and two NaNs
//: clear them trivially. That leaves eleven bits to carry the nominal exponent
//: through the same load, which turns two lookups per lane into one and keeps the
//: whole block decode in registers instead of a spilled float array.
constexpr int32_t kNominalBias = 1000;
constexpr uint32_t kValueMask = 0xFFF00000u;
constexpr uint32_t kNominalMask = 0x7FFu;

alignas(64) uint32_t g_packed[256];
std::once_flag g_packed_once;

void InitPacked() {
  for (int32_t byte = 0; byte < 256; ++byte) {
    const int32_t field = (byte >> 3) & 15;
    const int32_t mantissa = byte & 7;
    float value;
    if (field == 15 && mantissa == 7) {
      // The two NaN encodings. Their nominal exponent is still 8, because the
      // encoding field is 15 -- the reference reads the field, not the value.
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
    const int32_t carried =
        ((byte & 0x7F) != 0) ? nominal : kEmpty;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    // The low twenty bits have to be clear or the nominal would collide with
    // the significand; every encodable value satisfies that by construction.
    if ((bits & ~kValueMask) != 0) {
      std::abort();
    }
    g_packed[byte] = (bits & kValueMask) |
                     (static_cast<uint32_t>(carried + kNominalBias) & kNominalMask);
  }
}

inline void EnsurePacked() { std::call_once(g_packed_once, InitPacked); }

inline float PackedValue(uint32_t packed) {
  const uint32_t bits = packed & kValueMask;
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

//: The nominal exponent carried in a decode word, still biased by
//: ``kNominalBias``. Adding the activation's own nominal and subtracting the bias
//: once per half keeps the comparison order identical without a per-lane add.
inline int32_t PackedNominal(uint32_t packed) {
  return static_cast<int32_t>(packed & kNominalMask);
}

//: 2**exponent as the nearest representable binary32, built from its IEEE
//: encoding so no library `exp2` approximation can move a grid by an octave.
//: Exact for every exponent the scales and windows can produce, including the
//: subnormal range an E8M0 byte 0 needs and the infinity a byte 255 asks for.
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

//: floor(log2|value|) for a non-zero finite normal binary32.
inline int32_t FloorExponent(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return static_cast<int32_t>((bits >> 23) & 255) - 127;
}

//: `(inverse, step)` = (2**(13 - anchor), 2**(anchor - 13)), or (1, 1) when the
//: half has no anchor at all -- an anchorless half cuts nothing, which is the
//: identity the reference's sentinel branch produces.
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
//: down the exact NaN path, so the fast test is allowed to be conservative and is
//: never allowed to miss a NaN.
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

//: One output column's 32-term block reduction: the two halves, the running-sum
//: window, and the final renormalisation. `a_value`/`a_nominal` are this K block's
//: activation operands, decoded once for the whole call.
inline float BlockFragment(const float *a_value, const int32_t *a_nominal,
                           const uint8_t *codes) {
  // One decode word per lane, held in registers: the halves below read the
  // value bits and the nominal out of the same word instead of re-looking it up.
  uint32_t packed[kBlock];
  for (int32_t i = 0; i < kBlock; ++i) {
    packed[i] = g_packed[codes[i]];
  }

  // The anchor of a half is the largest summed nominal exponent in it. The
  // carried nominal is biased, so the bias is added to the activation's nominal
  // and removed once per half, which leaves the comparison order untouched.
  int32_t anchor_first = std::numeric_limits<int32_t>::min();
  for (int32_t i = 0; i < kHalf; ++i) {
    const int32_t summed = a_nominal[i] + PackedNominal(packed[i]);
    if (summed > anchor_first) {
      anchor_first = summed;
    }
  }
  anchor_first -= kNominalBias;
  float inverse_first;
  float step_first;
  CutPowers(anchor_first, &inverse_first, &step_first);
  int32_t whole_first = 0;
  for (int32_t i = 0; i < kHalf; ++i) {
    whole_first += static_cast<int32_t>(a_value[i] * PackedValue(packed[i]) * inverse_first);
  }
  const float total_first = static_cast<float>(whole_first) * step_first;
  const int32_t running =
      total_first != 0.0f ? FloorExponent(total_first) : kNoExponent;

  int32_t anchor_second = std::numeric_limits<int32_t>::min();
  for (int32_t i = kHalf; i < kBlock; ++i) {
    const int32_t summed = a_nominal[i] + PackedNominal(packed[i]);
    if (summed > anchor_second) {
      anchor_second = summed;
    }
  }
  anchor_second -= kNominalBias;
  const int32_t window = std::max(anchor_second, running);
  float inverse_second;
  float step_second;
  CutPowers(window, &inverse_second, &step_second);
  int32_t whole_second = 0;
  for (int32_t i = kHalf; i < kBlock; ++i) {
    whole_second += static_cast<int32_t>(a_value[i] * PackedValue(packed[i]) * inverse_second);
  }
  const float total =
      (static_cast<float>(whole_second) + std::trunc(total_first * inverse_second)) *
      step_second;
  if (total == 0.0f) {
    return total;
  }
  const int32_t exponent = FloorExponent(total);
  const float step = Pow2(exponent - kWindow);
  return std::trunc(total / step) * step;
}

//: One output column, accumulated in increasing K order exactly as the reference
//: does. Returns false when the column contains a NaN code, in which case the
//: caller writes NaN.
inline bool ColumnDot(const float *a_value, const int32_t *a_nominal, const float *a_scale,
                      const uint8_t *b_codes, const float *b_scale, int64_t blocks,
                      float *result) {
  float accumulator = 0.0f;
  for (int64_t block = 0; block < blocks; ++block) {
    const uint8_t *codes = b_codes + block * kBlock;
    if (HasNanCode(codes)) {
      return false;
    }
    const int64_t base = block * kBlock;
    const float fragment = BlockFragment(a_value + base, a_nominal + base, codes);
    // Activation scale first, weight scale second, then the K-block add: three
    // separately rounded operations, matching the reference's order.
    accumulator += (fragment * a_scale[block]) * b_scale[block];
  }
  *result = accumulator;
  return true;
}

struct Job {
  const float *a_value;
  const int32_t *a_nominal;
  const float *a_scale;
  const uint8_t *b_codes;
  const float *b_scale;
  int64_t n;
  int64_t k;
  int64_t blocks;
  int32_t block_rows;
  float *out;
  int64_t begin;
  int64_t end;
};

void RunJob(const Job *job) {
  for (int64_t column = job->begin; column < job->end; ++column) {
    const int64_t scale_row =
        job->block_rows == 1 ? column : column / job->block_rows;
    const float *weights_scale = job->b_scale + scale_row * job->blocks;
    float value = 0.0f;
    if (ColumnDot(job->a_value, job->a_nominal, job->a_scale,
                  job->b_codes + column * job->k, weights_scale, job->blocks,
                  &value)) {
      job->out[column] = value;
    } else {
      job->out[column] = std::numeric_limits<float>::quiet_NaN();
    }
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

//: ABI version of the compiled kernel. The Python loader refuses a library whose
//: version it does not know, so a stale artifact can never be silently used.
int32_t relic_cpu_block_gemv_abi_version(void) { return kAbiVersion; }

//: Exact M = 1 block-32 FP8 GEMV.
//:
//:   a_codes   [k]                        e4m3 bytes
//:   a_scale   [k / 32]                   raw E8M0 bytes
//:   b_codes   [n, k]                     e4m3 bytes, row-major
//:   b_scale   [n / 32, k / 32]           raw E8M0 bytes, block_rows == 32
//:             [n, k / 32]                raw E8M0 bytes, block_rows == 1
//:   out       [n]                        binary32 accumulators
//:
//: `threads` splits the output columns; every column is computed identically
//: whatever the thread count, so the result does not depend on it.
//: `threads_used` may be null and otherwise receives the number of worker threads
//: actually started.
int32_t relic_cpu_block_gemv_fp8(const uint8_t *a_codes, const uint8_t *a_scale,
                                 const uint8_t *b_codes, const uint8_t *b_scale,
                                 float *out, int64_t n, int64_t k,
                                 int32_t block_rows, int32_t threads,
                                 int32_t *threads_used) {
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
    EnsurePacked();
    const int64_t blocks = k / kBlock;

    // The activation side is shared by every output column, so it is decoded
    // once: one value and one nominal exponent per (K block, column of the K
    // block), and one scale power per K block.
    std::vector<float> a_value(static_cast<size_t>(k));
    std::vector<int32_t> a_nominal(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
      const uint32_t packed = g_packed[a_codes[i]];
      const float value = PackedValue(packed);
      if (value != value) {
        // A NaN anywhere in the activation row poisons every column, and the
        // zero-block loop below never has to run.
        for (int64_t column = 0; column < n; ++column) {
          out[column] = std::numeric_limits<float>::quiet_NaN();
        }
        if (threads_used != nullptr) {
          *threads_used = 1;
        }
        return kOk;
      }
      a_value[static_cast<size_t>(i)] = value;
      // Unbiased here: the packed weight word carries its own bias, which the
      // block loop removes once per half after summing the two nominals.
      a_nominal[static_cast<size_t>(i)] =
          PackedNominal(packed) - kNominalBias;
    }
    std::vector<float> a_scale_power(static_cast<size_t>(blocks));
    for (int64_t block = 0; block < blocks; ++block) {
      a_scale_power[static_cast<size_t>(block)] =
          Pow2(static_cast<int32_t>(a_scale[block]) - 127);
    }

    const int64_t scale_rows = block_rows == 1 ? n : n / block_rows;
    std::vector<float> b_scale_power(static_cast<size_t>(scale_rows * blocks));
    for (int64_t row = 0; row < scale_rows; ++row) {
      for (int64_t block = 0; block < blocks; ++block) {
        b_scale_power[static_cast<size_t>(row * blocks + block)] =
            Pow2(static_cast<int32_t>(b_scale[row * blocks + block]) - 127);
      }
    }

    const int32_t workers = static_cast<int32_t>(std::min<int64_t>(threads, n));
    if (threads_used != nullptr) {
      *threads_used = workers;
    }
    if (workers <= 1) {
      Job job{};
      job.a_value = a_value.data();
      job.a_nominal = a_nominal.data();
      job.a_scale = a_scale_power.data();
      job.b_codes = b_codes;
      job.b_scale = b_scale_power.data();
      job.n = n;
      job.k = k;
      job.blocks = blocks;
      job.block_rows = block_rows;
      job.out = out;
      job.begin = 0;
      job.end = n;
      RunJob(&job);
      return kOk;
    }

    std::vector<Job> jobs(static_cast<size_t>(workers));
    const int64_t chunk = (n + workers - 1) / workers;
    for (int32_t worker = 0; worker < workers; ++worker) {
      Job &job = jobs[static_cast<size_t>(worker)];
      job.a_value = a_value.data();
      job.a_nominal = a_nominal.data();
      job.a_scale = a_scale_power.data();
      job.b_codes = b_codes;
      job.b_scale = b_scale_power.data();
      job.n = n;
      job.k = k;
      job.blocks = blocks;
      job.block_rows = block_rows;
      job.out = out;
      job.begin = std::min<int64_t>(worker * chunk, n);
      job.end = std::min<int64_t>(job.begin + chunk, n);
    }

    // The calling thread takes the first chunk, so the result is produced even if
    // the platform refuses to start a single worker. A creation failure cannot be
    // allowed to escape while `pool` still holds joinable threads -- that would
    // terminate the process -- so the remaining chunks are drained here instead.
    RunJob(&jobs[0]);
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    size_t spawned = 0;
    try {
      for (int32_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back(RunJob, &jobs[static_cast<size_t>(worker)]);
        ++spawned;
      }
    } catch (...) {
      // Fall through: the chunks whose threads never started run below.
    }
    for (int32_t worker = static_cast<int32_t>(spawned) + 1; worker < workers; ++worker) {
      RunJob(&jobs[static_cast<size_t>(worker)]);
    }
    if (threads_used != nullptr) {
      *threads_used = static_cast<int32_t>(spawned) + 1;
    }
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