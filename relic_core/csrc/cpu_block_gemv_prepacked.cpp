// Prepacked, column-vectorised exact host-CPU block-32 FP8 GEMV (ABI v2).
//
// This is the second, opt-in implementation of the same reduction
// `cpu_block_gemv.cpp` (ABI v1) computes, and it is bit-identical to it. What it
// adds is a load-time prepack that removes the per-lane byte-table gather from the
// inner loop, plus a four-output-column SIMD path:
//
//   * prepack (`relic_cpu_block_gemv_prepacked_prepack`) turns every stored e4m3
//     weight byte into the same 32-bit decode word v1 builds in registers
//     (`g_packed[byte]`: the value's binary32 bits above bit 20, plus the biased
//     nominal exponent). It writes them as [block][column][lane], so one output
//     column's 32 lanes for one K block are 128 contiguous bytes, and it resolves
//     each column's NaN classification and each block scale power once.
//   * compute (`relic_cpu_block_gemv_prepacked_fp8`) then loads those words
//     straight into NEON registers and vectorises **across the 32 lanes of one
//     column**: the same `a_value`/`a_nominal` the activation decode produced are
//     contiguous vectors there too, so an anchor is an integer max tree and the
//     half sum is `trunc((value * a) * inverse)` accumulated as int32. The window,
//     carry, final cut and scale order stay the scalar statements v1 runs, because
//     they depend on the column's own anchor and are a small fraction of the work.
//
// The arithmetic is unchanged, and that is the point: every step below is exactly
// representable in binary32, so the result is the same bits as the Torch oracle
// rather than a tolerance-matched approximation.
//
//   1. A code's value and its nominal exponent F(v) = max(true exponent, -6) come
//      from the prepared word; a signed-zero code carries the kEmpty sentinel.
//   2. The 32 products of one K block are split into two 16-term halves. Each is
//      truncated toward zero onto 2**(anchor - 13), anchor being the largest
//      summed nominal exponent in that half.
//   3. The running sum rescales by the inverse step onto the second half's window,
//      which is max(second anchor, floor(log2|first half sum|)).
//   4. The block result is renormalised once more onto 2**(floor(log2|total|) - 13).
//   5. Activation scale first, weight scale second, both FP32, then the K blocks
//      accumulate one at a time in increasing K order.
//
// Exactness bounds (the same argument as v1): a product of two e4m3 codes has at
// most eight significant bits and lives in [2**-18, 2**18]; the anchor is at least
// the true exponent of every product in its half, so each truncated term is below
// 2**14 and a 16-term sum below 2**18 -- whole numbers, exact in binary32. The
// 13-bit windows keep the running and final results in the same range. The only
// rounding is the per-K-block FP32 accumulation, which is where the reference
// rounds too. No FMA may be introduced (`-ffp-contract=off`), and the two scale
// multiplies stay separate.
//
// The SIMD path only changes *how* the identical operands are combined: integer
// max for the anchors and per-lane `trunc((value * a) * inverse)` summed as int32
// are exact integer operations, so lane order inside a half cannot matter. The
// half sums are then combined by the same scalar sequence v1 uses.
//
// The word layout is [block][column][lane] rather than lane-major on purpose. A
// column-block is then one 128-byte contiguous run, which keeps the prepack a
// sequential write and the compute a sequential read of the derived form; the
// lane-major layout measured 3x slower than v1 because every vector load landed in
// a different cache line and the 4x footprint was paid with no locality to show for
// it. Vectorising the lanes also makes the activation operands contiguous vector
// loads instead of one broadcast per lane.
//
// Non-finite input. An e4m3 byte is a NaN code exactly when (byte & 0x7F) == 0x7F.
// A NaN in the activation row makes every column NaN; a NaN in a column's weights
// makes that column NaN. Both are the reference's own propagation. The weight side
// is resolved once at prepack into the per-column flag, so the compute loop never
// has to convert a NaN to an integer. Byte 255 scale bytes would mean 2**128,
// which no binary32 holds; the Python binding refuses them before they arrive, so
// this translation unit stays a plain primitive (it maps 255 to infinity exactly
// as v1 documents) while the public contract is finite grids only.
//
// Not here: no BLAS, no float64, no native FP8 cast, no BF16 expansion, no
// checkpoint rewrite. The prepacked words are a private derived form; the stored
// weight bytes stay the source of truth.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#define RELIC_GEMV_PREPACKED_NEON 1
#else
#define RELIC_GEMV_PREPACKED_NEON 0
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#define RELIC_GEMV_PREPACKED_ATFORK 1
#else
#define RELIC_GEMV_PREPACKED_ATFORK 0
#endif

namespace {

constexpr int32_t kBlock = 32;
constexpr int32_t kHalf = 16;
constexpr int32_t kWindow = 13;

//: The nominal exponent of a signed-zero code (see v1 for the full argument).
constexpr int32_t kEmpty = -1000;
constexpr int32_t kEmptyGuard = -500;
//: "This running sum has no exponent."
constexpr int32_t kNoExponent = -1000000;

constexpr int32_t kAbiVersion = 2;

constexpr int32_t kOk = 0;
constexpr int32_t kBadArguments = 1;
constexpr int32_t kOutOfMemory = 2;

//: kValueMask keeps the value's binary32 bits above bit 20; kNominalMask keeps the
//: biased nominal exponent. This is v1's decode word, reproduced bit for bit --
//: sharing the definition is what makes the two kernels agree about every code.
constexpr int32_t kNominalBias = 1000;
constexpr uint32_t kValueMask = 0xFFF00000u;
constexpr uint32_t kNominalMask = 0x7FFu;

//: Compute variants. `kAuto` uses the SIMD path where it was compiled in and the
//: portable path otherwise; `kScalar` is a test hook that forces the portable path
//: on any host, so the fallback can be exercised on a machine whose native path is
//: the SIMD one.
constexpr int32_t kAuto = 0;
constexpr int32_t kScalar = 1;

alignas(64) uint32_t g_packed[256];
std::once_flag g_packed_once;

void InitPacked() {
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

//: 2**exponent as the nearest representable binary32, from its IEEE encoding, so
//: no library `exp2` approximation can move a grid by an octave. Exact for every
//: exponent the scales and windows can produce, including the subnormal range an
//: E8M0 byte 0 needs and the infinity a byte 255 asks for.
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
//: half has no anchor at all. Identical to v1's, including the sentinel branch.
inline void CutPowers(int32_t anchor, float *inverse, float *step) {
  if (anchor < kEmptyGuard) {
    *inverse = 1.0f;
    *step = 1.0f;
    return;
  }
  *inverse = Pow2(kWindow - anchor);
  *step = Pow2(anchor - kWindow);
}

//: Whether any of 32 consecutive e4m3 bytes is a NaN code. Eight bytes at a time;
//: a false positive only marks the column invalid, which is the same answer.
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

//: The portable single-column block reduction, over prepacked words. This is v1's
//: `BlockFragment` with the table lookup already applied; the two halves, the
//: running window, the carry and the final cut are the same statements in the same
//: order, so the non-SIMD host and the SIMD host answer identically.
inline float FragmentScalar(const uint32_t *words, const float *a_value,
                            const int32_t *a_nominal) {
  int32_t anchor_first = std::numeric_limits<int32_t>::min();
  for (int32_t i = 0; i < kHalf; ++i) {
    const int32_t summed =
        a_nominal[i] + static_cast<int32_t>(words[i] & kNominalMask);
    anchor_first = std::max(anchor_first, summed);
  }
  anchor_first -= kNominalBias;
  float inverse_first;
  float step_first;
  CutPowers(anchor_first, &inverse_first, &step_first);
  int32_t whole_first = 0;
  for (int32_t i = 0; i < kHalf; ++i) {
    whole_first += static_cast<int32_t>(
        a_value[i] * PackedValue(words[i]) * inverse_first);
  }
  const float total_first = static_cast<float>(whole_first) * step_first;
  const int32_t running =
      total_first != 0.0f ? FloorExponent(total_first) : kNoExponent;

  int32_t anchor_second = std::numeric_limits<int32_t>::min();
  for (int32_t i = kHalf; i < kBlock; ++i) {
    const int32_t summed =
        a_nominal[i] + static_cast<int32_t>(words[i] & kNominalMask);
    anchor_second = std::max(anchor_second, summed);
  }
  anchor_second -= kNominalBias;
  const int32_t window = std::max(anchor_second, running);
  float inverse_second;
  float step_second;
  CutPowers(window, &inverse_second, &step_second);
  int32_t whole_second = 0;
  for (int32_t i = kHalf; i < kBlock; ++i) {
    whole_second += static_cast<int32_t>(
        a_value[i] * PackedValue(words[i]) * inverse_second);
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

#if RELIC_GEMV_PREPACKED_NEON

//: One 16-lane half of one column, loaded as four 128-bit registers.
struct HalfWords {
  uint32x4_t lane[4];
};

inline HalfWords LoadHalf(const uint32_t *words, int32_t first) {
  HalfWords half;
  for (int32_t group = 0; group < 4; ++group) {
    half.lane[group] = vld1q_u32(words + first + group * 4);
  }
  return half;
}

//: Largest summed nominal exponent in the half, still biased by kNominalBias.
//: Integer max over the 16 lanes; a max is order independent, so a tree over four
//: vectors gives the same value as v1's sequential max.
inline int32_t HalfAnchor(const HalfWords &half, const int32_t *a_nominal, int32_t first) {
  const uint32x4_t nominal_mask = vdupq_n_u32(kNominalMask);
  int32x4_t best = vdupq_n_s32(std::numeric_limits<int32_t>::min());
  for (int32_t group = 0; group < 4; ++group) {
    int32x4_t nominal =
        vreinterpretq_s32_u32(vandq_u32(half.lane[group], nominal_mask));
    nominal = vaddq_s32(nominal, vld1q_s32(a_nominal + first + group * 4));
    best = vmaxq_s32(best, nominal);
  }
  return vmaxvq_s32(best);
}

//: The half's truncated integer sum. Each lane is
//: `trunc((weight_value * activation_value) * inverse)` -- two explicit FP32
//: multiplies in the reference's order, then a truncation toward zero -- and the
//: per-lane integers are summed exactly (each below 2**14, the 16-term bound below
//: 2**18), so neither the lane order nor the reduction tree can change the result.
inline int32_t HalfWhole(const HalfWords &half, const float *a_value, int32_t first,
                         float inverse) {
  const uint32x4_t value_mask = vdupq_n_u32(kValueMask);
  const float32x4_t inverse_vector = vdupq_n_f32(inverse);
  int32x4_t whole = vdupq_n_s32(0);
  for (int32_t group = 0; group < 4; ++group) {
    const float32x4_t value =
        vreinterpretq_f32_u32(vandq_u32(half.lane[group], value_mask));
    const float32x4_t product =
        vmulq_f32(vmulq_f32(value, vld1q_f32(a_value + first + group * 4)),
                  inverse_vector);
    whole = vaddq_s32(whole, vcvtq_s32_f32(product));
  }
  return vaddvq_s32(whole);
}

#endif  // RELIC_GEMV_PREPACKED_NEON

//: One K block's fragment for one output column over its 32 prepacked words.
//:
//: The two 16-lane halves are vectorised (anchor max, product truncation and
//: integer sum); the per-column window, carry, final cut and renormalisation are
//: the scalar statements from `FragmentScalar`, because they depend on this
//: column's own anchor. That split is deliberate: the vector part is exactly
//: reproducible as integers, and the part that is not stays literally the same code
//: as the portable path.
inline float FragmentOne(const uint32_t *words, const float *a_value,
                         const int32_t *a_nominal) {
#if RELIC_GEMV_PREPACKED_NEON
  const HalfWords first = LoadHalf(words, 0);
  int32_t anchor_first = HalfAnchor(first, a_nominal, 0) - kNominalBias;
  float inverse_first;
  float step_first;
  CutPowers(anchor_first, &inverse_first, &step_first);
  const int32_t whole_first = HalfWhole(first, a_value, 0, inverse_first);

  const float total_first = static_cast<float>(whole_first) * step_first;
  const int32_t running =
      total_first != 0.0f ? FloorExponent(total_first) : kNoExponent;

  const HalfWords second = LoadHalf(words, kHalf);
  const int32_t anchor_second = HalfAnchor(second, a_nominal, kHalf) - kNominalBias;
  const int32_t window = std::max(anchor_second, running);
  float inverse_second;
  float step_second;
  CutPowers(window, &inverse_second, &step_second);
  const int32_t whole_second = HalfWhole(second, a_value, kHalf, inverse_second);

  const float total =
      (static_cast<float>(whole_second) + std::trunc(total_first * inverse_second)) *
      step_second;
  if (total == 0.0f) {
    return total;
  }
  const int32_t exponent = FloorExponent(total);
  const float step = Pow2(exponent - kWindow);
  return std::trunc(total / step) * step;
#else
  return FragmentScalar(words, a_value, a_nominal);
#endif
}

struct Columns {
  const uint32_t *words;
  const float *scale_powers;
  const uint8_t *valid;
  const float *a_value;
  const int32_t *a_nominal;
  const float *a_scale_power;
  float *out;
  int64_t n;
  int64_t blocks;
  int32_t block_rows;
  int32_t variant;
};

//: The columns `[begin, end)` of one call. A column is one independent dot
//: product, so splitting the output columns partitions the work exactly.
//:
//: ``words`` is indexed [block][column][lane]: one column's 32 lanes for one K
//: block are 128 contiguous bytes, so this loop walks the derived form once,
//: sequentially, in the same order it was written.
void RunColumns(const Columns &job, int64_t begin, int64_t end) {
  for (int64_t column = begin; column < end; ++column) {
    if (job.valid[column] == 0) {
      job.out[column] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    float accumulator = 0.0f;
    for (int64_t block = 0; block < job.blocks; ++block) {
      const uint32_t *block_words =
          job.words + (block * job.n + column) * kBlock;
      const float fragment = FragmentOne(block_words, job.a_value + block * kBlock,
                                         job.a_nominal + block * kBlock);
      const int64_t scale_row =
          job.block_rows == 1 ? column : column / job.block_rows;
      const float weight_scale = job.scale_powers[scale_row * job.blocks + block];
      // Activation scale first, weight scale second, then the K add.
      accumulator += (fragment * job.a_scale_power[block]) * weight_scale;
    }
    job.out[column] = accumulator;
  }
}

//: A process-wide worker pool, so a decode step does not pay thread creation per
//: token. Workers wait on a generation counter; the calling thread publishes one
//: `Columns` job and waits for every chunk to report back. If the pool cannot be
//: created the caller runs the chunks inline, which is slower but identical.
//:
//: The pool is process-wide state, so a `fork()` after first use is handled
//: explicitly: the child resets the pointer (leaking the parent's object, whose
//: threads do not exist there) and builds a fresh pool on its next call. A child
//: that never calls this operator never touches any of it.
struct Pool {
  explicit Pool(int32_t count) : workers(count) {}
  int32_t workers;
  std::mutex mutex;
  std::condition_variable work_cv;
  std::condition_variable done_cv;
  std::vector<std::thread> threads;
  bool stopping = false;
  uint64_t generation = 0;
  int32_t pending = 0;
  Columns job{};
};

std::atomic<Pool *> g_pool{nullptr};
std::atomic<int32_t> g_pool_workers{0};
std::mutex g_pool_build_mutex;
std::once_flag g_atfork_once;

void WorkerLoop(Pool *pool, int32_t index) {
  uint64_t seen = 0;
  for (;;) {
    Columns job{};
    int64_t begin = 0;
    int64_t end = 0;
    {
      std::unique_lock<std::mutex> lock(pool->mutex);
      pool->work_cv.wait(lock, [&] {
        return pool->stopping || pool->generation != seen;
      });
      if (pool->stopping) {
        return;
      }
      seen = pool->generation;
      job = pool->job;
      const int64_t chunk = (job.n + pool->workers - 1) / pool->workers;
      begin = std::min<int64_t>(static_cast<int64_t>(index) * chunk, job.n);
      end = std::min<int64_t>(begin + chunk, job.n);
    }
    if (begin < end) {
      RunColumns(job, begin, end);
    }
    {
      std::lock_guard<std::mutex> lock(pool->mutex);
      if (--pool->pending == 0) {
        pool->done_cv.notify_one();
      }
    }
  }
}

void DestroyPool(Pool *pool) {
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->stopping = true;
  }
  pool->work_cv.notify_all();
  for (std::thread &worker : pool->threads) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

#if RELIC_GEMV_PREPACKED_ATFORK
void AtForkChild() {
  // The parent's worker threads do not exist in the child and its mutexes may be
  // held by threads that are gone, so the object is abandoned rather than reused.
  g_pool.store(nullptr);
  g_pool_workers.store(0);
}
#endif

//: The pool for `workers`, rebuilt if the requested width changed. Null means the
//: platform refused to start any worker and the caller must run inline.
Pool *AcquirePool(int32_t workers) {
#if RELIC_GEMV_PREPACKED_ATFORK
  std::call_once(g_atfork_once, [] { pthread_atfork(nullptr, nullptr, AtForkChild); });
#endif
  Pool *existing = g_pool.load();
  if (existing != nullptr && existing->workers == workers) {
    return existing;
  }
  std::lock_guard<std::mutex> guard(g_pool_build_mutex);
  existing = g_pool.load();
  if (existing != nullptr) {
    if (existing->workers == workers) {
      return existing;
    }
    g_pool.store(nullptr);
    DestroyPool(existing);
    delete existing;
  }
  Pool *pool = new Pool(workers);
  try {
    pool->threads.reserve(static_cast<size_t>(workers));
    for (int32_t index = 0; index < workers; ++index) {
      pool->threads.emplace_back(WorkerLoop, pool, index);
    }
  } catch (...) {
    DestroyPool(pool);
    delete pool;
    return nullptr;
  }
  g_pool.store(pool);
  g_pool_workers.store(workers);
  return pool;
}

//: Run one call on the pool. Returns false when the pool could not be used.
bool RunOnPool(int32_t workers, const Columns &job) {
  Pool *pool = AcquirePool(workers);
  if (pool == nullptr) {
    return false;
  }
  {
    std::unique_lock<std::mutex> lock(pool->mutex);
    pool->job = job;
    pool->pending = workers;
    ++pool->generation;
    pool->work_cv.notify_all();
    pool->done_cv.wait(lock, [&] { return pool->pending == 0; });
  }
  return true;
}

int32_t Validate(int64_t n, int64_t k, int32_t block_rows, int32_t threads,
                 int32_t variant) {
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
  if (variant != kAuto && variant != kScalar) {
    return kBadArguments;
  }
  // The prepacked word count is blocks * 32 * n; keep the index arithmetic sound.
  const int64_t blocks = k / kBlock;
  if (n > std::numeric_limits<int64_t>::max() / (blocks * kBlock)) {
    return kBadArguments;
  }
  return kOk;
}

}  // namespace

extern "C" {

//: ABI version of this kernel: 2. The Python loader refuses any other value, and
//: the artifact name carries both the version and a fingerprint of this source, so
//: a v1 library can never be loaded in its place.
int32_t relic_cpu_block_gemv_prepacked_abi_version(void) { return kAbiVersion; }

//: Prepack one weight matrix.
//:
//:   b_codes       [n, k]                e4m3 bytes, row-major (read-only)
//:   b_scale       [n / 32, k / 32]      raw E8M0 bytes, block_rows == 32
//:                 [n, k / 32]           raw E8M0 bytes, block_rows == 1
//:   words         [k / 32, n, 32]       uint32 decode words, written
//:   scale_powers  [scale_rows, k / 32]  binary32 powers, written
//:   valid         [n]                   uint8: 0 when the column has a NaN code
//:
//: The buffers are sized and allocated by the caller; their extents are the
//: statement about how much host memory the prepacked form costs.
int32_t relic_cpu_block_gemv_prepacked_prepack(const uint8_t *b_codes,
                                               const uint8_t *b_scale,
                                               uint32_t *words, float *scale_powers,
                                               uint8_t *valid, int64_t n, int64_t k,
                                               int32_t block_rows) {
  if (b_codes == nullptr || b_scale == nullptr || words == nullptr ||
      scale_powers == nullptr || valid == nullptr) {
    return kBadArguments;
  }
  const int32_t status = Validate(n, k, block_rows, 1, kAuto);
  if (status != kOk) {
    return status;
  }
  try {
    EnsurePacked();
    const int64_t blocks = k / kBlock;
    // Every column starts valid; a NaN code in any of its blocks clears it, which
    // is the same answer as v1's per-block test (a NaN block returns NaN for the
    // whole column, and the value is NaN whichever block reported it).
    std::memset(valid, 1, static_cast<size_t>(n));
    for (int64_t block = 0; block < blocks; ++block) {
      for (int64_t column = 0; column < n; ++column) {
        const uint8_t *codes = b_codes + column * k + block * kBlock;
        if (valid[column] != 0 && HasNanCode(codes)) {
          valid[column] = 0;
        }
        uint32_t *block_words = words + (block * n + column) * kBlock;
        for (int32_t lane = 0; lane < kBlock; ++lane) {
          block_words[lane] = g_packed[codes[lane]];
        }
      }
    }
    const int64_t scale_rows = block_rows == 1 ? n : n / block_rows;
    for (int64_t row = 0; row < scale_rows; ++row) {
      for (int64_t block = 0; block < blocks; ++block) {
        const int64_t at = row * blocks + block;
        scale_powers[at] = Pow2(static_cast<int32_t>(b_scale[at]) - 127);
      }
    }
    return kOk;
  } catch (const std::bad_alloc &) {
    return kOutOfMemory;
  } catch (...) {
    return kBadArguments;
  }
}

//: Exact M = 1 block-32 FP8 GEMV against a prepacked weight.
//:
//:   a_codes       [k]                      e4m3 bytes
//:   a_scale       [k / 32]                 raw E8M0 bytes
//:   words         [k / 32, n, 32]          from the prepack above
//:   scale_powers  [scale_rows, k / 32]     from the prepack above
//:   valid         [n]                      from the prepack above
//:   out           [n]                      binary32 accumulators
//:   variant       kAuto (SIMD where compiled in, portable otherwise) or kScalar
//:
//: `threads` splits the output columns; every column is computed identically
//: whatever the thread count, so the result does not depend on it.
int32_t relic_cpu_block_gemv_prepacked_fp8(const uint8_t *a_codes,
                                           const uint8_t *a_scale,
                                           const uint32_t *words,
                                           const float *scale_powers,
                                           const uint8_t *valid, float *out, int64_t n,
                                           int64_t k, int32_t block_rows,
                                           int32_t threads, int32_t variant,
                                           int32_t *threads_used) {
  if (threads_used != nullptr) {
    *threads_used = 0;
  }
  if (a_codes == nullptr || a_scale == nullptr || words == nullptr ||
      scale_powers == nullptr || valid == nullptr || out == nullptr) {
    return kBadArguments;
  }
  const int32_t status = Validate(n, k, block_rows, threads, variant);
  if (status != kOk) {
    return status;
  }
  try {
    EnsurePacked();
    const int64_t blocks = k / kBlock;

    // The activation side is shared by every output column, so it is decoded once
    // -- the same decode v1 performs, from the same table.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> a_value(static_cast<size_t>(k));
    std::vector<int32_t> a_nominal(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
      const uint32_t packed = g_packed[a_codes[i]];
      const float value = PackedValue(packed);
      if (value != value) {
        for (int64_t column = 0; column < n; ++column) {
          out[column] = nan;
        }
        if (threads_used != nullptr) {
          *threads_used = 1;
        }
        return kOk;
      }
      a_value[static_cast<size_t>(i)] = value;
      a_nominal[static_cast<size_t>(i)] =
          static_cast<int32_t>(packed & kNominalMask) - kNominalBias;
    }
    std::vector<float> a_scale_power(static_cast<size_t>(blocks));
    for (int64_t block = 0; block < blocks; ++block) {
      a_scale_power[static_cast<size_t>(block)] =
          Pow2(static_cast<int32_t>(a_scale[block]) - 127);
    }

    Columns job{};
    job.words = words;
    job.scale_powers = scale_powers;
    job.valid = valid;
    job.a_value = a_value.data();
    job.a_nominal = a_nominal.data();
    job.a_scale_power = a_scale_power.data();
    job.out = out;
    job.n = n;
    job.blocks = blocks;
    job.block_rows = block_rows;
    job.variant = variant;

    const int32_t workers = static_cast<int32_t>(std::min<int64_t>(threads, n));
    if (workers <= 1) {
      RunColumns(job, 0, n);
      if (threads_used != nullptr) {
        *threads_used = 1;
      }
      return kOk;
    }
    if (RunOnPool(workers, job)) {
      if (threads_used != nullptr) {
        *threads_used = workers;
      }
      return kOk;
    }
    // The platform refused the worker threads. Running the columns here is slower
    // and produces the same numbers, which is why it is a fallback rather than an
    // error: nothing about the answer depends on how the columns were scheduled.
    RunColumns(job, 0, n);
    if (threads_used != nullptr) {
      *threads_used = 1;
    }
    return kOk;
  } catch (const std::bad_alloc &) {
    return kOutOfMemory;
  } catch (...) {
    return kBadArguments;
  }
}

}  // extern "C"