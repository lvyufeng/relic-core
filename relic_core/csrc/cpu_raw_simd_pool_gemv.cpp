// Raw-byte NEON host GEMV with a persistent worker pool.
//
// This computes exactly what `cpu_raw_simd_gemv.cpp` computes -- the same
// accepted block-32 truncation rule, over the same raw original bytes, with the
// same tables and the same sequential K order -- and differs in one place only:
// **where the threads come from**.
//
// The frozen kernel created `threads - 1` `std::thread`s and joined them on
// every call. For a repeated 96-thread projection that is 95 thread creations
// and joins per GEMV, and because the callers of a freshly spawned thread start
// claiming columns while the loop is still spawning the rest, the early part of
// a call runs with fewer workers than it asked for. This file keeps one
// process-local pool alive: the workers wait on a condition variable, a call
// publishes its plan, every worker starts from that same publication, and the
// call returns once the last of them has finished.
//
// The arithmetic is not re-derived here: the table, the block reduction, the
// plan and the range loop between the two `RELIC_RAW_SIMD_POOL_MATH` markers are
// byte-identical to the frozen translation unit, and
// `tests/test_cpu_raw_simd_pool_gemv.py` pins both that text and its SHA-256.
// Every output column is still computed independently of the thread count: the
// column partition changes, the K order inside a column does not.
//
// Pool contract:
//
//   * one active geometry per process -- a call that asks for a different
//     `threads` resizes the pool, and only while every worker is idle, so a
//     host never runs two 96-worker pools side by side;
//   * concurrent callers are serialized by a mutex and each is synchronous: no
//     pointer the caller passed outlives the call, and the plan is cleared
//     before returning;
//   * the calling thread takes part in the work, so the columns are produced
//     even if the platform refuses to start a worker, and the reported thread
//     count is the number that actually ran;
//   * a worker that throws sets an error the caller returns, after the pool is
//     quiesced; no exception crosses the C boundary;
//   * a fork leaves the child with inherited bookkeeping but no threads. The
//     child's first call refuses by name rather than joining workers that do not
//     exist; `relic_cpu_raw_simd_pool_gemv_reset()` (which the Python binding
//     calls from `os.register_at_fork`) discards that state so the child can
//     build its own pool;
//   * the pool is stopped and joined at normal process exit.
//
// No CPU affinity, NUMA policy or global scheduler setting is touched.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__ARM_NEON) && !defined(RELIC_RAW_SIMD_FORCE_SCALAR)
#define RELIC_RAW_SIMD_NEON 1
#include <arm_neon.h>
#endif

// RELIC_RAW_SIMD_POOL_MATH_BEGIN  (verbatim from cpu_raw_simd_gemv.cpp)
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
// RELIC_RAW_SIMD_POOL_MATH_END

// ---------------------------------------------------------------------------
// Persistent pool. Everything below is scheduling; nothing below computes.
// ---------------------------------------------------------------------------

//: A call found a pool created before a `fork()` and no
//: `relic_cpu_raw_simd_pool_gemv_reset()` since. Distinct from a bad argument so
//: a caller can tell "your operands are wrong" from "call reset in the child".
constexpr int32_t kStaleFork = 3;

struct Pool;

void PoolWorker(Pool *pool);

struct Pool {
  //: Serializes whole calls, so a second caller cannot resize the pool while
  //: the first one's session is still being worked on.
  std::mutex call_mutex;
  std::mutex mutex;
  std::condition_variable wake;
  std::condition_variable done;
  std::vector<std::thread> threads;
  //: The published call. Only read by a worker after it has seen the session
  //: counter advance, and only written by the caller while every worker waits.
  Plan plan{};
  int64_t n = 0;
  int64_t chunk = 1;
  std::atomic<int64_t> next{0};
  int64_t generation = 0;
  int32_t spawned = 0;    // workers this pool owns, excluding the caller
  int32_t remaining = 0;  // workers that have not finished this session
  int32_t error = 0;      // sticky status from a worker exception
  bool stop = false;
  int64_t owner_pid = 0;
};

void PoolWorker(Pool *pool) {
  int64_t seen = 0;
  std::unique_lock<std::mutex> lock(pool->mutex);
  for (;;) {
    pool->wake.wait(lock, [pool, seen] {
      return pool->stop || pool->generation != seen;
    });
    if (pool->stop) {
      return;
    }
    seen = pool->generation;
    // One consistent snapshot under the lock: the caller may clear these the
    // moment the session reports complete, and the worker must not read them
    // after that.
    const Plan plan = pool->plan;
    const int64_t n = pool->n;
    const int64_t chunk = pool->chunk;
    lock.unlock();
    int32_t failure = 0;
    for (;;) {
      const int64_t begin = pool->next.fetch_add(chunk);
      if (begin >= n) {
        break;
      }
      const int64_t end = std::min(begin + chunk, n);
      try {
        RunRange(plan, begin, end);
      } catch (...) {
        failure = kBadArguments;
        break;
      }
    }
    lock.lock();
    if (failure != 0) {
      pool->error = failure;
    }
    if (--pool->remaining == 0) {
      pool->done.notify_one();
    }
  }
}

//: Stop and join this process's workers. Must be called with `lock` held and
//: returns with it held; it releases the lock across the joins because a worker
//: needs the lock to leave its wait.
void PoolResize(Pool &pool, int32_t wanted, std::unique_lock<std::mutex> &lock) {
  if (pool.spawned == wanted) {
    return;
  }
  pool.stop = true;
  pool.wake.notify_all();
  lock.unlock();
  for (std::thread &worker : pool.threads) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  pool.threads.clear();
  lock.lock();
  pool.stop = false;
  pool.spawned = 0;
  pool.threads.reserve(static_cast<size_t>(wanted));
  for (int32_t index = 0; index < wanted; ++index) {
    // A creation failure keeps the workers that did start: they are the same
    // participating threads the frozen kernel's fallback leaves running.
    try {
      pool.threads.emplace_back(PoolWorker, &pool);
    } catch (...) {
      break;
    }
  }
  pool.spawned = static_cast<int32_t>(pool.threads.size());
}

void PoolShutdown(Pool *pool) {
  // An inherited pool must not be joined: in a forked child its thread objects
  // name threads that do not exist, and detaching or destroying one terminates
  // the process. Such a pool is left in place and a new one is built instead.
  if (pool->owner_pid != static_cast<int64_t>(::getpid())) {
    return;
  }
  std::unique_lock<std::mutex> lock(pool->mutex);
  pool->stop = true;
  pool->wake.notify_all();
  lock.unlock();
  for (std::thread &worker : pool->threads) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  pool->threads.clear();
}

struct PoolHolder {
  std::atomic<Pool *> current{nullptr};

  ~PoolHolder() {
    Pool *pool = current.load(std::memory_order_acquire);
    if (pool != nullptr) {
      PoolShutdown(pool);
    }
  }
};

PoolHolder &Holder() {
  static PoolHolder holder;
  return holder;
}

//: This process's pool, or nullptr if the only one there is predates a fork.
Pool *AcquirePool() {
  const int64_t pid = static_cast<int64_t>(::getpid());
  PoolHolder &holder = Holder();
  for (int32_t attempt = 0; attempt < 4; ++attempt) {
    Pool *pool = holder.current.load(std::memory_order_acquire);
    if (pool != nullptr) {
      return pool->owner_pid == pid ? pool : nullptr;
    }
    Pool *fresh = new (std::nothrow) Pool();
    if (fresh == nullptr) {
      return nullptr;
    }
    fresh->owner_pid = pid;
    Pool *expected = nullptr;
    if (holder.current.compare_exchange_strong(expected, fresh,
                                               std::memory_order_release,
                                               std::memory_order_acquire)) {
      return fresh;
    }
    delete fresh;  // lost the race; this one has started no thread
  }
  Pool *pool = holder.current.load(std::memory_order_acquire);
  return (pool != nullptr && pool->owner_pid == pid) ? pool : nullptr;
}

//: Discard this process's pool bookkeeping without touching it. Called in a
//: forked child (there, the inherited state names threads that do not exist) so
//: the next call builds a pool of its own. The old object is deliberately not
//: destroyed: doing so would join or detach nonexistent threads.
void ResetPool() {
  PoolHolder &holder = Holder();
  Pool *pool = holder.current.load(std::memory_order_acquire);
  // A parent-side reset must not abandon live workers and create a second pool.
  if (pool == nullptr || pool->owner_pid == static_cast<int64_t>(::getpid())) {
    return;
  }
  holder.current.store(nullptr, std::memory_order_release);
  // No other thread survived the fork. Its copied once flag may have been busy.
  new (&g_init_once) std::once_flag();
}

}  // namespace

extern "C" {

//: ABI version of the compiled kernel; the loader refuses an unknown one.
int32_t relic_cpu_raw_simd_pool_gemv_abi_version(void) { return kAbiVersion; }

//: 4 = the NEON decode across K lanes, 1 = the portable scalar form.
int32_t relic_cpu_raw_simd_pool_gemv_vector_width(void) {
#ifdef RELIC_RAW_SIMD_NEON
  return 4;
#else
  return 1;
#endif
}

//: Drop the pool state a forked child inherited; safe to call from
//: `os.register_at_fork(after_in_child=...)` because it only publishes a null
//: pointer and allocates nothing.
void relic_cpu_raw_simd_pool_gemv_reset(void) { ResetPool(); }

//: Exact M = 1 block-32 FP8 GEMV, reusing one process-local worker pool.
//:
//:   a_codes   [k]                        e4m3 bytes
//:   a_scale   [k / 32]                   raw E8M0 bytes
//:   b_codes   [n, k]                     e4m3 bytes, row-major, no prepack
//:   b_scale   [n / 32, k / 32]           raw E8M0 bytes, block_rows == 32
//:             [n, k / 32]                raw E8M0 bytes, block_rows == 1
//:   out       [n]                        binary32 accumulators
//:
//: The caller owns every buffer and must keep them alive until this returns;
//: nothing is retained afterwards. `threads` splits the output columns, and the
//: result is the same whatever it is -- the columns are independent.
int32_t relic_cpu_raw_simd_pool_gemv_fp8(const uint8_t *a_codes,
                                         const uint8_t *a_scale,
                                         const uint8_t *b_codes,
                                         const uint8_t *b_scale, float *out,
                                         int64_t n, int64_t k,
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
  Pool *pool = AcquirePool();
  if (pool == nullptr) {
    return kStaleFork;
  }
  // One call at a time per process: the resize below and the session that
  // follows it must not interleave with another caller's session.
  std::lock_guard<std::mutex> call_guard(pool->call_mutex);
  try {
    EnsureTables();
    const int64_t blocks = k / kBlock;

    std::vector<float> a_value(static_cast<size_t>(k));
    std::vector<int32_t> a_nominal(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
      const uint32_t packed = g_packed[a_codes[i]];
      const uint32_t bits = packed & kValueMask;
      float value;
      std::memcpy(&value, &bits, sizeof value);
      if (value != value) {
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
    const int64_t chunk =
        std::max<int64_t>(1, std::min<int64_t>(64, std::max<int64_t>(1, n / workers)));

    std::unique_lock<std::mutex> lock(pool->mutex);
    // The calling thread is one of the participants, exactly as before, so the
    // pool owns workers - 1 threads.
    PoolResize(*pool, workers - 1, lock);
    if (threads_used != nullptr) {
      *threads_used = pool->spawned + 1;
    }
    pool->plan = plan;
    pool->n = n;
    pool->chunk = chunk;
    pool->next.store(0, std::memory_order_relaxed);
    pool->remaining = pool->spawned;
    pool->error = 0;
    ++pool->generation;
    pool->wake.notify_all();
    lock.unlock();

    int32_t failure = 0;
    for (;;) {
      const int64_t begin = pool->next.fetch_add(chunk);
      if (begin >= n) {
        break;
      }
      const int64_t end = std::min(begin + chunk, n);
      try {
        RunRange(plan, begin, end);
      } catch (...) {
        failure = kBadArguments;
        break;
      }
    }

    lock.lock();
    if (failure != 0) {
      pool->error = failure;
    }
    pool->done.wait(lock, [pool] { return pool->remaining == 0; });
    const int32_t result = pool->error != 0 ? pool->error : kOk;
    // The call is synchronous: no pointer it was handed may outlive it.
    pool->plan = Plan{};
    pool->n = 0;
    pool->chunk = 1;
    pool->next.store(0, std::memory_order_relaxed);
    lock.unlock();
    return result;
  } catch (const std::bad_alloc &) {
    return kOutOfMemory;
  } catch (...) {
    return kBadArguments;
  }
}

}  // extern "C"
