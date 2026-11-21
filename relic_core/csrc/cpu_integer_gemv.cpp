// Exact compact host-CPU FP8 GEMV over a 16-bit-per-weight integer prepack.
//
// The accepted `cpu_block_gemv.cpp` (v1) reproduces the reference reduction but reads
// one byte per weight and decodes it through a 256-entry table of binary32 words. The
// prepacked variant (v2) removed the table with one `uint32` per weight -- four
// bytes, two more than the stored code, so the resident form became the problem
// instead of the arithmetic.
//
// This translation unit keeps the reduction and shrinks the representation: one
// `uint16` per weight carries exactly the four fields the reduction reads, so the
// prepack is two bytes per weight and the inner loop is integer shifts rather than
// float multiplies.
//
// Per-lane fields of an e4m3 code (`field` = exponent bits, `mantissa` = low three):
//
//   * significand `SIG` = `field == 0 ? mantissa : 8 + mantissa` (0..15);
//   * sign;
//   * power `P` with value `= SIG * 2**P`: `field == 0 ? -9 : field - 10`, so the
//     seven subnormal encodings share `P = -9`;
//   * nominal exponent `NOM` = `max(true exponent, -6)` = `field >= 1 ? field - 7 : -6`,
//     and a sentinel for a signed-zero code, which contributes nothing to an anchor.
//
// Word layout: `SIG` in bits 0..3, sign in bit 4, `P + 16` in bits 5..9 and
// `NOM + 32` in bits 10..15 -- sixteen bits exactly. The sentinel is stored as 0 and
// decodes arithmetically to `-32` with no branch: a live pair is at least
// `-6 + -6 = -12`, an all-sentinel half is exactly `-64`, and `kEmptyGuard` sits
// between them.
//
// The reduction is integer throughout. A half's anchor is the largest summed nominal,
// and each lane's term is `SIG_A * SIG_B` shifted by `P_A + P_B + 13 - anchor`,
// truncated toward zero on the magnitude and then signed:
//
//   * a lane's shift is at most `13 - 6 = 7`, so no term exceeds `225 << 7` and the
//     16-term sum stays below `2**19` -- an int32 with room to spare;
//   * the running sum's carry onto the second window and the final 14-bit cut are
//     exact integer shifts, because the second window is never below that carry's
//     exponent;
//   * therefore the only rounding left is the binary32 form of the finished fragment
//     (14-bit significand times a power of two, built exactly), the two E8M0 scale
//     multiplies and the K accumulation, which the reference rounds as well.
//
// The two scale multiplies and the K add stay separate, in that order. They are
// applied by bit arithmetic rather than by the FPU, so a host that flushes subnormals
// answers the same as one that does not, while a power-of-two multiply with one
// round-to-nearest-even is exactly what the reference's binary32 multiply does.
// `-ffp-contract=off` and `-fno-fast-math` remain required for the K accumulation.
//
// Non-finite input. An e4m3 code is a NaN exactly when `(byte & 0x7F) == 0x7F`. The
// prepack records that per output column, because a NaN in any of a column's blocks
// makes the whole column NaN; a NaN in the activation row makes every column NaN,
// which the caller-side decode detects before the reduction runs.
//
// Scale bytes. A raw E8M0 byte `b` means `2**(b - 127)`; every `b` in `[0, 254]` is
// an exact power of two and `255` would mean `2**128`, which no binary32 holds, so
// the Python binding refuses byte 255 before it arrives.
//
// The weights remain the checkpoint's stored bytes: nothing here rewrites a
// checkpoint, expands a weight to BF16/FP32, or quantizes an activation.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr int32_t kBlock = 32;
constexpr int32_t kHalf = 16;
constexpr int32_t kWindow = 13;

//: Output columns per prepacked group: four share one contiguous run of words, which
//: is what lets the vector path advance four columns with one load.
constexpr int32_t kGroup = 4;

//: Groups one worker walks per pass over the K blocks, so a block's words for the tile
//: arrive as one contiguous run. The prepacked layout is `[block][group][lane][4]`, so
//: a single group's 256 bytes per block sit `groups * 256` bytes apart -- 1.6 MiB for
//: the released `wkv`, far past any prefetcher -- while a tile of this many groups
//: reads 16 KiB contiguously per block. Every column still accumulates its K blocks in
//: increasing order, so the tiling changes the access pattern and not the arithmetic.
constexpr int64_t kGroupTile = 64;

constexpr uint32_t kSigMask = 0xFu;
constexpr uint32_t kSignShift = 4;
constexpr uint32_t kPowerShift = 5;
constexpr uint32_t kPowerMask = 0x1Fu;
constexpr int32_t kPowerBias = 16;
constexpr uint32_t kNomShift = 10;
constexpr uint32_t kNomMask = 0x3Fu;
constexpr int32_t kNomBias = 32;

//: Below this a half's anchor is the all-sentinel value and the reference's
//: window/step pair is the identity. Every live anchor is at least -12.
constexpr int32_t kEmptyGuard = -32;

//: "This running sum has no exponent." Never wins `max` against a live anchor.
constexpr int32_t kNoExponent = -1000000;

constexpr int32_t kAbiVersion = 1;
constexpr int32_t kOk = 0;
constexpr int32_t kBadArguments = 1;
constexpr int32_t kOutOfMemory = 2;

inline float AsFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

inline uint32_t AsBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

// ---------------------------------------------------------------- compact word

struct Fields {
  int32_t sig;
  int32_t sign;
  int32_t power;
  int32_t nominal;
};

//: Split one e4m3 byte into the four fields the reduction reads.
inline Fields DecodeCode(uint8_t byte) {
  const int32_t field = (byte >> 3) & 15;
  const int32_t mantissa = byte & 7;
  Fields decoded;
  decoded.sig = field == 0 ? mantissa : 8 + mantissa;
  decoded.sign = byte >> 7;
  decoded.power = field == 0 ? -9 : field - 10;
  // A signed zero has no nominal exponent; `kEmptyGuard - kNomBias` stores as 0 and
  // decodes to exactly `kEmptyGuard`.
  decoded.nominal = (byte & 0x7F) == 0 ? kEmptyGuard - kNomBias : (field >= 1 ? field - 7 : -6);
  return decoded;
}

//: The six-bit nominal field, with the signed-zero sentinel stored as the reserved
//: value 0. Every live nominal is at least `-6`, so `nominal + kNomBias` is at least
//: 26 and the reserved value cannot collide with one; writing the sentinel through the
//: ordinary bias would wrap to a value that decodes to a *live* nominal and would
//: silently raise the anchor of any half that contains a zero code.
inline uint32_t EncodeNominal(int32_t nominal) {
  if (nominal < kEmptyGuard) {
    return 0;
  }
  return static_cast<uint32_t>(nominal + kNomBias) & kNomMask;
}

inline uint16_t EncodeFields(const Fields &fields) {
  return static_cast<uint16_t>((static_cast<uint32_t>(fields.sig) & kSigMask) |
                               (static_cast<uint32_t>(fields.sign) << kSignShift) |
                               ((static_cast<uint32_t>(fields.power + kPowerBias) & kPowerMask)
                                << kPowerShift) |
                               (EncodeNominal(fields.nominal) << kNomShift));
}

inline int32_t WordSig(uint32_t word) { return static_cast<int32_t>(word & kSigMask); }
inline int32_t WordSign(uint32_t word) { return static_cast<int32_t>((word >> kSignShift) & 1u); }
inline int32_t WordPower(uint32_t word) {
  return static_cast<int32_t>((word >> kPowerShift) & kPowerMask) - kPowerBias;
}
inline int32_t WordNominal(uint32_t word) {
  return static_cast<int32_t>((word >> kNomShift) & kNomMask) - kNomBias;
}

inline bool IsNanCode(uint8_t byte) { return (byte & 0x7F) == 0x7F; }

// ------------------------------------------------------------- binary32 helpers

//: ``magnitude >> shift`` rounding to nearest even.
inline uint64_t ShiftRightRne(uint64_t magnitude, int32_t shift) {
  if (shift <= 0) {
    return magnitude;
  }
  if (shift >= 32) {
    // Every call site holds fewer than 32 significant bits, so the quotient is zero
    // and the remainder cannot reach the halfway point.
    return 0;
  }
  const uint64_t quotient = magnitude >> shift;
  const uint64_t remainder = magnitude & ((1ull << shift) - 1);
  const uint64_t halfway = 1ull << (shift - 1);
  const bool up = remainder > halfway || (remainder == halfway && (quotient & 1) != 0);
  return up ? quotient + 1 : quotient;
}

//: The correctly rounded (nearest, ties to even) binary32 of ``magnitude * 2**exponent``.
inline float MakeFloat(bool negative, uint64_t magnitude, int64_t exponent) {
  const uint32_t sign = negative ? 0x80000000u : 0u;
  if (magnitude == 0) {
    return AsFloat(sign);
  }
  const int32_t msb = 63 - static_cast<int32_t>(__builtin_clzll(magnitude));
  int64_t lead = static_cast<int64_t>(msb) + exponent;
  if (lead >= -126) {
    const int32_t drop = msb - 23;
    uint64_t significand = drop > 0 ? ShiftRightRne(magnitude, drop) : (magnitude << (-drop));
    if (significand == (1ull << 24)) {
      // The rounding carried into the next binade.
      significand >>= 1;
      ++lead;
    }
    if (lead > 127) {
      return AsFloat(sign | 0x7F800000u);
    }
    return AsFloat(sign | (static_cast<uint32_t>(lead + 127) << 23) |
                   (static_cast<uint32_t>(significand) & 0x7FFFFFu));
  }
  // Subnormal range: the grid is a multiple of 2**-149.
  const int64_t shift = exponent + 149;
  const uint64_t mantissa =
      shift >= 0 ? (magnitude << shift) : ShiftRightRne(magnitude, static_cast<int32_t>(-shift));
  // A carry onto 2**23 is the smallest normal, whose encoding is the same integer.
  return AsFloat(sign | static_cast<uint32_t>(mantissa));
}

//: ``value * 2**delta`` with one round-to-nearest-even, on the bit pattern, so no host
//: FPU mode can change the answer.
inline float MulByPow2(float value, int32_t delta) {
  const uint32_t bits = AsBits(value);
  const uint32_t sign = bits & 0x80000000u;
  const uint32_t biased = (bits >> 23) & 0xFFu;
  const uint32_t mantissa = bits & 0x7FFFFFu;
  if (biased == 0xFFu) {
    return value;  // infinity and NaN are fixed points of a power-of-two multiply
  }
  if (biased == 0) {
    if (mantissa == 0) {
      return value;  // a signed zero keeps its sign and stays zero
    }
    return MakeFloat(sign != 0, mantissa, -149 + static_cast<int64_t>(delta));
  }
  return MakeFloat(sign != 0, mantissa | 0x800000u,
                   static_cast<int64_t>(biased) - 150 + static_cast<int64_t>(delta));
}

inline int32_t FloorLog2(uint32_t magnitude) {
  return 31 - static_cast<int32_t>(__builtin_clz(magnitude));
}

inline int32_t Magnitude(int32_t value) { return value < 0 ? -value : value; }

//: ``value * 2**delta`` truncated toward zero. Deltas are clamped: a non-zero value
//: never reaches the clamp (a live anchor bounds the shift by 7 to the left and 13 to
//: the right), and a zero value is unchanged by every shift.
inline int32_t TruncShift(int32_t value, int32_t delta) {
  if (delta > 31) {
    delta = 31;
  } else if (delta < -31) {
    delta = -31;
  }
  if (delta >= 0) {
    return static_cast<int32_t>(static_cast<uint32_t>(value) << delta);
  }
  const int32_t shifted = Magnitude(value) >> (-delta);
  return value < 0 ? -shifted : shifted;
}

// ------------------------------------------------------------------- geometries

struct Plan {
  int64_t n = 0;
  int64_t k = 0;
  int64_t blocks = 0;
  int64_t groups = 0;
  int32_t block_rows = kBlock;
};

inline int64_t GroupsFor(int64_t n) { return (n + kGroup - 1) / kGroup; }

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

//: The prepack has no weight-scale grid, so only the operand shapes are checked.
int32_t ValidatePack(int64_t n, int64_t k, int32_t threads) {
  if (n <= 0 || k <= 0 || k % kBlock != 0 || threads < 1) {
    return kBadArguments;
  }
  return kOk;
}

//: Weight-scale byte of `column` at `block`, for either grid shape.
inline const uint8_t *ScaleRow(const uint8_t *b_scale, const Plan &plan, int64_t column) {
  const int64_t row = plan.block_rows == 1 ? column : column / plan.block_rows;
  return b_scale + row * plan.blocks;
}

//: The activation row decoded once per call: the four fields and the block scales.
struct Activation {
  std::vector<int32_t> sig;
  std::vector<int32_t> power;
  std::vector<int32_t> nominal;
  std::vector<int32_t> sign;
  std::vector<uint8_t> scale;
  bool has_nan = false;
};

//: One 32-wide block of one group: the two halves, the running-sum window and the
//: final 14-bit cut, all in integers. `fragment` receives the signed 14-bit cut and
//: `exponent` the power of two it is carried at.
inline void GroupFragment(const int32_t *a_sig, const int32_t *a_power, const int32_t *a_nominal,
                          const int32_t *a_sign, const uint16_t *words, int32_t *fragment,
                          int32_t *exponent) {
  int32_t anchor_first[kGroup];
  int32_t anchor_second[kGroup];
  for (int32_t j = 0; j < kGroup; ++j) {
    anchor_first[j] = std::numeric_limits<int32_t>::min();
    anchor_second[j] = std::numeric_limits<int32_t>::min();
  }
  for (int32_t lane = 0; lane < kBlock; ++lane) {
    int32_t *anchor = lane < kHalf ? anchor_first : anchor_second;
    const int32_t a_nom = a_nominal[lane];
    for (int32_t j = 0; j < kGroup; ++j) {
      const int32_t summed = a_nom + WordNominal(words[lane * kGroup + j]);
      if (summed > anchor[j]) {
        anchor[j] = summed;
      }
    }
  }

  int32_t whole_first[kGroup];
  for (int32_t j = 0; j < kGroup; ++j) {
    const int32_t anchor = anchor_first[j];
    int32_t whole = 0;
    for (int32_t lane = 0; lane < kHalf; ++lane) {
      const uint32_t word = words[lane * kGroup + j];
      const int32_t term = WordSig(word) * a_sig[lane];
      const int32_t shifted =
          TruncShift(term, a_power[lane] + WordPower(word) + kWindow - anchor);
      whole += (a_sign[lane] ^ WordSign(word)) != 0 ? -shifted : shifted;
    }
    whole_first[j] = whole;
  }

  for (int32_t j = 0; j < kGroup; ++j) {
    const int32_t anchor = anchor_first[j];
    const int32_t running =
        whole_first[j] == 0 ? kNoExponent : FloorLog2(Magnitude(whole_first[j])) + anchor - kWindow;
    // The second half's window is a peer of the running sum, not a prefix of it.
    const int32_t window = std::max(anchor_second[j], running);
    int32_t whole = 0;
    for (int32_t lane = kHalf; lane < kBlock; ++lane) {
      const uint32_t word = words[lane * kGroup + j];
      const int32_t term = WordSig(word) * a_sig[lane];
      const int32_t shifted =
          TruncShift(term, a_power[lane] + WordPower(word) + kWindow - window);
      whole += (a_sign[lane] ^ WordSign(word)) != 0 ? -shifted : shifted;
    }
    const int32_t carry = TruncShift(whole_first[j], anchor - window);
    const int32_t total = whole + carry;
    if (total == 0) {
      fragment[j] = 0;
      exponent[j] = 0;
      continue;
    }
    const int32_t lead = FloorLog2(Magnitude(total));
    // The final cut keeps 14 bits on the window's grid.
    fragment[j] = TruncShift(total, kWindow - lead);
    exponent[j] = lead + window - kWindow - kWindow;
  }
}

#if defined(__ARM_NEON)

//: Four output columns of one 32-wide block, in int32 lanes.
//:
//: The anchor pass keeps only the nominal, which lives in the top six bits, so it is
//: one shift and one compare-and-max. The product pass uses `vshlq_s32`, a per-lane
//: count, which is what lets four columns with four different anchors advance
//: together; a per-lane count shifts right for negative values, and the product is
//: kept non-negative so an arithmetic shift is the truncation the reference makes.
//: The sign is applied with the usual add-then-exclusive-or conditional negation.
inline void GroupFragmentNeon(const int32_t *a_sig, const int32_t *a_power,
                              const int32_t *a_nominal, const int32_t *a_sign,
                              const uint16_t *words, int32_t *fragment, int32_t *exponent) {
  int32x4_t anchor_first = vdupq_n_s32(std::numeric_limits<int32_t>::min());
  int32x4_t anchor_second = vdupq_n_s32(std::numeric_limits<int32_t>::min());
  for (int32_t lane = 0; lane < kBlock; ++lane) {
    const uint32x4_t word = vmovl_u16(vld1_u16(words + lane * kGroup));
    const int32x4_t nominal = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(word, kNomShift), vdupq_n_u32(kNomMask))),
        vdupq_n_s32(kNomBias));
    const int32x4_t summed = vaddq_s32(nominal, vdupq_n_s32(a_nominal[lane]));
    if (lane < kHalf) {
      anchor_first = vmaxq_s32(anchor_first, summed);
    } else {
      anchor_second = vmaxq_s32(anchor_second, summed);
    }
  }

  int32_t anchors[kGroup];
  int32_t whole0[kGroup];
  vst1q_s32(anchors, anchor_first);
  int32x4_t whole_first = vdupq_n_s32(0);
  for (int32_t lane = 0; lane < kHalf; ++lane) {
    const uint32x4_t word = vmovl_u16(vld1_u16(words + lane * kGroup));
    const int32x4_t sig = vreinterpretq_s32_u32(vandq_u32(word, vdupq_n_u32(kSigMask)));
    const int32x4_t power = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(word, kPowerShift), vdupq_n_u32(kPowerMask))),
        vdupq_n_s32(kPowerBias));
    const int32x4_t delta =
        vsubq_s32(vaddq_s32(vdupq_n_s32(a_power[lane] + kWindow), power), anchor_first);
    const int32x4_t term = vshlq_s32(vmulq_s32(sig, vdupq_n_s32(a_sig[lane])), delta);
    const int32x4_t sign =
        veorq_s32(vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(word, kSignShift), vdupq_n_u32(1))),
                  vdupq_n_s32(a_sign[lane]));
    // The conditional-negation mask has to be all ones, not the sign bit: negating
    // through `(term - 1) ^ -1` is what makes this branchless.
    const int32x4_t mask = vnegq_s32(sign);
    whole_first = vaddq_s32(whole_first, veorq_s32(vaddq_s32(term, mask), mask));
  }
  vst1q_s32(whole0, whole_first);

  int32_t running[kGroup];
  for (int32_t j = 0; j < kGroup; ++j) {
    running[j] = whole0[j] == 0 ? kNoExponent
                                : FloorLog2(Magnitude(whole0[j])) + anchors[j] - kWindow;
  }
  const int32x4_t window = vmaxq_s32(anchor_second, vld1q_s32(running));

  int32x4_t whole_second = vdupq_n_s32(0);
  for (int32_t lane = kHalf; lane < kBlock; ++lane) {
    const uint32x4_t word = vmovl_u16(vld1_u16(words + lane * kGroup));
    const int32x4_t sig = vreinterpretq_s32_u32(vandq_u32(word, vdupq_n_u32(kSigMask)));
    const int32x4_t power = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(word, kPowerShift), vdupq_n_u32(kPowerMask))),
        vdupq_n_s32(kPowerBias));
    const int32x4_t delta =
        vsubq_s32(vaddq_s32(vdupq_n_s32(a_power[lane] + kWindow), power), window);
    const int32x4_t term = vshlq_s32(vmulq_s32(sig, vdupq_n_s32(a_sig[lane])), delta);
    const int32x4_t sign =
        veorq_s32(vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(word, kSignShift), vdupq_n_u32(1))),
                  vdupq_n_s32(a_sign[lane]));
    // The conditional-negation mask has to be all ones, not the sign bit: negating
    // through `(term - 1) ^ -1` is what makes this branchless.
    const int32x4_t mask = vnegq_s32(sign);
    whole_second = vaddq_s32(whole_second, veorq_s32(vaddq_s32(term, mask), mask));
  }

  int32_t whole1[kGroup];
  int32_t windows[kGroup];
  vst1q_s32(whole1, whole_second);
  vst1q_s32(windows, window);
  for (int32_t j = 0; j < kGroup; ++j) {
    const int32_t carry = TruncShift(whole0[j], anchors[j] - windows[j]);
    const int32_t total = whole1[j] + carry;
    if (total == 0) {
      fragment[j] = 0;
      exponent[j] = 0;
      continue;
    }
    const int32_t lead = FloorLog2(Magnitude(total));
    fragment[j] = TruncShift(total, kWindow - lead);
    exponent[j] = lead + windows[j] - kWindow - kWindow;
  }
}

#endif  // __ARM_NEON

// ----------------------------------------------------------------------- driving

struct Job {
  const Activation *activation;
  const uint16_t *words;
  const uint8_t *nan_columns;
  const uint8_t *b_scale;
  Plan plan;
  bool scalar_only;
  float *out;
  int64_t begin;
  int64_t end;
};

//: One tile of groups, every K block accumulated in increasing order per column.
//:
//: `groups` is the tile width in groups (four columns each); a tile narrower than
//: `kGroupTile` is the tail of the output.
void RunTile(const Job &job, int64_t begin, int64_t end) {
  const Activation &activation = *job.activation;
  const int64_t width = end - begin;
  float accumulators[kGroupTile * kGroup];
  for (int64_t slot = 0; slot < width * kGroup; ++slot) {
    accumulators[slot] = 0.0f;
  }
  for (int64_t block = 0; block < job.plan.blocks; ++block) {
    const int64_t offset = static_cast<int64_t>(block) * kBlock;
    const int32_t a_scale_byte = static_cast<int32_t>(activation.scale[block]);
    for (int64_t group = begin; group < end; ++group) {
      const uint16_t *words =
          job.words + (static_cast<int64_t>(block) * job.plan.groups + group) * kBlock * kGroup;
      int32_t fragment[kGroup];
      int32_t exponent[kGroup];
#if defined(__ARM_NEON)
      if (!job.scalar_only) {
        GroupFragmentNeon(activation.sig.data() + offset, activation.power.data() + offset,
                          activation.nominal.data() + offset, activation.sign.data() + offset,
                          words, fragment, exponent);
      } else {
        GroupFragment(activation.sig.data() + offset, activation.power.data() + offset,
                      activation.nominal.data() + offset, activation.sign.data() + offset, words,
                      fragment, exponent);
      }
#else
      GroupFragment(activation.sig.data() + offset, activation.power.data() + offset,
                    activation.nominal.data() + offset, activation.sign.data() + offset, words,
                    fragment, exponent);
#endif
      const int64_t base = group * kGroup;
      const int64_t live = std::min<int64_t>(kGroup, job.plan.n - base);
      float *tile = accumulators + (group - begin) * kGroup;
      for (int64_t j = 0; j < live; ++j) {
        float value = MakeFloat(fragment[j] < 0, static_cast<uint64_t>(Magnitude(fragment[j])),
                                exponent[j]);
        // Activation scale first, weight scale second, then the K add: three
        // separately rounded operations, matching the reference's order.
        value = MulByPow2(value, a_scale_byte - 127);
        value = MulByPow2(value, static_cast<int32_t>(ScaleRow(job.b_scale, job.plan, base + j)[block]) - 127);
        tile[j] += value;
      }
    }
  }
  for (int64_t group = begin; group < end; ++group) {
    const int64_t base = group * kGroup;
    const int64_t live = std::min<int64_t>(kGroup, job.plan.n - base);
    const float *tile = accumulators + (group - begin) * kGroup;
    for (int64_t j = 0; j < live; ++j) {
      job.out[base + j] = job.nan_columns != nullptr && job.nan_columns[base + j] != 0
                              ? std::numeric_limits<float>::quiet_NaN()
                              : tile[j];
    }
  }
}

void RunJob(const Job &job) {
  for (int64_t begin = job.begin; begin < job.end; begin += kGroupTile) {
    RunTile(job, begin, std::min<int64_t>(begin + kGroupTile, job.end));
  }
}

//: Split `[0, total)` across `threads` workers, running the calling thread's share
//: first so a refused thread cannot leave the work undone. Returns how many workers
//: actually ran.
int32_t RunParallel(const Job &template_job, int64_t total, int32_t threads,
                    std::vector<Job> *storage) {
  const int32_t workers = static_cast<int32_t>(std::min<int64_t>(threads, std::max<int64_t>(total, 1)));
  if (workers <= 1) {
    Job job = template_job;
    job.begin = 0;
    job.end = total;
    RunJob(job);
    return 1;
  }
  storage->assign(static_cast<size_t>(workers), template_job);
  const int64_t chunk = (total + workers - 1) / workers;
  for (int32_t worker = 0; worker < workers; ++worker) {
    (*storage)[static_cast<size_t>(worker)].begin = std::min<int64_t>(worker * chunk, total);
    (*storage)[static_cast<size_t>(worker)].end =
        std::min<int64_t>((*storage)[static_cast<size_t>(worker)].begin + chunk, total);
  }
  RunJob((*storage)[0]);
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers - 1));
  size_t spawned = 0;
  try {
    for (int32_t worker = 1; worker < workers; ++worker) {
      pool.emplace_back(RunJob, std::cref((*storage)[static_cast<size_t>(worker)]));
      ++spawned;
    }
  } catch (...) {
    // Fall through: chunks whose threads never started run below.
  }
  for (int32_t worker = static_cast<int32_t>(spawned) + 1; worker < workers; ++worker) {
    RunJob((*storage)[static_cast<size_t>(worker)]);
  }
  for (std::thread &worker : pool) {
    worker.join();
  }
  return static_cast<int32_t>(spawned) + 1;
}

//: The activation row decoded once: four field arrays plus the K-block scale bytes.
bool DecodeActivation(const uint8_t *a_codes, const uint8_t *a_scale, int64_t k,
                      Activation *activation) {
  activation->sig.resize(static_cast<size_t>(k));
  activation->power.resize(static_cast<size_t>(k));
  activation->nominal.resize(static_cast<size_t>(k));
  activation->sign.resize(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    const uint8_t byte = a_codes[i];
    if (IsNanCode(byte)) {
      activation->has_nan = true;
      return false;
    }
    const Fields fields = DecodeCode(byte);
    activation->sig[static_cast<size_t>(i)] = fields.sig;
    activation->power[static_cast<size_t>(i)] = fields.power;
    activation->nominal[static_cast<size_t>(i)] = fields.nominal;
    activation->sign[static_cast<size_t>(i)] = fields.sign;
  }
  const int64_t blocks = k / kBlock;
  activation->scale.assign(a_scale, a_scale + blocks);
  return true;
}

struct PackJob {
  const uint8_t *b_codes;
  uint16_t *words;
  uint8_t *nan_columns;
  int64_t n;
  int64_t k;
  int64_t blocks;
  int64_t groups;
  int64_t begin;
  int64_t end;
};

void RunPack(const PackJob &job) {
  for (int64_t group = job.begin; group < job.end; ++group) {
    const int64_t base = group * kGroup;
    const int64_t live = std::min<int64_t>(kGroup, job.n - base);
    uint8_t nan_flags[kGroup] = {0, 0, 0, 0};
    for (int64_t block = 0; block < job.blocks; ++block) {
      const uint8_t *codes[kGroup];
      for (int64_t j = 0; j < live; ++j) {
        codes[j] = job.b_codes + (base + j) * job.k + block * kBlock;
      }
      uint16_t *words = job.words + (block * job.groups + group) * kBlock * kGroup;
      for (int32_t lane = 0; lane < kBlock; ++lane) {
        for (int64_t j = 0; j < live; ++j) {
          const uint8_t byte = codes[j][lane];
          words[lane * kGroup + j] = EncodeFields(DecodeCode(byte));
          nan_flags[j] = static_cast<uint8_t>(nan_flags[j] | (IsNanCode(byte) ? 1 : 0));
        }
        // Padding columns hold the zero-code word and are never read back.
        for (int64_t j = live; j < kGroup; ++j) {
          words[lane * kGroup + j] = EncodeFields(DecodeCode(0));
        }
      }
    }
    for (int64_t j = 0; j < live; ++j) {
      job.nan_columns[base + j] = nan_flags[j];
    }
  }
}

}  // namespace

extern "C" {

//: ABI version of the compiled kernel. The loader refuses an artifact whose version
//: it does not know, so a stale library can never be called with the wrong signature.
int32_t relic_cpu_integer_gemv_abi_version(void) { return kAbiVersion; }

//: Number of `uint16` words in a prepack for `[n, k]`.
int64_t relic_cpu_integer_gemv_words(int64_t n, int64_t k) {
  if (n <= 0 || k <= 0 || k % kBlock != 0) {
    return -1;
  }
  return GroupsFor(n) * kGroup * k;
}

//: Pack `[n, k]` e4m3 codes into the compact grouped 16-bit form.
//:
//:   b_codes     [n, k]   e4m3 bytes, row-major
//:   words       [blocks, groups, 32, 4]  uint16, `groups = ceil(n / 4)`
//:   nan_columns [n]      uint8, 1 when the column contains any NaN code
//:
//: The layout is transposed on purpose: the four columns of a group share one
//: contiguous run per K lane, which is one vector load for the four-column path. A
//: lane-major `[block, 32, n]` layout would put a column's words `n` apart and was
//: measured to be about three times slower on the vector path.
int32_t relic_cpu_integer_gemv_pack(const uint8_t *b_codes, int64_t n, int64_t k,
                                    uint16_t *words, uint8_t *nan_columns, int32_t threads) {
  if (b_codes == nullptr || words == nullptr || nan_columns == nullptr) {
    return kBadArguments;
  }
  if (ValidatePack(n, k, threads) != kOk) {
    return kBadArguments;
  }
  try {
    PackJob job{};
    job.b_codes = b_codes;
    job.words = words;
    job.nan_columns = nan_columns;
    job.n = n;
    job.k = k;
    job.blocks = k / kBlock;
    job.groups = GroupsFor(n);
    const int32_t workers = static_cast<int32_t>(std::min<int64_t>(threads, job.groups));
    if (workers <= 1) {
      job.begin = 0;
      job.end = job.groups;
      RunPack(job);
      return kOk;
    }
    std::vector<PackJob> jobs(static_cast<size_t>(workers), job);
    const int64_t chunk = (job.groups + workers - 1) / workers;
    for (int32_t worker = 0; worker < workers; ++worker) {
      PackJob &current = jobs[static_cast<size_t>(worker)];
      current.begin = std::min<int64_t>(worker * chunk, job.groups);
      current.end = std::min<int64_t>(current.begin + chunk, job.groups);
    }
    RunPack(jobs[0]);
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    size_t spawned = 0;
    try {
      for (int32_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back(RunPack, std::cref(jobs[static_cast<size_t>(worker)]));
        ++spawned;
      }
    } catch (...) {
    }
    for (int32_t worker = static_cast<int32_t>(spawned) + 1; worker < workers; ++worker) {
      RunPack(jobs[static_cast<size_t>(worker)]);
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

//: Exact M = 1 block-32 FP8 GEMV over a packed weight.
//:
//:   a_codes     [k]                       e4m3 bytes
//:   a_scale     [k / 32]                  raw E8M0 bytes
//:   words       [blocks, groups, 32, 4]   from `relic_cpu_integer_gemv_pack`
//:   nan_columns [n]                       from the same call
//:   b_scale     [n / 32, k / 32]          raw E8M0 bytes, `block_rows == 32`
//:               [n, k / 32]               raw E8M0 bytes, `block_rows == 1`
//:   out         [n]                       binary32 accumulators
//:
//: `mode` selects the vector path (0, the default where available) or the scalar
//: path (1), which exists as the independent oracle the vector path is checked
//: against. `threads` splits the output groups, and the result does not depend on it.
int32_t relic_cpu_integer_gemv_fp8(const uint8_t *a_codes, const uint8_t *a_scale,
                                   const uint16_t *words, const uint8_t *nan_columns,
                                   const uint8_t *b_scale, float *out, int64_t n, int64_t k,
                                   int32_t block_rows, int32_t threads, int32_t mode,
                                   int32_t *threads_used) {
  if (threads_used != nullptr) {
    *threads_used = 0;
  }
  if (a_codes == nullptr || a_scale == nullptr || words == nullptr || nan_columns == nullptr ||
      b_scale == nullptr || out == nullptr) {
    return kBadArguments;
  }
  if (Validate(n, k, block_rows, threads) != kOk || mode < 0 || mode > 1) {
    return kBadArguments;
  }
  try {
    Plan plan;
    plan.n = n;
    plan.k = k;
    plan.blocks = k / kBlock;
    plan.groups = GroupsFor(n);
    plan.block_rows = block_rows;

    Activation activation;
    if (!DecodeActivation(a_codes, a_scale, k, &activation)) {
      // A NaN anywhere in the activation row poisons every column.
      for (int64_t column = 0; column < n; ++column) {
        out[column] = std::numeric_limits<float>::quiet_NaN();
      }
      if (threads_used != nullptr) {
        *threads_used = 1;
      }
      return kOk;
    }

    Job job{};
    job.activation = &activation;
    job.words = words;
    job.nan_columns = nan_columns;
    job.b_scale = b_scale;
    job.plan = plan;
    job.scalar_only = mode == 1;
    job.out = out;
    std::vector<Job> storage;
    const int32_t used = RunParallel(job, plan.groups, threads, &storage);
    if (threads_used != nullptr) {
      *threads_used = used;
    }
    return kOk;
  } catch (const std::bad_alloc &) {
    return kOutOfMemory;
  } catch (...) {
    return kBadArguments;
  }
}

}  // extern "C"