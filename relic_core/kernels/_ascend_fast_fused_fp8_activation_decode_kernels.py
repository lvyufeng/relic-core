"""Triton Ascend kernel: the fused FP8 activation pass with no variable bit shifts.

``fast_fused_fp8_activation_decode.py`` owns validation, the Torch oracle and backend
dispatch; this module owns the device kernel and its two specialized bit helpers. It
replaces, for one pass, the two variable-shift helpers the frozen fused kernel calls,
and it changes **nothing** about the arithmetic those helpers decide: the emitted
E4M3 code bytes, the emitted E8M0 scale bytes and the decoded BF16 bits are the ones
``quantize_block_fp8`` followed by ``unpack_fp8_to_bf16(block_shape=(1, 32))``
produces on the same input.

Why the shifts can go
---------------------

The frozen encode rounds an E4M3 *subnormal* by ``significand >> (141 - exponent)``
with a lane-dependent shift. That quotient is exactly ``|value| * 512``: the
significand of a normal FP32 value is ``value * 2**(150 - exponent)``, so dividing by
``2**(141 - exponent)`` is multiplying by ``2**9``. Below the smallest E4M3 normal the
product is under 8, so its integer part *is* the unrounded code and its fraction
resolves the round-to-nearest-even tie with no shift at all. Multiplication by 512 is
an exact power-of-two scale, and the integer part and the fraction are both exact in
FP32, so the decision is identical bit for bit.

The frozen decode moves the significand by ``7 - msb``, ``>> right`` or ``<< left``,
all lane-dependent, because it handles the whole E4M3 range including values that land
in BF16's subnormal range. The **produced** pairs cannot land there. The released
``1e-4`` FP32 amax floor and the released ``fp32(1/448)`` reciprocal make the smallest
producible E8M0 byte ``105`` (``ceil(log2(fp32(1e-4) * fp32(1/448))) == -22``), and the
smallest nonzero E4M3 magnitude is ``2**-9``, so the smallest producible value is
``2**-31`` -- 95 octaves above the BF16 subnormal threshold ``2**-126``. Every produced
nonzero value is therefore a BF16 **normal**, and its exponent and 7-bit fraction are
constant shifts of the two bytes, given by the ladder below. ``scale >= 105`` is a
proof obligation, not a hope: this module's tests enumerate it, and they also show the
ladder is wrong below 105, which is what makes the restriction load bearing rather than
vacuous.

The `probability < 105` case is unreachable from this encoder, so the specialization is
safe *here* and this module is not a general E4M3/E8M0 decoder; that is why it lives
beside the fused pass rather than replacing the frozen decoder.

What the specialization keeps
-----------------------------

The ceil-power-of-two block scale, the FP32 amax floor and ``1/448``, the non-finite
block rule, the ``+-448`` clamp and the constant-shift normal encode are imported from
the frozen modules and are not retyped. The two new helpers take integers only -- no
pointers, no loads -- which is the form this backend lowers reliably. Nothing is
relaxed: no INT8 activation, no new scale grid, no normal-range assumption about the
*input*, and no new rounding anywhere. A block holding an infinity or NaN still becomes
the E4M3 NaN byte with the E8M0 NaN scale, decoded to BF16 NaN, and a negative zero
still keeps its sign through both the code byte and the decoded BF16 pattern.
"""
from __future__ import annotations

import triton
import triton.language as tl

# The released encoder's constant-shift normal rounding and the three scalar decisions
# it shares with the fused pass are imported, so this pass cannot drift from the
# operator it is a specialization of. None of them contains a pointer or a load.
from ._ascend_fp8_quant_kernels import (
    _nonfinite_bits,
    _pow2_bits,
    _round_half_even_shift,
    _scale_exponent_from_bits,
)

__all__ = [
    "fast_fused_fp8_activation_decode_kernel",
]


@triton.jit
def _e4m3_byte_from_bits_fast(bits):
    """E4M3FN byte of a finite FP32 value already clamped to ``|value| <= 448``.

    Same decision as the frozen ``_e4m3_byte_from_bits``, with the subnormal branch's
    ``significand >> (141 - exponent)`` replaced by ``|value| * 512`` and a truncate
    plus round-to-nearest-even on the fraction; see the module docstring for why the
    two are the same quotient. The normal branch is unchanged and still uses the
    frozen constant-shift rounding at 20 bits.

    Every shift and mask here is by a literal, so no lane value selects a shift
    amount. The float-to-integer cast is fed only lanes that take the subnormal
    branch, which are finite by construction, and a zero everywhere else, so it is
    defined for an infinity or NaN lane as well.
    """
    sign = (bits >> 24) & 128
    exponent = (bits >> 23) & 255
    significand = (bits & 0x7FFFFF) | 0x800000
    rounded = _round_half_even_shift(significand, 20)
    carry = rounded >> 4
    biased = exponent - 120 + carry
    normal = tl.where(biased > 15, 0x7E, (biased << 3) | (rounded - 8 - (carry << 3)))
    # ``|value| * 512`` for the lanes the frozen encoder rounds with a shift. The
    # product is below 8 there, and both the truncation and the subtraction are exact,
    # so ``fraction`` is the exact discarded remainder scaled to [0, 1).
    magnitude = (bits & 0x7FFFFFFF).to(tl.float32, bitcast=True) * 512.0
    lane = (exponent > 0) & (exponent <= 120)
    scaled = tl.where(lane, magnitude, 0.0)
    truncated = scaled.to(tl.int32)
    fraction = scaled - truncated.to(tl.float32)
    increment = tl.where(fraction > 0.5, 1,
                         tl.where(fraction == 0.5, truncated & 1, 0))
    # A rounded 8 is exactly the first E4M3 normal code, which is why the frozen form
    # clamps to 8 here rather than only to 7.
    subnormal = tl.minimum(truncated + increment, 8)
    return sign | tl.where(exponent <= 120, subnormal, normal)


@triton.jit
def _e4m3_e8m0_bf16_bits_fast(code, scale_code):
    """BF16 bits of an E4M3 code under an E8M0 byte, for the produced domain only.

    ``scale_code >= 105`` is required and is guaranteed by the encoder this kernel
    runs: the smallest producible value is ``2**-31``, so every nonzero lane is a BF16
    normal and the two constant ladders below cover it exactly. A code of magnitude
    zero stays a signed zero, ``byte 255`` and code ``0x7F`` stay NaN, and a biased
    exponent of 255 still becomes infinity -- the same three overrides the frozen
    decoder applies, in the same order.
    """
    sign = (code & 128) << 8
    exponent = (code >> 3) & 15
    mantissa = code & 7
    # Normal codes: significand (8 + m) * 2**(e - 10) times the block scale becomes
    # 1.fraction * 2**(e + scale - 7), with fraction m << 4.
    normal_biased = exponent + scale_code - 7
    normal_fraction = mantissa << 4
    # Subnormal codes: significand m * 2**-9 normalizes to a four-entry ladder. The
    # masks recover the displacement without subtracting past zero on a lane that
    # does not select the branch.
    subnormal_biased = tl.where(mantissa >= 4, scale_code - 7,
                               tl.where(mantissa >= 2, scale_code - 8, scale_code - 9))
    subnormal_fraction = tl.where(mantissa >= 4, (mantissa & 3) << 5,
                                  tl.where(mantissa >= 2, (mantissa & 1) << 6, 0))
    biased = tl.where(exponent == 0, subnormal_biased, normal_biased)
    fraction = tl.where(exponent == 0, subnormal_fraction, normal_fraction)
    magnitude = tl.where(biased >= 255, 0x7F80, (biased << 7) | fraction)
    # A zero significand is a signed zero, not the smallest subnormal: the subnormal
    # ladder above reads mantissa 0 as ``scale - 9`` without this override.
    magnitude = tl.where((exponent == 0) & (mantissa == 0), 0, magnitude)
    is_nan = ((code & 127) == 127) | (scale_code == 255)
    magnitude = tl.where(is_nan, 0x7FC0, magnitude)
    return (sign | magnitude).to(tl.uint16)


@triton.jit
def fast_fused_fp8_activation_decode_kernel(X, OUT, CODES, SCALES,
                                           NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
                                           BLOCKS: tl.constexpr, FLOOR: tl.constexpr,
                                           INV_FP8_MAX: tl.constexpr,
                                           WRITE_CODES: tl.constexpr,
                                           INDEX64: tl.constexpr):
    """Encode ``NUM_BLOCKS`` consecutive ``BLOCK``-wide blocks of ``X`` and decode them.

    The block tile, the striding, the masking and the two output modes are the frozen
    fused kernel's; only the two bit helpers differ, so the emitted bytes and decoded
    bits are identical while the lane-dependent shifts are gone.

    When ``WRITE_CODES`` is false the code and scale bytes stay in registers -- the
    common case, where only the dequantized operand is consumed. When it is true they
    are also stored, and are then the bytes ``block_fp8_quant`` would have written.
    """
    total = tl.cdiv(NUM_BLOCKS, BLOCKS)
    for program in range(tl.program_id(0), total, tl.num_programs(0)):
        if INDEX64:
            first_block = program.to(tl.int64) * BLOCKS
        else:
            first_block = program * BLOCKS
        block_ids = first_block + tl.arange(0, BLOCKS)
        active = block_ids < NUM_BLOCKS
        lane = tl.arange(0, BLOCK)
        offsets = block_ids[:, None] * BLOCK + lane[None, :]
        values = tl.load(X + offsets, mask=active[:, None], other=0.0).to(tl.float32)
        # An FP32 pattern orders exactly like its magnitude once the sign bit is
        # clear, so the block maximum is an integer reduction and a NaN operand
        # reports its pattern instead of defining a float max.
        magnitude_bits = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        nonfinite = tl.max(_nonfinite_bits(magnitude_bits), axis=1) != 0
        amax = tl.max(magnitude_bits, axis=1).to(tl.float32, bitcast=True)
        scaled = tl.maximum(amax, FLOOR) * INV_FP8_MAX
        exponent = _scale_exponent_from_bits(scaled.to(tl.int32, bitcast=True))
        # 1/s for s = 2**exponent: an exact FP32 reciprocal, then the released clamp.
        inverse_bits = _pow2_bits(-exponent)[:, None]
        normalized = tl.minimum(tl.maximum(values * inverse_bits.to(tl.float32, bitcast=True),
                                           -448.0), 448.0)
        codes = _e4m3_byte_from_bits_fast(normalized.to(tl.int32, bitcast=True))
        # The finite contract is extended explicitly, exactly as the frozen pass does:
        # a block holding an infinity or NaN has no finite 1/448 scale, so every lane
        # takes the E4M3 NaN byte and the block takes the E8M0 NaN byte.
        codes = tl.where(nonfinite[:, None], 0x7F, codes)
        scale_bytes = tl.where(nonfinite, 255, exponent + 127)
        # Decode the byte pair just produced, so no second rounding is introduced by
        # not going through memory. ``scale_bytes`` is at least 105 for every finite
        # block, which is the precondition of the constant ladder.
        bits = _e4m3_e8m0_bf16_bits_fast(codes, scale_bytes[:, None])
        value = tl.cast(bits, tl.bfloat16, bitcast=True)
        tl.store(OUT + offsets, value, mask=active[:, None])
        if WRITE_CODES:
            tl.store(CODES + offsets, codes.to(tl.uint8), mask=active[:, None])
            tl.store(SCALES + block_ids, scale_bytes.to(tl.uint8), mask=active)