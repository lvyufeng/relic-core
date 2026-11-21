"""Triton Ascend kernel for the original block-32 FP8 activation quantizer.

``block_fp8_quant.py`` owns validation, the Torch reference and backend dispatch;
this module owns only the fused device kernel and its scalar bit helpers. The
helpers take integers, so ``tests/test_block_fp8_quant_math.py`` executes their
exact bodies on a CPU without Triton and checks them against an independent
exact-rational oracle.

Every decision the released kernel implements with IEEE bit manipulation -- the
ceil-power-of-two block scale and the E4M3 round-to-nearest-even encode -- is
integer arithmetic here. There is no ``log2``, ``exp2``, ``frexp`` and no
division in this file, so neither the scale nor the code byte can inherit an
accelerator math library's approximation. The only floating-point operations are
the ones the released kernel performs on FP32 data: multiply the FP32 amax by
the rounded ``1/448`` constant, multiply by the exact power-of-two reciprocal of
the chosen scale, and clamp to ``+-448``.

That reciprocal multiply is bit-identical to the released ``x / s``. The block
scale is always an exact power of two whose reciprocal is exactly representable,
so IEEE-754 multiplication by it is the correctly rounded value of the same real
quotient -- including in the subnormal range, where both operations round the
same real number to nearest. A device that flushes FP32 subnormal results cannot
change the emitted byte either: every magnitude below ``2**-10`` rounds to a
signed FP8 zero whether or not it was flushed first, and the subnormal range
starts far below that at ``2**-126``.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _nonfinite_bits(bits):
    """One when an FP32 bit pattern is infinity or NaN, else zero."""
    return tl.where((bits & 0x7F800000) == 0x7F800000, 1, 0)


@triton.jit
def _scale_exponent_from_bits(scaled_bits):
    """``ceil(log2(value))`` from the bits of a positive finite FP32 value.

    ``scaled`` is never zero or subnormal: the released ``1e-4`` floor keeps it
    at or above ``2**-22``, and a finite FP32 input keeps it at most ``2**120``.
    Out-of-range patterns therefore only occur in masked or non-finite blocks,
    whose results are discarded or replaced.
    """
    exponent = (scaled_bits >> 23) & 255
    mantissa = scaled_bits & 0x7FFFFF
    return exponent - 127 + tl.where(mantissa != 0, 1, 0)


@triton.jit
def _pow2_bits(exponent):
    """FP32 bit pattern of ``2**exponent``, for a normal-range integer exponent."""
    return (exponent + 127) << 23


@triton.jit
def _round_half_even_shift(value, shift):
    """``value >> shift`` rounded to nearest with ties to even, value/shift >= 0.

    The bias form ``(value + 2**(shift-1) - 1 + lsb) >> shift`` is what makes the
    tie case fall out without a branch: adding ``half - 1`` can only carry when
    the remainder exceeds the half, or equals it while the retained value is odd.
    """
    quotient = value >> shift
    half = 1 << (shift - 1)
    remainder = value & ((1 << shift) - 1)
    return quotient + tl.where(remainder > half, 1,
                               tl.where(remainder == half, quotient & 1, 0))


@triton.jit
def _e4m3_byte_from_bits(bits):
    """E4M3FN byte of a finite FP32 value already clamped to ``|value| <= 448``.

    E4M3 has a 4-bit exponent field with bias 7, so an FP32 normal exponent field
    maps to ``field - 120`` and 23 - 3 = 20 significand bits are discarded with
    round-to-nearest-even. Below the smallest normal (``2**-6``, field 121) every
    representable magnitude is an integer multiple of ``2**-9``, so the value is
    ``significand >> (141 - field)`` with the same rounding; a rounded 8 is
    exactly the first normal code. Shifts are clamped into ``[1, 25]`` so every
    lane is defined for every input: shift 25 already returns zero for all
    24-bit significands, and the clamp only changes lanes whose branch is not
    selected. ``bits`` of a float32 subnormal or a zero reach the subnormal path
    with shift 25 and correctly produce a signed zero byte.
    """
    sign = (bits >> 24) & 128
    exponent = (bits >> 23) & 255
    significand = (bits & 0x7FFFFF) | 0x800000
    rounded = _round_half_even_shift(significand, 20)
    carry = rounded >> 4
    biased = exponent - 120 + carry
    normal = tl.where(biased > 15, 0x7E, (biased << 3) | (rounded - 8 - (carry << 3)))
    shift = tl.minimum(tl.maximum(141 - exponent, 1), 25)
    subnormal = tl.minimum(_round_half_even_shift(significand, shift), 8)
    return sign | tl.where(exponent <= 120, subnormal, normal)


@triton.jit
def block_fp8_quant_kernel(X, CODES, SCALES, NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
                           BLOCKS: tl.constexpr, FLOOR: tl.constexpr,
                           INV_FP8_MAX: tl.constexpr, INDEX64: tl.constexpr):
    """Encode ``NUM_BLOCKS`` consecutive ``BLOCK``-wide blocks of ``X``.

    The caller guarantees ``K % BLOCK == 0``, so every block is complete and only
    the final program of a non-multiple tile is masked. ``CODES`` receives one
    raw E4M3 byte per element and ``SCALES`` one raw E8M0 byte per block.
    """
    if INDEX64:
        first_block = tl.program_id(0).to(tl.int64) * BLOCKS
    else:
        first_block = tl.program_id(0) * BLOCKS
    block_ids = first_block + tl.arange(0, BLOCKS)
    active = block_ids < NUM_BLOCKS
    lane = tl.arange(0, BLOCK)
    offsets = block_ids[:, None] * BLOCK + lane[None, :]
    values = tl.load(X + offsets, mask=active[:, None], other=0.0).to(tl.float32)
    # Once the sign bit is clear, an FP32 bit pattern orders exactly like its
    # magnitude, so the block maximum is an integer reduction. A floating max
    # would additionally have to answer what a NaN operand means here; the
    # integer form simply reports the largest pattern and the non-finite test
    # below decides the block.
    magnitude_bits = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    nonfinite = tl.max(_nonfinite_bits(magnitude_bits), axis=1) != 0
    amax = tl.max(magnitude_bits, axis=1).to(tl.float32, bitcast=True)
    scaled = tl.maximum(amax, FLOOR) * INV_FP8_MAX
    exponent = _scale_exponent_from_bits(scaled.to(tl.int32, bitcast=True))
    # 1/s for s = 2**exponent: an exact FP32 reciprocal, then the released clamp.
    inverse_bits = _pow2_bits(-exponent)[:, None]
    normalized = tl.minimum(tl.maximum(values * inverse_bits.to(tl.float32, bitcast=True),
                                       -448.0), 448.0)
    codes = _e4m3_byte_from_bits(normalized.to(tl.int32, bitcast=True))
    # A block holding an infinity or NaN has no finite 1/448 scale. The finite
    # contract is extended explicitly: all its lanes become the E4M3 NaN code and
    # its scale the E8M0 NaN byte, never a saturated finite value.
    codes = tl.where(nonfinite[:, None], 0x7F, codes)
    scale_bytes = tl.where(nonfinite, 255, exponent + 127)
    tl.store(CODES + offsets, codes.to(tl.uint8), mask=active[:, None])
    tl.store(SCALES + block_ids, scale_bytes.to(tl.uint8), mask=active)