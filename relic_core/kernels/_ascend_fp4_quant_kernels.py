"""Triton Ascend kernels for the released block-wise FP4 activation quantizer.

``block_fp4_quant.py`` owns validation, the Torch reference and backend dispatch;
this module owns only the device kernels and their integer bit helpers, so the
helpers can also be re-executed as plain Python by
``tests/test_block_fp4_quant_math.py`` against an exact-rational oracle.

Two properties of the released kernel (``inference/kernel.py:127-205`` of the
checkpoint) are reproduced exactly here:

* the **E8M0** branch selects ``2**ceil(log2(amax * fp32(1/6)))``. The helper it
  calls is named ``fast_round_scale`` but computes a *ceiling* from the exponent
  field; the ceiling, not a nearest rounding, is what the source does.
* the **E4M3** branch selects ``fp32(e4m3(amax / 6.0))`` -- a rounding of the
  *ratio* -- and then normalizes by that rounded scale, not by the exact ratio.
  The ratio is clamped to the format's largest finite (448) before the encode, so
  an ``amax`` above 2688 saturates the scale instead of overflowing it. That is
  what a saturating float-to-FP8 cast does, and it is an **assumption about the
  released cast**, not something this repository has proven: the source writes
  ``T.Cast(FP8, amax / 6.0)`` and never states overflow behaviour, so the claim is
  only that the emitted byte matches a ``__NV_SATFINITE``-style conversion. If the
  released cast were instead to overflow to a NaN, the scale byte for such a block
  would differ; nothing else in this operator depends on the choice.

Neither branch casts a float to a float8 dtype on the device, and neither needs
native FP4 arithmetic: the E4M3 encode is integer bit arithmetic borrowed from
``_ascend_fp8_quant_kernels`` and the E2M1 encode is an integer comparison ladder
against the exact FP32 bit patterns of the codebook midpoints. Nibble packing is
deliberately absent: see ``block_fp4_quant.py`` for why the two-nibble merge is
an explicit device-side pack step rather than a stride-2 store.
"""
from __future__ import annotations

import triton
import triton.language as tl

# The E4M3 encoder, the E8M0 exponent ceiling and the shared float32 helpers are
# the ones already validated for the block-32 FP8 activation quantizer. Importing
# them keeps one implementation of each conversion in the package.
from ._ascend_fp8_quant_kernels import (
    _e4m3_byte_from_bits,
    _nonfinite_bits,
    _pow2_bits,
    _scale_exponent_from_bits,
)

#: FP32 bit patterns of the exact E2M1 magnitude midpoints. A midpoint is where
#: two neighbouring codes are equidistant, and the code chosen there is the one
#: with an even encoding (its low bit is the mantissa bit), which is the
#: round-to-nearest-even rule the released cast relies on. Every midpoint is a
#: dyadic rational, so its bit pattern is exact, and the first representable
#: magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 -- so the midpoints are:
#:
#: Every module-level value a kernel body reads has to be a ``tl.constexpr``: the
#: Triton front end refuses a plain global by name ("cannot access global variable
#: within @jit"), and the escape hatch that would relax that is deliberately not
#: used here. Annotating them keeps the decision at the constant instead of moving
#: seven midpoints into a kernel signature.
_MIDPOINT_0_25 = tl.constexpr(0x3E800000)  # 0.25, tie between code 0 (0.0) and code 1 (0.5)
_MIDPOINT_0_75 = tl.constexpr(0x3F400000)  # 0.75, tie between code 1 and code 2 (1.0)
_MIDPOINT_1_25 = tl.constexpr(0x3FA00000)  # 1.25, tie between code 2 and code 3 (1.5)
_MIDPOINT_1_75 = tl.constexpr(0x3FE00000)  # 1.75, tie between code 3 and code 4 (2.0)
_MIDPOINT_2_50 = tl.constexpr(0x40200000)  # 2.5, tie between code 4 and code 5 (3.0)
_MIDPOINT_3_50 = tl.constexpr(0x40600000)  # 3.5, tie between code 5 and code 6 (4.0)
_MIDPOINT_5_00 = tl.constexpr(0x40A00000)  # 5.0, tie between code 6 and code 7 (6.0)

#: The quiet-NaN FP32 bit pattern, kept as an **integer** on purpose. Triton's JIT
#: compares a cached kernel's globals for change by equality, and ``nan != nan``, so
#: a NaN *float* constexpr global fails the second compilation of the same kernel
#: with "global variable has changed since compiled". An integer pattern compares
#: equal to itself, so the constant is stable across recompiles, and the kernel
#: selects these bits and bitcasts once -- the same way the fused unpack kernels
#: build their BF16 NaN from ``0x7FC0``.
_NAN_BITS = tl.constexpr(0x7FC00000)

#: The largest finite E4M3 magnitude. The scale ratio is clamped to it *before* the
#: encode: the shared E4M3 encoder's contract is an input already within the format,
#: and a ratio in ``[448, 512)`` would otherwise encode to the NaN byte ``0x7F``
#: instead of saturating at ``0x7E``. Saturating is what the released kernel's FP8
#: cast does; see the module docstring for how far that is a claim.
_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _e2m1_code(signed_bits):
    """E2M1 code of an FP32 value already clamped to ``[-6, 6]``, as ``int32``.

    ``signed_bits`` is the value's IEEE bit pattern read as int32, so the sign is
    its arithmetic shift and the magnitude is the pattern with the sign cleared.
    For a non-negative finite float the bit pattern orders exactly like the value,
    which makes every midpoint decision an integer comparison: no FP32 compare can
    sit one ulp away from a tie, and no unselected branch shifts by an undefined
    amount. A tie goes to the even code, so the comparisons against ``0.75``,
    ``1.75`` and ``3.5`` are inclusive while the others are strict.

    E2M1 is a finite format: all sixteen codes are numbers, so a NaN or infinity
    reaching this helper has no NaN encoding and is simply classified by its
    magnitude bits. Non-finite blocks are replaced by the caller, never by this
    helper.
    """
    negative = (signed_bits >> 31) != 0
    magnitude = signed_bits & 0x7FFFFFFF
    code = tl.where(magnitude > _MIDPOINT_0_25, 1, 0)
    code = tl.where(magnitude >= _MIDPOINT_0_75, 2, code)
    code = tl.where(magnitude > _MIDPOINT_1_25, 3, code)
    code = tl.where(magnitude >= _MIDPOINT_1_75, 4, code)
    code = tl.where(magnitude > _MIDPOINT_2_50, 5, code)
    code = tl.where(magnitude >= _MIDPOINT_3_50, 6, code)
    code = tl.where(magnitude > _MIDPOINT_5_00, 7, code)
    return tl.where(negative, code + 8, code)


@triton.jit
def _e2m1_value(code):
    """The FP32 value of an E2M1 code, keeping the sign of the two zero codes.

    Codes 0 and 8 are ``+0.0`` and ``-0.0``. Negating a float zero in IEEE leaves
    the sign bit set, so the dequantized block keeps the distinction the packed
    bytes carry -- which the integer widening in ``unpack_fp4_to_int8`` cannot.
    """
    magnitude = code & 7
    value = tl.where(magnitude >= 1, 0.5, 0.0)
    value = tl.where(magnitude >= 2, 1.0, value)
    value = tl.where(magnitude >= 3, 1.5, value)
    value = tl.where(magnitude >= 4, 2.0, value)
    value = tl.where(magnitude >= 5, 3.0, value)
    value = tl.where(magnitude >= 6, 4.0, value)
    value = tl.where(magnitude >= 7, 6.0, value)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _with_zero_sign_f32(codes, raw_bits):
    """Give a zero-magnitude result the sign its code carries, in FP32 bits.

    A zero's sign cannot be left to the arithmetic. ``-x`` may be lowered to
    ``0 - x``, and ``0 - +0`` is ``+0``; a multiply by a zero operand is likewise
    foldable to a positive zero. The E2M1 code encodes the sign of zero as its high
    bit, so the sign is written into the result's bit pattern here instead of being
    computed.

    A NaN result keeps its NaN: the exponent field is checked, so a non-finite block
    (zero codes times a NaN scale) is never given a zero's sign. ``raw_bits`` of a
    zero is ``0x00000000`` or ``0x80000000``, so OR-ing the sign bit is exact, and
    OR-ing zero leaves the pattern untouched.
    """
    zero = (codes & 7) == 0
    finite = (raw_bits & 0x7F800000) != 0x7F800000
    # A negative int32 literal is used for the sign bit: ``0x80000000`` written as a
    # positive literal does not fit int32 and would promote the expression.
    sign_bit = tl.where((codes & 8) != 0, -2147483648, 0)
    return tl.where(zero & finite, raw_bits | sign_bit, raw_bits)


@triton.jit
def _signed_zero_bf16(codes):
    """The BF16 ``+-0`` whose sign is the code's sign bit.

    A zero-magnitude lane is exactly ``+-0`` in BF16 whatever the narrowing did, so
    the value is built from its 16-bit pattern instead of being recovered from the
    narrowed float. The bitcast runs ``uint16 -> bfloat16``, the direction the frozen
    unpack kernels already compile and run on this backend, and the pattern is
    produced by a small integer shift that cannot overflow.
    """
    return tl.cast(((codes & 8) << 12).to(tl.uint16), tl.bfloat16, bitcast=True)


@triton.jit
def _e4m3_byte_value(byte):
    """FP32 value of an E4M3 byte; the two NaN codes decode to NaN.

    Used only for the E4M3-scale branch, where the scale the reference divides by
    is the *rounded* byte's value rather than the exact ratio.
    """
    sign = (byte >> 7) & 1
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    significand = tl.where(exponent == 0, mantissa, 8 + mantissa)
    power = tl.where(exponent == 0, -9, exponent - 10)
    value = significand.to(tl.float32) * _pow2_bits(power).to(tl.float32, bitcast=True)
    # The two NaN codes are substituted in the *bit pattern* before the bitcast: a
    # float NaN constant would be an unstable JIT global (see _NAN_BITS).
    bits = value.to(tl.int32, bitcast=True)
    bits = tl.where((byte & 127) == 127, _NAN_BITS, bits)
    value = bits.to(tl.float32, bitcast=True)
    return tl.where(sign != 0, -value, value)


@triton.jit
def _block_scale(amax, FLOOR: tl.constexpr, INV_FP4_MAX: tl.constexpr,
                 E4M3_SCALE: tl.constexpr):
    """``(scale byte, FP32 scale)`` for a block amax, per the released branch."""
    amax = tl.maximum(amax, FLOOR)
    if E4M3_SCALE:
        # ``fp32(e4m3(amax / 6.0))``: the ratio is divided by the exact constant
        # 6.0 (the source divides, it does not multiply by a reciprocal) and the
        # E4M3 encode is round-to-nearest-even, saturating at 448.
        ratio = tl.minimum(amax / 6.0, _E4M3_MAX)
        scale_byte = _e4m3_byte_from_bits(ratio.to(tl.int32, bitcast=True))
        return scale_byte, _e4m3_byte_value(scale_byte)
    # ``fast_pow2(fast_log2_ceil(amax * fp32(1/6)))``: an FP32 multiply by the
    # rounded reciprocal, then an exponent-field ceiling -- not a divide, and not
    # a nearest rounding of the logarithm.
    exponent = _scale_exponent_from_bits((amax * INV_FP4_MAX).to(tl.int32, bitcast=True))
    return exponent + 127, _pow2_bits(exponent).to(tl.float32, bitcast=True)


@triton.jit
def _block_quantize(values, FLOOR: tl.constexpr, INV_FP4_MAX: tl.constexpr,
                    E4M3_SCALE: tl.constexpr):
    """Per-block E2M1 codes and scale bytes for a ``[BLOCKS, BLOCK]`` tile."""
    magnitude_bits = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    nonfinite = tl.max(_nonfinite_bits(magnitude_bits), axis=1) != 0
    amax = tl.max(magnitude_bits, axis=1).to(tl.float32, bitcast=True)
    # A non-finite block has no finite scale; computing one from its NaN/Inf
    # magnitude would be an undefined exponent, so the finite floor is used and
    # the block is replaced at the end. Nothing downstream then sees a NaN.
    amax = tl.where(nonfinite, FLOOR, amax)
    scale_byte, scale_value = _block_scale(amax, FLOOR, INV_FP4_MAX, E4M3_SCALE)
    if E4M3_SCALE:
        # The scale is the *rounded* ratio, so the reference divides by that
        # rounded value; a reciprocal multiply would round twice. This is the one
        # division the released kernel performs on this path.
        normalized = values / scale_value[:, None]
    else:
        # The scale is an exact power of two, so multiplying by its exactly
        # representable reciprocal is bit-identical to the released ``x / s`` --
        # in the subnormal range too, where both round the same real quotient to
        # nearest -- while leaving no division node in the kernel. The exponent is
        # recovered from the byte the scale decision just produced.
        inverse = _pow2_bits(127 - scale_byte).to(tl.float32, bitcast=True)
        normalized = values * inverse[:, None]
    normalized = tl.minimum(tl.maximum(normalized, -6.0), 6.0)
    codes = _e2m1_code(normalized.to(tl.int32, bitcast=True))
    # E2M1 has no NaN code. A non-finite block is extended to the *scale's* NaN
    # byte and all-zero codes, so a consumer that multiplies by the scale still
    # sees NaN rather than a plausible finite value.
    codes = tl.where(nonfinite[:, None], 0, codes)
    scale_byte = tl.where(nonfinite, 0x7F if E4M3_SCALE else 255, scale_byte)
    # The inplace path multiplies the codes by this value and never reads the byte,
    # so the value itself has to carry the NaN out: zero codes times a finite scale
    # would write a plausible zero block instead of the documented NaN. Selected in
    # the bit pattern, for the same reason as _NAN_BITS.
    scale_bits = tl.where(nonfinite, _NAN_BITS, scale_value.to(tl.int32, bitcast=True))
    scale_value = scale_bits.to(tl.float32, bitcast=True)
    return codes, scale_byte, scale_value


@triton.jit
def block_fp4_quant_kernel(X, CODES, SCALES, NUM_BLOCKS: tl.constexpr,
                           BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                           FLOOR: tl.constexpr, INV_FP4_MAX: tl.constexpr,
                           E4M3_SCALE: tl.constexpr, INDEX64: tl.constexpr):
    """Write one raw E2M1 code per element and one raw scale byte per block.

    ``CODES`` is ``x``-shaped and ``SCALES`` holds ``ceil(numel / BLOCK)`` bytes;
    both stores are contiguous, and the caller merges code pairs into packed
    nibbles afterwards.
    """
    if INDEX64:
        first_block = tl.program_id(0).to(tl.int64) * BLOCKS
    else:
        first_block = tl.program_id(0) * BLOCKS
    block_ids = first_block + tl.arange(0, BLOCKS)
    active = block_ids < NUM_BLOCKS
    offsets = block_ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    values = tl.load(X + offsets, mask=active[:, None], other=0.0).to(tl.float32)
    codes, scale_byte, _ = _block_quantize(values, FLOOR, INV_FP4_MAX, E4M3_SCALE)
    tl.store(CODES + offsets, codes.to(tl.uint8), mask=active[:, None])
    tl.store(SCALES + block_ids, scale_byte.to(tl.uint8), mask=active)


@triton.jit
def block_fp4_quant_dequant_kernel(X, NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
                                   BLOCKS: tl.constexpr, FLOOR: tl.constexpr,
                                   INV_FP4_MAX: tl.constexpr, E4M3_SCALE: tl.constexpr,
                                   IS_BF16: tl.constexpr, INDEX64: tl.constexpr):
    """Quantize and dequantize each block in place, without materializing codes.

    Each program reads its own ``BLOCK``-wide lanes into registers before storing
    the same lanes back, so the in-place write never races another program. The
    released kernel's ``inplace`` branch computes ``fp32(fp4_value) * scale`` and
    narrows once to the input dtype; that is what this does.
    """
    if INDEX64:
        first_block = tl.program_id(0).to(tl.int64) * BLOCKS
    else:
        first_block = tl.program_id(0) * BLOCKS
    block_ids = first_block + tl.arange(0, BLOCKS)
    active = block_ids < NUM_BLOCKS
    offsets = block_ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    values = tl.load(X + offsets, mask=active[:, None], other=0.0).to(tl.float32)
    codes, _, scale_value = _block_quantize(values, FLOOR, INV_FP4_MAX, E4M3_SCALE)
    dequantized = _e2m1_value(codes) * scale_value[:, None]
    # Restore the sign of every zero in the result's bits: see _with_zero_sign_f32.
    raw = _with_zero_sign_f32(codes, dequantized.to(tl.int32, bitcast=True))
    if IS_BF16:
        # The narrowing is a second place a zero's sign can be dropped, so a
        # zero-magnitude lane is replaced by an explicit ``+-0`` BF16 value selected
        # with the code's sign. A non-finite block also has zero codes, so the
        # substitution additionally requires a finite result: it must keep its NaN.
        narrowed = raw.to(tl.float32, bitcast=True).to(tl.bfloat16)
        zero = ((codes & 7) == 0) & ((raw & 0x7F800000) != 0x7F800000)
        stored = tl.where(zero, _signed_zero_bf16(codes), narrowed)
        tl.store(X + offsets, stored, mask=active[:, None])
    else:
        tl.store(X + offsets, raw.to(tl.float32, bitcast=True), mask=active[:, None])