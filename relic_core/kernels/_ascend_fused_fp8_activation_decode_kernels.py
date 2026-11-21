"""Triton Ascend kernel: original block-32 FP8 activation quantize, then exact BF16 decode.

``fused_fp8_activation_decode.py`` owns validation, the Torch reference and backend
dispatch; this module owns only the fused device kernel. It exists because the
promoted-BF16 path used to run two kernels per projection -- encode the original
E4M3/E8M0 bytes, then read them back and decode to BF16 -- and the second kernel
only ever read what the first had just written.

The two integer decisions that must not drift are imported from the frozen modules
rather than re-typed: the ceil-power-of-two block scale and the E4M3
round-to-nearest-even encode come from ``_ascend_fp8_quant_kernels``, and the
E4M3-times-E8M0-to-BF16 rounding comes from ``_ascend_unpack_kernels``. All five
helpers take integers and vectors only -- no pointers, no loads -- which is the form
this backend lowers reliably; the frozen kernels already call them, so the fused
pass adds a call graph rather than new arithmetic.

Because the decode consumes the very byte pair the encode just produced, the BF16
result is identical to quantize-then-decode by construction: there is one rounding
of the original value and it is the same rounding. No INT8 activation, no new scale
grid and no normal-range assumption are introduced: a block holding an infinity or
NaN still becomes the E4M3 NaN byte with the E8M0 NaN scale, and a zero or
subnormal magnitude still decodes through the same signed-zero and subnormal paths.
"""
from __future__ import annotations

import triton
import triton.language as tl

# The released encode and the frozen decode, imported so this pass cannot drift
# from the two operators it replaces.
from ._ascend_fp8_quant_kernels import (
    _e4m3_byte_from_bits,
    _nonfinite_bits,
    _pow2_bits,
    _scale_exponent_from_bits,
)
from ._ascend_unpack_kernels import _e4m3_e8m0_bf16_bits


@triton.jit
def fused_fp8_activation_decode_kernel(X, OUT, CODES, SCALES, NUM_BLOCKS: tl.constexpr,
                                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                                      FLOOR: tl.constexpr, INV_FP8_MAX: tl.constexpr,
                                      WRITE_CODES: tl.constexpr, INDEX64: tl.constexpr):
    """Encode ``NUM_BLOCKS`` consecutive ``BLOCK``-wide blocks of ``X`` and decode them.

    ``X`` and ``OUT`` are the BF16/FP32 input and the BF16 output over the same
    flattened elements. Each program strides the block tiles it was given, so a
    geometry whose grid would exceed the device cap still covers every block
    exactly once; only lanes past ``NUM_BLOCKS`` are masked.

    When ``WRITE_CODES`` is false the original code and scale bytes stay in
    registers -- the common case, where only the dequantized operand is consumed.
    When it is true they are also stored to ``CODES``/``SCALES``, which are then the
    same bytes ``block_fp8_quant`` would have written. The unused pointer arguments
    are left unread in that case; the caller passes the output buffer for them.
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
        # Clear the sign bit and an FP32 pattern orders exactly like its magnitude,
        # so the block maximum is an integer reduction. A NaN operand therefore
        # contributes its pattern to the max and the non-finite test below decides
        # the block, instead of a float max having to define NaN behaviour.
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
        # The finite contract is extended explicitly: a block holding an infinity or
        # NaN has no finite 1/448 scale, so all its lanes take the E4M3 NaN byte and
        # its scale the E8M0 NaN byte, which is what the frozen quantizer stores and
        # what the frozen decoder reads back as NaN.
        codes = tl.where(nonfinite[:, None], 0x7F, codes)
        scale_bytes = tl.where(nonfinite, 255, exponent + 127)
        # Decode the byte pair just produced. This is the frozen decoder's own body,
        # so no second rounding is introduced by not going through memory.
        bits = _e4m3_e8m0_bf16_bits(codes, scale_bytes[:, None])
        value = tl.cast(bits, tl.bfloat16, bitcast=True)
        tl.store(OUT + offsets, value, mask=active[:, None])
        if WRITE_CODES:
            tl.store(CODES + offsets, codes.to(tl.uint8), mask=active[:, None])
            tl.store(SCALES + block_ids, scale_bytes.to(tl.uint8), mask=active)