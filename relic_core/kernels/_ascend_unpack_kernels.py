"""Fused Ascend kernels for online floating-code unpacking.

These kernels consume byte aliases, never FP8/FP4 arithmetic types. E2M1 is
expanded exactly to integer values multiplied by two; the original E8M0 block
scales require a factor of one half. E4M3FN/E8M0 decoding produces BF16 bits by
integer exponent arithmetic, equivalent to FP32 decode/scaling followed by one
round-to-nearest-even BF16 conversion, including subnormal output. This avoids
relying on an accelerator's FP32 denormal/flush-to-zero behavior.

Import lazily from ascend_unpack.py only after a Triton Ascend backend is found.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _e2m1_x2(code):
    magnitude = code & 7
    high = (2 + (magnitude & 1)) << tl.maximum((magnitude >> 1) - 1, 0)
    absolute = tl.where(magnitude < 4, magnitude, high)
    return tl.where((code & 8) != 0, -absolute, absolute)


@triton.jit
def fp4_unpack_int8_kernel(INPUT, OUTPUT, NUMEL: tl.constexpr, BLOCK: tl.constexpr,
                          INDEX64: tl.constexpr):
    for program in range(tl.program_id(0), tl.cdiv(NUMEL, BLOCK), tl.num_programs(0)):
        if INDEX64:
            base = program.to(tl.int64) * BLOCK
        else:
            base = program * BLOCK
        # Store a contiguous output vector rather than two masked byte-stride-2
        # vectors. Triton Ascend 3.2's interleave-mask optimizer aborts on the latter.
        output_offset = 2 * base + tl.arange(0, 2 * BLOCK)
        packed = tl.load(INPUT + output_offset // 2, output_offset < 2 * NUMEL, other=0).to(tl.int32)
        nibble_shift = (output_offset & 1) * 4
        code = (packed >> nibble_shift) & 15
        values = _e2m1_x2(code)
        tl.store(OUTPUT + output_offset, values.to(tl.int8), output_offset < 2 * NUMEL)


@triton.jit
def _e4m3_e8m0_bf16_bits(code, scale_code):
    sign = (code & 128) << 8
    exponent = (code >> 3) & 15
    mantissa = code & 7
    significand = tl.where(exponent == 0, mantissa, 8 + mantissa)
    power = tl.where(exponent == 0, -9, exponent - 10) + scale_code - 127
    msb = tl.where(significand >= 8, 3,
                   tl.where(significand >= 4, 2, tl.where(significand >= 2, 1, 0)))
    biased = power + msb + 127
    normal = (tl.minimum(tl.maximum(biased, 0), 255) << 7) | ((significand << (7 - msb)) & 127)

    # BF16 subnormals are integer multiples of 2^-133. All shifts are clamped,
    # even in unselected branches, to avoid undefined oversized/negative shifts.
    shift = power + 133
    left = tl.minimum(tl.maximum(shift, 0), 7)
    right = tl.minimum(tl.maximum(-shift, 1), 4)
    truncated = significand >> right
    remainder = significand & ((1 << right) - 1)
    halfway = 1 << (right - 1)
    increment = (remainder > halfway) | ((remainder == halfway) & ((truncated & 1) != 0))
    rounded = truncated + increment.to(tl.int32)
    subnormal = tl.where(shift >= 0, significand << left, rounded)
    magnitude = tl.where(biased <= 0, subnormal, tl.where(biased >= 255, 0x7F80, normal))
    magnitude = tl.where(significand == 0, 0, magnitude)
    is_nan = ((code & 127) == 127) | (scale_code == 255)
    magnitude = tl.where(is_nan, 0x7FC0, magnitude)
    return (sign | magnitude).to(tl.uint16)


@triton.jit
def fp8_unpack_bf16_kernel(INPUT, SCALE, OUTPUT, NUMEL: tl.constexpr,
                          M: tl.constexpr, K: tl.constexpr,
                          SCALE_M: tl.constexpr, SCALE_K: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                          HAS_SCALE: tl.constexpr, BLOCK: tl.constexpr,
                          INDEX64: tl.constexpr):
    for program in range(tl.program_id(0), tl.cdiv(NUMEL, BLOCK), tl.num_programs(0)):
        if INDEX64:
            base = program.to(tl.int64) * BLOCK
        else:
            base = program * BLOCK
        offset = base + tl.arange(0, BLOCK)
        mask = offset < NUMEL
        code = tl.load(INPUT + offset, mask, other=0).to(tl.int32)
        if HAS_SCALE:
            matrix = offset // (M * K)
            row = (offset // K) % M
            column = offset % K
            scale_offset = (matrix * SCALE_M + row // BLOCK_M) * SCALE_K + column // BLOCK_K
            scale_code = tl.load(SCALE + scale_offset, mask, other=127).to(tl.int32)
        else:
            scale_code = tl.full((BLOCK,), 127, tl.int32)
        bits = _e4m3_e8m0_bf16_bits(code, scale_code)
        # A same-width bitcast is not a floating conversion and preserves BF16 ±0
        # and subnormal bit patterns without running arithmetic on those values.
        value = tl.cast(bits, tl.bfloat16, bitcast=True)
        tl.store(OUTPUT + offset, value, mask)


@triton.jit
def fp4_int8_scale_kernel(INPUT, OUTPUT, NUMEL: tl.constexpr, BLOCK: tl.constexpr,
                          INDEX64: tl.constexpr):
    for program in range(tl.program_id(0), tl.cdiv(NUMEL, BLOCK), tl.num_programs(0)):
        if INDEX64:
            base = program.to(tl.int64) * BLOCK
        else:
            base = program * BLOCK
        offset = base + tl.arange(0, BLOCK)
        mask = offset < NUMEL
        code = tl.load(INPUT + offset, mask, other=127).to(tl.int32)
        bits = tl.where(code == 0, 0x00200000,
                        tl.where(code == 1, 0x00400000, (tl.maximum(code, 1) - 1) << 23))
        bits = tl.where(code == 255, 0x7FC00000, bits)
        value = tl.cast(bits, tl.float32, bitcast=True)
        tl.store(OUTPUT + offset, value, mask)
