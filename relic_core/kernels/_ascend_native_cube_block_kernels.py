"""Native Ascend Cube kernels for the authorized different-reduction block GEMV.

Both kernels are M=1 and opt-in. What changes against the accepted operator is the
reduction, not the original encodings, the activation quantization or the weight
precision.

``native_cube_block32_kernel``
    One native BF16xBF16->FP32 Cube ``tl.dot`` sums each K-32 block's 32 products.
    The accepted reference instead cuts a block into two 16-term halves, anchors each
    half on a nominal exponent and truncates onto a window-13 grid. Same unit
    products, same separate FP32 ``SA`` then ``SB`` rescales, same sequential K-block
    addition: only the 32-term reduction is the hardware's.

``native_cube_scaled_bf16_kernel``
    The original E8M0 scale is merged into each unit value as an exact BF16 exponent
    adjustment before one K-major Cube dot, so no post-dot rescale happens. That
    changes the scaling order as well as the reduction, and is only valid where every
    ``code x scale`` pair is exactly a normal BF16 value; the Python layer certifies
    that on the host and refuses the mode otherwise.

Two hard constraints of this backend shape the code and must not be relaxed:

* **No pointer argument is passed to a ``@triton.jit`` helper.** A helper that loads
  through a pointer while its caller is assembling tiles mixes ranks across the
  interprocedural offset analysis (``OffsetAnalysis.cpp``) in this toolchain. Every
  load therefore happens in the kernel body and every helper is arithmetic only.
* **No vector-varying shift.** ``packed >> ((lane & 1) * 4)`` and the codebook
  helper's ``(magnitude >> 1) - 1`` shift are both computed per lane and were the
  known packed-decode regression, so the nibble select is a ``tl.where`` of two
  static shifts and every value decode is a comparison ladder.

Every helper is integer/bit-pattern arithmetic: no float8 cast, no FP4 instruction,
no integer activation quantization, no float64, and no NaN or plain scalar global
(an unstable JIT global). Only ``tl.constexpr`` values reach a kernel body.
"""
import triton
import triton.language as tl

from ._ascend_block_gemm_kernels import _original_scale_code, _scale_e8m0


@triton.jit
def _e4_unit_bf16_bits(code):
    """BF16 bits of an unscaled E4M3 value, sign included.

    E4M3 is 1/4/3 and BF16 is 1/8/7 with the sign in the same place, so a normal
    code maps to BF16 exponent ``exponent + 120`` and mantissa ``mantissa << 4``:
    both significands carry three fractional bits, so nothing is rounded. A subnormal
    code is ``mantissa * 2**-9``, whose normalized exponent and three-bit mantissa
    are exactly the ladder below (mantissa 1..7 -> exponent 118/119/119/120/120/120/
    120). Both ladders are static comparisons: no shift depends on a lane value.
    """
    sign = (code & 128) << 8
    exponent = (code >> 3) & 15
    mantissa = code & 7
    subnormal = exponent == 0
    sub_exponent = tl.where(mantissa >= 4, 120, tl.where(mantissa >= 2, 119, 118))
    sub_mantissa = tl.where(mantissa == 7, 0x60,
                            tl.where((mantissa == 3) | (mantissa == 6), 0x40,
                                     tl.where(mantissa == 5, 0x20, 0)))
    unit_exponent = tl.where(subnormal, sub_exponent, exponent + 120)
    unit_mantissa = tl.where(subnormal, sub_mantissa, mantissa << 4)
    # A code of magnitude zero is a signed zero, not the smallest subnormal: the
    # subnormal ladder above reads mantissa 0 as exponent 118 without this.
    magnitude_bits = tl.where(subnormal & (mantissa == 0), 0, (unit_exponent << 7) | unit_mantissa)
    finite = sign | magnitude_bits
    return tl.where((code & 127) == 127, sign | 0x7FC0, finite)


@triton.jit
def _e2m1_unit_bf16_bits(magnitude):
    """BF16 bits (unsigned) of an E2M1 magnitude index 0..7: 0, .5, 1, 1.5, 2, 3, 4, 6.

    Three fractional bits carry the widest value (``1.5 * 2**2``) exactly, so every
    magnitude is a BF16 value; the ladder sets the exponent field (126..129) and the
    half-bit mantissa. Static shifts only.
    """
    exponent = tl.where(magnitude >= 6, 129,
                        tl.where(magnitude >= 4, 128,
                                 tl.where(magnitude >= 2, 127, 126)))
    mantissa = tl.where((magnitude == 3) | (magnitude == 5) | (magnitude == 7), 0x40, 0)
    return tl.where(magnitude == 0, 0, (exponent << 7) | mantissa)


@triton.jit
def _e2m1_x2_magnitude(codes):
    """Codebook index 0..7 and validity of exact ``INT8 = 2 * E2M1`` values.

    The ladder maps the widened magnitude onto the codebook position in
    ``0, 1, 2, 3, 4, 6, 8, 12``, so a stored value outside that codebook is reported
    invalid rather than turned into a plausible weight.
    """
    magnitude = tl.abs(codes)
    index = tl.where(magnitude >= 12, 7,
                     tl.where(magnitude >= 8, 6,
                              tl.where(magnitude >= 6, 5,
                                       tl.where(magnitude >= 4, 4,
                                                tl.where(magnitude >= 3, 3,
                                                         tl.where(magnitude >= 2, 2,
                                                                  tl.where(magnitude >= 1, 1, 0)))))))
    valid = ((magnitude == 0) | (magnitude == 1) | (magnitude == 2) | (magnitude == 3)
             | (magnitude == 4) | (magnitude == 6) | (magnitude == 8) | (magnitude == 12))
    return index, valid


@triton.jit
def _scaled_bf16_bits(unit_bits, scale_code, zero):
    """Merge an E8M0 exponent into BF16 bits as an exact exponent-field add.

    ``scale_code`` is the original raw E8M0 byte, so the scale is ``2**(code - 127)``
    and the product is ``unit * 2**(code - 127)``: adding ``code - 127`` to the unit
    exponent is exact exactly when the result stays normal. The shift is clamped
    before it is applied, so no lane shifts by a negative amount. A lane outside the
    certified contract becomes NaN rather than a silently rounded value, and a NaN
    scale byte still wins over a zero lane.

    The arithmetic stays int32 and only the returned bit pattern is 16-bit, which is
    the idiom the hardware-validated unpack kernel uses for the same job: it keeps
    every operand of the buffer-limited Cube kernel 16-bit without introducing
    dtype-annotated constants or 16-bit shift amounts whose promotion rules differ.
    """
    sign = unit_bits & 0x8000
    exponent = (unit_bits >> 7) & 255
    mantissa = unit_bits & 127
    nan_scale = scale_code == 255
    shifted = exponent + scale_code - 127
    safe = tl.minimum(tl.maximum(shifted, 0), 255)
    scaled = sign | (safe << 7) | mantissa
    # A signed zero times a finite power of two keeps its sign (2**0 included).
    scaled = tl.where(zero & ~nan_scale, sign, scaled)
    valid = (~nan_scale) & (zero | ((shifted > 0) & (shifted < 255)))
    return tl.where(valid, scaled, sign | 0x7FC0).to(tl.uint16)


@triton.jit
def _bfloat16_tile(bits):
    """A BF16 value from a 16-bit pattern: the direction this backend compiles."""
    return tl.cast(bits.to(tl.uint16), tl.bfloat16, bitcast=True)


@triton.jit
def native_cube_block32_kernel(A, SA, B, SB, OUT, N: tl.constexpr, K: tl.constexpr,
                               BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr,
                               B_PACKED: tl.constexpr, B_SCALE_FLOAT: tl.constexpr,
                               BN: tl.constexpr, INDEX64: tl.constexpr):
    """One native Cube dot per K-32 block, then the frozen separate FP32 rescales.

    The A tile is ``[16, 32]`` with the activation row in row 0 and zeros below it,
    the smallest M a Cube dot accepts for one row. The row is recovered with a masked
    sum over the 15 zero rows, which adds ``+0`` to it: the same value for every
    result except an exactly-zero column, where ``+0`` may replace ``-0``.
    """
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    tiles = tl.cdiv(N, BN)
    lanes = tl.arange(0, 32)
    rows = tl.arange(0, 16)
    for tile in range(program, tiles, tl.num_programs(0)):
        columns = tile * BN + tl.arange(0, BN)
        active = columns < N
        accumulator = tl.zeros((BN,), tl.float32)
        invalid = tl.zeros((BN,), tl.int1)
        for block in range(K // 32):
            offset = block * 32 + lanes
            codes_a = tl.load(A + offset).to(tl.int32)
            row_invalid = tl.sum(((codes_a & 127) == 127).to(tl.int32)) != 0
            unit_a = tl.where((codes_a & 127) == 127, 0, _e4_unit_bf16_bits(codes_a))
            tile_a = _bfloat16_tile(tl.where(rows[:, None] == 0, unit_a[None, :], 0))
            if B_PACKED:
                # Low nibble is the even K index, the released weights' ordering.
                packed = tl.load(B + columns[None, :] * (K // 2) + (block * 16 + lanes // 2)[:, None],
                                 mask=active[None, :], other=0).to(tl.int32)
                code = tl.where((lanes & 1)[:, None] == 0, packed & 15, packed >> 4)
                magnitude = code & 7
                unit_b = _e2m1_unit_bf16_bits(magnitude) | ((code & 8) << 12)
                scale_b = tl.load(SB + columns * (K // 32) + block, mask=active, other=0)
                scale_b = scale_b.to(tl.int32)
                column_invalid = tl.zeros((BN,), tl.int1)
            else:
                codes_b = tl.load(B + columns[None, :] * K + offset[:, None],
                                  mask=active[None, :], other=0).to(tl.int32)
                if FP4:
                    magnitude, valid_b = _e2m1_x2_magnitude(codes_b)
                    negative = codes_b < 0
                    unit_b = tl.where(valid_b,
                                      _e2m1_unit_bf16_bits(magnitude) | ((negative.to(tl.int32)) << 15),
                                      0)
                    column_invalid = tl.sum((~valid_b).to(tl.int32), axis=0) != 0
                else:
                    nan_b = (codes_b & 127) == 127
                    unit_b = tl.where(nan_b, 0, _e4_unit_bf16_bits(codes_b))
                    column_invalid = tl.sum(nan_b.to(tl.int32), axis=0) != 0
                scale = tl.load(SB + (columns // BLOCK_ROWS) * (K // 32) + block,
                                mask=active, other=0)
                if B_SCALE_FLOAT:
                    scale_b = _original_scale_code(scale, FP4).to(tl.int32)
                else:
                    scale_b = scale.to(tl.int32)
            partial = tl.dot(tile_a, _bfloat16_tile(unit_b), out_dtype=tl.float32)
            fragment = tl.sum(tl.where(rows[:, None] == 0, partial, 0.0), axis=0)
            scale_a = tl.load(SA + block).to(tl.int32)
            # The accepted separate rescale order: FP32 * SA, then * SB, no fusion.
            scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
            accumulator = accumulator + scaled
            invalid = invalid | column_invalid | row_invalid
        tl.store(OUT + columns, tl.where(invalid, float("nan"), accumulator), mask=active)


@triton.jit
def native_cube_scaled_bf16_kernel(A, SA, B, SB, OUT, N: tl.constexpr, K: tl.constexpr,
                                   BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr,
                                   B_PACKED: tl.constexpr, B_SCALE_FLOAT: tl.constexpr,
                                   BK: tl.constexpr, BN: tl.constexpr, INDEX64: tl.constexpr):
    """One native Cube dot per ``BK`` columns of K, with the scales merged first.

    ``BK`` is a multiple of 32, so a lane's scale is the original per-K-32 grid value
    at ``k // 32``. Every operand is ``unit * 2**(code - 127)``, exact in BF16 by the
    host certification, so the dot is a native BF16 matmul of the exact dequantized
    operands and only the Cube's FP32 accumulation over ``BK`` terms is left.
    """
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    tiles = tl.cdiv(N, BN)
    rows = tl.arange(0, 16)
    k_lane = tl.arange(0, BK)
    for tile in range(program, tiles, tl.num_programs(0)):
        columns = tile * BN + tl.arange(0, BN)
        active = columns < N
        accumulator = tl.zeros((BN,), tl.float32)
        for step in range(K // BK):
            index = step * BK + k_lane
            block = index // 32
            codes_a = tl.load(A + index).to(tl.int32)
            scale_a = tl.load(SA + block).to(tl.int32)
            # A NaN E4M3 code has no finite value; forcing the NaN scale byte keeps
            # it NaN here instead of letting the exponent add land on a finite one.
            scale_a = tl.where((codes_a & 127) == 127, 255, scale_a)
            zero_a = (codes_a & 127) == 0
            bits_a = _scaled_bf16_bits(_e4_unit_bf16_bits(codes_a).to(tl.uint16),
                                       scale_a.to(tl.int16), zero_a)
            tile_a = _bfloat16_tile(tl.where(rows[:, None] == 0, bits_a[None, :], 0))
            if B_PACKED:
                packed = tl.load(B + columns[None, :] * (K // 2) + (index // 2)[:, None],
                                 mask=active[None, :], other=0).to(tl.int32)
                code = tl.where((index & 1)[:, None] == 0, packed & 15, packed >> 4)
                magnitude = code & 7
                unit_b = (_e2m1_unit_bf16_bits(magnitude) | ((code & 8) << 12)).to(tl.uint16)
                zero_b = magnitude == 0
                block_scale = tl.load(SB + columns[None, :] * (K // 32) + block[:, None],
                                      mask=active[None, :], other=0).to(tl.int32)
            else:
                codes_b = tl.load(B + columns[None, :] * K + index[:, None],
                                  mask=active[None, :], other=0).to(tl.int32)
                if FP4:
                    magnitude, valid_b = _e2m1_x2_magnitude(codes_b)
                    negative = (codes_b < 0).to(tl.int32)
                    unit_b = tl.where(valid_b,
                                      _e2m1_unit_bf16_bits(magnitude) | (negative << 15),
                                      0).to(tl.uint16)
                    zero_b = valid_b & (magnitude == 0)
                else:
                    unit_b = _e4_unit_bf16_bits(codes_b).to(tl.uint16)
                    zero_b = (codes_b & 127) == 0
                    valid_b = tl.full((BK, BN), True, tl.int1)
                scale = tl.load(SB + (columns // BLOCK_ROWS)[None, :] * (K // 32) + block[:, None],
                                mask=active[None, :], other=0)
                if B_SCALE_FLOAT:
                    block_scale = _original_scale_code(scale, FP4).to(tl.int32)
                else:
                    block_scale = scale.to(tl.int32)
                if not FP4:
                    block_scale = tl.where((codes_b & 127) == 127, 255, block_scale)
                # An out-of-codebook INT8 value is not an original E2M1 code: the
                # certification refuses it, and this keeps a bypassed check from
                # producing a plausible number instead of NaN.
                block_scale = tl.where(valid_b, block_scale, 255)
            bits_b = _scaled_bf16_bits(unit_b, block_scale.to(tl.int16), zero_b)
            partial = tl.dot(tile_a, _bfloat16_tile(bits_b), out_dtype=tl.float32)
            accumulator = accumulator + tl.sum(
                tl.where(rows[:, None] == 0, partial, 0.0), axis=0)
        tl.store(OUT + columns, accumulator, mask=active)