"""Ascend vector kernels for explicit byte-coded block-32 GEMM.

No FP8 cast, FP4 instruction, native integer activation quantization or float64
operation is used. Product slabs fit in a small per-program vector tile.
"""
import triton
import triton.language as tl


@triton.jit
def _ilog2(value):
    high = tl.where(value >= 65536, 16, 0)
    value = value >> high
    shift = tl.where(value >= 256, 8, 0)
    value = value >> shift
    high += shift
    shift = tl.where(value >= 16, 4, 0)
    value = value >> shift
    high += shift
    shift = tl.where(value >= 4, 2, 0)
    value = value >> shift
    high += shift
    return high + tl.where(value >= 2, 1, 0)


@triton.jit
def _pow2(exponent):
    return ((exponent + 127).to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e4m3(code):
    exponent = (code >> 3) & 15
    mantissa = code & 7
    value = tl.where(exponent == 0, mantissa, 8 + mantissa).to(tl.float32)
    value *= _pow2(tl.where(exponent == 0, -9, exponent - 10))
    value = tl.where((code & 128) != 0, -value, value)
    return value


@triton.jit
def _nominal_e4(code):
    nominal = tl.maximum(((code >> 3) & 15) - 7, -6)
    return tl.where((code & 127) != 0, nominal, -1000)


@triton.jit
def _floor_exp(value):
    bits = value.to(tl.int32, bitcast=True) & 0x7fffffff
    exponent = ((bits >> 23) & 255).to(tl.int32) - 127
    return tl.where(bits != 0, exponent, -1000)


@triton.jit
def _grid_exponent(anchor):
    return tl.where(anchor < -500, 0, anchor - 13)


@triton.jit
def _cut(value, anchor):
    exponent = _grid_exponent(anchor)
    return (value * _pow2(-exponent)).to(tl.int32).to(tl.float32) * _pow2(exponent)


@triton.jit
def _original_scale_code(scale, HALF: tl.constexpr):
    bits = scale.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7fffff
    if HALF:
        code = tl.where(bits == 0x00200000, 0,
                        tl.where(bits == 0x00400000, 1, exponent + 1))
        valid = (bits == 0x00200000) | (bits == 0x00400000)
        valid |= ((bits >> 31) == 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 253)
    else:
        code = tl.where(bits == 0x00400000, 0, exponent)
        valid = (bits == 0x00400000)
        valid |= ((bits >> 31) == 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254)
    return tl.where(valid, code, 255).to(tl.int32)


@triton.jit
def _scale_e8m0(value, code):
    # Correctly round *each* power-of-two scale to FP32. Combining two scale
    # exponents would change overflow/underflow and the reference's operation
    # order. Integer construction also avoids arithmetic FTZ of E8M0 byte zero.
    raw = value.to(tl.int32, bitcast=True)
    # Avoid an int64 literal: CANN vector casts cannot promote uint32 to uint64.
    sign = (raw >> 31) << 31
    exponent = ((raw >> 23) & 255).to(tl.int32)
    mantissa = raw & 0x7fffff
    significand = tl.where(exponent == 0, mantissa, mantissa | 0x800000)
    power = tl.where(exponent == 0, -149, exponent - 150) + code - 127
    msb = _ilog2(significand)
    biased = power + msb + 127
    normal = (tl.minimum(tl.maximum(biased, 0), 255).to(tl.int32) << 23)
    normal |= (significand << (23 - msb)) & 0x7fffff
    shift = power + 149
    left = tl.minimum(tl.maximum(shift, 0), 23)
    right = tl.minimum(tl.maximum(-shift, 1), 24)
    truncated = significand >> right
    remainder = significand & ((1 << right) - 1)
    halfway = 1 << (right - 1)
    increment = (remainder > halfway) | ((remainder == halfway) & ((truncated & 1) != 0))
    rounded = truncated + increment.to(tl.int32)
    rounded = tl.where(shift < -24, 0, rounded)
    subnormal = tl.where(shift >= 0, significand << left, rounded)
    magnitude = tl.where(biased <= 0, subnormal, tl.where(biased >= 255, 0x7f800000, normal))
    magnitude = tl.where(significand == 0, 0, magnitude)
    magnitude = tl.where(exponent == 255, raw & 0x7fffffff, magnitude)
    is_nan = ((exponent == 255) & (mantissa != 0)) | (code == 255)
    magnitude = tl.where(is_nan, 0x7fc00000, magnitude)
    return (sign | magnitude).to(tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def block_gemm_kernel(A, SA, B, SB, OUT, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr,
                      B_SCALE_FLOAT: tl.constexpr, BN: tl.constexpr, INDEX64: tl.constexpr):
    row = tl.program_id(1)
    base = tl.program_id(0) * BN
    if INDEX64:
        row = row.to(tl.int64)
        base = base.to(tl.int64)
    columns = base + tl.arange(0, BN)
    lanes = tl.arange(0, 32)
    accumulator = tl.full((BN,), 0.0, tl.float32)
    invalid = tl.full((BN,), False, tl.int1)
    for block in range(K // 32):
        offset = block * 32 + lanes
        ac = tl.load(A + row * K + offset).to(tl.int32)
        a_nan = (ac & 127) == 127
        av = tl.where(a_nan, 0.0, _decode_e4m3(ac))
        an = _nominal_e4(ac)
        bc = tl.load(B + columns[:, None] * K + offset[None, :], columns[:, None] < N, other=0).to(tl.int32)
        if FP4:
            magnitude = tl.abs(bc)
            valid = (magnitude == 0) | (magnitude == 1) | (magnitude == 2) | (magnitude == 3)
            valid |= (magnitude == 4) | (magnitude == 6) | (magnitude == 8) | (magnitude == 12)
            bv = tl.where(valid, bc.to(tl.float32) * 0.5, 0.0)
            bn = tl.where(magnitude >= 8, 2,
                          tl.where(magnitude >= 4, 1, tl.where(magnitude >= 2, 0, -1)))
            bn = tl.where(magnitude != 0, bn, -1000)
            b_nan = ~valid
        else:
            b_nan = (bc & 127) == 127
            bv = tl.where(b_nan, 0.0, _decode_e4m3(bc))
            bn = _nominal_e4(bc)
        product = av[None, :] * bv
        nominal = an[None, :] + bn
        first = tl.max(tl.where(lanes[None, :] < 16, nominal, -2000), 1)
        g0 = _grid_exponent(first)
        integers = (product * _pow2(-g0[:, None])).to(tl.int32)
        total = tl.sum(tl.where(lanes[None, :] < 16, integers, 0), 1).to(tl.float32) * _pow2(g0)
        second = tl.max(tl.where(lanes[None, :] >= 16, nominal, -2000), 1)
        second = tl.maximum(second, _floor_exp(total))
        g1 = _grid_exponent(second)
        integers = (product * _pow2(-g1[:, None])).to(tl.int32)
        whole = tl.sum(tl.where(lanes[None, :] >= 16, integers, 0), 1)
        whole += (total * _pow2(-g1)).to(tl.int32)
        total = whole.to(tl.float32) * _pow2(g1)
        fragment = _cut(total, _floor_exp(total))
        scale_a = tl.load(SA + row * (K // 32) + block).to(tl.int32)
        scale_b = tl.load(SB + (columns // BLOCK_ROWS) * (K // 32) + block, columns < N, other=0)
        if B_SCALE_FLOAT:
            scale_b = _original_scale_code(scale_b, FP4)
        else:
            scale_b = scale_b.to(tl.int32)
        scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
        # No multiply feeds this add: scale rounding is completed by the bit
        # constructors above, and the K blocks remain in their original order.
        accumulator = accumulator + scaled
        invalid |= tl.sum((b_nan | a_nan[None, :]).to(tl.int32), 1) != 0
    accumulator = tl.where(invalid, float("nan"), accumulator)
    tl.store(OUT + row * N + columns, accumulator, columns < N)
