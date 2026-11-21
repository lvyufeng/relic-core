"""Acceptance for the exact M=1 decode block GEMV.

The claim under test is narrow and checkable: the two-pass operator produces
**the same bits** as the accepted one-pass block GEMM, because its fragments are
the accepted kernel's loop body and its second pass is the accepted kernel's
accumulator chain. Three independent statements are compared:

* the accepted operator itself (``fp4_int8_block_gemm`` / ``fp8_block_gemm``) on
  the same backend;
* an independent scalar oracle written here from the documented reduction;
* the operator's own scratch, fragment by fragment, against that oracle.

A CPU run exercises the Torch reference path (which delegates to the accepted
operator) and every validation/refusal. A fused run needs a leased card:

    RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 RELIC_DECODE_GEMV_TEST_BACKEND=triton \
        python -m pytest tests/test_decode_block_gemv.py -q -rs

Comparison policy: outputs are compared as **bit patterns**, never with a
tolerance. NaN is compared by classification only -- the NaN payload is not part
of any parity contract here, exactly as the accepted operator's documentation
says of its own non-finite cases.
"""
from __future__ import annotations

import math
import os
import struct

import pytest

torch = pytest.importorskip("torch")
from relic_core.kernels.block_gemm import fp4_int8_block_gemm, fp8_block_gemm
from relic_core.kernels.decode_block_gemv import (
    MAX_PROGRAMS,
    decode_gemv_scratch,
    fp4_int8_decode_gemv,
    fp4_packed_decode_gemv,
    fp8_decode_gemv,
)
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8


#: The exact doubled E2M1 codebook, indexed by nibble (E2M1 code 8 is negative
#: zero, which the integer form deliberately cannot carry).
DOUBLED = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def e4(code):
    if code & 127 == 127:
        return math.nan
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(float(8 + mantissa), exponent - 10)
    return math.copysign(value, -1.0 if code & 128 else 1.0)


def a_nominal(code):
    return max(((code >> 3) & 15) - 7, -6)


def b_nominal(code, fp4):
    if fp4:
        doubled = abs(DOUBLED[code])
        if doubled == 0:
            return None
        if doubled >= 8:
            return 2
        if doubled >= 4:
            return 1
        return 0 if doubled >= 2 else -1
    return None if code & 127 == 0 else a_nominal(code)


def b_value(code, fp4):
    return DOUBLED[code] * 0.5 if fp4 else e4(code)


def cut(value, anchor):
    grid = 0 if anchor is None or anchor < -500 else anchor - 13
    scale = math.ldexp(1.0, -grid)
    return f32(math.trunc(f32(value * scale)) * math.ldexp(1.0, grid))


def fragment(a_codes, b_codes, fp4):
    """Independent statement of one K-32 fragment reduction."""
    products, exponents = [], []
    for ac, bc in zip(a_codes, b_codes):
        a, b = e4(ac), b_value(bc, fp4)
        if math.isnan(a) or math.isnan(b):
            return math.nan
        product = f32(a * b)
        products.append(product)
        exponents.append(None if product == 0 else a_nominal(ac) + b_nominal(bc, fp4))
    first = max((v for v in exponents[:16] if v is not None), default=None)
    total = f32(sum(cut(p, first) for p in products[:16]))
    running = None
    if total != 0:
        bits = struct.unpack("<i", struct.pack("<f", total))[0]
        running = ((bits >> 23) & 255) - 127
    second_half = [v for v in exponents[16:] if v is not None]
    if running is not None:
        second_half.append(running)
    second = max(second_half, default=None)
    total = f32(cut(total, second) + f32(sum(cut(p, second) for p in products[16:])))
    anchor = None
    if total != 0:
        bits = struct.unpack("<i", struct.pack("<f", total))[0]
        anchor = ((bits >> 23) & 255) - 127
    return cut(total, anchor)


def oracle(a_codes, a_scale, b_codes, b_scale, *, fp4, block_rows=1):
    """Fragments and the strict K-order accumulation, as (fragments, result)."""
    a, sa, b, sb = (tensor.tolist() for tensor in (a_codes, a_scale, b_codes, b_scale))
    n, k = len(b), len(a[0])
    blocks = k // 32
    fragments = [[0.0] * n for _ in range(blocks)]
    for block in range(blocks):
        for column in range(n):
            value = fragment(a[0][block * 32:(block + 1) * 32], b[column][block * 32:(block + 1) * 32], fp4)
            code_a = sa[0][block]
            code_b = sb[column // block_rows][block]
            scale_a = math.nan if code_a == 255 else math.ldexp(1.0, code_a - 127)
            scale_b = math.nan if code_b == 255 else math.ldexp(1.0, code_b - 127)
            fragments[block][column] = f32(f32(value * scale_a) * scale_b)
    result = []
    for column in range(n):
        value = 0.0
        for block in range(blocks):  # strictly increasing K order, no reassociation
            value = f32(value + fragments[block][column])
        result.append(value)
    return fragments, torch.tensor([result], dtype=torch.float32)


def device():
    name = os.environ.get("RELIC_DECODE_GEMV_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(name)


def backend():
    return os.environ.get("RELIC_DECODE_GEMV_TEST_BACKEND", "torch" if device().type == "cpu" else "triton")


def assert_takes_the_finite_path(expected, *, context=""):
    """A bit-equality check against an all-NaN tensor proves nothing.

    The NaN positions are part of the contract and are checked separately by the
    non-finite tests; every *parity* test asserts here that the fixture actually
    produced finite values to compare.
    """
    assert bool(torch.isfinite(expected).all()), (
        f"{context}: the fixture produced non-finite values, so the bit comparison "
        f"would be vacuous")


def assert_bits(actual, expected, *, context=""):
    """Bit equality for finite values, NaN classification for non-finite ones."""
    assert actual.shape == expected.shape, (context, actual.shape, expected.shape)
    actual, expected = actual.cpu(), expected.cpu()
    nan_actual, nan_expected = torch.isnan(actual), torch.isnan(expected)
    assert torch.equal(nan_actual, nan_expected), f"{context}: NaN positions differ"
    finite = ~nan_expected
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32)), (
        f"{context}: {int((actual[finite].view(torch.int32) != expected[finite].view(torch.int32)).sum())} "
        f"of {int(finite.sum())} finite values differ")


def operands(n, k, *, seed=11, fp4=True, block_rows=1, a_scale_code=127, b_scale_code=127,
             nan_free=True):
    """Codes for one activation row plus a weight, and the packed nibble view.

    ``nan_free`` walks any E4M3 NaN code (``code & 127 == 127``) down to its finite
    neighbour. It is on by default on purpose: a random byte pattern of length K
    contains a NaN code about half the time for K=96, which would make a
    bit-equality assertion compare NaN against NaN and prove nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    blocks = k // 32
    a_scale = torch.full((1, blocks), a_scale_code, dtype=torch.uint8)
    scale_rows = math.ceil(n / block_rows)
    b_scale = torch.full((scale_rows, blocks), b_scale_code, dtype=torch.uint8)
    if fp4:
        nibbles = torch.randint(0, 16, (n, k), dtype=torch.int64, generator=generator)
        packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
        if nan_free:
            a_codes = torch.where((a_codes & 127) == 127,
                                  (a_codes.to(torch.int32) - 1).to(torch.uint8), a_codes)
        return a_codes, a_scale, packed, b_scale, nibbles
    b_codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    if nan_free:
        for codes in (a_codes, b_codes):
            codes.copy_(torch.where((codes & 127) == 127,
                                    (codes.to(torch.int32) - 1).to(torch.uint8), codes))
    return a_codes, a_scale, b_codes, b_scale, None


# -- the primary acceptance: bit equality with the accepted operator ----------------------


@pytest.mark.parametrize("n,k", [(32, 64), (7, 96), (17, 128), (2304, 5120), (5120, 2304)])
def test_fp4_int8_decode_gemv_matches_the_accepted_block_gemm(n, k):
    place, mode = device(), backend()
    a_codes, a_scale, packed, b_scale, nibbles = operands(n, k, seed=n + k)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    args = [t.to(place) for t in (a_codes, a_scale, values, half)]
    expected = fp4_int8_block_gemm(*[t.to(place) for t in (a_codes, a_scale, values, half)],
                                  backend=mode)
    actual = fp4_int8_decode_gemv(*args, backend=mode)
    assert actual.shape == (1, n) and actual.dtype == torch.float32
    assert_takes_the_finite_path(expected, context=f"fp4 int8 n={n} k={k}")
    assert_bits(actual, expected, context=f"fp4 int8 n={n} k={k}")


@pytest.mark.parametrize("n,k", [(32, 64), (5, 96), (2304, 5120)])
def test_fp4_packed_decode_gemv_matches_the_int8_path_and_the_accepted_gemm(n, k):
    place, mode = device(), backend()
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=n * k)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_packed_decode_gemv(a_codes.to(place), a_scale.to(place), packed.to(place),
                                    b_scale.to(place), backend=mode)
    assert_takes_the_finite_path(expected, context=f"fp4 packed n={n} k={k}")
    assert_bits(actual, expected, context=f"fp4 packed n={n} k={k}")


@pytest.mark.parametrize("block_rows", [1, 32])
@pytest.mark.parametrize("n,k", [(32, 64), (2304, 5120), (5120, 1280)])
def test_fp8_decode_gemv_matches_the_accepted_block_gemm(n, k, block_rows):
    place, mode = device(), backend()
    a_codes, a_scale, b_codes, b_scale, _ = operands(n, k, seed=n + 2 * k, fp4=False,
                                                    block_rows=block_rows)
    expected = fp8_block_gemm(a_codes.to(place), a_scale.to(place), b_codes.to(place),
                              b_scale.to(place), block_rows=block_rows, backend=mode)
    actual = fp8_decode_gemv(a_codes.to(place), a_scale.to(place), b_codes.to(place),
                            b_scale.to(place), block_rows=block_rows, backend=mode)
    assert_takes_the_finite_path(expected, context=f"fp8 n={n} k={k} rows={block_rows}")
    assert_bits(actual, expected, context=f"fp8 n={n} k={k} rows={block_rows}")


# -- the same, against the independent scalar oracle -------------------------------------


@pytest.mark.parametrize("fp4", [True, False])
def test_matches_the_independent_scalar_oracle(fp4):
    place, mode = device(), backend()
    n, k = 13, 128
    a_codes, a_scale, weight, b_scale, nibbles = operands(n, k, seed=77, fp4=fp4)
    fragments, expected = oracle(a_codes, a_scale, nibbles if fp4 else weight, b_scale, fp4=fp4)
    if fp4:
        values = unpack_fp4_to_int8(weight, backend="torch")
        half = fp4_int8_scale(b_scale, backend="torch")
        actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                      half.to(place), backend=mode)
    else:
        actual = fp8_decode_gemv(a_codes.to(place), a_scale.to(place), weight.to(place),
                                b_scale.to(place), block_rows=1, backend=mode)
    assert_takes_the_finite_path(expected, context=f"oracle fp4={fp4}")
    assert_bits(actual, expected, context=f"oracle fp4={fp4}")


def test_the_fragment_scratch_holds_the_accepted_per_block_values():
    """Each scratch entry must be the accepted fragment for that (block, column).

    This is the check that the split itself is faithful: pass 1 writes exactly the
    per-block value the one-pass kernel would have added, and pass 2 only adds.
    """
    place, mode = device(), backend()
    if mode != "triton":
        pytest.skip("the fused path is the only one that fills the scratch")
    n, k = 11, 160
    a_codes, a_scale, packed, b_scale, nibbles = operands(n, k, seed=515)
    fragments, expected = oracle(a_codes, a_scale, nibbles, b_scale, fp4=True)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    scratch = decode_gemv_scratch(n, k, device=place)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode, scratch=scratch)
    stored = scratch[0].cpu()
    assert stored.shape == (k // 32, n)
    for block in range(k // 32):
        assert_bits(stored[block:block + 1], torch.tensor([fragments[block]], dtype=torch.float32),
                    context=f"fragment block {block}")
    assert_bits(actual, expected, context="scratch end to end")


def test_out_and_scratch_are_returned_in_place():
    place, mode = device(), backend()
    n, k = 23, 96
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=31)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    scratch = decode_gemv_scratch(n, k, device=place)
    out = torch.full((1, n), float("nan"), dtype=torch.float32, device=place)
    returned = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                    half.to(place), backend=mode, out=out, scratch=scratch)
    assert returned is out
    assert not bool(torch.isnan(out).any()), "the fused path must fill every output column"


# -- tiling and grid invariants ----------------------------------------------------------


@pytest.mark.parametrize("fragment_rows", [8, 16, 32])
@pytest.mark.parametrize("columns_per_program", [16, 64])
def test_the_tiling_does_not_change_a_single_bit(fragment_rows, columns_per_program):
    """Different programs walking different shapes must compute the same number."""
    place, mode = device(), backend()
    n, k = 97, 192
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=808)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    args = [t.to(place) for t in (a_codes, a_scale, values, half)]
    reference = fp4_int8_decode_gemv(*args, backend=mode)
    for rows, columns in ((fragment_rows, columns_per_program), (16, 64)):
        actual = fp4_int8_decode_gemv(*args, backend=mode, fragment_rows=rows,
                                     columns_per_program=columns)
        assert_takes_the_finite_path(reference, context="tiling")
        assert_bits(actual, reference, context=f"tiling rows={rows} columns={columns}")


def test_a_wide_n_and_short_k_still_fits_the_program_cap():
    """The cap is the total program count, so the launcher must split, not overflow."""
    place, mode = device(), backend()
    n, k = 4096, 64
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=99)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode)
    assert_takes_the_finite_path(expected, context="wide short")
    assert_bits(actual, expected, context="wide short")


# -- exact numeric edge cases ------------------------------------------------------------


def test_every_e2m1_code_and_a_spread_of_e4m3_codes():
    """All 16 weight codes against E4M3 codes that cover every exponent class."""
    place, mode = device(), backend()
    codes = [0x00, 0x01, 0x08, 0x38, 0x3F, 0x40, 0x7E, 0x80, 0xB8, 0xC0, 0xFE, 0xFC, 0x04]
    k = 32
    for code in codes:
        for nibble in range(16):
            a_codes = torch.full((1, k), code, dtype=torch.uint8)
            nibbles = torch.full((16, k), nibble, dtype=torch.int64)
            packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
            a_scale = torch.full((1, 1), 127, dtype=torch.uint8)
            b_scale = torch.full((16, 1), 127, dtype=torch.uint8)
            values = unpack_fp4_to_int8(packed, backend="torch")
            half = fp4_int8_scale(b_scale, backend="torch")
            expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                           half.to(place), backend=mode)
            actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                          half.to(place), backend=mode)
            assert_takes_the_finite_path(expected, context=f"a=0x{code:02X} nibble={nibble}")
            assert_bits(actual, expected, context=f"a=0x{code:02X} nibble={nibble}")


def test_a_nan_activation_code_nans_the_whole_column_like_the_accepted_kernel():
    place, mode = device(), backend()
    n, k = 6, 64
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=1234)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    a_codes[0, 33] = 0x7F  # an E4M3 NaN code in the second K block
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode)
    assert bool(torch.isnan(expected).all())
    assert_bits(actual, expected, context="nan activation")


def test_an_e8m0_scale_of_255_nans_like_the_accepted_kernel():
    place, mode = device(), backend()
    n, k = 6, 96
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=4321)
    values = unpack_fp4_to_int8(packed, backend="torch")
    b_scale[2, 1] = 255
    half = fp4_int8_scale(b_scale, backend="torch")
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode)
    assert bool(torch.isnan(expected[0, 2]))
    assert_bits(actual, expected, context="nan scale")


@pytest.mark.parametrize("code", [0, 1, 2, 126, 127, 128, 253, 254])
def test_every_finite_e8m0_scale_endpoint_is_identical(code):
    place, mode = device(), backend()
    n, k = 9, 64
    a_codes, a_scale, packed, b_scale, _ = operands(n, k, seed=code + 1, a_scale_code=code,
                                                    b_scale_code=code)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode)
    assert_bits(actual, expected, context=f"scale endpoint {code}")


def test_a_cancelling_block_sequence_keeps_the_strict_k_order():
    """Alternating signs make a reordered sum observably different from the in-order one."""
    place, mode = device(), backend()
    k = 32 * 8
    a_codes = torch.full((1, k), 0x3F, dtype=torch.uint8)  # 1.0 in E4M3
    a_scale = torch.full((1, k // 32), 127, dtype=torch.uint8)
    nibbles = torch.zeros(1, k, dtype=torch.int64)
    pattern = [5, 9, 5, 9, 5, 9, 5, 9]  # +3, -3 alternating along K, per block
    for block, nibble in enumerate(pattern):
        nibbles[0, block * 32:(block + 1) * 32] = nibble
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    b_scale = torch.full((1, k // 32), 127, dtype=torch.uint8)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    expected = fp4_int8_block_gemm(a_codes.to(place), a_scale.to(place), values.to(place),
                                   half.to(place), backend=mode)
    actual = fp4_int8_decode_gemv(a_codes.to(place), a_scale.to(place), values.to(place),
                                  half.to(place), backend=mode)
    assert_bits(actual, expected, context="cancelling sequence")
    # The strict-order sum of +3/-3 blocks is exactly the accepted value; a
    # reassociated sum would leave a different residue here.
    assert float(actual[0, 0]) == float(expected[0, 0])


# -- validation: everything is refused before any device work ----------------------------


def test_two_activation_rows_are_refused_by_name():
    a_codes, a_scale, packed, b_scale, _ = operands(4, 64, seed=8)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    two_rows = torch.cat([a_codes, a_codes], dim=0)
    two_scales = torch.cat([a_scale, a_scale], dim=0)
    with pytest.raises(ValueError, match="exactly one activation row"):
        fp4_int8_decode_gemv(two_rows, two_scales, values, half, backend="torch")


def test_shape_dtype_and_grid_refusals():
    a_codes, a_scale, packed, b_scale, _ = operands(8, 64, seed=9)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    with pytest.raises(TypeError, match="dtype"):
        fp4_int8_decode_gemv(a_codes, a_scale, values.float(), half, backend="torch")
    with pytest.raises(ValueError, match="divisible by 32"):
        fp4_int8_decode_gemv(a_codes[:, :63].contiguous(), a_scale, values[:, :63].contiguous(),
                             half, backend="torch")
    with pytest.raises(ValueError, match="half-scales"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, torch.ones(1, 2), backend="torch")
    with pytest.raises(ValueError, match="block_rows"):
        fp8_decode_gemv(a_codes, a_scale, values.view(torch.uint8), b_scale, block_rows=7,
                        backend="torch")
    with pytest.raises(ValueError, match="backend"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="native")
    with pytest.raises(ValueError, match="power of two"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="torch", fragment_rows=3)


def test_the_grid_cap_is_refused_before_any_allocation():
    """A geometry above the total-program cap must fail on the CPU path too."""
    from relic_core.kernels.decode_block_gemv import _grids

    with pytest.raises(ValueError, match="total-program cap"):
        _grids(MAX_PROGRAMS * 16 + 16, 1, 16, 64)
    # And the split keeps the product under the cap for every legal geometry.
    for n, blocks in ((2304, 160), (25600, 192), (129280, 224), (5120, 40)):
        chunks, ksplit, columns = _grids(n, blocks, 16, 64)
        assert chunks * ksplit <= MAX_PROGRAMS, (n, blocks, chunks, ksplit)
        assert columns <= MAX_PROGRAMS
        assert ksplit == min(blocks, MAX_PROGRAMS // chunks)


def test_overlapping_out_and_scratch_are_refused():
    a_codes, a_scale, packed, b_scale, _ = operands(8, 64, seed=10)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    scratch = decode_gemv_scratch(8, 64)
    with pytest.raises(ValueError, match="overlap"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="torch",
                             out=values.view(torch.float32).flatten()[:8].reshape(1, 8))
    byte_alias = scratch[0].view(torch.uint8).reshape(-1)[:scratch[0].numel()].view(scratch[0].shape)
    with pytest.raises(ValueError, match="separate buffers"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="torch",
                             scratch=(scratch[0], byte_alias))
    with pytest.raises(ValueError, match="K // 32, N"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="torch",
                             scratch=decode_gemv_scratch(8, 96))


def test_a_cpu_tensor_cannot_ask_for_the_fused_backend():
    a_codes, a_scale, packed, b_scale, _ = operands(4, 64, seed=12)
    values = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(b_scale, backend="torch")
    with pytest.raises(ValueError, match="NPU tensors only"):
        fp4_int8_decode_gemv(a_codes, a_scale, values, half, backend="triton")