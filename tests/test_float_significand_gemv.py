"""Contracts for the floating-significand decode GEMV.

The operator under test is :mod:`relic_core.kernels.floating_significand_gemv`. Its
kernel replaces the frozen integer prototype's *varying integer shifts* and int32
significand multiply with exact FP32 significand products times an exact
power-of-two factor, keeping every other statement of the accepted block-32
reduction. These tests prove that substitution is exact; none of them needs an
accelerator.

Three independent layers:

* a **Torch mirror** of the kernel's own statements (the FP32 term, the bitcast
  carry exponent, the exact dyadic rescales, and a faithful transcription of the
  frozen ``_scale_e8m0`` tail), compared **bit for bit** against the accepted CPU
  operator ``fp4_packed_decode_gemv`` / ``fp8_decode_gemv``. This is what proves
  the rewrite computes the accepted fragment and not merely something close to it;
* an **exact-rational oracle** (``fractions.Fraction``) that checks, from the
  definition and with no float arithmetic at all, that the FP32 product really is
  the exact rational value, that the cast really is truncation toward zero, and
  that the bitcast carry exponent really is ``floor(log2)`` -- over every E4M3 x
  E2M1 pair and every finite E4M3 x E4M3 pair;
* the **contract**: shapes, dtypes, aliasing, scratch reuse, the ROWS_N <= 16 and
  grid caps, and the static rules the backend enforces at compile time (no plain
  module global, no variable shift amount in the hot helpers, no operand float
  decode, no FP64).

The NPU comparison against the accepted operator on real hardware, and the speed
question the change exists for, are the root's measurements; nothing here is a
substitute for them.
"""
from __future__ import annotations

import ast
import math
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import floating_significand_gemv as G  # noqa: E402
from relic_core.kernels.decode_block_gemv import (  # noqa: E402
    fp4_packed_decode_gemv,
    fp8_decode_gemv,
)
from relic_core.kernels.integer_decode_gemv import (  # noqa: E402
    ROWS_N_MAX,
    integer_decode_gemv_scratch,
)

EMPTY = -1000
KERNEL_MODULE = Path(G.__file__).with_name("_ascend_float_significand_kernels.py")
INTEGER_MODULE = Path(G.__file__).with_name("_ascend_integer_decode_kernels.py")


# -- the kernel's statements, in Torch ---------------------------------------------------------


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    return ((exponent.to(torch.int32) + 127) << 23).contiguous().view(torch.float32)


def _ilog2(value: torch.Tensor) -> torch.Tensor:
    """The frozen kernel's integer floor-log2, transcribed (used by _scale_e8m0 only)."""
    high = torch.where(value >= 65536, torch.full_like(value, 16), torch.zeros_like(value))
    value = value >> high
    shift = torch.where(value >= 256, torch.full_like(value, 8), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    shift = torch.where(value >= 16, torch.full_like(value, 4), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    shift = torch.where(value >= 4, torch.full_like(value, 2), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    return high + torch.where(value >= 2, torch.ones_like(value), torch.zeros_like(value))


def _fields_e4(codes: torch.Tensor):
    codes = codes.to(torch.int32)
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    nan = (codes & 127) == 127
    zero = (codes & 127) == 0
    significand = torch.where(exponent == 0, mantissa, 8 + mantissa)
    significand = torch.where(nan, torch.zeros_like(significand), significand)
    power = torch.where(exponent == 0, -9, exponent - 10)
    nominal = torch.maximum(exponent - 7, torch.full_like(exponent, -6))
    nominal = torch.where(zero, torch.full_like(nominal, EMPTY), nominal)
    return significand, power, nominal, (codes & 128) != 0, nan


def _fields_e2(code: torch.Tensor):
    magnitude = code.to(torch.int32) & 7
    significand = torch.where(magnitude < 2, magnitude, 2 + (magnitude & 1))
    power = torch.where(magnitude < 2, -1, (magnitude >> 1) - 2)
    nominal = (magnitude >> 1) - 1
    nominal = torch.where(magnitude != 0, nominal, torch.full_like(nominal, EMPTY))
    return significand, power, nominal, (code & 8) != 0


def _grid_exponent(anchor: torch.Tensor) -> torch.Tensor:
    return torch.where(anchor < -500, torch.zeros_like(anchor), anchor - 13)


def _terms_float(significand_product, power, grid, negative, half):
    """The new term: exact FP32 product by an exact dyadic factor, then truncate to int."""
    factor = _pow2(power - grid[:, None])
    magnitude = (significand_product.to(torch.float32) * factor).to(torch.int32)
    magnitude = torch.where(half, magnitude, torch.zeros_like(magnitude))
    return torch.where(negative, -magnitude, magnitude)


def _scale_int_float(value, shift):
    """``trunc(value * 2**shift)`` through FP32, the kernel's exact dyadic rescale."""
    return (value.to(torch.float32) * _pow2(shift)).to(torch.int32)


def _floor_exp(value: torch.Tensor) -> torch.Tensor:
    bits = value.view(torch.int32) & 0x7FFFFFFF
    exponent = ((bits >> 23) & 255) - 127
    return torch.where(bits != 0, exponent, torch.full_like(exponent, EMPTY))


def _scale_e8m0(value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """The frozen ``_scale_e8m0``, transcribed so the mirror applies the accepted tail."""
    code = code.to(torch.int32)
    raw = value.view(torch.int32)
    sign = (raw >> 31) << 31
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7FFFFF
    significand = torch.where(exponent == 0, mantissa, mantissa | 0x800000)
    power = torch.where(exponent == 0, torch.full_like(exponent, -149),
                        exponent - 150) + code - 127
    msb = _ilog2(significand)
    biased = power + msb + 127
    normal = torch.minimum(torch.maximum(biased, torch.zeros_like(biased)),
                           torch.full_like(biased, 255)) << 23
    normal = normal | ((significand << (23 - msb)) & 0x7FFFFF)
    shift = power + 149
    left = torch.minimum(torch.maximum(shift, torch.zeros_like(shift)), torch.full_like(shift, 23))
    right = torch.minimum(torch.maximum(-shift, torch.ones_like(shift)), torch.full_like(shift, 24))
    truncated = significand >> right
    remainder = significand & ((1 << right) - 1)
    halfway = 1 << (right - 1)
    increment = (remainder > halfway) | ((remainder == halfway) & ((truncated & 1) != 0))
    rounded = truncated + increment.to(torch.int32)
    rounded = torch.where(shift < -24, 0, rounded)
    subnormal = torch.where(shift >= 0, significand << left, rounded)
    magnitude = torch.where(biased <= 0, subnormal,
                            torch.where(biased >= 255, 0x7F800000, normal))
    magnitude = torch.where(significand == 0, 0, magnitude)
    magnitude = torch.where(exponent == 255, raw & 0x7FFFFFFF, magnitude)
    is_nan = ((exponent == 255) & (mantissa != 0)) | (code == 255)
    magnitude = torch.where(is_nan, 0x7FC00000, magnitude)
    return (sign | magnitude).contiguous().view(torch.float32)


def _original_scale_code(scale: torch.Tensor, half: bool) -> torch.Tensor:
    """The frozen ``_original_scale_code``: exact FP32 power back to its E8M0 byte."""
    bits = scale.view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    if half:
        code = torch.where(bits == 0x00200000, 0,
                           torch.where(bits == 0x00400000, 1, exponent + 1))
        valid = (bits == 0x00200000) | (bits == 0x00400000)
        valid = valid | (((bits >> 31) == 0) & (mantissa == 0)
                         & (exponent >= 1) & (exponent <= 253))
    else:
        code = torch.where(bits == 0x00400000, 0, exponent)
        valid = (bits == 0x00400000)
        valid = valid | (((bits >> 31) == 0) & (mantissa == 0)
                         & (exponent >= 1) & (exponent <= 254))
    return torch.where(valid, code, 255)


def _mirror_fragment(a_codes, a_scale_code, b_codes, *, b_is_e4):
    """One fragment from the mirror's FP32 statements, then the accepted scale tail."""
    sig_a, pa, nom_a, neg_a, _nan_a = _fields_e4(a_codes)
    if b_is_e4:
        sig_b, pb, nom_b, neg_b = _fields_e4(b_codes)[:4]
    else:
        sig_b, pb, nom_b, neg_b = _fields_e2(b_codes)
    significand_product = sig_a[None, :] * sig_b
    power = pa[None, :] + pb
    nominal = nom_a[None, :] + nom_b
    negative = neg_a[None, :] ^ neg_b
    lanes = torch.arange(32)

    grid0 = _grid_exponent(nominal[:, :16].max(-1).values)
    total0 = _terms_float(significand_product, power, grid0, negative, lanes[None, :] < 16).sum(-1)
    carry = torch.where(total0 == 0, torch.full_like(total0, EMPTY),
                        _floor_exp(total0.to(torch.float32)) + grid0)
    second = torch.maximum(nominal[:, 16:].max(-1).values, carry)
    grid1 = _grid_exponent(second)
    whole = _terms_float(significand_product, power, grid1, negative, lanes[None, :] >= 16).sum(-1)
    whole = whole + _scale_int_float(total0, grid0 - grid1)

    exponent = torch.where(whole == 0, torch.full_like(whole, EMPTY),
                           _floor_exp(whole.to(torch.float32)) + grid1)
    window = _grid_exponent(exponent)
    return _scale_int_float(whole, grid1 - window).to(torch.float32) * _pow2(window)


def _mirror_packed(a, sa, b_packed, sb):
    """Accumulate in increasing block order, applying the accepted scale tail twice."""
    n, k = b_packed.shape[0], a.shape[1]
    out = torch.zeros((1, n), dtype=torch.float32)
    for block in range(k // 32):
        packed = b_packed[:, block * 16:(block + 1) * 16].to(torch.int32)
        codes = torch.empty((n, 32), dtype=torch.int32)
        codes[:, 0::2] = packed & 15
        codes[:, 1::2] = (packed >> 4) & 15
        fragment = _mirror_fragment(
            a[0, block * 32:(block + 1) * 32], sa[0, block], codes, b_is_e4=False
        )
        out = out + _scale_e8m0(_scale_e8m0(fragment, sa[0, block]), sb[:, block])
    return out


def _mirror_fp8(a, sa, b_codes, sb, block_rows):
    n, k = b_codes.shape[0], a.shape[1]
    out = torch.zeros((1, n), dtype=torch.float32)
    for block in range(k // 32):
        codes = b_codes[:, block * 32:(block + 1) * 32].to(torch.int32)
        fragment = _mirror_fragment(
            a[0, block * 32:(block + 1) * 32], sa[0, block], codes, b_is_e4=True
        )
        rows = torch.arange(n) // block_rows
        raw = sb[rows, block]
        code_b = raw if raw.dtype != torch.float32 else _original_scale_code(raw, False)
        out = out + _scale_e8m0(_scale_e8m0(fragment, sa[0, block]), code_b)
    return out


def _assert_same_reduction(mirror: torch.Tensor, accepted: torch.Tensor, note: str = "") -> None:
    """Bit equality on finite lanes, classification equality everywhere else.

    A column is finite or it is not, and that is decided by the *reduction*: an
    invalid operand or an overflowing rescale. The NaN **payload** is not a parity
    contract anywhere in this package (the frozen unpack and block modules say so
    explicitly: classification is preserved, payload bits are not), so the proof is
    bit-exact where the answer is a number and classification-exact where it is not.
    """
    expected_bits = accepted.view(torch.int32)
    actual_bits = mirror.view(torch.int32)
    finite = torch.isfinite(accepted) & torch.isfinite(mirror)
    mismatched = (expected_bits != actual_bits) & finite
    assert not bool(mismatched.any()), (
        f"{int(mismatched.sum())} finite lanes differ {note}: "
        f"{expected_bits[mismatched][:8].tolist()} != {actual_bits[mismatched][:8].tolist()}"
    )
    nan_expected, nan_actual = torch.isnan(accepted), torch.isnan(mirror)
    assert torch.equal(nan_expected, nan_actual), (
        f"NaN classification differs {note}: "
        f"{(nan_expected != nan_actual).sum().item()} lanes"
    )
    inf_expected, inf_actual = torch.isinf(accepted), torch.isinf(mirror)
    assert torch.equal(inf_expected & ~nan_expected, inf_actual & ~nan_actual), (
        f"infinity classification differs {note}"
    )

# -- fixtures ----------------------------------------------------------------------------------


def _codes(shape, generator):
    raw = torch.randint(0, 255, shape, generator=generator, dtype=torch.int32)
    raw[(raw & 127) == 127] = 0x38
    return raw.to(torch.uint8)


def _packed(n, k, generator):
    return torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.int32).to(torch.uint8)


def _operands(n=8, k=96, seed=3):
    generator = torch.Generator().manual_seed(seed)
    a = _codes((1, k), generator)
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    return a, sa, b, sb


# -- the mirror is the accepted operator --------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n,k", [(1, 32), (17, 64), (33, 128), (64, 256)])
def test_mirror_matches_the_accepted_fp4_packed_operator(seed, n, k):
    generator = torch.Generator().manual_seed(seed)
    a = _codes((1, k), generator)
    sa = torch.randint(0, 256, (1, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.randint(0, 256, (n, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    _assert_same_reduction(_mirror_packed(a, sa, b, sb), accepted, "fp4 packed")


@pytest.mark.parametrize("block_rows", [1, 32])
def test_mirror_matches_the_accepted_fp8_operator(block_rows):
    generator = torch.Generator().manual_seed(11)
    n, k = 40, 96
    a = _codes((1, k), generator)
    sa = torch.randint(0, 256, (1, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    b = _codes((n, k), generator)
    rows = math.ceil(n / block_rows)
    sb = torch.randint(0, 256, (rows, k // 32), generator=generator,
                       dtype=torch.int32).to(torch.uint8)
    accepted = fp8_decode_gemv(a, sa, b, sb, block_rows=block_rows, backend="torch")
    _assert_same_reduction(_mirror_fp8(a, sa, b, sb, block_rows), accepted, "fp8")


def test_mirror_matches_on_every_activation_code_against_every_nibble():
    """All 256 E4M3 activations against all 16 E2M1 nibbles in one K block."""
    n, k = 16, 32
    nibbles = torch.arange(16, dtype=torch.int32)
    b = torch.empty((n, k // 2), dtype=torch.uint8)
    for column in range(n):
        packed = torch.empty(k // 2, dtype=torch.int32)
        packed[0::2] = nibbles[:k // 4]
        packed[1::2] = nibbles.flip(0)[:k // 4]
        b[column] = packed.to(torch.uint8)
    sb = torch.full((n, 1), 127, dtype=torch.uint8)
    sa = torch.full((1, 1), 127, dtype=torch.uint8)
    for code in range(256):
        a = torch.full((1, k), code, dtype=torch.uint8)
        accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
        if (code & 127) == 127:
            assert bool(torch.isnan(accepted).all())  # invalid flag, not arithmetic
            continue
        _assert_same_reduction(_mirror_packed(a, sa, b, sb), accepted, f"byte {code:#04x}")


def test_first_half_cancellation_and_zero_carry_match_the_accepted_operator():
    """Opposite-sign first half cancels, carrying the zero sentinel into the second."""
    k = 32
    a = torch.tensor([[0x38] * k], dtype=torch.uint8)  # all 1.0
    packed = torch.zeros((2, k // 2), dtype=torch.int32)
    packed[:, :8] = 2 + 10 * 16  # +1.0 / -1.0 alternating, eight times
    packed[:, 8:] = torch.tensor([4 + 4 * 16] * 4 + [6 + 6 * 16] * 4, dtype=torch.int32)
    packed = packed.to(torch.uint8)
    sa = torch.full((1, 1), 127, dtype=torch.uint8)
    sb = torch.full((2, 1), 127, dtype=torch.uint8)
    accepted = fp4_packed_decode_gemv(a, sa, packed, sb, backend="torch")
    _assert_same_reduction(_mirror_packed(a, sa, packed, sb), accepted, "cancellation")


@pytest.mark.parametrize("scale", [0, 1, 2, 126, 127, 128, 253, 254])
def test_scale_endpoints_match_through_both_rescales(scale):
    """The two separate E8M0 rescales, including the smallest and largest finite byte."""
    generator = torch.Generator().manual_seed(5)
    n, k = 8, 64
    a = _codes((1, k), generator)
    sa = torch.full((1, k // 32), scale, dtype=torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.full((n, k // 32), scale, dtype=torch.uint8)
    accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    _assert_same_reduction(_mirror_packed(a, sa, b, sb), accepted, f"scale {scale}")


def test_float_weight_scales_are_accepted_and_match_the_code_path():
    """Exact FP32 powers of two are converted back to their byte, not multiplied twice."""
    generator = torch.Generator().manual_seed(13)
    n, k = 12, 64
    a = _codes((1, k), generator)
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b = _codes((n, k), generator)
    codes = torch.randint(120, 135, (1, k // 32), generator=generator, dtype=torch.int32)
    float_scales = _pow2(codes - 127)
    by_code = fp8_decode_gemv(a, sa, b, codes.to(torch.uint8), block_rows=32, backend="torch")
    by_float = fp8_decode_gemv(a, sa, b, float_scales, block_rows=32, backend="torch")
    assert torch.equal(by_float.view(torch.int32), by_code.view(torch.int32))
    _assert_same_reduction(_mirror_fp8(a, sa, b, float_scales, 32), by_float, "float scales")


# -- the exact-rational oracle -----------------------------------------------------------------


def _e4_exact(byte: int) -> Fraction:
    sign = -1 if byte & 128 else 1
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    if exponent == 0:
        return Fraction(sign * mantissa, 2**9)
    return Fraction(sign * (8 + mantissa), 1) * Fraction(2) ** (exponent - 10)


def _e2_exact(code: int) -> Fraction:
    magnitudes = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
                  Fraction(2), Fraction(3), Fraction(4), Fraction(6))
    value = magnitudes[code & 7]
    return -value if code & 8 else value


def _fields_e4_scalar(byte: int):
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    significand = 0 if (byte & 127) == 127 else (mantissa if exponent == 0 else 8 + mantissa)
    power = -9 if exponent == 0 else exponent - 10
    return significand, power, bool(byte & 128)


def _fields_e2_scalar(code: int):
    magnitude = code & 7
    significand = magnitude if magnitude < 2 else 2 + (magnitude & 1)
    power = -1 if magnitude < 2 else (magnitude >> 1) - 2
    return significand, power, bool(code & 8)


def test_fp32_significand_product_is_exact_for_every_e4_x_e2_pair():
    """No rounding is introduced before the term is built: the FP32 product is the rational."""
    for byte in range(256):
        if (byte & 127) == 127:
            continue  # NaN has no product to state
        sig_a, pa, neg_a = _fields_e4_scalar(byte)
        for code in range(16):
            sig_b, pb, neg_b = _fields_e2_scalar(code)
            sign = -1 if neg_a ^ neg_b else 1
            product = sig_a * sig_b
            # The significand decomposition is the definition of the encoding, so the
            # integer product is the exact value divided by its own power of two.
            exact = _e4_exact(byte) * _e2_exact(code) * sign
            assert exact == Fraction(product) * Fraction(2) ** (pa + pb), (
                f"byte {byte:#04x} code {code:#x}: significand x 2**power is not the value"
            )
            for shift in (-6, -1, 0, 3, 13, 26):
                built = Fraction(float(product)) * Fraction(2) ** shift
                assert built == Fraction(product) * Fraction(2) ** shift, (
                    f"byte {byte:#04x} code {code:#x} shift {shift}: FP32 product rounded"
                )


def test_float_term_is_the_exact_truncation_for_every_e4_x_e2_pair():
    """``trunc(sig_product * 2**exp)`` equals the exact rational truncation, all 4096 pairs."""
    for byte in range(256):
        if (byte & 127) == 127:
            continue
        sig_a, pa, neg_a = _fields_e4_scalar(byte)
        exact_a = _e4_exact(byte)
        for code in range(16):
            sig_b, pb, neg_b = _fields_e2_scalar(code)
            exact_b = _e2_exact(code)
            product = int(sig_a) * int(sig_b)
            negative = neg_a ^ neg_b
            for grid in (-20, -17, -13, -7, -4, 0, 5):
                exponent = int(pa) + int(pb) - grid
                magnitude = int(Fraction(float(product)) * Fraction(2) ** exponent)
                # The kernel casts the magnitude and applies the sign afterwards.
                term = -magnitude if negative else magnitude
                expected = int((exact_a * exact_b * Fraction(2) ** (-grid)).__trunc__())
                assert term == expected, (
                    f"byte {byte:#04x} code {code:#x} grid {grid}"
                )


def test_float_term_is_exact_for_every_finite_e4_x_e4_pair():
    """The FP8 path's products are ``<= 225`` too; prove them over all finite pairs."""
    grids = (-20, -13, -4)
    for first in range(256):
        if (first & 127) == 127:
            continue
        sig_a, pa, neg_a = _fields_e4_scalar(first)
        exact_a = _e4_exact(first)
        for second in range(256):
            if (second & 127) == 127:
                continue
            sig_b, pb, neg_b = _fields_e4_scalar(second)
            product = int(sig_a) * int(sig_b)
            assert product <= 225 and product == int(sig_a) * int(sig_b)
            negative = neg_a ^ neg_b
            exact_product = exact_a * _e4_exact(second)
            for grid in grids:
                exponent = int(pa) + int(pb) - grid
                magnitude = int(Fraction(float(product)) * Fraction(2) ** exponent)
                term = -magnitude if negative else magnitude
                expected = int((exact_product * Fraction(2) ** (-grid)).__trunc__())
                assert term == expected, (
                    f"bytes {first:#04x} x {second:#04x} grid {grid}"
                )


def test_bitcast_carry_exponent_is_exactly_floor_log2():
    """The replacement for ``_ilog2``: the FP32 exponent field of the exact integer."""
    def _ilog2_scalar(value: int) -> int:
        return max(0, value.bit_length() - 1)

    def _floor_exp_scalar(value: int) -> int:
        bits = int(torch.tensor(float(value), dtype=torch.float32).view(torch.int32))
        return ((bits & 0x7FFFFFFF) >> 23) - 127

    values = []
    for power in range(0, 20):
        for delta in (-2, -1, 0, 1, 2):
            candidate = 2**power + delta
            if 0 < candidate < 2**20:
                values.append(candidate)
    generator = torch.Generator().manual_seed(23)
    values.extend(int(v) for v in torch.randint(1, 2**20, (200,), generator=generator))
    for value in values:
        assert Fraction(float(value)) == Fraction(value), "the int -> FP32 cast rounded"
        assert _floor_exp_scalar(value) == _ilog2_scalar(value), f"value {value}"


def test_exact_dyadic_rescale_is_the_truncating_shift():
    """``(int.to(fp32) * 2**shift).to(int32)`` equals the frozen sign-preserving shift."""
    generator = torch.Generator().manual_seed(29)
    for shift in range(-26, 14):
        # The reachable domain: |value| < 2**24 (exact in FP32) and, for a left
        # shift, a result that still fits int32 -- which the reduction guarantees
        # because the dominating anchor is the carry's own exponent. Scaling an
        # arbitrary 2**20 integer by 2**12 is outside the operator's contract and
        # overflows int32, so the test stays inside it rather than widening a bound.
        limit = 2**20 if shift >= 0 else 2**24
        if shift > 0:
            limit = min(limit, (2**30) >> shift)
        values = torch.randint(-limit, limit, (512,), generator=generator).to(torch.int32)
        values = torch.cat([values, torch.tensor([0, 1, -1, limit - 1, -(limit - 1)])])
        factor = _pow2(torch.tensor([shift], dtype=torch.int32)).item()
        got = (values.to(torch.float32) * factor).to(torch.int32)
        magnitude = values.abs()
        if shift >= 0:
            scaled = magnitude << shift
        else:
            scaled = magnitude >> (-shift)
        expected = torch.where(values < 0, -scaled, scaled)
        assert torch.equal(got, expected), f"shift {shift}"


# -- contract ----------------------------------------------------------------------------------


def test_cpu_backend_delegates_and_rejects_the_fused_backend():
    a, sa, b, sb = _operands()
    out = G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch")
    assert out.shape == (1, 8) and out.dtype == torch.float32
    accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    assert torch.equal(out.view(torch.int32), accepted.view(torch.int32))
    with pytest.raises(ValueError, match="NPU tensors only"):
        G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="triton")


@pytest.mark.parametrize("variant", ["auto", "serial", "grouped"])
def test_every_variant_is_accepted_and_the_cpu_path_answers(variant):
    a, sa, b, sb = _operands()
    out = G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", variant=variant)
    assert torch.equal(out.view(torch.int32),
                       fp4_packed_decode_gemv(a, sa, b, sb, backend="torch").view(torch.int32))


def test_rows_beyond_the_measured_unified_buffer_budget_are_refused_by_name():
    """"32 columns was measured at 3,440,640 bits against a 1,572,864-bit budget."""
    a, sa, b, sb = _operands()
    assert ROWS_N_MAX == 16
    with pytest.raises(ValueError, match="exceeds the supported maximum 16"):
        G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", rows_n=32)
    for bad in (0, -1, 3):
        with pytest.raises(ValueError):
            G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", rows_n=bad)
    with pytest.raises(ValueError):
        G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", group_k=3)


def test_the_grid_cap_and_the_index_width_are_decided_before_launch():
    """The aggregate program bound and the 64-bit pointer decision are pure functions."""
    from relic_core.kernels.decode_block_gemv import MAX_PROGRAMS
    from relic_core.kernels.integer_decode_gemv import _index_width, _integer_grids

    # One column per program over more columns than the aggregate cap permits.
    with pytest.raises(ValueError, match="total-program cap"):
        _integer_grids(MAX_PROGRAMS + 1, 3, 1, 4)
    chunks, groups, ksplit = _integer_grids(4096, 8, 16, 4)
    assert chunks * ksplit <= MAX_PROGRAMS and groups == 2
    assert _index_width(2**17, 2**15, 2**15) is True   # N*K passes 2**31
    assert _index_width(8, 96, 3) is False
    a, sa, b, sb = _operands(n=8, k=96)
    assert G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch").shape == (1, 8)


def test_scratch_reuse_and_alias_refusals_come_from_the_frozen_decoder():
    a, sa, b, sb = _operands()
    scratch = G.floating_significand_gemv_scratch(8, 96)
    assert [tuple(t.shape) for t in scratch] == [(3, 8), (3, 8)]
    out = G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", scratch=scratch)
    assert out.shape == (1, 8)
    bad = (torch.empty((2, 8), dtype=torch.float32), torch.empty((3, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match="fragment scratch must be"):
        G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", scratch=bad)
    # ``out`` aliasing an input is refused by the frozen output contract.
    with pytest.raises((ValueError, TypeError)):
        G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch", out=a.float())


def test_an_int8_container_is_read_as_the_same_packed_bytes():
    """The released bytes may arrive in an ``int8`` container; it is a view, not a value."""
    a, sa, b, sb = _operands()
    signed = b.view(torch.int8)
    assert signed.dtype == torch.int8
    as_uint8 = G.fp4_packed_float_significand_gemv(a, sa, b, sb, backend="torch")
    as_int8 = G.fp4_packed_float_significand_gemv(a, sa, signed, sb, backend="torch")
    assert torch.equal(as_uint8.view(torch.int32), as_int8.view(torch.int32))
    # Expanded INT8-times-two storage is a different operator and is not accepted here.
    with pytest.raises(ValueError, match=r"\[N, 48\]|packed FP4"):
        G.fp4_packed_float_significand_gemv(a, sa, b[:, :16].to(torch.int8), sb)


def test_no_environment_flag_or_permission_argument_gates_the_operator():
    import inspect

    signature = inspect.signature(G.fp4_packed_float_significand_gemv)
    assert "permission" not in signature.parameters
    assert "allow_" not in " ".join(signature.parameters)
    source = Path(G.__file__).read_text()
    assert "os.environ" not in source and "getenv" not in source


# -- the backend's compile-time rules -----------------------------------------------------------


def test_the_kernel_module_compiles_the_static_rules_the_backend_enforces():
    """No plain module global, no operand float decode, no FP64, no matrix unit."""
    source = KERNEL_MODULE.read_text()
    tree = ast.parse(source)
    declared = {
        node.target.id for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.target, ast.Name)
    }
    jit_names = {
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    }
    imported = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
    bodies = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    assert bodies
    builtins = {"range", "float", "int", "bool", "min", "max", "abs"}
    for function in bodies:
        allowed = {"tl", "triton"} | jit_names | imported | builtins
        allowed |= {argument.arg for argument in function.args.args}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                pending = list(node.targets)
                while pending:
                    target = pending.pop()
                    if isinstance(target, ast.Name):
                        allowed.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        pending.extend(target.elts)
        for node in ast.walk(function):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                allowed.add(node.target.id)
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name in allowed:
                    continue
                if name in declared:
                    raise AssertionError(
                        f"{function.name} reads module global {name!r}; it must be a "
                        "tl.constexpr so the compiled-kernel cache key is stable"
                    )
                raise AssertionError(f"{function.name} reads unknown global {name!r}")
    for forbidden in ("_decode_e4m3", "unpack_fp4_to_int8", "_e2m1_x2",
                      "float64", "pow(", "tl.dot"):
        assert forbidden not in source, f"the exact path must not use {forbidden}"


def test_the_hot_helpers_use_no_variable_shift_amount():
    """The point of this module: shifts in the term/carry path are literal-only.

    A varying shift amount is what the frozen prototype lowered outside the vector
    unit; here the only shifts are the constant ``>> 23`` / ``<< 23`` of the
    exponent bitcast and the constant nibble selects, which are spelled in the
    kernels. Every other per-lane scaling goes through ``_pow2`` + an FP32 multiply.
    """
    tree = ast.parse(KERNEL_MODULE.read_text())
    hot = {"_float_terms", "_float_scale_int"}
    seen = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in hot:
            continue
        seen.add(node.name)
        for inner in ast.walk(node):
            if isinstance(inner, (ast.LShift, ast.RShift)):
                assert isinstance(inner.right, ast.Constant), (
                    f"{node.name} shifts by a non-literal amount "
                    f"({ast.unparse(inner.right)}); the term path must not"
                )
    assert seen == hot, "the hot helpers moved; update this proof with them"


def test_the_significands_are_cast_to_fp32_before_they_are_multiplied():
    """The whole hypothesis: no int32 multiply on the per-lane path.

    The frozen integer prototype multiplies two significands in ``int32`` and the
    reported cost is what this operator exists to avoid, so the product has to be
    formed in FP32 from operands that were cast *before* the multiply. Both
    significands are ``<= 15``, so the FP32 product is exact (``<= 225``).
    """
    tree = ast.parse(KERNEL_MODULE.read_text())
    fragment = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_float_fragment")
    products = [
        node for node in ast.walk(fragment)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.MatMult))
    ]
    significand_products = [
        node for node in products
        if any(isinstance(inner, ast.Name) and inner.id == "significand_product"
               for inner in ast.walk(node) if isinstance(inner, ast.Assign))
    ]
    assignment = next(
        node for node in ast.walk(fragment)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "significand_product"
                for target in node.targets)
    )
    value = assignment.value
    assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult)
    for operand in (value.left, value.right):
        assert isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute), (
            "each significand must be cast before the multiply: "
            f"got {ast.unparse(operand)}"
        )
        assert operand.func.attr == "to", f"expected a cast, got {ast.unparse(operand)}"
        target = ast.unparse(operand.args[0])
        assert target.startswith("tl.float32"), (
            f"the significand must be cast to FP32 before the multiply, got {target}"
        )
    # And the term helper must not introduce one either: it receives the FP32 product.
    terms = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_float_terms")
    term_multiplies = [
        node for node in ast.walk(terms)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
    ]
    assert term_multiplies, "the term must be a product"
    assert any(
        isinstance(operand, ast.Name) and operand.id == "significand_product"
        for node in term_multiplies for operand in (node.left, node.right)
    ), "the term multiply must consume the FP32 significand product"
    # The scaled factor comes from _pow2 (a bit-built power of two), never from an
    # integer shift of an int32 significand.
    assert "_pow2" in ast.unparse(terms)


def test_the_nibble_select_is_static_and_the_fragment_has_no_pointer_math():
    """Loads stay in the kernels; the arithmetic helper takes values only."""
    tree = ast.parse(KERNEL_MODULE.read_text())
    fragment = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_float_fragment")
    for node in ast.walk(fragment):
        if isinstance(node, ast.Attribute) and node.attr == "load":
            raise AssertionError("_float_fragment must not load: rank analysis aborts")
    source = KERNEL_MODULE.read_text()
    # The two packed nibble selects are the frozen constant-shift form.
    assert "weight & 15" in source and "weight >> 4" in source


def test_the_two_half_anchor_grid_and_window_are_the_frozen_ones():
    """The arithmetic contract is inherited, not re-derived: prove it by provenance."""
    source = KERNEL_MODULE.read_text()
    for name in ("_grid_exponent", "_floor_exp", "_pow2", "_scale_e8m0",
                 "_original_scale_code", "_e4_fields", "_e2m1_fields", "_scale_fast"):
        assert name in source, f"the frozen helper {name} must be imported, not re-derived"
    integer_source = INTEGER_MODULE.read_text()
    assert "def _e4_fields" in integer_source
    # The scaled fragment is rebuilt exactly once, through the exact dyadic path.
    assert "_float_scale_int(whole, grid1 - window)" in source
    assert "_float_scale_int(total0, grid0 - grid1)" in source