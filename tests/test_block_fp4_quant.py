"""The original block-wise FP4 activation quantizer, against an independent oracle.

Every case here is compared bit-for-bit with a second implementation written in
exact rational arithmetic: values are ``Fraction``s, FP32 rounding is modelled
explicitly, and the E2M1 code is a nearest-neighbour search over the codebook
rather than the kernel's integer comparison ladder. The oracle therefore shares
no rounding decision with the code under test -- it only shares the *contract*
(``inference/kernel.py:127-205``), which is what the assertions are about.

Device cases are opt-in: set ``RELIC_BLOCK_FP4_QUANT_TEST_DEVICE=npu:0`` on a
leased Ascend host to compare the fused Triton kernel with the Torch reference
bit-for-bit. Without it the file is a CPU-only suite and the NPU cases skip.
"""
from __future__ import annotations

import math
import os
import struct
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels.block_fp4_quant import (  # noqa: E402
    quantize_block_fp4,
    supports_triton_ascend,
)

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("the Torch reference needs float8_e4m3fn", allow_module_level=True)

from relic_core.kernels.unpack import unpack_fp4_to_int8  # noqa: E402

DEVICE = os.environ.get("RELIC_BLOCK_FP4_QUANT_TEST_DEVICE", "")
NPU_DEVICE = torch.device(DEVICE) if DEVICE.startswith("npu") else None
requires_npu = pytest.mark.skipif(
    NPU_DEVICE is None,
    reason="requires RELIC_BLOCK_FP4_QUANT_TEST_DEVICE=npu:0 on a real Ascend host")

FP4_MAX = 6.0
INV_FP4_MAX = struct.unpack("<f", struct.pack("<f", 1.0 / 6.0))[0]
FLOOR = {"e8m0": 6.0 * 2.0**-126, "e4m3": 6.0 * 2.0**-9}
NAN_SCALE = {"e8m0": 255, "e4m3": 0x7F}
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: Midpoints where two codes are equidistant; the tie goes to the even code.
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
DTYPES = (torch.float32, torch.bfloat16)


# --- exact oracle -------------------------------------------------------------

def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    return exponent


def _ceil_log2(value: Fraction) -> int:
    exponent = _floor_log2(value)
    return exponent if Fraction(2) ** exponent == value else exponent + 1


def _round_half_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _f32_round(value: Fraction) -> float:
    """The nearest float32 to an exact rational, ties to even, as a Python float."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    if exponent > 127:
        return math.copysign(math.inf, sign)
    if exponent >= -126:
        step = Fraction(2) ** (exponent - 23)
        significand = _round_half_even(magnitude / step)
        if significand == 2**24:  # rounded up a binade
            significand //= 2
            exponent += 1
            if exponent > 127:
                return math.copysign(math.inf, sign)
        bits = (exponent + 127) << 23 | (significand - 2**23)
        return sign * _value(bits)
    step = Fraction(2) ** -149
    significand = _round_half_even(magnitude / step)
    if significand >= 2**23:
        return sign * _value(0x00800000)  # rounded up into the smallest normal
    return sign * _value(significand)


def _e4m3_round(value: Fraction) -> int:
    """The E4M3 byte an exact rational rounds to (RNE, saturating at 448)."""
    sign = 0x80 if value < 0 else 0
    magnitude = min(abs(value), Fraction(int(448)))
    if magnitude == 0:
        return sign
    exponent = _floor_log2(magnitude)
    if exponent >= -6:
        step = Fraction(2) ** (exponent - 3)
        significand = _round_half_even(magnitude / step)
        if significand == 16:
            significand //= 2
            exponent += 1
        return sign | ((exponent + 7) << 3) | (significand - 8)
    step = Fraction(2) ** -9
    significand = _round_half_even(magnitude / step)
    if significand >= 8:
        return sign | (1 << 3)
    return sign | significand


def _e4m3_value(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    if (byte & 0x7F) == 0x7F:
        return math.nan
    if exponent == 0:
        return sign * math.ldexp(float(mantissa), -9)
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def _e2m1_index(value: Fraction) -> int:
    """Magnitude code 0..7 an exact rational rounds to, ties to the even code."""
    if value <= 0:
        return 0
    if value > Fraction(6):
        value = Fraction(6)
    upper = 1
    while upper < len(MAGNITUDES) and Fraction(MAGNITUDES[upper]).__lt__(value):
        upper += 1
    if upper == len(MAGNITUDES):
        return len(MAGNITUDES) - 1
    below = Fraction(MAGNITUDES[upper - 1])
    above = Fraction(MAGNITUDES[upper])
    if value - below < above - value:
        return upper - 1
    if value - below > above - value:
        return upper
    return upper if upper % 2 == 0 else upper - 1


def _oracle_block(values: list[float], scale_format: str) -> tuple[list[int], int]:
    """``(codes, scale byte)`` for one block of float32 values."""
    finite = [abs(value) for value in values if math.isfinite(value)]
    nonfinite = len(finite) != len(values)
    amax = max(finite) if finite else 0.0
    floor = FLOOR[scale_format]
    amax = max(amax, floor) if not nonfinite else floor
    if nonfinite:
        return [0] * len(values), NAN_SCALE[scale_format]
    if scale_format == "e8m0":
        product = _f32_round(Fraction(amax) * Fraction(INV_FP4_MAX))
        exponent = _ceil_log2(Fraction(product))
        scale_byte = exponent + 127
        scale = math.ldexp(1.0, exponent)
    else:
        ratio = _f32_round(Fraction(amax) / Fraction(FP4_MAX))
        scale_byte = _e4m3_round(Fraction(ratio))
        scale = _e4m3_value(scale_byte)
    codes = []
    for value in values:
        normalized = _f32_round(Fraction(value) / Fraction(scale))
        normalized = min(max(normalized, -FP4_MAX), FP4_MAX)
        index = _e2m1_index(Fraction(abs(normalized)))
        # The sign comes from the input, not from the ``Fraction``: a rational has
        # no signed zero, and IEEE division by a positive scale preserves the sign
        # of a zero exactly as it preserves the sign of a non-zero value.
        codes.append(index + (8 if math.copysign(1.0, value) < 0 else 0))
    return codes, scale_byte


def _oracle(values, block_size: int, scale_format: str):
    """Codes and scale bytes for a nested Python list of float32 values."""
    flat = [float(value) for value in _flatten(values)]
    codes, scales = [], []
    for start in range(0, len(flat), block_size):
        block = flat[start:start + block_size]
        block_codes, scale = _oracle_block(block, scale_format)
        codes.extend(block_codes)
        scales.append(scale)
    return codes, scales


def _flatten(values):
    for value in values:
        if isinstance(value, list):
            yield from _flatten(value)
        else:
            yield value


def _reference_dequantized(codes: list[int], scales: list[int], block_size: int,
                           scale_format: str) -> list[float]:
    """The reference's own inplace dequantization: fp4 value * scale, in FP32."""
    signed = (*MAGNITUDES, *(-magnitude for magnitude in MAGNITUDES))
    result = []
    for index, code in enumerate(codes):
        scale_byte = scales[index // block_size]
        if scale_format == "e8m0":
            scale = math.ldexp(1.0, scale_byte - 127) if scale_byte != 255 else math.nan
        else:
            scale = _e4m3_value(scale_byte)
        result.append(signed[code] * scale)
    return result


def _pack(codes: list[int]) -> list[int]:
    return [codes[index] | (codes[index + 1] << 4) for index in range(0, len(codes), 2)]


# --- fixture shapes ----------------------------------------------------------

def _blocks(count: int, block_size: int, generator: torch.Generator) -> torch.Tensor:
    """A ``[count, block_size]`` FP32 tensor covering the interesting magnitudes."""
    return (torch.randn(count, block_size, generator=generator) * 8).clamp(-40, 40)


def _assert_matches_oracle(x: torch.Tensor, block_size: int, scale_format: str) -> None:
    packed, scales = quantize_block_fp4(x, block_size, scale_format, backend="torch")
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32).tolist()
    expected_codes, expected_scales = _oracle(flat, block_size, scale_format)
    assert scales.reshape(-1).tolist() == expected_scales
    assert packed.reshape(-1).tolist() == _pack(expected_codes), "packed nibbles disagree"


# --- the oracle itself -------------------------------------------------------

def test_the_oracle_f32_rounding_matches_torch_division():
    """The oracle models FP32 division exactly; check it against the hardware op."""
    generator = torch.Generator().manual_seed(7)
    for _ in range(2000):
        numerator = torch.randn(1, generator=generator).item() * 10.0 ** int(
            torch.randint(-20, 20, (1,), generator=generator))
        denominator = torch.randn(1, generator=generator).item() * 10.0 ** int(
            torch.randint(-20, 20, (1,), generator=generator))
        if denominator == 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
            continue
        ratio = Fraction(_value(_bits(numerator))) / Fraction(_value(_bits(denominator)))
        expected = _f32_round(ratio)
        actual = (torch.tensor([numerator], dtype=torch.float32)
                  / torch.tensor([denominator], dtype=torch.float32))
        assert _bits(actual.item()) == _bits(expected) or (actual.item() != actual.item())


# --- operator behaviour ------------------------------------------------------

@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("dtype", DTYPES)
def test_random_blocks_match_the_exact_oracle(scale_format, block_size, dtype):
    generator = torch.Generator().manual_seed(1000 + block_size)
    x = _blocks(41, block_size, generator).to(dtype)
    _assert_matches_oracle(x, block_size, scale_format)


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_every_codebook_value_and_midpoint_rounds_as_the_reference(scale_format):
    """One block per case: an amax of 6 fixes scale 1, so the rest is the raw rule."""
    cases = []
    for value in MAGNITUDES + tuple(-magnitude for magnitude in MAGNITUDES):
        cases.append([6.0] * 32 if value == 0 else [6.0] + [value] * 31)
    for midpoint in MIDPOINTS:
        cases.append([6.0] + [midpoint] * 31)
        cases.append([6.0] + [-midpoint] * 31)
        # A few ulps either side of the tie, so a comparison that is off by one
        # unit in the last place cannot pass.
        above = _value(_bits(midpoint) + 1)
        below = _value(_bits(midpoint) - 1)
        cases.append([6.0, above] + [0.0] * 30)
        cases.append([6.0, below] + [0.0] * 30)
    for code, value in enumerate(MAGNITUDES):
        for sign in (1.0, -1.0):
            row = [6.0] + [sign * value] * 31
            packed, scales = quantize_block_fp4(torch.tensor([row], dtype=torch.float32),
                                               32, scale_format, backend="torch")
            expected_codes, expected_scales = _oracle([row], 32, scale_format)
            assert scales.reshape(-1).tolist() == expected_scales
            assert packed.reshape(-1).tolist() == _pack(expected_codes), (code, sign, value)
    x = torch.tensor(cases, dtype=torch.float32)
    _assert_matches_oracle(x, 32, scale_format)


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_clamp_and_extreme_magnitudes(scale_format):
    huge = torch.finfo(torch.float32).max
    rows = [
        [0.0] * 32,
        [-0.0] * 32,
        [huge] * 32,
        [-huge] + [0.0] * 31,
        [1e-38] * 32,
        [1e-45] * 32,
        [6.0] * 32,
        [-6.0] * 32,
        [6.0] + [6.0000005] * 31,
    ]
    x = torch.tensor(rows, dtype=torch.float32)
    _assert_matches_oracle(x, 32, scale_format)


@pytest.mark.parametrize("amax", [2688.0, 2784.0, 2816.0, 2880.0, 3000.0])
def test_e4m3_scale_saturates_above_the_ratio_that_overflows(amax):
    """A saturating scale byte, not the NaN byte, for an amax above 2688.

    The block is mixed so the amax is the only large lane: the remaining lanes still
    have to encode against a scale of 448.
    """
    values = [amax, 448.0, 224.0, 0.01171875, -1.0, 0.0, -0.0, 1e-30] * 4
    x = torch.tensor([values], dtype=torch.float32)
    packed, scales = quantize_block_fp4(x, 32, "e4m3", backend="torch")
    assert scales.reshape(-1).tolist() == [0x7E], "the scale byte must saturate to 448"
    _assert_matches_oracle(x, 32, "e4m3")
    modified = x.clone()
    quantize_block_fp4(modified, 32, "e4m3", inplace=True, backend="torch")
    assert torch.isfinite(modified).all(), "a saturated scale must not leak a NaN"


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_negative_zero_survives_in_the_code_and_the_inplace_result(scale_format):
    x = torch.tensor([[-0.0] * 32, [0.0] * 32], dtype=torch.float32)
    packed, _ = quantize_block_fp4(x, 32, scale_format, backend="torch")
    codes, _ = _oracle([[-0.0] * 32], 32, scale_format)
    assert packed[0].tolist() == _pack(codes), "the sign of zero must reach the code"
    assert packed[1].tolist() == [0] * 16
    modified = x.clone()
    quantize_block_fp4(modified, 32, scale_format, inplace=True, backend="torch")
    assert torch.signbit(modified[0]).all() and not torch.signbit(modified[1]).any()


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_nonfinite_blocks_follow_the_documented_extension(scale_format):
    rows = [[math.inf] + [1.0] * 31, [math.nan] * 32, [-math.inf] * 32, [2.0] * 32]
    x = torch.tensor(rows, dtype=torch.float32)
    packed, scales = quantize_block_fp4(x, 32, scale_format, backend="torch")
    nan_byte = NAN_SCALE[scale_format]
    assert scales.reshape(-1).tolist()[:3] == [nan_byte] * 3
    assert packed[:3].tolist() == [[0] * 16] * 3, "E2M1 has no NaN code; codes go to zero"
    modified = x.clone()
    quantize_block_fp4(modified, 32, scale_format, inplace=True, backend="torch")
    assert torch.isnan(modified[:3]).all() and torch.isfinite(modified[3]).all()


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_all_zero_block_keeps_a_nonzero_scale(scale_format):
    """The floors exist so that a zero block's scale can still be divided out."""
    packed, scales = quantize_block_fp4(torch.zeros(2, 32, dtype=torch.bfloat16), 32,
                                        scale_format, backend="torch")
    assert scales.reshape(-1).tolist() == [1, 1]
    assert packed.tolist() == [[0] * 16] * 2
    modified = torch.zeros(2, 32, dtype=torch.bfloat16)
    quantize_block_fp4(modified, 32, scale_format, inplace=True, backend="torch")
    assert torch.equal(modified, torch.zeros_like(modified))


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_inplace_matches_the_reference_dequantization_for_both_dtypes(scale_format):
    generator = torch.Generator().manual_seed(55)
    for dtype in DTYPES:
        x = _blocks(9, 32, generator).to(dtype)
        flat = x.to(torch.float32).reshape(-1).tolist()
        codes, scales = _oracle(flat, 32, scale_format)
        expected = _reference_dequantized(codes, scales, 32, scale_format)
        expected_tensor = torch.tensor(expected, dtype=torch.float32).to(dtype).reshape(x.shape)
        modified = x.clone()
        result = quantize_block_fp4(modified, 32, scale_format, inplace=True, backend="torch")
        assert result is modified, "inplace must return the tensor it modified"
        # Compare raw bits: BF16 narrowing is a rounding of the FP32 dequantized
        # value, and nothing here should be near a NaN payload.
        assert torch.equal(modified.view(torch.int16), expected_tensor.view(torch.int16)), dtype


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_packed_codes_read_back_through_the_frozen_unpack_operator(scale_format):
    """The nibble order is the released packed-FP4 order, checked by the unpack op."""
    generator = torch.Generator().manual_seed(77)
    x = _blocks(4, 32, generator)
    packed, scales = quantize_block_fp4(x, 32, scale_format, backend="torch")
    widened = unpack_fp4_to_int8(packed, backend="torch")
    flat = x.reshape(-1).tolist()
    codes, expected_scales = _oracle([flat], 32, scale_format)
    signed = (*MAGNITUDES, *(-magnitude for magnitude in MAGNITUDES))
    assert widened.reshape(-1).tolist() == [int(2 * signed[code]) for code in codes]
    assert scales.reshape(-1).tolist() == expected_scales


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_leading_axes_and_block_16_are_flattened_rows(scale_format):
    generator = torch.Generator().manual_seed(91)
    x = (torch.randn(2, 3, 16, generator=generator) * 3).to(torch.bfloat16)
    packed, scales = quantize_block_fp4(x, 16, scale_format, backend="torch")
    assert packed.shape == (2, 3, 8) and scales.shape == (2, 3, 1)
    flat = x.reshape(-1, 16).to(torch.float32).tolist()
    expected_codes, expected_scales = _oracle(flat, 16, scale_format)
    assert packed.reshape(-1).tolist() == _pack(expected_codes)
    assert scales.reshape(-1).tolist() == expected_scales


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_out_pair_is_written_in_place_and_returned(scale_format):
    x = torch.randn(2, 32, generator=torch.Generator().manual_seed(3))
    packed = torch.empty(2, 16, dtype=torch.uint8)
    scales = torch.empty(2, 1, dtype=torch.uint8)
    actual = quantize_block_fp4(x, 32, scale_format, backend="torch", out=(packed, scales))
    assert actual[0] is packed and actual[1] is scales
    reference = quantize_block_fp4(x, 32, scale_format, backend="torch")
    assert torch.equal(packed, reference[0]) and torch.equal(scales, reference[1])


def test_empty_rows_and_zero_length_leading_axis():
    x = torch.empty(0, 32, dtype=torch.float32)
    packed, scales = quantize_block_fp4(x, 32, "e8m0", backend="torch")
    assert packed.shape == (0, 16) and scales.shape == (0, 1)


# --- refusals ----------------------------------------------------------------

def test_refusals_name_the_contract():
    x = torch.zeros(1, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="block_size"):
        quantize_block_fp4(x, 8, "e8m0", backend="torch")
    with pytest.raises(ValueError, match="multiple of"):
        quantize_block_fp4(torch.zeros(1, 24, dtype=torch.float32), 16, "e8m0", backend="torch")
    with pytest.raises(ValueError, match="scale_format"):
        quantize_block_fp4(x, 32, "e5m2", backend="torch")
    with pytest.raises(TypeError, match="BF16 or FP32"):
        quantize_block_fp4(torch.zeros(1, 32, dtype=torch.float16), 32, "e8m0", backend="torch")
    with pytest.raises(ValueError, match="contiguous"):
        quantize_block_fp4(torch.zeros(2, 64, dtype=torch.float32)[:, ::2], 32, "e8m0",
                           backend="torch")
    with pytest.raises(ValueError, match="no out tensors"):
        quantize_block_fp4(x, 32, "e8m0", inplace=True, backend="torch",
                           out=(torch.empty(1, 16, dtype=torch.uint8),
                                torch.empty(1, 1, dtype=torch.uint8)))
    with pytest.raises(TypeError, match="pair of uint8"):
        quantize_block_fp4(x, 32, "e8m0", backend="torch", out=(torch.empty(1, 16),))
    with pytest.raises(ValueError, match="distinct"):
        shared = torch.empty(1, 16, dtype=torch.uint8)
        quantize_block_fp4(x, 32, "e8m0", backend="torch", out=(shared, shared))
    with pytest.raises(ValueError, match="shape"):
        quantize_block_fp4(x, 32, "e8m0", backend="torch",
                           out=(torch.empty(1, 8, dtype=torch.uint8),
                                torch.empty(1, 1, dtype=torch.uint8)))
    with pytest.raises(ValueError, match="overlap|alias"):
        quantize_block_fp4(x, 32, "e8m0", backend="torch",
                           out=(x.view(torch.uint8).flatten()[:16].reshape(1, 16),
                                torch.empty(1, 1, dtype=torch.uint8)))
    with pytest.raises(ValueError, match="backend"):
        quantize_block_fp4(x, 32, "e8m0", backend="native")
    with pytest.raises(ValueError, match="requires an NPU"):
        quantize_block_fp4(x, 32, "e8m0", backend="triton")


# --- device cases ------------------------------------------------------------

@requires_npu
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_npu_kernel_matches_the_torch_reference_bit_for_bit(scale_format, block_size):
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    generator = torch.Generator().manual_seed(404 + block_size)
    x = _blocks(33, block_size, generator).to(torch.bfloat16)
    expected_packed, expected_scales = quantize_block_fp4(x, block_size, scale_format,
                                                          backend="torch")
    packed, scales = quantize_block_fp4(x.to(NPU_DEVICE), block_size, scale_format,
                                        backend="triton")
    assert torch.equal(packed.cpu(), expected_packed), "codes disagree"
    assert torch.equal(scales.cpu(), expected_scales), "scale bytes disagree"


@requires_npu
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_npu_inplace_keeps_the_sign_of_a_negative_zero_bit_exactly(scale_format):
    """The final BF16/FP32 inplace bits of a zero must carry the code's sign.

    On the real device this returned ``0`` where the reference returns ``0x8000`` for
    code 8, in both scale formats, while every non-zero value matched -- the
    arithmetic drops a zero's sign. The kernel now writes that sign into the result
    bits, so this compares the *patterns* rather than the values.
    """
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    rows = [[-0.0] * 32, [0.0] * 32, [-1.0] * 32, [1.0] * 32]
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.tensor(rows, dtype=torch.float32).to(dtype)
        expected = x.clone()
        quantize_block_fp4(expected, 32, scale_format, inplace=True, backend="torch")
        device_x = x.to(NPU_DEVICE)
        quantize_block_fp4(device_x, 32, scale_format, inplace=True, backend="triton")
        torch.npu.synchronize()
        result = device_x.cpu()
        # The integer view turns the sign bit into an ordinary negative number, which
        # is the same test in either dtype and cannot overflow a 16-bit mask.
        bits = result.view(torch.int16) if dtype == torch.bfloat16 else result.view(torch.int32)
        expected_bits = (expected.view(torch.int16) if dtype == torch.bfloat16
                         else expected.view(torch.int32))
        assert torch.equal(bits, expected_bits), (
            f"{scale_format}/{dtype}: inplace bits differ from the reference")
        assert bool((bits[0] < 0).all()), "the negative zero lost its sign"
        assert bool((bits[1] >= 0).all()), "a positive zero gained a sign"


@requires_npu
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_npu_inplace_zero_sign_fix_never_turns_a_nan_into_a_signed_zero(scale_format):
    """A non-finite block has zero codes, so the sign fix must exclude it by exponent.

    This is the trap in the fix: the NaN is carried by the scale, and a blind "zero
    code implies signed zero" rewrite would replace the NaN with ``+-0``.
    """
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    x = torch.tensor([[math.inf] + [1.0] * 31, [math.nan] * 32, [-0.0] * 32],
                     dtype=torch.float32).to(torch.bfloat16)
    device_x = x.to(NPU_DEVICE)
    quantize_block_fp4(device_x, 32, scale_format, inplace=True, backend="triton")
    torch.npu.synchronize()
    result = device_x.cpu()
    assert torch.isnan(result[0]).all() and torch.isnan(result[1]).all()
    assert (result[2].view(torch.int16) == 0x8000).all(), "the real zero lost its sign"


@requires_npu
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_npu_nonfinite_inplace_writes_nan_like_the_reference(scale_format):
    """The inplace kernel carries the NaN out through the scale value.

    Zero codes times a *finite* scale would write a plausible zero block, so this is
    the case that separates "documented extension" from "silently plausible".
    """
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    x = torch.tensor([[math.inf] + [1.0] * 31, [2.0] * 32, [math.nan] * 32],
                     dtype=torch.float32).to(torch.bfloat16)
    device_x = x.to(NPU_DEVICE)
    quantize_block_fp4(device_x, 32, scale_format, inplace=True, backend="triton")
    torch.npu.synchronize()
    result = device_x.cpu()
    assert torch.isnan(result[0]).all() and torch.isnan(result[2]).all()
    assert torch.isfinite(result[1]).all()


@requires_npu
def test_npu_kernel_saturates_the_e4m3_scale_where_the_ratio_overflows():
    """The compiled kernel must clamp the encode input, not emit the NaN byte.

    Until this pinch point was found the device kernel produced ``0x7F`` for a ratio
    in ``[448, 512)`` while the reference produced the saturated ``0x7E``, so this is
    the case that separates them.
    """
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    rows = []
    for amax in (2688.0, 2784.0, 2816.0, 2880.0, 3000.0, 10000.0):
        rows.append([amax, 448.0, 224.0, 0.01171875, -amax, -1.0] + [0.0] * 26)
    x = torch.tensor(rows, dtype=torch.float32).to(torch.bfloat16)
    expected_packed, expected_scales = quantize_block_fp4(x, 32, "e4m3", backend="torch")
    assert expected_scales.reshape(-1).tolist() == [0x7E] * len(rows)
    packed, scales = quantize_block_fp4(x.to(NPU_DEVICE), 32, "e4m3", backend="triton")
    torch.npu.synchronize()
    assert torch.equal(scales.cpu(), expected_scales), "the NPU scale byte diverged"
    assert torch.equal(packed.cpu(), expected_packed), "the NPU codes diverged"


@requires_npu
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_npu_inplace_matches_the_reference_dequantization(scale_format):
    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    generator = torch.Generator().manual_seed(505)
    x = _blocks(7, 32, generator).to(torch.bfloat16)
    expected = x.clone()
    quantize_block_fp4(expected, 32, scale_format, inplace=True, backend="torch")
    device_x = x.to(NPU_DEVICE)
    result = quantize_block_fp4(device_x, 32, scale_format, inplace=True, backend="triton")
    torch.npu.synchronize()
    assert result is device_x
    assert torch.equal(device_x.cpu().view(torch.int16), expected.view(torch.int16))


@requires_npu
def test_npu_kernel_follows_the_current_stream_and_restores_the_device():
    import torch_npu
    import triton

    if not supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    torch.npu.set_device(NPU_DEVICE)
    generator = torch.Generator().manual_seed(606)
    x_host = _blocks(4, 32, generator).to(torch.bfloat16)
    expected_packed, expected_scales = quantize_block_fp4(x_host, 32, "e4m3", backend="torch")
    stream = torch.npu.Stream(device=NPU_DEVICE)
    with torch.npu.stream(stream):
        before = torch.npu.current_device()
        x = x_host.to(NPU_DEVICE)
        target = triton.runtime.driver.active.get_current_target()
        assert target.backend.lower() in {"npu", "ascend"}, target.backend
        current = triton.runtime.driver.active.get_current_stream(NPU_DEVICE.index)
        expected_stream = torch_npu._C._npu_getCurrentRawStream(NPU_DEVICE.index)
        assert current == expected_stream, "Triton and Torch launch on different streams"
        packed, scales = quantize_block_fp4(x, 32, "e4m3", backend="triton")
        consumed_packed = packed.to(torch.int16) + torch.zeros((), dtype=torch.int16,
                                                              device=NPU_DEVICE)
        consumed_scales = scales.to(torch.int16) + torch.zeros((), dtype=torch.int16,
                                                              device=NPU_DEVICE)
        assert torch.npu.current_device() == before, "the quantizer changed the current device"
    stream.synchronize()
    assert torch.equal(consumed_packed.cpu(), expected_packed.to(torch.int16))
    assert torch.equal(consumed_scales.cpu(), expected_scales.to(torch.int16))


@requires_npu
def test_npu_torch_backend_is_refused_before_an_unsupported_cast():
    x = torch.ones(1, 32, dtype=torch.bfloat16, device=NPU_DEVICE)
    with pytest.raises(ValueError, match="CPU/CUDA-only"):
        quantize_block_fp4(x, 32, "e4m3", backend="torch")