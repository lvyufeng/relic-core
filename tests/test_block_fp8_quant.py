"""Operator tests for the original block-32 FP8 activation quantizer.

The CPU tests need only torch: the ``torch`` backend is the same-device FP32
reference ending in the native FP8 cast, and it is checked here against an
independent oracle written in exact rational arithmetic rather than against
another cast of the same expression. The NPU tests are opt-in and run only when
``RELIC_BLOCK_FP8_QUANT_TEST_DEVICE=npu:0`` asks for a real device; they compare
the fused Triton kernel against the CPU reference bit for bit and are the only
place that can claim anything about hardware.

Nothing here claims a GEMM, an expert forward or model parity. This operator
produces the original E4M3 code bytes and E8M0 scale bytes for activations; how a
matmul consumes them is a separate contract.
"""
from __future__ import annotations

import bisect
import math
import os
from fractions import Fraction
import struct

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import block_fp8_quant  # noqa: E402
from relic_core.kernels.block_fp8_quant import quantize_block_fp8  # noqa: E402

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("the Torch reference needs float8_e4m3fn", allow_module_level=True)

DEVICE = os.environ.get("RELIC_BLOCK_FP8_QUANT_TEST_DEVICE", "")
NPU_DEVICE = torch.device(DEVICE) if DEVICE.startswith("npu") else None
requires_npu = pytest.mark.skipif(
    NPU_DEVICE is None,
    reason="requires RELIC_BLOCK_FP8_QUANT_TEST_DEVICE=npu:0 on a real Ascend host")

BLOCK = 32
FP8_MAX = 448.0
FLOOR = struct.unpack("<f", struct.pack("<f", 1e-4))[0]
INV_FP8_MAX = struct.unpack("<f", struct.pack("<f", 1.0 / 448.0))[0]
DTYPES = (torch.float32, torch.bfloat16)


# --- independent exact oracle -------------------------------------------------
# Values are exact ``Fraction``s and the E4M3 encode is a nearest-neighbour search
# over the enumerated representable magnitudes, so the arithmetic under test is
# never reused by the oracle.

def _f32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    return exponent


def _round_half_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _ceil_log2(value: Fraction) -> int:
    exponent = _floor_log2(value)
    return exponent if Fraction(2) ** exponent == value else exponent + 1


def _f32_round(value: Fraction) -> int:
    """Round an exact rational to float32 (ties to even) and return its bits."""
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    if exponent < -126:
        mantissa = _round_half_even(magnitude * Fraction(2**149))
        if mantissa == 0:
            return sign
        if mantissa >= 2**23:
            return sign | (1 << 23)
        return sign | mantissa
    mantissa = _round_half_even(magnitude * Fraction(2) ** (23 - exponent))
    if mantissa == 2**24:
        exponent += 1
        mantissa = 2**23
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (mantissa - 2**23)


def _e4m3_magnitude(code: int) -> Fraction:
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 0:
        return Fraction(mantissa, 512)
    return Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)


_E4M3_TABLE = tuple((_e4m3_magnitude(code), code) for code in range(127))
_E4M3_VALUES = tuple(value for value, _ in _E4M3_TABLE)


def _e4m3_nearest(magnitude: Fraction) -> int:
    if magnitude <= 0:
        return 0
    index = bisect.bisect_left(_E4M3_VALUES, magnitude)
    if index == len(_E4M3_VALUES):
        return 0x7E
    if _E4M3_VALUES[index] == magnitude:
        return _E4M3_TABLE[index][1]
    lower, upper = _E4M3_TABLE[index - 1], _E4M3_TABLE[index]
    below, above = magnitude - lower[0], upper[0] - magnitude
    if below < above:
        return lower[1]
    if above < below:
        return upper[1]
    return lower[1] if (lower[1] & 7) % 2 == 0 else upper[1]


def exact_block_quantize(row: list[float]) -> tuple[list[int], int]:
    """Exact reference for one 32-element block, from the float32 input values."""
    assert len(row) == BLOCK
    if any(not math.isfinite(value) for value in row):
        # The documented finite-contract extension, not a claim about the
        # released kernel: a block with infinity or NaN becomes the NaN pair.
        return [0x7F] * BLOCK, 255
    magnitude = Fraction(max((abs(value) for value in row), default=0.0))
    # The released kernel floors amax at 1e-4 in FP32 before the reciprocal.
    floored = max(magnitude, Fraction(FLOOR))
    scaled = Fraction(_f32_value(_f32_round(floored * Fraction(INV_FP8_MAX))))
    exponent = _ceil_log2(scaled)
    codes = []
    for value in row:
        # Fraction(-0.0) == Fraction(0) and would drop the sign of a zero, so the
        # sign comes from the input float's sign bit. The scale is a positive
        # power of two, so the product's sign is always the input's sign: this is
        # what makes a normalized underflow keep its signed zero (code 0x80).
        negative = math.copysign(1.0, value) < 0
        product = _f32_round(Fraction(abs(value)) * Fraction(2) ** -exponent)
        clamped = min(max(_f32_value(product), -FP8_MAX), FP8_MAX)
        sign = 0x80 if negative else 0
        codes.append(sign | _e4m3_nearest(Fraction(abs(clamped))))
    return codes, exponent + 128 - 1


def exact_quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reference over an FP32/BF16 tensor, as (codes, scales) uint8."""
    width = values.shape[-1]
    flat = values.reshape(-1, width).to(torch.float32).tolist()
    codes = torch.empty(tuple(values.shape), dtype=torch.uint8)
    scales = torch.empty((*values.shape[:-1], width // BLOCK), dtype=torch.uint8)
    flat_codes = codes.view(-1)
    flat_scales = scales.view(-1)
    for row_index, row in enumerate(flat):
        for start in range(0, width, BLOCK):
            block_codes, scale_byte = exact_block_quantize(row[start:start + BLOCK])
            offset = row_index * width + start
            flat_codes[offset:offset + BLOCK] = torch.tensor(block_codes, dtype=torch.uint8)
            flat_scales[row_index * (width // BLOCK) + start // BLOCK] = scale_byte
    return codes, scales


# --- test data ----------------------------------------------------------------

def _spanning_blocks(rows: int, width: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Random blocks whose per-block magnitude walks a wide power-of-two range."""
    generator = torch.Generator().manual_seed(seed)
    exponents = (-140, -60, -22, -9, -3, 0, 1, 6, 20, 40, 100, 120)
    blocks = []
    for index in range(rows * (width // BLOCK)):
        scale = 2.0 ** exponents[index % len(exponents)]
        blocks.append(torch.randn(BLOCK, generator=generator) * scale)
    return torch.cat(blocks).reshape(rows, width).to(dtype)


def _special_blocks() -> torch.Tensor:
    """Blocks covering zeros, signed zeros, non-finite values and RNE ties."""
    huge, tiny = 1e30, 1e-30
    rows = [
        [0.0] * BLOCK,                                   # all zero -> floored scale
        [-0.0] * BLOCK,                                  # signed zeros keep 0x80
        [float("inf")] + [0.0] * (BLOCK - 1),            # non-finite block
        [float("nan")] + [1.0] * (BLOCK - 1),            # non-finite block
        [3.4028234663852886e38, 1.0] + [0.0] * (BLOCK - 2),   # maximum finite amax
        [huge] + [-tiny] * (BLOCK - 1),                  # normalised underflow to -0
        [448.0, -448.0] + [1.0] * (BLOCK - 2),           # exact clamps
        [2.0**-10] + [0.0] * (BLOCK - 1),                # tie at the first midpoint
        [1.5 * 2.0**-9] + [0.0] * (BLOCK - 1),           # tie between codes 1 and 2
        [1.0] * BLOCK,                                   # all equal
        [2.0**-22, 1e-4, 0.0] + [0.0] * (BLOCK - 3),     # at and below the floor
        [1e-45, 0.0] + [0.0] * (BLOCK - 2),              # FP32 subnormal input
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _cases() -> list[tuple[str, torch.Tensor]]:
    return [
        ("single_block", torch.randn(1, BLOCK)),
        ("cells", _spanning_blocks(6, 64, torch.float32, 11)),
        ("wide_matrix", _spanning_blocks(4, 320, torch.float32, 12)),
        ("three_axes", _spanning_blocks(2 * 3, 128, torch.float32, 13).reshape(2, 3, 128)),
        ("single_row", _spanning_blocks(1, 128, torch.float32, 14)),
        ("special", _special_blocks()),
    ]


# --- CPU reference tests ------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,values", _cases(), ids=[case[0] for case in _cases()])
def test_cpu_backend_matches_the_exact_rational_oracle(name, values, dtype):
    x = values.to(dtype).contiguous()
    codes, scales = quantize_block_fp8(x, backend="torch")
    expected_codes, expected_scales = exact_quantize(x)
    assert codes.shape == x.shape
    assert scales.shape == (*x.shape[:-1], x.shape[-1] // BLOCK)
    assert torch.equal(codes, expected_codes), name
    assert torch.equal(scales, expected_scales), name


def test_nonfinite_blocks_become_the_documented_nan_pair():
    x = torch.tensor([[float("inf")] + [0.0] * 31, [float("nan")] + [1.0] * 31,
                      [1.0] * 32, [0.0] * 32], dtype=torch.float32)
    codes, scales = quantize_block_fp8(x, backend="torch")
    assert torch.all(codes[0] == 0x7F) and torch.all(codes[1] == 0x7F)
    assert scales[0].item() == 255 and scales[1].item() == 255
    # A neighbouring finite block is not affected by its neighbour's non-finite data.
    # amax 1.0 gives scale 2**-8 (byte 119) and code 0x78 for 256; the floored
    # all-zero block gives the floor scale 2**-22 (byte 105) and positive zeros.
    assert scales[2].item() == 127 - 8 and scales[3].item() == 127 - 22
    assert torch.all(codes[2] == 0x78)
    assert torch.all(codes[3] == 0x00)


def test_scale_is_the_released_ceil_power_of_two():
    x = _spanning_blocks(8, 64, torch.float32, 21)
    _, scales = quantize_block_fp8(x, backend="torch")
    flat = x.reshape(-1, BLOCK)
    flat_scales = scales.view(-1)
    for row_index in range(flat.shape[0]):
        amax = flat[row_index].abs().max().item()
        exponent = int(flat_scales[row_index]) - 127
        scale = math.ldexp(1.0, exponent)
        if amax < FLOOR:
            assert exponent == -22
            continue
        # ceil(log2(amax / 448)) leaves the block maximum in (224, 448] after
        # scaling, up to one FP32 rounding of the reciprocal product: the scale is
        # chosen from the rounded product, so the exact ratio can miss either edge
        # by at most half an ulp of that product.
        epsilon = 2.0**-24
        assert amax / scale <= FP8_MAX * (1.0 + epsilon)
        assert amax / scale >= FP8_MAX / 2 * (1.0 - epsilon)


def test_signed_negative_zero_survives_in_the_code_byte():
    x = torch.tensor([[-0.0] * BLOCK, [0.0] * BLOCK, [1.0] + [-0.0] * (BLOCK - 1)],
                     dtype=torch.float32)
    codes, scales = quantize_block_fp8(x, backend="torch")
    assert torch.all(codes[0] == 0x80)
    assert torch.all(codes[1] == 0x00)
    assert codes[2, 0].item() == 0x78 and torch.all(codes[2, 1:] == 0x80)
    assert torch.all(scales[0] == 127 - 22) and torch.all(scales[1] == 127 - 22)


def test_input_is_never_modified_and_results_are_deterministic():
    x = _spanning_blocks(4, 64, torch.float32, 31)
    before = x.clone()
    first = quantize_block_fp8(x, backend="torch")
    second = quantize_block_fp8(x, backend="torch")
    assert torch.equal(x, before)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_auto_off_npu_is_the_torch_reference():
    x = _spanning_blocks(3, 96, torch.bfloat16, 41)
    auto_codes, auto_scales = quantize_block_fp8(x)
    torch_codes, torch_scales = quantize_block_fp8(x, backend="torch")
    assert torch.equal(auto_codes, torch_codes) and torch.equal(auto_scales, torch_scales)


def test_out_pair_is_validated_and_returned_by_identity():
    x = _spanning_blocks(2, 64, torch.float32, 51)
    codes = torch.empty(x.shape, dtype=torch.uint8)
    scales = torch.empty((2, 2), dtype=torch.uint8)
    actual_codes, actual_scales = quantize_block_fp8(x, backend="torch", out=(codes, scales))
    assert actual_codes is codes and actual_scales is scales
    expected_codes, expected_scales = quantize_block_fp8(x, backend="torch")
    assert torch.equal(codes, expected_codes) and torch.equal(scales, expected_scales)


def test_out_rejects_the_wrong_contract():
    x = _spanning_blocks(2, 64, torch.float32, 61)
    codes = torch.empty(x.shape, dtype=torch.uint8)
    scales = torch.empty((2, 2), dtype=torch.uint8)
    with pytest.raises(TypeError):
        quantize_block_fp8(x, backend="torch", out=codes)
    with pytest.raises(TypeError):
        quantize_block_fp8(x, backend="torch", out=(codes.to(torch.int8), scales))
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="torch", out=(codes[:, :32], scales))
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="torch", out=(codes, scales[:, :1]))
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="torch", out=(codes, codes))
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="torch", out=(codes, scales.t()))


def test_out_rejects_storage_shared_with_the_input():
    x = _spanning_blocks(2, 64, torch.float32, 71)
    codes = torch.empty(x.shape, dtype=torch.uint8)
    scales = torch.empty((2, 2), dtype=torch.uint8)
    # A contiguous uint8 view of the input's own bytes: the ranges overlap, so a
    # conservative storage check must refuse it. A disjoint view would be equally
    # unsafe to accept, which is why the check is not a pointer-range test.
    aliased = x.view(torch.uint8).view(-1)[:x.numel()].view(x.shape)
    assert aliased.is_contiguous() and aliased.data_ptr() == x.data_ptr()
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="torch", out=(aliased, scales))


def test_input_contract_is_enforced():
    good = torch.randn(2, 64)
    with pytest.raises(TypeError):
        quantize_block_fp8(good.to(torch.float16))
    with pytest.raises(TypeError):
        quantize_block_fp8(good.to(torch.int32))
    with pytest.raises(ValueError):
        quantize_block_fp8(torch.tensor(1.0))
    with pytest.raises(ValueError):
        quantize_block_fp8(torch.randn(2, 48))
    with pytest.raises(ValueError):
        quantize_block_fp8(torch.randn(2, 0))
    with pytest.raises(ValueError):
        quantize_block_fp8(torch.randn(2, 64).t())
    with pytest.raises(TypeError):
        quantize_block_fp8([1.0] * 64)
    with pytest.raises(TypeError):
        quantize_block_fp8(good.view(torch.uint16))


def test_backend_names_are_validated_and_triton_needs_an_npu():
    x = torch.randn(1, 32)
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="cuda")
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend=None)
    with pytest.raises(ValueError):
        quantize_block_fp8(x, backend="triton")     # a CPU tensor never selects a kernel


def test_empty_leading_dimension_returns_empty_outputs():
    for shape in ((0, 32), (3, 0, 64)):
        x = torch.empty(shape, dtype=torch.float32)
        codes, scales = quantize_block_fp8(x, backend="torch")
        assert codes.shape == shape
        assert scales.shape == (*shape[:-1], shape[-1] // BLOCK)
        assert codes.numel() == 0 and scales.numel() == 0


def test_supports_triton_ascend_is_a_pure_probe():
    # `False` on a CPU host, but the call must never require a device or raise.
    assert isinstance(block_fp8_quant.supports_triton_ascend(), bool)


# --- opt-in real Ascend device ------------------------------------------------

def _to_device(values: torch.Tensor) -> torch.Tensor:
    import torch_npu  # noqa: F401 - installing the torch.npu namespace
    torch.npu.set_device(NPU_DEVICE)
    return values.to(NPU_DEVICE)


@requires_npu
@pytest.mark.parametrize("dtype", DTYPES)
def test_npu_fused_kernel_matches_the_cpu_reference_bit_for_bit(dtype):
    if not block_fp8_quant.supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    cases = _cases()
    special = _special_blocks()
    for name, values in cases:
        x = values.to(dtype).contiguous()
        expected_codes, expected_scales = quantize_block_fp8(x, backend="torch")
        actual_codes, actual_scales = quantize_block_fp8(_to_device(x), backend="triton")
        assert torch.equal(actual_codes.cpu(), expected_codes), name
        assert torch.equal(actual_scales.cpu(), expected_scales), name
    for start in range(0, special.shape[0], 2):
        x = special[start:start + 2].to(dtype).contiguous()
        expected_codes, expected_scales = quantize_block_fp8(x, backend="torch")
        actual_codes, actual_scales = quantize_block_fp8(_to_device(x), backend="auto")
        assert torch.equal(actual_codes.cpu(), expected_codes), start
        assert torch.equal(actual_scales.cpu(), expected_scales), start


@requires_npu
def test_npu_kernel_follows_the_current_stream_and_restores_the_device():
    import torch_npu
    import triton

    if not block_fp8_quant.supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    torch.npu.set_device(NPU_DEVICE)
    x_host = _spanning_blocks(4, 128, torch.bfloat16, 81)
    expected_codes, expected_scales = quantize_block_fp8(x_host, backend="torch")
    stream = torch.npu.Stream(device=NPU_DEVICE)
    with torch.npu.stream(stream):
        before = torch.npu.current_device()
        x = x_host.to(NPU_DEVICE)
        target = triton.runtime.driver.active.get_current_target()
        assert target.backend.lower() in {"npu", "ascend"}, target.backend
        current = triton.runtime.driver.active.get_current_stream(NPU_DEVICE.index)
        expected_stream = torch_npu._C._npu_getCurrentRawStream(NPU_DEVICE.index)
        assert current == expected_stream, "Triton and Torch launch on different streams"
        codes, scales = quantize_block_fp8(x, backend="triton")
        # Consume both outputs with the next Torch operation before any explicit
        # synchronisation: a backend on another stream would race these reads.
        consumed_codes = codes.to(torch.int32) + torch.zeros((), dtype=torch.int32, device=NPU_DEVICE)
        consumed_scales = scales.to(torch.int32) + torch.zeros((), dtype=torch.int32,
                                                              device=NPU_DEVICE)
        assert torch.npu.current_device() == before, "the quantizer changed the current device"
    stream.synchronize()
    assert torch.equal(consumed_codes.cpu(), expected_codes.to(torch.int32))
    assert torch.equal(consumed_scales.cpu(), expected_scales.to(torch.int32))


@requires_npu
def test_npu_explicit_torch_backend_is_refused_before_an_unsupported_cast():
    x = _to_device(torch.ones((1, 32), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CPU/CUDA-only"):
        quantize_block_fp8(x, backend="torch")


@requires_npu
def test_npu_out_pair_dispatch_and_large_index_path():
    if not block_fp8_quant.supports_triton_ascend():
        pytest.fail("an explicitly requested NPU run requires the installed Triton Ascend backend")
    x = _to_device(_spanning_blocks(2, 64, torch.float32, 91))
    codes = torch.empty(x.shape, dtype=torch.uint8, device=NPU_DEVICE)
    scales = torch.empty((2, 2), dtype=torch.uint8, device=NPU_DEVICE)
    actual_codes, actual_scales = quantize_block_fp8(x, backend="triton", out=(codes, scales))
    assert actual_codes is codes and actual_scales is scales
    expected_codes, expected_scales = quantize_block_fp8(x.cpu(), backend="torch")
    assert torch.equal(codes.cpu(), expected_codes)
    assert torch.equal(scales.cpu(), expected_scales)
    # A single row wider than one program tile exercises the masked tail without
    # leaving the int32 index range; the int64 selection itself is unit-tested in
    # test_ascend_unpack_launch.py's padded-boundary pattern.
    wide = _to_device(_spanning_blocks(1, 32 * (block_fp8_quant.BLOCKS * 3 + 5), torch.float32, 92))
    wide_codes, wide_scales = quantize_block_fp8(wide, backend="triton")
    expected_wide = quantize_block_fp8(wide.cpu(), backend="torch")
    assert torch.equal(wide_codes.cpu(), expected_wide[0])
    assert torch.equal(wide_scales.cpu(), expected_wide[1])