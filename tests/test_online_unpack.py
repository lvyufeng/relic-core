"""Independent byte-level contracts for online floating-format unpacking.

The default run is CPU-only. A lease-owning caller may opt in to another device
with RELIC_ONLINE_UNPACK_TEST_DEVICE=npu:0 (or cuda:0), and select the backend
with RELIC_ONLINE_UNPACK_TEST_BACKEND=torch/auto/triton. Tests may copy results
back to the CPU for comparison; the operators themselves must not copy payloads
there or invoke a float8 device cast. No environment or lease is created here.

Expected encodings are derived with Python math/struct, not from the operator's
lookup tables or a float8 Torch conversion. BF16 comparisons include signed-zero
bits, finite/subnormal rounding and infinities; NaNs are checked by classification.
"""
from __future__ import annotations

import math
import os
import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from relic_core.kernels import unpack  # noqa: E402


@pytest.fixture
def device():
    requested = os.environ.get("RELIC_ONLINE_UNPACK_TEST_DEVICE", "cpu")
    if requested.startswith("npu"):
        pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("explicitly requested NPU is unavailable; this is not a pass")
    elif requested.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("explicitly requested CUDA device is unavailable; this is not a pass")
    return torch.device(requested)


@pytest.fixture
def backend():
    return os.environ.get("RELIC_ONLINE_UNPACK_TEST_BACKEND", "torch")


def _fp4_x2(code: int) -> int:
    magnitudes = (0, 1, 2, 3, 4, 6, 8, 12)
    magnitude = magnitudes[code & 7]
    return -magnitude if code & 8 else magnitude


def _e4m3(byte: int) -> float:
    negative, exponent, fraction = bool(byte & 128), (byte >> 3) & 15, byte & 7
    if exponent == 15 and fraction == 7:
        value = float("nan")
    elif exponent == 0:
        value = fraction * 2.0 ** -9
    else:
        value = (8 + fraction) * 2.0 ** (exponent - 10)
    return -value if negative else value


def _e8m0(byte: int, *, half: bool = False) -> float:
    return float("nan") if byte == 255 else 2.0 ** (byte - 127 - int(half))


def _f32_bits(value: float) -> int:
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        packed = struct.pack("<f", math.copysign(float("inf"), value))
    return struct.unpack("<I", packed)[0]


def _bf16_bits(value: float) -> int:
    bits = _f32_bits(value)
    if bits & 0x7F800000 == 0x7F800000:
        return (bits >> 16) | (0x40 if bits & 0x7FFFFF else 0)
    # Round-to-nearest-even from FP32, including subnormal carries and negative zero.
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _oracle_fp8(raw: torch.Tensor, scale_raw: torch.Tensor | None = None,
                 block_shape=(32, 32)) -> tuple[list[int], list[bool]]:
    codes = raw.cpu().reshape(-1).tolist()
    scale_codes = None if scale_raw is None else scale_raw.cpu().reshape(-1).tolist()
    if scale_codes is not None:
        bm, bk = block_shape
        rows, columns = raw.shape[-2:]
        grid_rows, grid_columns = (rows + bm - 1) // bm, (columns + bk - 1) // bk
    bits, nan = [], []
    for index, byte in enumerate(codes):
        value = _e4m3(byte)
        if scale_codes is not None:
            batch = index // (rows * columns)
            row, column = (index // columns) % rows, index % columns
            offset = (batch * grid_rows + row // bm) * grid_columns + column // bk
            value *= _e8m0(scale_codes[offset])
        bits.append(_bf16_bits(value))
        nan.append(math.isnan(value))
    return bits, nan


def _assert_bf16(actual: torch.Tensor, expected_bits: list[int], expected_nan: list[bool]):
    assert actual.dtype == torch.bfloat16
    observed = [int(value) & 0xFFFF for value in actual.view(torch.int16).cpu().reshape(-1).tolist()]
    assert len(observed) == len(expected_bits)
    for index, (got, want, is_nan) in enumerate(zip(observed, expected_bits, expected_nan)):
        if is_nan:
            assert got & 0x7F80 == 0x7F80 and got & 0x7F, f"expected NaN at {index}, got {got:#06x}"
        else:
            assert got == want, f"BF16 bits differ at {index}: {got:#06x} != {want:#06x}"


def _storage_view(raw: torch.Tensor, name: str) -> torch.Tensor:
    if name == "uint8":
        return raw
    if name == "int8":
        return raw.view(torch.int8)
    dtype = getattr(torch, name, None)
    if dtype is None:
        pytest.skip(f"Torch does not expose optional packed dtype {name}; this is not a pass")
    return raw.view(dtype)


@pytest.mark.parametrize("storage", ["uint8", "int8", "float4_e2m1fn_x2"])
def test_fp4_all_256_bytes_are_exact_low_first_int8_x2(device, backend, storage):
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(4, 64).to(device)
    source = _storage_view(raw, storage)
    before = raw.cpu().clone()
    observed = unpack.unpack_fp4_to_int8(source, backend=backend)
    expected = [_fp4_x2(code) for byte in range(256) for code in (byte & 15, byte >> 4)]
    assert observed.device == raw.device and observed.shape == (4, 128)
    assert observed.dtype == torch.int8 and observed.is_contiguous()
    assert observed.cpu().reshape(-1).tolist() == expected
    assert torch.equal(raw.cpu(), before)


def test_fp4_zero_codes_have_numeric_not_sign_bit_preservation(device, backend):
    raw = torch.tensor([0x80, 0x08, 0x88, 0x00], dtype=torch.uint8, device=device)
    before = raw.cpu().clone()
    observed = unpack.unpack_fp4_to_int8(raw, backend=backend)
    assert observed.cpu().tolist() == [0] * 8
    assert torch.equal(raw.cpu(), before)  # The unchanged source still distinguishes 0 and 8.


@pytest.mark.parametrize("storage", ["uint8", "int8", "float8_e4m3fn"])
def test_fp8_all_256_encodings_match_independent_bf16_bits(device, backend, storage):
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).to(device)
    source = _storage_view(raw, storage)
    expected = _oracle_fp8(raw)
    before = raw.cpu().clone()
    observed = unpack.unpack_fp8_to_bf16(source, backend=backend)
    assert observed.device == source.device and observed.shape == source.shape
    assert observed.is_contiguous()
    _assert_bf16(observed, *expected)
    assert torch.equal(raw.cpu(), before)
    assert int(observed.view(torch.int16)[0].cpu()) == 0
    assert int(observed.view(torch.int16)[128].cpu()) == -32768


@pytest.mark.parametrize("storage", ["uint8", "int8", "float8_e8m0fnu"])
def test_all_e8m0_half_scales_include_subnormals_and_nan(device, backend, storage):
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(8, 32).to(device)
    source = _storage_view(raw, storage)
    before = raw.cpu().clone()
    observed = unpack.fp4_int8_scale(source, backend=backend)
    assert observed.dtype == torch.float32 and observed.shape == raw.shape and observed.device == raw.device
    bits = [int(value) & 0xFFFFFFFF for value in observed.view(torch.int32).cpu().reshape(-1).tolist()]
    for byte in range(255):
        assert bits[byte] == _f32_bits(_e8m0(byte, half=True)), byte
    assert math.isnan(float(observed.reshape(-1)[255].cpu()))
    assert bits[0] == 0x00200000 and bits[1] == 0x00400000  # Not flushed to zero.
    assert torch.equal(raw.cpu(), before)


@pytest.mark.parametrize("byte", [0, 1, 127, 254, 255])
def test_scalar_fp4_half_scale(device, backend, byte):
    source = torch.tensor(byte, dtype=torch.uint8, device=device)
    result = unpack.fp4_int8_scale(source, backend=backend)
    assert result.shape == torch.Size([])
    if byte == 255:
        assert math.isnan(float(result.cpu()))
    else:
        assert float(result.cpu()) == _e8m0(byte, half=True)


def test_fp4_original_scale_compensation_preserves_values_without_row_requantization(device, backend):
    raw = torch.tensor([[0x21, 0xF7, 0x98, 0xBA] * 8, [0x43, 0x65, 0xDC, 0xFE] * 8], dtype=torch.uint8, device=device)
    scales = torch.tensor([[126, 129], [127, 128]], dtype=torch.uint8, device=device)
    before_raw, before_scales = raw.cpu().clone(), scales.cpu().clone()
    integers = unpack.unpack_fp4_to_int8(raw, backend=backend)
    half_scale = unpack.fp4_int8_scale(scales, backend=backend)
    reconstructed = integers.float() * half_scale.repeat_interleave(32, dim=-1)
    expected = []
    for row, packed_row in enumerate(before_raw.tolist()):
        expected_row = []
        for index, byte in enumerate(packed_row):
            for code in (byte & 15, byte >> 4):
                k = len(expected_row)
                expected_row.append((_fp4_x2(code) / 2) * _e8m0(int(before_scales[row, k // 32])))
        expected.append(expected_row)
    assert torch.equal(reconstructed.cpu(), torch.tensor(expected, dtype=torch.float32))
    assert half_scale.shape == scales.shape  # The K-block grid was not collapsed to one row scale.
    assert torch.equal(raw.cpu(), before_raw) and torch.equal(scales.cpu(), before_scales)


@pytest.mark.parametrize("block_shape", [(32, 32), (1, 32), (2, 4)])
def test_fp8_ragged_batched_scale_grids_and_extreme_scaling(device, backend, block_shape):
    shape = (2, 3, 5, 9)
    raw = (torch.arange(math.prod(shape), dtype=torch.int64) % 256).to(torch.uint8).reshape(shape).to(device)
    bm, bk = block_shape
    scale_shape = (*shape[:-2], (shape[-2] + bm - 1) // bm, (shape[-1] + bk - 1) // bk)
    values = [0, 127, 254, 255, 1, 128, 126, 100, 160]
    scales = torch.tensor([values[i % len(values)] for i in range(math.prod(scale_shape))], dtype=torch.uint8).reshape(scale_shape).to(device)
    expected = _oracle_fp8(raw, scales, block_shape)
    before_raw, before_scales = raw.cpu().clone(), scales.cpu().clone()
    result = unpack.unpack_fp8_to_bf16(raw, scales, block_shape=block_shape, backend=backend)
    _assert_bf16(result, *expected)
    assert result.shape == raw.shape and result.device == raw.device
    assert torch.equal(raw.cpu(), before_raw) and torch.equal(scales.cpu(), before_scales)


def test_fp8_row_scales_are_not_square_block_scales(device, backend):
    raw = torch.full((4, 64), 0x38, dtype=torch.uint8, device=device)  # E4M3 1.0.
    row_scales = torch.tensor([[127, 128], [128, 129], [129, 130], [130, 131]], dtype=torch.uint8, device=device)
    square_scales = torch.tensor([[127, 128]], dtype=torch.uint8, device=device)
    row = unpack.unpack_fp8_to_bf16(raw, row_scales, block_shape=(1, 32), backend=backend)
    square = unpack.unpack_fp8_to_bf16(raw, square_scales, backend=backend)
    _assert_bf16(row, *_oracle_fp8(raw, row_scales, (1, 32)))
    _assert_bf16(square, *_oracle_fp8(raw, square_scales))
    assert not torch.equal(row.cpu(), square.cpu())
    with pytest.raises(ValueError, match="scale must have shape"):
        unpack.unpack_fp8_to_bf16(raw, row_scales, backend=backend)


def test_fp8_all_code_scale_pairs_match_fp32_then_bf16_rounding(device, backend):
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    raw = codes[:, None].repeat(1, 256).contiguous().to(device)
    scales = codes[None, :].repeat(256, 1).contiguous().to(device)
    expected = _oracle_fp8(raw, scales, (1, 1))
    result = unpack.unpack_fp8_to_bf16(raw, scales, block_shape=(1, 1), backend=backend)
    _assert_bf16(result, *expected)


def test_fp8_scale_edges_round_subnormals_overflow_and_nan(device, backend):
    raw = torch.tensor([[0x00, 0x80, 0x01, 0x08, 0x30, 0x38, 0x7E, 0xFE, 0x7F, 0xFF]], dtype=torch.uint8, device=device)
    for byte in (0, 1, 126, 127, 128, 254, 255):
        scale = torch.full(raw.shape, byte, dtype=torch.uint8, device=device)
        result = unpack.unpack_fp8_to_bf16(raw, scale, block_shape=(1, 1), backend=backend)
        _assert_bf16(result, *_oracle_fp8(raw, scale, (1, 1)))


@pytest.mark.parametrize("kind", ["fp4", "fp8", "scale"])
def test_empty_inputs_keep_shape_dtype_and_device(device, backend, kind):
    raw = torch.empty((2, 0), dtype=torch.uint8, device=device)
    if kind == "fp4":
        result, dtype = unpack.unpack_fp4_to_int8(raw, backend=backend), torch.int8
    elif kind == "fp8":
        scales = torch.empty((1, 0), dtype=torch.uint8, device=device)
        result, dtype = unpack.unpack_fp8_to_bf16(raw, scales, backend=backend), torch.bfloat16
    else:
        result, dtype = unpack.fp4_int8_scale(raw, backend=backend), torch.float32
    assert result.shape == (2, 0) and result.numel() == 0 and result.dtype == dtype and result.device == device


def test_frontend_torch_chunks_cross_matrix_and_batch_boundaries(device, monkeypatch):
    monkeypatch.setattr(unpack, "_TORCH_CHUNK_ELEMENTS", 7)
    raw = torch.arange(90, dtype=torch.uint8).reshape(2, 5, 9).to(device)
    scales = torch.tensor([127, 128, 126, 129] * 5, dtype=torch.uint8).reshape(2, 5, 2).to(device)
    result = unpack.unpack_fp8_to_bf16(raw, scales, block_shape=(1, 5), backend="torch")
    _assert_bf16(result, *_oracle_fp8(raw, scales, (1, 5)))
    packed = raw.reshape(3, 30)
    actual = unpack.unpack_fp4_to_int8(packed, backend="torch")
    expected = [_fp4_x2(code) for byte in raw.cpu().reshape(-1).tolist() for code in (byte & 15, byte >> 4)]
    assert actual.cpu().reshape(-1).tolist() == expected


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
def test_valid_preallocated_output_is_returned_by_identity(device, backend, kind):
    raw = torch.tensor([[0x38, 0xB8, 0x30, 0xB0]], dtype=torch.uint8, device=device)
    shape, dtype = ((1, 8), torch.int8) if kind == "fp4" else ((1, 4), torch.bfloat16)
    out = torch.empty(shape, dtype=dtype, device=device)
    result = (unpack.unpack_fp4_to_int8(raw, out=out, backend=backend) if kind == "fp4"
              else unpack.unpack_fp8_to_bf16(raw, out=out, backend=backend))
    assert result is out


@pytest.mark.parametrize("kind", ["fp4", "fp8", "scale"])
def test_rejects_wrong_payload_dtype_and_non_tensor(kind):
    operation = {"fp4": unpack.unpack_fp4_to_int8, "fp8": unpack.unpack_fp8_to_bf16,
                 "scale": unpack.fp4_int8_scale}[kind]
    for source in ([1, 2], torch.zeros(2, 4, dtype=torch.float32), torch.zeros(2, 4, dtype=torch.int16)):
        with pytest.raises(TypeError):
            operation(source)


@pytest.mark.parametrize("kind", ["fp4", "fp8", "scale"])
def test_rejects_noncontiguous_input_instead_of_copying_payload(kind):
    source = torch.zeros(4, 4, dtype=torch.uint8).t()
    operation = {"fp4": unpack.unpack_fp4_to_int8, "fp8": unpack.unpack_fp8_to_bf16,
                 "scale": unpack.fp4_int8_scale}[kind]
    with pytest.raises(ValueError, match="contiguous"):
        operation(source)


def test_rejects_scalar_packed_weight_and_scaled_vector():
    with pytest.raises(ValueError, match="rank"):
        unpack.unpack_fp4_to_int8(torch.tensor(0, dtype=torch.uint8))
    with pytest.raises(ValueError, match="rank"):
        unpack.unpack_fp8_to_bf16(torch.zeros(4, dtype=torch.uint8), torch.zeros(1, 1, dtype=torch.uint8))


@pytest.mark.parametrize("block_shape", [(0, 32), (32, -1), (True, 32), (1.5, 32), (32,), None, "32"])
def test_rejects_invalid_block_shape(block_shape):
    with pytest.raises(ValueError, match="block_shape"):
        unpack.unpack_fp8_to_bf16(torch.zeros(2, 4, dtype=torch.uint8), block_shape=block_shape)


def test_rejects_scale_broadcasting_wrong_dtype_and_noncontiguity():
    weight = torch.zeros(2, 5, 9, dtype=torch.uint8)
    with pytest.raises(ValueError, match="scale must have shape"):
        unpack.unpack_fp8_to_bf16(weight, torch.zeros(1, 1, dtype=torch.uint8))
    with pytest.raises(TypeError, match="byte codes"):
        unpack.unpack_fp8_to_bf16(weight, torch.ones(2, 1, 1, dtype=torch.float32))
    with pytest.raises(ValueError, match="contiguous"):
        unpack.unpack_fp8_to_bf16(torch.zeros(64, 64, dtype=torch.uint8), torch.zeros(2, 2, dtype=torch.uint8).t())


@pytest.mark.parametrize("cpu_operand", ["weight", "scale"])
def test_scaled_fp8_rejects_physical_cpu_accelerator_device_mismatch(device, monkeypatch, cpu_operand):
    if device.type == "cpu":
        pytest.skip("physical mixed-device validation requires an explicitly requested accelerator; not a pass")
    weight_device = "cpu" if cpu_operand == "weight" else device
    scale_device = "cpu" if cpu_operand == "scale" else device
    weight = torch.zeros((2, 4), dtype=torch.uint8, device=weight_device)
    scales = torch.full((1, 1), 127, dtype=torch.uint8, device=scale_device)

    def forbid_dispatch_or_output(*args, **kwargs):
        raise AssertionError("mixed-device inputs reached backend selection or output allocation")

    monkeypatch.setattr(unpack, "_select_backend", forbid_dispatch_or_output)
    monkeypatch.setattr(unpack, "_output", forbid_dispatch_or_output)
    with pytest.raises(ValueError, match="same device"):
        unpack.unpack_fp8_to_bf16(weight, scales, backend="torch")


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
@pytest.mark.parametrize("problem", ["dtype", "shape", "noncontiguous", "device", "non_tensor"])
def test_rejects_invalid_output_metadata(kind, problem):
    raw = torch.zeros(2, 4, dtype=torch.uint8)
    shape, dtype = ((2, 8), torch.int8) if kind == "fp4" else ((2, 4), torch.bfloat16)
    if problem == "dtype":
        out = torch.empty(shape, dtype=torch.float32)
    elif problem == "shape":
        out = torch.empty((1, 1), dtype=dtype)
    elif problem == "noncontiguous":
        out = torch.empty((shape[1], shape[0]), dtype=dtype).t()
    elif problem == "device":
        out = torch.empty(shape, dtype=dtype, device="meta")
    else:
        out = [0]
    operation = unpack.unpack_fp4_to_int8 if kind == "fp4" else unpack.unpack_fp8_to_bf16
    with pytest.raises((TypeError, ValueError)):
        operation(raw, out=out)


@pytest.mark.parametrize("disjoint", [False, True])
def test_fp4_output_rejects_shared_storage_even_for_disjoint_views(disjoint):
    arena = torch.zeros(32, dtype=torch.uint8)
    source = arena[:8].reshape(2, 4)
    out = arena[16:32].view(torch.int8).reshape(2, 8) if disjoint else arena[:16].view(torch.int8).reshape(2, 8)
    with pytest.raises(ValueError, match="alias|share storage"):
        unpack.unpack_fp4_to_int8(source, out=out)


def test_fp8_output_rejects_alias_with_weight_or_scale():
    arena = torch.zeros(64, dtype=torch.uint8)
    source = arena[:16].reshape(1, 16)
    out = arena.view(torch.bfloat16)[:16].reshape(1, 16)
    with pytest.raises(ValueError, match="alias|share storage"):
        unpack.unpack_fp8_to_bf16(source, out=out)
    separate_source = torch.zeros(1, 16, dtype=torch.uint8)
    scale = arena[:2].reshape(1, 2)
    with pytest.raises(ValueError, match="alias|share storage"):
        unpack.unpack_fp8_to_bf16(separate_source, scale, block_shape=(1, 8), out=out)


@pytest.mark.parametrize("kind", ["fp4", "fp8", "scale"])
def test_unknown_backend_is_not_silently_accepted(kind):
    operation = {"fp4": unpack.unpack_fp4_to_int8, "fp8": unpack.unpack_fp8_to_bf16,
                 "scale": unpack.fp4_int8_scale}[kind]
    with pytest.raises(ValueError, match="backend"):
        operation(torch.zeros(2, 4, dtype=torch.uint8), backend="unknown")


def test_explicit_triton_is_npu_only_even_for_an_empty_input():
    with pytest.raises(ValueError, match="requires an NPU"):
        unpack.unpack_fp4_to_int8(torch.empty(0, dtype=torch.uint8), backend="triton")


def test_auto_cpu_does_not_import_optional_accelerator_backend(monkeypatch):
    def forbid_import(name):
        raise AssertionError(f"CPU auto unexpectedly imported {name}")
    monkeypatch.setattr(unpack.importlib, "import_module", forbid_import)
    result = unpack.unpack_fp4_to_int8(torch.tensor([0x21], dtype=torch.uint8), backend="auto")
    assert result.tolist() == [1, 2]


def test_backend_availability_is_selected_before_execution(monkeypatch):
    unavailable = SimpleNamespace(supports_triton_ascend=lambda: False)
    monkeypatch.setattr(unpack.importlib, "import_module", lambda name: unavailable)
    device = SimpleNamespace(type="npu")
    assert unpack._select_backend("auto", device) is None
    with pytest.raises(RuntimeError, match="unavailable"):
        unpack._select_backend("triton", device)
    available = SimpleNamespace(supports_triton_ascend=lambda: True)
    monkeypatch.setattr(unpack.importlib, "import_module", lambda name: available)
    assert unpack._select_backend("auto", device) is available


@pytest.mark.parametrize("kind", ["fp4", "fp8", "scale"])
def test_kernel_execution_failure_is_not_retried_through_torch(monkeypatch, kind):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate accelerator execution failure")
    implementation = SimpleNamespace(fp4_unpack_int8=fail, fp8_unpack_bf16=fail, fp4_int8_scale=fail)
    monkeypatch.setattr(unpack, "_select_backend", lambda backend, device: implementation)
    operation = {"fp4": unpack.unpack_fp4_to_int8, "fp8": unpack.unpack_fp8_to_bf16,
                 "scale": unpack.fp4_int8_scale}[kind]
    with pytest.raises(RuntimeError, match="deliberate accelerator execution failure"):
        operation(torch.tensor([[0x38]], dtype=torch.uint8), backend="auto")


def test_torch_payload_path_uses_neither_float8_cast_nor_cpu_copy(device, monkeypatch):
    raw = torch.tensor([[0x38, 0xB8, 0x00, 0x80]], dtype=torch.uint8, device=device)
    scales = torch.tensor([[127]], dtype=torch.uint8, device=device)
    original_to = torch.Tensor.to
    float8_dtypes = {getattr(torch, name) for name in ("float8_e4m3fn", "float8_e8m0fnu") if hasattr(torch, name)}
    def checked_to(self, *args, **kwargs):
        dtype, destination = kwargs.get("dtype"), kwargs.get("device")
        tensor_target = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor_target is not None:
            destination = tensor_target.device
            if dtype is None:
                dtype = tensor_target.dtype
        else:
            if dtype is None:
                dtype = next((arg for arg in args if isinstance(arg, torch.dtype)), None)
            if destination is None:
                destination = next((arg for arg in args if isinstance(arg, (str, torch.device))), None)
        if dtype in float8_dtypes:
            raise AssertionError("the payload path performed a float8 cast")
        if destination is not None and torch.device(destination).type == "cpu" and self.device.type != "cpu":
            raise AssertionError("the payload path copied an accelerator tensor to the CPU with .to()")
        return original_to(self, *args, **kwargs)
    def no_cpu(self, *args, **kwargs):
        raise AssertionError("the payload path copied a tensor to the CPU")
    # Exercise every CPU-target overload even in CPU-only runs. This synthetic
    # device metadata checks the guard, not accelerator execution or correctness.
    accelerator_source = SimpleNamespace(device=SimpleNamespace(type="npu"))
    cpu_target = torch.empty(1, dtype=torch.uint8)
    for args, kwargs in ((("cpu",), {}), ((torch.device("cpu"),), {}),
                         ((), {"device": "cpu"}), ((cpu_target,), {})):
        with pytest.raises(AssertionError, match="copied.*CPU"):
            checked_to(accelerator_source, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", checked_to)
        patch.setattr(torch.Tensor, "cpu", no_cpu)
        fp4 = unpack.unpack_fp4_to_int8(raw, backend="torch")
        fp8 = unpack.unpack_fp8_to_bf16(raw, scales, block_shape=(1, 4), backend="torch")
        half = unpack.fp4_int8_scale(scales, backend="torch")
    assert fp4.device == device and fp8.device == device and half.device == device
