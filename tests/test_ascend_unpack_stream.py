"""Hardware-only current-stream and device-context contract for fused unpack."""
from __future__ import annotations

import os

import pytest
import torch

from relic_core.kernels import ascend_unpack
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8, unpack_fp8_to_bf16


pytestmark = pytest.mark.skipif(
    os.environ.get("RELIC_ONLINE_UNPACK_TEST_BACKEND") != "triton"
    or not str(os.environ.get("RELIC_ONLINE_UNPACK_TEST_DEVICE", "")).startswith("npu"),
    reason="requires an explicitly requested real Triton Ascend device; not a CPU acceptance test",
)


def test_fused_unpack_follows_torch_nondefault_stream_without_operator_sync():
    import torch_npu
    import triton
    if not ascend_unpack.supports_triton_ascend():
        pytest.fail("explicit fused test requires the actual installed Ascend backend")
    device = torch.device(os.environ["RELIC_ONLINE_UNPACK_TEST_DEVICE"])
    torch.npu.set_device(device)
    raw_host = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(4, 64)
    scale_host = torch.full((1, 2), 127, dtype=torch.uint8)
    fp4_expected = unpack_fp4_to_int8(raw_host, backend="torch")
    fp8_expected = unpack_fp8_to_bf16(raw_host, scale_host, backend="torch")
    scale_expected = fp4_int8_scale(scale_host, backend="torch")
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        before = torch.npu.current_device()
        raw = raw_host.to(device)
        scales = scale_host.to(device)
        target = triton.runtime.driver.active.get_current_target()
        assert target.backend.lower() in {"npu", "ascend"}
        current = triton.runtime.driver.active.get_current_stream(device.index)
        expected_stream = torch_npu._C._npu_getCurrentRawStream(device.index)
        assert current == expected_stream, "Triton and Torch launch on different current streams"
        print("FUSED_STREAM_PROVENANCE", {"target_backend": target.backend, "target_arch": target.arch,
                                           "device_index": device.index, "raw_stream": int(current)}, flush=True)
        # Consume each output with the next Torch operation BEFORE any explicit
        # synchronize. A backend on a different stream would race these reads.
        fp4 = unpack_fp4_to_int8(raw, backend="triton") + torch.zeros((), dtype=torch.int8, device=device)
        fp8 = unpack_fp8_to_bf16(raw, scales, backend="triton")
        fp8_bits = fp8.view(torch.int16) | torch.zeros((), dtype=torch.int16, device=device)
        half = fp4_int8_scale(scales, backend="triton")
        half_bits = half.view(torch.int32) | torch.zeros((), dtype=torch.int32, device=device)
        assert torch.npu.current_device() == before, "unpack changed the caller's current device"
    stream.synchronize()
    assert torch.equal(fp4.cpu(), fp4_expected)
    torch.testing.assert_close(fp8.cpu(), fp8_expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(fp8_expected)
    assert torch.equal(fp8_bits.cpu()[finite], fp8_expected.view(torch.int16)[finite])
    assert torch.equal(half_bits.cpu(), scale_expected.view(torch.int32))
