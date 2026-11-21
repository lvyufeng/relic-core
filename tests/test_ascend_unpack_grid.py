"""Real Ascend grid-stride/tail checks added after large-checkpoint loading.

These require an explicit leased device. The large FP8 case crosses the actual
65535-program ceiling; smaller FP4/scale cases force the same loop with two
programs and compare every output bit against the independent CPU backend.
"""
from __future__ import annotations

import os

import pytest
import torch

from relic_core.kernels import ascend_unpack
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8, unpack_fp8_to_bf16


@pytest.fixture
def device():
    name = os.environ.get("RELIC_ONLINE_UNPACK_TEST_DEVICE", "")
    if not name.startswith("npu") or os.environ.get("RELIC_ONLINE_UNPACK_TEST_BACKEND") != "triton":
        pytest.skip("requires an explicitly leased NPU with the fused Triton backend")
    import torch_npu  # noqa: F401

    if not ascend_unpack.supports_triton_ascend():
        pytest.fail("an explicitly requested fused grid test needs the installed Ascend backend")
    return torch.device(name)


def test_fp8_matrix_crosses_the_actual_grid_limit_with_complete_output_and_guards(device):
    rows, width = 65536, ascend_unpack.BLOCK
    assert ascend_unpack._grid(rows * width) == (65535,)
    raw = torch.full((rows, width), 0x38, dtype=torch.uint8).to(device)
    scales = torch.full((rows // 32, width // 32), 127, dtype=torch.uint8).to(device)
    guard = 64
    storage = torch.full((rows * width + 2 * guard,), -85.0, dtype=torch.bfloat16, device=device)
    output = storage[guard:-guard].reshape(rows, width)
    assert unpack_fp8_to_bf16(raw, scales, backend="triton", out=output) is output
    torch.npu.synchronize(device)
    observed = storage.cpu()
    assert torch.equal(observed[:guard], torch.full((guard,), -85.0, dtype=torch.bfloat16))
    assert torch.equal(observed[-guard:], torch.full((guard,), -85.0, dtype=torch.bfloat16))
    # E4M3 0x38 is exactly 1.0, and E8M0 127 exactly one; inspect every lane,
    # including tiles assigned after the first grid pass, not a sampled tail.
    assert bool((observed[guard:-guard].view(torch.int16) == 0x3F80).all())


def test_fp4_forced_grid_stride_preserves_every_value_and_tail_guard(device, monkeypatch):
    monkeypatch.setattr(ascend_unpack, "MAX_PROGRAMS", 2)
    count = 8193
    host = (torch.arange(count, dtype=torch.int32) % 256).to(torch.uint8)
    expected = unpack_fp4_to_int8(host, backend="torch")
    raw = host.to(device)
    guard = 64
    storage = torch.full((2 * count + 2 * guard,), -85, dtype=torch.int8, device=device)
    output = storage[guard:-guard]
    assert ascend_unpack._grid(count) == (2,)
    assert unpack_fp4_to_int8(raw, backend="triton", out=output) is output
    torch.npu.synchronize(device)
    observed = storage.cpu()
    assert torch.equal(observed[:guard], torch.full((guard,), -85, dtype=torch.int8))
    assert torch.equal(observed[-guard:], torch.full((guard,), -85, dtype=torch.int8))
    assert torch.equal(observed[guard:-guard], expected)


def test_scale_forced_grid_stride_preserves_all_bits_and_tail_guard(device, monkeypatch):
    monkeypatch.setattr(ascend_unpack, "MAX_PROGRAMS", 2)
    count = 4097
    host = (torch.arange(count, dtype=torch.int32) % 256).to(torch.uint8)
    expected = fp4_int8_scale(host, backend="torch")
    raw = host.to(device)
    guard = 64
    storage = torch.full((count + 2 * guard,), -85.0, dtype=torch.float32, device=device)
    output = storage[guard:-guard]
    assert ascend_unpack._grid(count) == (2,)
    # The public scale API allocates its result; the internal launcher accepts
    # this guarded allocation for the grid/tail contract under test.
    assert ascend_unpack.fp4_int8_scale(raw, output) is output
    torch.npu.synchronize(device)
    observed = storage.cpu()
    assert torch.equal(observed[:guard], torch.full((guard,), -85.0, dtype=torch.float32))
    assert torch.equal(observed[-guard:], torch.full((guard,), -85.0, dtype=torch.float32))
    assert torch.equal(observed[guard:-guard].view(torch.int32), expected.view(torch.int32))
