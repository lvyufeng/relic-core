"""Real NPU memory-tail guards for the fused FP4 byte-unpack kernel."""
from __future__ import annotations

import os

import pytest
import torch

from relic_core.kernels.unpack import unpack_fp4_to_int8


pytestmark = pytest.mark.skipif(
    os.environ.get("RELIC_ONLINE_UNPACK_TEST_BACKEND") != "triton"
    or not str(os.environ.get("RELIC_ONLINE_UNPACK_TEST_DEVICE", "")).startswith("npu"),
    reason="requires an explicitly requested fused Ascend device; not a CPU acceptance test",
)


@pytest.mark.parametrize("count", [1, 3, 255, 513, 1025, 2047])
def test_fused_fp4_output_mask_preserves_both_guard_regions(count):
    import torch_npu
    device = torch.device(os.environ["RELIC_ONLINE_UNPACK_TEST_DEVICE"])
    raw_host = (torch.arange(count, dtype=torch.int32) % 256).to(torch.uint8)
    expected = unpack_fp4_to_int8(raw_host, backend="torch")
    raw = raw_host.to(device)
    # Guards share the output allocation, not an input allocation. A contiguous
    # slice keeps a valid out= contract while exposing an out-of-bounds tail write.
    guard = 64
    storage = torch.full((2 * count + 2 * guard,), -85, dtype=torch.int8, device=device)
    out = storage[guard:guard + 2 * count]
    actual = unpack_fp4_to_int8(raw, backend="triton", out=out)
    torch.npu.synchronize(device)
    assert actual is out
    observed = storage.cpu()
    assert torch.equal(observed[:guard], torch.full((guard,), -85, dtype=torch.int8))
    assert torch.equal(observed[guard + 2 * count:], torch.full((guard,), -85, dtype=torch.int8))
    assert torch.equal(observed[guard:guard + 2 * count], expected)
