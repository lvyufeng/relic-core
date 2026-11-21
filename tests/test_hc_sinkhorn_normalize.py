"""Original HC nonlinearities and exact CPU normalization ordering."""
import pytest
import torch

from relic_core.kernels.hc_sinkhorn_normalize import sinkhorn_normalize,hc_split_sinkhorn_normalized
from relic_core.kernels.ops import hc_split_sinkhorn_torch


@pytest.mark.parametrize("iterations",[1,2,20,32])
def test_cpu_normalization_is_the_original_explicit_loop_bit_for_bit(iterations):
    raw=torch.randn(3,4,4,generator=torch.Generator().manual_seed(1521))
    initial=torch.softmax(raw,dim=-1)+1e-6
    expected=initial/(initial.sum(dim=-2,keepdim=True)+1e-6)
    for _ in range(iterations-1):
        expected=expected/(expected.sum(dim=-1,keepdim=True)+1e-6)
        expected=expected/(expected.sum(dim=-2,keepdim=True)+1e-6)
    assert torch.equal(sinkhorn_normalize(initial,iterations=iterations,backend="torch"),expected)


def test_cpu_whole_hc_wrapper_stays_on_the_original_body():
    mixes=torch.randn(1,1,24,generator=torch.Generator().manual_seed(1522))
    scales=torch.randn(3,generator=torch.Generator().manual_seed(1523))
    base=torch.randn(24,generator=torch.Generator().manual_seed(1524))
    expected=hc_split_sinkhorn_torch(mixes,scales,base)
    actual=hc_split_sinkhorn_normalized(mixes,scales,base)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
