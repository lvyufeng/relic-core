"""Exact original FP32-weight bit reconstruction, not rounded BF16 weights."""
import pytest
import torch

from relic_core.kernels.exact_weight_plane_gemv import prepare_exact_fp32_planes,exact_weight_plane_gemv


def test_gaussian_fp32_weights_are_preserved_as_three_exact_planes():
    weight=torch.randn(24,256,generator=torch.Generator().manual_seed(1511)).float()*0.01
    prepared=prepare_exact_fp32_planes(weight)
    reconstructed=(prepared.planes[0][:,:24].float()+prepared.planes[1][:,:24].float()+prepared.planes[2][:,:24].float()).t()
    assert torch.equal(reconstructed.view(torch.int32),weight.view(torch.int32))
    assert not torch.equal(weight.bfloat16().float().view(torch.int32),weight.view(torch.int32))
    x=torch.randn(1,256,generator=torch.Generator().manual_seed(1512)).bfloat16()
    # Layout/native accumulation may differ; only source reconstruction is a bit contract.
    torch.testing.assert_close(exact_weight_plane_gemv(x,prepared,backend="torch"),x.float()@weight.t(),rtol=1e-5,atol=1e-7)


def test_unrepresentable_weights_are_not_silently_narrowed():
    weight=torch.zeros((24,256),dtype=torch.float32)
    weight[0,0]=torch.finfo(torch.float32).max
    with pytest.raises(ValueError,match="exactly reconstruct"):prepare_exact_fp32_planes(weight)


def test_fp32_activation_is_refused_rather_than_cast_to_bf16():
    prepared=prepare_exact_fp32_planes(torch.ones((24,256),dtype=torch.float32))
    with pytest.raises(ValueError,match="BF16 activation"):
        exact_weight_plane_gemv(torch.ones((1,256),dtype=torch.float32),prepared,backend="torch")
