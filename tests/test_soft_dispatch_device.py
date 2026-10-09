"""`impl="auto"` has to answer for the tensor, not only for the card.

`_auto_impl` reads the *device capability* and nothing else, so on an sm_8x machine it
answers `triton` for a weight that may be sitting on the host. A triton kernel cannot
take a host pointer, and the failure is not a fallback: the first argument raises
`ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`.

The bug is arch-shaped, which is why it went unseen. On sm_75 `major < 8` sends every
fp8 op down the torch path whatever the tensor's device, so a host tensor works by
accident and the missing guard is invisible; the same code on the RTX 4090 reaches the
kernel and raises. The relicllm suite builds its checkpoint fixtures on the CPU, so the
24 errors it hit on sm_89 were all this one dispatch -- and a test gated on the card
would have skipped on the one this library was written for.

`_soft_gemm` and `hc_split_sinkhorn` already guard on `is_cuda`; the two fp8 entry
points did not. The tests below cover those two, and the third covers the guard's own
risk: making every fp8 call soft would pass the first two just as well.
"""

import pytest
import torch

from relic_core.kernels import ops


def _fp8_pair(device: str, rows: int = 128, cols: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp8 weight and one E8M0 scale per 128x128 block, the way a checkpoint stores it."""
    generator = torch.Generator().manual_seed(20261009)
    weight = (torch.rand(rows, cols, generator=generator) * 4 - 2).to(torch.float8_e4m3fn)
    scale = torch.ones(-(-rows // 128), -(-cols // 128), dtype=torch.float32)
    return weight.to(device), scale.to(device)


def test_an_auto_dequant_on_a_host_tensor_answers_instead_of_raising():
    """The regression. A host tensor is what the loader hands these ops when a test writes a
    miniature checkpoint to a temporary directory -- the call the sm_89 run found."""
    weight, scale = _fp8_pair("cpu")

    out = ops.soft_fp8_blockfp8_weight_dequant(weight, scale, 128, impl="auto")

    assert out.device.type == "cpu"
    assert out.shape == weight.shape


def test_the_auto_path_and_the_soft_path_agree_on_a_host_tensor():
    """A fallback that produced different numbers would be worse than the exception it replaced."""
    weight, scale = _fp8_pair("cpu")

    auto = ops.soft_fp8_blockfp8_weight_dequant(weight, scale, 128, impl="auto")
    soft = ops.soft_fp8_blockfp8_weight_dequant_torch(weight, scale, 128)

    assert torch.equal(auto, soft)


def test_an_auto_gemm_on_a_host_tensor_answers_instead_of_raising():
    """The same missing guard on the other fp8 entry point, reached by the same shape of call."""
    weight, scale = _fp8_pair("cpu", rows=128, cols=256)
    x = torch.randn(4, 256) * 0.5

    out = ops.soft_fp8_blockfp8_gemm(x, weight, scale, impl="auto")

    assert out.device.type == "cpu"
    assert out.shape == (4, 128)
    assert torch.equal(out, ops.soft_fp8_blockfp8_gemm_torch(x, weight, scale))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a device tensor needs a device")
def test_a_device_dequant_still_reaches_the_kernel_auto_chose():
    """The guard must not cost sm_8x the triton path, which is the reason `auto` reads the card.

    Without this, "make the condition check the device" and "always take the soft path" pass
    the tests above equally well.
    """
    if ops._resolve_impl("fp8", "auto") != "triton":
        pytest.skip("this build resolves fp8 to the soft path; the guard is not what is under test")

    weight, scale = _fp8_pair("cuda")
    on_card = ops.soft_fp8_blockfp8_weight_dequant(weight, scale, 128, impl="auto")
    on_host = ops.soft_fp8_blockfp8_weight_dequant_torch(weight.cpu(), scale.cpu(), 128)

    assert on_card.device.type == "cuda"
    # The card path returns a half-precision dtype, so the comparison is of values and not bits.
    assert torch.allclose(on_card.float().cpu(), on_host, atol=1e-2, rtol=1e-3)