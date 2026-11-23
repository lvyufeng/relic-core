"""CPU oracle tests for the opt-in sparse-attention operand layout candidate.

Two layers are checked here. First, the layouts themselves, exactly: the fused
gather must reproduce the reference's fancy-index gather (negative entries from
row 0, then masked), with contiguous FP32 full blocks and an exact-size tail that
is never padded to 64. Second, end-to-end output against the released
``ops._sparse_attn_torch``. The candidate feeds ``torch.bmm`` the same product
with contiguous operands instead of strided views, so a BLAS kernel may schedule
differently; the comparison here is tight but tolerance-based, and the root
harness owns the bitwise component gate (all eight ranks, full model).

No NPU is required for this module: the fused Triton path is skipped unless a
Triton Ascend compiler and an NPU device are present.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from relic_core.kernels import _ascend_sparse_attention_layout_kernels as kernels
from relic_core.kernels import ascend_sparse_attention_layout as layout
from relic_core.kernels import ops

#: Head dimension of the released sparse-attention path.
D = 512
#: Tolerance for the end-to-end comparison: same operands and same arithmetic,
#: but contiguous ``bmm`` operands are allowed a different BLAS scheduling.
ATOL = RTOL = 1e-5


def make(*, heads=2, n=96, topk=96, dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, heads, D, generator=gen, dtype=torch.float32).to(dtype)
    kv = torch.randn(1, n, D, generator=gen, dtype=torch.float32).to(dtype)
    indices = torch.randint(0, n, (1, 1, topk), generator=gen, dtype=torch.int32)
    sink = torch.randn(heads, generator=gen, dtype=torch.float32)
    return q, kv, indices, sink


def budget(topk, heads=2):
    return layout.layout_workspace_bytes(topk, heads=heads)


def run(q, kv, indices, sink, *, scale=0.75, backend="torch", **kwargs):
    # Not setdefault: its default argument is evaluated eagerly even when the key
    # is present, which would compute a budget from a deliberately bad q/topk and
    # raise before the guard under test.
    if "max_workspace_bytes" not in kwargs:
        kwargs["max_workspace_bytes"] = budget(indices.shape[-1], q.shape[2])
    return layout.sparse_attn_layout(q, kv, indices, sink, scale, backend=backend, **kwargs)


def reference(q, kv, indices, sink, *, scale=0.75):
    return ops._sparse_attn_torch(q, kv, sink, indices, scale)


def build(q, kv, indices):
    return layout.build_sparse_attention_layouts(
        q, kv, indices, backend="torch",
        max_workspace_bytes=budget(indices.shape[-1], q.shape[2]))


# ------------------------------------------------------------------- layouts


@pytest.mark.parametrize("topk", [1, 63, 64, 65, 127, 128, 129, 288])
def test_layouts_are_the_reference_gather_with_exact_full_blocks_and_tail(topk):
    q, kv, indices, _ = make(heads=3, n=320, topk=topk)
    built = build(q, kv, indices)
    nfull, tail = divmod(topk, layout.BLOCK)
    gathered = kv[0, indices[0, 0].clamp_min(0)].to(torch.float32)  # (T, 512)

    assert built.nfull == nfull and built.tail == tail
    assert built.straight_full.shape == (nfull, layout.BLOCK, D)
    assert built.transposed_full.shape == (nfull, D, layout.BLOCK)
    assert built.straight_full.dtype == torch.float32 and built.straight_full.is_contiguous()
    assert built.transposed_full.is_contiguous()
    if nfull:
        assert torch.equal(built.straight_full, gathered[:nfull * layout.BLOCK].reshape(nfull, layout.BLOCK, D))
        assert torch.equal(built.transposed_full, built.straight_full.transpose(1, 2))
    if tail:
        assert built.straight_tail.shape == (tail, D) and built.straight_tail.is_contiguous()
        assert built.transposed_tail.shape == (D, tail) and built.transposed_tail.is_contiguous()
        assert torch.equal(built.straight_tail, gathered[nfull * layout.BLOCK:])
        assert torch.equal(built.transposed_tail, built.straight_tail.transpose(0, 1))
    else:
        assert built.straight_tail is None and built.transposed_tail is None
    assert torch.equal(built.valid, (indices[0, 0] >= 0).reshape(1, topk))


def test_the_tail_is_the_true_width_and_never_padded_to_a_block():
    q, kv, indices, _ = make(heads=2, n=200, topk=100)
    built = build(q, kv, indices)
    assert built.nfull == 1 and built.tail == 36
    assert built.straight_tail.shape == (36, D)
    assert built.transposed_tail.shape == (D, 36)
    # The only 64-wide tensors are the complete blocks.
    assert built.straight_full.shape[1] == layout.BLOCK


def test_a_multiple_of_the_block_has_no_tail_tensors():
    q, kv, indices, _ = make(heads=2, n=192, topk=192)
    built = build(q, kv, indices)
    assert built.nfull == 3 and built.tail == 0
    assert built.straight_tail is None and built.transposed_tail is None


def test_topk_zero_builds_empty_layouts_and_reduces_to_the_sink_only():
    q, kv, indices, sink = make(heads=2, n=8, topk=0)
    built = build(q, kv, indices)
    assert built.nfull == 0 and built.tail == 0 and built.valid.shape == (1, 0)
    out = run(q, kv, indices, sink)
    ref = reference(q, kv, indices, sink)
    assert torch.equal(out, ref)
    assert torch.equal(out, torch.zeros_like(out))


# -------------------------------------------------------------- exact arithmetic


@pytest.mark.parametrize("topk", [1, 63, 64, 65, 100, 128, 129, 192, 288, 512])
def test_output_matches_the_released_torch_reference(topk):
    q, kv, indices, sink = make(heads=2, n=max(topk, 64), topk=topk, seed=topk)
    out = run(q, kv, indices, sink)
    ref = reference(q, kv, indices, sink)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_all_invalid_indices_reduce_to_zeros_like_the_reference():
    q, kv, indices, sink = make(heads=2, n=64, topk=96)
    indices = torch.full_like(indices, -1)
    out = run(q, kv, indices, sink)
    ref = reference(q, kv, indices, sink)
    assert torch.equal(out, ref)
    assert torch.equal(out, torch.zeros_like(out))


def test_negative_indices_below_the_sentinel_are_still_invalid():
    q, kv, indices, sink = make(heads=2, n=64, topk=70)
    indices = indices.clone()
    indices[0, 0, 0] = -5
    indices[0, 0, 1] = -1
    out = run(q, kv, indices, sink)
    ref = reference(q, kv, indices, sink)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_a_non_contiguous_q_with_a_unit_last_stride_is_accepted():
    heads = 2
    q, kv, indices, sink = make(heads=heads, n=96, topk=96)
    base = torch.randn(1, 1, heads * 2, D)
    q_view = base[:, :, ::2, :]
    assert q_view.shape == q.shape and q_view.stride(-1) == 1 and not q_view.is_contiguous()
    out = run(q_view, kv, indices, sink)
    torch.testing.assert_close(out, reference(q_view, kv, indices, sink), rtol=RTOL, atol=ATOL)


def test_bfloat16_inputs_return_bfloat16_outputs():
    q, kv, indices, sink = make(heads=2, n=96, topk=100, dtype=torch.bfloat16)
    out = run(q, kv, indices, sink)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_a_grad_enabled_sink_parameter_is_read_only_and_accepted_in_inference():
    """The real attention sink is an nn.Parameter with requires_grad=True.

    The layout candidate never writes it and never inspects requires_grad; it only
    reads the value, so it is admitted in an explicit inference context. Callers
    still choose the context: outside inference the reference arithmetic tracks
    gradients, so the runtime guard requires torch.is_grad_enabled() to be False.
    """
    q, kv, indices, sink = make(heads=2, n=96, topk=100)
    sink_parameter = torch.nn.Parameter(sink.clone())
    before = sink_parameter.detach().clone()
    with torch.inference_mode():
        out = run(q, kv, indices, sink_parameter)
        ref = reference(q, kv, indices, sink_parameter)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    assert torch.equal(sink_parameter.detach(), before)


def test_operands_are_treated_as_read_only():
    q, kv, indices, sink = make(heads=2, n=96, topk=100)
    q_before, kv_before, idx_before, sink_before = (value.clone() for value in (q, kv, indices, sink))
    run(q, kv, indices, sink)
    assert torch.equal(q, q_before) and torch.equal(kv, kv_before)
    assert torch.equal(indices, idx_before) and torch.equal(sink, sink_before)


# ------------------------------------------------------------------- the guards


def test_the_workspace_estimate_is_a_declared_tensor_budget():
    q, kv, indices, sink = make(heads=2, n=64, topk=96)
    needed = budget(96, heads=2)
    assert needed > 0 and isinstance(needed, int)
    # A declared per-call budget: less than it is refused, and the check is real.
    with pytest.raises(ValueError, match="per-call scratch"):
        run(q, kv, indices, sink, max_workspace_bytes=needed - 1)


def test_the_budget_is_declared_as_an_estimate_with_explicit_exclusions():
    """The helper must not claim to be a hard cap or a measured absolute peak."""
    words = " ".join(_LAYOUT_SOURCE.split())
    assert "declared per-call estimate" in words
    assert "not a guaranteed upper bound" in words
    for excluded in ("allocator rounding", "graph memory pools", "BMM workspace",
                     "Ascend UB", "absolute peak"):
        assert excluded in words, excluded
    assert layout.ALLOCATION_ALLOWANCE > 0


def test_backend_and_certificate_are_explicit():
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="backend must be one of"):
        run(q, kv, indices, sink, backend="auto")
    with pytest.raises(ValueError, match="explicit boolean"):
        run(q, kv, indices, sink, certified_valid_indices=1)
    with pytest.raises(ValueError, match="NPU device"):
        run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)


@pytest.mark.parametrize("bad", [0, -1, True, None, 3.5, "4096"])
def test_max_workspace_bytes_must_be_an_explicit_positive_integer(bad):
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="positive integer"):
        run(q, kv, indices, sink, max_workspace_bytes=bad)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), True, "0.7"])
def test_the_scale_must_be_finite(scale):
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="finite number"):
        run(q, kv, indices, sink, scale=scale)


def test_shape_dtype_and_contiguity_guards():
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="B1 q"):
        run(q.expand(2, 1, 2, D).contiguous(), kv, indices, sink)
    with pytest.raises(ValueError, match="unit last stride"):
        bad_q = torch.randn(1, 1, D, 2).transpose(-1, -2)
        assert bad_q.shape == (1, 1, 2, D) and bad_q.stride(-1) != 1
        run(bad_q, kv, indices, sink)
    with pytest.raises(ValueError, match="FP32/BF16"):
        run(q.to(torch.float16), kv, indices, sink)
    with pytest.raises(ValueError, match="kv of shape"):
        run(q, kv.unsqueeze(0), indices, sink)
    with pytest.raises(ValueError, match="contiguous FP32/BF16"):
        noncontig_kv = torch.randn(1, D, kv.shape[1]).transpose(1, 2)
        assert noncontig_kv.shape == kv.shape and not noncontig_kv.is_contiguous()
        run(q, noncontig_kv, indices, sink)
    with pytest.raises(ValueError, match="int32 indices"):
        run(q, kv, indices.long(), sink)
    with pytest.raises(ValueError, match="contiguous int32 indices"):
        run(q, kv, torch.zeros(1, 1, 64, dtype=torch.int32)[:, :, ::2], sink)
    with pytest.raises(ValueError, match=r"\(H,\)"):
        run(q, kv, indices, torch.randn(4))
    with pytest.raises(ValueError, match="contiguous FP32/BF16/FP16"):
        run(q, kv, indices, sink.to(torch.int32))
    with pytest.raises(ValueError, match="at least one head"):
        run(q[:, :, :0], kv, indices, sink, max_workspace_bytes=8 * 1024 * 1024)


def test_n_and_topk_are_bounded():
    q, kv, indices, sink = make(heads=2, n=8, topk=8)
    with pytest.raises(ValueError, match="1 <= N <="):
        run(q, kv[:, :0], indices, sink)
    with pytest.raises(ValueError, match="0 <= T <="):
        run(q, kv, torch.zeros(1, 1, layout.MAX_TOPK + 1, dtype=torch.int32), sink,
            max_workspace_bytes=8 * 1024 * 1024)


# ------------------------------------------------------- static NATIVE contract
#
# The fused path must stay a pure forward gather/prepack: no host synchronization,
# no device-to-host copy, no fallback, and the exact 32x128 tile the private root
# harness expects.

_KERNEL_SOURCE = Path(kernels.__file__).read_text()
_LAYOUT_SOURCE = Path(layout.__file__).read_text()


def test_the_kernel_is_the_experimental_16x64_forward_gather_writing_both_layouts():
    """The 32x128 prototype overflowed the UB budget; the shipped tile is 16x64."""
    assert kernels.TILE_W == 16 and kernels.TILE_D == 64
    assert kernels.BLOCK == 64 and kernels.HEAD_DIM == 512
    assert "TILE_W = 16" in _KERNEL_SOURCE and "TILE_D = 64" in _KERNEL_SOURCE
    assert "HEAD_DIM = 512" in _KERNEL_SOURCE
    assert "WIDTH: tl.constexpr" in _KERNEL_SOURCE
    assert "STRAIGHT" in _KERNEL_SOURCE and "TRANSPOSED" in _KERNEL_SOURCE
    assert "tl.trans(" in _KERNEL_SOURCE
    assert "tl.store" in _KERNEL_SOURCE
    assert "BLOCK = 64" in _KERNEL_SOURCE
    # 32x128 was measured to overflow; it must not reappear as the active tile.
    assert "TILE_W = 32" not in _KERNEL_SOURCE and "TILE_D = 128" not in _KERNEL_SOURCE


def test_the_fused_path_has_no_host_sync_and_no_fallback():
    for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize", "cudaStreamSynchronize"):
        assert forbidden not in _KERNEL_SOURCE, forbidden
        assert forbidden not in _LAYOUT_SOURCE, forbidden
    assert "except Exception" in _KERNEL_SOURCE  # compiler import guard only


def test_the_public_contract_names_are_stable():
    assert layout.BLOCK == 64 and layout.HEAD_DIM == 512
    assert layout.MAX_TOPK == 512 and layout.MAX_KV_ROWS == 4096
    assert "certified_valid_indices" in _LAYOUT_SOURCE
    assert "max_workspace_bytes" in _LAYOUT_SOURCE
    assert layout.layout_workspace_bytes(64) > layout.layout_workspace_bytes(0)


def test_the_fused_path_runs_under_the_npu_launch_guard():
    """The launch must switch to this tensor's NPU device and the current stream."""
    assert "_npu_launch" in _LAYOUT_SOURCE
    assert "with _npu_launch(kv):" in _LAYOUT_SOURCE


def test_the_full_block_and_exact_tail_launches_are_explicit():
    """One launch for complete blocks, plus one exact-tail launch when needed."""
    assert _KERNEL_SOURCE.count("_gather_layouts_kernel[") == 2
    assert "exact-tail launch" in _LAYOUT_SOURCE
    assert "never padded" in _LAYOUT_SOURCE or "never pads to a block" in _KERNEL_SOURCE


def test_the_workspace_estimate_takes_the_real_head_count():
    q, kv, indices, _ = make(heads=6, n=64, topk=100)
    one_head = layout.layout_workspace_bytes(100, heads=1)
    six_heads = layout.layout_workspace_bytes(100, heads=q.shape[2])
    assert six_heads > one_head
    assert layout.layout_workspace_bytes(100, heads=6) == six_heads


def test_the_fused_path_is_skipped_without_an_ascend_triton_backend():
    if not kernels.triton_available():
        pytest.skip("a Triton compiler is not installed on this host")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("no Ascend NPU is available on this host")
    q, kv, indices, sink = make(heads=2, n=96, topk=100)
    q, kv, indices, sink = (value.npu() for value in (q, kv, indices, sink))
    out = run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)
    # Same-device reference: a CPU reference would run different bmm kernels and
    # could hide (or invent) a difference. Bitwise parity is the preference; the
    # root harness's component gate is what accepts or rejects the candidate.
    ref = ops._sparse_attn_torch(q, kv, sink, indices, 0.75)
    if not torch.equal(out, ref):
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)