"""Tests for the S2 fused sparse-attention softmax state (Ascend B1 prototype).

The CPU oracle layer checks that the S2 composition is the accepted S1 arithmetic:
same S1 operand layouts, same native ``torch.bmm`` boundaries, same FP32 denominator
taken before the BF16 narrowing, same sink join and final division. On a device the
only accepted outcome is a bit-identical output against the accepted S1 control run
on the same NPU tensors — a deviation is a rejection, not a tolerance to loosen.

No NPU is required for the module: the Triton kernels are skipped unless a Triton
Ascend compiler and an NPU device are present, and the static layer only reads the
two S2 sources.
"""
from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest
import torch

from relic_core.kernels import _ascend_sparse_attention_state_kernels as state_kernels
from relic_core.kernels import ascend_sparse_attention_layout as layout
from relic_core.kernels import ascend_sparse_attention_state as state
from relic_core.kernels import ops

#: Head dimension of the released sparse-attention path.
D = 512
#: Tight tolerance for the CPU oracle comparison (same ops, same layouts).
ATOL = RTOL = 1e-5

_STATE_SOURCE = Path(state.__file__).read_text()
_KERNEL_SOURCE = Path(state_kernels.__file__).read_text()


def _code_only(source):
    """Tokenize a module and keep only code tokens, dropping comments and strings.

    Static contract checks must not read documentation: a docstring is allowed to
    explain "stay native torch.bmm" or "enable_fp_fusion=False" without being a
    call site. Tokenizing keeps the actual identifiers, operators and literals.
    """
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return "".join(pieces)


def _called_names(source):
    """Dotted names of every direct call in a module, e.g. ``torch.bmm``."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        while isinstance(func, ast.Subscript):  # kernel[grid](...) launch form
            func = func.value
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        if parts:
            names.add(".".join(reversed(parts)))
    return names


def _launch_keywords(source):
    """``kw.value`` for every ``kernel[grid](...)`` keyword named enable_fp_fusion."""
    values = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        for keyword in node.keywords:
            if keyword.arg == "enable_fp_fusion":
                values.append(keyword.value)
    return values


_STATE_CODE = _code_only(_STATE_SOURCE)
_KERNEL_CODE = _code_only(_KERNEL_SOURCE)




def make(*, heads=2, n=96, topk=96, dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, heads, D, generator=gen, dtype=torch.float32).to(dtype)
    kv = torch.randn(1, n, D, generator=gen, dtype=torch.float32).to(dtype)
    indices = torch.randint(0, n, (1, 1, topk), generator=gen, dtype=torch.int32)
    sink = torch.randn(heads, generator=gen, dtype=torch.float32)
    return q, kv, indices, sink


def state_budget(topk, heads=2):
    return state.state_workspace_bytes(topk, heads=heads)


def s1_budget(topk, heads=2):
    return layout.layout_workspace_bytes(topk, heads=heads)


def run(q, kv, indices, sink, *, scale=0.75, backend="torch", **kwargs):
    # Deliberately not setdefault: its default argument is evaluated eagerly even
    # when the key is present, which would compute a budget from a bad q/topk and
    # raise before the guard under test.
    if "max_workspace_bytes" not in kwargs:
        kwargs["max_workspace_bytes"] = state_budget(indices.shape[-1], q.shape[2])
    return state.sparse_attn_state(q, kv, indices, sink, scale, backend=backend, **kwargs)


def s1(q, kv, indices, sink, *, scale=0.75, backend="torch"):
    return layout.sparse_attn_layout(
        q, kv, indices, sink, scale, backend=backend,
        max_workspace_bytes=s1_budget(indices.shape[-1], q.shape[2]))


# ------------------------------------------------------------- CPU oracle


@pytest.mark.parametrize("topk", [0, 1, 63, 64, 65, 100, 128, 129, 192, 288, 512])
def test_cpu_oracle_is_the_accepted_s1_arithmetic(topk):
    q, kv, indices, sink = make(heads=2, n=max(topk, 64), topk=topk, seed=topk)
    out = run(q, kv, indices, sink)
    ref = s1(q, kv, indices, sink)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("topk", [1, 64, 100, 288])
def test_cpu_oracle_also_matches_the_released_torch_reference(topk):
    q, kv, indices, sink = make(heads=2, n=max(topk, 64), topk=topk, seed=topk + 1)
    out = run(q, kv, indices, sink)
    ref = ops._sparse_attn_torch(q, kv, sink, indices, 0.75)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_all_invalid_indices_reduce_to_zeros():
    q, kv, indices, sink = make(heads=2, n=64, topk=96)
    indices = torch.full_like(indices, -1)
    out = run(q, kv, indices, sink)
    ref = s1(q, kv, indices, sink)
    assert torch.equal(out, ref)
    assert torch.equal(out, torch.zeros_like(out))


def test_negative_indices_below_the_sentinel_are_invalid():
    q, kv, indices, sink = make(heads=2, n=64, topk=70)
    indices = indices.clone()
    indices[0, 0, 0] = -5
    indices[0, 0, 1] = -1
    out = run(q, kv, indices, sink)
    ref = s1(q, kv, indices, sink)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_a_grad_enabled_sink_parameter_is_read_only_in_inference():
    q, kv, indices, sink = make(heads=2, n=96, topk=100)
    sink_parameter = torch.nn.Parameter(sink.clone())
    before = sink_parameter.detach().clone()
    with torch.inference_mode():
        out = run(q, kv, indices, sink_parameter)
        ref = s1(q, kv, indices, sink_parameter)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    assert torch.equal(sink_parameter.detach(), before)


def test_operands_are_read_only():
    q, kv, indices, sink = make(heads=2, n=96, topk=100)
    q_before, kv_before, idx_before, sink_before = (
        value.clone() for value in (q, kv, indices, sink))
    run(q, kv, indices, sink)
    assert torch.equal(q, q_before) and torch.equal(kv, kv_before)
    assert torch.equal(indices, idx_before) and torch.equal(sink, sink_before)


# ------------------------------------------------------------------ guards


def test_backend_certificate_and_budget_are_explicit():
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="backend must be one of"):
        run(q, kv, indices, sink, backend="auto")
    with pytest.raises(ValueError, match="explicit boolean"):
        run(q, kv, indices, sink, certified_valid_indices=1)
    for bad in (0, -1, True, None):
        with pytest.raises(ValueError, match="positive integer"):
            run(q, kv, indices, sink, max_workspace_bytes=bad)
    with pytest.raises(ValueError, match="NPU device"):
        run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)


def test_geometry_and_dtype_guards_are_the_s1_guards():
    q, kv, indices, sink = make(heads=2, n=64, topk=32)
    with pytest.raises(ValueError, match="B1 q"):
        run(q.expand(2, 1, 2, D).contiguous(), kv, indices, sink)
    with pytest.raises(ValueError, match="contiguous FP32/BF16"):
        noncontig_kv = torch.randn(1, D, kv.shape[1]).transpose(1, 2)
        assert noncontig_kv.shape == kv.shape and not noncontig_kv.is_contiguous()
        run(q, noncontig_kv, indices, sink)
    with pytest.raises(ValueError, match="int32 indices"):
        run(q, kv, indices.long(), sink)
    with pytest.raises(ValueError, match="contiguous FP32/BF16/FP16"):
        run(q, kv, indices, sink.to(torch.int32))
    with pytest.raises(ValueError, match="0 <= T <="):
        run(q, kv, torch.zeros(1, 1, layout.MAX_TOPK + 1, dtype=torch.int32), sink,
            max_workspace_bytes=8 * 1024 * 1024)


def test_the_state_budget_covers_the_layout_budget_and_is_enforced():
    q, kv, indices, sink = make(heads=2, n=64, topk=96)
    needed = state_budget(96, heads=2)
    assert needed > s1_budget(96, heads=2)
    assert state_budget(96, heads=6) > state_budget(96, heads=1)
    with pytest.raises(ValueError, match="sparse-attention state needs"):
        run(q, kv, indices, sink, max_workspace_bytes=needed - 1)


# ------------------------------------------------------------ static contract


def test_s2_keeps_native_bmm_boundaries_and_s1_layouts():
    state_calls = _called_names(_STATE_SOURCE)
    kernel_calls = _called_names(_KERNEL_SOURCE)
    # The oracle composes the accepted S1 layouts; it does not delegate to S1's
    # attention function (which would make the comparison tautological).
    assert any(name.endswith("build_sparse_attention_layouts") for name in state_calls)
    assert not any(name.endswith("sparse_attn_layout") for name in state_calls)
    # Both products stay torch.bmm in the attention module, and the fused kernels
    # never call a matmul at all. AST, so docstrings cannot satisfy or break this.
    assert "torch.bmm" in state_calls
    assert not any(name.endswith("bmm") for name in kernel_calls), kernel_calls
    assert "_state_prepare_kernel" in kernel_calls
    assert "_state_post_pv_kernel" in kernel_calls


def test_the_fused_kernels_are_exact_state_math_without_host_sync():
    # S2b: the kernel has no reduction at all (the denominator is torch-side).
    assert "tl.sum" not in _KERNEL_CODE
    assert "numerator.to(tl.bfloat16).to(tl.float32)" in _KERNEL_CODE
    assert "tl.max(scores,axis=0)" in _KERNEL_CODE
    assert "tl.maximum(old_max,block_max)" in _KERNEL_CODE
    assert "TODO" not in _KERNEL_CODE and "placeholder" not in _KERNEL_CODE.lower()
    for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize"):
        assert forbidden not in _KERNEL_CODE, forbidden
        assert forbidden not in _STATE_CODE, forbidden


def test_fma_is_disabled_on_both_kernel_launches():
    values = _launch_keywords(_KERNEL_SOURCE)
    assert len(values) == 2, values  # exactly the prepare and post-pv launches
    assert all(isinstance(value, ast.Constant) and value.value is False
               for value in values), values


def test_the_denominator_is_a_torch_sum_of_the_raw_fp32_numerator():
    """S2b keeps the reference denominator expression, and it is not in the kernel.

    In the first real BF16 gate the old fused denominator stage differed from the
    reference by one FP32 denominator element (4.768371582e-07); that difference did
    not isolate the TL exp lower bits from the in-kernel reduction. S2b therefore
    restores the reference's ATen reduction tree as a hypothesis under test: the
    kernel writes the raw FP32 numerator and does not reduce, and the caller sums that
    buffer with torch.sum(-1), shape (1, H, true_width) contiguous, then multiplies
    by the scale and adds — the accepted reference expression, with no new BF16
    narrowing. Bitwise acceptance is still gated by the device comparison.
    """
    assert "OUT_RAW" in _KERNEL_CODE
    assert "tl.sum" not in _KERNEL_CODE
    assert "OLD_SUM" not in _KERNEL_CODE
    # Raw numerator is stored before the BF16-rounded P.V operand.
    assert _KERNEL_CODE.index("tl.store(OUT_RAW") < _KERNEL_CODE.index("tl.store(OUT_P")
    # The torch denominator is the reference expression on the raw (1, H, width) view.
    assert "raw_out.view(1,h,width).sum(dim=-1)" in _STATE_CODE
    assert "sum_exp=sum_exp*scale_buf.view(1,h)" in _STATE_CODE
    # The BF16-only NPU guard is still direct (no silent FP32 generalization).
    assert "supports BF16 q and kv only" in _STATE_SOURCE


def test_the_npu_domain_is_bf16_only_in_code_and_docs():
    """FP32 is refused by name, not silently dropped: code, docs and receipt agree."""
    assert "supports BF16 q and kv only" in _STATE_SOURCE
    words = " ".join(_STATE_SOURCE.split())
    assert "BF16-only" in words
    assert "952/4096" in words  # the recorded prior FP32 failure receipt
    # The refusal must run before the S1 layouts are allocated.
    assert (_STATE_SOURCE.index("supports BF16 q and kv only")
            < _STATE_SOURCE.index("build_sparse_attention_layouts"))


def test_the_budget_is_a_declared_estimate_not_a_physical_cap():
    words = " ".join(_STATE_SOURCE.split())
    assert "declared estimate" in words
    assert "not a hard cap" in words
    for excluded in ("allocator rounding", "graph pools", "backend BMM workspace",
                     "UB/local"):
        assert excluded in words, excluded


def test_lanes_is_a_constexpr_kernel_parameter():
    """Ascend rejects a module global inside tl.arange; LANES must be passed in."""
    assert state_kernels.LANES == 64
    assert "LANES:tl.constexpr" in _KERNEL_CODE
    assert "tl.arange(0,LANES)" in _KERNEL_CODE


def _device_ready_or_skip():
    if not state_kernels.triton_available():
        pytest.skip("a Triton compiler is not installed on this host")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("no Ascend NPU is available on this host")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_fused_path_device_gate_by_dtype(dtype):
    """Device contract, per dtype: BF16 must be bit-identical; FP32 is refused by name.

    The runtime calls this path with BF16 query/KV only. FP32 is not silently
    dropped from the contract: the NPU path raises a named ValueError before any
    allocation (a prior FP32 synthetic failed the strict gate at T=100, 952/4096
    raw output bits), and the CPU oracle still accepts FP32. For BF16 the gate is
    strict raw bits: `torch.equal` on floats treats -0.0/+0.0 as equal, so the
    comparison runs on the integer view of the raw bits; the finite check keeps
    infinities and NaNs out of a payload comparison.
    """
    _device_ready_or_skip()
    q, kv, indices, sink = make(heads=2, n=96, topk=100, dtype=dtype)
    q, kv, indices, sink = (value.npu() for value in (q, kv, indices, sink))
    if dtype is torch.float32:
        with pytest.raises(ValueError, match="supports BF16 q and kv only"):
            run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)
        return
    out = run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)
    ref = layout.sparse_attn_layout(
        q, kv, indices, sink, 0.75, backend="triton",
        certified_valid_indices=True,
        max_workspace_bytes=layout.layout_workspace_bytes(100, heads=2))
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert ref.dtype == torch.bfloat16
    assert torch.isfinite(out).all() and torch.isfinite(ref).all()
    out_bits = out.detach().cpu().contiguous().view(torch.int16)
    ref_bits = ref.detach().cpu().contiguous().view(torch.int16)
    if not torch.equal(out_bits, ref_bits):
        differing = int((out_bits != ref_bits).sum())
        raise AssertionError(
            f"S2 BF16 is not bit-identical to the accepted S1 control on this device "
            f"({differing} differing elements); reject the candidate rather than "
            f"loosening the comparison")


@pytest.mark.parametrize("q_dtype,kv_dtype",
                         [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)])
def test_the_fused_path_rejects_mixed_q_kv_dtypes(q_dtype, kv_dtype):
    """Mixed BF16/FP32 q-kv is outside the BF16-only NPU domain and refused by name."""
    _device_ready_or_skip()
    q, kv, indices, sink = make(heads=2, n=96, topk=100, dtype=q_dtype)
    kv = kv.to(kv_dtype)
    q, kv, indices, sink = (value.npu() for value in (q, kv, indices, sink))
    with pytest.raises(ValueError, match="supports BF16 q and kv only"):
        run(q, kv, indices, sink, backend="triton", certified_valid_indices=True)
