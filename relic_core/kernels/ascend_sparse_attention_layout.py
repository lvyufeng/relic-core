"""Opt-in exact sparse-attention operand layout for the Ascend B1 decode path.

The reference path (``ops._sparse_attn_torch``, lines 1283..1363) gathers the
selected KV rows and casts them to FP32 ONCE, before the block loop (its
1337..1340); what is rebuilt inside the loop is only the per-block KV view/slice
and its transposed view. That per-block operand traffic is the measured cost on
the Ascend B1 decode profile (Transpose 512x64 ~2.26 ms, Slice ~1.51 ms, masked
fill ~1.09 ms rank-0, four forwards excluded), while the arithmetic itself is
only a handful of ``bmm`` calls.

This module replaces *only* that shuffling. One fused NPU gather/prepack writes
the two contiguous FP32 operand layouts the unchanged ``torch.bmm`` calls
consume. It is one kernel launch for the complete blocks and, only when the
top-k is not a multiple of 64, one extra exact-tail launch (never a padded
launch). The tile is an experimental 16x64 program: the first 32x128 prototype
overflowed the 192 KiB UB budget at Ascend compile time, so UB fit is claimed by
the root harness's compile, never by this source:

* ``straight[nfull, 64, 512]`` -- gathered KV rows in original order (P.V operand);
* ``transposed[nfull, 512, 64]`` -- the same values in QK operand order;
* an optional exact-size tail pair ``(tail, 512)`` / ``(512, tail)``.

The tail is the important part: the reference's last block reduces exactly the
true number of keys. The tail is therefore a separate exact-size allocation and a
separate ``bmm`` with ``N = tail``; the reduction is never padded to 64. A
``topk`` that is a multiple of 64 has no tail tensors at all.

What is unchanged, deliberately: every score/max/exp/FP32-denominator step, the
running max that starts at -1e30, the per-block BF16 narrowing of the numerator
(cast to BF16 and back before P.V), the sum taken on the FP32 numerator before
that narrowing, the attention sink joining after the loop, and the final FP32
division. The only intentional difference from the reference is that the two
``bmm`` operands arrive contiguous instead of as strided views/slices, so a BLAS
kernel may schedule differently while computing the same product; bitwise parity
is gated by the component tests, not assumed here.

Index contract, matching the reference exactly: only negative entries are
invalid (``valid = indices >= 0``; -1 and any other negative are gathered from
row 0 and then masked to -inf). The reference throws on a positive out-of-range
index; the torch oracle here still throws (fancy-index bounds check), while the
NPU kernel requires ``certified_valid_indices=True`` because checking on device
would need a device-to-host copy and would break graph capture. No dropping, no
sorting, no pruning, no hidden fallback: a failed NPU launch is an error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import _ascend_sparse_attention_layout_kernels as _kernels
from .ascend_unpack import _npu_launch

#: Reference block width; part of the arithmetic (the BF16 numerator narrows once
#: per block) and therefore fixed, not a tuning knob.
BLOCK = 64
#: Head dimension of the released sparse-attention path.
HEAD_DIM = 512
#: Largest top-k this candidate admits (compressed top-k is <= 288 in practice).
MAX_TOPK = 512
#: Largest stored KV row count this candidate admits.
MAX_KV_ROWS = 4096

_DTYPES = (torch.float32, torch.bfloat16)
_SINK_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_BACKENDS = ("torch", "triton")


@dataclass(frozen=True)
class SparseAttentionLayouts:
    """The two FP32 operand layouts plus the unchanged validity mask."""

    straight_full: torch.Tensor
    transposed_full: torch.Tensor
    straight_tail: "torch.Tensor | None"
    transposed_tail: "torch.Tensor | None"
    valid: torch.Tensor
    nfull: int
    tail: int


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


#: Allocation-slack allowance folded into the declared per-call estimate.
ALLOCATION_ALLOWANCE = 64 * 1024


def layout_workspace_bytes(topk, *, block=BLOCK, head_dim=HEAD_DIM, heads=1):
    """Declared per-call estimate of Python-visible tensor storage, in bytes.

    This is a *declared estimate* for the caller's ``max_workspace_bytes``, not a
    guaranteed upper bound and not a physical cap. It counts the storage this
    module and its torch reference visibly allocate:

    * both FP32 operand layouts (complete blocks plus the exact tail);
    * the larger of the torch oracle's gather/cast temporary (~2*T*D*4) and the
      steady-state attention tensors: the FP32 ``q`` copy plus four ``H*D``
      accumulators (old/scaled/P.V/next), the per-block score/numerator/cast
      scratch (8*H*block*4), several H-vectors, and the T-byte valid mask;
    * the final worst-case ``H*D`` FP32 output;
    * a fixed allocation-slack allowance.

    What it deliberately EXCLUDES: allocator rounding and fragmentation, graph
    memory pools (the Ascend graph pool can retain more), any backend BMM
    workspace peak, the Ascend UB/local memory the fused kernel uses, and any
    concurrent-stream overlap. A caller sizing a budget from this number should
    treat it as the declared per-call estimate the candidate promises, not as a
    measured absolute peak.
    """
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0 or topk > MAX_TOPK:
        raise ValueError(f"topk must be an integer within 0..{MAX_TOPK}")
    if block != BLOCK or head_dim != HEAD_DIM:
        raise ValueError("the sparse-attention layout is defined for the reference 64/512 geometry")
    heads = _positive_int(heads, "heads")
    nfull, tail = divmod(topk, block)
    layouts = 2 * (nfull * block + tail) * head_dim * 4
    # The torch oracle materializes the gathered/cast rows once; the fused path
    # does not, but the declared estimate must cover whichever backend runs.
    oracle_gather = 2 * topk * head_dim * 4
    # q_f32 copy + four H*D accumulators + per-block score/numerator/cast scratch
    # + a few H-vectors + the one-byte-per-index validity mask.
    steady_state = (5 * heads * head_dim * 4 + 8 * heads * block * 4
                    + 8 * heads * 4 + topk)
    final_output = heads * head_dim * 4
    return layouts + max(oracle_gather, steady_state) + final_output + ALLOCATION_ALLOWANCE


def _checked_inputs(q, kv, indices, sink, scale, *, backend, certified_valid_indices,
                    max_workspace_bytes):
    """Validate the B1 contract and return the normalized pieces; never launches."""
    if backend not in _BACKENDS:
        raise ValueError(f"sparse-attention layout backend must be one of {_BACKENDS}")
    if not isinstance(certified_valid_indices, bool):
        raise ValueError("certified_valid_indices must be an explicit boolean")
    if (isinstance(max_workspace_bytes, bool) or not isinstance(max_workspace_bytes, int)
            or max_workspace_bytes <= 0):
        raise ValueError("max_workspace_bytes must be an explicit positive integer")
    for name, value in (("q", q), ("kv", kv), ("indices", indices)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"sparse-attention layout requires tensor {name}")
    if sink is not None and not isinstance(sink, torch.Tensor):
        raise ValueError("sparse-attention layout requires tensor sink")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ValueError("softmax_scale must be a finite number")
    if q.dim() != 4 or q.shape[0] != 1 or q.shape[1] != 1 or q.shape[3] != HEAD_DIM:
        raise ValueError("sparse-attention layout requires B1 q of shape (1, 1, H, 512)")
    if q.dtype not in _DTYPES or q.stride(-1) != 1:
        raise ValueError("sparse-attention layout q must be FP32/BF16 with a unit last stride")
    heads = q.shape[2]
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ValueError("sparse-attention layout requires at least one head")
    if kv.dim() != 3 or kv.shape[0] != 1 or kv.shape[2] != HEAD_DIM:
        raise ValueError("sparse-attention layout requires kv of shape (1, N, 512)")
    if kv.dtype not in _DTYPES or not kv.is_contiguous():
        raise ValueError("sparse-attention layout kv must be contiguous FP32/BF16")
    if not 1 <= kv.shape[1] <= MAX_KV_ROWS:
        raise ValueError(f"sparse-attention layout requires 1 <= N <= {MAX_KV_ROWS} stored KV rows")
    if (indices.dim() != 3 or indices.shape[0] != 1 or indices.shape[1] != 1
            or indices.dtype != torch.int32 or not indices.is_contiguous()):
        raise ValueError("sparse-attention layout requires contiguous int32 indices of shape (1, 1, T)")
    topk = indices.shape[2]
    if isinstance(topk, bool) or not isinstance(topk, int) or not 0 <= topk <= MAX_TOPK:
        raise ValueError(f"sparse-attention layout requires 0 <= T <= {MAX_TOPK}")
    if sink is not None:
        if sink.dim() not in (1, 2) or (sink.dim() == 1 and sink.shape[0] != heads) \
                or (sink.dim() == 2 and tuple(sink.shape) != (1, heads)):
            raise ValueError("sparse-attention layout sink must be (H,) or (1, H)")
        if sink.dtype not in _SINK_DTYPES or not sink.is_contiguous():
            raise ValueError("sparse-attention layout sink must be contiguous FP32/BF16/FP16")
    for value in (kv, indices, sink):
        if value is not None and value.device != q.device:
            raise ValueError("sparse-attention layout requires every operand on one device")
    needed = layout_workspace_bytes(topk, head_dim=HEAD_DIM, heads=heads)
    if needed > max_workspace_bytes:
        raise ValueError(
            f"sparse-attention layout needs {needed} bytes of per-call scratch, "
            f"max_workspace_bytes is {max_workspace_bytes}")
    if backend == "triton":
        if q.device.type != "npu":
            raise ValueError("the fused sparse-attention layout requires an NPU device")
        if not certified_valid_indices:
            raise ValueError(
                "the fused sparse-attention layout requires certified_valid_indices=True; "
                "positive index bounds are checked without a device-to-host copy")
        if not _kernels.triton_available():
            raise RuntimeError("the fused sparse-attention layout needs a Triton Ascend compiler")
    return heads, topk


def build_sparse_attention_layouts(q, kv, indices, *, backend="torch",
                                   certified_valid_indices=False, max_workspace_bytes):
    """Build the two FP32 operand layouts (no attention arithmetic).

    This is the layout-only entry point the CPU oracle tests compare against a
    manual gather. It never writes ``q``/``kv``/``indices``; both layouts and the
    validity mask are fresh tensors.
    """
    heads, topk = _checked_inputs(q, kv, indices, None, 1.0, backend=backend,
                                  certified_valid_indices=certified_valid_indices,
                                  max_workspace_bytes=max_workspace_bytes)
    d = HEAD_DIM
    nfull, tail = divmod(topk, BLOCK)
    device = q.device
    if nfull:
        straight_full = torch.empty((nfull, BLOCK, d), dtype=torch.float32, device=device)
        transposed_full = torch.empty((nfull, d, BLOCK), dtype=torch.float32, device=device)
    else:
        straight_full = torch.empty((0, BLOCK, d), dtype=torch.float32, device=device)
        transposed_full = torch.empty((0, d, BLOCK), dtype=torch.float32, device=device)
    straight_tail = (torch.empty((tail, d), dtype=torch.float32, device=device) if tail else None)
    transposed_tail = (torch.empty((d, tail), dtype=torch.float32, device=device) if tail else None)
    if backend == "torch":
        _gather_layouts_torch(kv, indices, straight_full, transposed_full,
                              straight_tail, transposed_tail, nfull=nfull, tail=tail, topk=topk)
    else:
        # The launch must run under this tensor's device and the current stream;
        # `_npu_launch` switches the NPU device context and refuses a non-Ascend
        # Triton target. It performs no synchronization and is capture-safe.
        with _npu_launch(kv):
            _kernels.gather_layouts(kv, indices, straight_full, transposed_full,
                                    straight_tail, transposed_tail,
                                    n=kv.shape[1], nfull=nfull, tail=tail)
    valid = (indices >= 0).reshape(1, topk).contiguous()
    return SparseAttentionLayouts(straight_full, transposed_full, straight_tail,
                                  transposed_tail, valid, nfull, tail)


def _gather_layouts_torch(kv, indices, straight_full, transposed_full,
                          straight_tail, transposed_tail, *, nfull, tail, topk):
    """CPU oracle: the reference's own fancy-index gather, then exact block copies."""
    rows = 1
    d = HEAD_DIM
    safe = indices.clamp_min(0)
    batch = torch.arange(1, device=kv.device).view(1, 1, 1).expand_as(safe)
    gathered = kv[batch, safe].to(torch.float32).reshape(rows, topk, d)
    if nfull:
        straight_full.copy_(gathered[:, :nfull * BLOCK, :].reshape(nfull, BLOCK, d))
        transposed_full.copy_(straight_full.transpose(1, 2))
    if tail:
        straight_tail.copy_(gathered[:, nfull * BLOCK:, :].reshape(tail, d))
        transposed_tail.copy_(straight_tail.transpose(0, 1))


def sparse_attn_layout(q, kv, indices, sink, scale, *, backend="torch",
                       certified_valid_indices=False, max_workspace_bytes):
    """Sparse attention using the prebuilt layouts; identical arithmetic to the reference.

    Returns the same shape and dtype as ``q``. ``backend="torch"`` is the explicit
    CPU oracle (also usable on NPU for bitwise comparison); ``backend="triton"``
    fuses the gather/prepack on the NPU and then runs the unchanged torch.bmm loop.
    """
    _heads, topk = _checked_inputs(q, kv, indices, sink, scale, backend=backend,
                                   certified_valid_indices=certified_valid_indices,
                                   max_workspace_bytes=max_workspace_bytes)
    layouts = build_sparse_attention_layouts(
        q, kv, indices, backend=backend,
        certified_valid_indices=certified_valid_indices,
        max_workspace_bytes=max_workspace_bytes)
    b, s, h, d = q.shape
    rows = b * s
    if rows != 1:
        raise ValueError("the sparse-attention layout candidate is defined for the B1 decode row")
    q_f32 = q.to(torch.float32).reshape(rows, h, d)
    valid = layouts.valid
    scores_max = torch.full((rows, h), -1e30, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros((rows, h), device=q.device, dtype=torch.float32)
    acc_o = torch.zeros((rows, h, d), device=q.device, dtype=torch.float32)
    for start in range(0, topk, BLOCK):
        stop = min(start + BLOCK, topk)
        width = stop - start
        if width == BLOCK:
            kv_blk = layouts.straight_full[start // BLOCK]
            kv_t = layouts.transposed_full[start // BLOCK]
        else:
            kv_blk = layouts.straight_tail
            kv_t = layouts.transposed_tail
        # The reference's exact ``bmm(q_f32, kv_blk.transpose(1, 2))``, with the
        # transposed operand already materialized contiguously.
        scores = torch.bmm(q_f32, kv_t.unsqueeze(0)) * float(scale)
        scores = scores.masked_fill(~valid[:, None, start:stop], float("-inf"))
        max_prev = scores_max
        scores_max = torch.maximum(max_prev, scores.amax(dim=-1))
        scores_scale = torch.exp(max_prev - scores_max)
        numerator = torch.exp(scores - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * scores_scale + numerator.sum(dim=-1)
        # The per-block BF16 narrowing of the numerator is part of the reference's
        # arithmetic: cast to BF16 and back before the P.V product, as the kernel does.
        acc_o = acc_o * scores_scale.unsqueeze(-1) + torch.bmm(
            numerator.to(torch.bfloat16).to(torch.float32), kv_blk.unsqueeze(0))
    sum_exp = sum_exp + torch.exp(sink.to(torch.float32).view(1, h) - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).reshape(b, s, h, d).to(q.dtype)