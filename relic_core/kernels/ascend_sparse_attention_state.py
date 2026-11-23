"""Fused sparse-attention softmax state (S2 prototype) for the Ascend B1 decode path.

Candidate S2 keeps the two boundaries the accepted S1 candidate already fixed — the
QK and P.V products stay native ``torch.bmm`` over the S1 FP32 KV layouts, and the
exact tail keeps its own ``bmm`` with ``N = tail`` (never padded to 64) — and fuses
only the tiny per-block bookkeeping the reference runs around them. In the
reference that is about a dozen ATen calls per 64-key block: scale multiply, mask,
block max, running max, rescale exp, numerator exp, denominator sum, BF16 narrowing
of the P.V operand, accumulator rescale and add. S2 replaces them with two Triton
kernels per block:

* ``_ascend_sparse_attention_state_kernels.prepare_state`` — score scale, invalid
  ``-inf`` mask, block max, ``maximum(max_prev, new_max)``, ``exp(max_prev -
  new_max)``, raw FP32 numerator written to device, and the BF16-rounded P.V
  operand. The denominator is deliberately outside the kernel (S2b): the old
  fused denominator was measured to change one FP32 denominator element
  (``4.768371582e-07``) relative to the accepted control, so the caller restores
  the reference's ATen denominator tree — ``torch.sum(-1)`` on the raw FP32
  numerator, then multiply and add in the reference's own shape. That
  attribution covered only the denominator delta: it did not isolate the raw FP32
  ``tl.exp`` from the in-kernel reduction, so S2b retains the reference's ATen
  reduction tree and leaves the raw numerator to be measured separately; the
  numerator's ``tl.exp`` remains unproven;
* ``_ascend_sparse_attention_state_kernels.post_pv`` — ``acc = acc * scale + pv``
  over the 512-wide accumulator.

The attention sink join and the final division stay in torch, in the reference's own
order, so no new division rounding is introduced. The accepted numerator ordering is
preserved: the denominator sums the FP32 numerator, and only then is the P.V operand
rounded to BF16 and widened back to FP32.

Numerical contract: no new headroom is claimed. The first real BF16 B1 gate
rejected S2 with 1 of 4096 BF16 output elements different (max abs
``0.0009765625``), traceable in that run to one changed FP32 denominator element
(``4.768371582e-07``). The old fused denominator differs from the reference; that
attribution did not isolate the raw FP32 ``tl.exp`` from the in-kernel reduction.
S2b retains the reference's ATen reduction tree (the same expression: ``sum_exp *
scores_scale + torch.sum(raw_numerator, -1)``, raw FP32, before any narrowing) and
the raw numerator is measured separately. S2b is accepted only if its complete
output is bit-identical to the accepted S1 control on the same NPU; bitwise
acceptance is still gated by that device comparison, not claimed by this source. A
deviation is a rejection, not a looser tolerance.

NPU domain (SPEC specialization): the fused path is BF16-only for both ``q`` and
``kv``, because that is the runtime's actual domain — every real B1 decode call
passes BF16 query and BF16 KV, and the model never feeds FP32. FP32 ``q``/``kv`` is
therefore refused by name before any allocation on the ``triton`` backend rather
than silently accepted or silently dropped. That refusal is not a quality waiver:
a prior FP32 synthetic run was rejected by the strict bit gate at T=100 (952/4096
raw output bits differed from the accepted S1 control), so FP32 is an unsupported
path here, not a looser-quality one. The CPU oracle (``backend="torch"``) still
accepts both BF16 and FP32 and reproduces the reference arithmetic from the S1
layouts, so the composed arithmetic cannot drift silently from the accepted
``sparse_attn_layout``.
"""
from __future__ import annotations

import torch

from . import _ascend_sparse_attention_state_kernels as _state_kernels
from . import ascend_sparse_attention_layout as _layout
from .ascend_unpack import _npu_launch

#: Accepted S1 geometry, re-exported so callers import one module.
BLOCK = _layout.BLOCK
HEAD_DIM = _layout.HEAD_DIM
MAX_TOPK = _layout.MAX_TOPK
MAX_KV_ROWS = _layout.MAX_KV_ROWS


def state_workspace_bytes(topk, *, heads=1):
    """Declared per-call tensor estimate for the S2 state path, in bytes.

    It is the accepted S1 layout estimate plus the tensors this path adds visibly:
    the BF16-rounded P.V operand for the full block and the exact tail, the QK score
    rows, the P.V result rows, the per-head state/scale buffers and the one-byte
    validity mask. Like the S1 helper this is a declared estimate, not a hard cap:
    allocator rounding, graph pools, backend BMM workspace and the kernel's UB/local
    memory are excluded, and the caller's ``max_workspace_bytes`` records that
    declaration rather than a fixed physical limit.
    """
    heads = _layout._positive_int(heads, "heads")
    layout_bytes = _layout.layout_workspace_bytes(topk, heads=heads)
    p_full = heads * BLOCK * 4
    p_tail = heads * (topk % BLOCK) * 4
    raw_full = heads * BLOCK * 4
    raw_tail = heads * (topk % BLOCK) * 4
    scores = heads * BLOCK * 4
    pv = heads * HEAD_DIM * 4
    state = 3 * heads * 4
    valid = topk
    return (layout_bytes + p_full + p_tail + raw_full + raw_tail
            + scores + pv + state + valid)


def _checked(q, kv, indices, sink, scale, *, backend, certified_valid_indices,
             max_workspace_bytes):
    """Run the accepted S1 validation, then enforce the S2 declared estimate."""
    heads, topk = _layout._checked_inputs(
        q, kv, indices, sink, scale, backend=backend,
        certified_valid_indices=certified_valid_indices,
        max_workspace_bytes=max_workspace_bytes)
    needed = state_workspace_bytes(topk, heads=heads)
    if needed > max_workspace_bytes:
        raise ValueError(
            f"sparse-attention state needs {needed} bytes of per-call scratch, "
            f"max_workspace_bytes is {max_workspace_bytes}")
    return heads, topk


def sparse_attn_state(q, kv, indices, sink, scale, *, backend="torch",
                      certified_valid_indices=False, max_workspace_bytes):
    """Sparse attention via the S1 layouts with the fused S2 state kernels.

    Returns the same shape and dtype as ``q``. ``backend="torch"`` is the explicit
    CPU oracle over the S1 layouts and accepts BF16 or FP32. ``backend="triton"`` is
    the NPU path: BF16 ``q`` and BF16 ``kv`` only (the runtime's actual domain), and
    requires ``certified_valid_indices=True`` (no device to host bound check, so
    graph capture stays intact). Any other dtype pair is refused by name on the NPU
    path; the CPU oracle remains available for FP32 comparisons.
    """
    heads, topk = _checked(q, kv, indices, sink, scale, backend=backend,
                           certified_valid_indices=certified_valid_indices,
                           max_workspace_bytes=max_workspace_bytes)
    if backend == "triton" and (q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16):
        # BF16-only NPU domain: the runtime's real calls are BF16 q/KV. Refuse by
        # name before any allocation instead of computing an unaudited FP32 path;
        # a prior FP32 synthetic failed the strict gate at T=100 (952/4096 bits).
        raise ValueError(
            "sparse_attn_state backend='triton' supports BF16 q and kv only; got "
            f"q={q.dtype}, kv={kv.dtype}. FP32 q/kv is unsupported on the NPU path "
            "(previously failed bit parity at T=100: 952/4096 output bits); use "
            "backend='torch' for the FP32 oracle.")
    layouts = _layout.build_sparse_attention_layouts(
        q, kv, indices, backend=backend,
        certified_valid_indices=certified_valid_indices,
        max_workspace_bytes=max_workspace_bytes)
    b, s, h, d = q.shape
    rows = b * s
    if rows != 1:
        raise ValueError("the sparse-attention state candidate is defined for the B1 decode row")
    q_f32 = q.to(torch.float32).reshape(rows, h, d)
    valid = layouts.valid
    scores_max = torch.full((rows, h), -1e30, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros((rows, h), device=q.device, dtype=torch.float32)
    acc_o = torch.zeros((rows, h, d), device=q.device, dtype=torch.float32)

    if backend == "torch":
        for start in range(0, topk, BLOCK):
            stop = min(start + BLOCK, topk)
            width = stop - start
            if width == BLOCK:
                kv_blk = layouts.straight_full[start // BLOCK]
                kv_t = layouts.transposed_full[start // BLOCK]
            else:
                kv_blk = layouts.straight_tail
                kv_t = layouts.transposed_tail
            scores = torch.bmm(q_f32, kv_t.unsqueeze(0)) * float(scale)
            scores = scores.masked_fill(~valid[:, None, start:stop], float("-inf"))
            max_prev = scores_max
            scores_max = torch.maximum(max_prev, scores.amax(dim=-1))
            scores_scale = torch.exp(max_prev - scores_max)
            numerator = torch.exp(scores - scores_max.unsqueeze(-1))
            sum_exp = sum_exp * scores_scale + numerator.sum(dim=-1)
            acc_o = acc_o * scores_scale.unsqueeze(-1) + torch.bmm(
                numerator.to(torch.bfloat16).to(torch.float32), kv_blk.unsqueeze(0))
    else:
        valid_u8 = valid.reshape(1, topk).to(torch.uint8)
        max_buf = scores_max.view(h)
        scale_buf = torch.empty(h, device=q.device, dtype=torch.float32)
        p_full = torch.empty((h, BLOCK), device=q.device, dtype=torch.float32)
        raw_full = torch.empty((h, BLOCK), device=q.device, dtype=torch.float32)
        tail = topk % BLOCK
        p_tail = (torch.empty((h, tail), device=q.device, dtype=torch.float32)
                  if tail else None)
        raw_tail = (torch.empty((h, tail), device=q.device, dtype=torch.float32)
                    if tail else None)
        acc_view = acc_o.view(h, d)
        # One device context for the whole loop: the current NPU device and stream
        # stay pinned, and none of this needs a device-to-host copy.
        with _npu_launch(kv):
            for start in range(0, topk, BLOCK):
                stop = min(start + BLOCK, topk)
                width = stop - start
                if width == BLOCK:
                    kv_blk = layouts.straight_full[start // BLOCK]
                    kv_t = layouts.transposed_full[start // BLOCK]
                    p_out, raw_out = p_full, raw_full
                else:
                    kv_blk = layouts.straight_tail
                    kv_t = layouts.transposed_tail
                    p_out, raw_out = p_tail, raw_tail
                # Native QK bmm exactly as accepted S1; the scale multiply is fused
                # into prepare, where the reference applies it before the mask.
                scores = torch.bmm(q_f32, kv_t.unsqueeze(0))
                _state_kernels.prepare_state(
                    scores.view(h, width), valid_u8[0, start:stop],
                    max_buf, scale_buf, raw_out, p_out, float(scale), width)
                # Reference denominator expression, unchanged and in torch. It
                # consumes the RAW FP32 numerator (shape (1, H, width), contiguous),
                # not the BF16-rounded P operand: the same arithmetic the accepted
                # S1 path applies before any narrowing.
                sum_exp = sum_exp * scale_buf.view(1, h) + raw_out.view(1, h, width).sum(dim=-1)
                # Native P.V bmm over the exact width, with the BF16-rounded-then-
                # widened numerator the reference feeds it.
                pv = torch.bmm(p_out.view(1, h, width), kv_blk.unsqueeze(0))
                _state_kernels.post_pv(acc_view, pv.view(h, d), scale_buf)

    # Sink join and final division stay in torch, in the reference's order.
    sum_exp = sum_exp + torch.exp(sink.to(torch.float32).view(1, h) - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).reshape(b, s, h, d).to(q.dtype)