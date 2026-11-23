"""Optional fused sparse-attention softmax-state kernels for the Ascend NPU (S2 prototype).

This module owns only the two tiny per-block state kernels. The QK and P.V products
stay native ``torch.bmm`` on the accepted S1 KV layouts; the exact tail width stays
a separate ``bmm`` with N = tail. What is fused here is the sixteen-or-so tiny
ATen calls the reference runs per 64-key block:

* ``prepare``: ``scores * softmax_scale``, the invalid-lane ``-inf`` mask, the
  running max (``maximum(max_prev, block_max)``), ``exp(max_prev - new_max)``, the
  FP32 numerator ``exp(scores - new_max)`` written raw to device, and the
  per-element BF16 rounding of the numerator that the P.V operand needs. The
  denominator sum is NOT in the kernel (S2b): it stays ``torch.sum(-1)`` on the raw
  buffer in the caller. The old fused denominator stage differed from the reference
  by one FP32 element (4.768371582e-07) in the real B1 gate; that difference did not
  isolate the TL exp lower bits from the reduction, so the reference's ATen
  reduction tree is restored as a hypothesis under test, not a proven TLSum cause.
  All per-head state lives on device and is updated in place, read-before write.
* ``post_pv``: ``acc = acc * scores_scale + pv`` for the 512-wide accumulator.

The launch option ``enable_fp_fusion=False`` is deliberate: the reference multiplies
then adds as two FP32 roundings, not as an FMA, and this repository already uses the
same option on its Ascend Triton kernels. Whether the compiler honours it is a
property the root harness's device tests have to measure; this source states the
intent, not a proof.

Numerical risk is real and is not papered over: ``tl.exp`` is not guaranteed to
round identically to CANN's ``torch.exp``. In the real B1 gate the old fused
denominator stage differed from the reference by one FP32 element (4.768371582e-07);
that difference did not isolate the TL exp lower bits from the in-kernel reduction,
so the reference's ATen reduction tree is restored in torch as a hypothesis under
test, and the raw numerator is left on device so the exponentials can be measured
separately;
the exponentials themselves are still ``tl.exp``, so acceptance remains gated by the
bit-identical same-NPU comparison against the accepted S1 control. A deviation is a
rejection, not extra headroom.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except Exception:  # pragma: no cover - the CPU-only host has no Triton
    triton = None
    tl = None
    _USE_TRITON = False

#: Reference block width, fixed by the accepted S1 layout.
BLOCK = 64
#: Head dimension of the released sparse-attention path.
HEAD_DIM = 512
#: Post-P.V program tile along the head dimension.
TILE_D = 128
#: One reference block has 64 lanes, so the prepare reduction is always 64 wide.
#: It is passed to the kernel as a ``tl.constexpr`` parameter, not read as a global:
#: the Ascend backend rejects a module global inside ``tl.arange``.
LANES = 64


if _USE_TRITON:

    @triton.jit
    def _state_prepare_kernel(
        SCORES, VALID, OLD_MAX, SCALE_OUT, OUT_RAW, OUT_P,
        SCALAR_SCALE,
        WIDTH: tl.constexpr,
        LANES: tl.constexpr,
    ):
        """Fused per-block softmax state for one head, one 64-key block.

        ``SCORES`` is the native QK ``bmm`` result for this head, ``WIDTH`` is the
        block's true width (64 for complete blocks, the exact tail otherwise).
        Lanes past ``WIDTH`` are masked out and contribute ``-inf`` to the max, so
        the reduction stays 64 wide without reading past the operand.

        The kernel writes the raw FP32 numerator to ``OUT_RAW``; the DENOMINATOR is
        deliberately NOT computed here. The old fused denominator stage differed
        from the reference by one FP32 element (4.768371582e-07) in the real B1
        gate; that difference did not isolate the TL exp lower bits from the
        in-kernel reduction, so the reference's ATen reduction tree is restored in
        torch as a hypothesis under test, with the reference's own expression and
        shape, and the raw numerator is left on device to be measured separately.
        ``OLD_MAX`` is read first and written last, so the running max is updated in
        place.
        """
        head = tl.program_id(0)
        lanes = tl.arange(0, LANES)  # constexpr parameter: Ascend rejects module globals here
        lane_mask = lanes < WIDTH
        scores = tl.load(SCORES + head * WIDTH + lanes, mask=lane_mask, other=0.0)
        valid = tl.load(VALID + lanes, mask=lane_mask, other=0)
        # The reference multiplies the bmm result by the float32 scale BEFORE the
        # mask; this is the same two-step FP32 multiply-then-select.
        scores = scores * SCALAR_SCALE
        scores = tl.where(valid != 0, scores, float("-inf"))
        old_max = tl.load(OLD_MAX + head)
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(old_max, block_max)
        scores_scale = tl.exp(old_max - new_max)
        numerator = tl.exp(scores - new_max)
        narrowed = numerator.to(tl.bfloat16).to(tl.float32)
        tl.store(SCALE_OUT + head, scores_scale)
        tl.store(OLD_MAX + head, new_max)
        # Raw FP32 numerator first, then the BF16-rounded P.V operand: the caller
        # sums the raw buffer with torch.sum(-1) using the same reference expression.
        # That restores the ATen reduction tree as the hypothesis under test; bitwise
        # acceptance is still gated by the device comparison, because the raw
        # exponentials come from tl.exp.
        tl.store(OUT_RAW + head * WIDTH + lanes, numerator, mask=lane_mask)
        tl.store(OUT_P + head * WIDTH + lanes, narrowed, mask=lane_mask)

    @triton.jit
    def _state_post_pv_kernel(
        ACC, PV, SCALE,
        DIM: tl.constexpr, TILE_D: tl.constexpr,
    ):
        """``acc = acc * scores_scale + pv`` for one head and one D tile."""
        head = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * TILE_D + tl.arange(0, TILE_D)
        mask = offs < DIM
        scores_scale = tl.load(SCALE + head)
        acc = tl.load(ACC + head * DIM + offs, mask=mask, other=0.0)
        pv = tl.load(PV + head * DIM + offs, mask=mask, other=0.0)
        # Two roundings (multiply, then add), matching the reference. The launch
        # passes enable_fp_fusion=False as the intended FMA control; whether the
        # compiler honours it is witnessed only by the root's reference post-PV
        # check, not proved in general by this source.
        tl.store(ACC + head * DIM + offs, acc * scores_scale + pv, mask=mask)


def triton_available() -> bool:
    """Report whether this host exposes a Triton compiler for the fused state."""
    return _USE_TRITON


def prepare_state(scores, valid, old_max, scale_out, raw_out, p_out, scalar_scale, width):
    """Launch the prepare kernel; all buffers are written in place.

    The kernel updates the running max and scale and writes the raw FP32 numerator
    (``raw_out``) plus the BF16-rounded P.V operand (``p_out``). It does not reduce:
    the denominator update stays in torch, like the reference. No device to host
    copy — ``width`` is a host integer from the block loop, so it can be a
    compile-time constant without a synchronization.
    """
    if not _USE_TRITON:
        raise RuntimeError("the fused sparse-attention state needs a Triton Ascend compiler")
    heads = old_max.numel()
    _state_prepare_kernel[(heads,)](
        scores, valid, old_max, scale_out, raw_out, p_out,
        float(scalar_scale),
        WIDTH=width, LANES=LANES, enable_fp_fusion=False,
    )


def post_pv(acc, pv, scores_scale):
    """Launch the fused post-P.V rescale/add kernel; ``acc`` is updated in place."""
    if not _USE_TRITON:
        raise RuntimeError("the fused sparse-attention state needs a Triton Ascend compiler")
    heads = scores_scale.numel()
    grid = (heads, triton.cdiv(HEAD_DIM, TILE_D))
    _state_post_pv_kernel[grid](
        acc, pv, scores_scale,
        DIM=HEAD_DIM, TILE_D=TILE_D, enable_fp_fusion=False,
    )