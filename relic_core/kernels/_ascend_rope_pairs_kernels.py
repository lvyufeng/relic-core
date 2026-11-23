"""Optional fused Ascend pair rotation for the single-position decode table.

The reference rotates adjacent elements of ``x`` as one complex number -- ``even * cos - odd * sin``,
``even * sin + odd * cos``, with ``inverse`` negating the sine -- computes in FP32 and writes the
narrowed result back into ``x``'s own storage through strided ``[..., 0::2]`` / ``[..., 1::2]`` views.

One kernel performs the same four products and two sums in the same order, writing into the same
storage.

The kernel compiles at the widths this model actually rotates (head dimension 512). It is not claimed
for arbitrary widths: the Ascend compiler rejects some shapes with ``'memref.subview' op operation
destroyed but still has uses``, which is a property of the compiler rather than of this source, and
the only proof for another width is the harness's own compile. Nothing moves: the table, the angles and the arithmetic order are the reference's. The fused
module's own measurement was bit-identical against the eager body on all three shapes the model uses
(query, key and the inverse on the attention output), and the component tests re-make that comparison
per shape rather than trusting this comment.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except Exception:  # pragma: no cover - a CPU-only host has no Triton
    triton = None
    tl = None
    _USE_TRITON = False


def triton_available() -> bool:
    """Whether this interpreter can compile the fused pair rotation."""
    return _USE_TRITON


if _USE_TRITON:

    @triton.jit
    def rope_pairs_kernel(X, COS, SIN, PAIRS: tl.constexpr, ROW_STRIDE,
                          ROWS, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
        """Rotate each pair in place.

        A lane only writes the two elements it read, so several lanes of one row cannot collide and
        the in-place contract holds without a scratch buffer.
        """
        row = tl.program_id(0)
        if row < ROWS:
            lane = tl.arange(0, BLOCK)
            mask = lane < PAIRS
            base = row * ROW_STRIDE
            even = tl.load(X + base + 2 * lane, mask=mask, other=0.0).to(tl.float32)
            odd = tl.load(X + base + 2 * lane + 1, mask=mask, other=0.0).to(tl.float32)
            cosine = tl.load(COS + lane, mask=mask, other=0.0)
            sine = tl.load(SIN + lane, mask=mask, other=0.0)
            if INVERSE:
                sine = -sine
            new_even = even * cosine - odd * sine
            new_odd = even * sine + odd * cosine
            tl.store(X + base + 2 * lane, new_even.to(X.dtype.element_ty), mask=mask)
            tl.store(X + base + 2 * lane + 1, new_odd.to(X.dtype.element_ty), mask=mask)


def rotate_in_place(x, cosine, sine, *, row_stride: int, inverse: bool, pairs: int):
    """Launch the rotation over ``x``'s rows; ``cosine``/``sine`` are contiguous ``[pairs]`` tables."""
    if not _USE_TRITON:  # pragma: no cover - guarded by the caller
        raise RuntimeError("triton is not available on this host")
    rows = x.numel() // max(row_stride, 1)
    tile = max(triton.next_power_of_2(pairs), 32)
    rope_pairs_kernel[(rows,)](x, cosine, sine, PAIRS=pairs, ROW_STRIDE=row_stride,
                               ROWS=rows, INVERSE=inverse, BLOCK=tile)
    return x