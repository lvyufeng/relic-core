"""Optional fused Ascend row normalization for the exact two-stage case.

``RMSNorm.forward`` casts to FP32, squares, takes the mean over the row, adds epsilon, takes rsqrt,
scales, applies the weight and casts back: eight dispatches, each microseconds, moving a few tens of
kilobytes -- the whole row is 5120 floats, so it fits in one program.

Two kernels replace the front half, and the rsqrt stays in the framework, because that combination
was measured to reproduce the eager chain exactly:

* the row sum of squares in **one block** matches the framework's sum; a tile-wise accumulation does
  not, so this kernel never splits a row across blocks;
* the mean is the sum **multiplied by the reciprocal of the length**, not the sum divided by it --
  those differ, and only the multiplication reproduces the framework's mean;
* the framework's ``rsqrt`` differs from a kernel ``rsqrt`` on these inputs in 7 rows of 40, which is
  enough to change the last BF16 bit, so it stays outside the kernel;
* the elementwise tail is plain multiplication, which is exact.

The claim this module supports is narrow and stated: with the rsqrt kept in the framework and the mean
formed by reciprocal multiplication, the two kernels reproduce the eager chain bit for bit. Measured
over four trials of 40 rows of 5120: 0 of 819200 BF16 elements differed. That is the measurement the
component tests re-make per shape rather than a property this source asserts.
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
    """Whether this interpreter can compile the fused row normalization."""
    return _USE_TRITON


if _USE_TRITON:

    @triton.jit
    def row_sqsum_kernel(X, OUT, K: tl.constexpr, KT: tl.constexpr):
        """Sum of squares for one row, accumulated in a single block (see the module doc)."""
        row = tl.program_id(0)
        lanes = tl.arange(0, KT)
        mask = lanes < K
        x = tl.load(X + row * K + lanes, mask=mask, other=0).to(tl.float32)
        tl.store(OUT + row, tl.sum(x * x, axis=0))

    @triton.jit
    def scale_kernel(X, WEIGHT, RSTD, OUT, K: tl.constexpr, KT: tl.constexpr,
                     WEIGHTED: tl.constexpr):
        """The elementwise tail: the exact product the eager expression forms, then one narrowing."""
        row = tl.program_id(0)
        lanes = tl.arange(0, KT)
        mask = lanes < K
        x = tl.load(X + row * K + lanes, mask=mask, other=0).to(tl.float32)
        scaled = x * tl.load(RSTD + row)
        if WEIGHTED:
            scaled = scaled * tl.load(WEIGHT + lanes, mask=mask, other=0).to(tl.float32)
        tl.store(OUT + row * K + lanes, scaled.to(OUT.dtype.element_ty), mask=mask)


def row_sqsum(x, out, *, dim: int):
    """Launch the single-block sum of squares; ``x`` is ``[rows, dim]`` and ``out`` is ``[rows]``."""
    if not _USE_TRITON:  # pragma: no cover - guarded by the caller
        raise RuntimeError("triton is not available on this host")
    rows = x.numel() // dim
    tile = max(triton.next_power_of_2(dim), 1024)
    row_sqsum_kernel[(rows,)](x, out, K=dim, KT=tile)
    return out


def scale(x, weight, rstd, out, *, dim: int, weighted: bool = True):
    """Launch the elementwise tail into ``out``, narrowing once on the way out."""
    if not _USE_TRITON:  # pragma: no cover - guarded by the caller
        raise RuntimeError("triton is not available on this host")
    rows = x.numel() // dim
    tile = max(triton.next_power_of_2(dim), 1024)
    scale_kernel[(rows,)](x, weight, rstd, out, K=dim, KT=tile, WEIGHTED=weighted)
    return out