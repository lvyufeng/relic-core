"""Optional fused Ascend build for the sliding-window top-k index table.

The reference builds a decode step's table with eight launches -- an ``arange``, an add, a modulo, a
compare, a select, a cast, a broadcast and a materializing copy -- for a table of 128 int32 values,
512 bytes, once a layer. The launches are most of what the table costs.

The values are integer arithmetic, so one kernel produces them with no floating-point question:
``(lane + oldest) % window_size``, then ``-1`` where the slot is ahead of the position, cast to int32
-- the same order and the same widths the eager expression uses. The modulo has exactly one
alternative because both operands are bounded: ``lane`` is below ``window_size`` and ``oldest`` is at
most ``window_size - 1``, so the sum is below ``2 * window_size`` and subtracting ``window_size`` once
is the same result without a division. Every value fits int32, so nothing travels on emulated 64-bit
lanes, and no width change can alter what the values are.

The broadcast and its copy also disappear, because the kernel writes a real row and the batch
dimension is one.
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
    """Whether this interpreter can compile the fused table build."""
    return _USE_TRITON


if _USE_TRITON:

    @triton.jit
    def window_idxs_kernel(OUT, SLOT, ROW, WINDOW: tl.constexpr, BLOCK: tl.constexpr):
        """Build the ring's decode table in one program, without int64 lanes and without a modulo."""
        lane = tl.arange(0, BLOCK)
        mask = lane < WINDOW
        oldest = tl.load(SLOT).to(tl.int32) + 1
        row = tl.load(ROW).to(tl.int32)
        total = lane.to(tl.int32) + oldest
        index = tl.where(total >= WINDOW, total - WINDOW, total)
        index = tl.where(index > row, -1, index)
        tl.store(OUT + lane, index, mask=mask)


def build_table(out, slot, row, *, window_size: int):
    """Launch the table build into ``out``; ``slot`` and ``row`` are single-element int32 tensors."""
    if not _USE_TRITON:  # pragma: no cover - guarded by the caller
        raise RuntimeError("triton is not available on this host")
    tile = max(triton.next_power_of_2(window_size), 32)
    window_idxs_kernel[(1,)](out, slot, row, WINDOW=window_size, BLOCK=tile)
    return out