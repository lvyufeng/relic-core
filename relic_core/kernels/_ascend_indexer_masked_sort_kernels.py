"""Optional fused Ascend selection for the sparse indexer's mask-then-sort step.

The reference builds a small int array, masks the entries the query cannot reach,
sorts it ascending and casts to int32. On the Ascend build each of those steps
becomes an AI CPU operator, and an AI CPU operator occupies no cube or vector unit
while serialising against the stream: measured on the registered decode lane that
is 9 Greater + 9 Sort + 9 Cast + 4 Index = 31 operators a forward, 2762 us, 3.5%
of the step, almost all of it fixed dispatch cost rather than work.

This module replaces the sort with a rank. An entry's output position is the
number of reachable entries below it, so a reader scatters instead of sorting: one
pass per element with the whole row compared against it, which is O(width) a
program and needs no intra-block exchange. (This Triton build offers no tensor
slicing and no shuffles, so a bitonic sort in registers is not available either.)

Correctness rests on a property of the input rather than on the algorithm: the ids
are unique, and every tie is between sentinels that all collapse to the same
padding value, so the sorted result is uniquely determined and no tie-break rule
has to match ``torch.sort``'s.

Two padding conventions exist because the two call sites sort against different
sentinels: a sentinel past every real position leaves the padding *after* the real
ids, while the ``-1`` sentinel sorts below every real block id and leaves it
*before* them. Both are reproduced here.
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

#: Elements compared per iteration inside the rank kernel. Only affects register pressure.
COMPARE_BLOCK = 256


def triton_available() -> bool:
    """Whether this interpreter can compile the fused selection."""
    return _USE_TRITON


if _USE_TRITON:

    @triton.jit
    def rank_scatter_kernel(VALUES, KEYS, OUT, WIDTH: tl.constexpr, OFFSET: tl.constexpr,
                            PAD_FIRST: tl.constexpr, COMPARE: tl.constexpr):
        """One program per row element.

        ``OUT`` must already hold the padding value in every slot; this kernel only
        writes the reachable ids into their ranked slots, so the padding positions
        keep what the caller filled in (the tail of the ranking that no reachable
        entry claims).

        The rank is the number of reachable entries strictly below this one, plus -
        when the padding sorts first, as at the ``-1``-sentinel site - the number of
        unreachable entries, which all sit ahead of every real id.
        """
        index = tl.program_id(0)
        row = index // WIDTH
        lane = index % WIDTH
        base = row * WIDTH
        value = tl.load(VALUES + base + lane)
        reachable = value > float('-inf')
        ident = (tl.load(KEYS + base + lane) + OFFSET).to(tl.int32)
        # The reference sorts the *ids*, not the scores: it builds
        # `where(reachable, keys + offset, tail)` and sorts that, so an entry's rank
        # is the number of reachable ids below it.
        below = tl.zeros((1,), dtype=tl.int32)
        padding = tl.zeros((1,), dtype=tl.int32)
        for start in tl.static_range(0, WIDTH, COMPARE):
            offset = tl.arange(0, COMPARE)
            other_lane = start + offset
            live = other_lane < WIDTH
            other_value = tl.load(VALUES + base + other_lane, live, other=float('-inf'))
            other_reachable = live & (other_value > float('-inf'))
            other_ident = (tl.load(KEYS + base + other_lane, live, other=0) + OFFSET).to(tl.int32)
            below += tl.sum((other_reachable & (other_ident < ident)).to(tl.int32), axis=0)
            padding += tl.sum((live & ~other_reachable).to(tl.int32), axis=0)
        position = below
        if PAD_FIRST:
            position = position + padding
        tl.store(OUT + base + position, ident, mask=reachable)


def rank_scatter(values, keys, out, *, offset: int, pad_first: bool):
    """Launch the rank-and-scatter kernel; ``out`` must be pre-filled with the padding value."""
    if not _USE_TRITON:  # pragma: no cover - guarded by the caller
        raise RuntimeError("triton is not available on this host")
    width = values.shape[-1]
    rows = values.numel() // width
    rank_scatter_kernel[(rows * width,)](
        values.contiguous(), keys.contiguous(), out, width, int(offset), pad_first,
        COMPARE=COMPARE_BLOCK,
    )
    return out