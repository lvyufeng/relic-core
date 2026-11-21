"""Ascend kernel reading packed FP4 weights from their resident tile-major layout.

The frozen packed paths read the checkpoint's row-major weight bytes:
``[N, K // 2]`` one byte per adjacent K pair, with the K block a row's scale
covers being 16 consecutive bytes. Two lanes then share a byte and consecutive
tile rows are separate short bursts, so a tile's worth of weight bytes is
``ROWS_N`` disjoint 16-byte segments at a row stride of ``K // 2``.

This module reads the *same bytes* from a permuted resident layout instead:

    BT[tile, block, r, c]  -- uint8, the packed byte of row ``tile * 16 + r``
                              at K block ``block``, byte ``c`` in the block
    ST[tile, block, r]     -- uint8, the original E8M0 byte of that row/block

so one program's whole weight tile is one **contiguous** run of
``GROUP_K * 256`` bytes at ``(tile * KBLOCKS + group_base) * 256``, and its scale
tile is ``GROUP_K * 16`` contiguous bytes at ``(tile * KBLOCKS + group_base) * 16``.
The permutation is done once on the host when the bank is staged; it changes no
byte value and does not change the byte count for an ``N`` that is already a
multiple of 16, which every released expert shape is.

What is *not* changed: the arithmetic. The fragment, the packed-nibble select and
the exact doubled-E2M1 codebook are imported from the frozen modules, the
fragment tile is the same ``[GROUP_K * 16, 32]`` shape the frozen grouped kernel
already compiles on this backend, and pass 2 is the frozen
``decode_accumulate_kernel`` unchanged, so the strict increasing-K accumulator
chain is the accepted one by construction.

Layout deviation, stated explicitly
-----------------------------------

The core task's proposal wrote the resident shape as ``[K // 32, N // 16, 16, 16]``
(K-block outer) with a 256-byte tile load at ``(block * NTILES + tile) * 256``.
This module uses the **tile-outer** permutation ``[NTILES, K // 32, 16, 16]``
instead, and that is a deliberate deviation: only tile-outer makes the blocks of
one K group contiguous, which is what turns ``GROUP_K`` blocks into a single
``GROUP_K * 256``-byte load and one ``GROUP_K * 16``-byte scale load. Under
K-block-outer the same group is ``GROUP_K`` separate 256-byte loads strided by
``NTILES * 256``, and the scales are strided by ``NTILES * 16``. Both layouts
satisfy "one contiguous 256-byte tile per program"; this one additionally makes
the group contiguous. The arithmetic and the fragment set are identical either
way.

Access notes, measured on this backend rather than assumed:

* the tile is loaded as a **flat 1-D** run of ``GROUP_K * 256`` bytes and only
  then reshaped to ``[GROUP_K * 16, 16]``, so the tile addresses are
  ``arange``-contiguous with no per-row stride term;
* the raw tile stays one byte per element through the local nibble expansion, and
  the backend's gather **refuses uint8** (measured: "Expected dtype
  fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8, but got uint8"), so it crosses through a
  same-width ``int8`` bitcast -- the bits are untouched -- and the widen masks
  with ``0xff`` *after* the gather, which is also what restores the unsigned value
  the frozen nibble select needs on its odd lane;
* the tail of a ragged final group is masked on every load
  (``group_bytes < remaining * 256``, ``group_bytes < remaining * 16``,
  ``block < KBLOCKS``) and again on the store, which is the frozen grouped kernel's
  own idiom -- a masked lane forms no access and its zero ``other`` fragment never
  reaches the scratch.

Not a speed claim: no measurement of this layout exists yet. It is a memory-access
variant of unchanged arithmetic, and only an actual decode on the target build can
say whether burst shape was the cost.
"""
import triton
import triton.language as tl

# The fragment (reduction over both 16-term halves, the value decoders and both
# E8M0 rescales), the frozen nibble select and the exact doubled-E2M1 codebook are
# the accepted ones, imported rather than re-typed so this layout cannot drift from
# the accepted arithmetic.
from ._ascend_decode_grouped_kernels import (
    _e2m1_x2_lut,
    _fragment_rows,
    _packed_nibble,
)


@triton.jit
def tile_major_packed_fragment_kernel(
    A, SA, BT, ST, FRAG, BAD,
    N: tl.constexpr, KBLOCKS: tl.constexpr, INDEX64: tl.constexpr,
    GROUP_K: tl.constexpr, GROUP_STRIDE: tl.constexpr,
):
    """Write one K group's fragments for one 16-column tile per iteration.

    ``FRAG`` is ``[KBLOCKS, N]`` FP32 and ``BAD`` ``[KBLOCKS, N]`` uint8 -- the
    frozen prototype's scratch layout, so pass 2 is its kernel unchanged.

    ``BT``/``ST`` are the resident tile-major bank, ``[NTILES, KBLOCKS, 16, 16]``
    and ``[NTILES, KBLOCKS, 16]``; ``A``/``SA`` are the original activation codes
    ``[1, K]`` and E8M0 bytes ``[1, K // 32]``. Flat row ``j`` of the group tile is
    K-group block ``j >> 4`` and tile column ``j & 15``, which is exactly the
    ``[GROUP_K * 16, 16]`` order a contiguous ``GROUP_K * 256``-byte run reshapes
    to -- so no per-row index arithmetic is needed to line the rows up.

    Each program strides the K groups it was given, so a geometry whose grid would
    exceed the device cap still covers every block exactly once. Every ``(block,
    column)`` pair in range is written by exactly one program.
    """
    lanes = tl.arange(0, 32)
    # Every shape/`arange` argument must be a `tl.constexpr` expression, and a plain
    # local is not one: `tile_rows = GROUP_K * 16` then `tl.arange(0, tile_rows)`
    # failed the first real compile with "arange's arguments must be of type
    # tl.constexpr" *before* any arithmetic ran. `GROUP_K` is a constexpr kernel
    # argument, so `GROUP_K * 16` written in place is constexpr -- the same form the
    # frozen prefetch kernel uses for its `(TILE, 32)` shape tuples -- and that is why
    # there is deliberately no `tile_rows` binding here. The group tile's row count.
    flat = tl.arange(0, GROUP_K * 16)
    group_of_row = flat >> 4
    within = flat & 15

    # The tile-major bank: one program owns one 16-column tile.
    tile = tl.program_id(0)
    columns = tile * 16 + within
    column_valid = columns < N
    if INDEX64:
        # Widened before the `* KBLOCKS`, `* 256`, `* 16` and `* N` products below.
        tile = tile.to(tl.int64)
        columns = columns.to(tl.int64)

    group_bytes = tl.arange(0, GROUP_K * 256)
    group_scale_bytes = tl.arange(0, GROUP_K * 16)
    # Lane j of the expanded block reads byte j >> 1: one byte serves the two
    # nibbles that are its two K elements.
    pair_lanes = tl.arange(0, 32) >> 1
    byte_shift = lanes[None, :]

    for group_base in range(tl.program_id(1) * GROUP_K, KBLOCKS, GROUP_STRIDE * GROUP_K):
        if INDEX64:
            group_base = group_base.to(tl.int64)
        block = group_base + group_of_row
        # The frozen idiom for a ragged final group: mask the loads, never clamp an
        # index. A masked lane forms no access, so nothing outside the bank is read,
        # and `active` also keeps the zero `other` fragment out of the scratch.
        block_active = block < KBLOCKS
        active = column_valid & block_active
        remaining = KBLOCKS - group_base

        # One activation block per group, gathered onto the group's 16 rows. Only
        # GROUP_K * 32 distinct bytes exist behind this load; the duplication is
        # what saves a 3-D operand, which this backend has no proven reduction for.
        ac = tl.load(A + block[:, None] * 32 + lanes[None, :], mask=block_active[:, None],
                     other=0).to(tl.int32)
        scale_a = tl.load(SA + block, mask=block_active, other=0).to(tl.int32)

        # The whole group tile in one contiguous run of raw bytes.
        tile_base = (tile * KBLOCKS + group_base) * 256
        raw = tl.load(
            BT + tile_base + group_bytes,
            mask=group_bytes < remaining * 256,
            other=0,
        )
        # Both the reshape and the broadcast shape are written in place for the same
        # reason as the arange above: a shape argument has to be a constexpr
        # expression, and `GROUP_K` is one.
        pairs = tl.reshape(raw, (GROUP_K * 16, 16))
        gather_index = tl.broadcast_to(pair_lanes[None, :], (GROUP_K * 16, 32))
        gathered = (
            tl.gather(pairs.to(tl.int8, bitcast=True), gather_index, 1).to(tl.int32) & 255
        )
        bc = _e2m1_x2_lut(_packed_nibble(gathered, byte_shift))

        # The original E8M0 bytes of the same group rows, also contiguous.
        scale_base = (tile * KBLOCKS + group_base) * 16
        scale_b = tl.load(
            ST + scale_base + group_scale_bytes,
            mask=group_scale_bytes < remaining * 16,
            other=0,
        ).to(tl.int32)

        scaled, invalid = _fragment_rows(ac, bc, scale_a, scale_b, lanes, True)
        offsets = block * N + columns
        tl.store(FRAG + offsets, scaled, mask=active)
        tl.store(BAD + offsets, tl.where(invalid, 1, 0).to(tl.uint8), mask=active)