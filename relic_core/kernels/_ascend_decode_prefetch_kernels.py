"""Ascend kernels for the prefetch-access grouped decode block GEMV.

The frozen grouped kernel (:mod:`_ascend_decode_grouped_kernels`) reads the packed
FP4 weight one *fragment* at a time: a tile row loads the 16 bytes of a single K
block, and two lanes address the same byte (``lanes >> 1``). On the target build
that measured 18.62 ms at the released ``2304 x 5120`` shape against 12.63 ms for
the identical arithmetic over an ``INT8 x 2`` operand, while a decode graph of the
same kernel saved only about 0.2 ms -- so the difference is device work, not host
launch. One candidate is the access shape: 16 consecutive bytes per row that two
lanes share is a half-used 32-byte burst, and consecutive tile rows are separate
short bursts.

This module changes **only where the weight bytes are read**:

* the tile is loaded **column-major**: one contiguous row of ``GROUP_K * 16`` raw
  bytes per tile column (64 bytes at ``GROUP_K = 4``, two full 32-byte bursts), so
  one load covers the whole K group instead of ``GROUP_K`` short ones;
* the raw tile stays one byte per element from the load through the local
  expansion and is widened only **after** the gather -- widening first would
  expand the raw tile four times in the unified buffer, which is already the
  binding constraint. The backend's gather accepts ``int8`` and **refuses
  ``uint8``** (measured: "Expected dtype fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8,
  but got uint8"), so the tile crosses it through a same-width bitcast -- the
  bytes' bits are untouched -- and the widen masks with ``0xff`` to recover the
  unsigned value;
* the two nibbles of one byte are produced by a **local gather** of the 16 bytes
  into 32 lanes (lane ``j`` reads byte ``j >> 1``) and then the frozen
  :func:`_packed_nibble` constant-shift select, so no per-lane variable shift
  remains.

The arithmetic is not re-typed: the fragment, the packed decode and both rescales
are imported from the frozen modules, and pass 2 is the frozen
``decode_accumulate_kernel`` unchanged, so the strict increasing-K accumulation is
the accepted one by construction.

Layout contract: the wide tile is ``[ROWS_N, GROUP_K * 16]``, and its reshape to
``[TILE, 16]`` orders flat rows as ``column * GROUP_K + group``. Every index here
therefore uses the column-major mapping

    column = program_id(0) * ROWS_N + (flat >> GROUP_SHIFT)
    block  = group_base + (flat & (GROUP_K - 1))

which covers the same fragment set as the grouped kernel, only permuted, so pass 2
still sees one ``[K // 32, N]`` scratch.

Not a speed claim: no measurement of this schedule exists yet. It is a
memory-access variant of unchanged arithmetic, and only an actual decode on the
target build can say whether the burst shape was the cost.
"""
import triton
import triton.language as tl

# The fragment (reduction, value decoders, both rescales), the packed-nibble
# select and the exact doubled-E2M1 codebook are the frozen ones, imported rather
# than re-typed so this schedule cannot drift from the accepted arithmetic.
from ._ascend_block_gemm_kernels import _original_scale_code
from ._ascend_decode_grouped_kernels import (
    _e2m1_x2_lut,
    _fragment_rows,
    _packed_nibble,
)


@triton.jit
def decode_prefetch_fragment_kernel(
    A, SA, B, SB, FRAG, BAD,
    N: tl.constexpr, K: tl.constexpr, KBLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr, PACKED: tl.constexpr,
    B_SCALE_FLOAT: tl.constexpr,
    ROWS_N: tl.constexpr, GROUP_SHIFT: tl.constexpr, GROUP_K: tl.constexpr,
    TILE: tl.constexpr, GROUP_STRIDE: tl.constexpr, INDEX64: tl.constexpr,
):
    """Write ``GROUP_K`` blocks x ``ROWS_N`` columns of fragments per iteration.

    ``FRAG`` is ``[KBLOCKS, N]`` FP32 and ``BAD`` ``[KBLOCKS, N]`` uint8, the
    frozen prototype's layout, so pass 2 is its kernel unchanged.

    The weight tile is fetched as one wide row per column (see the module
    docstring); flat row ``i`` then owns column ``i >> GROUP_SHIFT`` and block
    ``group_base + (i & (GROUP_K - 1))``. Each program strides the remaining K
    groups, so a geometry whose grid would exceed the device cap still covers
    every block exactly once.
    """
    lanes = tl.arange(0, 32)
    flat = tl.arange(0, TILE)
    group_in_tile = flat & (GROUP_K - 1)
    within = flat >> GROUP_SHIFT

    columns = tl.program_id(0) * ROWS_N + within
    column_active = columns < N
    # One entry per tile column: the row axis of the wide weight tile.
    tile_columns = tl.program_id(0) * ROWS_N + tl.arange(0, ROWS_N)
    tile_column_active = tile_columns < N
    if INDEX64:
        # Widened before every `* K`, `* (K // 2)`, `* 32`, `* 16` and `* N`.
        columns = columns.to(tl.int64)
        tile_columns = tile_columns.to(tl.int64)

    if PACKED:
        # GROUP_K blocks x 16 packed bytes, one byte per adjacent K pair.
        raw_bytes = tl.arange(0, GROUP_K * 16)
        # Lane j duplicates byte j >> 1; the nibble pick stays a constant shift.
        pair_lanes = tl.arange(0, 32) >> 1
    else:
        raw_codes = tl.arange(0, GROUP_K * 32)

    for group_base in range(tl.program_id(1) * GROUP_K, KBLOCKS, GROUP_STRIDE * GROUP_K):
        if INDEX64:
            group_base = group_base.to(tl.int64)
        block = group_base + group_in_tile
        block_active = block < KBLOCKS
        active = column_active & block_active

        ac = tl.load(A + block[:, None] * 32 + lanes[None, :],
                     mask=block_active[:, None], other=0).to(tl.int32)
        scale_a = tl.load(SA + block, mask=block_active, other=0).to(tl.int32)

        if PACKED:
            # One contiguous raw-byte row per column: 16 bytes a block, so a
            # GROUP_K group is 64 bytes at the default. The tail bytes of a
            # partial trailing group are masked here and their rows masked out
            # by `block_active` below.
            byte = group_base * 16 + raw_bytes
            byte_offsets = tile_columns[:, None] * (K // 2) + byte[None, :]
            byte_tile = tl.load(
                B + byte_offsets,
                mask=tile_column_active[:, None] & (byte < K // 2)[None, :],
                other=0,
            )
            pairs = tl.reshape(byte_tile, (TILE, 16))
            gather_index = tl.broadcast_to(pair_lanes[None, :], (TILE, 32))
            # The gather refuses uint8, so the raw tile reaches it through a
            # same-width bitcast: one byte in, one byte out, bits unchanged. The
            # widen and the 0xff mask happen only *after* the gather -- widening
            # first would quadruple the tile in the unified buffer, and the mask
            # restores the unsigned value the frozen select needs (its odd lane
            # shifts right by four with no mask, so a sign-extended byte would be
            # wrong there).
            gathered = (
                tl.gather(pairs.to(tl.int8, bitcast=True), gather_index, 1).to(tl.int32) & 255
            )
            bc = _e2m1_x2_lut(_packed_nibble(gathered, lanes[None, :]))
        else:
            # The dense form already carries one code per K element.
            code = group_base * 32 + raw_codes
            code_offsets = tile_columns[:, None] * K + code[None, :]
            code_tile = tl.load(
                B + code_offsets,
                mask=tile_column_active[:, None] & (code < K)[None, :],
                other=0,
            )
            bc = tl.reshape(code_tile, (TILE, 32)).to(tl.int32)

        scale_offsets = (columns // BLOCK_ROWS) * KBLOCKS + block
        scale_b = tl.load(SB + scale_offsets, mask=active, other=0)
        if B_SCALE_FLOAT:
            scale_b = _original_scale_code(scale_b, FP4)
        else:
            scale_b = scale_b.to(tl.int32)

        scaled, invalid = _fragment_rows(ac, bc, scale_a, scale_b, lanes, FP4)
        offsets = block * N + columns
        tl.store(FRAG + offsets, scaled, mask=active)
        tl.store(BAD + offsets, tl.where(invalid, 1, 0).to(tl.uint8), mask=active)