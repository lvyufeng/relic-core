"""Ascend kernels for the grouped-K exact M=1 block-32 decode GEMV.

The frozen two-pass prototype (``_ascend_decode_block_gemv_kernels``) computes one
``(column, K block)`` fragment per program. Measured on the target build that
shape was exact but **slower** than the accepted block GEMM for FP4: 23040
one-fragment programs cost more in launch and per-program setup than the serial
work they removed (FP4 ``w1`` 12.73 -> 49.64 ms, ``w2`` 11.38 -> 49.87 ms, while
the FP8 case at 12800 programs moved 6.24 -> 6.90 ms). The fragment count was not
the only difference between those cases, so no cause is claimed here; what is
claimed is only that a one-fragment program is too small.

These kernels keep the arithmetic and change only the granule size:

* :func:`decode_grouped_fragment_kernel` -- pass 1. One program owns
  ``ROWS_N`` columns **and** ``GROUP_K`` K blocks, as one flat 2-D tile of
  ``GROUP_K * ROWS_N`` rows by 32 K lanes. Row ``g * ROWS_N + r`` is exactly the
  ``(block_base + g, column_base + r)`` fragment, and every reduction is along the
  32-lane axis of that row, so grouping blocks inside a program multiplies the
  work per program without touching a single arithmetic statement. The tile is
  strictly 2-D: the backend's known lack of a 3-D reduction is why the
  ``(group, column, lane)`` cube is flattened to ``(row, lane)`` up front.
* :func:`decode_serial_gemv_kernel` -- the other granule: the accepted kernel's
  shape (one program per column tile, a serial loop over every K block in
  increasing order), but the weight is read from the checkpoint's **packed E2M1
  nibbles** and the **raw E8M0 grid** in register. That removes the 39 MiB
  global ``INT8 x2`` widening pass and the FP32 half-scale table the two-pass
  prototype has to build before it can start, and it needs no K scratch at all,
  at the cost of the K loop being serial again.

Both kernels call the *same* frozen reduction, value decoders and rescales:
``_cut``, ``_decode_e4m3``, ``_floor_exp``, ``_grid_exponent``, ``_nominal_e4``,
``_original_scale_code``, ``_pow2`` and ``_scale_e8m0`` are imported from
``_ascend_block_gemm_kernels``. The fragment is therefore the accepted fragment by
construction: only how many of them one program computes changed.

The packed-nibble widening is the one arithmetic step written here rather than
imported, and it is a **spelling** change with no numerical change. The frozen
``_ascend_unpack_kernels._e2m1_x2`` and the obvious nibble extraction
``packed >> ((lane & 1) * 4)`` between them use per-lane **variable** shifts, and
CANN lowers a variable vector shift badly: measured, the packed variant cost about
37 ms more than the identical INT8 operand at the same grid while both hold the
same numbers. :func:`_packed_nibble` selects between two **constant** shifts and
:func:`_e2m1_x2_lut` is the same 16-entry codebook as a comparison ladder, so no
dynamic shift remains. Every magnitude is an exact small integer, so the two
spellings agree bit for bit on all sixteen codes -- including E2M1 code 8, whose
negative zero both spellings turn into integer zero. The frozen helper is left
untouched: it is the proven codebook, and this is a private spelling of it.
Pass 2 is the frozen ``decode_accumulate_kernel`` unchanged, so the fragments are
still added back in increasing K-block order -- no tree reduction, no reassociated
sum, no fused multiply-add, no requantization, no native FP4/FP8 arithmetic.

Indexing notes that are contract, not style:

* **The grid bound is on the total program count.** Ascend rejects a launch
  whose total block count exceeds 65535, so a second grid axis is not a way
  around it. The launcher shapes ``(column chunks, group stride)`` so the
  product fits and lets each program stride the remaining K groups.
* **Every index is widened before any offset product.** ``columns`` and ``block``
  become ``int64`` before ``* K``, ``* 32``, ``* (K // 2)`` or ``* N``; doing the
  product first and widening after would already have overflowed.
* **Both tile sides are powers of two**, so the flat row decomposes with shifts
  and a mask. Division by a non-power-of-two constant has no proven lowering on
  this backend, which is the same reason the frozen prototype strides instead of
  flattening a program id.
* **The tile has a hard size ceiling of 64 rows.** A 128-row tile
  (``GROUP_K = 8`` x ``ROWS_N = 16``) does not compile: ``ub overflow, requires
  2675968 bits while 1572864 bits available``, and the compiler's own retry with
  code motion and multi-buffering disabled fails too. That is why the wrapper
  bounds ``GROUP_K * ROWS_N`` at 64 and refuses anything larger before launch.
  The bound lives in the wrapper so the CPU reference path rejects the same
  geometry, not here -- these kernels only assume the caller respected it.
* **Ragged K and N are masked at load time** (``other=0``), so a partial group at
  ``K = 160`` (5 blocks against ``GROUP_K = 8``) or a partial column tile reads a
  defined zero and is then masked out of the store.

Legacy edge semantics are preserved rather than repaired: ``_pow2`` wraps on an
out-of-range exponent and ``_cut`` inherits that wrap, exactly as the accepted
kernel does. Nothing here "fixes" the wrap, and a validator that produces NaN
still turns the whole column into NaN in pass 2.

Deliberately absent: any performance claim. These are schedule variants; only an
actual measured decode on the target build can say which is faster.
"""
import triton
import triton.language as tl

from ._ascend_block_gemm_kernels import (
    _cut,
    _decode_e4m3,
    _floor_exp,
    _grid_exponent,
    _nominal_e4,
    _original_scale_code,
    _pow2,
    _scale_e8m0,
)

#: Sentinels the frozen reduction maximizes against. ``_BELOW_ALL`` sits under every
#: reachable nominal and ``_EMPTY`` is the nominal of a zero operand. Both are
#: ``tl.constexpr`` because the Triton front end refuses to read any plain module
#: scalar from a ``@jit`` body, and both are **integers** rather than floats because
#: the JIT compares a cached kernel's globals for change by equality -- ``nan != nan``
#: would read as "changed since compiled" on the second call of the same kernel.
_BELOW_ALL = tl.constexpr(-2000)
_EMPTY = tl.constexpr(-1000)


@triton.jit
def _fp4_code_valid(magnitude):
    """Whether a doubled E2M1 magnitude is in the exact INT8-times-two codebook.

    ``magnitude`` is ``|x2|`` already, so ``0, 1, 2, 3, 4, 6, 8, 12`` are the only
    legal values. A packed nibble always decodes to one of them; an INT8-times-two
    operand can carry anything a caller staged, so it is checked.
    """
    valid = (magnitude == 0) | (magnitude == 1) | (magnitude == 2) | (magnitude == 3)
    valid |= (magnitude == 4) | (magnitude == 6) | (magnitude == 8) | (magnitude == 12)
    return valid


@triton.jit
def _fp4_value(x2, magnitude):
    """FP32 E2M1 value of an exact doubled code; an out-of-codebook code is zero.

    ``x2 * 0.5`` is exact for every codebook magnitude, and the invalid case is
    zeroed *here* while the nominal still reads the magnitude -- the accepted
    kernel's split, kept because only the flag turns such a column into NaN.
    """
    return tl.where(_fp4_code_valid(magnitude), x2.to(tl.float32) * 0.5, 0.0)


@triton.jit
def _fp4_nominal(magnitude):
    """E2M1 nominal exponent of a doubled code, with the frozen empty sentinel."""
    nominal = tl.where(magnitude >= 8, 2,
                       tl.where(magnitude >= 4, 1, tl.where(magnitude >= 2, 0, -1)))
    return tl.where(magnitude != 0, nominal, _EMPTY)


@triton.jit
def _packed_nibble(packed, lanes):
    """Low nibble of a packed byte on an even lane, high nibble on an odd one.

    ``packed >> ((lanes & 1) * 4)`` is the obvious spelling and computes a
    **per-lane variable shift amount**, which this backend lowers badly -- measured
    as the dominant cost of the packed variant. Selecting between two
    constant-shift expressions produces exactly the same nibble with no dynamic
    shift, because the parity of a lane is a compile-time-known alternation.
    """
    return tl.where((lanes & 1) == 0, packed & 15, packed >> 4)


@triton.jit
def _e2m1_x2_lut(nibble):
    """Exact ``INT8 = 2 * E2M1`` value of one nibble, by comparison ladder.

    The frozen ``_ascend_unpack_kernels._e2m1_x2`` builds the same codebook as
    ``(2 + (m & 1)) << max((m >> 1) - 1, 0)`` -- a second per-lane variable shift.
    This ladder is the identical 16-entry table

        code   0  1  2  3  4  5  6  7   8   9  10  11  12  13  14  15
        x2     0  1  2  3  4  6  8 12   0  -1  -2  -3  -4  -6  -8 -12

    with no shift at all. Every magnitude is an exact small integer, so the two
    spellings are bit-identical on all sixteen codes, including code 8: E2M1's
    negative zero becomes integer zero, losing the sign, exactly as the frozen
    helper does and as the INT8 operand form documents.
    """
    magnitude = nibble & 7
    absolute = tl.where(magnitude < 4, magnitude,
                        tl.where(magnitude == 4, 4,
                                 tl.where(magnitude == 5, 6,
                                          tl.where(magnitude == 6, 8, 12))))
    return tl.where((nibble & 8) != 0, -absolute, absolute)


@triton.jit
def _activation_values(a_codes):
    """``(value, nominal, is_nan)`` of one block of original E4M3 activation codes.

    Shape-agnostic on purpose: a ``(32,)`` vector and a ``(rows, 32)`` matrix are
    both elementwise inputs, which is what lets the grouped and serial kernels
    share this decoder verbatim.
    """
    a_nan = (a_codes & 127) == 127
    return tl.where(a_nan, 0.0, _decode_e4m3(a_codes)), _nominal_e4(a_codes), a_nan


@triton.jit
def _weight_values(b_codes, FP4: tl.constexpr):
    """``(value, nominal, is_invalid)`` of one block of weight codes.

    ``FP4`` selects the exact doubled-E2M1 codebook; otherwise the operand is
    original E4M3. Same shape-agnostic contract as :func:`_activation_values`.
    """
    if FP4:
        magnitude = tl.abs(b_codes)
        return _fp4_value(b_codes, magnitude), _fp4_nominal(magnitude), ~_fp4_code_valid(magnitude)
    b_nan = (b_codes & 127) == 127
    return tl.where(b_nan, 0.0, _decode_e4m3(b_codes)), _nominal_e4(b_codes), b_nan


@triton.jit
def _reduce_two_halves(product, nominal, lanes):
    """The accepted kernel's two-16-term reduction of one row, ending in ``_cut``.

    Every statement is the accepted ``block_gemm_kernel`` body: the first half is
    anchored on its own largest nominal, the second half on its own plus the
    first half's carry, and the final ``_cut`` normalizes the joined total. The
    reductions are along the lane axis only, so a row is independent of how many
    other rows share the tile.
    """
    first = tl.max(tl.where(lanes[None, :] < 16, nominal, _BELOW_ALL), 1)
    g0 = _grid_exponent(first)
    integers = (product * _pow2(-g0[:, None])).to(tl.int32)
    total = tl.sum(tl.where(lanes[None, :] < 16, integers, 0), 1).to(tl.float32) * _pow2(g0)
    second = tl.max(tl.where(lanes[None, :] >= 16, nominal, _BELOW_ALL), 1)
    second = tl.maximum(second, _floor_exp(total))
    g1 = _grid_exponent(second)
    integers = (product * _pow2(-g1[:, None])).to(tl.int32)
    whole = tl.sum(tl.where(lanes[None, :] >= 16, integers, 0), 1)
    whole += (total * _pow2(-g1)).to(tl.int32)
    total = whole.to(tl.float32) * _pow2(g1)
    return _cut(total, _floor_exp(total))


@triton.jit
def _fragment_single_row(a_codes, b_codes, scale_a, scale_b, lanes, FP4: tl.constexpr):
    """Scale one block's fragment: A codes are a 32-lane vector broadcast over columns.

    This is the serial path's shape, where every column of the tile reads the same
    activation block. The two rescales are separate integer-exponent
    multiplications in the accepted order -- A first, then B -- because combining
    the exponents would change the reference's overflow and rounding behaviour.
    """
    av, an, a_nan = _activation_values(a_codes)
    bv, bn, b_nan = _weight_values(b_codes, FP4)
    fragment = _reduce_two_halves(av[None, :] * bv, an[None, :] + bn, lanes)
    scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
    return scaled, tl.sum((b_nan | a_nan[None, :]).to(tl.int32), 1) != 0


@triton.jit
def _fragment_rows(a_codes, b_codes, scale_a, scale_b, lanes, FP4: tl.constexpr):
    """Scale one block's fragment: A codes already carry one row per tile row.

    This is the grouped path's shape. Each row of ``a_codes`` is the block its own
    row owns, so the arithmetic below is the accepted expression with the
    activation broadcast already materialized -- the same per-row statements the
    serial helper makes, not a different reduction.
    """
    av, an, a_nan = _activation_values(a_codes)
    bv, bn, b_nan = _weight_values(b_codes, FP4)
    fragment = _reduce_two_halves(av * bv, an + bn, lanes)
    scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
    return scaled, tl.sum((b_nan | a_nan).to(tl.int32), 1) != 0


@triton.jit
def decode_grouped_fragment_kernel(
    A, SA, B, SB, FRAG, BAD,
    N: tl.constexpr, K: tl.constexpr, KBLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr, PACKED: tl.constexpr,
    B_SCALE_FLOAT: tl.constexpr,
    ROWS_N: tl.constexpr, ROW_SHIFT: tl.constexpr, GROUP_K: tl.constexpr,
    TILE: tl.constexpr, GROUP_STRIDE: tl.constexpr, INDEX64: tl.constexpr,
):
    """Write ``GROUP_K`` blocks x ``ROWS_N`` columns of fragments per iteration.

    ``FRAG`` is ``[KBLOCKS, N]`` FP32 and ``BAD`` ``[KBLOCKS, N]`` uint8; entry
    ``(block, column)`` is the accepted kernel's ``scaled`` and its per-column
    invalid flag for that block -- the same buffers and the same layout the frozen
    two-pass prototype uses, so pass 2 is that prototype's kernel unchanged.

    The flat tile row ``i`` owns column ``i & (ROWS_N - 1)`` and block
    ``group_base + (i >> ROW_SHIFT)``. Each program walks the K groups it was
    given with a stride, so a geometry whose grid would exceed the device cap
    still covers every block exactly once.
    """
    lanes = tl.arange(0, 32)
    flat = tl.arange(0, TILE)
    group = flat >> ROW_SHIFT
    within = flat & (ROWS_N - 1)
    columns = tl.program_id(0) * ROWS_N + within
    column_active = columns < N
    if INDEX64:
        # Widened before the `* K`, `* (K // 2)` and `* N` products below.
        columns = columns.to(tl.int64)

    for group_base in range(tl.program_id(1) * GROUP_K, KBLOCKS, GROUP_STRIDE * GROUP_K):
        if INDEX64:
            group_base = group_base.to(tl.int64)
        block = group_base + group
        block_active = block < KBLOCKS
        active = column_active & block_active

        # One activation block per tile row, gathered as a flat 2-D load: rows that
        # share a group read the same 32 codes. Not a 3-D operand, which is the
        # point -- the backend has no proven 3-D reduction. ``block`` is a [TILE]
        # vector, so it needs the explicit ``[:, None]``: ``block * 32 + lanes[None, :]``
        # would try to broadcast a [TILE] vector against the 32-wide lane axis and be
        # rejected as "shape incompatible in dimension 1".
        ac = tl.load(A + block[:, None] * 32 + lanes[None, :], mask=block_active[:, None],
                     other=0).to(tl.int32)
        scale_a = tl.load(SA + block, mask=block_active, other=0).to(tl.int32)

        if PACKED:
            # Low nibble is the even K element, high nibble the odd one: one byte
            # per adjacent pair along K, the checkpoint's own layout.
            byte_offsets = (columns[:, None] * (K // 2) + block[:, None] * 16
                            + (lanes[None, :] >> 1))
            packed = tl.load(B + byte_offsets, mask=active[:, None], other=0).to(tl.int32)
            bc = _e2m1_x2_lut(_packed_nibble(packed, lanes[None, :]))
        else:
            offsets = columns[:, None] * K + block[:, None] * 32 + lanes[None, :]
            bc = tl.load(B + offsets, mask=active[:, None], other=0).to(tl.int32)

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


@triton.jit
def decode_serial_gemv_kernel(
    A, SA, B, SB, OUT,
    N: tl.constexpr, K: tl.constexpr, KBLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr, PACKED: tl.constexpr,
    B_SCALE_FLOAT: tl.constexpr, BN: tl.constexpr, INDEX64: tl.constexpr,
):
    """Accumulate every K block in increasing order, one column tile per program.

    The accepted ``block_gemm_kernel`` shape -- one program per column tile and a
    serial K loop -- with two differences that are about where the bytes are read,
    not what is computed:

    * the weight is the checkpoint's packed E2M1 nibble pair, widened in register
      by the exact :func:`_e2m1_x2_lut` ladder, so no global ``INT8 x2`` pass has to
      run first;
    * the weight scale is the raw E8M0 byte, so no FP32 half-scale table and no
      ``_original_scale_code`` decode is needed for the packed form.

    That removes both preparation passes and the K-block scratch, at the cost of
    the K chain being serial in one program again. The accumulation statement is
    the accepted one and the order is strictly increasing K, so nothing is
    reassociated.
    """
    lanes = tl.arange(0, 32)
    columns = tl.program_id(0) * BN + tl.arange(0, BN)
    active = columns < N
    if INDEX64:
        # Widened before the `* K`, `* (K // 2)` and per-block scale products.
        columns = columns.to(tl.int64)
    accumulator = tl.full((BN,), 0.0, tl.float32)
    invalid = tl.full((BN,), False, tl.int1)

    for block in range(KBLOCKS):
        if INDEX64:
            block = block.to(tl.int64)
        ac = tl.load(A + block * 32 + lanes).to(tl.int32)
        if PACKED:
            byte_offsets = columns[:, None] * (K // 2) + block * 16 + (lanes[None, :] >> 1)
            packed = tl.load(B + byte_offsets, mask=active[:, None], other=0).to(tl.int32)
            bc = _e2m1_x2_lut(_packed_nibble(packed, lanes[None, :]))
        else:
            offsets = columns[:, None] * K + block * 32 + lanes[None, :]
            bc = tl.load(B + offsets, mask=active[:, None], other=0).to(tl.int32)
        scale_a = tl.load(SA + block).to(tl.int32)
        scale_b = tl.load(SB + (columns // BLOCK_ROWS) * KBLOCKS + block,
                          mask=active, other=0)
        if B_SCALE_FLOAT:
            scale_b = _original_scale_code(scale_b, FP4)
        else:
            scale_b = scale_b.to(tl.int32)

        scaled, bad = _fragment_single_row(ac, bc, scale_a, scale_b, lanes, FP4)
        # No multiply feeds this add: scale rounding is completed by the bit
        # constructors above, and the K blocks remain in their original order.
        accumulator = accumulator + scaled
        invalid |= bad

    accumulator = tl.where(invalid, float("nan"), accumulator)
    tl.store(OUT + columns, accumulator, mask=active)