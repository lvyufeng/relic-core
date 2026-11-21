"""Ascend kernels for the exact M=1 block-32 decode GEMV.

The accepted block GEMM (``_ascend_block_gemm_kernels.block_gemm_kernel``) runs
one program per 16-column tile and walks **every** K block inside that program,
accumulating strictly in increasing block order. For a single decode row that
leaves only ``ceil(N / 16)`` programs -- a few hundred, against eight cards --
and each one owns a long serial chain of reduction and scale bit operations per
K block. The work is right; the parallelism is not.

These two kernels split that chain without changing a single arithmetic
statement, and without associating anything across K:

* :func:`decode_fragment_kernel` -- pass 1. Each program owns one column tile and
  a strided subset of the K blocks, and computes the corresponding fragments
  independently. One fragment is exactly one iteration of the accepted kernel's
  loop body.
* :func:`decode_accumulate_kernel` -- pass 2. Adds those fragments back in
  increasing K-block order into one FP32 accumulator per column, which is
  literally the accepted kernel's ``accumulator = accumulator + scaled`` chain.

The reductions inside a fragment are the frozen helpers from
``_ascend_block_gemm_kernels`` imported by name (``_cut``, ``_decode_e4m3``,
``_floor_exp``, ``_grid_exponent``, ``_nominal_e4``, ``_original_scale_code``,
``_pow2``, ``_scale_e8m0``) and the packed-nibble widening is the frozen
``_ascend_unpack_kernels._e2m1_x2``. Re-using them is what makes the fragment the
accepted fragment *by construction*; ``decode_block_gemv``'s NPU test
additionally asserts bit equality against the accepted kernel.

Deliberately absent: any split-K tree reduction, any fused multiply-add, any
reassociated sum, any native FP4/FP8 instruction, and any FP64. The scratch is
K-block major rather than ``[N, KBLOCKS]`` because a pass-1 program owns a run of
columns and writes one row per block it computed, so pass 2 reads one contiguous
row per block; a column-major scratch would make both passes strided.

**The grid bound is on the total program count, not on one axis.** Ascend rejects
a launch whose total block count exceeds 65535 (the earlier one-dimensional
"grid should be less than 65536" failure is that limit), so a second grid axis is
*not* a way around it: ``decode_column_chunks * KBLOCKS`` can exceed the cap even
when each axis is under it. The launcher therefore splits the K blocks into
``KSPLIT`` pieces and gives pass 1 a grid of ``(column chunks, KSPLIT)`` with
``chunks * KSPLIT <= 65535``; each program then walks the blocks it owns with
``range(pid, KBLOCKS, KSPLIT)``. Doing it that way needs no integer division to
recover an index from a flattened program id -- division by a non-power-of-two
constant is the one integer operation this backend has no proven lowering for --
so the block is always a loop scalar and the A codes stay a broadcast vector.

Legacy edge semantics are preserved rather than repaired: ``_pow2`` wraps on
out-of-range exponents and ``_cut`` inherits that wrap, exactly as the accepted
kernel does. The two passes reproduce it because they call the same helpers with
the same arguments; nothing here "fixes" the wrap.
"""
from __future__ import annotations

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
from ._ascend_unpack_kernels import _e2m1_x2

#: Sentinels the frozen reduction maximizes against. ``_BELOW_ALL`` sits under every
#: reachable nominal and ``_EMPTY`` is the nominal of a zero operand;
#: ``_grid_exponent`` maps anything below -500 to a zero grid exponent, which is
#: truncation to an integer.
#:
#: Both are ``tl.constexpr`` because the Triton front end refuses to read **any**
#: plain module scalar from a ``@jit`` body -- it fails with "cannot access global
#: variable ... within @jit", which is how the first device run of the FP4
#: quantizer in this tree failed -- and the environment flag that relaxes that rule
#: is deliberately unused here. They are kept as *integers* rather than floats for a
#: second, independent reason: the JIT compares a cached kernel's globals for change
#: by equality, and ``nan != nan`` would make a float-NaN global read as "changed
#: since compiled" on the second call of the same kernel.
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
    zeroed *here* while the nominal still reads the magnitude -- the frozen
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
def decode_fragment_kernel(
    A, SA, B, SB, FRAG, BAD,
    N: tl.constexpr, K: tl.constexpr, KBLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, FP4: tl.constexpr, PACKED: tl.constexpr,
    B_SCALE_FLOAT: tl.constexpr, ROWS: tl.constexpr, KSPLIT: tl.constexpr,
    INDEX64: tl.constexpr,
):
    """Write one scaled K-32 fragment per (column, K block) into ``FRAG``/``BAD``.

    ``FRAG`` is ``[KBLOCKS, N]`` FP32 and ``BAD`` ``[KBLOCKS, N]`` uint8; entry
    ``(block, column)`` is the accepted kernel's ``scaled`` and its per-column
    invalid flag for that block. Each program owns ``ROWS`` columns and the K
    blocks ``pid(1), pid(1) + KSPLIT, ...``, so the A codes of a block are one
    32-lane vector loaded once and broadcast -- never a 2D gather.

    ``INDEX64`` widens the column and block indices *before* the offset products,
    which is the only safe order when ``N * K`` reaches 2**31; the launch chooses
    it from the reach.
    """
    lanes = tl.arange(0, 32)
    columns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    active = columns < N
    if INDEX64:
        columns = columns.to(tl.int64)

    # One program walks a strided slice of the K blocks; the iterations are
    # independent (each writes its own scratch row), so there is no recurrence
    # here even though they share a program.
    for block in range(tl.program_id(1), KBLOCKS, KSPLIT):
        if INDEX64:
            block = block.to(tl.int64)
        ac = tl.load(A + block * 32 + lanes).to(tl.int32)
        a_nan = (ac & 127) == 127
        av = tl.where(a_nan, 0.0, _decode_e4m3(ac))
        an = _nominal_e4(ac)
        scale_a = tl.load(SA + block).to(tl.int32)
        if PACKED:
            # Low nibble is the even K element, high nibble the odd one, which is
            # the frozen packed layout: one byte per adjacent pair along K.
            byte_offsets = columns[:, None] * (K // 2) + block * 16 + (lanes[None, :] >> 1)
            packed = tl.load(B + byte_offsets, mask=active[:, None], other=0).to(tl.int32)
            bc = _e2m1_x2((packed >> ((lanes[None, :] & 1) * 4)) & 15)
        else:
            offsets = columns[:, None] * K + block * 32 + lanes[None, :]
            bc = tl.load(B + offsets, mask=active[:, None], other=0).to(tl.int32)
        if FP4:
            magnitude = tl.abs(bc)
            bv = _fp4_value(bc, magnitude)
            bn = _fp4_nominal(magnitude)
            b_nan = ~_fp4_code_valid(magnitude)
        else:
            b_nan = (bc & 127) == 127
            bv = tl.where(b_nan, 0.0, _decode_e4m3(bc))
            bn = _nominal_e4(bc)

        product = av[None, :] * bv
        nominal = an[None, :] + bn
        # First 16 lanes: nominal anchor from their own exponents only.
        first = tl.max(tl.where(lanes[None, :] < 16, nominal, _BELOW_ALL), 1)
        g0 = _grid_exponent(first)
        integers = (product * _pow2(-g0[:, None])).to(tl.int32)
        total = tl.sum(tl.where(lanes[None, :] < 16, integers, 0), 1).to(tl.float32) * _pow2(g0)
        # Second 16 lanes: the first-half carry joins them inside their own window.
        second = tl.max(tl.where(lanes[None, :] >= 16, nominal, _BELOW_ALL), 1)
        second = tl.maximum(second, _floor_exp(total))
        g1 = _grid_exponent(second)
        integers = (product * _pow2(-g1[:, None])).to(tl.int32)
        whole = tl.sum(tl.where(lanes[None, :] >= 16, integers, 0), 1)
        whole += (total * _pow2(-g1)).to(tl.int32)
        total = whole.to(tl.float32) * _pow2(g1)
        fragment = _cut(total, _floor_exp(total))

        scale_offsets = (columns // BLOCK_ROWS) * KBLOCKS + block
        scale_b = tl.load(SB + scale_offsets, mask=active, other=0)
        if B_SCALE_FLOAT:
            scale_b = _original_scale_code(scale_b, FP4)
        else:
            scale_b = scale_b.to(tl.int32)
        # Two separate rescales in the accepted order: scale A rounds, then scale B.
        scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
        invalid = tl.sum((b_nan | a_nan[None, :]).to(tl.int32), 1) != 0
        offsets = block * N + columns
        tl.store(FRAG + offsets, scaled, mask=active)
        tl.store(BAD + offsets, tl.where(invalid, 1, 0).to(tl.uint8), mask=active)


@triton.jit
def decode_accumulate_kernel(
    FRAG, BAD, OUT,
    N: tl.constexpr, KBLOCKS: tl.constexpr, COLUMNS: tl.constexpr,
    INDEX64: tl.constexpr,
):
    """Add the fragments back in increasing K-block order, one column tile at a time.

    Nothing here may be reordered: ``acc = acc + fragment`` in block order is the
    accepted kernel's accumulator chain, and a tree or ``tl.sum`` reduction over
    the block axis would be a different (associative) statement. The invalid flags
    OR together and replace the whole column with NaN, which is also the accepted
    kernel's final ``tl.where``.
    """
    columns = tl.program_id(0) * COLUMNS + tl.arange(0, COLUMNS)
    active = columns < N
    if INDEX64:
        columns = columns.to(tl.int64)
    accumulator = tl.full((COLUMNS,), 0.0, tl.float32)
    invalid = tl.full((COLUMNS,), False, tl.int1)
    for block in range(KBLOCKS):
        if INDEX64:
            block = block.to(tl.int64)
        offsets = block * N + columns
        accumulator = accumulator + tl.load(FRAG + offsets, mask=active, other=0.0)
        invalid |= tl.load(BAD + offsets, mask=active, other=0) != 0
    accumulator = tl.where(invalid, float("nan"), accumulator)
    tl.store(OUT + columns, accumulator, mask=active)