"""Ascend fragment kernels: the same exact reduction with FP32 significand products.

``_ascend_integer_decode_kernels`` proved the integer rewrite is bit-exact but
slower than the accepted FP32 fragment path (62.90 ms grouped, 85.40 ms serial,
17.90 ms packed FP32). The hypothesis for the gap is that the *varying* integer
shift amounts and the int32 significand multiply do not lower to vector AIV code
on this backend, while an FP32 multiply does. This module keeps the identical
arithmetic and replaces exactly those two operations:

* the significand multiply itself moves out of ``int32``: the two fields are cast
  to FP32 and multiplied there (``<= 225``, exact), and a per-lane
  ``significand_product << shift`` / ``>> shift`` becomes ``product * _pow2(power
  - grid)`` followed by one cast to ``int32``. No integer multiply and no varying
  shift amount are left on the per-lane path;
* ``_ilog2(abs(integer_sum))`` becomes ``_floor_exp(float(integer_sum))``, i.e. a
  bitcast, a constant ``>> 23`` and a constant subtract;
* the carry and the window cut scale by an exact dyadic factor through
  ``(integer.to(fp32) * _pow2(shift)).to(int32)`` instead of a shift whose amount
  is a runtime value.

The arithmetic is unchanged, and the bound is what makes it exact rather than
merely close:

* every significand is ``<= 15`` (E4M3) or ``<= 3`` (E2M1), so the FP32 product of
  the two cast significands is at most ``225`` -- eight significant bits, *exact*
  in FP32. The product is never formed in ``int32``: a varying integer multiply is
  one of the two operations this module exists to remove;
* the per-lane factor is ``2**(power - grid)``; the encoding bounds a nominal
  exponent sum at ``[-12, 16]`` (FP8 x FP8) and the first-half carry can raise the
  second half's anchor to at most ``21``, so the grid stays inside ``[-25, 8]`` and
  the per-lane exponent inside ``[-26, 10]``. Every factor is therefore a *normal*
  exact dyadic float and the product is exact in FP32;
* conservatively ``|per-lane term| < 2**15`` and a 16-term partial sum is
  ``< 2**19`` (a raw ``240 x 240`` product with a nominal ``14`` anchor is
  ``57600 / 2 = 28800``), so the sum is exact as ``int32`` and as the FP32 a cast
  produces, which is what lets the carry exponent come from a bitcast instead of a
  binary-search ``ilog2``;
* the two rescales by ``2**(grid0 - grid1)`` and ``2**(grid1 - window)`` have
  exponents ``|shift| <= ~33`` -- still normal dyadic factors -- and the shifted
  value stays below ``2**20`` because the anchor that dominates the second half is
  the carry's own exponent, so the ``int32`` result cannot overflow;
* the cast is applied to the *magnitude* and the sign afterwards, so a negative
  term truncates toward zero exactly as the accepted cast does, without depending
  on how this backend lowers a negative float-to-int conversion.

Everything else is the frozen, hardware-proven shape: the same loads issued in the
kernels (not in the arithmetic helper) for rank analysis, the same two-half
nominal-exponent anchor and 13-bit window, the same guarded two-step rescale with
only global stores inside the branch, the same frozen ``decode_accumulate_kernel``
for the ordered second pass, and the same generic ``_scale_e8m0`` proof path.

Deliberately absent: no operand float decode, no native FP4/FP8 instruction, no
matrix/Cube unit, no FP64, no log/exp, no FMA, no reassociated K sum, and no
precision relaxation -- every value entering the reduction is still the exact
integer the accepted kernel computes.
"""
from __future__ import annotations

import triton
import triton.language as tl

from ._ascend_block_gemm_kernels import (
    _floor_exp,
    _grid_exponent,
    _original_scale_code,
    _pow2,
    _scale_e8m0,
)
from ._ascend_integer_decode_kernels import (
    _e2m1_fields,
    _e4_fields,
    _scale_fast,
)


@triton.jit
def _float_terms(significand_product, power, grid, negative, half):
    """The accepted ``(product * _pow2(-grid)).to(tl.int32)`` without a varying shift.

    ``significand_product`` is the exact FP32 product of two significands (each
    ``<= 15`` or ``<= 3``), ``power`` the sum of the two exponent fields and
    ``grid`` the per-column window anchor. The dyadic factor is exact and the
    product of an eight-bit significand product with it is exact in FP32, so the
    value cast is the integer the accepted kernel's cast produces.

    The cast is taken on the magnitude and the sign applied afterwards, which is
    the frozen integer helper's order: it makes truncation independent of the
    backend's lowering for negative float-to-int, and for a truncating conversion
    the two orders are identical.
    """
    factor = _pow2(power - grid[:, None])
    magnitude = (significand_product * factor).to(tl.int32)
    magnitude = tl.where(half, magnitude, 0)
    return tl.where(negative, -magnitude, magnitude)


@triton.jit
def _float_scale_int(value, shift):
    """``trunc(value * 2**shift)`` for an integer ``|value| < 2**24``.

    ``value.to(tl.float32)`` is exact in that range and ``_pow2`` builds an exact
    normal dyadic factor, so the product is exact and the cast is a pure
    truncation. The shift amount is per-lane data but it is only ever *added into
    an exponent field*, never used as an integer shift, which is the change this
    module exists to measure.
    """
    return (value.to(tl.float32) * _pow2(shift)).to(tl.int32)


@triton.jit
def _float_fragment(ac, weight, lanes, PACKED: tl.constexpr):
    """One accepted fragment from already-loaded operands, with FP32 significands.

    Pointer arithmetic and every ``tl.load`` stay in the kernels below: the Ascend
    compiler's ``OffsetAnalysis`` pass aborts when it has to combine pointer
    offsets of different ranks across a call boundary, which is why the frozen
    fragment kernel has this split too.

    Returns the **unscaled** 13-bit-window fragment and the invalid flag. The two
    E8M0 rescales belong to the kernels because a vector value returned out of a
    conditional inside a helper is the shape this compiler failed to legalize.
    """
    sig_a, pa, nom_a, neg_a, nan_a = _e4_fields(ac)
    if PACKED:
        # Low nibble is the even K element and the high nibble the odd one, which
        # is the released packed layout. The two nibbles are selected with two
        # *constant* shifts and a lane mask, never with a per-lane shift amount:
        # a varying shift measured 49.75 ms against 18.62 ms for the constant form
        # on the same packed group of four blocks.
        codes = tl.where((lanes[None, :] & 1) == 0, weight & 15, weight >> 4)
        sig_b, pb, nom_b, neg_b = _e2m1_fields(codes)
        # Every nibble is a valid E2M1 code, so the packed path has no invalid
        # weight lane; only the activation can be NaN.
        nan_b = weight * 0
    else:
        sig_b, pb, nom_b, neg_b, nan_b_e4 = _e4_fields(weight)
        nan_b = nan_b_e4.to(tl.int32)

    significand_product = sig_a[None, :].to(tl.float32) * sig_b.to(tl.float32)
    power = pa[None, :] + pb
    nominal = nom_a[None, :] + nom_b
    negative = neg_a[None, :] ^ neg_b

    # First half: anchor from its own nominal exponents, exactly as accepted.
    first = tl.max(tl.where(lanes[None, :] < 16, nominal, -2000), 1)
    grid0 = _grid_exponent(first)
    total0 = tl.sum(_float_terms(
        significand_product, power, grid0, negative, lanes[None, :] < 16), 1)

    # Second half: the first-half carry's exponent joins this half's window. The
    # carry exponent is the accepted ``ilog2(abs(total0)) + grid0`` written as a
    # bitcast exponent: ``abs(total0) < 2**20`` is exact in FP32, so its exponent
    # field *is* floor(log2), read with a constant shift and no binary search.
    carry = tl.where(total0 == 0, -1000, _floor_exp(total0.to(tl.float32)) + grid0)
    second = tl.max(tl.where(lanes[None, :] >= 16, nominal, -2000), 1)
    second = tl.maximum(second, carry)
    grid1 = _grid_exponent(second)
    whole = tl.sum(_float_terms(
        significand_product, power, grid1, negative, lanes[None, :] >= 16), 1)
    whole += _float_scale_int(total0, grid0 - grid1)

    # The accepted window cut: truncate the whole to 13 bits below its exponent.
    exponent = tl.where(whole == 0, -1000, _floor_exp(whole.to(tl.float32)) + grid1)
    window = _grid_exponent(exponent)
    fragment = _float_scale_int(whole, grid1 - window).to(tl.float32) * _pow2(window)

    invalid = tl.sum(nan_b + nan_a[None, :].to(tl.int32), 1) != 0
    return fragment, invalid


@triton.jit
def float_fragment_kernel(A, SA, B, SB, FRAG, BAD, N: tl.constexpr,
                          K: tl.constexpr, KBLOCKS: tl.constexpr,
                          BLOCK_ROWS: tl.constexpr, PACKED: tl.constexpr,
                          FP4: tl.constexpr, B_SCALE_FLOAT: tl.constexpr,
                          ROWS: tl.constexpr, GROUP: tl.constexpr,
                          KSPLIT: tl.constexpr, INDEX64: tl.constexpr):
    """Pass 1: one program owns ``ROWS`` columns and a strided run of K groups.

    ``GROUP`` K blocks are walked serially inside a program -- a static unroll
    count that bounds the work per program and the grid, not a fused ``GROUP x
    ROWS`` tile. The only resident tile is the fragment's ``ROWS x 32``, which is
    why ``ROWS`` is capped at 16 by the measured unified-buffer budget (32 needed
    3,440,640 bits against 1,572,864 on this build).

    Every load is issued here with the rank its own tensor has; the arithmetic call
    receives loaded values only.
    """
    columns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    active = columns < N
    if INDEX64:
        columns = columns.to(tl.int64)
    lanes = tl.arange(0, 32)
    groups = (KBLOCKS + GROUP - 1) // GROUP
    for group in range(tl.program_id(1), groups, KSPLIT):
        for inner in range(GROUP):
            block = group * GROUP + inner
            if INDEX64:
                # Widen before the ``block * N`` product, not after it.
                block = block.to(tl.int64)
            reached = block < KBLOCKS
            safe = tl.minimum(block, KBLOCKS - 1)
            ac = tl.load(A + safe * 32 + lanes).to(tl.int32)
            scale_a = tl.load(SA + safe).to(tl.int32)
            if PACKED:
                byte_offsets = columns[:, None] * (K // 2) + safe * 16 + (lanes[None, :] >> 1)
                weight = tl.load(B + byte_offsets, mask=active[:, None], other=0).to(tl.int32)
            else:
                offsets = columns[:, None] * K + safe * 32 + lanes[None, :]
                weight = tl.load(B + offsets, mask=active[:, None], other=0).to(tl.int32)
            scale_offsets = (columns // BLOCK_ROWS) * KBLOCKS + safe
            scale_b = tl.load(SB + scale_offsets, mask=active, other=0)
            if B_SCALE_FLOAT:
                scale_b = _original_scale_code(scale_b, FP4)
            else:
                scale_b = scale_b.to(tl.int32)
            fragment, bad = _float_fragment(ac, weight, lanes, PACKED)
            # The guarded fast path is decided HERE and the two schedules differ on
            # purpose: this one takes the two sequential exact power-of-two
            # rescalings when the whole tile is safe for both, and the frozen
            # generic helper otherwise. Only *stores* cross the branch -- a vector
            # value returned from inside a conditional is what this compiler failed
            # to legalize -- and the invalid flag is stored outside it either way.
            bits_a, safe_a = _scale_fast(fragment, scale_a)
            midway = bits_a.to(tl.float32, bitcast=True)
            bits_b, safe_b = _scale_fast(midway, scale_b)
            offset = block * N + columns
            store = active & reached
            if tl.min(tl.where(safe_a & safe_b, 1, 0).to(tl.int32), 0) != 0:
                tl.store(FRAG + offset, bits_b.to(tl.float32, bitcast=True), mask=store)
            else:
                # Generic: the frozen helper, applied twice separately, in the
                # accepted order. Never one combined exponent.
                generic = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
                tl.store(FRAG + offset, generic, mask=store)
            tl.store(BAD + offset, tl.where(bad, 1, 0).to(tl.uint8), mask=store)


@triton.jit
def float_serial_kernel(A, SA, B, SB, OUT, N: tl.constexpr, K: tl.constexpr,
                        KBLOCKS: tl.constexpr, BLOCK_ROWS: tl.constexpr,
                        PACKED: tl.constexpr, FP4: tl.constexpr,
                        B_SCALE_FLOAT: tl.constexpr, ROWS: tl.constexpr,
                        INDEX64: tl.constexpr):
    """Single-pass variant: one program owns ``ROWS`` columns and every K block.

    The accumulator chain is the accepted ``acc = acc + scaled`` in increasing
    block order; nothing is added out of order and nothing is tree-reduced. This is
    the proof schedule: no dynamic branch and no fast path, just the frozen helper
    applied twice in the accepted order.
    """
    columns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    active = columns < N
    if INDEX64:
        columns = columns.to(tl.int64)
    lanes = tl.arange(0, 32)
    accumulator = tl.full((ROWS,), 0.0, tl.float32)
    invalid = tl.full((ROWS,), False, tl.int1)
    for block in range(KBLOCKS):
        ac = tl.load(A + block * 32 + lanes).to(tl.int32)
        scale_a = tl.load(SA + block).to(tl.int32)
        if PACKED:
            byte_offsets = columns[:, None] * (K // 2) + block * 16 + (lanes[None, :] >> 1)
            weight = tl.load(B + byte_offsets, mask=active[:, None], other=0).to(tl.int32)
        else:
            offsets = columns[:, None] * K + block * 32 + lanes[None, :]
            weight = tl.load(B + offsets, mask=active[:, None], other=0).to(tl.int32)
        scale_offsets = (columns // BLOCK_ROWS) * KBLOCKS + block
        scale_b = tl.load(SB + scale_offsets, mask=active, other=0)
        if B_SCALE_FLOAT:
            scale_b = _original_scale_code(scale_b, FP4)
        else:
            scale_b = scale_b.to(tl.int32)
        fragment, bad = _float_fragment(ac, weight, lanes, PACKED)
        scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
        accumulator = accumulator + scaled
        invalid |= bad
    accumulator = tl.where(invalid, float("nan"), accumulator)
    tl.store(OUT + columns, accumulator, mask=active)