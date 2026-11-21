"""Ascend integer-significand fragment kernels for the exact M=1 decode GEMV.

The accepted fragment (``decode_block_gemv``) decodes every operand to FP32 and
then re-derives the same information the encoding already carries: an E4M3 byte
is a 4-bit significand and a 4-bit power, and an E2M1 nibble is a 2-bit
significand and a 2-bit power. This module keeps those two fields as integers and
performs the whole 32-term partial reduction with integer products, adds and
shifts, converting to FP32 only once per (column, K block) when the 13-bit window
fragment is rebuilt, and re-using the frozen ``_scale_e8m0`` tail unchanged.

The arithmetic is not an approximation of the accepted one; it is the same
statement written in the integers the accepted FP32 values are built from:

* every E4M3 value is exactly ``significand * 2**power`` with
  ``significand <= 15`` and every E2M1 value is exactly ``significand * 2**power``
  with ``significand <= 3``, so their product is exactly representable in FP32
  (``<= 45``) and no rounding happens between the two forms;
* the accepted kernel's ``(product * _pow2(-grid)).to(tl.int32)`` is therefore
  ``significand_product * 2**(power - grid)`` truncated toward zero, which is an
  exact left shift or a truncating right shift of a small integer;
* the 16-term sum of those integers is ``<= 2**20`` and is exact in FP32, so the
  same integer is what the accepted kernel sums and what the carry into the
  second half consumes;
* the second-half total and the window cut are truncations of that integer by a
  power of two, which the helper below performs with a sign-preserving shift;
* ``_grid_exponent``/``_pow2``/``_ilog2``/``_scale_e8m0`` are the frozen helpers,
  imported by name, so the sentinel, wrapping and two-rescale behaviours are the
  accepted ones rather than re-derivations.

Both halves keep the accepted anchor rule (nominal is the halo exponent, not the
true exponent), so the window and the truncation points are identical.

Only one FP32 conversion happens per fragment, and it is exact: the 13-bit
window significand times a power of two. Everything else on the fragment path is
integer.

Deliberately absent: no native FP4/FP8 instruction, no FP64, no float decode of
an individual operand, no log/exp, no reassociated K sum, no fused multiply-add.
The pass-2 accumulate kernel is the frozen one, so the K-block chain is
literally the accepted chain.
"""
from __future__ import annotations

import triton
import triton.language as tl

from ._ascend_block_gemm_kernels import (
    _grid_exponent,
    _ilog2,
    _original_scale_code,
    _pow2,
    _scale_e8m0,
)


@triton.jit
def _e4_fields(codes):
    """``(significand, power, nominal, negative, not_a_number)`` of E4M3 bytes.

    ``value = significand * 2**power`` exactly, with ``significand`` in ``[0, 15]``
    (subnormals carry ``power = -9``). ``nominal`` is the accepted halo exponent
    ``max(exponent - 7, -6)``; a zero code is the only code excluded from the
    anchor, and a NaN code keeps its (large) nominal because that is what the
    accepted ``_nominal_e4`` does -- the NaN is carried by the invalid flag and
    the zero significand, not by moving the anchor.
    """
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    nan = (codes & 127) == 127
    zero = (codes & 127) == 0
    significand = tl.where(exponent == 0, mantissa, 8 + mantissa)
    significand = tl.where(nan, 0, significand)
    power = tl.where(exponent == 0, -9, exponent - 10)
    nominal = tl.maximum(exponent - 7, -6)
    nominal = tl.where(zero, -1000, nominal)
    negative = (codes & 128) != 0
    return significand, power, nominal, negative, nan


@triton.jit
def _e2m1_fields(codes):
    """``(significand, power, nominal, negative)`` of E2M1 nibbles.

    The released magnitudes are ``0, 0.5, 1, 1.5, 2, 3, 4, 6``; each is exactly
    ``significand * 2**power`` with the two-bit significand ``magnitude < 2 ?
    magnitude : 2 + (magnitude & 1)`` and the power below. ``nominal`` equals the
    accepted ``bn`` ladder (``(magnitude >> 1) - 1``) and a zero magnitude sits
    below every anchor, so it contributes no exponent.
    """
    magnitude = codes & 7
    significand = tl.where(magnitude < 2, magnitude, 2 + (magnitude & 1))
    power = tl.where(magnitude < 2, -1, (magnitude >> 1) - 2)
    nominal = (magnitude >> 1) - 1
    nominal = tl.where(magnitude != 0, nominal, -1000)
    return significand, power, nominal, (codes & 8) != 0


@triton.jit
def _scale_fast(value, code):
    """One *unguarded* exact power-of-two rescale, plus the predicate that says whether it is.

    Returns ``(bits, safe)``: ``bits`` is the FP32 pattern after moving the exponent
    by ``code - 127`` and ``safe`` says, per lane, whether that move was exact and
    stayed inside the normal range. Nothing is branched on here, so this helper has
    no conditional result to return; the caller decides.

    ``bits`` is deliberately still *built* on unsafe lanes, with the new exponent
    clamped to ``[0, 255]`` so a speculative pattern can never overflow the shift.
    Those lanes' bits are unused, and the second rescale's predicate reads the
    actual first-intermediate exponent, which is what makes the two-step predicate
    exact rather than a guess about the composition.

    A signed zero keeps its sign; ``code == 255`` (a NaN scale) is never safe, so a
    zero under a NaN scale cannot be reported as a signed zero.
    """
    raw = value.to(tl.int32, bitcast=True)
    exponent = (raw >> 23) & 255
    new_exponent = exponent + code - 127
    magnitude = raw & 0x7fffffff
    zero = magnitude == 0
    safe = (code != 255) & (zero | ((exponent > 0) & (exponent < 255)
                                    & (new_exponent > 0) & (new_exponent < 255)))
    # Clamped so an unsafe lane's speculative shift amount stays in range; the
    # value is discarded because ``safe`` fails for the whole tile.
    clamped = tl.minimum(tl.maximum(new_exponent, 0), 255)
    # Sign and significand are reassembled from two literals that both fit in
    # int32: a single sign-and-magnitude mask would need a value above 2**31 - 1,
    # and this compiler refuses the wider-than-int32 promotion that forces. The
    # frozen ``_scale_e8m0`` builds the sign the same way.
    mantissa = raw & 0x7fffff
    sign = (raw >> 31) << 31
    bits = tl.where(zero, raw, sign | mantissa | (clamped << 23))
    return bits, safe


@triton.jit
def _integral_terms(significand_product, power, grid, negative, half):
    """The accepted ``(product * _pow2(-grid)).to(tl.int32)``, in integers.

    ``grid`` is the per-column window anchor; ``half`` masks the 16 lanes this
    half owns. A left shift is exact; a right shift truncates toward zero, which
    is the accepted cast's direction, and the sign is applied after the magnitude
    so a negative term truncates toward zero rather than toward negative infinity.

    Both shift amounts are clamped. The reachable range is small -- the anchor is
    at least the largest lane's ``power + 3``, so a left shift never exceeds 13 and
    a right shift never exceeds 4 -- and the clamp only guards the compiler.
    """
    shift = power - grid[:, None]
    left = tl.minimum(tl.maximum(shift, 0), 31)
    right = tl.minimum(tl.maximum(-shift, 1), 31)
    magnitude = tl.where(shift >= 0, significand_product << left, significand_product >> right)
    magnitude = tl.where(half, magnitude, 0)
    return tl.where(negative, -magnitude, magnitude)


@triton.jit
def _truncating_shift(value, shift):
    """``trunc(value * 2**shift)`` for an integer, in the accepted cast direction."""
    left = tl.minimum(tl.maximum(shift, 0), 31)
    right = tl.minimum(tl.maximum(-shift, 1), 31)
    magnitude = tl.abs(value)
    scaled = tl.where(shift >= 0, magnitude << left, magnitude >> right)
    return tl.where(value < 0, -scaled, scaled)


@triton.jit
def _fragment_scaled(ac, weight, lanes, PACKED: tl.constexpr):
    """One accepted fragment from **already loaded** operands.

    Pointer arithmetic and every ``tl.load`` live in the kernels that call this.
    An earlier revision loaded A (1-D), SA (scalar), B (2-D) and SB (1-D) through
    this helper and the Ascend compiler's ``OffsetAnalysis`` pass aborted combining
    pointer offsets of different ranks across the call boundary
    (``lhs.getRank() == rhs.getRank()``); keeping the fragment arithmetic, and only
    the arithmetic, behind the call is the shape the frozen fragment kernel already
    compiles on this stack.

    ``ac`` is the 32 activation bytes of this K block and ``weight`` the 2-D
    ``[columns, 32]`` weight codes -- packed nibbles or raw E4M3 bytes. Returns the
    **unscaled** 13-bit-window fragment and the invalid flag: the two E8M0
    rescales belong to the kernels, because a value returned out of a conditional
    inside a helper is what this compiler failed to legalize (an unresolved
    ``memref<16xf32> -> tensor<16xf32>`` materialization at that helper's return).
    """
    sig_a, pa, nom_a, neg_a, nan_a = _e4_fields(ac)
    if PACKED:
        # Low nibble is the even K element and the high nibble the odd one, which
        # is the released packed layout. The two nibbles are selected with two
        # *constant* shifts and a lane mask, never with a per-lane shift amount:
        # a varying shift was measured at 49.75 ms against 18.62 ms for the
        # constant form on the same packed group of four blocks.
        codes = tl.where((lanes[None, :] & 1) == 0, weight & 15, weight >> 4)
        sig_b, pb, nom_b, neg_b = _e2m1_fields(codes)
        # Every nibble is a valid E2M1 code, so the packed path has no invalid
        # weight lane; only the activation can be NaN.
        nan_b = weight * 0
    else:
        sig_b, pb, nom_b, neg_b, nan_b_e4 = _e4_fields(weight)
        nan_b = nan_b_e4.to(tl.int32)

    significand_product = sig_a[None, :] * sig_b
    power = pa[None, :] + pb
    nominal = nom_a[None, :] + nom_b
    negative = neg_a[None, :] ^ neg_b

    # First half: anchor from its own nominal exponents, exactly as accepted.
    first = tl.max(tl.where(lanes[None, :] < 16, nominal, -2000), 1)
    grid0 = _grid_exponent(first)
    total0 = tl.sum(_integral_terms(
        significand_product, power, grid0, negative, lanes[None, :] < 16), 1)

    # Second half: the first-half carry's exponent joins this half's window.
    carry = tl.where(total0 == 0, -1000, _ilog2(tl.abs(total0)) + grid0)
    second = tl.max(tl.where(lanes[None, :] >= 16, nominal, -2000), 1)
    second = tl.maximum(second, carry)
    grid1 = _grid_exponent(second)
    whole = tl.sum(_integral_terms(
        significand_product, power, grid1, negative, lanes[None, :] >= 16), 1)
    whole += _truncating_shift(total0, grid0 - grid1)

    # The accepted window cut: truncate the whole to 13 bits below its exponent.
    exponent = tl.where(whole == 0, -1000, _ilog2(tl.abs(whole)) + grid1)
    window = _grid_exponent(exponent)
    fragment = _truncating_shift(whole, grid1 - window).to(tl.float32) * _pow2(window)

    invalid = tl.sum(nan_b + nan_a[None, :].to(tl.int32), 1) != 0
    return fragment, invalid


@triton.jit
def integer_fragment_kernel(A, SA, B, SB, FRAG, BAD, N: tl.constexpr,
                            K: tl.constexpr, KBLOCKS: tl.constexpr,
                            BLOCK_ROWS: tl.constexpr, PACKED: tl.constexpr,
                            FP4: tl.constexpr, B_SCALE_FLOAT: tl.constexpr,
                            ROWS: tl.constexpr, GROUP: tl.constexpr,
                            KSPLIT: tl.constexpr, INDEX64: tl.constexpr):
    """Pass 1: one program owns ``ROWS`` columns and a strided run of K groups.

    ``GROUP`` K blocks are walked **serially** inside a program -- it is a static
    unroll count that bounds how much work one program takes and how many programs
    the grid needs, not a fused ``GROUP x ROWS`` tile. The only resident tile is
    the fragment's ``ROWS x 32``; that is why ``ROWS`` can be raised on its own
    later without changing the arithmetic. The grid is
    ``(ceil(N / ROWS), KSPLIT)`` and the total program count is bounded by the
    caller before launch.

    Every load is issued here, in this kernel, with the rank its own tensor has:
    the activation row is 1-D, the weight codes are 2-D, the two scale loads are
    scalar and 1-D. The frozen fragment kernel has exactly this shape and compiles
    on this backend; the arithmetic call below takes the loaded values only.
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
                # Widen before the ``block * N`` product, not after it: the
                # product is one of the reaches that can pass 2**31.
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
            fragment, bad = _fragment_scaled(ac, weight, lanes, PACKED)
            # The guarded fast path is decided HERE, in the kernel, and the two
            # schedules differ on purpose: this is the measured schedule, so it
            # takes the two sequential exact power-of-two rescalings when the whole
            # tile is safe for both, and the frozen generic helper otherwise. Only
            # *stores* cross the branch -- a vector value returned from inside a
            # conditional is what the compiler failed to legalize -- and the
            # invalid flag is stored outside it either way.
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
def integer_serial_kernel(A, SA, B, SB, OUT, N: tl.constexpr, K: tl.constexpr,
                          KBLOCKS: tl.constexpr, BLOCK_ROWS: tl.constexpr,
                          PACKED: tl.constexpr, FP4: tl.constexpr,
                          B_SCALE_FLOAT: tl.constexpr, ROWS: tl.constexpr,
                          INDEX64: tl.constexpr):
    """Single-pass variant: one program owns ``ROWS`` columns and every K block.

    The accumulator chain is the accepted ``acc = acc + scaled`` in increasing
    block order; nothing is added out of order and nothing is tree-reduced. All
    loads are issued here, for the same rank-analysis reason pass 1 explains.
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
        fragment, bad = _fragment_scaled(ac, weight, lanes, PACKED)
        # The serial schedule is the proof schedule: no dynamic branch and no fast
        # path, just the frozen helper applied twice in the accepted order. It is
        # the variant the grouped schedule's fast path has to agree with.
        scaled = _scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)
        accumulator = accumulator + scaled
        invalid |= bad
    accumulator = tl.where(invalid, float("nan"), accumulator)
    tl.store(OUT + columns, accumulator, mask=active)
