"""Exact M=1 decode block GEMV with integer significand arithmetic on Ascend.

This is the same reduction as :mod:`relic_core.kernels.block_gemm` -- the explicit
two-half block-32 rule with a nominal-exponent anchor, a 13-bit truncation window,
a carry from the first half into the second, a final window cut, two sequential
E8M0 rescales and an ordered K-block accumulation -- written so the per-term work
is done on the integers the encoded operands already are instead of floating
decode of every operand.

Why this exists: the accepted kernel decodes each E4M3 byte and each E2M1 code to
FP32 per element and then derives ``product * 2**-grid`` in FP32 before casting to
``int32``. An E4M3 value is ``significand * 2**power`` with ``significand <= 15``
and an E2M1 value is ``significand * 2**power`` with ``significand <= 3``, so the
product significand is ``<= 45`` and the term the accepted kernel computes is an
exact shift of that small integer. Doing it that way removes the per-element float
decode, the float product tile and the float-to-int cast; it does not change a
single bit of the result, and the module keeps the frozen scale tail
(``_scale_e8m0`` twice, in the accepted order) and the frozen pass-2 add chain
literally imported rather than reimplemented.

The equivalence is exact rather than approximate:

* ``significand_a * significand_b <= 45``, so the product is exact in FP32 as well
  as in the integer;
* ``power_a + power_b`` spans ``[-10, 6]``, so no operand or product is subnormal
  and no scaling by a window power rounds;
* a 16-term partial sum of those terms is ``<= 2**20``, exact in FP32, which is
  what lets the carry be taken from the integer instead of from a float;
* the window cut is a truncation of that integer by a power of two, and truncation
  toward zero is the accepted cast's direction, so the sign is applied after the
  magnitude rather than to a negative shift.

Two schedules are provided and they compute the same fragments:

* ``variant="grouped"`` (default) -- pass 1 computes each ``(column, K block)``
  fragment in a program that owns a column run and a group of K blocks; pass 2 is
  the frozen ``decode_accumulate_kernel``, which adds the fragments in increasing
  block order. This is the schedule that gives a decode step enough programs.
* ``variant="serial"`` -- one program owns a column run and walks every K block
  itself, accumulating in order. No scratch, less launch work, fewer programs;
  useful for small N or as the reference schedule.

Neither variant reassociates the K sum, fuses a multiply-add, requantizes an
operand, or uses native FP4/FP8 or FP64 arithmetic. The CPU backend delegates to
the frozen :func:`decode_block_gemv.fp4_packed_decode_gemv` /
:func:`decode_block_gemv.fp8_decode_gemv`, which in turn delegate to the accepted
block GEMM, so the oracle for this operator is the accepted operator rather than a
second transcription of the rule.

This module makes **no performance claim**, and the measurement that was taken
says the opposite of fast: on the target build the exact integer fragment path is
slower than the frozen FP32 fragment path (62.90 ms grouped, 85.40 ms serial,
against 17.90 ms for the packed FP32 schedule) even though it is bit-identical.
The integer significand arithmetic buys precision, not speed, on this backend --
an int32 multiply and a variable per-lane shift are not cheaper there than an
FP32 multiply. Nothing in the repository binds this operator into a model or a
serving path, and it must not be bound as the fast path without a new
measurement.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch
from .block_gemm import _bytes, _dense
from .decode_block_gemv import (
    ACCUMULATE_COLUMNS,
    BLOCK,
    MAX_PROGRAMS,
    _backend,
    _device,
    _operands,
    _output,
    _prepare_scratch,
    decode_gemv_scratch,
    fp4_packed_decode_gemv,
    fp8_decode_gemv,
)

__all__ = [
    "fp4_packed_integer_decode_gemv",
    "fp8_integer_decode_gemv",
    "integer_decode_gemv_scratch",
]

#: Columns per program. 16 is the column tile the accepted block GEMM already
#: compiles; it is a parameter because the best value is a build measurement.
#: A program holds one ``ROWS_N x 32`` fragment tile, so this is the only axis
#: that sizes resident work; it may be raised up to `ROWS_N_MAX` on its own.
DEFAULT_ROWS_N = 16
#: K blocks a program walks serially before it reads its next group. This is a
#: loop count, not a second tile axis: the fragment tile stays ``ROWS_N x 32``.
DEFAULT_GROUP_K = 4
#: Largest ``ROWS_N`` the fragment tile is allowed to ask for. 32 is refused by
#: name rather than compiled: on the target backend a 32-column fragment needed
#: 3,440,640 bits of unified buffer against the 1,572,864-bit (192 KiB) budget,
#: and disabling code motion / multi-buffering did not bring it under. 16 is the
#: largest tile the measured build actually runs.
ROWS_N_MAX = 16
#: Largest serial group a program may walk in one visit.
GROUP_K_MAX = 32

_VARIANTS = ("auto", "serial", "grouped")


def integer_decode_gemv_scratch(n, k, *, device=None):
    """The two-pass scratch, re-exported from the frozen decoder: ``[K // 32, N]``.

    A caller running the grouped variant every step can keep one pair per
    geometry; a ``variant="serial"`` call accepts it too so both schedules take
    the same argument.
    """
    return decode_gemv_scratch(n, k, device=device)


def _tile(rows_n, group_k):
    """Validate the two schedule knobs before anything is allocated.

    Both are **powers of two**: ``rows_n`` sizes a ``tl.arange`` lane vector, which
    this backend rejects for a non-power-of-two extent, and ``group_k`` is a static
    unroll count. Refusing them here means a typo fails on the CPU reference path
    instead of as a compile error on a leased card.
    """
    for name, value, limit in (
        ("rows_n", rows_n, ROWS_N_MAX), ("group_k", group_k, GROUP_K_MAX)
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two, got {value}")
        if value > limit:
            if name == "rows_n":
                raise ValueError(
                    f"rows_n = {value} exceeds the supported maximum {limit}: a "
                    "32-column fragment needs more unified buffer than this backend has, "
                    "measured at 3,440,640 bits against a 1,572,864-bit budget"
                )
            raise ValueError(f"{name} = {value} exceeds the supported maximum {limit}")
    return rows_n, group_k


def _variant(variant, blocks):
    if not isinstance(variant, str):
        raise TypeError(f"variant must be a string, got {type(variant).__name__}")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    if variant == "auto":
        # A grouped schedule only pays when there is more than one block to split;
        # otherwise the serial program is the same work without a second pass.
        return "grouped" if blocks >= 2 else "serial"
    return variant


def _index_width(n, k, blocks):
    """Whether the pointer index has to be 64-bit for this geometry.

    ``N * K`` and ``KBLOCKS * N`` are the two reaches a launch derives addresses
    from, and both are products that can pass ``2**31`` while every individual
    axis is small. The decision is made once, from those products, before any
    arithmetic runs.
    """
    return max(n * k, n * blocks) >= 2**31


def _integer_grids(n, blocks, rows_n, group_k):
    """Grid shapes for both schedules, bounded by the *total* program cap.

    Ascend refuses a launch whose total block count exceeds 65535 (the earlier
    one-dimensional "grid should be less than 65536" failure was that limit), so
    pass 1 splits the block groups instead of growing an axis. The check runs
    before any allocation, so the CPU path rejects an impossible geometry too.
    """
    chunks = math.ceil(n / rows_n)
    if chunks > MAX_PROGRAMS:
        raise ValueError(
            f"N / rows_n = {chunks} exceeds the {MAX_PROGRAMS} total-program cap; "
            "pass a larger rows_n or split the operand"
        )
    groups = math.ceil(blocks / group_k)
    ksplit = max(1, min(groups, MAX_PROGRAMS // chunks))
    assert chunks * ksplit <= MAX_PROGRAMS
    return chunks, groups, ksplit


def _run(a, sa, b, sb, *, fp4, packed, block_rows, b_scale_float, n, k, device,
         out, scratch, backend, rows_n, group_k, variant):
    rows_n, group_k = _tile(rows_n, group_k)
    if fp4 and not packed:
        # ``INT8 = 2 * E2M1`` storage is served by the frozen entry point, which
        # already has its own oracle; this operator reads the packed released
        # nibbles, so that combination is refused rather than compiled as dead
        # device code nobody exercises.
        raise ValueError(
            "integer decode GEMV reads packed FP4 nibbles; use the frozen INT8-times-two "
            "entry point for expanded INT8 storage"
        )
    blocks = k // BLOCK
    variant = _variant(variant, blocks)
    protected = (a, sa, b, sb)
    chunks, groups, ksplit = _integer_grids(n, blocks, rows_n, group_k)
    if backend == "torch":
        # The frozen decoder validates a supplied scratch even on the CPU path, so
        # a caller that switches backend for testing gets the same errors.
        _prepare_scratch(scratch, n, blocks, device, protected)
        target = _output(out, n, device, protected)
        if packed:
            return fp4_packed_decode_gemv(a, sa, b, sb, backend="torch", out=target)
        if fp4:
            raise ValueError("use the frozen INT8-times-two entry point for that storage")
        return fp8_decode_gemv(
            a, sa, b, sb, block_rows=block_rows, backend="torch", out=target
        )
    fragments, invalid = (None, None)
    if variant == "grouped":
        fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    elif scratch is not None:
        # A supplied scratch is validated even though the serial schedule does not
        # write it, so a caller that switches schedule for testing gets the same
        # shape/overlap errors either way -- the same rule the frozen decoder uses.
        _prepare_scratch(scratch, n, blocks, device, protected)
    target = _output(out, n, device, protected if fragments is None else protected + (fragments, invalid))
    with _npu_launch(a):
        from ._ascend_integer_decode_kernels import (
            integer_fragment_kernel,
            integer_serial_kernel,
        )

        index64 = _index_width(n, k, blocks)
        common = (a, sa, b, sb)
        flags = (k, blocks, block_rows, packed, fp4, b_scale_float, rows_n)
        if variant == "grouped":
            from ._ascend_decode_block_gemv_kernels import decode_accumulate_kernel

            integer_fragment_kernel[(chunks, ksplit)](
                *common, fragments, invalid, n, *flags, group_k, ksplit, index64
            )
            decode_accumulate_kernel[(math.ceil(n / ACCUMULATE_COLUMNS),)](
                fragments, invalid, target, n, blocks, ACCUMULATE_COLUMNS, index64
            )
        else:
            assert chunks <= MAX_PROGRAMS
            integer_serial_kernel[(chunks,)](
                *common, target, n, *flags, index64
            )
    return target


def fp4_packed_integer_decode_gemv(
    a_codes, a_scale, b_packed, b_e8m0, *,
    backend="auto", out=None, scratch=None,
    rows_n=DEFAULT_ROWS_N, group_k=DEFAULT_GROUP_K, variant="auto",
):
    """Original packed E2M1 weight bytes with the original E8M0 byte grid.

    ``b_packed`` is ``[N, K // 2]`` ``uint8`` (an ``int8`` container is accepted),
    one byte per adjacent K pair, low nibble first -- the checkpoint's own layout.
    ``b_e8m0`` is the original ``[N, K // 32]`` raw E8M0 grid, **not** halved. The
    kernel never widens a nibble to a float and never builds an expanded INT8
    weight: the nibble's own significand and power are the product terms.

    ``variant`` selects the schedule (``"auto"``, ``"serial"``, ``"grouped"``) and
    ``rows_n`` / ``group_k`` size its tile; the arithmetic is identical for every
    choice.
    """
    a, sa, k = _operands(a_codes, a_scale)
    if not isinstance(b_packed, torch.Tensor):
        raise TypeError("packed FP4 weight bytes must be a torch.Tensor")
    _dense(b_packed, "packed FP4 weight bytes", (torch.uint8, torch.int8))
    b = b_packed if b_packed.dtype == torch.uint8 else b_packed.view(torch.uint8)
    n, blocks = b.shape[0], k // BLOCK
    if n <= 0 or b.shape[1] != k // 2:
        raise ValueError(f"packed FP4 weight bytes must be [N, {k // 2}], two K codes per byte")
    sb = _bytes(b_e8m0, "FP4 E8M0 weight scales", scale=True)
    if tuple(sb.shape) != (n, blocks):
        raise ValueError(f"FP4 E8M0 scales must be [N, {blocks}] and hold the original grid")
    device = _device(a, sa, b, sb)
    backend = _backend(backend, device)
    return _run(
        a, sa, b, sb, fp4=True, packed=True, block_rows=1, b_scale_float=False,
        n=n, k=k, device=device, out=out, scratch=scratch, backend=backend,
        rows_n=rows_n, group_k=group_k, variant=variant,
    )


def fp8_integer_decode_gemv(
    a_codes, a_scale, b_codes, b_scale, *,
    block_rows=32, backend="auto", out=None, scratch=None,
    rows_n=DEFAULT_ROWS_N, group_k=DEFAULT_GROUP_K, variant="auto",
):
    """E4M3FN weight codes with a square or row-wise original E8M0 grid (M=1).

    ``b_scale`` may be raw E8M0 bytes, ``[ceil(N / block_rows), K // 32]``, or the
    exact FP32 decoded powers of that grid in the same shape. ``block_rows`` is 32
    for the square dense grid and 1 for the row-scaled Engram-style grid; it is
    part of the contract rather than inferred, so a wrong grid cannot silently
    become legal.
    """
    a, sa, k = _operands(a_codes, a_scale)
    b = _bytes(b_codes, "weight codes")
    n, blocks = b.shape[0], k // BLOCK
    if n <= 0 or b.shape[1] != k:
        raise ValueError(f"weight codes must be [N, {k}] with positive N")
    if block_rows not in {1, 32}:
        raise ValueError("weight block_rows must be 1 or 32")
    if not isinstance(b_scale, torch.Tensor):
        raise TypeError(f"weight scales must be a torch.Tensor, got {type(b_scale).__name__}")
    if b_scale.dtype == torch.float32:
        sb = _dense(b_scale, "weight scales", (torch.float32,))
        b_scale_float = True
    else:
        sb = _bytes(b_scale, "weight scales", scale=True)
        b_scale_float = False
    if tuple(sb.shape) != (math.ceil(n / block_rows), blocks):
        raise ValueError("weight scale grid disagrees with its block-row/K layout")
    device = _device(a, sa, b, sb)
    backend = _backend(backend, device)
    return _run(
        a, sa, b, sb, fp4=False, packed=False, block_rows=block_rows,
        b_scale_float=b_scale_float, n=n, k=k, device=device, out=out,
        scratch=scratch, backend=backend, rows_n=rows_n, group_k=group_k,
        variant=variant,
    )