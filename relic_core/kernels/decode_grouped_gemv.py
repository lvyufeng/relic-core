"""Grouped-K exact M=1 decode block GEMV, with an in-register packed serial variant.

This is a **schedule** layer over the accepted block GEMM's arithmetic, not a new
numerical path. The frozen prototype in :mod:`decode_block_gemv` proved the
two-pass split (parallel K fragments, then the accepted sequential add) is exact,
but measured it slower than the accepted operator for FP4 on the target build
(FP4 ``w1`` 12.73 -> 49.64 ms, ``w2`` 11.38 -> 49.87 ms, while FP8 ``wq_a`` moved
6.24 -> 6.90 ms). Those two FP4 cases both launched 23040 one-fragment programs,
which is the shape this module exists to shrink; the FP8 case had fewer, so the
fragment size is *a* difference and no single cause is claimed.

Two variants, selected explicitly:

``variant="grouped"``
    Still two passes, still the accepted fragment and the accepted sequential
    add, but one pass-1 program owns ``GROUP_K`` K blocks and ``ROWS_N`` columns
    as one flat 2-D tile. Fewer, larger programs; the K scratch and the strict
    increasing-K accumulation of the accepted kernel are unchanged.

``variant="serial"``
    The accepted kernel's own shape -- one program per column tile, a serial loop
    over every K block in increasing order -- reading the checkpoint's **packed
    E2M1 nibbles** and **raw E8M0 bytes** in register. This drops both
    preparation passes the two-pass form needs (the global ``INT8 x2`` widening
    and the FP32 half-scale table) and needs no K scratch, at the cost of the K
    chain being serial again.

Exactness is structural: the fragment reduction, the value decoders and both
rescales are the frozen helpers of ``_ascend_block_gemm_kernels``, and the
packed-nibble widening is the frozen ``_ascend_unpack_kernels._e2m1_x2``. What
changes between the variants is only how many fragments a program computes and
where the weight bytes are read from. The NPU tests assert bit equality against
the accepted ``block_gemm`` operator and against the frozen two-pass prototype;
the CPU path **is** the frozen prototype's CPU oracle, so it cannot disagree with
the accepted operator by construction.

Nothing here is wired into a model or serving path, and **no performance claim is
made**. Only an actual measured decode on the target build can say whether either
variant is faster than the accepted operator, and which of the two wins.

One optimisation deliberately not implemented: an "all-normal scales, no cut
needed" fast path for the rescale/truncation helpers. It is only legal behind a
common-block predicate that proves every scale byte and every intermediate
exponent is normal, and that predicate has not been proven against the reference
edge cases yet, so every block still calls the frozen helpers in full. Relaxing
it without that proof would change the output on exactly the inputs the frozen
helpers exist to handle.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch
from .block_gemm import _bytes, _dense
from .decode_block_gemv import (
    BLOCK,
    MAX_PROGRAMS,
    _backend,
    _device,
    _operands,
    _output,
    _prepare_scratch,
    _torch_reference,
    decode_gemv_scratch,
)

__all__ = [
    "decode_grouped_gemv_scratch",
    "fp4_int8_grouped_decode_gemv",
    "fp4_packed_grouped_decode_gemv",
    "fp8_grouped_decode_gemv",
]

#: K blocks per pass-1 program in the grouped variant. 4 blocks x 16 columns is
#: 64 tile rows, and it is the fastest grouped setting measured on the target
#: build (18.62 ms at the released ``2304 x 5120`` shape, against 22.54 ms at
#: ``GROUP_K = 1``); it is also inside the unified-buffer budget below.
GROUP_K = 4
#: Columns per program: the grouped tile's column side, and the serial variant's
#: whole tile width.
ROWS_N = 16
#: Columns per pass-2 program; the frozen accumulate kernel's own parameter.
ACCUMULATE_COLUMNS = 64
#: Upper bound on ``GROUP_K * ROWS_N`` -- how many tile rows one pass-1 program
#: holds -- and the reason the default is 4 x 16 rather than 8 x 16.
#:
#: **Measured, not guessed.** A ``GROUP_K = 8``, ``ROWS_N = 16`` tile (128 rows)
#: does not compile on the target build:
#:
#:     ub overflow, requires 2675968 bits while 1572864 bits available
#:
#: The compiler auto-disabled code motion and multi-buffering and retried, and it
#: still failed, so this is a hard 192 KiB unified-buffer ceiling rather than a
#: flag that can be turned off. 64 rows passes and 128 rows fails; the ratio
#: between them (1.70x over budget at 128) puts the breaking point near 75 rows
#: *if* the requirement grows linearly in tile rows, which has not been measured,
#: so the bound here is the largest power-of-two-safe value below it: **64**. A
#: geometry above it is refused by name before any allocation, because the
#: alternative is a default configuration that fails to compile on the target.
MAX_TILE_ROWS = 64
#: The two supported schedules. Spelled out because a typo must not silently pick
#: one of them.
VARIANTS = ("grouped", "serial")


def _power_of_two(value, name, *, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")
    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _schedule(variant, group_k, rows_n, accumulate_columns):
    """Validate the schedule knobs and return them normalized.

    Both tile sides must be powers of two because the flattened tile row
    decomposes into ``(group, column)`` with a shift and a mask -- division by a
    non-power-of-two constant has no proven lowering on this backend. The serial
    variant still validates ``group_k`` so a caller can sweep one parameter set
    across both variants, but only ``rows_n`` affects its geometry.
    """
    if not isinstance(variant, str):
        raise TypeError(f"variant must be a string, got {type(variant).__name__}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    group_k = _power_of_two(group_k, "group_k", maximum=32)
    rows_n = _power_of_two(rows_n, "rows_n", maximum=64)
    accumulate_columns = _power_of_two(
        accumulate_columns, "accumulate_columns", maximum=1024
    )
    if variant == "grouped" and group_k * rows_n > MAX_TILE_ROWS:
        # The serial variant has no group tile at all -- one column tile, one K
        # loop -- so ``group_k`` does not enter its unified-buffer requirement and
        # must not be part of its budget either.
        raise ValueError(
            f"group_k * rows_n = {group_k * rows_n} exceeds the {MAX_TILE_ROWS}-row "
            f"per-program tile budget ({MAX_TILE_ROWS} rows is the largest that "
            "compiles: 128 rows overflows the 192 KiB unified buffer, 64 rows fits); "
            "use group_k * rows_n <= 64, e.g. group_k=4 with rows_n=16"
        )
    return variant, group_k, rows_n, accumulate_columns


def _geometry(n, blocks, *, variant, group_k, rows_n):
    """``(column chunks, K groups, group stride)`` whose product fits the device cap.

    Ascend rejects a launch whose **total** block count exceeds 65535, so the
    second grid axis is not a way around it either. The grouped variant therefore
    gives pass 1 a grid of ``(column chunks, group stride)`` with
    ``column chunks * group stride <= MAX_PROGRAMS`` and lets each program stride
    the remaining K groups; when the whole group axis fits, the stride is the
    group count and each program owns exactly one group.

    The check runs before any allocation or device work, so the CPU reference
    path rejects an unlaunchable geometry too rather than only the fused one.
    """
    column_chunks = math.ceil(n / rows_n)
    if column_chunks > MAX_PROGRAMS:
        raise ValueError(
            f"N / rows_n = {column_chunks} exceeds the {MAX_PROGRAMS} total-program "
            f"cap; pass a larger rows_n or split the operand"
        )
    if variant == "serial":
        return column_chunks, 1, 1
    group_chunks = math.ceil(blocks / group_k)
    group_stride = max(1, min(group_chunks, MAX_PROGRAMS // column_chunks))
    assert column_chunks * group_stride <= MAX_PROGRAMS
    return column_chunks, group_chunks, group_stride


def decode_grouped_gemv_scratch(n, k, *, device=None):
    """The reusable two-pass scratch, ``([K // 32, N], [K // 32, N])``.

    Identical to :func:`decode_block_gemv.decode_gemv_scratch` -- the fragment
    layout is the frozen prototype's, because pass 2 is its kernel -- and exposed
    under this module's name so a caller does not have to reach across modules
    for it. The serial variant writes no scratch; a pair passed to it is
    validated and then left untouched.
    """
    return decode_gemv_scratch(n, k, device=device)


def _launch_grouped(a, sa, b, sb, out, fragments, invalid, *, fp4, packed,
                   block_rows, b_scale_float, n, k, blocks, rows_n, group_k,
                   group_stride, column_chunks, accumulate_columns):
    with _npu_launch(a):
        from ._ascend_decode_block_gemv_kernels import decode_accumulate_kernel
        from ._ascend_decode_grouped_kernels import decode_grouped_fragment_kernel

        # The reaches that can pass 2**31: the widest weight offset is N * K, the
        # widest scratch offset KBLOCKS * N. Widening is chosen from them here and
        # applied to the indices before their products inside the kernels.
        index64 = max(n * k, n * blocks) >= 2**31
        tile = group_k * rows_n
        decode_grouped_fragment_kernel[(column_chunks, group_stride)](
            a, sa, b, sb, fragments, invalid,
            n, k, blocks, block_rows, fp4, packed, b_scale_float,
            rows_n, rows_n.bit_length() - 1, group_k, tile, group_stride, index64,
        )
        decode_accumulate_kernel[(math.ceil(n / accumulate_columns),)](
            fragments, invalid, out, n, blocks, accumulate_columns, index64
        )
    return out


def _launch_serial(a, sa, b, sb, out, *, fp4, packed, block_rows, b_scale_float,
                   n, k, blocks, rows_n, column_chunks):
    with _npu_launch(a):
        from ._ascend_decode_grouped_kernels import decode_serial_gemv_kernel

        index64 = max(n * k, n * blocks) >= 2**31
        decode_serial_gemv_kernel[(column_chunks,)](
            a, sa, b, sb, out,
            n, k, blocks, block_rows, fp4, packed, b_scale_float, rows_n, index64,
        )
    return out


def _run(a, sa, b, sb, *, fp4, packed, block_rows, b_scale_float, n, k, device,
         out, scratch, backend, variant, group_k, rows_n, accumulate_columns):
    """Validate, then dispatch the CPU oracle or one of the two fused schedules."""
    variant, group_k, rows_n, accumulate_columns = _schedule(
        variant, group_k, rows_n, accumulate_columns
    )
    blocks = k // BLOCK
    protected = (a, sa, b, sb)
    column_chunks, _, group_stride = _geometry(
        n, blocks, variant=variant, group_k=group_k, rows_n=rows_n
    )
    if backend == "torch":
        if scratch is not None:
            # Validated even though this path does not write it, so a caller that
            # switches backend for testing gets the same shape errors either way.
            _prepare_scratch(scratch, n, blocks, device, protected)
        out = _output(out, n, device, protected)
        return _torch_reference(
            a, sa, b, sb, fp4=fp4, packed=packed, block_rows=block_rows, out=out
        )
    if variant == "serial":
        if scratch is not None:
            _prepare_scratch(scratch, n, blocks, device, protected)
        out = _output(out, n, device, protected)
        return _launch_serial(
            a, sa, b, sb, out, fp4=fp4, packed=packed, block_rows=block_rows,
            b_scale_float=b_scale_float, n=n, k=k, blocks=blocks, rows_n=rows_n,
            column_chunks=column_chunks,
        )
    fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    out = _output(out, n, device, protected + (fragments, invalid))
    return _launch_grouped(
        a, sa, b, sb, out, fragments, invalid, fp4=fp4, packed=packed,
        block_rows=block_rows, b_scale_float=b_scale_float, n=n, k=k, blocks=blocks,
        rows_n=rows_n, group_k=group_k, group_stride=group_stride,
        column_chunks=column_chunks, accumulate_columns=accumulate_columns,
    )


def fp4_int8_grouped_decode_gemv(
    a_codes, a_scale, b_int8_x2, b_half_scale, *,
    variant="grouped", group_k=GROUP_K, rows_n=ROWS_N,
    accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """Exact ``INT8 = 2 * E2M1`` weights with the original half FP32 scale grid.

    The operand contract is the frozen :func:`decode_block_gemv.fp4_int8_decode_gemv`
    one: ``b_int8_x2`` is ``[N, K]`` ``int8`` holding exactly twice each E2M1 value
    and ``b_half_scale`` is ``[N, K // 32]`` FP32 holding the original E8M0 grid
    divided by two. The pair is required together because mixing doubled values
    with the undivided grid would halve every K block's contribution.

    This variant exists for parity: it lets the grouped and serial schedules be
    checked against the accepted operator without the packed widening in the way.
    """
    a, sa, k = _operands(a_codes, a_scale)
    _dense(b_int8_x2, "FP4 INT8-times-two values", (torch.int8,))
    _dense(b_half_scale, "FP4 original half-scales", (torch.float32,))
    n, blocks = b_int8_x2.shape[0], k // BLOCK
    if n <= 0 or b_int8_x2.shape[1] != k:
        raise ValueError(f"FP4 INT8-times-two values must be [N, {k}] with positive N")
    if tuple(b_half_scale.shape) != (n, blocks):
        raise ValueError(f"FP4 half-scales must be [N, {blocks}] and hold the original half grid")
    device = _device(a, sa, b_int8_x2, b_half_scale)
    backend = _backend(backend, device)
    return _run(
        a, sa, b_int8_x2, b_half_scale, fp4=True, packed=False, block_rows=1,
        b_scale_float=True, n=n, k=k, device=device, out=out, scratch=scratch,
        backend=backend, variant=variant, group_k=group_k, rows_n=rows_n,
        accumulate_columns=accumulate_columns,
    )


def fp4_packed_grouped_decode_gemv(
    a_codes, a_scale, b_packed, b_e8m0, *,
    variant="grouped", group_k=GROUP_K, rows_n=ROWS_N,
    accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """Original packed E2M1 weight bytes with the original E8M0 byte grid.

    ``b_packed`` is ``[N, K // 2]`` ``uint8`` (or an ``int8`` container), one byte
    per adjacent K pair, low nibble first -- the checkpoint's own layout, the one
    ``unpack_fp4_to_int8`` reads. ``b_e8m0`` is the original ``[N, K // 32]`` raw
    E8M0 grid, **not** halved: the kernel widens the nibbles to ``2 * E2M1`` and
    pairs them with the half-scale it derives from that grid, so the loaded tile is
    the original value under the original scale. Nothing is requantized, and for
    ``variant="serial"`` neither the ``INT8 x2`` widening nor the FP32 half-scale
    table is ever materialized.
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
        variant=variant, group_k=group_k, rows_n=rows_n,
        accumulate_columns=accumulate_columns,
    )


def fp8_grouped_decode_gemv(
    a_codes, a_scale, b_codes, b_scale, *,
    block_rows=32, variant="grouped", group_k=GROUP_K, rows_n=ROWS_N,
    accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """E4M3FN weights with a square or row-wise original E8M0 grid.

    ``b_scale`` may be raw E8M0 bytes, ``[ceil(N / block_rows), K // 32]``, or the
    exact FP32 decoded powers of that grid in the same shape. ``block_rows`` is 32
    for the square dense grid and 1 for the row-scaled Engram-style grid; it is part
    of the contract rather than inferred, because inferring it would make a wrong
    grid silently legal.
    """
    a, sa, k = _operands(a_codes, a_scale)
    b = _bytes(b_codes, "weight codes")
    n, blocks = b.shape[0], k // BLOCK
    if n <= 0 or b.shape[1] != k:
        raise ValueError(f"weight codes must be [N, {k}] with positive N")
    if block_rows not in {1, 32}:
        raise ValueError("weight block_rows must be 1 or 32")
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
        scratch=scratch, backend=backend, variant=variant, group_k=group_k,
        rows_n=rows_n, accumulate_columns=accumulate_columns,
    )