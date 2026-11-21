"""Prefetch-access grouped-K exact M=1 decode GEMV over original checkpoint bytes.

A **memory-access** variant of :mod:`decode_grouped_gemv`, on the same accepted
arithmetic. The measured starting point is that module's own record: at the
released ``2304 x 5120`` shape the grouped schedule took 18.62 ms reading packed
FP4 bytes against 12.63 ms for the identical fragments over an ``INT8 x 2``
operand, and a decode graph of it saved only about 0.2 ms -- so the gap is device
work, not host launch. One candidate is the access shape: the grouped kernel reads
one K block's 16 bytes per tile row, with two lanes sharing each byte, which is a
half-used 32-byte burst per row and a separate short burst per row.

This module therefore keeps every arithmetic statement and changes only how the
weight tile is fetched: one contiguous row of ``GROUP_K * 16`` raw bytes per
column (64 bytes at the default, two full 32-byte bursts), held as ``uint8`` until
after the local expansion. See
:mod:`_ascend_decode_prefetch_kernels` for the layout contract and the two-pass
arrangement, which is unchanged: pass 1 writes the frozen ``[K // 32, N]``
fragment scratch, pass 2 is the frozen ``decode_accumulate_kernel`` and adds the
fragments in strictly increasing K order.

Validated scope
---------------

Only the **packed FP4** entry point, :func:`fp4_packed_prefetch_decode_gemv`, is
validated on hardware. At the released ``2304 x 5120`` (w1) and ``5120 x 2304``
(w2) shapes its output was **bit-identical** to the grouped schedule -- 0 FP32
bit differences -- while taking **16.47 ms and 16.42 ms** against **18.60 ms and
18.11 ms** for the grouped kernels on the target build. The arithmetic is the
frozen fragment, so the whole difference is the memory access.

The two **dense** entry points are refused. The wide-row tile over a dense code
matrix is not validated: at ``1280 x 5120`` the FP8 form differed from the
accepted operator on **71 finite FP32 values**, while the grouped FP8 kernel
matched, so this schedule's dense load/reshape is the suspect and the cause is not
yet isolated. :func:`fp8_prefetch_decode_gemv` and
:func:`fp4_int8_prefetch_decode_gemv` therefore raise a named ``RuntimeError``
instead of returning a value that could differ; there is no silent fallback to the
old operator and no tolerance anywhere. Only the packed FP4 path may be wired into
a model.

Exactness of the packed path is structural rather than re-argued: the fragment, the
packed decode and both rescales are the frozen helpers imported by the kernel, and
the fragment scratch and its accumulator are the frozen ones. The CPU backend is
the frozen prototype's CPU oracle -- it proves operand normalisation and dispatch,
and *nothing* about the Ascend kernel; only the hardware parity test in
``tests/test_decode_prefetch_gemv.py`` does that.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch
from .block_gemm import _bytes, _dense
from .decode_block_gemv import (
    BLOCK,
    _backend,
    _device,
    _operands,
    _output,
    _prepare_scratch,
    _torch_reference,
    decode_gemv_scratch,
)
from .decode_grouped_gemv import (
    ACCUMULATE_COLUMNS,
    GROUP_K,
    MAX_TILE_ROWS,
    ROWS_N,
    _geometry,
    _power_of_two,
)

__all__ = [
    "decode_prefetch_gemv_scratch",
    "fp4_int8_prefetch_decode_gemv",
    "fp4_packed_prefetch_decode_gemv",
    "fp8_prefetch_decode_gemv",
]


def _refuse_unvalidated(entry, accepted, grouped):
    """Refuse an unvalidated schedule by name, never a silent substitution."""
    return RuntimeError(
        f"{entry} is not validated on hardware and is refused rather than returning a value that "
        "may differ: the wide-row tile over a dense code matrix mismatched the accepted operator on "
        "71 finite FP32 values at 1280x5120 (FP8) while the grouped kernel matched, so this "
        "schedule's dense load/reshape is unproven. The packed FP4 entry point is bit-exact and is "
        f"unaffected. Use {accepted} or {grouped} instead; no fallback is applied here and no "
        "tolerance is used anywhere."
    )


def _schedule(group_k, rows_n, accumulate_columns):
    """Validate the schedule knobs and return them normalized.

    Both tile sides must be powers of two because the flat row decomposes into
    ``(group, column)`` with a shift and a mask, and because the wide row lengths
    ``GROUP_K * 16`` / ``GROUP_K * 32`` passed to ``tl.arange`` have to be powers
    of two as well. The row budget is the same measured bound the grouped
    schedule uses: a 128-row tile overflows the 192 KiB unified buffer on the
    target build, 64 rows fits.
    """
    group_k = _power_of_two(group_k, "group_k", maximum=32)
    rows_n = _power_of_two(rows_n, "rows_n", maximum=64)
    accumulate_columns = _power_of_two(
        accumulate_columns, "accumulate_columns", maximum=1024
    )
    if group_k * rows_n > MAX_TILE_ROWS:
        raise ValueError(
            f"group_k * rows_n = {group_k * rows_n} exceeds the {MAX_TILE_ROWS}-row "
            f"per-program tile budget ({MAX_TILE_ROWS} rows is the largest that "
            "compiles: 128 rows overflows the 192 KiB unified buffer, 64 rows fits); "
            "use group_k * rows_n <= 64, e.g. group_k=4 with rows_n=16"
        )
    return group_k, rows_n, accumulate_columns


def decode_prefetch_gemv_scratch(n, k, *, device=None):
    """The reusable two-pass scratch, ``([K // 32, N], [K // 32, N])``.

    The fragment layout is the frozen prototype's, because pass 2 is its kernel;
    the column-major tile only changes which program writes an entry.
    """
    return decode_gemv_scratch(n, k, device=device)


def _needs_index64(n, k, blocks, rows_n, group_k):
    """Whether the offset arithmetic needs 64-bit indexing, from the **padded** reach.

    A masked tail lane still forms its address before the mask suppresses the
    access, so the padded tile extent is what can overflow -- the same reasoning
    the fused unpack launchers use for their pointer width. Deciding from the
    logical ``n`` would leave the last tile's dead lanes one block away from a
    wrapped index.
    """
    padded_n = math.ceil(n / rows_n) * rows_n
    padded_blocks = math.ceil(blocks / group_k) * group_k
    return max(padded_n * k, padded_blocks * n) >= 2**31


def _launch(a, sa, b, sb, out, fragments, invalid, *, fp4, packed, block_rows,
            b_scale_float, n, k, blocks, rows_n, group_k, group_stride,
            column_chunks, accumulate_columns):
    with _npu_launch(a):
        from ._ascend_decode_block_gemv_kernels import decode_accumulate_kernel
        from ._ascend_decode_prefetch_kernels import decode_prefetch_fragment_kernel

        # The widest weight offset is N * K (the packed form is narrower) and the
        # widest scratch offset KBLOCKS * N; both decided from the padded extent.
        index64 = _needs_index64(n, k, blocks, rows_n, group_k)
        decode_prefetch_fragment_kernel[(column_chunks, group_stride)](
            a, sa, b, sb, fragments, invalid,
            n, k, blocks, block_rows, fp4, packed, b_scale_float,
            rows_n, group_k.bit_length() - 1, group_k, group_k * rows_n,
            group_stride, index64,
        )
        decode_accumulate_kernel[(math.ceil(n / accumulate_columns),)](
            fragments, invalid, out, n, blocks, accumulate_columns, index64
        )
    return out


def _run(a, sa, b, sb, *, fp4, packed, block_rows, b_scale_float, n, k, device,
         out, scratch, backend, group_k, rows_n, accumulate_columns):
    """Validate, then dispatch the CPU oracle or the fused prefetch schedule."""
    group_k, rows_n, accumulate_columns = _schedule(
        group_k, rows_n, accumulate_columns
    )
    blocks = k // BLOCK
    protected = (a, sa, b, sb)
    column_chunks, _, group_stride = _geometry(
        n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n
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
    fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    out = _output(out, n, device, protected + (fragments, invalid))
    return _launch(
        a, sa, b, sb, out, fragments, invalid, fp4=fp4, packed=packed,
        block_rows=block_rows, b_scale_float=b_scale_float, n=n, k=k, blocks=blocks,
        rows_n=rows_n, group_k=group_k, group_stride=group_stride,
        column_chunks=column_chunks, accumulate_columns=accumulate_columns,
    )


def fp4_int8_prefetch_decode_gemv(
    a_codes, a_scale, b_int8_x2, b_half_scale, *,
    group_k=GROUP_K, rows_n=ROWS_N, accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """Exact ``INT8 = 2 * E2M1`` weights with the original half FP32 scale grid.

    **Refused: not validated on hardware.** Shares the dense tile branch whose FP8
    form mismatched the accepted operator, and it was never measured itself; see
    the module docstring. Use ``block_gemm.fp4_int8_block_gemm`` or
    ``decode_grouped_gemv.fp4_int8_grouped_decode_gemv``.
    """
    raise _refuse_unvalidated(
        "fp4_int8_prefetch_decode_gemv",
        "block_gemm.fp4_int8_block_gemm",
        "decode_grouped_gemv.fp4_int8_grouped_decode_gemv",
    )


def fp4_packed_prefetch_decode_gemv(
    a_codes, a_scale, b_packed, b_e8m0, *,
    group_k=GROUP_K, rows_n=ROWS_N, accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """Original packed E2M1 weight bytes with the original E8M0 byte grid.

    ``b_packed`` is ``[N, K // 2]`` ``uint8`` (or an ``int8`` container), one byte
    per adjacent K pair, low nibble first -- the checkpoint's own layout. ``b_e8m0``
    is the original ``[N, K // 32]`` raw E8M0 grid, **not** halved: the kernel
    widens the nibbles to ``2 * E2M1`` and pairs them with the half-scale it derives
    from that grid, so the loaded tile is the original value under the original
    scale. Nothing is requantized, and no ``INT8 x 2`` or FP32 half-scale table is
    ever materialized.
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
        group_k=group_k, rows_n=rows_n, accumulate_columns=accumulate_columns,
    )


def fp8_prefetch_decode_gemv(
    a_codes, a_scale, b_codes, b_scale, *,
    block_rows=32, group_k=GROUP_K, rows_n=ROWS_N,
    accumulate_columns=ACCUMULATE_COLUMNS,
    backend="auto", out=None, scratch=None,
):
    """E4M3FN weights with a square or row-wise original E8M0 grid.

    **Refused: not validated on hardware.** At ``1280 x 5120`` this schedule's
    output differed from the accepted operator on 71 finite FP32 values while the
    grouped kernel matched; see the module docstring. Use
    ``block_gemm.fp8_block_gemm`` or
    ``decode_grouped_gemv.fp8_grouped_decode_gemv``.
    """
    raise _refuse_unvalidated(
        "fp8_prefetch_decode_gemv",
        "block_gemm.fp8_block_gemm",
        "decode_grouped_gemv.fp8_grouped_decode_gemv",
    )
