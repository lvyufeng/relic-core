"""Exact M=1 decode block GEMV: parallel K fragments, sequential K accumulation.

The accepted block GEMM (``block_gemm.fp4_int8_block_gemm`` / ``fp8_block_gemm``)
is written for prefill: one program per 16-column tile that walks every K block
inside the program. A decode step has a single activation row, so that shape
leaves a few hundred programs on eight cards and gives each one a long serial
chain of reduction and scale bit operations. The arithmetic is right and the
utilisation is not.

This operator keeps the arithmetic and changes only the schedule, in two passes:

1. every ``(column, K block)`` fragment is computed independently, one program
   per (column tile, K block) pair -- bit for bit the loop body of the accepted
   kernel;
2. the fragments are added back **in increasing K-block order** into one FP32
   accumulator per column -- bit for bit the accepted kernel's ``acc = acc +
   scaled`` chain.

There is no split-K tree reduction, no reassociated sum, no fused multiply-add,
no requantization and no native FP4/FP8 arithmetic. The per-fragment reduction
and both rescales are the frozen helpers of ``_ascend_block_gemm_kernels``, and
the packed-nibble widening is the frozen ``unpack_fp4_to_int8`` codebook, so the
fragment is the accepted fragment by construction. The NPU test asserts bit
equality against the accepted operator, and the scratch buffer is compared
against an independent scalar oracle fragment by fragment.

The three entry points differ only in how the weight is stored:

* :func:`fp4_int8_decode_gemv` -- exact ``INT8 = 2 * E2M1`` values plus the
  original **half** FP32 scale grid (what online unpack produces).
* :func:`fp4_packed_decode_gemv` -- the original packed E2M1 bytes plus the
  original E8M0 byte grid, widened inside the kernel. Nothing is requantized:
  the same codes and the same original scales the checkpoint stores.
* :func:`fp8_decode_gemv` -- E4M3FN weight codes with a square or row-wise
  E8M0 grid, raw bytes or their exact FP32 decoded powers.

Scratch layout: ``(K // 32, N)``, K-block major, returned by
:func:`decode_gemv_scratch`. A pass-1 program owns a run of columns and writes one
scratch row per K block it computes, so K-block major makes pass 2's per-block
read contiguous; a ``(N, K // 32)`` scratch would make it strided. Pass
``scratch=`` from a caller that runs this every step, or leave it ``None`` to
allocate per call. The CPU reference path validates a supplied scratch and does
not write it.

Launch geometry is bounded by the device's **total** program cap (65535), not by
one grid axis: the launcher splits the K blocks so that
``ceil(N / fragment_rows) * ksplit <= 65535`` and each pass-1 program walks its own
strided slice of blocks. A geometry that cannot fit is refused by name rather than
launched with missing work, and the check runs before any allocation so the CPU
reference path rejects it too. For the released V4.1-Flash shapes the largest case
(129,280 x 7,168) uses 64,640 of the 65,535 programs.

This is an explicitly opt-in operator: it is not wired into any model or serving
path here, and it makes no performance claim. Only an actual, measured decode
benchmark on the target build can say whether it is faster than the accepted
block GEMM.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch
from .block_gemm import _bytes, _dense, fp4_int8_block_gemm, fp8_block_gemm
from .unpack import fp4_int8_scale, unpack_fp4_to_int8

__all__ = [
    "decode_gemv_scratch",
    "fp4_int8_decode_gemv",
    "fp4_packed_decode_gemv",
    "fp8_decode_gemv",
]

BLOCK = 32
#: Ascend refuses a launch grid above 65535 on an axis; both kernels stride any
#: remaining work instead of growing the grid past this.
MAX_PROGRAMS = 65535
#: Columns per pass-1 program. 16 x 32 lanes is the tile the accepted kernel
#: already compiles as, so the default adds no new tile shape to the backend; the
#: parameter exists because the best value is a measurement on the target build.
FRAGMENT_ROWS = 16
#: Columns per pass-2 program.
ACCUMULATE_COLUMNS = 64


def _power_of_two(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")
    return value


def _operands(a_codes, a_scale):
    """Validate the single activation row and its block-32 E8M0 grid."""
    a = _bytes(a_codes, "activation codes")
    sa = _bytes(a_scale, "activation scales", scale=True)
    if a.shape[0] != 1:
        raise ValueError("the decode GEMV is defined for exactly one activation row [1,K]")
    k = a.shape[1]
    if k <= 0 or k % BLOCK:
        raise ValueError("the decode GEMV needs a positive K divisible by 32")
    if tuple(sa.shape) != (1, k // BLOCK):
        raise ValueError(f"activation scales must have shape {(1, k // BLOCK)}")
    return a, sa, k


def _device(*tensors):
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all decode GEMV operands must be on the same physical device")
    return device


def _backend(backend, device):
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "auto":
        backend = "triton" if device.type == "npu" else "torch"
    if backend == "torch" and device.type != "cpu":
        raise ValueError("the bounded decode GEMV Torch oracle is CPU-only; use the fused NPU backend")
    if backend == "triton" and device.type != "npu":
        raise ValueError("the fused decode GEMV backend is for NPU tensors only")
    return backend


def decode_gemv_scratch(n, k, *, device=None):
    """Allocate the reusable two-pass scratch, ``([K // 32, N], [K // 32, N])``.

    The first tensor carries the FP32 fragments, the second the per-fragment
    invalid flag. Both are fully written by the fused path, so a caller may keep
    one pair per geometry and reuse it; nothing here is stateful.
    """
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("decode GEMV scratch needs an int N and an int K")
    if n <= 0 or k <= 0 or k % BLOCK:
        raise ValueError("decode GEMV scratch needs positive N and a K divisible by 32")
    blocks = k // BLOCK
    return (
        torch.empty((blocks, n), dtype=torch.float32, device=device),
        torch.empty((blocks, n), dtype=torch.uint8, device=device),
    )


def _prepare_scratch(scratch, n, blocks, device, protected):
    if scratch is None:
        return (
            torch.empty((blocks, n), dtype=torch.float32, device=device),
            torch.empty((blocks, n), dtype=torch.uint8, device=device),
        )
    if not isinstance(scratch, (tuple, list)) or len(scratch) != 2:
        raise TypeError("scratch must be the (fragment, invalid) pair from decode_gemv_scratch")
    fragments, invalid = scratch
    _dense(fragments, "fragment scratch", (torch.float32,))
    _dense(invalid, "invalid scratch", (torch.uint8,))
    for name, tensor in (("fragment scratch", fragments), ("invalid scratch", invalid)):
        if tuple(tensor.shape) != (blocks, n):
            raise ValueError(f"{name} must be [K // 32, N] = {(blocks, n)}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on the input device")
        for other in protected:
            if torch._C._overlaps(tensor, other):
                raise ValueError("the decode GEMV scratch must not overlap an operand or the output")
    if torch._C._overlaps(fragments, invalid):
        raise ValueError("the fragment and invalid scratch must be separate buffers")
    return fragments, invalid


def _output(out, n, device, protected):
    if out is None:
        return torch.empty((1, n), dtype=torch.float32, device=device)
    _dense(out, "output", (torch.float32,))
    if tuple(out.shape) != (1, n):
        raise ValueError(f"output must be [1, N] = {(1, n)}")
    if out.device != device:
        raise ValueError("output must be on the input device")
    for other in protected:
        if torch._C._overlaps(out, other):
            raise ValueError("the decode GEMV output must not overlap an operand or the scratch")
    return out


def _torch_reference(a, sa, b, sb, *, fp4, packed, block_rows, out):
    """Delegate to the accepted CPU operator; no second reduction is implemented here.

    The packed form is expressed as the frozen byte widening plus the frozen
    half-scale decode, then the accepted FP4 operator -- a composition of two
    operators that already have their own oracles, not a new numerical path.
    """
    if packed:
        values = unpack_fp4_to_int8(b, backend="torch")
        return fp4_int8_block_gemm(
            a, sa, values, fp4_int8_scale(sb, backend="torch"), backend="torch", out=out
        )
    if fp4:
        return fp4_int8_block_gemm(a, sa, b, sb, backend="torch", out=out)
    return fp8_block_gemm(a, sa, b, sb, block_rows=block_rows, backend="torch", out=out)


def _grids(n, blocks, fragment_rows, columns_per_program):
    """Grid shapes whose *total* program count stays under the device cap.

    Ascend rejects a launch whose total block count exceeds 65535 -- the earlier
    one-dimensional "grid should be less than 65536" failure was that limit -- so
    the product is what has to fit, not each axis on its own. Pass 1 therefore
    splits the K blocks into ``ksplit`` pieces and walks them with
    ``range(pid, KBLOCKS, ksplit)`` instead of flattening a program id and
    dividing it back apart, because division by a non-power-of-two constant has no
    proven lowering on this backend.
    """
    chunks = math.ceil(n / fragment_rows)
    if chunks > MAX_PROGRAMS:
        raise ValueError(
            f"N / fragment_rows = {chunks} exceeds the {MAX_PROGRAMS} total-program cap; "
            f"pass a larger fragment_rows or split the operand"
        )
    ksplit = max(1, min(blocks, MAX_PROGRAMS // chunks))
    column_grid = math.ceil(n / columns_per_program)
    if column_grid > MAX_PROGRAMS:
        raise ValueError(
            f"N / columns_per_program = {column_grid} exceeds the {MAX_PROGRAMS} "
            f"total-program cap; pass a larger columns_per_program or split the operand"
        )
    assert chunks * ksplit <= MAX_PROGRAMS
    return chunks, ksplit, column_grid


def _launch(a, sa, b, sb, out, fragments, invalid, *,
            fp4, packed, block_rows, b_scale_float, n, k,
            fragment_rows, columns_per_program):
    with _npu_launch(a):
        from ._ascend_decode_block_gemv_kernels import (
            decode_accumulate_kernel,
            decode_fragment_kernel,
        )

        blocks = k // BLOCK
        chunks, ksplit, column_grid = _grids(n, blocks, fragment_rows, columns_per_program)
        # Widen the index arithmetic before any product: ``N * K`` and
        # ``KBLOCKS * N`` are the reaches that can pass 2**31.
        index64 = max(n * k, n * blocks) >= 2**31
        decode_fragment_kernel[(chunks, ksplit)](
            a, sa, b, sb, fragments, invalid,
            n, k, blocks, block_rows, fp4, packed, b_scale_float,
            fragment_rows, ksplit, index64,
        )
        decode_accumulate_kernel[(column_grid,)](
            fragments, invalid, out, n, blocks, columns_per_program, index64
        )
    return out


def _run(a, sa, b, sb, *, fp4, packed, block_rows, b_scale_float, n, k, device,
         out, scratch, backend, fragment_rows, columns_per_program):
    fragment_rows = _power_of_two(fragment_rows, "fragment_rows")
    columns_per_program = _power_of_two(columns_per_program, "columns_per_program")
    blocks = k // BLOCK
    protected = (a, sa, b, sb)
    # The grid/shape bound is checked before any allocation or device work, so a
    # geometry this decoder cannot launch fails on the CPU reference path too.
    _grids(n, blocks, fragment_rows, columns_per_program)
    if backend == "torch":
        if scratch is not None:
            # Validated even though this path does not write it, so a caller that
            # switches backend for testing gets the same shape errors either way.
            _prepare_scratch(scratch, n, blocks, device, protected)
        out = _output(out, n, device, protected)
        return _torch_reference(a, sa, b, sb, fp4=fp4, packed=packed, block_rows=block_rows, out=out)
    fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    out = _output(out, n, device, protected + (fragments, invalid))
    return _launch(
        a, sa, b, sb, out, fragments, invalid,
        fp4=fp4, packed=packed, block_rows=block_rows, b_scale_float=b_scale_float,
        n=n, k=k, fragment_rows=fragment_rows, columns_per_program=columns_per_program,
    )


def fp4_int8_decode_gemv(
    a_codes, a_scale, b_int8_x2, b_half_scale, *,
    backend="auto", out=None, scratch=None,
    fragment_rows=FRAGMENT_ROWS, columns_per_program=ACCUMULATE_COLUMNS,
):
    """Exact ``INT8 = 2 * E2M1`` weights with the original half FP32 scale grid.

    ``b_int8_x2`` is ``[N, K]`` ``int8`` and holds exactly twice each E2M1 value;
    ``b_half_scale`` is ``[N, K // 32]`` FP32 and holds the original E8M0 grid
    divided by two. Keeping the pair together is what stops a caller from mixing
    doubled values with the original (undivided) grid, which would halve every
    K block's contribution.
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
        backend=backend, fragment_rows=fragment_rows,
        columns_per_program=columns_per_program,
    )


def fp4_packed_decode_gemv(
    a_codes, a_scale, b_packed, b_e8m0, *,
    backend="auto", out=None, scratch=None,
    fragment_rows=FRAGMENT_ROWS, columns_per_program=ACCUMULATE_COLUMNS,
):
    """Original packed E2M1 weight bytes with the original E8M0 byte grid.

    ``b_packed`` is ``[N, K // 2]`` ``uint8`` (or an ``int8`` container), one byte
    per adjacent K pair, low nibble first -- the checkpoint's own layout, the one
    ``unpack_fp4_to_int8`` reads. ``b_e8m0`` is the original ``[N, K // 32]`` raw
    E8M0 grid, **not** halved: the kernel widens the nibbles to ``2 * E2M1`` and
    pairs them with the half-scale it derives from that grid, so the loaded tile
    is the original value under the original scale. Nothing is requantized.
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
        fragment_rows=fragment_rows, columns_per_program=columns_per_program,
    )


def fp8_decode_gemv(
    a_codes, a_scale, b_codes, b_scale, *,
    block_rows=32, backend="auto", out=None, scratch=None,
    fragment_rows=FRAGMENT_ROWS, columns_per_program=ACCUMULATE_COLUMNS,
):
    """E4M3FN weights with a square or row-wise original E8M0 grid.

    ``b_scale`` may be raw E8M0 bytes, ``[ceil(N / block_rows), K // 32]``, or the
    exact FP32 decoded powers of that grid in the same shape. ``block_rows`` is
    32 for the square dense grid and 1 for the row-scaled Engram-style grid; it is
    part of the contract, not inferred, because inferring it would make a wrong
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
        scratch=scratch, backend=backend, fragment_rows=fragment_rows,
        columns_per_program=columns_per_program,
    )