"""Exact M=1 decode block GEMV whose per-lane scaling is an FP32 significand product.

This is the same reduction as :mod:`relic_core.kernels.block_gemm` -- the explicit
two-half block-32 rule with a nominal-exponent anchor, a 13-bit truncation window,
a carry from the first half into the second, a final window cut, two sequential
E8M0 rescales and an ordered K-block accumulation -- and it computes the same
integers, bit for bit, as :mod:`relic_core.kernels.integer_decode_gemv`.

The difference is how a term is built. The frozen integer prototype shifts a small
integer by a per-lane runtime amount and multiplies significands in ``int32``; on
the target build that measured 62.90 ms grouped and 85.40 ms serial against 17.90 ms
for the accepted FP32 fragment schedule, so this operator replaces exactly the two
suspected non-vectorized operations and keeps everything else identical:

* the term is ``significand_product.to(fp32) * _pow2(power - grid)`` cast to
  ``int32`` -- an exact FP32 product by an exact normal dyadic factor, not a
  variable shift;
* the carry and window exponents come from a bitcast of the exact integer's FP32
  image (``>> 23``, constant) instead of the binary-search ``ilog2``.

It is exact rather than approximate, and the module's tests prove each step against
an independent ``fractions.Fraction`` statement:

* significand products are ``<= 225`` (eight significant bits) and every
  power-of-two factor is normal, so the FP32 product is the exact rational value;
* ``|term| <= 2**19`` and a 16-term partial sum is ``<= 2**20``, exact as ``int32``
  and as the FP32 the cast produces, so the carry exponent is the true floor;
* the cast is taken on the magnitude with the sign applied afterwards, which is a
  truncation toward zero -- the accepted cast's direction -- without relying on this
  backend's lowering of a negative float-to-int conversion.

Two schedules are provided and they compute the same fragments:

* ``variant="grouped"`` -- pass 1 computes each ``(column, K block)`` fragment in a
  program that owns a column run and a group of K blocks; pass 2 is the frozen
  ``decode_accumulate_kernel``, which adds the fragments in increasing block order.
* ``variant="serial"`` -- one program owns a column run and walks every K block
  itself, accumulating in order. No scratch, fewer programs, used as the proof
  schedule.

Neither variant reassociates the K sum, fuses a multiply-add, requantizes an
operand, or uses native FP4/FP8, FP64, or matrix-unit arithmetic. The CPU backend
delegates to the frozen :func:`decode_block_gemv.fp4_packed_decode_gemv` /
:func:`decode_block_gemv.fp8_decode_gemv`, which delegate to the accepted block
GEMM, so the oracle for this operator is the accepted operator rather than a second
transcription of the rule.

**No performance claim is made here.** The variable-shift lowering is a hypothesis
about why the integer prototype was slow; only a measurement on the target build
decides it, and nothing in the repository binds this operator into a model or a
serving path.
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
from .integer_decode_gemv import (
    DEFAULT_GROUP_K,
    DEFAULT_ROWS_N,
    GROUP_K_MAX,
    ROWS_N_MAX,
    _index_width,
    _integer_grids,
    _tile,
    _variant,
)

__all__ = [
    "fp4_packed_float_significand_gemv",
    "fp8_float_significand_gemv",
    "floating_significand_gemv_scratch",
]


def floating_significand_gemv_scratch(n, k, *, device=None):
    """The two-pass scratch of the frozen decoder: ``[K // 32, N]`` fragments plus flags.

    A caller running the grouped variant every step can keep one pair per geometry;
    a ``variant="serial"`` call accepts it too so both schedules take the same
    argument.
    """
    return decode_gemv_scratch(n, k, device=device)


def _run(a, sa, b, sb, *, fp4, packed, block_rows, b_scale_float, n, k, device,
         out, scratch, backend, rows_n, group_k, variant):
    rows_n, group_k = _tile(rows_n, group_k)
    if fp4 and not packed:
        # ``INT8 = 2 * E2M1`` storage is served by the frozen entry point; this
        # operator reads the packed released nibbles, so the other combination is
        # refused rather than compiled as dead device code nobody exercises.
        raise ValueError(
            "floating-significand decode GEMV reads packed FP4 nibbles; use the frozen "
            "INT8-times-two entry point for expanded INT8 storage"
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
        return fp8_decode_gemv(
            a, sa, b, sb, block_rows=block_rows, backend="torch", out=target
        )
    fragments, invalid = (None, None)
    if variant == "grouped":
        fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    elif scratch is not None:
        # A supplied scratch is validated even though the serial schedule does not
        # write it, so a caller that switches schedule for testing gets the same
        # shape/overlap errors either way.
        _prepare_scratch(scratch, n, blocks, device, protected)
    target = _output(out, n, device,
                     protected if fragments is None else protected + (fragments, invalid))
    with _npu_launch(a):
        from ._ascend_float_significand_kernels import (
            float_fragment_kernel,
            float_serial_kernel,
        )

        index64 = _index_width(n, k, blocks)
        common = (a, sa, b, sb)
        flags = (k, blocks, block_rows, packed, fp4, b_scale_float, rows_n)
        if variant == "grouped":
            from ._ascend_decode_block_gemv_kernels import decode_accumulate_kernel

            float_fragment_kernel[(chunks, ksplit)](
                *common, fragments, invalid, n, *flags, group_k, ksplit, index64
            )
            # The frozen ordered add chain, literally imported: the K-block
            # accumulation is the accepted one, not a re-derivation.
            decode_accumulate_kernel[(math.ceil(n / ACCUMULATE_COLUMNS),)](
                fragments, invalid, target, n, blocks, ACCUMULATE_COLUMNS, index64
            )
        else:
            assert chunks <= MAX_PROGRAMS
            float_serial_kernel[(chunks,)](*common, target, n, *flags, index64)
    return target


def fp4_packed_float_significand_gemv(
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


def fp8_float_significand_gemv(
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