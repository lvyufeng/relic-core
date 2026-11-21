"""Exact M=1 decode GEMV over packed FP4 weights in their resident tile-major bank.

Same accepted arithmetic as :mod:`decode_grouped_gemv`, reading the *same* original
weight bytes from a permuted resident layout instead of the checkpoint's row-major
``[N, K // 2]`` matrix. See
:mod:`_ascend_tile_major_packed_kernels` for the layout and why it is tile-outer.

Resident layout (one bank, same byte count for an ``N`` that is a multiple of 16)::

    b_tile : uint8 [NTILES, K // 32, 16, 16]
    s_tile : uint8 [NTILES, K // 32, 16]

``b_tile[tile, block, r, c]`` is the packed byte of logical row ``tile * 16 + r``
at K block ``block``, byte ``c`` inside that block -- so the flat run
``(tile * KBLOCKS + block) * 256 + [0, 256)`` is one row-tile of one K block, and
``GROUP_K`` consecutive blocks of one tile are ``GROUP_K * 256`` contiguous bytes.
``s_tile[tile, block, r]`` is that row/block's **original, unhalved E8M0 byte**.
Nothing is requantized, and no ``INT8 x 2`` or FP32 half-scale bank is built.

Logical geometry is derived from the tile shapes: ``K = shape[1] * 32`` and
``N = shape[0] * 16``. A padded bank (``N`` not a multiple of 16) passes
``n_output`` to trim to the real row count; the padding rows are zero code bytes
and scale byte ``127`` (identity) and are masked out, never sampled.

Validated scope
---------------

Only the **packed FP4** entry point exists. There is no dense form: the wide-tile
dense load of the prefetch prototype mismatched the accepted operator on 71 finite
FP32 values in its FP8 form and is unbound, and this layout has no dense variant to
validate. Nothing here is a performance claim -- it is a memory-access variant of
unchanged arithmetic, and only an actual decode on the target build can say whether
the burst shape was the cost.

The CPU backend is the frozen prototype's accepted operator reached through the
exact inverse permutation, so it proves operand normalisation, the permutation and
the dispatch -- and *nothing* about the Ascend kernel. Only the hardware test in
``tests/test_tile_major_packed_gemv.py`` does that.
"""
from __future__ import annotations


import torch

from .ascend_unpack import _npu_launch
from .block_gemm import _bytes, fp4_int8_block_gemm
from .decode_block_gemv import (
    BLOCK,
    MAX_PROGRAMS,
    _backend,
    _device,
    _operands,
    _output,
    _prepare_scratch,
    decode_gemv_scratch,
)
from .unpack import fp4_int8_scale, unpack_fp4_to_int8

__all__ = [
    "ACCUMULATE_COLUMNS",
    "GROUP_K",
    "NTILE_ROWS",
    "fp4_packed_tile_major_decode_gemv",
    "pack_tile_major",
    "tile_major_packed_reference",
    "tile_major_packed_scratch",
    "unpack_tile_major",
]

#: Logical rows a single resident tile holds. Fixed by the layout: ``[..., 16, 16]``
#: is 16 rows x 16 packed bytes = 32 K values per row.
NTILE_ROWS = 16
#: K blocks per pass-1 program. Four blocks x 16 rows is the frozen ``[64, 32]``
#: fragment tile the grouped kernel already compiles on this backend, and it makes
#: one group a single contiguous ``4 * 256``-byte load and one 64-byte scale load.
GROUP_K = 4
#: Columns per pass-2 program; the frozen accumulate kernel's own parameter.
ACCUMULATE_COLUMNS = 64
#: The original E8M0 byte of ``2**0``: the identity scale used for padded rows.
IDENTITY_SCALE = 127
#: Packed bytes per logical row per K block: 32 codes, two per byte.
BLOCK_BYTES = BLOCK // 2


def _tiles(n: int) -> int:
    return -(-n // NTILE_ROWS)


def _padded_n(n: int) -> int:
    return _tiles(n) * NTILE_ROWS


def _contiguous(tensor, name, ndim, dtypes=(torch.uint8, torch.int8)):
    """Validate one resident tile tensor and return it as ``uint8`` container bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected {dtypes}")
    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a contiguous strided {ndim}-D tile tensor")
    if tensor.requires_grad:
        raise ValueError("the tile-major packed GEMV is an inference operator, not an autograd operation")
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _tile_operands(b_tile, s_tile):
    """Validate the resident bank and derive ``(b, sb, blocks, ntiles)`` from its shapes."""
    b = _contiguous(b_tile, "tile-major packed FP4 weight", 4)
    s = _contiguous(s_tile, "tile-major E8M0 scales", 3)
    if b.shape[2] != 16 or b.shape[3] != BLOCK_BYTES:
        raise ValueError(
            "tile-major packed FP4 weight must be [NTILES, K // 32, 16, "
            f"{BLOCK_BYTES}], got {tuple(b.shape)}"
        )
    ntiles, blocks = b.shape[0], b.shape[1]
    if ntiles <= 0 or blocks <= 0:
        raise ValueError("the tile-major bank needs a positive tile count and K-block count")
    if tuple(s.shape) != (ntiles, blocks, 16):
        raise ValueError(
            f"tile-major E8M0 scales must be [NTILES, K // 32, 16] = {(ntiles, blocks, 16)}, "
            f"got {tuple(s.shape)}"
        )
    if b.device != s.device:
        raise ValueError("the tile-major weight and scale banks must be on the same device")
    for other in (b, s):
        if other.requires_grad:
            raise ValueError("the tile-major packed GEMV is an inference operator")
    return b, s, blocks, ntiles


def _logical_n(n_output, ntiles):
    """Resolve the logical row count: the physical tile extent, or a caller's trim."""
    physical = ntiles * NTILE_ROWS
    if n_output is None:
        return physical
    if isinstance(n_output, bool) or not isinstance(n_output, int):
        raise TypeError(f"n_output must be an int, got {type(n_output).__name__}")
    if n_output <= 0 or n_output > physical:
        raise ValueError(
            f"n_output must be in (0, {physical}] for a {ntiles}-tile bank, got {n_output}"
        )
    return n_output


def _geometry(n, blocks, group_k, accumulate_columns):
    """Grid shapes whose *total* program count stays under the device cap.

    Ascend rejects a launch whose total block count exceeds 65535, so pass 1 gets
    ``(NTILES, group stride)`` with the product inside the cap and each program
    strides the remaining K groups. The check runs before allocation or device
    work, so the CPU path rejects an unlaunchable geometry too.
    """
    if isinstance(group_k, bool) or not isinstance(group_k, int):
        raise TypeError("group_k must be an int")
    if group_k not in {1, 2, 4}:
        raise ValueError(
            "group_k must be 1, 2 or 4 (the 4 x 16 = 64-row fragment tile is the "
            f"largest that compiles on this backend), got {group_k}"
        )
    if isinstance(accumulate_columns, bool) or not isinstance(accumulate_columns, int):
        raise TypeError("accumulate_columns must be an int")
    if accumulate_columns <= 0 or accumulate_columns & (accumulate_columns - 1):
        raise ValueError("accumulate_columns must be a positive power of two")
    ntiles = _tiles(n)
    if ntiles > MAX_PROGRAMS:
        raise ValueError(
            f"N / {NTILE_ROWS} = {ntiles} exceeds the {MAX_PROGRAMS} total-program cap"
        )
    group_chunks = -(-blocks // group_k)
    group_stride = max(1, min(group_chunks, MAX_PROGRAMS // ntiles))
    assert ntiles * group_stride <= MAX_PROGRAMS
    column_grid = -(-n // accumulate_columns)
    if column_grid > MAX_PROGRAMS:
        raise ValueError(
            f"N / accumulate_columns = {column_grid} exceeds the {MAX_PROGRAMS} "
            f"total-program cap; pass a larger accumulate_columns"
        )
    return ntiles, group_stride, column_grid


def _needs_index64(n, blocks, ntiles):
    """Whether the offsets need 64-bit indexing, from the **padded** reach.

    The widest weight offset is the resident bank itself (``NTILES * KBLOCKS * 256``,
    i.e. the padded ``N * K / 2``) and the widest scratch offset is ``KBLOCKS * N``.
    A masked tail lane still forms its address before the mask suppresses the
    access, so the padded extent is what can overflow.
    """
    return max(ntiles * blocks * 256, blocks * n) >= 2**31


def pack_tile_major(b_packed, b_e8m0):
    """Permute the checkpoint's row-major packed weight into the resident bank.

    ``b_packed`` is ``[N, K // 2]`` (one byte per adjacent K pair, low nibble first)
    and ``b_e8m0`` is ``[N, K // 32]`` raw E8M0 bytes. Returns
    ``(b_tile[NTILES, K // 32, 16, 16], s_tile[NTILES, K // 32, 16])``.

    This is **only a permutation**: every code nibble and every scale byte keeps its
    exact value, both outputs are contiguous, and ``unpack_tile_major`` recovers the
    inputs byte for byte. No checkpoint file is touched and no new quantization grid
    is introduced.

    ``N`` is padded up to a multiple of 16 -- the tile rows -- with zero code bytes
    (code 0) and scale byte ``127`` (``2**0``, the identity), which is why the
    returned byte count equals the input's for an ``N`` that is already a multiple
    of 16 and only grows by the padding otherwise.
    """
    b = _bytes(b_packed, "packed FP4 weight bytes")
    sb = _bytes(b_e8m0, "FP4 E8M0 weight scales", scale=True)
    if b.ndim != 2 or sb.ndim != 2 or b.shape[0] != sb.shape[0]:
        raise ValueError("packed weight and scale bytes must be [N, K // 2] and [N, K // 32]")
    n, bytes_per_row = b.shape
    if n <= 0 or bytes_per_row <= 0:
        raise ValueError("packed FP4 weight bytes need a positive N and K")
    if bytes_per_row % BLOCK_BYTES:
        raise ValueError(
            f"a row holds {bytes_per_row} packed bytes, which is not a whole number of "
            f"{BLOCK_BYTES}-byte K blocks"
        )
    blocks = bytes_per_row // BLOCK_BYTES
    if tuple(sb.shape) != (n, blocks):
        raise ValueError(f"FP4 E8M0 scales must be [N, {blocks}] = {(n, blocks)}, got {tuple(sb.shape)}")
    device = b.device
    if sb.device != device:
        raise ValueError("the packed weight and its scales must be on the same device")

    ntiles = _tiles(n)
    padded = ntiles * NTILE_ROWS
    if padded != n:
        b_pad = torch.zeros((padded, bytes_per_row), dtype=torch.uint8, device=device)
        s_pad = torch.full((padded, blocks), IDENTITY_SCALE, dtype=torch.uint8, device=device)
        b_pad[:n] = b
        s_pad[:n] = sb
        b, sb = b_pad, s_pad
    # [tile, row, block, byte] -> [tile, block, row, byte]: the row axis is the
    # innermost logical axis of a tile, and moving it inside the block axis is what
    # makes one K block of one tile a single contiguous 256-byte run.
    b_tile = (
        b.reshape(ntiles, NTILE_ROWS, blocks, BLOCK_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    s_tile = (
        sb.reshape(ntiles, NTILE_ROWS, blocks).permute(0, 2, 1).contiguous()
    )
    return b_tile, s_tile


def unpack_tile_major(b_tile, s_tile, *, n_output=None):
    """Recover the checkpoint's row-major packed bytes; exact inverse of the pack.

    Returns ``(b_packed[N, K // 2], b_e8m0[N, K // 32])`` as ``uint8``. Used by the
    CPU reference and by any consumer that needs the logical matrix back (a
    multi-row prefill that reuses the accepted dense path, for instance); the
    round trip is byte-identical, not merely numerically equal.
    """
    b, s, blocks, ntiles = _tile_operands(b_tile, s_tile)
    n = _logical_n(n_output, ntiles)
    padded = b.permute(0, 2, 1, 3).reshape(ntiles * NTILE_ROWS, blocks * BLOCK_BYTES)
    scales = s.permute(0, 2, 1).reshape(ntiles * NTILE_ROWS, blocks)
    return padded[:n].contiguous(), scales[:n].contiguous()


def tile_major_packed_scratch(n, k, *, device=None):
    """The reusable two-pass scratch, ``([K // 32, N], [K // 32, N])``.

    The frozen fragment layout, because pass 2 is the frozen accumulate kernel.
    """
    return decode_gemv_scratch(n, k, device=device)


def tile_major_packed_reference(a_codes, a_scale, b_tile, s_tile, *, n_output=None, out=None):
    """Pure-Torch reference: invert the permutation, then run the accepted operator.

    No second reduction is written here. The tile bank is permuted back to the
    checkpoint's own bytes, then the frozen FP4 byte widening and half-scale decode
    feed ``fp4_int8_block_gemm`` -- a composition of operators that already have
    their own oracles. Intended for CPU fixtures; it says nothing about the kernel.
    """
    a, sa, _ = _operands(a_codes, a_scale)
    b, s, blocks, ntiles = _tile_operands(b_tile, s_tile)
    n = _logical_n(n_output, ntiles)
    device = _device(a, sa, b, s)
    if device.type != "cpu":
        raise ValueError("the tile-major Torch reference is CPU-only; use the fused NPU backend")
    b_packed, b_e8m0 = unpack_tile_major(b_tile, s_tile, n_output=n_output)
    out = _output(out, n, device, (a, sa, b, s))
    values = unpack_fp4_to_int8(b_packed, backend="torch")
    return fp4_int8_block_gemm(
        a, sa, values, fp4_int8_scale(b_e8m0, backend="torch"), backend="torch", out=out
    )


def _launch(a, sa, b, s, out, fragments, invalid, *, n, blocks, ntiles, group_k,
            group_stride, column_grid, accumulate_columns):
    with _npu_launch(a):
        from ._ascend_decode_block_gemv_kernels import decode_accumulate_kernel
        from ._ascend_tile_major_packed_kernels import tile_major_packed_fragment_kernel

        index64 = _needs_index64(n, blocks, ntiles)
        tile_major_packed_fragment_kernel[(ntiles, group_stride)](
            a, sa, b, s, fragments, invalid,
            n, blocks, index64, group_k, group_stride,
        )
        # Pass 2 is the frozen kernel: it adds the fragments in strictly increasing
        # K order, so no reassociation is introduced by the new layout.
        decode_accumulate_kernel[(column_grid,)](
            fragments, invalid, out, n, blocks, accumulate_columns, index64
        )
    return out


def fp4_packed_tile_major_decode_gemv(
    a_codes, a_scale, b_tile, s_tile,
    backend="auto", *, n_output=None, group_k=GROUP_K,
    accumulate_columns=ACCUMULATE_COLUMNS, out=None, scratch=None,
):
    """Exact original packed FP4 weight bytes, resident in the tile-major bank.

    ``a_codes``/``a_scale`` are the unchanged original E4M3 activation codes
    ``[1, K]`` and raw E8M0 bytes ``[1, K // 32]``. ``b_tile``/``s_tile`` are the
    resident bank from :func:`pack_tile_major` -- ``K = b_tile.shape[1] * 32`` and
    ``N = b_tile.shape[0] * 16``, with ``n_output`` trimming a padded bank.

    ``backend`` is positional-or-keyword and defaults to ``auto`` (NPU fused, CPU
    reference), so a caller can pass it as the fifth positional argument or as a
    keyword. Returns the FP32 ``[1, N]`` accumulator; the caller owns the final
    narrowing and the surrounding activation ops.
    """
    a, sa, k = _operands(a_codes, a_scale)
    b, s, blocks, ntiles = _tile_operands(b_tile, s_tile)
    if blocks * BLOCK != k:
        raise ValueError(
            f"the tile-major bank holds K = {blocks * BLOCK} (K blocks x {BLOCK}) and the "
            f"activation row holds K = {k}"
        )
    n = _logical_n(n_output, ntiles)
    device = _device(a, sa, b, s)
    backend = _backend(backend, device)
    protected = (a, sa, b, s)
    # The grid bound is checked before any allocation or device work.
    _, group_stride, column_grid = _geometry(n, blocks, group_k, accumulate_columns)
    if backend == "torch":
        if scratch is not None:
            # Validated even though this path writes no scratch, so a caller that
            # switches backend for testing sees the same shape errors either way.
            _prepare_scratch(scratch, n, blocks, device, protected)
        out = _output(out, n, device, protected)
        return tile_major_packed_reference(a, sa, b, s, n_output=n, out=out)
    fragments, invalid = _prepare_scratch(scratch, n, blocks, device, protected)
    out = _output(out, n, device, protected + (fragments, invalid))
    return _launch(
        a, sa, b, s, out, fragments, invalid, n=n, blocks=blocks, ntiles=ntiles,
        group_k=group_k, group_stride=group_stride, column_grid=column_grid,
        accumulate_columns=accumulate_columns,
    )