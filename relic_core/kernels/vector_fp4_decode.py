"""Exact original FP4 byte widening computed from bits, without a global lookup.

The frozen ``unpack_fp4_to_int8`` widens each packed E2M1 byte to the pair
``(2 * low_nibble_value, 2 * high_nibble_value)`` as two signed INT8 values, and
the static acceleration of it reads that pair from a 256-entry table in global
memory -- one gather per input byte. This module produces the same bytes from the
byte's own bits instead:

* ``mode="arithmetic"`` -- one integer expression per nibble, no table at all;
* ``mode="lut"`` -- a 16-entry signed table built once per program in on-chip
  memory and read with a local ``tl.gather``.

Values, ordering, dtype and layout are the frozen ones: the table is
``X2_TABLE``, the low nibble is the even K index, both negative-zero codes fold
to integer zero, and no scale is read. Nothing here is a new quantization, and
``unpack_fp4_to_int8`` keeps its own default behaviour untouched. Whether the
removed lookup is actually faster is a hardware measurement, not a claim of this
module; the target microbenchmark belongs to the caller.
"""
from __future__ import annotations

import math

import torch

from . import unpack
from .ascend_unpack import _npu_launch

__all__ = ["X2_TABLE", "unpack_fp4_vector"]

_REFERENCE_UNPACK = unpack.unpack_fp4_to_int8

#: Twice each original E2M1 magnitude, then the negative encodings in the same
#: order, with both zero codes folded to integer zero. Stated here so the tests can
#: hold the frozen widening to it; the kernels derive the same table from code bits.
X2_TABLE = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

#: Output bytes one packed byte turns into.
_PAIR = 2
#: Tiles each mode launches by default: the arithmetic kernel keeps more live
#: integer temporaries, the LUT kernel keeps a gathered index array. 4096 is
#: measured on both released expert geometries, at 1.34x the throughput of 1024
#: (w1/w3 0.3478 ms against 0.4724, w2 0.3418 against 0.4578, full byte identity
#: at every tile), and it is the largest tile that stays inside the unified-buffer
#: budget: 8192 aborts the Ascend Triton compiler. ``lut`` keeps 2048 because that
#: mode's gathered index array was never re-timed at the larger tile.
_DEFAULT_TILE = {"arithmetic": 4096, "lut": 2048}
_TILES = (512, 1024, 2048, 4096)
_MODES = ("auto", "arithmetic", "lut")


def _pair_bits(byte: int) -> int:
    """The exact 16-bit INT8-times-two pair of one packed byte.

    Low nibble in the low byte, which is the released little-endian I8x2 layout.
    """
    low, high = X2_TABLE[byte & 15], X2_TABLE[byte >> 4]
    return (low & 255) | ((high & 255) << 8)


def _lut_rows(tile: int) -> int:
    """Rows of the gathered tile: enough to keep the inner width at 16.

    Mirrors the kernel module's ``MAX_LUT_ROWS``/``MIN_LUT_WIDTH`` without importing
    it: the wrapper validates the tile, and the kernel derives nothing.
    """
    return min(64, tile // 16)


def _index64(numel: int, tile: int) -> bool:
    """Whether the padded output element index can leave signed 32-bit range.

    The kernels address the byte axis and store one 16-bit pair per byte, so the
    widest *element* index the pointer arithmetic can form is twice the padded byte
    count -- the conservative quantity, decided before any launch.
    """
    return -(-numel // tile) * tile * _PAIR >= 2**31


def unpack_fp4_vector(packed: torch.Tensor, *, backend: str = "auto",
                     out: torch.Tensor | None = None, mode: str = "auto",
                     tile: int | None = None) -> torch.Tensor:
    """Widen original packed E2M1 bytes exactly, without a per-element global lookup.

    The result is bit-identical to :func:`relic_core.kernels.unpack.unpack_fp4_to_int8`
    for every one of the 256 input patterns: same values, same K order, same
    ``int8`` dtype, last dimension doubled. Keep the original E8M0 block scales and
    multiply by ``fp4_int8_scale`` exactly as with the frozen widening.

    ``mode="auto"`` selects ``"arithmetic"``, the variant with no gather at all;
    ``mode="lut"`` is the alternative probe that trades the arithmetic for a local
    16-entry table read. ``tile`` overrides the per-mode program size from
    ``{512, 1024, 2048, 4096}``; 4096 is the measured default for ``arithmetic`` and
    8192 aborts the Ascend Triton compiler, so 4096 is the ceiling rather than an
    unmeasured option. ``backend="torch"`` (or a non-NPU device) runs the frozen
    reference widening, so validation and refusals are shared with it.
    """
    raw = unpack._byte_input(packed, "packed", encoding="fp4", min_rank=1)
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    selected = "arithmetic" if mode == "auto" else mode
    if tile is None:
        tile = _DEFAULT_TILE[selected]
    elif isinstance(tile, bool) or not isinstance(tile, int) or tile not in _TILES:
        raise ValueError(f"tile must be one of {_TILES}, got {tile!r}")
    shape = (*raw.shape[:-1], raw.shape[-1] * _PAIR)
    result = unpack._output(out, shape, torch.int8, raw.device, (packed,))
    if raw.numel() == 0:
        return result
    implementation = unpack._select_backend(backend, raw.device)
    if implementation is None:
        return _REFERENCE_UNPACK(packed, backend="torch", out=result)
    from . import _ascend_vector_fp4_decode_kernels as kernels

    size = raw.numel()
    grid = (min(math.ceil(size / tile), 65535),)
    widened = result.view(torch.uint16)
    width64 = _index64(size, tile)
    with _npu_launch(raw):
        if selected == "arithmetic":
            kernels.fp4_vector_arithmetic_kernel[grid](raw, widened, size, tile, width64)
        else:
            kernels.fp4_vector_lut_kernel[grid](raw, widened, size, tile,
                                                _lut_rows(tile), width64)
    return result