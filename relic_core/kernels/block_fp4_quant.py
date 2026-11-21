"""Original block-wise FP4 activation quantization for the released V4.1 kernels.

This reproduces both branches of the released ``fp4_quant_kernel``
(``inference/kernel.py:127-205`` of the checkpoint), which differ in how a block's
scale is chosen and stored:

``scale_format="e8m0"`` -- the indexer path
    ``amax`` (FP32) is floored at ``6 * 2**-126`` and the scale is
    ``2**ceil(log2(amax * fp32(1/6)))``, stored as one raw E8M0 byte per block.
    The helper the source calls is named ``fast_round_scale`` but computes a
    **ceiling** from the FP32 exponent field; it is not a nearest rounding.
    ``amax`` is compared to the floor *before* the reciprocal multiply, exactly as
    the source orders it, and that ordering is what keeps the smallest reachable
    block from producing a zero scale.

``scale_format="e4m3"`` -- the compressed-KV path
    ``amax`` is floored at ``6 * 2**-9`` and the scale is
    ``fp32(e4m3(amax / 6.0))``, stored as one raw E4M3 byte per block. The scale is
    the **rounded ratio**, so normalizing divides by the rounded scale and not by
    the exact ``amax / 6``. The floor is what keeps an all-zero block's scale
    nonzero, which the compressed-KV cache depends on: a zero scale cannot be
    divided out on the way back. The ratio is clamped to 448 before the encode, so
    an ``amax`` above 2688 saturates the scale: that models a saturating FP8 cast,
    which is what the released kernel is expected to lower to, and is an assumption
    about that cast rather than a property stated in the source.

Both branches then clamp ``x / scale`` to ``[-6, 6]`` and encode E2M1 with
round-to-nearest, ties to the even code. Codes are packed two per byte along the
last axis, low nibble first, which is the layout the released packed-FP4 weights
use and the layout ``relic_core.kernels.unpack.unpack_fp4_to_int8`` reads back.

What this is *not*: it is not an INT8 activation path, not W4A8, not a row-wise or
per-tensor substitution, and it never rewrites the checkpoint. It also does not
provide a native FP4 matmul -- it produces the original encoding, and a consumer
still has to exist.

Outputs are raw bytes, never a torch float8 dtype and never a native FP4 dtype:
``codes``/``packed`` are ``uint8`` and ``scales`` are ``uint8``. That keeps the NPU
path clear of the float8 casts this stack cannot perform, and keeps the scale
exactly representable in the consumer. ``torch`` runs a same-device reference and
is the independent CPU oracle; ``triton`` runs the fused Ascend kernel whose
encodes are integer bit arithmetic. ``auto`` selects by device, never falls back
after a backend has been chosen, and never silently substitutes one for the other
after a kernel or compiler failure.

Nibble packing is a separate device-side step rather than a kernel store. The two
nibbles of one byte come from two neighbouring lanes, so a fused pack would need
either a cross-lane reduction or a masked stride-2 store; a masked stride-2 store
tripped an Ascend Triton interleave-optimization assertion in this stack (see the
online-unpack work), so the quantize kernel writes one contiguous code byte per
element and the merge is an explicit ``int16`` bitwise step. It is part of the
implementation rather than a fallback, costs one activation-sized pass, and does
not touch the values.

Non-finite input is outside the released finite contract. This operator extends
that contract explicitly: a block containing any infinity or NaN gets all-zero
code bytes and the *NaN* scale byte of its format (``255`` for E8M0, ``0x7F`` for
E4M3), and ``inplace=True`` writes NaNs for it. E2M1 itself has no NaN code, so a
consumer multiplying the packed codes by that scale still sees NaN instead of a
plausible finite value. No claim is made about how the released kernel treats such
a block.
"""
from __future__ import annotations

import importlib
import struct

import torch

__all__ = ["quantize_block_fp4", "supports_triton_ascend"]

#: Scales are per block of this many consecutive elements. The release uses 16 for
#: the compressed-KV cache and 32 for the indexer; other widths are not validated
#: against the reference and are refused rather than guessed.
_BLOCK_SIZES = (16, 32)
_BLOCKS = 16  # blocks per kernel program: 16 * 32 = 512 active lanes at most
_CHUNK_ELEMENTS = 1 << 20

_FP4_MAX = 6.0
#: The two amax floors, as FP32 literals. Both are exactly representable.
_FLOOR_E8M0 = 6.0 * 2.0**-126
_FLOOR_E4M3 = 6.0 * 2.0**-9
#: ``1/6`` as the FP32 literal the released kernel multiplies by.
_INV_FP4_MAX = struct.unpack("<f", struct.pack("<f", 1.0 / 6.0))[0]

#: The eight E2M1 magnitudes in code order; codes 8..15 are their negatives.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_SIGNED = (*_E2M1_MAGNITUDES, *(-value for value in _E2M1_MAGNITUDES))

_NAN_SCALE = {"e8m0": 255, "e4m3": 0x7F}

_LEVELS: dict[torch.device, torch.Tensor] = {}
_SIGNED_LEVELS: dict[torch.device, torch.Tensor] = {}


def _float32(value: float) -> float:
    """The FP32 rounding of a Python float, returned as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _torch_fp8_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("this torch build has no float8_e4m3fn dtype; the Torch "
                           "reference needs it and the Triton backend is NPU-only")
    return dtype


def supports_triton_ascend() -> bool:
    """Report an installed Ascend Triton backend without starting device work."""
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError:
        return False
    return bool(module.supports_triton_ascend())


def _checked_input(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must have a dense strided layout")
    if x.device.type not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"x uses unsupported device {x.device}")
    if x.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"x must be BF16 or FP32 activations, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError("x must have rank >= 1")
    width = x.shape[-1]
    if width <= 0:
        raise ValueError("the last dimension must be positive")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an int")
    if block_size not in _BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {_BLOCK_SIZES}, got {block_size}")
    if width % block_size:
        raise ValueError(f"the last dimension must be a multiple of {block_size}, got {width}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous; no implicit copy is made")
    return x


def _overlaps(first: torch.Tensor, second: torch.Tensor) -> bool:
    structural = getattr(torch._C, "_overlaps", None)
    if structural is not None:
        return bool(structural(first, second))
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _outputs(out, shapes, device: torch.device, source: torch.Tensor):
    if out is None:
        return tuple(torch.empty(shape, dtype=torch.uint8, device=device) for shape in shapes)
    if (not isinstance(out, (tuple, list)) or len(out) != 2
            or not all(isinstance(item, torch.Tensor) for item in out)):
        raise TypeError("out must be a (packed, scales) pair of uint8 tensors or None")
    packed, scales = out
    if packed is scales:
        raise ValueError("out packed and scales must be distinct tensors")
    for name, tensor, shape in zip(("packed", "scales"), (packed, scales), shapes):
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"out {name} must be a contiguous dense strided tensor")
        if tensor.dtype != torch.uint8:
            raise TypeError(f"out {name} must have dtype torch.uint8, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"out {name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"out {name} must be on the input device ({device})")
    for first, second in ((packed, scales), (packed, source), (scales, source)):
        if _overlaps(first, second):
            raise ValueError("out tensors must not alias or share storage with an input")
    return packed, scales


def _select_backend(backend: str, device: torch.device):
    if not isinstance(backend, str) or backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of 'auto', 'torch', or 'triton'")
    if backend == "torch":
        if device.type == "npu":
            raise ValueError("the Torch FP4 reference is CPU/CUDA-only; NPU execution "
                             "requires the fused backend")
        return None
    if device.type != "npu":
        if backend == "triton":
            raise ValueError("the Triton Ascend backend requires an NPU input")
        return None
    # The NPU stack this package targets cannot cast to float8, so `auto` cannot
    # fall back to the Torch reference here: it needs the fused kernel or an
    # explicit failure. Silence would look like a working cast.
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError as error:
        raise RuntimeError("NPU block FP4 quantization requires the fused Triton Ascend "
                           "backend, which is not importable") from error
    if not module.supports_triton_ascend():
        raise RuntimeError("NPU block FP4 quantization requires an installed Triton Ascend "
                           "backend; there is no NPU float8 cast fallback")
    return module


def _levels(device: torch.device, signed: bool) -> torch.Tensor:
    cache = _SIGNED_LEVELS if signed else _LEVELS
    levels = cache.get(device)
    if levels is None:
        levels = torch.tensor(_E2M1_SIGNED if signed else _E2M1_MAGNITUDES,
                              dtype=torch.float32, device=device)
        cache[device] = levels
    return levels


def _ceil_log2(value: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(value))`` for positive finite FP32 values, from their bits."""
    bits = value.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    return exponent - 127 + (mantissa != 0).to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """FP32 values of ``2**exponent`` for integer exponents, from their bits."""
    return ((exponent + 127) << 23).view(torch.float32)


def _e2m1_codes(normalized: torch.Tensor) -> torch.Tensor:
    """E2M1 codes of values already clamped to ``[-6, 6]``, ties to the even code.

    Written as a nearest-level search rather than as the kernel's bit ladder, so
    the oracle and the implementation are not the same expression. Both
    differences and comparisons are exact: every neighbouring magnitude pair here
    is within a factor of two of itself, so Sterbenz's lemma makes each subtraction
    exact in FP32 and the tie test can never be decided by a rounding.
    """
    magnitude = normalized.abs()
    levels = _levels(normalized.device, signed=False)
    upper = torch.searchsorted(levels, magnitude, right=False).clamp(1, len(_E2M1_MAGNITUDES) - 1)
    lower = upper - 1
    upper_value = levels[upper]
    lower_value = levels[lower]
    above = (magnitude - lower_value) > (upper_value - magnitude)
    tie = (magnitude - lower_value) == (upper_value - magnitude)
    even = (upper & 1) == 0
    index = torch.where(above | (tie & even), upper, lower)
    negative = torch.signbit(normalized)
    return (index + torch.where(negative, 8, 0)).to(torch.uint8)


def _torch_blocks(block: torch.Tensor, scale_format: str):
    """``(codes, scale bytes, scale values)`` for a ``[blocks, width]`` FP32 tensor.

    A non-finite block keeps finite arithmetic by moving its amax to the floor and
    is then replaced: codes 0 and the NaN scale byte of the format.
    """
    floor = _FLOOR_E8M0 if scale_format == "e8m0" else _FLOOR_E4M3
    nan_scale = torch.tensor(_NAN_SCALE[scale_format], dtype=torch.uint8, device=block.device)
    nonfinite = (~torch.isfinite(block)).any(dim=1)
    floored = block.abs().amax(dim=1).clamp_min(floor)
    amax = torch.where(nonfinite, torch.full_like(floored, floor), floored)
    if scale_format == "e4m3":
        # The released kernel's float32-to-E4M3 cast is the saturating hardware
        # conversion (CUDA ``__NV_SATFINITE`` above 448), but torch's CPU cast
        # turns a large magnitude into the E4M3 *NaN* byte instead of saturating.
        # Clamping the ratio first is what the hardware cast does, so the
        # reference keeps the released semantics for the out-of-range ratios an
        # amax above 2688 produces; in-range ratios are unaffected.
        ratio = (amax / _FP4_MAX).clamp(max=448.0)
        scale_byte = ratio.to(_torch_fp8_dtype()).view(torch.uint8)
        scale_value = scale_byte.view(_torch_fp8_dtype()).to(torch.float32)
    else:
        exponent = _ceil_log2(amax * _INV_FP4_MAX)
        scale_value = _pow2(exponent)
        scale_byte = ((exponent + 127) & 0xFF).to(torch.uint8)
    normalized = (block / scale_value[:, None]).clamp(-_FP4_MAX, _FP4_MAX)
    codes = _e2m1_codes(normalized)
    codes = torch.where(nonfinite[:, None], torch.zeros_like(codes), codes)
    scale_byte = torch.where(nonfinite, nan_scale, scale_byte)
    scale_value = torch.where(nonfinite, torch.full_like(scale_value, float("nan")), scale_value)
    return codes, scale_byte, scale_value


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Merge adjacent code pairs into one byte, low nibble first.

    Written as ``low + high * 16`` rather than as a shift and an OR: the two
    nibbles cannot overlap or carry into each other (``low <= 15``), so the sum is
    the byte, and it uses only multiply/add on a small integer type, which every
    backend supports. ``int16`` keeps the intermediate from being mistaken for a
    bit pattern before the ``uint8`` narrowing.
    """
    low = codes[..., 0::2].to(torch.int16)
    high = codes[..., 1::2].to(torch.int16)
    return (low + high * 16).to(torch.uint8)


def _torch_reference(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                     block_size: int, scale_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.shape[-1]
    blocks_per_row = width // block_size
    rows = x.numel() // width
    flat_x = x.reshape(rows, width)
    flat_codes = codes.view(-1)
    flat_scales = scales.view(-1)
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // width)
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        block = flat_x[start:stop].to(torch.float32).view(-1, block_size)
        block_codes, block_scales, _ = _torch_blocks(block, scale_format)
        flat_codes[start * width:stop * width].copy_(block_codes.reshape(-1))
        first = start * blocks_per_row
        flat_scales[first:first + (stop - start) * blocks_per_row].copy_(block_scales)
    return codes, scales


def _torch_reference_inplace(x: torch.Tensor, block_size: int, scale_format: str) -> torch.Tensor:
    width = x.shape[-1]
    rows = x.numel() // width
    flat_x = x.reshape(rows, width)
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // width)
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        block = flat_x[start:stop].to(torch.float32).view(-1, block_size)
        codes, _, scale_value = _torch_blocks(block, scale_format)
        values = _levels(block.device, signed=True)[codes.long()]
        dequantized = values * scale_value[:, None]
        flat_x[start:stop].copy_(dequantized.to(x.dtype).view(stop - start, width))
    return x


def _triton_codes(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                  block_size: int, scale_format: str, module) -> tuple[torch.Tensor, torch.Tensor]:
    from ._ascend_fp4_quant_kernels import block_fp4_quant_kernel

    blocks = x.numel() // block_size
    tile = _BLOCKS * block_size
    padded = ((x.numel() + tile - 1) // tile) * tile
    floor = _FLOOR_E8M0 if scale_format == "e8m0" else _FLOOR_E4M3
    with module._npu_launch(x):
        block_fp4_quant_kernel[((blocks + _BLOCKS - 1) // _BLOCKS,)](
            x, codes, scales, blocks, block_size, _BLOCKS, floor, _INV_FP4_MAX,
            scale_format == "e4m3", padded >= 2**31)
    return codes, scales


def _triton_inplace(x: torch.Tensor, block_size: int, scale_format: str, module) -> torch.Tensor:
    from ._ascend_fp4_quant_kernels import block_fp4_quant_dequant_kernel

    blocks = x.numel() // block_size
    tile = _BLOCKS * block_size
    padded = ((x.numel() + tile - 1) // tile) * tile
    floor = _FLOOR_E8M0 if scale_format == "e8m0" else _FLOOR_E4M3
    with module._npu_launch(x):
        block_fp4_quant_dequant_kernel[((blocks + _BLOCKS - 1) // _BLOCKS,)](
            x, blocks, block_size, _BLOCKS, floor, _INV_FP4_MAX, scale_format == "e4m3",
            x.dtype == torch.bfloat16, padded >= 2**31)
    return x


@torch.no_grad()
def quantize_block_fp4(x: torch.Tensor, block_size: int = 32, scale_format: str = "e8m0",
                       inplace: bool = False, backend: str = "auto",
                       out: tuple[torch.Tensor, torch.Tensor] | None = None,
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP32 activations to original block-wise FP4 bytes.

    ``x`` must be contiguous, on CPU/CUDA/NPU, of dtype BF16 or FP32, of rank at
    least one, with a positive last dimension that is a multiple of ``block_size``.
    ``block_size`` is 16 (compressed-KV) or 32 (indexer); those are the two widths
    the released model uses and other widths are refused rather than guessed.
    ``scale_format`` is ``"e8m0"`` (power-of-two scales, the indexer) or ``"e4m3"``
    (rounded-ratio scales, the compressed-KV cache).

    With ``inplace=False`` it returns ``(packed, scales)``: ``packed`` holds two raw
    E2M1 codes per byte along the last axis, low nibble first, shape
    ``x.shape[:-1] + (K // 2,)``; ``scales`` holds one raw scale byte per block,
    shape ``x.shape[:-1] + (K // block_size,)``. With ``inplace=True`` the
    quantize-dequantize result is written back into ``x`` in ``x``'s own dtype and
    ``x`` is returned, as the released kernel's ``inplace`` branch does; ``out`` is
    then not allowed.

    ``out``, if given, must be a ``(packed, scales)`` pair of contiguous uint8
    tensors of exactly those shapes on the input device, disjoint from ``x`` and
    from each other; the same two objects are returned. ``backend`` selects the
    same-device Torch reference (``"torch"``), the fused Triton Ascend kernel
    (``"triton"``), or ``"auto"``: Torch off NPU, and on NPU the fused kernel,
    raising rather than silently substituting an unsupported float8 cast.

    These are quantization outputs, not INT8 activations and not a replacement
    weight format. A block containing infinity or NaN becomes all-zero codes with
    the format's NaN scale byte (``255`` for E8M0, ``0x7F`` for E4M3) and, when
    ``inplace``, NaN values -- an explicit extension of the released finite
    contract, not a claim about the released kernel's behaviour.
    """
    values = _checked_input(x, block_size)
    if not isinstance(scale_format, str) or scale_format not in _NAN_SCALE:
        raise ValueError("scale_format must be 'e8m0' or 'e4m3'")
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a bool")
    if inplace and out is not None:
        raise ValueError("inplace=True writes into x and takes no out tensors")
    shapes = ((*values.shape[:-1], values.shape[-1] // 2),
              (*values.shape[:-1], values.shape[-1] // block_size))
    codes = scales = packed = None
    if not inplace:
        packed, scales = _outputs(out, shapes, values.device, values)
        # Codes are an implementation detail of the packed result; the kernel
        # writes them contiguously and the nibble merge happens after.
        codes = torch.empty(values.shape, dtype=torch.uint8, device=values.device)
    implementation = _select_backend(backend, values.device)
    if values.numel() == 0:
        if inplace:
            return values
        return packed, scales
    if inplace:
        if implementation is None:
            return _torch_reference_inplace(values, block_size, scale_format)
        return _triton_inplace(values, block_size, scale_format, implementation)
    if implementation is None:
        _torch_reference(values, codes, scales, block_size, scale_format)
    else:
        _triton_codes(values, codes, scales, block_size, scale_format, implementation)
    packed.copy_(_pack_codes(codes))
    return packed, scales