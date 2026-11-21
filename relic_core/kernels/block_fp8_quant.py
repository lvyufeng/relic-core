"""Original block-32 FP8 activation quantization for the released V4.1 kernels.

This reproduces the released inference kernel's ``act_quant_kernel``. For each
consecutive block of 32 values along the last axis it computes, in FP32,

1. ``amax = max(|x|)``, floored at ``1e-4``;
2. ``scale = 2**ceil(log2(amax * fp32(1/448)))`` -- a power of two selected by
   IEEE bit manipulation, never ``amax / 448`` and never ``log2``/``exp2``;
3. each value multiplied by the exact reciprocal of that scale, clamped to
   ``[-448, 448]``;
4. the clamped value encoded as E4M3 with round-to-nearest-even.

The outputs are original bytes, not a torch floating dtype: ``codes`` holds one
raw E4M3FN byte per element and ``scales`` one raw E8M0 byte per block, whose
value is exactly ``2**(byte - 127)``. Keeping bytes is what lets the NPU path
avoid an unsupported float8 cast and avoids relying on native FP8 arithmetic,
which this operator does not need.

This is activation *quantization*, not an integer substitution: it produces no
row amax, no static per-tensor scale, no zero point and no INT8 activation, and
it never modifies its input. ``torch`` runs a same-device reference that ends in
the native FP8 cast and is the independent oracle on CPU; ``triton`` runs the
fused Ascend kernel whose encode is integer bit arithmetic. ``auto`` picks the
fused kernel for NPU input and the Torch reference elsewhere, never falls back
after a backend has been chosen, and never silently substitutes one for the
other after a kernel or compiler failure.

Non-finite input is outside the released finite contract. This operator extends
that contract explicitly: a block containing any infinity or NaN maps every one
of its 32 codes to the E4M3 NaN byte ``0x7F`` and its scale to the E8M0 NaN byte
``255``. No claim is made about how the released kernel treats such a block.
"""
from __future__ import annotations

import importlib
import struct

import torch

__all__ = ["quantize_block_fp8", "supports_triton_ascend"]

#: One E8M0 scale per 32 consecutive elements of the last axis, as in the
#: released kernel's block size.
BLOCK = 32
#: Blocks encoded per program: 16 * 32 = 512 active lanes, a power of two that
#: amortises the block reduction without asking the backend for a huge tile.
BLOCKS = 16
LANES = BLOCK * BLOCKS

_FP8_MAX = 448.0
_CHUNK_ELEMENTS = 1 << 20
_CONSTANTS: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _float32(value: float) -> float:
    """The FP32 rounding of a Python float, returned as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


# Both constants are the ones the released kernel multiplies in FP32: the amax
# floor and ``1/448`` as an FP32 literal. Rounding them here keeps the Torch
# reference and the constexpr kernel arguments bit-identical.
_FLOOR = _float32(1e-4)
_INV_FP8_MAX = _float32(1.0 / 448.0)


def _torch_fp8_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("this torch build has no float8_e4m3fn dtype; the Torch "
                           "reference needs it and the Triton backend is NPU-only")
    return dtype


def supports_triton_ascend() -> bool:
    """Report an installed Ascend Triton backend without starting device work.

    Delegates to the fused online-unpack backend's probe so the package keeps one
    definition of "a compatible Triton Ascend install is present".
    """
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError:
        return False
    return bool(module.supports_triton_ascend())


def _checked_input(x: torch.Tensor) -> torch.Tensor:
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
    if width % BLOCK:
        raise ValueError(f"the last dimension must be a multiple of {BLOCK}, got {width}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous; no implicit copy is made")
    return x


def _overlaps(first: torch.Tensor, second: torch.Tensor) -> bool:
    structural = getattr(torch._C, "_overlaps", None)
    if structural is not None:
        return bool(structural(first, second))
    # Conservative fallback: refuse any shared storage rather than guess ranges.
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _outputs(out, shapes: tuple[tuple[int, ...], tuple[int, ...]], device: torch.device,
             source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        return tuple(torch.empty(shape, dtype=torch.uint8, device=device) for shape in shapes)
    if (not isinstance(out, (tuple, list)) or len(out) != 2
            or not all(isinstance(item, torch.Tensor) for item in out)):
        raise TypeError("out must be a (codes, scales) pair of uint8 tensors or None")
    codes, scales = out
    if codes is scales:
        raise ValueError("out codes and scales must be distinct tensors")
    for name, tensor, shape in zip(("codes", "scales"), (codes, scales), shapes):
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"out {name} must be a contiguous dense strided tensor")
        if tensor.dtype != torch.uint8:
            raise TypeError(f"out {name} must have dtype torch.uint8, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"out {name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"out {name} must be on the input device ({device})")
    for first, second in ((codes, scales), (codes, source), (scales, source)):
        if _overlaps(first, second):
            raise ValueError("out tensors must not alias or share storage with an input")
    return codes, scales


def _select_backend(backend: str, device: torch.device):
    if not isinstance(backend, str) or backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of 'auto', 'torch', or 'triton'")
    if backend == "torch":
        if device.type == "npu":
            raise ValueError("the Torch FP8 reference is CPU/CUDA-only; NPU execution requires the fused backend")
        return None
    if device.type != "npu":
        if backend == "triton":
            raise ValueError("the Triton Ascend backend requires an NPU input")
        return None
    # The NPU stack this package targets does not provide a float8 cast, so
    # `auto` cannot fall back to the Torch reference here: it needs the fused
    # kernel or an explicit failure. Silence would look like a working cast.
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError as error:
        raise RuntimeError("NPU block FP8 quantization requires the fused Triton Ascend "
                           "backend, which is not importable") from error
    if not module.supports_triton_ascend():
        raise RuntimeError("NPU block FP8 quantization requires an installed Triton Ascend "
                           "backend; there is no NPU float8 cast fallback")
    return module


def _constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device)
    cached = _CONSTANTS.get(key)
    if cached is None:
        cached = (torch.tensor(_FLOOR, dtype=torch.float32, device=device),
                  torch.tensor(_INV_FP8_MAX, dtype=torch.float32, device=device))
        _CONSTANTS[key] = cached
    return cached


def _ceil_log2(value: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(value))`` for positive finite FP32 values, from their bits."""
    bits = value.view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    return exponent - 127 + (mantissa != 0).to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """FP32 values of ``2**exponent`` for integer exponents, from their bits."""
    return ((exponent + 127) << 23).view(torch.float32)


def _torch_reference(x: torch.Tensor, codes: torch.Tensor,
                     scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference on the input device, ending in the native E4M3 cast."""
    width = x.shape[-1]
    blocks_per_row = width // BLOCK
    rows = x.numel() // width
    floor, inverse = _constants(x.device)
    flat_x = x.reshape(rows, width)
    flat_codes = codes.view(-1)
    flat_scales = scales.view(-1)
    nan_codes = torch.tensor(0x7F, dtype=torch.uint8, device=x.device)
    nan_scale = torch.tensor(255, dtype=torch.uint8, device=x.device)
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // width)
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        block = flat_x[start:stop].to(torch.float32).view(-1, BLOCK)
        nonfinite = (~torch.isfinite(block)).any(dim=1)
        amax = block.abs().amax(dim=1).clamp_min(floor)
        exponent = _ceil_log2(amax * inverse)
        normalized = block * _pow2(-exponent)[:, None]
        normalized = normalized.clamp(-_FP8_MAX, _FP8_MAX)
        encoded = normalized.to(_torch_fp8_dtype()).view(torch.uint8)
        encoded = torch.where(nonfinite[:, None], nan_codes, encoded)
        # The mask keeps the byte cast defined: a non-finite block's exponent is
        # meaningless, and its value is replaced by the NaN byte anyway.
        scale_bytes = torch.where(nonfinite, nan_scale, ((exponent + 127) & 0xFF).to(torch.uint8))
        flat_codes[start * width:stop * width].copy_(encoded.reshape(-1))
        first_block = start * blocks_per_row
        flat_scales[first_block:first_block + (stop - start) * blocks_per_row].copy_(scale_bytes)
    return codes, scales


def _triton_quantize(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                     module) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Ascend kernel on the caller's current stream and device."""
    if x.numel() == 0:
        return codes, scales
    from ._ascend_fp8_quant_kernels import block_fp8_quant_kernel

    blocks = x.numel() // BLOCK
    tile = BLOCKS * BLOCK
    # Masked tail lanes count towards the pointer width, exactly like the fused
    # unpack kernels: an index that fits only in int64 must never be int32.
    padded = ((x.numel() + tile - 1) // tile) * tile
    # Reuses the fused-unpack NPU launch context: same device scope, same
    # "Triton's active target really is Ascend" check, no stream or sync changes.
    with module._npu_launch(x):
        block_fp8_quant_kernel[((blocks + BLOCKS - 1) // BLOCKS,)](
            x, codes, scales, blocks, BLOCK, BLOCKS, _FLOOR, _INV_FP8_MAX,
            padded >= 2**31)
    return codes, scales


@torch.no_grad()
def quantize_block_fp8(x: torch.Tensor, backend: str = "auto",
                       out: tuple[torch.Tensor, torch.Tensor] | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP32 activations to original block-32 FP8 bytes and scales.

    ``x`` must be contiguous, on CPU/CUDA/NPU, of dtype BF16 or FP32, of rank at
    least one, with a positive last dimension that is a multiple of 32. Its
    leading axes are flattened into blocks rows, so any shape works.

    Returns ``(codes, scales)``: ``codes`` has ``x``'s shape and holds one raw
    E4M3FN byte per element; ``scales`` has ``x.shape[:-1] + (K // 32,)`` and
    holds one raw E8M0 byte per block, whose value is ``2**(byte - 127)``. A
    block containing infinity or NaN becomes 32 copies of the NaN code ``0x7F``
    with the NaN scale byte ``255`` -- an explicit extension of the released
    finite contract, not a claim about the released kernel's behaviour.

    ``out``, if given, must be a ``(codes, scales)`` pair of contiguous uint8
    tensors of exactly those shapes on the input device, disjoint from ``x`` and
    from each other; the same two objects are returned. ``backend`` selects the
    same-device Torch reference (``"torch"``), the fused Triton Ascend kernel
    (``"triton"``), or ``"auto"``: Torch off NPU, and on NPU the fused kernel,
    raising rather than silently substituting an unsupported float8 cast.

    Negative zero keeps its sign in the code byte (``0x80``); the amax floor
    keeps the scale positive, so no finite block loses that distinction. These
    are quantization outputs, not INT8 activations and not a dequantized tensor.
    """
    values = _checked_input(x)
    shapes = (tuple(values.shape), (*values.shape[:-1], values.shape[-1] // BLOCK))
    codes, scales = _outputs(out, shapes, values.device, values)
    if values.numel() == 0:
        return codes, scales
    implementation = _select_backend(backend, values.device)
    if implementation is None:
        return _torch_reference(values, codes, scales)
    return _triton_quantize(values, codes, scales, implementation)