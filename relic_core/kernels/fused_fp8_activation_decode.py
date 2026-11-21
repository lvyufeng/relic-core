"""Original block-32 FP8 activation quantization fused with its exact BF16 decode.

The promoted-BF16 path runs two device kernels per projection: ``quantize_block_fp8``
writes the original E4M3 code bytes and E8M0 scale bytes, then
``unpack_fp8_to_bf16`` reads them back and emits BF16. When the dequantized operand
is the only thing consumed, the second kernel reads exactly the bytes the first just
wrote, so the round trip through memory buys nothing.

This module fuses them into one kernel. The encode decisions are the frozen ones,
imported rather than re-typed: one raw E8M0 byte per 32 consecutive elements, chosen
by ``ceil(log2(amax * fp32(1/448)))`` in IEEE bit arithmetic with the FP32 ``1e-4``
amax floor; each value multiplied by the exact power-of-two reciprocal, clamped to
``[-448, 448]`` and encoded as E4M3 with round-to-nearest-even. The decode then
consumes the byte pair just produced, so the BF16 result is bit-identical to
quantize-then-decode: one rounding of the original value, and it is the same
rounding.

Nothing is relaxed. This is not an INT8 activation, not W8A8/W4A8 and not a new
scale grid; the emitted BF16 values are the original FP8 values the released kernel
would have produced, and a block holding an infinity or NaN keeps the frozen
semantics -- the E4M3 NaN byte with the E8M0 NaN scale byte, decoded to BF16 NaN. No
normal-range assumption is made: zero, negative zero and subnormal magnitudes go
through the same signed-zero and subnormal decode paths as the two-kernel form.

``backend="torch"`` composes the two frozen Torch references and is the CPU/CUDA
oracle. ``backend="triton"`` runs the fused Ascend kernel and requires an NPU;
``"auto"`` selects Torch off NPU and the fused kernel on NPU, raising rather than
substituting an unsupported float8 cast or falling back after a kernel or compiler
failure.
"""
from __future__ import annotations

import importlib

import torch

from .block_fp8_quant import BLOCK
# The released FP32 constants, taken from the operator that defines them: the amax
# floor and ``1/448`` exactly as the frozen quantizer rounds them. Re-deriving them
# here would be a second definition of the same released number.
from .block_fp8_quant import _FLOOR as FLOOR
from .block_fp8_quant import _INV_FP8_MAX as INV_FP8_MAX

__all__ = ["BLOCK", "BLOCKS", "MAX_PROGRAMS", "quantize_decode_fp8"]

#: Blocks encoded and decoded per program: 16 * 32 = 512 active lanes, the tile the
#: frozen quantizer already compiles on this backend, which amortises the block
#: reduction without asking for a large buffer.
BLOCKS = 16
#: Ascend refuses a launch grid above this; the kernel strides the remaining tiles.
MAX_PROGRAMS = 65535


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


def _checked_out(out, x: torch.Tensor) -> torch.Tensor:
    """The BF16 output buffer, validated, or a fresh one."""
    if out is None:
        return torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    if out.layout != torch.strided or not out.is_contiguous():
        raise ValueError("out must be a contiguous dense strided tensor")
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must have dtype torch.bfloat16, got {out.dtype}")
    if tuple(out.shape) != tuple(x.shape):
        raise ValueError(f"out must have shape {tuple(x.shape)}, got {tuple(out.shape)}")
    if out.device != x.device:
        raise ValueError(f"out must be on the input device ({x.device})")
    if _overlaps(out, x):
        raise ValueError("out must not alias or share storage with x")
    return out


def _select_backend(backend: str, device: torch.device):
    if not isinstance(backend, str) or backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of 'auto', 'torch', or 'triton'")
    if backend == "torch":
        if device.type == "npu":
            raise ValueError("the Torch fused reference is CPU/CUDA-only; NPU execution "
                             "requires the fused backend")
        return None
    if device.type != "npu":
        if backend == "triton":
            raise ValueError("the Triton Ascend backend requires an NPU input")
        return None
    # The NPU stack this package targets has no float8 cast, so ``auto`` cannot fall
    # back to the Torch reference here: it needs the fused kernel or an explicit
    # failure. Silence would look like a working cast.
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError as error:
        raise RuntimeError("NPU fused FP8 activation quantization requires the fused Triton "
                           "Ascend backend, which is not importable") from error
    if not module.supports_triton_ascend():
        raise RuntimeError("NPU fused FP8 activation quantization requires an installed Triton "
                           "Ascend backend; there is no NPU float8 cast fallback")
    return module


def _torch_reference(x: torch.Tensor, result: torch.Tensor, *,
                     return_codes: bool):
    """The two frozen Torch references composed, on the input device.

    This is the oracle the fused kernel is held to, and it is literally the pair of
    operators the fusion replaces rather than a third implementation of them.
    """
    from .block_fp8_quant import quantize_block_fp8
    from .unpack import unpack_fp8_to_bf16

    width = x.shape[-1]
    # Both frozen operators describe a matrix: one row of scale per output row. The
    # block structure is along the last axis, so flattening the leading axes is a
    # relabelling of which row a block belongs to, never a regrouping of blocks.
    flat = x.reshape(-1, width)
    codes, scales = quantize_block_fp8(flat, backend="torch")
    decoded = unpack_fp8_to_bf16(codes, scales, block_shape=(1, BLOCK), backend="torch")
    result.copy_(decoded.reshape(x.shape))
    if not return_codes:
        return result
    return (result, codes.reshape(x.shape), scales.reshape(*x.shape[:-1], width // BLOCK))


def _triton_fused(x: torch.Tensor, result: torch.Tensor, codes, scales, module):
    """Run the fused kernel on the caller's current stream and device."""
    from ._ascend_fused_fp8_activation_decode_kernels import (
        fused_fp8_activation_decode_kernel,
    )

    blocks = x.numel() // BLOCK
    # Masked tail lanes count towards the pointer width: an index that fits only in
    # int64 must never be int32, and the masking does not make it unreachable.
    padded = ((blocks + BLOCKS - 1) // BLOCKS) * BLOCKS * BLOCK
    # Reuses the fused-unpack NPU launch context: same device scope, same
    # "Triton's active target really is Ascend" check, no stream or sync changes.
    with module._npu_launch(x):
        fused_fp8_activation_decode_kernel[(min(MAX_PROGRAMS, -(-blocks // BLOCKS)),)](
            x, result,
            # The dead branch reads neither buffer; the output stands in so the
            # kernel never receives a pointer it could touch by accident.
            result if codes is None else codes,
            result if scales is None else scales,
            blocks, BLOCK, BLOCKS, FLOOR, INV_FP8_MAX, codes is not None,
            padded >= 2**31)
    return result


@torch.no_grad()
def quantize_decode_fp8(x: torch.Tensor, backend: str = "auto",
                        out: torch.Tensor | None = None,
                        return_codes: bool = False):
    """Quantize BF16/FP32 activations to original block-32 FP8 and decode to BF16.

    ``x`` must be contiguous, on CPU/CUDA/NPU, of dtype BF16 or FP32, of rank at
    least one, with a positive last dimension that is a multiple of 32. Its leading
    axes are flattened into rows of blocks, so any shape works.

    Returns the BF16 tensor of ``x``'s shape holding exactly what
    ``quantize_block_fp8`` followed by ``unpack_fp8_to_bf16(block_shape=(1, 32))``
    would produce on the same backend: the original FP8 values, not an integer
    approximation of them. With ``return_codes=True`` the return value is
    ``(decoded, codes, scales)``, where ``codes`` has ``x``'s shape and ``scales``
    has ``x.shape[:-1] + (K // 32,)``; both hold the same raw bytes the frozen
    quantizer emits. They are filled from the same pass and are not reused buffers.

    ``out``, if given, must be a contiguous BF16 tensor of ``x``'s shape on the input
    device, disjoint from ``x``; the same object is returned as the decoded tensor.
    ``backend`` selects the composed Torch reference (``"torch"``), the fused Triton
    Ascend kernel (``"triton"``), or ``"auto"``: Torch off NPU, and on NPU the fused
    kernel, raising rather than silently substituting an unsupported float8 cast.
    """
    if not isinstance(return_codes, bool):
        raise TypeError(f"return_codes must be a bool, got {type(return_codes).__name__}")
    values = _checked_input(x)
    result = _checked_out(out, values)
    width = values.shape[-1]
    if values.numel() == 0:
        if not return_codes:
            return result
        return (result,
                torch.empty(tuple(values.shape), dtype=torch.uint8, device=values.device),
                torch.empty((*values.shape[:-1], width // BLOCK), dtype=torch.uint8,
                            device=values.device))
    implementation = _select_backend(backend, values.device)
    if implementation is None:
        return _torch_reference(values, result, return_codes=return_codes)
    # Only the fused path consumes these buffers; the reference gets its own from
    # the frozen quantizer it composes.
    codes = torch.empty(tuple(values.shape), dtype=torch.uint8,
                        device=values.device) if return_codes else None
    scales = torch.empty((*values.shape[:-1], width // BLOCK), dtype=torch.uint8,
                         device=values.device) if return_codes else None
    decoded = _triton_fused(values, result, codes, scales, implementation)
    return (decoded, codes, scales) if return_codes else decoded