"""Original block-32 FP8 activation quantization fused with its exact BF16 decode.

This is the specialized variant of :mod:`fused_fp8_activation_decode`: same contract,
same bytes, same BF16 bits, and the same Torch oracle, but the device kernel decides the
two lane-dependent branches with constant shifts instead of per-lane shift amounts. See
:mod:`_ascend_fast_fused_fp8_activation_decode_kernels` for the two algebraic facts that
make that legal (``|value| * 512`` for the subnormal encode, and the ``scale >= 105``
proof for the decode).

Validation, the backend selection and the Torch oracle are the frozen wrapper's own --
imported, not retyped -- so this module cannot disagree with the operator it specializes
about what a valid input is or what the reference answer is. Only the Triton launch is
new. The frozen default path is untouched: nothing here is selected unless a caller asks
for it by name.

``backend="torch"`` composes the two frozen Torch references and is the CPU/CUDA
oracle. ``backend="triton"`` runs the fused Ascend kernel and requires an NPU;
``"auto"`` selects Torch off NPU and the fused kernel on NPU, raising rather than
substituting an unsupported float8 cast or falling back after a kernel or compiler
failure.
"""
from __future__ import annotations

import torch

from .fused_fp8_activation_decode import (
    BLOCK,
    BLOCKS,
    FLOOR,
    INV_FP8_MAX,
    MAX_PROGRAMS,
    _checked_input,
    _checked_out,
    _select_backend,
    _torch_reference,
)

__all__ = ["BLOCK", "BLOCKS", "MAX_PROGRAMS", "quantize_decode_fp8_fast"]


def _triton_fast(x: torch.Tensor, result: torch.Tensor, codes, scales, module):
    """Run the fused constant-shift kernel on the caller's current stream and device."""
    from ._ascend_fast_fused_fp8_activation_decode_kernels import (
        fast_fused_fp8_activation_decode_kernel,
    )

    blocks = x.numel() // BLOCK
    # Masked tail lanes count towards the pointer width: an index that fits only in
    # int64 must never be int32, and the masking does not make it unreachable.
    padded = ((blocks + BLOCKS - 1) // BLOCKS) * BLOCKS * BLOCK
    # Reuses the fused-unpack NPU launch context: same device scope, same
    # "Triton's active target really is Ascend" check, no stream or sync changes.
    with module._npu_launch(x):
        fast_fused_fp8_activation_decode_kernel[(min(MAX_PROGRAMS, -(-blocks // BLOCKS)),)](
            x, result,
            # The dead branch reads neither buffer; the output stands in so the
            # kernel never receives a pointer it could touch by accident.
            result if codes is None else codes,
            result if scales is None else scales,
            blocks, BLOCK, BLOCKS, FLOOR, INV_FP8_MAX, codes is not None,
            padded >= 2**31)
    return result


@torch.no_grad()
def quantize_decode_fp8_fast(x: torch.Tensor, backend: str = "auto",
                             out: torch.Tensor | None = None,
                             return_codes: bool = False):
    """Quantize BF16/FP32 activations to original block-32 FP8 and decode to BF16.

    The contract is :func:`fused_fp8_activation_decode.quantize_decode_fp8`'s, argument
    for argument and bit for bit; only the device kernel behind ``backend="triton"``
    differs, and it produces the same bytes and the same decoded values. ``x`` must be
    contiguous, on CPU/CUDA/NPU, of dtype BF16 or FP32, of rank at least one, with a
    positive last dimension that is a multiple of 32; its leading axes are flattened
    into rows of blocks.

    Returns the BF16 tensor of ``x``'s shape holding exactly what
    ``quantize_block_fp8`` followed by ``unpack_fp8_to_bf16(block_shape=(1, 32))``
    would produce on the same backend. With ``return_codes=True`` the return value is
    ``(decoded, codes, scales)``: ``codes`` has ``x``'s shape and ``scales`` has
    ``x.shape[:-1] + (K // 32,)``, both holding the raw bytes the frozen quantizer
    emits.

    ``out``, if given, must be a contiguous BF16 tensor of ``x``'s shape on the input
    device, disjoint from ``x``; the same object is returned as the decoded tensor.
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
    decoded = _triton_fast(values, result, codes, scales, implementation)
    return (decoded, codes, scales) if return_codes else decoded