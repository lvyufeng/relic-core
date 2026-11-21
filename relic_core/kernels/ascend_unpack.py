"""Optional fused Triton Ascend backend for online original-format unpacking.

The public validation/reference contract is in unpack.py. This module owns only
the NPU launch, and never falls back after a kernel or compiler failure. Native
FP8/FP4 matmul support is not required: kernels read original bytes and produce
BF16 or exact widened INT8 values on the current NPU stream.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import math


BLOCK = 1024
MAX_PROGRAMS = 65535


def _grid(numel):
    """Bound the launch below Ascend's grid limit; kernels stride remaining tiles."""
    return (min(MAX_PROGRAMS, math.ceil(numel / BLOCK)),)


def _padded_elements(numel: int) -> int:
    """Include masked tail lanes when selecting the pointer-index integer width."""
    return ((numel + BLOCK - 1) // BLOCK) * BLOCK


def supports_triton_ascend() -> bool:
    """Report an installed Ascend backend without starting an accelerator job."""
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.backends import backends
    except ImportError:
        return False
    return any(name.lower() in {"ascend", "npu"} for name in backends)


@contextmanager
def _npu_launch(raw):
    if raw.device.type != "npu":
        raise ValueError("the fused unpack backend is for NPU tensors only")
    if not supports_triton_ascend():
        raise RuntimeError("a compatible Triton Ascend backend is required for fused online unpack")
    import torch
    import triton
    with torch.npu.device(raw.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend.lower() not in {"ascend", "npu"}:
            raise RuntimeError(f"Triton's active target is not Ascend: {target.backend}")
        yield


def fp4_unpack_int8(raw, out):
    """Expand low/high E2M1 nibbles to exact signed INT8 values multiplied by two."""
    with _npu_launch(raw):
        if raw.numel() == 0:
            return out
        from ._ascend_unpack_kernels import fp4_unpack_int8_kernel
        fp4_unpack_int8_kernel[_grid(raw.numel())](
            raw, out, raw.numel(), BLOCK, _padded_elements(raw.numel()) * 2 >= 2**31)
    return out


def fp8_unpack_bf16(raw, scale_raw, out, block_shape):
    """Decode original E4M3FN/E8M0 directly to correctly rounded BF16 output."""
    with _npu_launch(raw):
        if raw.numel() == 0:
            return out
        from ._ascend_unpack_kernels import fp8_unpack_bf16_kernel
        has_scale = scale_raw is not None
        m = raw.shape[-2] if has_scale else 1
        k = raw.shape[-1]
        block_m, block_k = block_shape
        scale_m, scale_k = math.ceil(m / block_m), math.ceil(k / block_k)
        fp8_unpack_bf16_kernel[_grid(raw.numel())](
            raw, scale_raw if has_scale else raw, out, raw.numel(), m, k,
            scale_m, scale_k, block_m, block_k, has_scale, BLOCK,
            _padded_elements(raw.numel()) >= 2**31)
    return out


def fp4_int8_scale(raw, out):
    """Decode original E8M0 scales divided by two, without a new quantization grid."""
    with _npu_launch(raw):
        if raw.numel() == 0:
            return out
        from ._ascend_unpack_kernels import fp4_int8_scale_kernel
        fp4_int8_scale_kernel[_grid(raw.numel())](
            raw, out, raw.numel(), BLOCK, _padded_elements(raw.numel()) >= 2**31)
    return out
