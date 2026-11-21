"""Exact online decoding of released floating-format weight bytes.

``unpack_fp4_to_int8`` widens each original E2M1 nibble to twice its numeric
value, not to a newly quantized row. Keep the original E8M0 block scales and
multiply the widened values by ``fp4_int8_scale(scales)``. Integer zero cannot
retain E2M1's negative-zero sign; the input packed bytes are never changed.

``unpack_fp8_to_bf16`` decodes E4M3FN codes, optionally applies the original
E8M0 scale grid in FP32, and narrows once to BF16. This is wider execution of
original floating encodings, not native FP4/FP8 arithmetic or an integer
activation-quantization scheme. It does not establish GEMM or model parity.

The Torch path runs payload operations on the input device. Only constant
lookup tables are constructed on the CPU and copied once per device. The largest
is a 128 KiB joint code/scale BF16 table, which preserves subnormal output bits
without relying on the NPU's FP32 arithmetic denormal mode. The optional Triton
Ascend path is selected before execution; kernel failures are
never swallowed and retried through Torch. No CUDA extension or model module
is imported here. These byte-decoding operations are non-differentiable.
"""
from __future__ import annotations

import importlib
import math
import struct
from typing import Any

import torch

__all__ = ["unpack_fp4_to_int8", "unpack_fp8_to_bf16", "fp4_int8_scale"]

_FP4_X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
# Work buffers, unlike the requested output, are bounded even for a large weight.
_TORCH_CHUNK_ELEMENTS = 65536
_TABLES: dict[tuple[str, torch.device], torch.Tensor] = {}


def _e4m3_value(byte: int) -> float:
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        magnitude = float("nan")
    elif exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return math.copysign(magnitude, -1.0 if byte & 128 else 1.0)


_FP8_VALUES = tuple(_e4m3_value(byte) for byte in range(256))
_E8M0_VALUES = tuple(math.ldexp(1.0, byte - 127) for byte in range(255)) + (float("nan"),)
_FP4_HALF_SCALES = tuple(math.ldexp(1.0, byte - 128) for byte in range(255)) + (float("nan"),)
_JOINT_FP8_BITS: tuple[int, ...] | None = None


def _bf16_storage_bits(value: float) -> int:
    """FP32 evaluation followed by BF16 RNE, returned as a signed INT16 container.

    Host integer packing preserves tiny BF16 values independently of a device's
    FP32 flush-to-zero mode. NaN payload bits are not a public parity contract.
    """
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        packed = struct.pack("<f", math.copysign(float("inf"), value))
    bits = struct.unpack("<I", packed)[0]
    if bits & 0x7F800000 == 0x7F800000:
        encoded = (bits >> 16) | (0x40 if bits & 0x7FFFFF else 0)
    else:
        encoded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return encoded if encoded < 0x8000 else encoded - 0x10000


def _joint_fp8_bits() -> tuple[int, ...]:
    global _JOINT_FP8_BITS
    if _JOINT_FP8_BITS is None:
        _JOINT_FP8_BITS = tuple(_bf16_storage_bits(code * scale)
                               for code in _FP8_VALUES for scale in _E8M0_VALUES)
    return _JOINT_FP8_BITS


def _table(name: str, device: torch.device) -> torch.Tensor:
    key = (name, device)
    table = _TABLES.get(key)
    if table is None:
        if name == "fp8":
            values, dtype = tuple(_bf16_storage_bits(value) for value in _FP8_VALUES), torch.int16
        elif name == "fp8_scaled":
            values, dtype = _joint_fp8_bits(), torch.int16
        else:
            values, dtype = {
                "fp4_x2": (_FP4_X2, torch.int8),
                "fp4_half_scale": (_FP4_HALF_SCALES, torch.float32),
            }[name]
        table = torch.tensor(values, dtype=dtype, device="cpu")
        if name in {"fp8", "fp8_scaled"}:
            table = table.view(torch.bfloat16)
        table = table.to(device=device)
        _TABLES[key] = table
    return table


def _byte_input(value: torch.Tensor, name: str, *, encoding: str, min_rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must have a dense strided layout")
    if value.ndim < min_rank:
        raise ValueError(f"{name} must have rank >= {min_rank}")
    if value.device.type not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"{name} uses unsupported device {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous; no implicit payload copy is made")
    allowed = {torch.uint8, torch.int8}
    optional = {
        "fp4": "float4_e2m1fn_x2",
        "fp8": "float8_e4m3fn",
        "e8m0": "float8_e8m0fnu",
    }[encoding]
    dtype = getattr(torch, optional, None)
    if dtype is not None:
        allowed.add(dtype)
    if value.dtype not in allowed:
        raise TypeError(f"{name} must contain original {encoding} byte codes, got {value.dtype}")
    # A dtype view preserves the bit pattern; .to(uint8) would be a numeric cast.
    return value.view(torch.uint8)


def _block_shape(value: tuple[int, int]) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value)):
        raise ValueError("block_shape must contain two positive integer dimensions")
    return value[0], value[1]


def _output(out: torch.Tensor | None, shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    if out.layout != torch.strided or not out.is_contiguous():
        raise ValueError("out must be a contiguous dense strided tensor")
    if out.dtype != dtype:
        raise TypeError(f"out must have dtype {dtype}, got {out.dtype}")
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    if out.device != device:
        raise ValueError(f"out must be on the same device as the inputs ({device})")
    # Conservative storage overlap also refuses disjoint views of one allocation.
    if any(torch._C._overlaps(out, source) for source in inputs):
        raise ValueError("out must not alias or share storage with an input")
    return out


def _select_backend(backend: str, device: torch.device) -> Any | None:
    if not isinstance(backend, str) or backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of 'auto', 'torch', or 'triton'")
    if backend == "torch" or (backend == "auto" and device.type != "npu"):
        return None
    if device.type != "npu":
        raise ValueError("the Triton Ascend backend requires an NPU input")
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError as error:
        if backend == "auto":
            return None
        raise RuntimeError("the Triton Ascend backend is unavailable") from error
    if not module.supports_triton_ascend():
        if backend == "auto":
            return None
        raise RuntimeError("the Triton Ascend backend is unavailable")
    return module


@torch.no_grad()
def unpack_fp4_to_int8(packed: torch.Tensor, *, backend: str = "auto",
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """Widen packed E2M1 values exactly to INT8 x2, low nibble first.

    Input is contiguous original I8/U8/optional float4 storage of rank >= 1.
    The output doubles only the last dimension. No scales are read or rewritten;
    multiply by the separately decoded original E8M0 half-scale to reconstruct
    the weight. Both signed-zero nibbles become integer zero. ``out``, if given,
    must be disjoint contiguous INT8 storage on the same device.
    """
    raw = _byte_input(packed, "packed", encoding="fp4", min_rank=1)
    implementation = _select_backend(backend, raw.device)
    shape = (*raw.shape[:-1], raw.shape[-1] * 2)
    result = _output(out, shape, torch.int8, raw.device, (packed,))
    if raw.numel() == 0:
        return result
    if implementation is not None:
        implementation.fp4_unpack_int8(raw, result)
        return result
    lookup, flat, target = _table("fp4_x2", raw.device), raw.reshape(-1), result.reshape(-1)
    for start in range(0, flat.numel(), _TORCH_CHUNK_ELEMENTS):
        stop = min(start + _TORCH_CHUNK_ELEMENTS, flat.numel())
        chunk = flat[start:stop]
        target[2 * start:2 * stop:2].copy_(lookup[(chunk & 15).long()])
        target[2 * start + 1:2 * stop:2].copy_(lookup[(chunk >> 4).long()])
    return result


@torch.no_grad()
def unpack_fp8_to_bf16(weight: torch.Tensor, scale: torch.Tensor | None = None, *,
                       block_shape: tuple[int, int] = (32, 32), backend: str = "auto",
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """Decode E4M3FN bytes and optional E8M0 blocks in FP32, then narrow to BF16.

    An unscaled input may have any rank >= 1. A scaled matrix/batch requires
    ``scale.shape == weight.shape[:-2] + (ceil(M/bm), ceil(K/bk))``. Thus
    ``block_shape=(32,32)`` describes dense projections and ``(1,32)`` describes
    row-scaled tables. Leading batch dimensions do not broadcast. Ragged final
    blocks are supported. All payload tensors must be contiguous on one device.
    The original codes/scales are not modified; no NPU float8 cast is used.
    Torch gathers pre-rounded BF16 bit patterns from a joint code/scale table,
    rather than multiplying subnormal FP32 payloads on a flush-to-zero device.
    """
    raw = _byte_input(weight, "weight", encoding="fp8", min_rank=1)
    bm, bk = _block_shape(block_shape)
    scale_raw = None
    inputs = (weight,)
    if scale is not None:
        if raw.ndim < 2:
            raise ValueError("a scaled FP8 weight must have rank >= 2")
        scale_raw = _byte_input(scale, "scale", encoding="e8m0", min_rank=2)
        if scale_raw.device != raw.device:
            raise ValueError("weight and scale must be on the same device")
        rows, columns = raw.shape[-2:]
        grid_rows, grid_columns = (rows + bm - 1) // bm, (columns + bk - 1) // bk
        expected = (*raw.shape[:-2], grid_rows, grid_columns)
        if tuple(scale_raw.shape) != expected:
            raise ValueError(f"scale must have shape {expected}, got {tuple(scale_raw.shape)}")
        inputs = (weight, scale)
    implementation = _select_backend(backend, raw.device)
    result = _output(out, tuple(raw.shape), torch.bfloat16, raw.device, inputs)
    if raw.numel() == 0:
        return result
    if implementation is not None:
        implementation.fp8_unpack_bf16(raw, scale_raw, result, (bm, bk))
        return result
    lookup = _table("fp8" if scale_raw is None else "fp8_scaled", raw.device)
    flat, target = raw.reshape(-1), result.reshape(-1)
    scale_flat = None if scale_raw is None else scale_raw.reshape(-1)
    for start in range(0, flat.numel(), _TORCH_CHUNK_ELEMENTS):
        stop = min(start + _TORCH_CHUNK_ELEMENTS, flat.numel())
        code_indices = flat[start:stop].long()
        if scale_flat is not None:
            indices = torch.arange(start, stop, dtype=torch.int64, device=raw.device)
            batch = indices // (rows * columns)
            row = (indices // columns) % rows
            column = indices % columns
            grid_indices = (batch * grid_rows + row // bm) * grid_columns + column // bk
            code_indices = code_indices * 256 + scale_flat[grid_indices].long()
        target[start:stop].copy_(lookup[code_indices])
    return result


@torch.no_grad()
def fp4_int8_scale(scale: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """Decode the original E8M0 block scale divided by two as FP32.

    Byte 0 is ``2**-128`` after compensation, not zero; byte 255 is NaN.
    Keep this grid's original shape and K-block layout. This is not a row amax,
    new quantization scale, or integer zero-point. Scalar grids are accepted.
    """
    raw = _byte_input(scale, "scale", encoding="e8m0", min_rank=0)
    implementation = _select_backend(backend, raw.device)
    result = torch.empty(tuple(raw.shape), dtype=torch.float32, device=raw.device)
    if raw.numel() == 0:
        return result
    if implementation is not None:
        implementation.fp4_int8_scale(raw, result)
        return result
    lookup, flat, target = _table("fp4_half_scale", raw.device), raw.reshape(-1), result.reshape(-1)
    for start in range(0, flat.numel(), _TORCH_CHUNK_ELEMENTS):
        stop = min(start + _TORCH_CHUNK_ELEMENTS, flat.numel())
        target[start:stop].copy_(lookup[flat[start:stop].long()])
    return result
