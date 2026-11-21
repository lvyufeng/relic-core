"""Byte-coded block-32 GEMM with the explicit two-half reduction rule.

This is a vector implementation of a reference reduction, not native FP4/FP8
hardware arithmetic or W8A8. FP4 operands are the exact INT8-times-two values
produced by online unpack; their original half-scale grid is retained.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend

__all__ = ["fp4_int8_block_gemm", "fp8_block_gemm"]
BLOCK = 32
BLOCK_N = 16
WINDOW = 13
EMPTY = -1000


def _dense(tensor, name, dtypes):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected {dtypes}")
    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.ndim != 2:
        raise ValueError(f"{name} must be a contiguous strided matrix")
    if tensor.requires_grad:
        raise ValueError("byte-coded block GEMM is an inference operator, not an autograd operation")
    return tensor


def _bytes(tensor, name, *, scale=False):
    allowed = [torch.uint8, torch.int8]
    floating = getattr(torch, "float8_e8m0fnu", None) if scale else torch.float8_e4m3fn
    if floating is not None:
        allowed.append(floating)
    _dense(tensor, name, tuple(allowed))
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _prepare(a, sa, b, sb, *, fp4, block_rows, out, backend):
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    a = _bytes(a, "activation codes")
    sa = _bytes(sa, "activation scales", scale=True)
    if not isinstance(sb, torch.Tensor):
        raise TypeError("weight scales must be a torch.Tensor")
    if fp4:
        _dense(b, "FP4 INT8-times-two values", (torch.int8,))
        _dense(sb, "FP4 original half-scales", (torch.float32,))
        block_rows = 1
    else:
        b = _bytes(b, "weight codes")
        if sb.dtype != torch.float32:
            sb = _bytes(sb, "weight scales", scale=True)
        else:
            _dense(sb, "weight scales", (torch.float32,))
        if block_rows not in {1, 32}:
            raise ValueError("weight block_rows must be 1 or 32")
    m, k = a.shape
    n, kb = b.shape
    if min(m, n, k) <= 0 or k % BLOCK or kb != k:
        raise ValueError("positive [M,K] and [N,K] matrices need the same K, divisible by 32")
    if tuple(sa.shape) != (m, k // BLOCK):
        raise ValueError(f"activation scales must have shape {(m, k // BLOCK)}")
    if tuple(sb.shape) != (math.ceil(n / block_rows), k // BLOCK):
        raise ValueError("weight scale grid disagrees with its block-row/K layout")
    tensors = (a, sa, b, sb)
    if any(t.device != a.device for t in tensors):
        raise ValueError("all code and scale tensors must be on the same physical device")
    if backend == "auto":
        backend = "triton" if a.device.type == "npu" else "torch"
    if backend == "torch" and a.device.type != "cpu":
        raise ValueError("the bounded Torch oracle is CPU-only; use the fused NPU backend")
    if backend == "triton" and a.device.type != "npu":
        raise ValueError("the fused block GEMM backend is for NPU tensors only")
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    else:
        _dense(out, "output", (torch.float32,))
        if tuple(out.shape) != (m, n) or out.device != a.device:
            raise ValueError("output must match [M,N] and the input device")
        if any(torch._C._overlaps(out, t) for t in tensors):
            raise ValueError("output must not overlap an input")
    return a, sa, b, sb, out, backend, block_rows


def _e8m0_values(raw):
    values = [math.ldexp(1.0, b - 127) for b in range(255)] + [float("nan")]
    table = torch.tensor(values, dtype=torch.float32)
    return table[raw.long()]


def _decode_codes(raw):
    from .unpack import unpack_fp8_to_bf16
    return unpack_fp8_to_bf16(raw.contiguous(), backend="torch").float()


def _nominal(raw):
    nominal = ((raw.to(torch.int32) >> 3) & 15).clamp_min(1) - 7
    return torch.where((raw & 127) != 0, nominal, torch.full_like(nominal, EMPTY))


def _pow2(exponent):
    # Grid exponents are small normal powers. Integer IEEE construction avoids
    # log/exp approximations and never asks a float8 backend to cast a byte.
    exponent = exponent.to(torch.int32)
    return ((exponent + 127) << 23).contiguous().view(torch.float32)


def _cut(value, anchor):
    exponent = torch.where(anchor < -500, torch.zeros_like(anchor), anchor - WINDOW)
    inverse = _pow2(-exponent)
    step = _pow2(exponent)
    return torch.trunc(value * inverse) * step


def _floor_exponent(value):
    raw = value.abs().contiguous().view(torch.int32)
    result = ((raw >> 23) & 255) - 127
    return torch.where(value != 0, result, torch.full_like(result, EMPTY))


def _original_float_scale(values, *, half):
    raw = values.contiguous().view(torch.int32)
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7fffff
    if half:
        special = (raw == 0x00200000) | (raw == 0x00400000)
        valid = special | ((raw >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 253))
        decoded = values * 2.0
    else:
        valid = (raw == 0x00400000) | ((raw >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254))
        decoded = values
    # An invalid FP32 grid is not an original E8M0 scale. Match the fused
    # bit decoder's explicit NaN result rather than silently using another grid.
    return torch.where(valid, decoded, torch.full_like(decoded, float("nan")))


def _torch_gemm(a, sa, b, sb, out, fp4, block_rows):
    m, k = a.shape
    n = b.shape[0]
    out.zero_()
    # The product is never a complete [M,N,K] tensor. A maximum 16x64x32
    # product slab bounds temporary storage independently of checkpoint size.
    for row0 in range(0, m, 16):
        row1 = min(row0 + 16, m)
        for col0 in range(0, n, 64):
            col1 = min(col0 + 64, n)
            tile = out[row0:row1, col0:col1]
            for block in range(k // BLOCK):
                span = slice(block * BLOCK, (block + 1) * BLOCK)
                ac = a[row0:row1, span]
                av, an = _decode_codes(ac), _nominal(ac)
                if fp4:
                    bi = b[col0:col1, span].to(torch.int32)
                    ab = bi.abs()
                    valid = (ab == 0) | (ab == 1) | (ab == 2) | (ab == 3) | (ab == 4) | (ab == 6) | (ab == 8) | (ab == 12)
                    bv = torch.where(valid, bi.float() * 0.5, torch.full_like(bi.float(), float("nan")))
                    bn = torch.where(ab >= 8, 2, torch.where(ab >= 4, 1, torch.where(ab >= 2, 0, -1)))
                    bn = torch.where(ab != 0, bn, torch.full_like(bn, EMPTY))
                else:
                    bc = b[col0:col1, span]
                    bv, bn = _decode_codes(bc), _nominal(bc)
                product = av[:, None, :] * bv[None, :, :]
                nominal = an[:, None, :] + bn[None, :, :]
                first = nominal[..., :16].amax(-1)
                total = _cut(product[..., :16], first[..., None]).sum(-1)
                second = torch.maximum(nominal[..., 16:].amax(-1), _floor_exponent(total))
                total = _cut(total, second) + _cut(product[..., 16:], second[..., None]).sum(-1)
                fragment = _cut(total, _floor_exponent(total))
                scale_a = _e8m0_values(sa[row0:row1, block])
                scale_rows = torch.arange(col0, col1) // block_rows
                scale_b = sb[scale_rows, block]
                if sb.dtype == torch.float32:
                    scale_b = _original_float_scale(scale_b, half=fp4)
                else:
                    scale_b = _e8m0_values(scale_b)
                # Separate rescale/rescale/add is the existing CPU rule's order;
                # combining scale exponents or contracting FMA changes edge cases.
                scaled = (fragment * scale_a[:, None]) * scale_b[None, :]
                tile.add_(scaled)
    return out


def _launch(a, sa, b, sb, out, fp4, block_rows):
    with _npu_launch(a):
        from ._ascend_block_gemm_kernels import block_gemm_kernel
        m, k = a.shape
        n = b.shape[0]
        padded_n = math.ceil(n / BLOCK_N) * BLOCK_N
        index64 = max(m * k, padded_n * k, m * padded_n) >= 2**31
        block_gemm_kernel[(math.ceil(n / BLOCK_N), m)](
            a, sa, b, sb, out, m, n, k, block_rows, fp4,
            sb.dtype == torch.float32, BLOCK_N, index64)
    return out


def fp4_int8_block_gemm(a_codes, a_scale, b_int8_x2, b_half_scale, *, backend="auto", out=None):
    """Reduce original FP8 activations against exact widened FP4 values.

    The FP32 half-scale is an original E8M0 grid divided by two, not a new
    per-channel/row calibration. INT8 operands retain floating E2M1 meaning.
    Values outside the doubled E2M1 codebook produce NaNs. This returns FP32
    accumulators; the model owns its final output narrowing and activation ops.
    """
    a, sa, b, sb, out, backend, rows = _prepare(
        a_codes, a_scale, b_int8_x2, b_half_scale, fp4=True,
        block_rows=1, out=out, backend=backend)
    return _torch_gemm(a, sa, b, sb, out, True, rows) if backend == "torch" else _launch(a, sa, b, sb, out, True, rows)


def fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, *, block_rows=32, backend="auto", out=None):
    """Block-32 E4M3 GEMM with a square or row-wise original E8M0 grid.

    Weight scales may be raw E8M0 bytes or their exact FP32 decoded powers.
    This does not requantize operands or use expanded BF16 weights as a proxy
    for source codes. It implements the same explicit reduction as the CPU
    oracle, not a claim about an unexecuted native TileLang kernel.
    """
    a, sa, b, sb, out, backend, rows = _prepare(
        a_codes, a_scale, b_codes, b_scale, fp4=False,
        block_rows=block_rows, out=out, backend=backend)
    return _torch_gemm(a, sa, b, sb, out, False, rows) if backend == "torch" else _launch(a, sa, b, sb, out, False, rows)
