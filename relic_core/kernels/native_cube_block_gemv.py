"""Native Ascend Cube block-32 GEMV prototype with an explicitly different reduction.

This is **opt-in and unbound**: nothing in the runtime, the loader or a serving path
calls it, and the accepted operators (``block_gemm``, ``decode_block_gemv``) keep
their arithmetic and their defaults. It exists to measure one question — what a
single M=1 decode column costs when the K-32 partial dot product is a native
BF16xBF16->FP32 Cube instruction instead of the accepted vector emulation — and to
let the model owner compare the resulting logits against the accepted rule.

The released checkpoint's operands are unchanged: the activation is the original
E4M3 code bytes plus original E8M0 block-32 scale bytes, and the weight is either
the original packed E2M1 nibbles plus original E8M0 bytes, or the exact
``INT8 = 2 * E2M1`` values plus the original FP32 half-scale grid (what online
unpack produces), or E4M3 code bytes plus a raw/decoded E8M0 grid. Nothing is
requantized: no W8A8/W4A8 activation integers, no new weight precision and no
checkpoint rewrite.

Two accumulation rules are provided, and **both differ from the accepted rule**.
The accepted reference cuts each K-32 block into two 16-term halves, anchors each
half on a nominal exponent, truncates every product onto a window-13 grid and then
applies the two FP32 E8M0 rescales. Neither kernel here does any of that:

``mode="block32"`` (default)
    One native Cube dot per K-32 block sums all 32 products in the hardware's own
    FP32 accumulation order; the 15 unused M rows are zeros. The separate FP32
    ``SA`` then ``SB`` rescales and the sequential K-block addition are the frozen
    accepted helpers in the accepted order.

``mode="scaled_bf16"`` (requires an explicit certification)
    The original scale is merged into each unit value as an exact BF16 exponent
    adjustment *before* one K-major Cube dot, so the scales are part of the operands
    and there is no post-dot rescale. This changes the scaling order as well as the
    reduction: it is a different numerical contract, authorised for testing by the
    human owner explicitly, not a drop-in replacement. It is refused unless the
    caller certifies every ``code x scale`` pair with
    :func:`scaled_bf16_weight_representable` / :func:`scaled_bf16_activation_representable`
    at load time. The certification is a host check, so no device read is added to a
    hot loop; a lane that still turns out unsafe becomes NaN, never a silent value.

Module-level attributes: :data:`NUMERICAL_CONTRACT` names the changed reduction,
:data:`SCALED_BF16_CONTRACT` names the changed scaling order, and
:data:`SUPPORTED_BLOCK_N` / :data:`SUPPORTED_BLOCK_K` are the accepted tiles. There
is no performance claim here: only a measured benchmark on the target build can
price it, and this file is not that measurement.
"""
from __future__ import annotations

import math
import struct

import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend
from .block_gemm import _bytes, _dense

__all__ = [
    "NUMERICAL_CONTRACT",
    "SCALED_BF16_CONTRACT",
    "native_cube_fp4_gemv",
    "native_cube_fp8_gemv",
    "scaled_bf16_activation_representable",
    "scaled_bf16_weight_representable",
]

BLOCK = 32
#: Cube N tiles the kernel compiles.
SUPPORTED_BLOCK_N = (32, 64)
#: K strides for the merged-scale mode, both multiples of the original 32 grid.
SUPPORTED_BLOCK_K = (128, 256)
#: Ascend refuses a launch grid above 65535; the kernels stride the remaining tiles.
MAX_PROGRAMS = 65535
NUMERICAL_CONTRACT = "native_cube_no_reference_window"
SCALED_BF16_CONTRACT = "native_cube_scaled_bf16_pre_mma_scaling"


def _e4m3_value(byte: int) -> float:
    """Exact value of an E4M3FN byte, NaN codes included."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        magnitude = math.nan
    elif exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -magnitude if byte & 128 else magnitude


_E4M3_VALUES = tuple(_e4m3_value(byte) for byte in range(256))
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: INT8 = 2 * E2M1, the widening ``unpack_fp4_to_int8`` produces.
_E2M1_X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _bf16_exact(value: float) -> bool:
    """Whether ``value`` is exactly a normal BF16 value, or a signed zero."""
    if value == 0.0:
        return True
    if not math.isfinite(value):
        return False
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return False
    exponent = (bits >> 23) & 255
    return 1 <= exponent <= 254 and (bits & 0xFFFF) == 0


def _exact_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean ``[code, scale byte]`` tables: is ``unit(code) * 2**(s - 127)`` exact?

    Computed once, vectorised, in exact double arithmetic: every E4M3/E2M1 unit value
    and every scale is a dyadic rational, so the product below is exact and the
    question is only whether FP32 then BF16 carry it. Byte 255 is the E8M0 NaN and is
    never exact, and the E4M3 NaN codes have no finite product. This table *is* the
    merged-scale contract; the kernel turns any lane it refuses into NaN.
    """
    scales = torch.arange(256, dtype=torch.float64)
    e4_unit = torch.tensor(_E4M3_VALUES, dtype=torch.float64)
    table = _safe_table(e4_unit, scales)
    e2_unit = torch.tensor(
        [_E2M1_MAGNITUDES[code & 7] * (-1.0 if code & 8 else 1.0) for code in range(16)],
        dtype=torch.float64)
    return table, _safe_table(e2_unit, scales)


def _safe_table(unit: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``[code, 256]`` bool: the product is an exact BF16 normal value or a true zero."""
    product = unit[:, None] * torch.pow(2.0, scales[None, :] - 127.0)
    narrow = product.float()
    bits = narrow.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    carried = narrow.double() == product
    normal = (exponent >= 1) & (exponent <= 254) & ((bits & 0xFFFF) == 0)
    # Byte 255 is the E8M0 NaN, not an exponent, so no pair with it is exact.
    finite_scale = (torch.arange(256) < 255)[None, :]
    return ((carried & normal) | (product == 0)) & finite_scale


#: Built once from exact arithmetic: ``table[code, scale_byte]`` is the contract.
_E4_SAFE, _E2M1_SAFE = _exact_tables()


def _row_codes(value, name: str) -> torch.Tensor:
    """A one-row code vector ``[K]`` from original typed storage or an integer container."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    allowed = {torch.uint8, torch.int8}
    optional = getattr(torch, "float8_e4m3fn", None)
    if optional is not None:
        allowed.add(optional)
    if value.dtype not in allowed:
        raise TypeError(f"{name} must hold original E4M3 code bytes, got {value.dtype}")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous dense strided tensor")
    if value.requires_grad:
        raise ValueError("native Cube GEMV is an inference operator, not autograd")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.reshape(-1)
    if value.ndim != 1:
        raise ValueError(f"native Cube GEMV is M=1: {name} must be [K] or [1, K]")
    return value.view(torch.uint8)


def _row_scales(value, name: str, k: int) -> torch.Tensor:
    """A ``[K // 32]`` vector of raw E8M0 activation scale bytes."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    allowed = {torch.uint8, torch.int8}
    optional = getattr(torch, "float8_e8m0fnu", None)
    if optional is not None:
        allowed.add(optional)
    if value.dtype not in allowed:
        raise TypeError(f"{name} must hold original E8M0 scale bytes, got {value.dtype}")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous dense strided tensor")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.reshape(-1)
    if value.ndim != 1 or value.shape[0] != k // BLOCK:
        raise ValueError(f"{name} must be [{k // BLOCK}] for an activation of width {k}")
    return value.view(torch.uint8)


def _output(out, shape, device, inputs) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=torch.float32, device=device)
    _dense(out, "output", (torch.float32,))
    if tuple(out.shape) != shape:
        raise ValueError(f"output must have shape {shape}, got {tuple(out.shape)}")
    if out.device != device:
        raise ValueError("output must be on the input device")
    if any(torch._C._overlaps(out, source) for source in inputs):
        raise ValueError("output must not alias or share storage with an input")
    return out


def _fused(a: torch.Tensor, backend: str) -> bool:
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or (backend == "auto" and a.device.type != "npu"):
        if a.device.type == "npu":
            raise ValueError("the CPU natural oracle is CPU-only; use the fused NPU backend")
        return False
    if a.device.type != "npu":
        raise ValueError("the native Cube backend is for NPU tensors only")
    if not supports_triton_ascend():
        # The exact wording is a contract with the runtime adapter, which maps this
        # to its own named "packed decode unavailable" fallback.
        raise RuntimeError("Triton Ascend backend is required for the native Cube GEMV")
    return True


def _index64(n: int, k: int, packed: bool) -> bool:
    """Pointer-index width from the padded reach, decided before any product."""
    return max(n * (k // 2 if packed else k), 2 * k) >= 2**31


def _launch(name: str, a: torch.Tensor, args, n: int, block_n: int) -> None:
    with _npu_launch(a):
        from . import _ascend_native_cube_block_kernels as kernels

        getattr(kernels, name)[(min(math.ceil(n / block_n), MAX_PROGRAMS),)](*args)


def _check(mode, block_n, scaled_bf16_safe) -> None:
    if mode not in {"block32", "scaled_bf16"}:
        raise ValueError("mode must be 'block32' or 'scaled_bf16'")
    if block_n not in SUPPORTED_BLOCK_N:
        raise ValueError(f"block_n must be one of {SUPPORTED_BLOCK_N}")
    if mode == "scaled_bf16":
        if scaled_bf16_safe is not True:
            raise ValueError(
                "mode='scaled_bf16' merges the original scales into the BF16 operands "
                "before the matmul, so it requires scaled_bf16_safe=True, certified at "
                "load time with scaled_bf16_weight_representable() and "
                "scaled_bf16_activation_representable(); an uncertified call is refused "
                "rather than silently rounded"
            )
    elif scaled_bf16_safe:
        raise ValueError("scaled_bf16_safe only applies to mode='scaled_bf16'")


# -- host decoding used by the natural oracle and the merged-scale certification ---


def _e8m0_table() -> torch.Tensor:
    return torch.tensor([math.ldexp(1.0, byte - 127) for byte in range(255)] + [math.nan],
                        dtype=torch.float32)


_E8M0 = _e8m0_table()


def _e8m0_values(raw: torch.Tensor) -> torch.Tensor:
    return _E8M0[raw.long()]


def _half_scale_codes(half: torch.Tensor) -> torch.Tensor:
    """Full E8M0 byte of a stored FP32 half-scale; 255 where it is not original."""
    raw = half.contiguous().view(torch.int32)
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7FFFFF
    special = (raw == 0x00200000) | (raw == 0x00400000)
    valid = special | ((raw >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 253))
    full = torch.where(raw == 0x00200000, 0,
                       torch.where(raw == 0x00400000, 1, exponent + 1))
    return torch.where(valid, full, torch.full_like(full, 255)).to(torch.uint8)


def _float_scale_codes(grid: torch.Tensor) -> torch.Tensor:
    """Full E8M0 byte of an exact FP32 power-of-two grid; 255 when it is not one."""
    raw = grid.contiguous().view(torch.int32)
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7FFFFF
    valid = (raw >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254)
    return torch.where(valid, exponent, torch.full_like(exponent, 255)).to(torch.uint8)


def _unit_values(codes: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_E4M3_VALUES, dtype=torch.float32)[codes.long()]


def _packed_values(codes: torch.Tensor) -> torch.Tensor:
    """Exact E2M1 values of packed nibbles, low nibble first."""
    table = torch.tensor(_E2M1_X2, dtype=torch.float32)
    low, high = (codes & 15).long(), (codes >> 4).long()
    return torch.stack((table[low], table[high]), dim=-1).reshape(codes.shape[0], -1) * 0.5


def _per_row(grid: torch.Tensor, n: int, block_rows: int) -> torch.Tensor:
    return grid.repeat_interleave(block_rows, dim=0)[:n]


def _natural_block32(a_codes, a_scale, values, grid, *, block_rows) -> torch.Tensor:
    """Natural per-K-32 FP32 reduction: sum 32 exact products, then rescale.

    This is the oracle for the changed-accumulation rule: a plain FP32 sum of a
    block's exact products, followed by the accepted separate ``SA`` then ``SB``
    multiplies in K-block order. It is deliberately not the accepted window-13 rule.
    """
    n, k = values.shape
    a_unit = _unit_values(a_codes)
    rows = _e8m0_values(_per_row(grid, n, block_rows))
    out = torch.zeros(n, dtype=torch.float32)
    for block in range(k // BLOCK):
        span = slice(block * BLOCK, (block + 1) * BLOCK)
        partial = torch.sum(a_unit[span][None, :] * values[:, span], dim=-1)
        # Separate rescales: FP32 * SA, then * SB, then add. Never fused.
        out = out + (partial * _e8m0_values(a_scale[block : block + 1])[0]) * rows[:, block]
    return out


def _natural_scaled(a_codes, a_scale, values, grid, *, block_rows) -> torch.Tensor:
    """Natural FP32 dot of the exact scaled operands, in ascending K order."""
    n, k = values.shape
    a_values = _unit_values(a_codes) * _e8m0_values(a_scale).repeat_interleave(BLOCK)
    rows = _e8m0_values(_per_row(grid, n, block_rows)).repeat_interleave(BLOCK, dim=1)
    return torch.sum(values * rows * a_values[None, :], dim=-1)


def native_cube_fp4_gemv(a_codes, a_scale, b, b_scale, *, b_format="packed", mode="block32",
                         block_n=64, block_k=256, backend="auto", out=None,
                         scaled_bf16_safe=None):
    """One decode row against an original FP4 weight store, M=1.

    ``a_codes``/``a_scale`` are the original E4M3 activation codes ``[K]`` and E8M0
    block-32 scale bytes ``[K // 32]``. ``b_format="packed"`` takes the original
    packed E2M1 nibbles ``[N, K // 2]`` (low nibble first) with raw E8M0 bytes
    ``[N, K // 32]``; ``b_format="int8_x2"`` takes the exact widened values
    ``[N, K]`` with the original FP32 half-scale grid ``[N, K // 32]``. Returns FP32
    ``[1, N]``.
    """
    _check(mode, block_n, scaled_bf16_safe)
    codes_a = _row_codes(a_codes, "activation codes")
    k = codes_a.shape[0]
    scales_a = _row_scales(a_scale, "activation scales", k)
    if b_format not in {"packed", "int8_x2"}:
        raise ValueError("b_format must be 'packed' or 'int8_x2'")
    packed = b_format == "packed"
    if packed:
        codes_b = _bytes(b, "packed FP4 weight")
        if codes_b.ndim != 2 or codes_b.shape[1] * 2 != k:
            raise ValueError("packed FP4 weight must be [N, K // 2] with the activation's K")
        scales_b = _bytes(b_scale, "FP4 original scales", scale=True)
    else:
        codes_b = _dense(b, "FP4 INT8-times-two values", (torch.int8,))
        if codes_b.ndim != 2 or codes_b.shape[1] != k:
            raise ValueError("INT8-times-two weight must be [N, K] with the activation's K")
        scales_b = _dense(b_scale, "FP4 original half-scales", (torch.float32,))
    n = codes_b.shape[0]
    if tuple(scales_b.shape) != (n, k // BLOCK):
        raise ValueError(f"weight scales must be [{n}, {k // BLOCK}], got {tuple(scales_b.shape)}")
    if not (codes_b.device == codes_a.device == scales_a.device == scales_b.device):
        raise ValueError("all code and scale tensors must be on the same device")
    result = _output(out, (1, n), codes_a.device, (a_codes, a_scale, b, b_scale))
    if n == 0 or k == 0:
        return result
    if not _fused(codes_a, backend):
        values = _packed_values(codes_b) if packed else codes_b.to(torch.float32) * 0.5
        grid = scales_b if packed else _half_scale_codes(scales_b)
        natural = _natural_scaled if mode == "scaled_bf16" else _natural_block32
        result[0].copy_(natural(codes_a, scales_a, values, grid, block_rows=1))
        return result
    width64 = _index64(n, k, packed)
    if mode == "block32":
        _launch("native_cube_block32_kernel", codes_a,
                (codes_a, scales_a, codes_b, scales_b, result, n, k, 1, True, packed,
                 not packed, block_n, width64), n, block_n)
    else:
        block_k = int(block_k)
        if block_k not in SUPPORTED_BLOCK_K or k % block_k:
            raise ValueError(f"block_k must be one of {SUPPORTED_BLOCK_K} and divide K")
        _launch("native_cube_scaled_bf16_kernel", codes_a,
                (codes_a, scales_a, codes_b, scales_b, result, n, k, 1, True, packed,
                 not packed, block_k, block_n, width64), n, block_n)
    return result


def native_cube_fp8_gemv(a_codes, a_scale, b_codes, b_code_scale, *, block_rows=32,
                         mode="block32", block_n=64, block_k=256, backend="auto", out=None,
                         scaled_bf16_safe=None):
    """One decode row against an E4M3 weight, M=1, square or row-wise E8M0 grid.

    ``b_code_scale`` is raw E8M0 bytes or their exact FP32 decoded powers;
    ``block_rows`` is 32 for the dense square grid and 1 for row-scaled tables.
    Returns FP32 ``[1, N]``.
    """
    _check(mode, block_n, scaled_bf16_safe)
    codes_a = _row_codes(a_codes, "activation codes")
    k = codes_a.shape[0]
    scales_a = _row_scales(a_scale, "activation scales", k)
    codes_b = _bytes(b_codes, "weight codes")
    if codes_b.ndim != 2 or codes_b.shape[1] != k:
        raise ValueError("E4M3 weight codes must be [N, K] with the activation's K")
    n = codes_b.shape[0]
    if block_rows not in {1, 32}:
        raise ValueError("weight block_rows must be 1 or 32")
    if not isinstance(b_code_scale, torch.Tensor):
        raise TypeError("weight scales must be a torch.Tensor")
    raw = b_code_scale.dtype != torch.float32
    scales_b = (_bytes(b_code_scale, "weight scales", scale=True) if raw
                else _dense(b_code_scale, "weight scales", (torch.float32,)))
    expected = (math.ceil(n / block_rows), k // BLOCK)
    if tuple(scales_b.shape) != expected:
        raise ValueError(f"weight scale grid must be {expected}, got {tuple(scales_b.shape)}")
    if not (codes_b.device == codes_a.device == scales_a.device == scales_b.device):
        raise ValueError("all code and scale tensors must be on the same device")
    result = _output(out, (1, n), codes_a.device, (a_codes, a_scale, b_codes, b_code_scale))
    if n == 0 or k == 0:
        return result
    if not _fused(codes_a, backend):
        grid = scales_b if raw else _float_scale_codes(scales_b)
        natural = _natural_scaled if mode == "scaled_bf16" else _natural_block32
        result[0].copy_(natural(codes_a, scales_a, _unit_values(codes_b), grid,
                                block_rows=block_rows))
        return result
    width64 = _index64(n, k, False)
    if mode == "block32":
        _launch("native_cube_block32_kernel", codes_a,
                (codes_a, scales_a, codes_b, scales_b, result, n, k, block_rows, False,
                 False, not raw, block_n, width64), n, block_n)
    else:
        block_k = int(block_k)
        if block_k not in SUPPORTED_BLOCK_K or k % block_k:
            raise ValueError(f"block_k must be one of {SUPPORTED_BLOCK_K} and divide K")
        _launch("native_cube_scaled_bf16_kernel", codes_a,
                (codes_a, scales_a, codes_b, scales_b, result, n, k, block_rows, False,
                 False, not raw, block_k, block_n, width64), n, block_n)
    return result


def _safe_mask(table: torch.Tensor, codes: torch.Tensor, grid: torch.Tensor) -> bool:
    """Every ``(code, scale byte)`` pair in a chunk is on the exact table."""
    flat_codes, flat_grid = codes.reshape(-1).long(), grid.reshape(-1).long()
    if flat_codes.shape != flat_grid.shape:
        raise ValueError("the code and scale grids must cover the same elements")
    for start in range(0, flat_codes.numel(), 1 << 22):
        stop = min(start + (1 << 22), flat_codes.numel())
        if not bool(table[flat_codes[start:stop], flat_grid[start:stop]].all()):
            return False
    return True


def scaled_bf16_weight_representable(b, b_scale, *, fp4, b_format="packed", block_rows=32) -> bool:
    """Whether every original ``weight code x original scale`` pair is exact in BF16.

    One host pass over the store, meant to be called once at load time. The
    merged-scale mode requires it: that mode must not round a product which the
    accepted post-dot rescale keeps in FP32. ``False`` means the caller must use
    ``mode="block32"`` for that store, the per-K-32 proof path.
    """
    if fp4 and b_format == "packed":
        codes = _bytes(b, "packed FP4 weight")
        raw = _bytes(b_scale, "FP4 original scales", scale=True)
        n, k = codes.shape[0], codes.shape[1] * 2
        if codes.shape[1] * 2 % BLOCK or tuple(raw.shape) != (n, k // BLOCK):
            raise ValueError("packed FP4 weight and scale grid disagree")
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            span = raw[start:stop].long().repeat_interleave(BLOCK // 2, dim=1)
            low, high = (codes[start:stop] & 15), (codes[start:stop] >> 4)
            if not _safe_mask(_E2M1_SAFE, low, span) or not _safe_mask(_E2M1_SAFE, high, span):
                return False
        return True
    if fp4:
        widened = _dense(b, "FP4 INT8-times-two values", (torch.int8,)).to(torch.float32)
        half = _dense(b_scale, "FP4 original half-scales", (torch.float32,))
        return _safe_mask(_E2M1_SAFE, _x2_indices(widened), _half_scale_codes(half)
                          .repeat_interleave(BLOCK, dim=1))
    codes = _bytes(b, "weight codes")
    n, k = codes.shape
    grid = (_bytes(b_scale, "weight scales", scale=True) if b_scale.dtype != torch.float32
            else _float_scale_codes(_dense(b_scale, "weight scales", (torch.float32,))))
    if tuple(grid.shape) != (math.ceil(n / block_rows), k // BLOCK):
        raise ValueError("weight scale grid disagrees with the weight shape")
    return _safe_mask(_E4_SAFE, codes, _per_row(grid, n, block_rows).repeat_interleave(BLOCK, dim=1))


def _x2_indices(values: torch.Tensor) -> torch.Tensor:
    """Codebook index 0..7 of exact ``INT8 = 2 * E2M1`` magnitudes; 8 marks a non-code.

    Takes the widened values as stored (``0, 1, 2, 3, 4, 6, 8, 12``), not the halved
    E2M1 values: the index is that position, and a stored value outside the codebook
    is reported as 8 so the caller refuses it instead of accepting a wrong weight.
    """
    magnitude = values.abs().round().to(torch.int64)
    index = torch.full_like(magnitude, 8)
    for position, allowed in enumerate(_E2M1_X2[:8]):
        index = torch.where(magnitude == allowed, torch.full_like(magnitude, position), index)
    return index


def scaled_bf16_activation_representable(a_codes, a_scale) -> bool:
    """Whether every ``activation code x original E8M0 byte`` pair is exact in BF16.

    The activation is one row, so this is cheap enough to call for a real row before
    the merged-scale mode is used; ``False`` means that row must take
    ``mode="block32"``. A zero code is exact under any finite power-of-two scale.
    """
    codes = _row_codes(a_codes, "activation codes")
    scales = _row_scales(a_scale, "activation scales", codes.shape[0])
    return _safe_mask(_E4_SAFE, codes, scales.repeat_interleave(BLOCK))