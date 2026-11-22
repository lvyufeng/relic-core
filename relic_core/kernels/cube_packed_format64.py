"""Byte-exact, padding-free N64/K128 Cube order for one original FP4 projection.

This is a distinct physical format, not an alias for the N32 layout. Conversion
and recovery operate on CPU byte tensors only; code nibbles and all E8M0 scale
encodings remain uninterpreted. No device allocation, quantization or kernel is
performed, and every materialized payload has exactly the original byte count.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cube_packed_format import (
    _checked_count, _checked_codes, _checked_scales, _checked_payload_tensor,
    _check_no_overlap, _require_device, _resolve_output_device, _torch,
)

__all__ = ["FORMAT", "CubeGeometry64", "CubePackedFP464", "cube_geometry64",
           "convert_fp4_cube64", "recover_fp4_cube64"]

FORMAT = "relic.fp4.cube-n64-k128.v1"
CUBE_ROWS, CUBE_K, SCALE_K = 64, 128, 32
_CODE_PERMUTE = (0, 3, 4, 1, 2, 5)
_SCALE_PERMUTE = (0, 2, 3, 1)
_SCALE_INVERSE = (0, 3, 1, 2)


@dataclass(frozen=True)
class CubeGeometry64:
    """Complete N64/K128 extents and a same-size original byte budget."""
    n: int
    k: int

    def __post_init__(self):
        n, k = _checked_count(self.n, "n"), _checked_count(self.k, "k")
        if n % 64 or k % 128:
            raise ValueError("N64 Cube format requires complete N64/K128 geometry; no padding is permitted")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)

    @property
    def source_codes_shape(self):
        return (self.n, self.k // 2)

    @property
    def source_scales_shape(self):
        return (self.n, self.k // 32)

    @property
    def codes_view_shape(self):
        return (self.n // 64, 4, 16, self.k // 128, 8, 8)

    @property
    def scales_view_shape(self):
        return (self.n // 64, 64, self.k // 128, 4)

    @property
    def codes_shape(self):
        return (self.n // 64, self.k // 128, 8, 4, 16, 8)

    @property
    def scales_shape(self):
        return (self.n // 64, self.k // 128, 4, 64)

    @property
    def codes_bytes(self):
        return self.n * self.k // 2

    @property
    def scales_bytes(self):
        return self.n * self.k // 32

    @property
    def payload_bytes(self):
        return self.codes_bytes + self.scales_bytes

    @property
    def source_bytes(self):
        return self.payload_bytes

    @property
    def padding_bytes(self):
        return 0

    def report(self):
        return {"format": FORMAT, "n": self.n, "k": self.k, "cube_rows": 64,
                "cube_k": 128, "scale_k": 32,
                "source_codes_shape": list(self.source_codes_shape),
                "source_scales_shape": list(self.source_scales_shape),
                "codes_shape": list(self.codes_shape), "scales_shape": list(self.scales_shape),
                "codes_bytes": self.codes_bytes, "scales_bytes": self.scales_bytes,
                "source_bytes": self.source_bytes, "payload_bytes": self.payload_bytes,
                "padding_bytes": 0, "size_exact": True}


def cube_geometry64(n, k):
    """Validate one projection without importing the tensor library."""
    return CubeGeometry64(n, k)


@dataclass(frozen=True)
class CubePackedFP464:
    """Typed CPU byte payload; N32 payloads cannot be silently reinterpreted."""
    codes: "torch.Tensor"
    scales: "torch.Tensor"
    geometry: CubeGeometry64
    source_codes_dtype: str = "uint8"
    format: str = FORMAT

    def __post_init__(self):
        if self.format != FORMAT:
            raise ValueError("N64 payload must declare the distinct N64 Cube format")
        if not isinstance(self.geometry, CubeGeometry64):
            raise TypeError("N64 payload requires CubeGeometry64")
        if self.source_codes_dtype not in ("uint8", "int8"):
            raise ValueError("source_codes_dtype must retain the uint8/int8 byte spelling")
        _checked_payload_tensor(self.codes, "N64 codes", self.geometry.codes_shape)
        _checked_payload_tensor(self.scales, "N64 scales", self.geometry.scales_shape)
        _check_no_overlap(self.codes, self.scales, "N64 codes", "N64 scales")

    @property
    def n(self):
        return self.geometry.n

    @property
    def k(self):
        return self.geometry.k

    @property
    def payload_bytes(self):
        return self.geometry.payload_bytes

    def report(self):
        return {**self.geometry.report(), "source_codes_dtype": self.source_codes_dtype}


def convert_fp4_cube64(codes, scales, *, device=None):
    """Permute one original CPU projection; neither nibble nor scale is decoded."""
    torch = _torch()
    _resolve_output_device(device)
    raw_codes, raw_scales = _checked_codes(codes), _checked_scales(scales)
    _require_device(raw_codes, "N64 source codes", ("cpu",))
    _require_device(raw_scales, "N64 source scales", ("cpu",))
    geometry = cube_geometry64(raw_codes.shape[0], raw_codes.shape[1] * 2)
    if tuple(raw_scales.shape) != geometry.source_scales_shape:
        raise ValueError("N64 scales must retain the complete original [N,K/32] extent")
    result = CubePackedFP464(
        raw_codes.view(geometry.codes_view_shape).permute(_CODE_PERMUTE).contiguous(),
        raw_scales.view(geometry.scales_view_shape).permute(_SCALE_PERMUTE).contiguous(),
        geometry, source_codes_dtype="int8" if codes.dtype == torch.int8 else "uint8")
    for tensor in (result.codes, result.scales):
        for source in (codes, scales):
            _check_no_overlap(tensor, source, "N64 payload", "source")
    return result


def recover_fp4_cube64(payload, *, codes_dtype=None):
    """Recover original CPU bytes into fresh storage, including signed zeros."""
    torch = _torch()
    if not isinstance(payload, CubePackedFP464):
        raise TypeError("N64 recovery requires a typed CubePackedFP464 payload")
    payload.__post_init__()
    geometry = payload.geometry
    codes = payload.codes.permute(_CODE_PERMUTE).contiguous().view(geometry.source_codes_shape)
    scales = payload.scales.permute(_SCALE_INVERSE).contiguous().view(geometry.source_scales_shape)
    requested = payload.source_codes_dtype if codes_dtype is None else str(codes_dtype).replace("torch.", "")
    if requested not in ("uint8", "int8"):
        raise ValueError("N64 recovery codes_dtype must be uint8 or int8")
    if requested == "int8":
        codes = codes.view(torch.int8)
    for tensor in (codes, scales):
        for source in (payload.codes, payload.scales):
            _check_no_overlap(tensor, source, "recovered bytes", "N64 payload")
    return codes, scales
