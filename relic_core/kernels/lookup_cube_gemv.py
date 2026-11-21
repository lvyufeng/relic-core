"""Opt-in native Cube GEMV using exact, pre-certified scaled BF16 lookup bits.

This is a changed full-K accumulation contract, not the reference window rule.
Original FP8 activation bytes/E8M0 scales and packed FP4 bytes/E8M0 scales are
read directly. Tiny lookup tables replace wide per-lane integer exponent walks;
unsafe nonzero pairs become NaN, never rounded BF16 or a finite substitute.
"""
from __future__ import annotations

import math
import threading

import torch

from . import native_cube_block_gemv as native
from .ascend_unpack import _npu_launch

CONTRACT = "native_cube_lookup_scaled_bf16_full_K"
_TABLES = {}
_LOCK = threading.Lock()


def _joint_bits(values, safe):
    result = torch.empty((len(values),256),dtype=torch.int16)
    for code, unit in enumerate(values):
        for scale in range(256):
            if bool(safe[code,scale]):
                value = math.ldexp(unit,scale-127)
                result[code,scale] = torch.tensor(value,dtype=torch.bfloat16).view(torch.int16)
            else:
                result[code,scale] = 0x7fc0
    return result.contiguous().view(torch.uint16)


_E4_BITS = _joint_bits(native._E4M3_VALUES,native._E4_SAFE)
_E2_VALUES = tuple(float(value)*0.5 for value in native._E2M1_X2)
_E2_BITS = _joint_bits(_E2_VALUES,native._E2M1_SAFE)


def _tables(device):
    key=str(device)
    with _LOCK:
        if key not in _TABLES:
            _TABLES[key] = (_E4_BITS.to(device),_E2_BITS.to(device))
        return _TABLES[key]


def fp4_lookup_cube_gemv(a_codes,a_scale,b_packed,b_e8m0,*,backend="auto",block_n=32,block_k=128,out=None):
    """One row, same raw four-operand provider seam, exact input values only.

    Different full-K FP32 native accumulation and pre-matmul scale placement are
    intentional. Output is FP32. The caller must separately assess quality/TPS.
    """
    if block_n not in {16,32,64} or block_k not in {32,64,128}:
        raise ValueError("native lookup Cube tiles must be N16/32/64 and K32/64/128")
    a=native._row_codes(a_codes,"activation codes")
    k=a.numel()
    sa=native._row_scales(a_scale,"activation scales",k)
    b=native._bytes(b_packed,"packed FP4 weight")
    sb=native._bytes(b_e8m0,"original E8M0 grid",scale=True)
    if b.ndim!=2 or b.shape[1]*2!=k or k%block_k:
        raise ValueError("packed weight K must match activation and be divisible by block_k")
    n=b.shape[0]
    if sb.shape!=(n,k//32):
        raise ValueError("weight scale grid must be [N,K//32]")
    if not (a.device==sa.device==b.device==sb.device):
        raise ValueError("all four original operands must be on one device")
    result=native._output(out,(1,n),a.device,(a_codes,a_scale,b_packed,b_e8m0))
    if n==0 or k==0:return result
    if not native._fused(a,backend):
        activation = _E4_BITS[a.long(),sa.long().repeat_interleave(32)].view(torch.bfloat16).float()
        low, high = b & 15, b >> 4
        codes = torch.stack((low,high),dim=-1).reshape(n,k)
        weights = _E2_BITS[codes.long(),sb.long().repeat_interleave(32,dim=1)].view(torch.bfloat16).float()
        result.copy_(activation.unsqueeze(0) @ weights.t())
        return result
    from . import _ascend_lookup_cube_kernels as kernels
    at,bt=_tables(a.device)
    with _npu_launch(a):
        kernels.fp4_lookup_cube_kernel[(min(math.ceil(n/block_n),65535),)](
            a,sa,b,sb,at,bt,result,n,k,block_n,block_k,native._index64(n,k,True))
    return result
