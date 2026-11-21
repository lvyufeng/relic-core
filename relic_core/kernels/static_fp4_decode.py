"""Exact original FP4 byte widening with static nibble decoding, opt-in only."""
from __future__ import annotations

import math
import torch

from . import unpack
from .ascend_unpack import _npu_launch

_REFERENCE_UNPACK = unpack.unpack_fp4_to_int8
_PAIR_LUT = {}
_X2 = (0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12)
_PAIR_BITS = torch.tensor([(_X2[code & 15] & 255) | ((_X2[code >> 4] & 255) << 8)
                           for code in range(256)],dtype=torch.int32).to(torch.uint16)


def unpack_fp4_static(packed,*,backend="auto",out=None,tile=2048):
    """Same INT8-times-two bits as the frozen widening; no new quantization."""
    raw=unpack._byte_input(packed,"packed",encoding="fp4",min_rank=1)
    if isinstance(tile,bool) or not isinstance(tile,int) or tile not in {512,1024,2048,4096}:
        raise ValueError("static FP4 widening tiles must be 512,1024,2048 or4096")
    shape=(*raw.shape[:-1],raw.shape[-1]*2)
    result=unpack._output(out,shape,torch.int8,raw.device,(packed,))
    if raw.numel()==0:return result
    implementation=unpack._select_backend(backend,raw.device)
    if implementation is None:
        return _REFERENCE_UNPACK(packed,backend="torch",out=result)
    from . import _ascend_static_fp4_decode_kernels as kernels
    with _npu_launch(raw):
        kernels.fp4_static_decode_kernel[(min(math.ceil(raw.numel()/tile),65535),)](
            raw,result,raw.numel(),tile,((raw.numel()+tile-1)//tile)*tile*2>=2**31)
    return result


def unpack_fp4_pairs(packed,*,backend="auto",out=None,tile=2048,programs=None):
    """One 16-bit pair store per original byte; identical little-endian I8x2 bits."""
    raw=unpack._byte_input(packed,"packed",encoding="fp4",min_rank=1)
    if isinstance(tile,bool) or not isinstance(tile,int) or tile not in {512,1024,2048,4096,8192}:
        raise ValueError("pair widening tiles must be powers of two from512 through8192")
    result=unpack._output(out,(*raw.shape[:-1],raw.shape[-1]*2),torch.int8,raw.device,(packed,))
    if raw.numel()==0:return result
    implementation=unpack._select_backend(backend,raw.device)
    if implementation is None:
        return _REFERENCE_UNPACK(packed,backend="torch",out=result)
    from . import _ascend_static_fp4_decode_kernels as kernels
    key=str(raw.device)
    if key not in _PAIR_LUT:
        _PAIR_LUT[key]=_PAIR_BITS.to(raw.device)
    grid = min(math.ceil(raw.numel()/tile),65535)
    if programs is not None:
        if isinstance(programs,bool) or not isinstance(programs,int) or not 1<=programs<=65535:
            raise ValueError("programs must be a positive bounded launch grid")
        grid = min(grid,programs)
    with _npu_launch(raw):
        kernels.fp4_pair_decode_kernel[(grid,)](
            raw,_PAIR_LUT[key],result.view(torch.uint16),raw.numel(),tile,
            ((raw.numel()+tile-1)//tile)*tile*2>=2**31)
    return result
