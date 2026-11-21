"""Native Cube reduction with exactly reconstructed FP32 weights, opt-in only.

Three BF16 residual planes preserve each original FP32 weight value exactly.
BF16 activation values are unchanged. Summing the three native FP32 dot products
has a different reduction from a single FP32 GEMV; it is a numerical experiment,
not a precision downgrade or the exact reference path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .ascend_unpack import _npu_launch
from .native_cube_block_gemv import _output,_fused

CONTRACT="native_cube_exact_FP32_weight_planes_changed_reduction"


@dataclass
class ExactFP32Planes:
    planes: tuple
    rows: int
    columns: int
    original_dtype: str="torch.float32"
    reconstructs_original_bits: bool=True


def prepare_exact_fp32_planes(weight):
    if weight.dtype!=torch.float32 or weight.ndim!=2 or not weight.is_contiguous():
        raise ValueError("exact weight planes require a contiguous FP32 matrix")
    original=weight.detach().cpu()
    residual=original.clone()
    parts=[]
    for _ in range(3):
        part=residual.to(torch.bfloat16)
        parts.append(part)
        residual=residual-part.float()
    reconstructed=parts[0].float()+parts[1].float()+parts[2].float()
    if not bool(torch.isfinite(original).all()) or not torch.equal(reconstructed.view(torch.int32),original.view(torch.int32)):
        raise ValueError("three BF16 planes do not exactly reconstruct this original FP32 weight")
    if any(bool(((part.float().abs()<torch.finfo(torch.bfloat16).tiny)&(part!=0)).any()) for part in parts):
        raise ValueError("exact weight planes contain BF16 subnormals that could be flushed")
    n,k=original.shape
    if not 1<=n<=32 or k>65535:
        raise ValueError("this exact-plane prototype is scoped to N1..32 and K<=65535")
    padded=math.ceil(n/32)*32
    transposed=[]
    for part in parts:
        result=torch.zeros((k,padded),dtype=torch.bfloat16)
        result[:,:n]=part.t()
        transposed.append(result.to(weight.device))
    return ExactFP32Planes(tuple(transposed),n,k)


def exact_weight_plane_gemv(x,weight,*,backend="auto",out=None,block_k=256):
    if x.dtype!=torch.bfloat16 or x.ndim!=2 or x.shape!=(1,weight.columns) or not x.is_contiguous():
        raise ValueError("the exact-plane Cube projection needs one contiguous BF16 activation row")
    if block_k not in {128,256} or weight.columns%block_k:
        raise ValueError("K must be divisible by a supported128/256 Cube tile")
    if any(plane.device!=x.device for plane in weight.planes):raise ValueError("all exact planes must be on the activation device")
    result=_output(out,(1,weight.rows),x.device,(x,*weight.planes))
    if not _fused(x,backend):
        source=sum(plane[:,:weight.rows].float() for plane in weight.planes)
        result.copy_(x.float()@source)
        return result
    from . import _ascend_exact_weight_plane_kernels as kernels
    with _npu_launch(x):
        kernels.exact_weight_plane_kernel[(math.ceil(weight.rows/32),)](
            x,*weight.planes,result,weight.rows,weight.columns,weight.planes[0].shape[1],block_k)
    return result
