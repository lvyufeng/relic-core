"""Fused FP32 Sinkhorn normalizations; sigmoid and softmax stay on the original path.

This removes repeated tiny normalization launches without changing input values,
softmax/sigmoid implementations, iteration count or FP32 storage. The four-term
reduction order is the kernel's native order and is verified separately on NPU.
"""
from __future__ import annotations

import math
import torch

from .ascend_unpack import _npu_launch
from . import ops


def sinkhorn_normalize(comb,*,iterations=20,eps=1e-6,backend="auto"):
    if comb.dtype!=torch.float32 or not comb.is_contiguous() or comb.shape[-2:]!=(4,4):
        raise ValueError("Sinkhorn normalization requires contiguous FP32[...,4,4]")
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1 or iterations>64:
        raise ValueError("iterations must be an integer from1 through64")
    if not math.isfinite(eps) or eps<=0:raise ValueError("epsilon must be finite and positive")
    if backend not in {"auto","torch","triton"}:raise ValueError("unsupported Sinkhorn backend")
    if backend=="torch" or (backend=="auto" and comb.device.type=="cpu"):
        result=comb/(comb.sum(dim=-2,keepdim=True)+eps)
        for _ in range(iterations-1):
            result=result/(result.sum(dim=-1,keepdim=True)+eps)
            result=result/(result.sum(dim=-2,keepdim=True)+eps)
        return result
    if comb.device.type!="npu":raise ValueError("the fused normalization requires an NPU")
    from . import _ascend_hc_sinkhorn_kernels as kernels
    result=torch.empty_like(comb)
    rows=comb.numel()//16
    with _npu_launch(comb):
        kernels.sinkhorn_normalize_kernel[(min(rows,65535),)](comb,result,rows,iterations,eps)
    return result


def hc_split_sinkhorn_normalized(mixes,hc_scale,hc_base,hc_mult=4,sinkhorn_iters=20,eps=1e-6):
    """Original affine/sigmoid/softmax, then one native FP32 normalization kernel."""
    if mixes.device.type!="npu" or hc_mult!=4:
        return ops.hc_split_sinkhorn_torch(mixes,hc_scale,hc_base,hc_mult,sinkhorn_iters,eps)
    flat=mixes.view(-1,24).float()
    scale,base=hc_scale.float(),hc_base.float()
    pre=torch.sigmoid(flat[:,:4]*scale[0]+base[:4])+eps
    post=2*torch.sigmoid(flat[:,4:8]*scale[1]+base[4:8])
    comb=(flat[:,8:]*scale[2]+base[8:]).reshape(-1,4,4)
    initial=torch.softmax(comb,dim=-1)+eps
    final=sinkhorn_normalize(initial,iterations=max(1,sinkhorn_iters),eps=eps,backend="triton")
    shape=mixes.shape[:-1]
    return pre.view(*shape,4),post.view(*shape,4),final.view(*shape,4,4)
