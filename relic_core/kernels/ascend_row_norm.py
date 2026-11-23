"""Row normalization matching ``RMSNorm.forward`` exactly, with a fused Ascend path.

The reference is four lines -- cast to FP32, scale by ``rsqrt(square().mean(-1) + eps)``, multiply by
the weight, cast back -- and it costs eight dispatches for a 5120-wide row. This module offers the
same result through two kernels, keeping the rsqrt in the framework because a kernel rsqrt does not
reproduce it on these inputs (it differs in 7 rows of 40, enough to change the last BF16 bit).

:func:`row_norm_reference` is the eager expression itself, so the fused path has an oracle rather than
a restatement of itself; :func:`fused_row_norm` is the fused path. Which one a caller gets is explicit
through ``backend``, never inferred from the device.
"""
from __future__ import annotations

import torch

from . import _ascend_row_norm_kernels as _kernels

_BACKENDS = ("triton", "torch")
_DTYPES = (torch.bfloat16, torch.float16)


def row_norm_reference(x, weight, eps, *, weighted=True):
    """The eager chain, transcribed from ``RMSNorm.forward``: the oracle, not an approximation."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + float(eps))
    if weighted:
        value = weight * value
    return value.to(dtype)


def _checked(x, weight, eps, *, backend, weighted):
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of %r, got %r" % (_BACKENDS, backend))
    if not torch.is_tensor(x) or not torch.is_tensor(weight):
        raise TypeError("x and weight have to be tensors")
    if x.dim() < 2:
        raise ValueError("the fused row normalization needs a leading row axis, got %dd" % x.dim())
    if x.dtype not in _DTYPES:
        raise TypeError("x has to be bfloat16 or float16, got %s" % (x.dtype,))
    if weight.dtype != torch.float32:
        raise TypeError("the weight is kept FP32 by the reference, got %s" % (weight.dtype,))
    if weight.dim() != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("the weight has to be a vector of the row width, got %r for width %d"
                         % (tuple(weight.shape), x.shape[-1]))
    if not weight.is_contiguous():
        raise ValueError("the fused tail reads a contiguous weight")
    return int(x.shape[-1])


def fused_row_norm(x, weight, eps, *, backend="triton", weighted=True):
    """The row normalization through the fused kernels, or through the eager chain on ``torch``."""
    dim = _checked(x, weight, eps, backend=backend, weighted=weighted)
    if backend == "torch":
        return row_norm_reference(x, weight, eps, weighted=weighted)
    if not _kernels.triton_available():
        # Explicit rather than a silent fallback: a caller that asked for the fused path on a host
        # without Triton gets told, instead of measuring the eager chain it thought it replaced.
        raise RuntimeError("the fused row normalization needs Triton on this host; ask for "
                           "backend='torch' to take the eager chain deliberately")
    rows = x.numel() // dim
    flat = x.reshape(rows, dim)
    # The eager chain reads `x.float()`; the kernels read the stored dtype and widen in-register,
    # which is the same value, but the row sum has to be over the widened values.
    sums = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _kernels.row_sqsum(flat.contiguous(), sums, dim=dim)
    # The mean is the sum times the reciprocal, not the sum divided by the length: only the
    # multiplication reproduces `torch.mean` on these inputs.
    means = sums * (1.0 / dim)
    rstd = torch.rsqrt(means + float(eps))
    out = torch.empty((rows, dim), dtype=x.dtype, device=x.device)
    _kernels.scale(flat.contiguous(), weight, rstd, out, dim=dim, weighted=weighted)
    return out.view(x.shape)