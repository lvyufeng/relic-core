"""Pair rotation matching ``apply_rotary_emb_pairs`` exactly, with a fused Ascend path.

The reference rotates adjacent elements of ``x`` as one complex number, computes in FP32 and writes
the narrowed result back into ``x``'s own storage. This module offers the same rotation through one
kernel writing into the same storage, so the copy the eager path needs disappears without the values
moving.

:func:`rope_pairs_reference` is the eager expression itself and is the oracle for
:func:`fused_rope_pairs`. Both mutate ``x`` in place and return it, which is the contract the model's
call sites rely on.
"""
from __future__ import annotations

import torch

from . import _ascend_rope_pairs_kernels as _kernels

_BACKENDS = ("triton", "torch")
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def rope_pairs_reference(x, freqs, *, inverse=False):
    """The eager pair rotation, transcribed: FP32 products, one narrowing, written back in place."""
    dtype = x.dtype
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    cosine = freqs[..., 0].float()
    sine = freqs[..., 1].float()
    if inverse:
        sine = -sine
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    x[..., 0::2] = rotated_even.to(dtype)
    x[..., 1::2] = rotated_odd.to(dtype)
    return x


def _checked(x, freqs, *, backend):
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of %r, got %r" % (_BACKENDS, backend))
    if not torch.is_tensor(x) or not torch.is_tensor(freqs):
        raise TypeError("x and freqs have to be tensors")
    if x.dtype not in _DTYPES:
        raise TypeError("unsupported rotation dtype %s" % (x.dtype,))
    if x.dim() < 2 or x.shape[-1] % 2:
        raise ValueError("the rotated width has to be even and lead by at least one axis, got %r"
                         % (tuple(x.shape),))
    if freqs.dim() != 3 or freqs.shape[-1] != 2:
        raise ValueError("freqs has to be [1, pairs, 2], got %r" % (tuple(freqs.shape),))
    if freqs.shape[0] != 1:
        raise ValueError("the fused form covers the single-position decode table only, got %r"
                         % (tuple(freqs.shape),))
    pairs = x.shape[-1] // 2
    if freqs.shape[1] != pairs:
        raise ValueError("the table has to carry a cosine and a sine for every pair: %d rows for %d "
                         "pairs" % (freqs.shape[1], pairs))
    if x.stride(-1) != 1:
        raise ValueError("the rotated axis has to be contiguous to read adjacent pairs")
    return pairs


def fused_rope_pairs(x, freqs, *, inverse=False, backend="triton"):
    """Rotate ``x`` in place through the fused kernel, or through the eager chain on ``torch``."""
    pairs = _checked(x, freqs, backend=backend)
    if backend == "torch":
        return rope_pairs_reference(x, freqs, inverse=inverse)
    if not _kernels.triton_available():
        raise RuntimeError("the fused pair rotation needs Triton on this host; ask for "
                           "backend='torch' to take the eager chain deliberately")
    shape = x.shape
    if x.is_contiguous():
        row_stride = shape[-1]
    else:
        # The model rotates `q[..., -rope:]` slices, which are not contiguous: the leading extents
        # have to be regularly spaced for one row stride to describe them.
        row_stride = x.stride(-2)
        for axis in range(x.dim() - 2):
            if x.stride(axis) != x.stride(axis + 1) * shape[axis + 1]:
                raise ValueError("the leading extents are not regularly spaced for a row stride")
    cosine = freqs[0, :, 0].contiguous()
    sine = freqs[0, :, 1].contiguous()
    return _kernels.rotate_in_place(x, cosine, sine, row_stride=row_stride,
                                    inverse=inverse, pairs=pairs)