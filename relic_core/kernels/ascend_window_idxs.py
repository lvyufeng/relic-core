"""Sliding-window top-k index table matching ``get_window_topk_idxs``'s decode case, fused.

The reference builds the decode table with eight launches for 128 int32 values. One kernel produces
the same integers: ``(lane + oldest) % window_size``, then ``-1`` where the slot is ahead of the
position.

Everything here is integer arithmetic over bounded operands, so no width change and no rounding can
alter the result -- bitwise parity is a property of the expression rather than a measurement. It is
still checked: :func:`window_topk_idxs_reference` is the eager expression and the oracle for
:func:`window_topk_idxs`.
"""
from __future__ import annotations

import torch

from . import _ascend_window_idxs_kernels as _kernels

_BACKENDS = ("triton", "torch")


def window_topk_idxs_reference(window_size, slot, row, *, device=None):
    """The eager decode expression: ring modulo, then ``-1`` for slots ahead of the position."""
    lanes = torch.arange(int(window_size), dtype=torch.int64)
    total = lanes + (int(slot) + 1)
    index = torch.where(total >= int(window_size), total - int(window_size), total)
    index = torch.where(index > int(row), torch.full_like(index, -1), index)
    table = index.to(torch.int32).view(1, 1, int(window_size))
    return table if device is None else table.to(device)


def _checked(window_size, slot, row, *, backend):
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of %r, got %r" % (_BACKENDS, backend))
    if int(window_size) < 1:
        raise ValueError("the window has to be at least one slot, got %r" % (window_size,))
    if not 0 <= int(slot) < int(window_size):
        # `slot` is the caller's position reduced into the ring; an out-of-range value would make
        # the single-subtraction modulo in the kernel wrong rather than merely different.
        raise ValueError("slot %r is not inside the ring of %r slots" % (slot, window_size))
    if int(row) < 0:
        raise ValueError("row %r cannot be negative" % (row,))


def window_topk_idxs(window_size, slot, row, *, backend="triton", device=None):
    """The decode table, ``[1, 1, window_size]`` int32, through the fused kernel or the eager chain."""
    _checked(window_size, slot, row, backend=backend)
    if backend == "torch":
        return window_topk_idxs_reference(window_size, slot, row, device=device)
    if not _kernels.triton_available():
        raise RuntimeError("the fused table build needs Triton on this host; ask for "
                           "backend='torch' to take the eager chain deliberately")
    if device is None:
        raise ValueError("the fused table build needs an explicit device to allocate on")
    target = torch.device(device)
    out = torch.empty((int(window_size),), dtype=torch.int32, device=target)
    slot_tensor = torch.tensor([int(slot)], dtype=torch.int32, device=target)
    row_tensor = torch.tensor([int(row)], dtype=torch.int32, device=target)
    _kernels.build_table(out, slot_tensor, row_tensor, window_size=int(window_size))
    return out.view(1, 1, int(window_size))