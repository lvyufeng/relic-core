"""Mask-then-sort selection for the sparse indexer, with an exact CPU oracle.

The reference selection builds a small int array, masks the entries the query
cannot reach, sorts it ascending and casts to int32. On the Ascend build each of
those steps becomes an AI CPU operator, and an AI CPU operator occupies no cube or
vector unit while serialising against the stream; measured on the registered decode
lane it is 31 operators a forward, 2762 us, 3.5% of the step, almost all of it
fixed dispatch cost rather than work.

:mod:`._ascend_indexer_masked_sort_kernels` replaces the sort with a rank. This
module is the callable surface: the ``triton`` backend for the fused selection, and
an explicit ``torch`` backend that computes the reference's answer directly, so the
fused path has a bitwise oracle on CPU and on the device alike.

Two padding conventions exist because the two call sites sort against different
sentinels:

* :func:`masked_sort_trailing` — a sentinel past every real position, so the
  padding trails the real ids. Matches
  ``where(values > -inf, keys + offset, tail).sort().values`` followed by
  ``tail -> -1``.
* :func:`masked_sort_leading` — the ``-1`` sentinel sorts below every real block
  id, so the padding leads them. Matches
  ``where(values > -inf, keys, -1).sort().values.int()``.

Both return int32 with ``-1`` as the padding value. The arithmetic is integer
throughout: no width change and no rounding can alter the result, which is why
bitwise parity here is a property of the algorithm rather than a measurement.
"""
from __future__ import annotations

import torch

from . import _ascend_indexer_masked_sort_kernels as _kernels

#: Padding value in the returned table, matching the reference's ``-1``.
PADDING = -1
_BACKENDS = ("triton", "torch")


def _checked(values, keys, *, backend):
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of %r, got %r" % (_BACKENDS, backend))
    if not torch.is_tensor(values) or not torch.is_tensor(keys):
        raise TypeError("values and keys have to be tensors")
    if values.shape != keys.shape:
        raise ValueError("values and keys have to have the same shape, got %r and %r"
                         % (tuple(values.shape), tuple(keys.shape)))
    if values.dim() < 1:
        raise ValueError("the selection needs at least a width axis")
    if values.is_floating_point() is False:
        raise TypeError("values have to be floating point: reachability is 'value > -inf'")
    if keys.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                          torch.uint8):
        raise TypeError("keys have to be integers, got %s" % (keys.dtype,))
    if values.shape[-1] < 1:
        raise ValueError("the width has to be at least one")
    return values.shape[-1]


def _oracle_trailing(values, keys, offset):
    """The reference's answer for a trailing sentinel, without the sentinel itself."""
    ident = (keys + int(offset)).to(torch.int32)
    reachable = values > float("-inf")
    # The reference sorts ``where(reachable, ident, tail)`` with ``tail`` above every
    # real id, then maps the sentinels to -1. Sorting with a maximal sentinel and
    # replacing the positions no reachable entry claimed is the same table.
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.sort(torch.where(reachable, ident, sentinel), dim=-1).values
    count = reachable.sum(dim=-1, keepdim=True)
    lanes = torch.arange(ordered.shape[-1], device=ordered.device).expand_as(ordered)
    return torch.where(lanes < count, ordered, torch.full_like(ordered, PADDING))


def _oracle_leading(values, keys):
    """The reference's answer for the ``-1`` sentinel, which sorts below every real id."""
    reachable = values > float("-inf")
    if reachable.any() and bool((keys[reachable] < 0).any()):
        # The fused form ranks the padding ahead of every real id, which is only the
        # reference's answer while the ``-1`` sentinel sorts below real ids.
        raise ValueError("masked_sort_leading needs reachable keys to be non-negative")
    ordered = torch.sort(torch.where(reachable, keys, torch.full_like(keys, PADDING)),
                         dim=-1).values
    return ordered.to(torch.int32)


def masked_sort_trailing(values, keys, offset, *, backend="triton"):
    """Reachable ids ascending as int32 with ``-1`` padding after them.

    ``keys`` are added to ``offset`` first, exactly as the reference call site does,
    so the result is a drop-in for that expression's ``values``.
    """
    _checked(values, keys, backend=backend)
    if backend == "torch" or not _kernels.triton_available():
        return _oracle_trailing(values, keys, offset)
    out = torch.full(values.shape, PADDING, dtype=torch.int32, device=values.device)
    return _kernels.rank_scatter(values, keys, out, offset=offset, pad_first=False)


def masked_sort_leading(values, keys, *, backend="triton"):
    """``-1`` padding first, then reachable block ids ascending, as int32."""
    _checked(values, keys, backend=backend)
    if backend == "torch" or not _kernels.triton_available():
        return _oracle_leading(values, keys)
    out = torch.full(values.shape, PADDING, dtype=torch.int32, device=values.device)
    return _kernels.rank_scatter(values, keys, out, offset=0, pad_first=True)


def masked_sort_trailing_with_tail(values, keys_plus_offset, width, tail, *, backend="triton"):
    """The exact call-site form: ids already offset and the sentinel already named.

    ``width`` has to agree with the table's own width and ``tail`` has to be above
    every real id, because that is what makes the sentinel trail the real ids; both
    are checked rather than assumed, since this is the drop-in for one expression
    and a silent mismatch would move real ids to the padding positions.
    """
    if int(width) != int(values.shape[-1]):
        raise ValueError("width %r does not match the table width %r"
                         % (width, values.shape[-1]))
    ident = keys_plus_offset.to(torch.int64)
    reachable = values > float("-inf")
    if reachable.any() and bool((ident[reachable] >= int(tail)).any()):
        raise ValueError("tail %r has to be above every reachable id, as the reference requires"
                         % (tail,))
    if backend == "torch" or not _kernels.triton_available():
        # The ids are already offset, so this path passes an offset of zero; `tail` is
        # only a bound here, because the sentinel positions become -1 either way.
        return _oracle_trailing(values, keys_plus_offset, 0)
    out = torch.full(values.shape, PADDING, dtype=torch.int32, device=values.device)
    return _kernels.rank_scatter(values, keys_plus_offset, out, offset=0, pad_first=False)
    out = torch.full(values.shape, PADDING, dtype=torch.int32, device=values.device)
    return _kernels.rank_scatter(values, keys_plus_offset, out, offset=0, pad_first=False)