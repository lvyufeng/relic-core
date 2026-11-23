"""The fused sparse-indexer selection against the reference expressions it replaces.

The oracle is checked against the literal expression each call site runs, so the
fused path is validated against the reference rather than against itself. The
Triton backend is then checked against that oracle on the device, which is what
makes "integer arithmetic, so bitwise parity is a property rather than a
measurement" a checked claim instead of an assertion.

Nothing here needs a checkpoint or a model: the selection is a table over a small
integer array.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relic_core.kernels import _ascend_indexer_masked_sort_kernels as _kernels  # noqa: E402
from relic_core.kernels.ascend_indexer_masked_sort import (  # noqa: E402
    PADDING,
    masked_sort_leading,
    masked_sort_trailing,
    masked_sort_trailing_with_tail,
)


def _reference_trailing(values, keys, offset, tail):
    """`attention.py` site A, transcribed: sort against a sentinel past every real id."""
    idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
    return torch.where(idxs < tail, idxs, -1).int()


def _reference_leading(values, keys):
    """`attention.py` site B, transcribed: the `-1` sentinel sorts below every real id."""
    return torch.where(values > -torch.inf, keys, -1).sort(dim=-1).values.int()


def _cases(seed=0):
    """Reachable-id tables at the widths and reachabilities the indexer actually produces."""
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for width, offset, reachable_fraction in (
        (32, 0, 1.0), (32, 0, 0.0), (32, 0, 0.5), (97, 5, 0.3),
        (128, 0, 0.75), (2048, 7, 0.6), (1, 0, 1.0), (1, 0, 0.0),
    ):
        values = torch.zeros(2, width)
        keep = torch.rand(2, width, generator=generator) < reachable_fraction
        values[keep] = torch.rand(int(keep.sum()), generator=generator)
        values[~keep] = float("-inf")
        # Unique ids, as the reference's input has them, and small enough that the
        # non-negativity contract for the leading site holds.
        keys = torch.arange(2 * width).reshape(2, width) % width
        cases.append((values, keys.to(torch.int32), offset))
    return cases


@pytest.mark.parametrize("values, keys, offset", _cases())
def test_trailing_oracle_matches_the_reference_expression(values, keys, offset):
    tail = int(width := values.shape[-1]) + int(offset) + 1
    expected = _reference_trailing(values, keys, offset, tail)
    actual = masked_sort_trailing(values, keys, offset, backend="torch")
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.int32


@pytest.mark.parametrize("values, keys, offset", _cases(seed=1))
def test_leading_oracle_matches_the_reference_expression(values, keys, offset):
    expected = _reference_leading(values, keys)
    actual = masked_sort_leading(values, keys, backend="torch")
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.int32


@pytest.mark.parametrize("values, keys, offset", _cases(seed=2))
def test_trailing_with_tail_is_the_call_site_form(values, keys, offset):
    tail = int(values.shape[-1]) + int(offset) + 1
    expected = _reference_trailing(values, keys, offset, tail)
    actual = masked_sort_trailing_with_tail(values, keys + offset, values.shape[-1], tail,
                                            backend="torch")
    assert torch.equal(actual, expected)


def test_padding_value_is_where_the_reference_puts_it():
    """A trailing sentinel leaves -1 after the real ids; a leading one puts it first."""
    values = torch.tensor([[1.0, 2.0, float("-inf"), 3.0]])
    keys = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    trailing = masked_sort_trailing(values, keys, 0, backend="torch")
    leading = masked_sort_leading(values, keys, backend="torch")
    assert trailing.tolist() == [[10, 11, 13, -1]]
    assert leading.tolist() == [[-1, 10, 11, 13]]
    assert int(trailing[0, -1]) == PADDING == int(leading[0, 0])


def test_every_entry_unreachable_stays_padding():
    values = torch.full((3, 16), float("-inf"))
    keys = torch.arange(3 * 16, dtype=torch.int32).reshape(3, 16)
    for backend in ("torch",):
        assert torch.equal(masked_sort_trailing(values, keys, 0, backend=backend),
                           torch.full((3, 16), -1, dtype=torch.int32))
        assert torch.equal(masked_sort_leading(values, keys, backend=backend),
                           torch.full((3, 16), -1, dtype=torch.int32))


def test_rejects_a_leading_table_with_negative_reachable_keys():
    """The fused form ranks padding ahead of every real id, so a negative id invalidates it."""
    values = torch.tensor([[1.0, 2.0]])
    keys = torch.tensor([[-5, 3]], dtype=torch.int32)
    with pytest.raises(ValueError, match="non-negative"):
        masked_sort_leading(values, keys, backend="torch")


def test_rejects_a_tail_that_is_not_above_every_reachable_id():
    values = torch.tensor([[1.0, 2.0, float("-inf")]])
    ids = torch.tensor([[10, 11, 99]], dtype=torch.int32)
    with pytest.raises(ValueError, match="above every reachable id"):
        masked_sort_trailing_with_tail(values, ids, 3, 11, backend="torch")


def test_rejects_a_width_that_does_not_match_the_table():
    values = torch.tensor([[1.0, 2.0]])
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="does not match the table width"):
        masked_sort_trailing_with_tail(values, ids, 4, 99, backend="torch")


def test_rejects_an_unknown_backend():
    values = torch.tensor([[1.0]])
    keys = torch.tensor([[0]], dtype=torch.int32)
    with pytest.raises(ValueError, match="backend must be one of"):
        masked_sort_trailing(values, keys, 0, backend="cuda")


def test_rejects_shape_and_dtype_mismatches():
    values = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="same shape"):
        masked_sort_trailing(values, torch.tensor([[0]], dtype=torch.int32), 0, backend="torch")
    with pytest.raises(TypeError, match="floating point"):
        masked_sort_trailing(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]), 0, backend="torch")
    with pytest.raises(TypeError, match="integers"):
        masked_sort_trailing(values, torch.tensor([[0.5, 1.5]]), 0, backend="torch")


def test_triton_availability_is_a_boolean():
    assert isinstance(_kernels.triton_available(), bool)


@pytest.mark.skipif(not _kernels.triton_available(), reason="this host has no Triton")
def test_fused_selection_matches_the_oracle_on_the_device():
    """The whole point of the deterministic-arithmetic claim: same table, on device."""
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else None
    if device is None:
        pytest.skip("needs an Ascend device")
    for values, keys, offset in _cases(seed=3):
        values = values.to(device)
        keys = keys.to(device)
        fused = masked_sort_trailing(values, keys, offset, backend="triton")
        oracle = masked_sort_trailing(values, keys, offset, backend="torch")
        assert torch.equal(fused.cpu(), oracle.cpu())