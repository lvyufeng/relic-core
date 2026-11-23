"""The three decode fusion kernels against the eager expressions they replace.

Each module ships an explicit eager oracle, so the fused path is validated against the reference
rather than against itself. The CPU tests cover the oracle and every rejection path; the device tests
compare the fused kernel to the oracle where an Ascend device is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relic_core.kernels import _ascend_rope_pairs_kernels as rope_kernels  # noqa: E402
from relic_core.kernels import _ascend_row_norm_kernels as norm_kernels  # noqa: E402
from relic_core.kernels import _ascend_window_idxs_kernels as window_kernels  # noqa: E402
from relic_core.kernels.ascend_rope_pairs import (  # noqa: E402
    fused_rope_pairs,
    rope_pairs_reference,
)
from relic_core.kernels.ascend_row_norm import (  # noqa: E402
    fused_row_norm,
    row_norm_reference,
)
from relic_core.kernels.ascend_window_idxs import (  # noqa: E402
    window_topk_idxs,
    window_topk_idxs_reference,
)

_DEVICE = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else None


def _widths():
    """The widths this model normalizes and rotates: head 512, hidden 5120, plus an odd pair count."""
    return (512, 5120, 96)


# --------------------------------------------------------------------------- row norm


def test_row_norm_reference_is_the_eager_chain():
    """The oracle has to be `RMSNorm.forward`, so state it as that expression and compare."""
    torch.manual_seed(0)
    for dim in _widths():
        x = torch.randn(4, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.float32)
        eps = 1e-6
        dtype = x.dtype
        value = x.float()
        expected = (weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))).to(dtype)
        assert torch.equal(row_norm_reference(x, weight, eps), expected)


def test_row_norm_reference_without_weight_is_the_plain_scale():
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    weight = torch.ones(512, dtype=torch.float32)
    eps = 1e-6
    value = x.float()
    expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    assert torch.equal(row_norm_reference(x, weight, eps, weighted=False), expected)


def test_row_norm_rejects_what_the_fused_form_does_not_cover():
    weight = torch.ones(512, dtype=torch.float32)
    with pytest.raises(ValueError, match="backend must be one of"):
        fused_row_norm(torch.zeros(1, 512, dtype=torch.bfloat16), weight, 1e-6, backend="cuda")
    with pytest.raises(TypeError, match="bfloat16 or float16"):
        fused_row_norm(torch.zeros(1, 512, dtype=torch.float32), weight, 1e-6, backend="torch")
    with pytest.raises(TypeError, match="kept FP32"):
        fused_row_norm(torch.zeros(1, 512, dtype=torch.bfloat16),
                       weight.to(torch.bfloat16), 1e-6, backend="torch")
    with pytest.raises(ValueError, match="vector of the row width"):
        fused_row_norm(torch.zeros(1, 512, dtype=torch.bfloat16),
                       torch.ones(256, dtype=torch.float32), 1e-6, backend="torch")
    with pytest.raises(ValueError, match="leading row axis"):
        fused_row_norm(torch.zeros(512, dtype=torch.bfloat16), weight, 1e-6, backend="torch")


@pytest.mark.skipif(_DEVICE is None or not norm_kernels.triton_available(),
                    reason="needs an Ascend device and Triton")
@pytest.mark.parametrize("dim", _widths())
@pytest.mark.parametrize("rows", [1, 8, 40])
def test_fused_row_norm_matches_the_oracle_on_the_device(dim, rows):
    torch.manual_seed(dim + rows)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device=_DEVICE)
    weight = torch.randn(dim, dtype=torch.float32, device=_DEVICE)
    eps = 1e-6
    fused = fused_row_norm(x, weight, eps, backend="triton")
    oracle = row_norm_reference(x, weight, eps)
    assert torch.equal(fused, oracle), (
        "the fused row normalization has to reproduce the eager chain bit for bit; a mismatch means "
        "the single-block sum, the reciprocal mean or the framework rsqrt did not survive"
    )


# --------------------------------------------------------------------------- rope


def test_rope_pairs_reference_rotates_by_the_table():
    torch.manual_seed(1)
    x = torch.randn(2, 1, 3, 8, dtype=torch.bfloat16)
    freqs = torch.randn(1, 4, 2, dtype=torch.float32)
    original = x.clone()
    rotated = rope_pairs_reference(x, freqs)
    even = original[..., 0::2].float()
    odd = original[..., 1::2].float()
    cosine, sine = freqs[..., 0].float(), freqs[..., 1].float()
    assert torch.equal(rotated, x), "the eager rotation writes into the input and returns it"
    assert torch.equal(x[..., 0::2], (even * cosine - odd * sine).to(torch.bfloat16))
    assert torch.equal(x[..., 1::2], (even * sine + odd * cosine).to(torch.bfloat16))


def test_rope_pairs_reference_inverse_negates_the_sine():
    x = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)
    freqs = torch.randn(1, 4, 2, dtype=torch.float32)
    forward = rope_pairs_reference(x.clone(), freqs)
    inverse = rope_pairs_reference(x.clone(), freqs, inverse=True)
    assert not torch.equal(forward, inverse)


def test_rope_pairs_rejects_what_the_fused_form_does_not_cover():
    x = torch.zeros(1, 1, 2, 8, dtype=torch.bfloat16)
    good = torch.zeros(1, 4, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="backend must be one of"):
        fused_rope_pairs(x, good, backend="cuda")
    with pytest.raises(ValueError, match="even"):
        fused_rope_pairs(torch.zeros(1, 1, 3, dtype=torch.bfloat16), good, backend="torch")
    with pytest.raises(ValueError, match="single-position decode table"):
        fused_rope_pairs(x, torch.zeros(2, 4, 2, dtype=torch.float32), backend="torch")
    with pytest.raises(ValueError, match="cosine and a sine for every pair"):
        fused_rope_pairs(x, torch.zeros(1, 3, 2, dtype=torch.float32), backend="torch")
    with pytest.raises(ValueError, match="has to be .1, pairs, 2."):
        fused_rope_pairs(x, torch.zeros(4, 2, dtype=torch.float32), backend="torch")


@pytest.mark.skipif(_DEVICE is None or not rope_kernels.triton_available(),
                    reason="needs an Ascend device and Triton")
@pytest.mark.parametrize("shape", [(1, 1, 1, 512), (1, 1, 4, 512), (2, 1, 3, 512)])
def test_fused_rope_pairs_matches_the_oracle_on_the_device(shape):
    torch.manual_seed(sum(shape))
    x = torch.randn(*shape, dtype=torch.bfloat16, device=_DEVICE)
    freqs = torch.randn(1, shape[-1] // 2, 2, dtype=torch.float32, device=_DEVICE)
    for inverse in (False, True):
        fused = fused_rope_pairs(x.clone(), freqs, inverse=inverse, backend="triton")
        oracle = rope_pairs_reference(x.clone(), freqs, inverse=inverse)
        assert torch.equal(fused, oracle)


# --------------------------------------------------------------------------- window idxs


def test_window_reference_is_the_ring_expression():
    window, slot, row = 128, 100, 101
    table = window_topk_idxs_reference(window, slot, row)
    assert table.shape == (1, 1, window) and table.dtype == torch.int32
    ids = table.view(-1).tolist()
    assert ids[0] == (0 + slot + 1) % window
    # Everything at or past the ring head is a real slot; the rest is padding.
    for lane, value in enumerate(ids):
        expected = (lane + slot + 1) % window
        assert value == (expected if expected <= row else -1)


def test_window_reference_marks_every_slot_ahead_of_the_position():
    """A position at the ring head sees no future slot at all."""
    table = window_topk_idxs_reference(8, 0, 0).view(-1).tolist()
    # Lane 7 wraps to slot 0, which is the only slot at or behind the head; 1..7 are ahead.
    assert table == [-1, -1, -1, -1, -1, -1, -1, 0]


def test_window_rejects_a_slot_outside_the_ring():
    with pytest.raises(ValueError, match="not inside the ring"):
        window_topk_idxs(8, 8, 3, backend="torch")
    with pytest.raises(ValueError, match="cannot be negative"):
        window_topk_idxs(8, 0, -1, backend="torch")
    with pytest.raises(ValueError, match="at least one slot"):
        window_topk_idxs(0, 0, 0, backend="torch")
    with pytest.raises(ValueError, match="backend must be one of"):
        window_topk_idxs(8, 0, 3, backend="cuda")


@pytest.mark.skipif(_DEVICE is None or not window_kernels.triton_available(),
                    reason="needs an Ascend device and Triton")
@pytest.mark.parametrize("window,slot,row", [(128, 100, 101), (8, 0, 0), (128, 127, 200), (512, 3, 7)])
def test_fused_window_matches_the_oracle_on_the_device(window, slot, row):
    fused = window_topk_idxs(window, slot, row, backend="triton", device=_DEVICE)
    oracle = window_topk_idxs_reference(window, slot, row, device=_DEVICE)
    assert torch.equal(fused, oracle)