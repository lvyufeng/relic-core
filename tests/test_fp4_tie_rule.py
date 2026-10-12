"""The E2M1 tie rule: nearest code, ties to the even one.

`fp4_act_quant` gives each element the nearest code in the E2M1 table. A value exactly halfway
between two codes has no nearest one, so the tie has to be broken by a rule, and the two obvious
rules disagree on half the midpoints -- which for a normal-distributed activation is about 0.5% of
the elements, each one a full level out of sixteen, in every fp4-quantized activation of a model.

The reference rounds half to even. Its source does not say so: the behaviour falls out of the E2M1
cast its kernel ends in, and was read off that kernel's own output at each midpoint rather than
assumed. `_quantize_fp4_codes` used to break ties with `argmin`, which returns the *lowest* index
and so took the lower code whatever its parity -- the same answer as the reference on a tie landing
on an even index, a different one on the rest.

The table is `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` in indices 0..7 with the negatives in 8..15, so "the
even code" is a parity of the index in both halves. Both signs are checked, because `argmin`'s
low-index habit is not symmetric and neither is the correction that replaced it.

No card is needed: the code assignment is arithmetic on a constant table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relic_core.kernels import ops

BLOCK = 32
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIDPOINTS = tuple((low + high) / 2 for low, high in zip(LEVELS, LEVELS[1:]))

# Recorded from the reference's `fp4_quant_kernel` (tilelang) on a block whose amax pins the scale
# at 1, so the expectation is the oracle's answer and not a restatement of the rule under test.
REFERENCE = (0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0)


def _quantize(value: float) -> float:
    """One block: the 6.0 that pins the scale, the value under test, then zeros to fill.

    An amax of 6.0 puts the scale at `2 ** ceil(log2(6 / 6)) == 1`, so the value reaches the code
    table unscaled -- which is what makes these assertions statements about codes, not about scales.
    """
    x = torch.tensor([6.0, value] + [0.0] * (BLOCK - 2), dtype=torch.float32).reshape(1, BLOCK)
    return float(ops.fp4_act_quant(x, BLOCK, True).reshape(-1)[1])


def test_matches_the_reference_at_every_midpoint():
    """The seven midpoints, against what the reference kernel actually produced for them."""
    assert tuple(_quantize(mid) for mid in MIDPOINTS) == REFERENCE


def test_the_even_code_wins_a_tie():
    """The same rule restated as a rule, so a midpoint the recorded block misses cannot slip by."""
    for index, mid in enumerate(MIDPOINTS):
        expected = LEVELS[index] if index % 2 == 0 else LEVELS[index + 1]
        assert _quantize(mid) == expected, f"+{mid} between indices {index} and {index + 1}"
        assert _quantize(-mid) == -expected, f"-{mid} between indices {index} and {index + 1}"


@pytest.mark.parametrize("sign", [1, -1])
def test_a_clear_nearest_still_wins(sign: int):
    """A correction that only moves ties must not move anything else.

    A value a quarter of the way to the neighbouring code is the closest thing to a tie that is
    not one, so it is where an over-eager correction would show up first.
    """
    for low, high in zip(LEVELS, LEVELS[1:]):
        assert _quantize(sign * (low + (high - low) * 0.25)) == sign * low
        assert _quantize(sign * (low + (high - low) * 0.75)) == sign * high
