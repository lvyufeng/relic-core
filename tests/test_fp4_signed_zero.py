"""The E2M1 sign bit: a negative value that rounds to zero is `-0.0`, not `+0.0`.

E2M1 has sixteen codes and only fifteen values -- `0x0` and `0x8` are `+0.0` and `-0.0`. Which one
a quantizer emits for a value in `(-0.25, 0.0)` is invisible to arithmetic and visible in the packed
payload, and the reference picks `0x8`: its kernel ends in `T.Cast(FP4, clamp(x / s, -6, 6))`, and a
sign-preserving cast keeps the sign of anything it rounds onto zero.

The port did not, and could not: index 8 of `_FP4_LEVELS` is written `0.0`, so it is a *second*
`+0.0` rather than a `-0.0`, and `argmin` never returns an index it is not closest to.
Measured against the reference on 2.1M activations of a normal-
distributed tensor, 112675 codes differed -- 5.4% -- and every one of them was code 0 against code
8, with no value difference anywhere in the tensor. `_quantize_fp4_codes` now takes the magnitude
from the table and the sign from the input's sign bit.

The dequantizer cannot see any of this. `_FP4_LEVELS[8]` being `+0.0` means a round trip through
`fp4_act_quant(..., inplace=True)` -- which multiplies the code back out by the block scale, as
`tests/test_fp4_tie_rule.py`'s helper does -- returns `+0.0` either way, so these assertions read
the packed payload instead. That is also why the payload is worth comparing at all.

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

# The model quantizes bf16 activations; float32 is here so a failure can be read without asking
# whether the input itself was representable.
DTYPES = (torch.float32, torch.bfloat16)


def _codes(values: list[float], dtype: torch.dtype) -> list[int]:
    """The 4-bit code the quantizer assigns to each value, read off the packed payload.

    The block's first element is a 6.0, which pins the scale at `2 ** ceil(log2(6 / 6)) == 1` so the
    values under test reach the code table unscaled -- the same pin
    `tests/test_fp4_tie_rule.py` uses, and for the same reason: it makes these assertions statements
    about codes and not about scales.
    """
    x = torch.tensor([6.0, *values] + [0.0] * (BLOCK - 1 - len(values)), dtype=dtype).reshape(1, BLOCK)
    packed, _ = ops.fp4_act_quant(x, BLOCK, False)
    raw = packed.reshape(-1).view(torch.uint8).to(torch.int64)
    # `_pack_fp4_codes` puts the even-indexed codes in the low nibble and the odd ones in the high,
    # so code `i` is the low half of byte `i // 2` for even `i` and the high half for odd `i`. The
    # 6.0 at index 0 is skipped by starting at 1.
    return [int(raw[i // 2] >> 4 if i % 2 else raw[i // 2] & 0xF)
            for i in range(1, len(values) + 1)]


@pytest.mark.parametrize("dtype", DTYPES)
def test_negative_zero_is_negative(dtype: torch.dtype):
    """The whole assertion, at the input that is its own case."""
    assert _codes([0.0, -0.0], dtype) == [0, 8]


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_small_negative_rounds_to_negative_zero(dtype: torch.dtype):
    """A value that rounds onto zero for its magnitude, not a literal zero.

    This is the one that a fix aimed at `-0.0` alone would miss: the sign has to come from the input
    rather than from having happened to land on the table's zero.
    """
    assert _codes([0.1, -0.1], dtype) == [0, 8]
    assert _codes([0.24, -0.24], dtype) == [0, 8]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("index,level", list(enumerate(LEVELS))[1:])
def test_the_sign_bit_is_unchanged_where_the_magnitude_carries_it(
    dtype: torch.dtype, index: int, level: float
):
    """Taking the sign from the input must not disturb the codes that already had it.

    For every nonzero level the table's index carries the sign, so reconstructing the byte from
    magnitude and sign bit has to reproduce that index exactly -- in both halves.
    """
    assert _codes([level, -level], dtype) == [index, index + 8]


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_zero_code_keeps_the_pairing_the_table_assumes(dtype: torch.dtype):
    """`0x0` and `0x8` are the two zeros, and neither is reachable for a nonzero magnitude.

    Swept across the zero boundary, so the one input where the tie rule and the sign rule meet --
    -0.25, equidistant from codes 0, 8 and 9 -- is asserted here rather than on a card.
    """
    codes = _codes([0.25, -0.25, 0.0, -0.0, 0.24, -0.24], dtype)
    assert codes == [0, 8, 0, 8, 0, 8]
