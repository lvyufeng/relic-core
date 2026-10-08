"""The operator layer's dispatch: which implementation runs for which card.

`ops.py` has a layer above the kernels that decides *what code runs* before any tensor is
touched -- `_auto_impl` picks Triton or Torch per op kind, and `_choose_compute_dtype` picks the
accumulator width. That layer is the one that forks on the card: on a capability below 8 an FP8
matmul takes the Torch path and the compute dtype is fp16, and on 8 or above both change. It is
also, until this file, the *only* part of this package with no test at all -- every other test
drives a real kernel against a real build, this one reads the numbers.

That matters because the two cards this repository is validated on sit on opposite sides of the
fork. A test that needs no card is the only kind that can assert both sides from one machine, so
this one injects the capability rather than reading it: `major` comes from `torch.cuda` at run
time in the code under test, and from the test's own argument here.

What this does **not** cover, stated so a green run is not read as more than it is: the Triton and
Torch kernels those paths return are still untested against each other. This asserts the *choice*,
which is testable everywhere and was tested nowhere; it does not assert that the faster choice is
correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relic_core.kernels import ops


# ---------------------------------------------------------------------------------------------
# Injecting the card
# ---------------------------------------------------------------------------------------------
#
# `_auto_impl` and `_choose_compute_dtype` read the card through `torch.cuda` at call time, which
# makes the fork two small patches away from being reachable from a machine with no CUDA device at
# all -- and reachable from *both* sides from the same machine, which is the point. A stub rather
# than a real device, because the question is which branch the code takes and not what the branch
# computes.


class _FakeCuda:
    """Stands in for `torch.cuda` as far as the dispatch layer reads it."""

    def __init__(self, *, available: bool, capability: tuple[int, int] | None) -> None:
        self._available = available
        self._capability = capability
        self.asked = 0

    def is_available(self) -> bool:
        return self._available

    def get_device_capability(self, device=None) -> tuple[int, int]:
        # The real one takes an optional device and the dispatch layer calls both spellings --
        # `_auto_impl` with no argument, `_choose_compute_dtype` with one. Counting the calls is
        # how the "below 8 short-circuits before it reads the card" case is asserted.
        self.asked += 1
        assert self._capability is not None, "the card was read when the test did not offer one"
        return self._capability


@pytest.fixture
def card(monkeypatch):
    """Install a fake card and hand back a setter.

    Returns a callable rather than a value because one test needs to ask the same question with
    no card at all after asking it with one.
    """

    def install(*, available: bool, capability: tuple[int, int] | None = None) -> _FakeCuda:
        stub = _FakeCuda(available=available, capability=capability)
        monkeypatch.setattr(ops.torch, "cuda", stub)
        return stub

    return install


@pytest.fixture
def impls(monkeypatch):
    """Force `_USE_TRITON` to a value without importing Triton.

    The real flag is set once at import from whether `import triton` succeeded, so a test that
    wants the Triton branch on a host without Triton -- or the Torch branch on a host with it --
    has to set it directly. That is honest: the flag is a module-level fact, and pinning it is
    how the choice becomes deterministic instead of host-dependent.
    """

    def set_value(value: bool) -> None:
        monkeypatch.setattr(ops, "_USE_TRITON", value)

    return set_value


# ---------------------------------------------------------------------------------------------
# _auto_impl -- the FP8 fork
# ---------------------------------------------------------------------------------------------


def test_fp8_takes_torch_below_capability_8_and_triton_at_or_above(impls, card):
    """The fork the two validated cards sit on opposite sides of.

    sm_75 is below the cut and sm_89 is at it, so this one assertion is the whole reason the
    capability check exists in this layer. Both are asserted from one machine because the card is
    injected.
    """
    impls(True)

    card(available=True, capability=(7, 5))
    assert ops._auto_impl("fp8") == "torch", "an sm_75 card has no Triton FP8 path to take"

    card(available=True, capability=(8, 9))
    assert ops._auto_impl("fp8") == "triton", "an sm_89 card takes the Triton path"


def test_fp8_below_capability_8_answers_torch_even_with_triton_available(impls, card):
    """The below-8 answer does not depend on Triton being importable.

    This is the assertion that fails if the flag is consulted before the card: with Triton
    available, a flag-first ordering returns "triton" on an sm_75 card, which is exactly the
    code the fork exists to route around.
    """
    impls(True)
    card(available=True, capability=(7, 5))

    assert ops._auto_impl("fp8") == "torch"


def test_fp8_with_no_cuda_device_falls_through_to_the_triton_flag(impls, card):
    """No card is not the same as an old card, and the two land in different places.

    `torch.cuda.is_available()` false skips the capability read entirely, so the answer is the
    Triton flag alone -- which is what a CPU-side test of the Triton path relies on.
    """
    card(available=False)

    impls(True)
    assert ops._auto_impl("fp8") == "triton"
    impls(False)
    assert ops._auto_impl("fp8") == "torch"


def test_the_other_triton_kinds_do_not_ask_the_card(impls, card):
    """fp4, int8 and hc_split are capability-independent; only fp8 forks.

    Worth pinning because the fork being in exactly one kind is what keeps the sm_75/sm_89
    difference narrow. If a second kind grew a card check, this is where it would show up.
    """
    impls(True)
    stub = card(available=True, capability=(7, 5))

    assert ops._auto_impl("fp4") == "triton"
    assert ops._auto_impl("int8") == "triton"
    assert ops._auto_impl("hc_split") == "triton"
    assert stub.asked == 0, "none of these three read the card"


def test_fp8_quant_is_always_torch(impls, card):
    """The quantiser has no Triton path at all, on any card."""
    impls(True)
    stub = card(available=True, capability=(9, 0))

    assert ops._auto_impl("fp8_quant") == "torch"
    assert stub.asked == 0


def test_an_unknown_kind_defaults_to_torch(impls):
    """An op kind with no entry takes the safe path rather than guessing Triton."""
    impls(True)
    assert ops._auto_impl("something-new") == "torch"


# ---------------------------------------------------------------------------------------------
# _resolve_impl -- an explicit request against the flag
# ---------------------------------------------------------------------------------------------


def test_an_explicit_triton_request_is_downgraded_when_triton_is_absent(impls):
    """`impl="triton"` on a host without Triton returns torch rather than lying.

    The layer must not hand back "triton" for code that then cannot import -- the caller uses the
    string to pick a function, so a wrong answer here is an ImportError one frame away.
    """
    impls(False)
    assert ops._resolve_impl("fp8", "triton") == "torch"

    impls(True)
    assert ops._resolve_impl("fp8", "triton") == "triton"


def test_auto_delegates_to_the_fork(impls, card):
    """`"auto"` is the fork; anything else is taken as given."""
    impls(True)
    card(available=True, capability=(7, 5))
    assert ops._resolve_impl("fp8", "auto") == "torch"

    card(available=True, capability=(8, 9))
    assert ops._resolve_impl("fp8", "auto") == "triton"

    # An explicit "torch" is never upgraded, whatever the card could do.
    assert ops._resolve_impl("fp8", "torch") == "torch"


# ---------------------------------------------------------------------------------------------
# _choose_compute_dtype / _to_output_dtype -- the accumulator width
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "capability, default_is_bf16, expected",
    [
        ((7, 5), True, "float16"),
        ((7, 5), False, "float16"),
        ((8, 0), True, "bfloat16"),
        ((8, 0), False, "float16"),
        ((8, 6), True, "bfloat16"),
        ((9, 0), True, "bfloat16"),
    ],
)
def test_the_compute_dtype_follows_the_same_cut(impls, card, capability, default_is_bf16, expected, monkeypatch):
    """Below 8 is always fp16; at or above it is bf16 only when the default dtype is bf16.

    Parametrised across the boundary because the two validated cards (7.5 and 8.9) are its two
    ends, and a test that checked only those two would pass without pinning where the cut is or
    that the flag is consulted at all. Rows with `default_is_bf16=False` are the ones that catch
    an "8 or above is always bf16" mistake.
    """
    card(available=True, capability=capability)
    monkeypatch.setattr(
        ops.torch, "get_default_dtype", lambda: ops.torch.bfloat16 if default_is_bf16 else ops.torch.float32
    )

    assert ops._choose_compute_dtype(ops.torch.device("cuda")) == getattr(ops.torch, expected)


def test_the_compute_dtype_does_not_read_the_card_off_cuda(card):
    """A non-CUDA device never reaches the capability read, whatever the flag says."""
    stub = card(available=True, capability=(7, 5))
    assert ops._choose_compute_dtype(ops.torch.device("cpu")) == ops.torch.float32
    assert stub.asked == 0


def test_the_output_dtype_narrows_to_fp16_below_8(card, monkeypatch):
    """`_to_output_dtype` cuts in the same place, so the two never disagree about the card."""
    monkeypatch.setattr(ops.torch, "get_default_dtype", lambda: ops.torch.bfloat16)

    card(available=True, capability=(7, 5))
    assert ops._to_output_dtype(ops.torch.device("cuda")) == ops.torch.float16

    card(available=True, capability=(8, 9))
    assert ops._to_output_dtype(ops.torch.device("cuda")) == ops.torch.bfloat16