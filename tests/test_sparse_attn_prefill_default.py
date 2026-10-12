"""`sparse_attn`'s prefill branch is opt-in, and what it is opted out of.

A prefill-shaped call -- `q` on a card, more than one query row, head_dim 512 -- has two
implementations behind it. One is `_sparse_attn_torch`, the transcription of the released
`sparse_attn_kernel` that `test_sparse_attn_reference_arithmetic.py` pins against a recording of
that kernel. The other is `prefill_sparse_attn_headpair_kernel`, and it is a different function of
the same inputs: one key per warp through `__shfl_xor_sync`, one global max with the attention sink
folded into it, one fp32 denominator, and no bf16 narrowing of the numerator before P.V. What that
missing narrowing is worth has been measured against the reference -- 5.5e-3..1.0e-2 mean relative
on a real `attn_sink` at head_dim 512, a fifth of a bf16 ulp, in every layer of every forward pass
-- so which of the two answers a prefill call is not a scheduling detail. It decides whether this
library reproduces the released model or a nearby one.

`DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA` is that switch, and it now defaults to off. These tests pin the
default, pin that the env var still reaches the kernel, and pin the two things the change must not
have moved: a decode-shaped call never takes the prefill branch, and the other knobs keep their own
defaults.

The gate reads `q.is_cuda`, so the tests inject the card the way `test_ops_dispatch.py` injects a
capability: `_FakeCudaTensor` is a real CPU tensor whose `is_cuda` answers True, carrying real
values, and the extension is a stub that records which entry point was asked rather than a build.
What is asserted is *which* implementation answers, which is what the knob decides. What each of
them computes is the subject of the recording test next door and of every card-backed test in this
directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from relic_core.kernels import ops


class _FakeCudaTensor(torch.Tensor):
    """A CPU tensor that answers `is_cuda` with True. The one property the gate reads."""

    @property
    def is_cuda(self) -> bool:
        return True


class _StubExtension:
    """Stands in for the built extension: records the entry point, answers a sentinel."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sentinel: torch.Tensor | None = None

    def prefill_sparse_attn_headpair_forward(self, q, kv, attn_sink, topk_idxs, scale):
        self.calls.append("prefill_sparse_attn_headpair_forward")
        return self.sentinel

    def prefill_sparse_attn_forward(self, q, kv, attn_sink, topk_idxs, scale):
        self.calls.append("prefill_sparse_attn_forward")
        return self.sentinel


@pytest.fixture
def extension(monkeypatch):
    """A stub extension in the loader's slot, and a loader that refuses to be reached."""
    stub = _StubExtension()

    def refuse():
        raise AssertionError(
            "load_cuda_kernel() was called: the gate was supposed to short-circuit before the "
            "loader, and a build on this host must not decide what this test means"
        )

    monkeypatch.setattr(ops, "_INT8_CUDA_EXT", stub)
    monkeypatch.setattr(ops, "load_cuda_kernel", refuse)
    return stub


def _prefill_inputs(s: int = 2):
    """`(q, kv, attn_sink, topk_idxs, scale)` at the gate's own shape: `d = 512`, `s > 1`.

    Built from `arange` arithmetic with no RNG, as in the recording test next door. `topk = 128`
    spans two blocks and the second block of the second row is entirely invalid, so the answer is
    one the blocked running max had to produce -- the fixture is not a case the two forms agree on
    by accident.
    """
    b, h, d, topk, n = 1, 2, 512, 128, 8
    idx = ((torch.arange(b * s * topk).reshape(b, s, topk) * 3 + 1) % n).to(torch.int32)
    idx[0, 1, 64:] = -1
    q = (((torch.arange(b * s * h * d) % 17) - 8).float() / 8).to(torch.bfloat16)
    kv = (((torch.arange(b * n * d) % 11) - 5).float() / 8).to(torch.bfloat16)
    sink = ((torch.arange(h) % 5) - 2).float() / 4
    return q.reshape(b, s, h, d), kv.reshape(b, n, d), sink, idx, (1.0 / d) ** 0.5


def test_a_prefill_call_answers_with_the_reference_arithmetic_by_default(extension):
    """The card-shaped call, with the extension present, is `_sparse_attn_torch`'s answer."""
    q, kv, sink, idx, scale = _prefill_inputs()
    extension.sentinel = torch.full_like(q, 1234.0)

    got = ops.sparse_attn(_FakeCudaTensor(q), kv, sink, idx, scale)
    want = ops._sparse_attn_torch(q, kv, sink, idx, scale)

    assert extension.calls == [], (
        f"the fused prefill kernel answered an unopted-in prefill call ({extension.calls}); its "
        f"arithmetic is not the released kernel's (no bf16 numerator) and this is the default a "
        f"logits-parity port has to run"
    )
    assert torch.equal(got, want)


def test_the_env_var_still_reaches_the_kernel(extension, monkeypatch):
    """The knob stays a real switch: opted in, the same call is the extension's."""
    monkeypatch.setattr(ops, "_PREFILL_SPARSE_ATTN_CUDA", True)
    q, kv, sink, idx, scale = _prefill_inputs()
    extension.sentinel = torch.full_like(q, 1234.0)

    got = ops.sparse_attn(_FakeCudaTensor(q), kv, sink, idx, scale)

    assert extension.calls == ["prefill_sparse_attn_headpair_forward"]
    assert torch.equal(got, extension.sentinel)


def test_a_decode_shaped_call_never_takes_the_prefill_branch(extension, monkeypatch):
    """`s == 1` is the decode path: the prefill branch is not reachable from it, knob or not.

    The decode knobs are all still off, so the decode answer is the reference form too -- which is
    the point of the pin: this change moved the prefill default and nothing else.
    """
    monkeypatch.setattr(ops, "_PREFILL_SPARSE_ATTN_CUDA", True)
    q, kv, sink, idx, scale = _prefill_inputs()
    q, idx = q[:, :1], idx[:, :1]

    got = ops.sparse_attn(_FakeCudaTensor(q), kv, sink, idx, scale)

    assert extension.calls == []
    assert torch.equal(got, ops._sparse_attn_torch(q, kv, sink, idx, scale))


# Read at import into module constants, so the default can only be asked of a fresh interpreter --
# and only with the variable *absent*, since the A/B that motivated this default exports it. One
# child does both reads: the second reloads the module from the same import with the variable set.
_ENV_PROBE = """
import importlib
import json
import os

import relic_core.kernels.ops as ops

NAMES = (
    "_PREFILL_SPARSE_ATTN_CUDA",
    "_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA",
    "_FUSED_DECODE_ATTN_CUDA",
    "_TENSOR_CORE_ATTN_CUDA",
    "_FLASHINFER_STYLE_ATTN_HEADPAIR_CUDA",
)

out = {"unset": {name: bool(getattr(ops, name)) for name in NAMES}}
os.environ["DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA"] = "1"
importlib.reload(ops)
out["exported_1"] = {name: bool(getattr(ops, name)) for name in NAMES}
print(json.dumps(out))
"""


def test_the_knob_defaults_off_and_the_environment_still_overrides_it():
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    done = subprocess.run(
        [sys.executable, "-c", _ENV_PROBE], env=env, capture_output=True, text=True, check=True
    )
    got = json.loads(done.stdout.strip().splitlines()[-1])

    assert got["unset"]["_PREFILL_SPARSE_ATTN_CUDA"] is False, (
        "the CUDA prefill kernel is reachable without asking for it: it implements a one-global-max "
        "fp32 form, not the released kernel's blocked form with the bf16-narrowed numerator, so a "
        "default that picks it cannot reproduce the reference's logits"
    )
    # The knobs this change does not touch keep their own defaults: the decode set stays off, and
    # the headpair selector still means "prefer the headpair kernel" -- once the branch is opted into.
    assert got["unset"]["_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA"] is True
    assert got["unset"]["_FUSED_DECODE_ATTN_CUDA"] is False
    assert got["unset"]["_TENSOR_CORE_ATTN_CUDA"] is False
    assert got["unset"]["_FLASHINFER_STYLE_ATTN_HEADPAIR_CUDA"] is False

    assert got["exported_1"]["_PREFILL_SPARSE_ATTN_CUDA"] is True, (
        "DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA no longer reaches the kernel; the two-knob A/B and every "
        "launch script that opts into the fused prefill kernel depend on it being a real switch"
    )