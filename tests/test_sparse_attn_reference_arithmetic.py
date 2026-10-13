"""`ops.sparse_attn`'s torch path, held to the released reference kernel's own output.

The V4.1-Flash checkpoint ships the kernel this repository's `sparse_attn` stands in for:
`inference/kernel.py`'s tilelang `sparse_attn_kernel`. It is a *blocked* online softmax --
`block = 64`, a running max over the blocks with `exp(m_prev - m)` rescaling both the output
accumulator and the denominator, and the numerator narrowed to bf16 (`T.copy(acc_s, acc_s_cast)`)
before the P.V gemm, while the denominator is summed from the fp32 numerators (`T.reduce_sum`
runs first). That order is visible in the result and it is *not* the fp32-exact softmax a first
reading of the formula gives: on the fixture below, an fp32-exact softmax reproduces 325 of the
reference's 384 values; the blocked form reproduces all 384.

Nothing here runs that kernel. TileLang is not installed in the environments this suite runs in
(it wants `torch>=2.10`), and the kernel needs a CUDA device and the `sparse_attn` wrapper's
16-head padding to compile. So the expectation below is a *recording* -- 384 bf16 values taken
from `sparse_attn_kernel` -- and the test asserts this repository's torch path against it, the way
`test_fp4_tie_rule.py` pins the reference's fp4 tie rounding. It cannot catch a change that moves
both sides, and it says nothing about the CUDA paths `sparse_attn` takes on a card; it is a pin on
the implementation every host can run.

What the fixture is chosen to cover:

* `topk_idxs` is 192 wide, so the block loop runs three times and the *running* max with its
  per-block rescale is live. A fixture of one block or less would pass for the fp32-exact form,
  which is the one thing this test exists to exclude.
* Three row patterns: a fully valid row, a row of all `-1` entries (the `-1e30` floor's own case
  -- exact zeros, not NaN), and a row whose second block is entirely `-1`, so the running max has
  to stay where it was while the block contributes nothing.
* `h = 2`, under the 16 heads the reference's wrapper pads to, so the recording went through that
  padding too.

The inputs are rebuilt from `arange` arithmetic with no RNG: every entry is a multiple of 1/8,
exactly representable in bf16, so any torch build rebuilds the same fixture. Only the reference's
output is embedded, and it is embedded as what it is -- bf16 values written out as Python floats,
which hold them exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relic_core.kernels import ops


# Recorded from the released checkpoint's `inference/kernel.py::sparse_attn` (tilelang) on an RTX
# 4090, on the `_fixture()` below, flattened from (1, 3, 2, 64). bf16 in, bf16 out.
REFERENCE_OUTPUT = (
    -0.3359375, 0.44140625, 0.5078125, 0.150390625, -0.111328125, -0.484375, -0.54296875, 0.2451171875,
    0.62890625, 0.326171875, 0.050048828125, -0.265625, -0.60546875, -0.3359375, 0.44140625, 0.5078125,
    0.150390625, -0.111328125, -0.484375, -0.54296875, 0.2451171875, 0.62890625, 0.326171875, 0.050048828125,
    -0.265625, -0.60546875, -0.3359375, 0.44140625, 0.5078125, 0.150390625, -0.111328125, -0.484375,
    -0.54296875, 0.2451171875, 0.62890625, 0.326171875, 0.050048828125, -0.265625, -0.60546875, -0.3359375,
    0.44140625, 0.5078125, 0.150390625, -0.111328125, -0.484375, -0.54296875, 0.2451171875, 0.62890625,
    0.326171875, 0.050048828125, -0.265625, -0.60546875, -0.3359375, 0.44140625, 0.5078125, 0.150390625,
    -0.111328125, -0.484375, -0.54296875, 0.2451171875, 0.62890625, 0.326171875, 0.050048828125, -0.265625,
    0.01129150390625, 0.515625, 0.28125, 0.0283203125, -0.22265625, -0.55078125, -0.255859375, 0.47265625,
    0.482421875, 0.166015625, -0.050537109375, -0.408203125, -0.466796875, 0.01129150390625, 0.515625, 0.28125,
    0.0283203125, -0.22265625, -0.55078125, -0.255859375, 0.47265625, 0.482421875, 0.166015625, -0.050537109375,
    -0.408203125, -0.466796875, 0.01129150390625, 0.515625, 0.28125, 0.0283203125, -0.22265625, -0.55078125,
    -0.255859375, 0.47265625, 0.482421875, 0.166015625, -0.050537109375, -0.408203125, -0.466796875, 0.01129150390625,
    0.515625, 0.28125, 0.0283203125, -0.22265625, -0.55078125, -0.255859375, 0.47265625, 0.482421875,
    0.166015625, -0.050537109375, -0.408203125, -0.466796875, 0.01129150390625, 0.515625, 0.28125, 0.0283203125,
    -0.22265625, -0.55078125, -0.255859375, 0.47265625, 0.482421875, 0.166015625, -0.050537109375, -0.408203125,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.28515625, -0.034423828125, -0.31640625, -0.625, -0.271484375, 0.470703125, 0.50390625, 0.12890625,
    -0.1396484375, -0.498046875, -0.451171875, 0.27734375, 0.671875, 0.28515625, -0.034423828125, -0.31640625,
    -0.625, -0.271484375, 0.470703125, 0.50390625, 0.12890625, -0.1396484375, -0.498046875, -0.451171875,
    0.27734375, 0.671875, 0.28515625, -0.034423828125, -0.31640625, -0.625, -0.271484375, 0.470703125,
    0.50390625, 0.12890625, -0.1396484375, -0.498046875, -0.451171875, 0.27734375, 0.671875, 0.28515625,
    -0.034423828125, -0.31640625, -0.625, -0.271484375, 0.470703125, 0.50390625, 0.12890625, -0.1396484375,
    -0.498046875, -0.451171875, 0.27734375, 0.671875, 0.28515625, -0.034423828125, -0.31640625, -0.625,
    -0.271484375, 0.470703125, 0.50390625, 0.12890625, -0.1396484375, -0.498046875, -0.451171875, 0.27734375,
    0.0096435546875, -0.314453125, -0.64453125, -0.33984375, 0.41796875, 0.5546875, 0.193359375, -0.125,
    -0.50390625, -0.50390625, 0.2080078125, 0.65234375, 0.396484375, 0.0096435546875, -0.314453125, -0.64453125,
    -0.33984375, 0.41796875, 0.5546875, 0.193359375, -0.125, -0.50390625, -0.50390625, 0.2080078125,
    0.65234375, 0.396484375, 0.0096435546875, -0.314453125, -0.64453125, -0.33984375, 0.41796875, 0.5546875,
    0.193359375, -0.125, -0.50390625, -0.50390625, 0.2080078125, 0.65234375, 0.396484375, 0.0096435546875,
    -0.314453125, -0.64453125, -0.33984375, 0.41796875, 0.5546875, 0.193359375, -0.125, -0.50390625,
    -0.50390625, 0.2080078125, 0.65234375, 0.396484375, 0.0096435546875, -0.314453125, -0.64453125, -0.33984375,
    0.41796875, 0.5546875, 0.193359375, -0.125, -0.50390625, -0.50390625, 0.2080078125, 0.65234375,
)


def _fixture():
    """The recorded fixture's inputs, `(q, kv, attn_sink, topk_idxs, scale)`.

    Built from `arange` arithmetic rather than a seeded RNG: the values are multiples of 1/8, so
    they are exact in bf16 and the fixture is byte-identical on any torch build. The two `-1`
    assignments are the row patterns the docstring describes.
    """
    b, s, h, d, topk, n = 1, 3, 2, 64, 192, 32
    idx = ((torch.arange(b * s * topk, dtype=torch.int64).reshape(b, s, topk) * 7 + 3) % n).to(torch.int32)
    idx[0, 1] = -1                       # the empty row: every index -1
    idx[0, 2, 64:] = -1                  # a row whose second block is entirely invalid
    q = ((((torch.arange(b * s * h * d, dtype=torch.float32).reshape(b, s, h, d) * 5) % 19) - 9) / 4).to(torch.bfloat16)
    kv = ((((torch.arange(b * n * d, dtype=torch.float32).reshape(b, n, d) * 11) % 13) - 6) / 4).to(torch.bfloat16)
    sink = (((torch.arange(h, dtype=torch.float32) * 3) % 7) - 3) / 8
    return q, kv, sink, idx, (1.0 / d) ** 0.5


def test_the_torch_path_reproduces_the_recorded_reference_output():
    """The reference's blocked softmax with its bf16 numerator, value for value.

    The threshold is the part that carries the claim. On the recording host this path matched all
    384 values bit-for-bit, and the fp32-exact softmax matched 325 (84.6%, worst case 3.9e-3 --
    one bf16 ulp). 95% sits between them with room on both sides, so a rewrite back to the exact
    form fails here while ordinary elementwise-order noise across torch builds does not.

    The value is read through `_sparse_attn_torch` rather than through `sparse_attn` so that the
    pin is on the arithmetic and not on a host: this fixture is `d = 64`, under the 512 the fused
    prefill branch needs, so the two names are one call here -- but only because of the shape, and
    naming the implementation keeps that from being load-bearing. (The fused branch is opt-in now,
    through `DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA`; `test_sparse_attn_prefill_default.py` pins that.)
    The public entry point still runs in the padding test below, where the claim is an equality
    between two of its own answers and holds either way.
    """
    q, kv, sink, idx, scale = _fixture()
    got = ops._sparse_attn_torch(q, kv, sink, idx, scale)
    want = torch.tensor(REFERENCE_OUTPUT, dtype=torch.bfloat16).reshape(1, 3, 2, 64)

    assert torch.isfinite(got).all()
    # The all `-1` row is exact zeros -- the floor's case, and the reason it is -1e30 and not -inf.
    assert (got[0, 1] == 0).all()

    identical = float((got == want).float().mean())
    assert identical >= 0.95, (
        f"only {identical:.1%} of the {got.numel()} values are the reference's; the blocked "
        f"bf16-numerator form reproduces 100% of them and the fp32-exact softmax 84.6%, so this "
        f"is neither -- and a value-for-value regression here is a logits regression everywhere "
        f"`sparse_attn` is called"
    )
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=0)


def test_the_under_16_head_padding_the_reference_does_is_a_no_op_here():
    """`h = 2` padded to the wrapper's 16 heads, then narrowed back, is the same answer.

    The released `sparse_attn` pads `q` and `attn_sink` up to 16 heads because its kernel's layout
    wants them, runs, and narrows the result back. The heads are computed independently of one
    another, so the padded ones cannot move the real ones -- this asserts that on the fixture,
    bit for bit, rather than asserting it in prose.
    """
    q, kv, sink, idx, scale = _fixture()
    direct = ops.sparse_attn(q, kv, sink, idx, scale)
    q_padded = torch.cat([q, q.new_zeros(q.size(0), q.size(1), 16 - q.size(2), q.size(3))], dim=2)
    sink_padded = torch.cat([sink, sink.new_zeros(16 - sink.numel())])
    padded = ops.sparse_attn(q_padded, kv, sink_padded, idx, scale)[:, :, : q.size(2)]

    assert torch.equal(direct, padded)
