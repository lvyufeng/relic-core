"""Exactness contracts for `qwen4_exp_gated_rms_norm_exact`.

The operator replaces the elementwise tail of `RMSNormGated` for the Qwen4 GatedDeltaNet layers.
Its whole justification is bit-identity with the reference sequence, so these tests compare bit
patterns and not tolerances. Three groups run here:

* **Rounding rule, on the CPU.** `f32_bits_to_bf16_rn` in the kernel is integer software rounding,
  because the project's `nvcc` flags carry `--use_fast_math` (which implies `-ftz`). The Python
  port of that rule is checked against torch's own float32->bfloat16 cast over the whole float
  space, including subnormals, the subnormal/normal boundary and the overflow-to-infinity edge.
  This runs without a card and catches a wrong rounding rule before any GPU time is spent.
* **Source contracts.** The flags question above is not visible in a passing GPU test on a machine
  where the kernel happens to agree; it is visible in the source. These assertions fail if a later
  edit reintroduces a compiler-flag-dependent multiply or an `expf` call in the exact path.
* **CUDA bit-identity.** Skipped unless a card and the built extension are present; the parent runs
  them on the leased eight-4090 host. They are written to compare *bits*, treat NaN as equal to
  NaN, and cover the real `[rows, 128]` shape (`rows = query_rows * 6` value heads a rank), both
  gate dtypes, both activation modes, eager and captured replay, and the guard errors.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from relic_core.kernels.cuda_loader import load_cuda_kernel

EXACT = "qwen4_exp_gated_rms_norm_exact"
FUSED_ACT = "qwen4_exp_gated_rms_norm_exact_act"
HIDDEN = 128
EPS = 1e-6


# ---------------------------------------------------------------------------------------------
# The rounding rule, mirrored in Python so it can be checked without a device.
# ---------------------------------------------------------------------------------------------


def f32_bits_to_bf16_rn(bits: np.ndarray) -> np.ndarray:
    """The kernel's integer round-to-nearest-even float32 -> bfloat16, element for element."""
    bits = np.asarray(bits, dtype=np.uint64)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    nan_input = (exponent == 0xFF) & (mantissa != 0)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF
    quiet_nan = ((bits >> 16) | 0x0040) & 0xFFFF
    return np.where(nan_input, quiet_nan, rounded).astype(np.uint16)


def test_rounding_rule_matches_torch_over_the_whole_float_space():
    # The rounding decision depends on the low sixteen mantissa bits, so every one of them is
    # walked at every exponent (that is also every exact tie and every near-tie), with the full
    # mantissa tails sampled at random underneath the sign bit.
    rng = np.random.default_rng(20261118)
    exponents = np.arange(0, 256, dtype=np.uint64)
    low_bits = np.arange(0, 1 << 16, dtype=np.uint64)
    full_random = rng.integers(0, 1 << 23, size=4096, dtype=np.uint64)
    mantissas = np.concatenate([
        full_random,
        np.array([0x7FFFFF, 0x7FFF80, 0x7FFF7F, 0x7FFF81, 0x000001, 0x000080], dtype=np.uint64),
    ])
    positive = ((exponents[:, None] << 23) | low_bits[None, :]).reshape(-1)
    sign_bit = np.uint64(0x80000000)
    signed = (sign_bit
              | ((exponents[:, None] << 23) | mantissas[None, :]).reshape(-1))
    bits = np.concatenate([positive, signed])
    assert np.any(bits & np.uint64(0x80000000)), "negative float32 bit patterns must be covered"
    assert np.any((bits & np.uint64(0x80000000)) == 0), "positive float32 bit patterns must be covered"
    assert bits.size < (1 << 25), "keep the sweep large but not absurd"

    expected_bf16 = f32_bits_to_bf16_rn(bits)
    as_float = bits.astype(np.uint32).view(np.float32)
    reference = torch.from_numpy(as_float.copy()).to(torch.bfloat16)
    reference_bits = reference.view(torch.int16).numpy().astype(np.uint16)

    nan_input = np.isnan(as_float)
    finite = ~nan_input
    # NaN payloads are not a contract; that an input NaN stays NaN is.
    assert np.array_equal(np.isnan(reference.float().numpy()), nan_input)
    mismatched = finite & (expected_bf16 != reference_bits)
    if mismatched.any():
        index = int(np.flatnonzero(mismatched)[0])
        pytest.fail(
            f"rounding differs at bits=0x{int(bits[index]):08x} "
            f"(value={as_float[index]!r}): kernel 0x{int(expected_bf16[index]):04x} "
            f"torch 0x{int(reference_bits[index]):04x}"
        )


def test_rounding_rule_handles_overflow_into_infinity_and_signed_zero():
    values = np.array([
        0.0, -0.0, 3.3895313892515355e38, -3.3895313892515355e38,
        3.4028234663852886e38, -3.4028234663852886e38,     # fp32 max: rounds to bf16 inf
        1.0e-40, -1.0e-40,                                  # subnormal both ways
        1.1682471e-38,                                      # just below bf16's smallest normal
    ], dtype=np.float32)
    bits = values.view(np.uint32).astype(np.uint64)
    got = f32_bits_to_bf16_rn(bits)
    reference = torch.from_numpy(values.copy()).to(torch.bfloat16)
    assert np.array_equal(got, reference.view(torch.int16).numpy().astype(np.uint16))
    assert np.array_equal(got[:2], np.array([0x0000, 0x8000], dtype=np.uint16)), "signed zero kept"


# ---------------------------------------------------------------------------------------------
# Source contracts: the exactness argument depends on instructions no compiler flag can retarget.
# ---------------------------------------------------------------------------------------------


def _kernel_source() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return (root / "relic_core/csrc/qwen4_exp_gated_norm_exact.cu").read_text()


def test_exact_path_pins_the_multiply_and_avoids_flag_sensitive_math():
    source = _kernel_source()
    assert 'asm("mul.rn.f32 %0, %1, %2;"' in source, "float multiplies must stay explicit PTX"
    assert "0x7fffu + lsb" in source, "BF16 rounding must stay integer round-to-nearest-even"
    # `expf` under `--use_fast_math` is the approximate intrinsic, and torch's is not. The exact
    # entry must not reach any activation math at all.
    assert "expf(" not in source
    assert "1.0f / (1.0f" not in source, "no fast-math-rewritable float division in the exact path"


def test_new_source_is_registered_and_old_operations_are_untouched():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    setup = (root / "setup.py").read_text()
    binding = (root / "relic_core/csrc/cuda_kernel.cpp").read_text()
    assert '"relic_core/csrc/qwen4_exp_gated_norm_exact.cu"' in setup
    assert 'TORCH_CUDA_ARCH_LIST", "7.5;8.6;8.9"' in setup, "the default arch list is a contract"
    assert 'm.def("qwen4_exp_gated_rms_norm_exact"' in binding
    assert 'm.def("qwen4_exp_gated_rms_norm_exact_act"' in binding
    # The existing norm op keeps its name and behaviour; nothing here replaces it.
    assert 'm.def("qwen4_exp_grouped_rms_norm"' in binding


# ---------------------------------------------------------------------------------------------
# CUDA bit-identity.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, EXACT):
        pytest.skip("needs the built qwen4_exp_gated_rms_norm_exact entry")
    return ext


def bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare all device-side bits, including signed zeros and NaN canonicalization."""
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.view(torch.int16), expected.view(torch.int16)))


@pytest.mark.parametrize("hidden", [1, 128])
def test_device_subnormal_rounding_overflow_and_nan_bits(extension, hidden):
    # Walk each BF16 rounding tie and both neighbors throughout the entire FP32 subnormal
    # interval; include signed counterparts, overflow boundaries and noncanonical NaN payloads.
    bases = np.arange(0, 0x00810000, 0x10000, dtype=np.uint32)
    tail = np.array([0, 1, 0x7fff, 0x8000, 0x8001, 0xffff], dtype=np.uint32)
    positive = (bases[:, None] + tail[None, :]).reshape(-1)
    edge = np.array([0x7f7fffff, 0x7f7f7fff, 0x7f7f8000, 0x7f800000,
                     0x7fc00000, 0x7fffffff, 0x7f800001], dtype=np.uint32)
    positive = np.concatenate([positive, edge])
    raw = np.concatenate([positive, positive | np.uint32(0x80000000)])
    inverse = torch.from_numpy(raw.view(np.float32).copy()).cuda().view(-1, 1)
    x = torch.ones(inverse.shape[0], hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    activated = torch.ones_like(x, dtype=torch.float32)
    actual = getattr(extension, EXACT)(x, inverse, weight, activated)
    expected = reference(x, inverse, weight, activated)
    assert bits_equal(actual, expected), "all CUDA bits, including NaN, must match the actual tensor path"


def reference(x, inv_rms, weight, gate_activated):
    """`RMSNormGated.__call__` with the reduction supplied, transcribed from layers.py."""
    h = x.to(torch.float32)
    h = h * inv_rms                       # caller supplies the exact torch rsqrt(...)
    h = weight * h.to(torch.bfloat16)     # rounding boundary 1, then the BF16 weight product
    return (h * gate_activated).to(torch.bfloat16)   # boundary 3


def torch_inv_rms(x, eps=EPS):
    return torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)


def make_inputs(rows=6, hidden=HIDDEN, seed=41, scale=3.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    x = x * scale
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    return x, weight, gate


@pytest.mark.parametrize("query_rows", [1, 2, 4, 5, 6, 8])
def test_exact_entry_matches_the_reference_bit_for_bit(extension, query_rows):
    # The runtime flattens [batch, seq, value_heads, head_v_dim] to rows = seq * 6 heads a rank.
    rows = query_rows * 6
    x, weight, gate = make_inputs(rows=rows, seed=41 + query_rows)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    expected = reference(x, inv_rms, weight, gate_activated)
    actual = getattr(extension, EXACT)(x, inv_rms, weight, gate_activated)
    assert actual.dtype == torch.bfloat16 and actual.shape == x.shape
    assert bits_equal(actual, expected), "the fused epilogue must not change one bit"


@pytest.mark.parametrize("hidden", [128, 256, 192])
def test_exact_entry_matches_for_other_head_widths(extension, hidden):
    x, weight, gate = make_inputs(rows=6, hidden=hidden, seed=hidden)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    expected = reference(x, inv_rms, weight, gate_activated)
    assert bits_equal(getattr(extension, EXACT)(x, inv_rms, weight, gate_activated), expected)


def test_exact_entry_takes_a_float32_gate_and_cancellation_heavy_values(extension):
    # Values that makes the normalised product land close to a BF16 tie, so a wrong rounding rule
    # or a fused multiply-add shows up as a one-ulp difference rather than hiding in the noise.
    rows = 12
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1.0
    x[:, 1] = -1.0
    x[:, 2:] = torch.randn(rows, HIDDEN - 2, device="cuda", dtype=torch.bfloat16,
                           generator=generator) * 4.0
    weight = (torch.rand(HIDDEN, device="cuda", generator=generator) * 2 - 1).to(torch.bfloat16)
    gate = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.float32, generator=generator) * 6
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate)
    assert gate_activated.dtype is torch.float32
    expected = reference(x, inv_rms, weight, gate_activated)
    assert bits_equal(getattr(extension, EXACT)(x, inv_rms, weight, gate_activated), expected)


def test_exact_entry_preserves_infinities_and_nans(extension):
    x, weight, gate = make_inputs(rows=4)
    x[0, 0] = float("inf")
    x[1, 0] = float("-inf")
    x[2, 0] = float("nan")
    x[3, 5] = 0.0
    x[3, 6] = -0.0
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    expected = reference(x, inv_rms, weight, gate_activated)
    actual = getattr(extension, EXACT)(x, inv_rms, weight, gate_activated)
    assert bits_equal(actual, expected)


def test_exact_entry_is_repeatable_inside_a_cuda_graph(extension):
    x, weight, gate = make_inputs(rows=24)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    expected = reference(x, inv_rms, weight, gate_activated)
    op = getattr(extension, EXACT)
    # Warm up outside the recording, then capture: the op allocates its output, so it is exercised
    # the same way a captured verify step would exercise it.
    warm = op(x, inv_rms, weight, gate_activated)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, inv_rms, weight, gate_activated)
    graph.replay()
    assert bits_equal(captured, expected)
    graph.replay()
    assert bits_equal(captured, expected)
    assert bits_equal(warm, expected)


def test_exact_entry_accepts_a_4d_batch_sequence_shape(extension):
    # [batch, seq, value_heads, head_v_dim] is the shape the layer holds before its reshape.
    generator = torch.Generator(device="cuda").manual_seed(311)
    x = torch.randn(1, 4, 6, HIDDEN, device="cuda", dtype=torch.bfloat16,
                    generator=generator) * 2.5
    weight = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(1, 4, 6, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    expected = reference(x, inv_rms, weight, gate_activated)
    assert bits_equal(getattr(extension, EXACT)(x, inv_rms, weight, gate_activated), expected)


@pytest.mark.parametrize("activation,name", [(0, "sigmoid"), (1, "silu")])
def test_fused_activation_entry_is_consistent_with_torch_activation(extension, activation, name):
    """Bounded check, not an exactness claim: this entry is explicitly unproven.

    It evaluates the activation through double `exp` rather than torch's float32 `expf`, so a
    one-ulp float difference is possible and can flip a BF16 result at a rounding boundary. The
    assertion here is that it is within one BF16 ulp everywhere and exact almost everywhere; the
    count is printed so an oracle run can quantify it instead of guessing.
    """
    if not hasattr(extension, FUSED_ACT):
        pytest.skip("the extension does not carry the fused-activation entry")
    rows = 24
    x, weight, gate = make_inputs(rows=rows, seed=99)
    inv_rms = torch_inv_rms(x)
    gate_float = gate.to(torch.float32)
    torch_activated = torch.sigmoid(gate_float) if activation == 0 else torch.nn.functional.silu(
        gate_float)
    expected = reference(x, inv_rms, weight, torch_activated)
    actual = getattr(extension, FUSED_ACT)(x, inv_rms, weight, gate, activation)
    assert actual.dtype == torch.bfloat16 and actual.shape == x.shape
    equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if not equal:
        # One BF16 ulp is one unit of the low sixteen bits only within the same exponent; compare
        # in float and require the relative gap stay under 2**-7 (BF16's unit roundoff is 2**-8).
        gap = (actual.float() - expected.float()).abs()
        limit = expected.float().abs() * (2.0 ** -7) + 1e-6
        assert bool((gap <= limit).all()), (
            f"{name}: fused activation moves the result by more than one BF16 ulp"
        )
    matched = float((actual.view(torch.int16) == expected.view(torch.int16)).float().mean())
    print(f"\n{name}: bit-identical fraction {matched:.6f} (exactness is NOT claimed)")


def test_guard_errors_are_reported(extension):
    x, weight, gate = make_inputs(rows=6)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    op = getattr(extension, EXACT)
    with pytest.raises(RuntimeError):
        op(x.to(torch.float32), inv_rms, weight, gate_activated)          # x dtype
    with pytest.raises(RuntimeError):
        op(x, inv_rms.to(torch.bfloat16), weight, gate_activated)         # inv_rms dtype
    with pytest.raises(RuntimeError):
        op(x, inv_rms[:3], weight, gate_activated)                        # one value per row
    with pytest.raises(RuntimeError):
        op(x, inv_rms, weight[:-1], gate_activated)                       # weight width
    with pytest.raises(RuntimeError):
        op(x, inv_rms, weight, gate_activated[:3])                        # gate shape
    with pytest.raises(RuntimeError):
        op(x, inv_rms, weight, gate_activated.to(torch.bfloat16))         # needs fp32 activation
    with pytest.raises(RuntimeError):
        op(x.t(), inv_rms, weight, gate_activated)                        # non-contiguous rows


def test_transposed_rows_are_refused_rather_than_hidden_copied(extension):
    x, weight, gate = make_inputs(rows=6)
    inv_rms = torch_inv_rms(x)
    gate_activated = torch.sigmoid(gate.to(torch.float32))
    transposed = x.t().contiguous().t()          # [rows, hidden] with hidden-major strides
    assert not transposed.is_contiguous()
    with pytest.raises(RuntimeError):
        getattr(extension, EXACT)(transposed, inv_rms, weight, gate_activated)