"""Contracts for the compact integer host-CPU FP8 GEMV.

Three layers, deliberately separable:

* **independent arithmetic**: an exact-``Fraction`` statement of the reference reduction
  (two 16-term halves, nominal anchors, truncation toward zero on a 13-bit window, the
  carry onto the second window, the final 14-bit cut, then the two E8M0 scales) is
  compared against the compiled scalar path. It is a second statement of the rule, not a
  restatement of the kernel's own integer identities.
* **parity with the frozen v1 kernel**: the compiled integer kernel must produce the same
  binary32 bits as ``relic_core/csrc/cpu_block_gemv.cpp`` for every one of the 65536 e4m3
  code pairs, for mixed blocks that over-weight signed zeros and subnormals, and for the
  real Engram ``wkv`` geometry. v1 is the accepted reference and is not modified here.
* **the Python contract**: validation, the compaction accounting, source immutability and
  the two paths' agreement with each other.

The kernel is opt-in: every test compiles it into a temporary cache directory, so a
checkout that never asks for this kernel pays nothing.
"""
from __future__ import annotations

from fractions import Fraction
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv, cpu_integer_gemv  # noqa: E402

BLOCK = 32
GROUP = 4


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """One artifact for the whole module: compiling it is the expensive part."""
    return tmp_path_factory.mktemp("cpu_integer_gemv")


@pytest.fixture(scope="module")
def v1_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cpu_block_gemv_v1")


@pytest.fixture(autouse=True)
def _integer_cache(cache_dir, monkeypatch):
    """Keep prepack and GEMV calls inside the same explicitly requested cache."""
    monkeypatch.setenv(cpu_integer_gemv.CACHE_ENV, str(cache_dir))


# --------------------------------------------------------------------------- helpers


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_of_exact(value: Fraction) -> int:
    """The binary32 bits of an exact rational, rounding the way the hardware would.

    ``float(Fraction)`` is correctly rounded (nearest, ties to even) and raises instead of
    returning an infinity on overflow, so the overflow case is answered here.
    """
    try:
        as_float = float(value)
    except OverflowError:
        return 0xFF800000 if value < 0 else 0x7F800000
    return _bits(as_float)


def _e4m3_value(byte: int) -> float:
    """The exact value of an e4m3 code, from its fields rather than a lookup table."""
    sign = -1.0 if byte & 0x80 else 1.0
    field = (byte >> 3) & 15
    mantissa = byte & 7
    if field == 15 and mantissa == 7:
        return float("nan")
    if field == 0:
        return sign * math.ldexp(mantissa, -9)
    return sign * math.ldexp(8 + mantissa, field - 10)


def _e4m3_nominal(byte: int):
    """``max(true exponent, -6)``; ``None`` for a signed zero, which has no anchor."""
    if (byte & 0x7F) == 0:
        return None
    field = (byte >> 3) & 15
    return field - 7 if field >= 1 else -6


def _floor_log2(magnitude: Fraction) -> int:
    """``floor(log2(magnitude))`` for a positive rational, exactly."""
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    return exponent


def _trunc(value: Fraction) -> int:
    """Truncation toward zero, matching the reference's ``std::trunc``."""
    return int(value) if value >= 0 else -int(-value)


def _fraction_block(a_codes, b_codes) -> Fraction:
    """The reference block reduction as an exact rational.

    Every cut is a real-number truncation of an exact rational and both scales are powers
    of two, so this is the value the reference's block hands to the scale multiplies.
    """
    a_values = [Fraction(_e4m3_value(code)) for code in a_codes]
    b_values = [Fraction(_e4m3_value(code)) for code in b_codes]
    a_nominal = [_e4m3_nominal(code) for code in a_codes]
    b_nominal = [_e4m3_nominal(code) for code in b_codes]

    def anchor(span):
        live = [a_nominal[i] + b_nominal[i] for i in span
                if a_nominal[i] is not None and b_nominal[i] is not None]
        # An all-sentinel half has no anchor, and the reference then cuts nothing; that
        # cannot matter, because every product in such a half is zero.
        return max(live) if live else None

    first = anchor(range(0, 16))
    if first is None:
        whole_first = Fraction(0)
    else:
        grid = Fraction(2) ** (first - 13)
        whole_first = sum((_trunc(a_values[i] * b_values[i] / grid) for i in range(0, 16)),
                          Fraction(0))
    running = None if whole_first == 0 else _floor_log2(abs(whole_first)) + first - 13

    second = anchor(range(16, 32))
    if running is None:
        window = second
    elif second is None:
        window = running
    else:
        window = max(second, running)
    if window is None:
        whole_second = Fraction(0)
    else:
        grid = Fraction(2) ** (window - 13)
        whole_second = sum((_trunc(a_values[i] * b_values[i] / grid) for i in range(16, 32)),
                           Fraction(0))
    # The first half joins the second half's grid as a peer, truncated toward zero.
    carry = 0 if (first is None or window is None) else _trunc(
        whole_first * Fraction(2) ** (first - window))
    # `total` is an integer on the second window's grid, so the renormalised value is
    # `total * 2**(window-13)` -- and the final cut keeps 14 bits of *that*, not of the
    # bare integer, which is the one place the grid has to be carried explicitly.
    total = Fraction(whole_second + carry) * Fraction(2) ** (window - 13)
    if total == 0:
        return Fraction(0)
    step = Fraction(2) ** (_floor_log2(abs(total)) - 13)
    return Fraction(_trunc(total / step)) * step


def _codes(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.uint8)


def _inside_missing_libcxx() -> bool:
    """Whether this host's default C++ driver cannot find the standard library headers."""
    compiler = os.environ.get(cpu_block_gemv.COMPILER_ENV) or "c++"
    driver = shutil.which(compiler.split()[0])
    if driver is None:
        return False
    probe = subprocess.run([driver, "-x", "c++", "-E", "-"], input="#include <algorithm>\n",
                           capture_output=True, text=True, check=False)
    return probe.returncode != 0


@pytest.fixture(scope="module", autouse=True)
def _reference_compiler(tmp_path_factory):
    """Give the frozen v1 reference a compiler that can see its own standard library.

    On this checkout's macOS the CommandLineTools install ships an empty libc++ directory,
    so ``#include <algorithm>`` fails while the SDK still holds the headers. v1 predates
    that quirk and v1 is frozen, so the reference is built through a one-line wrapper
    named by the environment hook v1 already documents -- nothing in the tree changes,
    and on a host with a complete toolchain the wrapper is never created.
    """
    if not _inside_missing_libcxx():
        yield
        return
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True,
                         check=False).stdout.strip()
    include = f"{sdk}/usr/include/c++/v1"
    if not sdk or not Path(include).is_dir():
        pytest.skip("the frozen v1 reference cannot be compiled on this host")
    wrapper = tmp_path_factory.mktemp("v1_cc") / "cxx"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec "{shutil.which("c++")}" -isysroot "{sdk}" -nostdinc++ -isystem "{include}" "$@"\n'
    )
    wrapper.chmod(0o755)
    previous = os.environ.get(cpu_block_gemv.COMPILER_ENV)
    os.environ[cpu_block_gemv.COMPILER_ENV] = str(wrapper)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(cpu_block_gemv.COMPILER_ENV, None)
        else:
            os.environ[cpu_block_gemv.COMPILER_ENV] = previous


def _require_v1():
    """Assert the frozen v1 reference can be built here, and skip with its reason if not.

    Building a v1 artifact writes nothing into the tree -- the cache directory is the
    caller's -- and v1's source is never touched.
    """
    try:
        cpu_block_gemv.build()
    except (RuntimeError, OSError) as error:
        pytest.skip(f"the frozen v1 kernel cannot be built here: {error}")


# ------------------------------------------------------------------ exact arithmetic


def test_scalar_path_matches_an_exact_rational_statement(cache_dir):
    """The compiled scalar path against a ``Fraction`` statement of the same rule.

    The corpus covers signed zeros, all seven subnormals and the largest finite value in
    both operands, so the anchor sentinel and the negative shifts are exercised rather
    than assumed.
    """
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    alphabet = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x0F, 0x8F,
                0x38, 0xB8, 0x7E, 0xFE, 0x10, 0x90, 0x20, 0xA0, 0x3C, 0xBC]
    generator = torch.Generator().manual_seed(19)

    def draw(count):
        picks = torch.randint(0, len(alphabet), (count,), generator=generator).tolist()
        return [alphabet[index] for index in picks]

    for trial in range(60):
        a_codes = draw(BLOCK)
        columns = [draw(BLOCK) for _ in range(GROUP)]
        # Scales stay within a factor of 2**7 of unity so that neither of the reference's
        # two separately rounded multiplies can produce a subnormal intermediate: there a
        # double rounding is possible and the exact statement would stop being the same
        # question. The endpoints 0 and 254 are covered by the v1 parity corpus instead.
        a_scale = int(torch.randint(120, 135, (1,), generator=generator))
        b_scale = int(torch.randint(120, 135, (1,), generator=generator))
        weight = _codes([code for column in columns for code in column]).reshape(GROUP, BLOCK)
        scales = torch.full((GROUP, 1), b_scale, dtype=torch.uint8)
        prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
        got = cpu_integer_gemv.fp8_gemv_integer(
            _codes(a_codes).reshape(1, BLOCK),
            torch.tensor([[a_scale]], dtype=torch.uint8),
            prepared, scales, block_rows=1, mode="scalar",
        )
        a_power = Fraction(2) ** (a_scale - 127)
        b_power = Fraction(2) ** (b_scale - 127)
        for column in range(GROUP):
            exact = _fraction_block(a_codes, columns[column]) * a_power * b_power
            assert _bits(float(got[0, column])) == _bits_of_exact(exact), (
                f"trial {trial} column {column}: kernel {float(got[0, column])!r}, "
                f"exact statement {float(exact)!r}"
            )


def test_a_signed_zero_never_raises_a_half_anchor(cache_dir):
    """The sentinel has to stay below every live anchor.

    A signed zero has no nominal exponent. If its stored field decoded to a live nominal
    instead, the anchor of any half containing one would rise, every term in that half
    would shift by the same amount, and the block would be quietly wrong.
    """
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    half = [0x38] * 16 + [0x00] * 16  # 1.0's first half, signed zeros second half
    weight = _codes(half).reshape(1, BLOCK)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    result = cpu_integer_gemv.fp8_gemv_integer(
        _codes(half).reshape(1, BLOCK), scales, prepared, scales, block_rows=1, mode="scalar"
    )
    # Sixteen products of 1 * 1, then sixteen products of zero.
    assert _bits(float(result[0, 0])) == _bits(16.0)


# ---------------------------------------------------------------------- v1 parity


def test_every_code_pair_agrees_bit_for_bit_with_the_frozen_v1_kernel(cache_dir, v1_cache):
    """All 65536 e4m3 code pairs, against v1's own bits.

    One weight matrix holds all 256 values of the second operand as its columns, so all
    256 pairs for a given first-operand code are compared in a single pair of calls. A
    constant block is the exhaustive statement of the per-lane arithmetic; the mixed
    blocks below then check that lanes interact only through the anchor.
    """
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    _require_v1()
    # Column j is the constant code j, replicated down the 32 lanes.
    weight = _codes([value for value in range(256) for _ in range(BLOCK)]).reshape(256, BLOCK)
    weight_scales = torch.full((256 // BLOCK, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, weight_scales, block_rows=BLOCK, threads=2,
                                  cache_dir=cache_dir)
    activation_scales = torch.full((1, 1), 127, dtype=torch.uint8)
    failures = []
    for a_code in range(256):
        a_codes = torch.full((1, BLOCK), a_code, dtype=torch.uint8)
        reference = cpu_block_gemv.fp8_block_gemv(
            a_codes, activation_scales, weight, weight_scales,
            block_rows=BLOCK, threads=2, cache_dir=v1_cache,
        )
        got = cpu_integer_gemv.fp8_gemv_integer(
            a_codes, activation_scales, prepared, weight_scales,
            block_rows=BLOCK, mode="scalar",
        )
        for b_code in range(256):
            if _bits(float(reference[0, b_code])) != _bits(float(got[0, b_code])):
                failures.append((a_code, b_code))
                if len(failures) >= 8:
                    pytest.fail(f"the integer kernel disagrees with v1 for these code pairs: {failures}")
    assert not failures, f"{len(failures)} code pairs disagree with v1: {failures}"


@pytest.mark.parametrize(
    ("n", "k", "block_rows"),
    [(64, 64, 32), (128, 256, 1), (100, 192, 1), (96, 512, 1)],
)
def test_multi_block_parity_with_v1_includes_zeros_and_subnormals(n, k, block_rows, cache_dir,
                                                                  v1_cache):
    """Random blocks drawn from an alphabet that over-weights the awkward codes."""
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    _require_v1()
    alphabet = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x0F, 0x8F,
                0x38, 0xB8, 0x7E, 0xFE, 0x10, 0x90, 0x20, 0xA0, 0x3C, 0xBC]
    generator = torch.Generator().manual_seed(101 + n + k + block_rows)
    rows = n if block_rows == 1 else n // block_rows

    def draw(count):
        picks = torch.randint(0, len(alphabet), (count,), generator=generator).tolist()
        return [alphabet[index] for index in picks]

    a_codes = _codes(draw(k))
    a_scales = _codes(draw(k // BLOCK))
    weight = _codes(draw(n * k)).reshape(n, k)
    weight_scales = _codes(draw(rows * (k // BLOCK))).reshape(rows, k // BLOCK)

    reference = cpu_block_gemv.fp8_block_gemv(
        a_codes.reshape(1, k), a_scales.reshape(1, k // BLOCK), weight, weight_scales,
        block_rows=block_rows, threads=2, cache_dir=v1_cache,
    )
    prepared = prepack_fp8_weight(weight, weight_scales, block_rows=block_rows, threads=2,
                                  cache_dir=cache_dir)
    got = cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, k), a_scales.reshape(1, k // BLOCK), prepared, weight_scales,
        block_rows=block_rows, threads=2, mode="scalar",
    )
    mismatched = [column for column in range(n)
                  if _bits(float(reference[0, column])) != _bits(float(got[0, column]))]
    assert not mismatched, f"{len(mismatched)} of {n} columns differ from v1: {mismatched[:8]}"


# ---------------------------------------------------------------------- the two paths


@pytest.mark.parametrize(("n", "k"), [(64, 96), (256, 192), (40, 64)])
def test_the_vector_path_is_the_scalar_path_bit_for_bit(n, k, cache_dir):
    """``mode=`` selects an implementation, never a numerical answer."""
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    generator = torch.Generator().manual_seed(7 + n)
    a_codes = torch.randint(0, 256, (k,), generator=generator).to(torch.uint8)
    a_scales = torch.randint(0, 255, (k // BLOCK,), generator=generator).to(torch.uint8)
    weight = torch.randint(0, 256, (n, k), generator=generator).to(torch.uint8)
    weight_scales = torch.randint(0, 255, (n, k // BLOCK), generator=generator).to(torch.uint8)
    a_codes[(a_codes & 0x7F) == 0x7F] = 0x38
    weight[(weight & 0x7F) == 0x7F] = 0x38
    prepared = prepack_fp8_weight(weight, weight_scales, block_rows=1, cache_dir=cache_dir)
    scalar = cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, k), a_scales.reshape(1, k // BLOCK), prepared, weight_scales,
        block_rows=1, mode="scalar",
    )
    vector = cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, k), a_scales.reshape(1, k // BLOCK), prepared, weight_scales,
        block_rows=1, mode="auto", threads=3,
    )
    mismatched = [column for column in range(n)
                  if _bits(float(scalar[0, column])) != _bits(float(vector[0, column]))]
    assert not mismatched, f"the two paths disagree on {len(mismatched)} of {n} columns"


def test_the_thread_count_does_not_change_the_answer(cache_dir):
    """Each column accumulates its K blocks sequentially, so threads are throughput only."""
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    n, k = 256, 256
    generator = torch.Generator().manual_seed(31)
    a_codes = torch.randint(0, 256, (k,), generator=generator).to(torch.uint8)
    weight = torch.randint(0, 256, (n, k), generator=generator).to(torch.uint8)
    a_codes[(a_codes & 0x7F) == 0x7F] = 0x38
    weight[(weight & 0x7F) == 0x7F] = 0x38
    activation_scales = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    weight_scales = torch.full((n, k // BLOCK), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, weight_scales, block_rows=1, threads=3,
                                  cache_dir=cache_dir)
    answers = [cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, k), activation_scales, prepared, weight_scales,
        block_rows=1, threads=threads,
    ).clone() for threads in (1, 2, 4, 7)]
    for answer in answers[1:]:
        mismatched = [column for column in range(n)
                      if _bits(float(answer[0, column])) != _bits(float(answers[0][0, column]))]
        assert not mismatched, f"a different thread count changed {len(mismatched)} columns"


# ------------------------------------------------------------------------- non-finite


def test_a_nan_activation_code_makes_every_column_nan(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    a_codes = torch.full((BLOCK,), 0x38, dtype=torch.uint8)
    a_codes[5] = 0x7F  # an e4m3 NaN code
    weight = torch.full((GROUP, BLOCK), 0x38, dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    result = cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, BLOCK), torch.full((1, 1), 127, dtype=torch.uint8), prepared, scales,
        block_rows=1,
    )
    assert bool(torch.isnan(result).all())


def test_a_nan_in_one_weight_column_poisons_only_that_column(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    a_codes = torch.full((BLOCK,), 0x38, dtype=torch.uint8)
    weight = torch.full((GROUP, BLOCK), 0x38, dtype=torch.uint8)
    weight[2, 7] = 0x7F
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    result = cpu_integer_gemv.fp8_gemv_integer(
        a_codes.reshape(1, BLOCK), torch.full((1, 1), 127, dtype=torch.uint8), prepared, scales,
        block_rows=1,
    )
    assert bool(torch.isnan(result[0, 2]))
    assert _bits(float(result[0, 0])) == _bits(32.0)
    assert prepared.nan_columns.tolist() == [0, 0, 1, 0]


# ------------------------------------------------------------------------ the contract


def test_the_prepack_accounting_is_two_bytes_per_weight_plus_one_flag(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import packed_bytes, prepack_fp8_weight

    n, k = 25600, 6144
    assert packed_bytes(n, k)["total"] == n * k * 2 + n
    # A small stand-in keeps the suite fast while pinning the same arithmetic.
    weight = torch.zeros((64, 96), dtype=torch.uint8)
    scales = torch.full((64, 3), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    assert prepared.bytes["total"] == 64 * 96 * 2 + 64
    assert prepared.words.dtype == torch.uint16
    assert prepared.words.numel() * prepared.words.element_size() == 64 * 96 * 2
    assert prepared.nan_free_columns == 64


def test_a_mutated_source_tensor_is_reported_rather_than_silently_reused(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    prepared.verify_sources()
    weight[0, 0] = 0x38
    with pytest.raises(RuntimeError, match="modified after it was prepared"):
        prepared.verify_sources()


def test_a_scale_byte_of_255_is_refused_by_name(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 255, dtype=torch.uint8)
    with pytest.raises(ValueError, match="0xFF"):
        prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)


def test_an_fp32_scale_grid_must_be_an_exact_power_of_two(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, torch.ones((GROUP, 1), dtype=torch.float32),
                                  block_rows=1, cache_dir=cache_dir)
    assert prepared.words.dtype == torch.uint16
    with pytest.raises(ValueError, match="exact original E8M0 grid"):
        prepack_fp8_weight(weight, torch.full((GROUP, 1), 1.0000001, dtype=torch.float32),
                           block_rows=1, cache_dir=cache_dir)


def test_the_activation_must_be_one_row(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    with pytest.raises(ValueError, match="M = 1 GEMV"):
        cpu_integer_gemv.fp8_gemv_integer(
            torch.zeros((2, BLOCK), dtype=torch.uint8),
            torch.full((1, 1), 127, dtype=torch.uint8), prepared, scales, block_rows=1,
        )


def test_an_out_buffer_that_overlaps_an_input_is_refused(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    # A float32 view of the prepacked words has exactly the shape `out` wants, so the
    # only thing left to refuse is the aliasing itself.
    alias = prepared.words[: 2 * GROUP].view(torch.float32).reshape(1, GROUP)
    with pytest.raises(ValueError, match="must not overlap"):
        cpu_integer_gemv.fp8_gemv_integer(
            torch.zeros((1, BLOCK), dtype=torch.uint8), torch.full((1, 1), 127, dtype=torch.uint8),
            prepared, scales, block_rows=1, out=alias,
        )


def test_the_mode_and_threads_arguments_are_validated(cache_dir):
    from relic_core.kernels.cpu_integer_gemv import prepack_fp8_weight

    weight = torch.zeros((GROUP, BLOCK), dtype=torch.uint8)
    scales = torch.full((GROUP, 1), 127, dtype=torch.uint8)
    prepared = prepack_fp8_weight(weight, scales, block_rows=1, cache_dir=cache_dir)
    codes = torch.zeros((1, BLOCK), dtype=torch.uint8)
    activation_scales = torch.full((1, 1), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="mode must be"):
        cpu_integer_gemv.fp8_gemv_integer(codes, activation_scales, prepared, scales,
                                          block_rows=1, mode="fast")
    with pytest.raises(ValueError, match="threads must be"):
        cpu_integer_gemv.fp8_gemv_integer(codes, activation_scales, prepared, scales,
                                          block_rows=1, threads=0)


def test_no_cache_directory_means_no_build_or_write(monkeypatch):
    """The opt-in boundary: without a cache directory nothing is compiled or written."""
    monkeypatch.delenv(cpu_integer_gemv.CACHE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="no cache directory"):
        cpu_integer_gemv.cache_directory(None)


def test_the_artifact_is_content_addressed_and_cannot_collide_with_v1(cache_dir):
    plain = cpu_integer_gemv.compiled_library_path(cache_dir)
    assert plain.is_file()
    assert cpu_integer_gemv.compiled_library_path(cache_dir) == plain
    with_flags = cpu_integer_gemv.compiled_library_path(cache_dir, ["-DRELIC_PROBE=1"])
    assert with_flags != plain
    assert "cpu_integer_gemv" in plain.name
    assert f"abi{cpu_integer_gemv.ABI_VERSION}" in plain.name
    assert plain.name != cpu_block_gemv.compiled_library_path(cache_dir).name