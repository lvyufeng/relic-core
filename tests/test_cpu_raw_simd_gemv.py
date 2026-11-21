"""Acceptance tests for the compiled raw-byte SIMD host CPU FP8 GEMV.

The claim is narrow and checkable: given the same raw e4m3 codes and the same
original E8M0 scale grids, this kernel produces the bits the scalar kernel
(:mod:`relic_core.kernels.cpu_block_gemv`) and the accepted Torch oracle
(``block_gemm.fp8_block_gemm``) produce. The comparison is ``torch.equal`` on the
FP32 bit patterns, never a tolerance, and it is run on the vector path and on the
forced-scalar build of the same source.

Nothing here needs a GPU, a checkpoint or a network. A C++ compiler has to be
present, because the kernel is compiled from ``csrc/cpu_raw_simd_gemv.cpp`` into a
temporary directory; without one the module skips with the reason, and a skip is
not a pass.

What the tests deliberately do not claim: a speedup. Timing is the caller's
measurement on its own host; the only performance-shaped assertion here is that
nothing in this design needs a prepacked operand, which is visible in the API
(it takes the checkpoint's ``[N, K]`` bytes) rather than timed.
"""
from __future__ import annotations

from fractions import Fraction
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv as scalar  # noqa: E402
from relic_core.kernels import cpu_raw_simd_gemv as simd  # noqa: E402
from relic_core.kernels.block_gemm import fp8_block_gemm  # noqa: E402

BLOCK = 32
FORCE_SCALAR = ["-DRELIC_RAW_SIMD_FORCE_SCALAR"]


def _inside_missing_libcxx() -> bool:
    """Whether this host's default C++ driver cannot find the standard library headers."""
    compiler = os.environ.get(simd.COMPILER_ENV) or "c++"
    driver = shutil.which(compiler.split()[0])
    if driver is None:
        return False
    probe = subprocess.run([driver, "-x", "c++", "-E", "-"], input="#include <algorithm>\n",
                           capture_output=True, text=True, check=False)
    return probe.returncode != 0


@pytest.fixture(scope="session", autouse=True)
def _reference_compiler(tmp_path_factory):
    """Give this checkout's macOS toolchain a driver that can see its own libc++.

    On this checkout's macOS the CommandLineTools install ships an empty libc++ directory,
    so ``#include <algorithm>`` fails while the SDK still holds the headers. The
    workaround is the one the compiled-GEMV tests already use: a one-line wrapper named by
    the module's own ``COMPILER_ENV`` hook, so nothing in the tree changes and a host with
    a complete toolchain never creates it. This is a *test* shim, not a global environment
    change: the variable is set for this module's compiler lookups and restored after.
    """
    if not _inside_missing_libcxx():
        yield
        return
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True,
                         check=False).stdout.strip()
    include = f"{sdk}/usr/include/c++/v1"
    if not sdk or not Path(include).is_dir():
        pytest.skip("the compiler cannot see a C++ standard library on this host")
    wrapper = tmp_path_factory.mktemp("raw_simd_cc") / "cxx"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec "{shutil.which("c++")}" -isysroot "{sdk}" -nostdinc++ -isystem "{include}" "$@"\n'
    )
    wrapper.chmod(0o755)
    # Both modules own a compiler hook: this one, and the frozen scalar kernel's
    # that is used as a reference here. Neither is a process-wide setting; each is
    # restored below.
    previous = {module: os.environ.get(module.COMPILER_ENV) for module in (simd, scalar)}
    for module in (simd, scalar):
        os.environ[module.COMPILER_ENV] = str(wrapper)
    try:
        yield
    finally:
        for module, value in previous.items():
            if value is None:
                os.environ.pop(module.COMPILER_ENV, None)
            else:
                os.environ[module.COMPILER_ENV] = value


@pytest.fixture(scope="session")
def cache():
    """A session-private build directory, or a skip that names the toolchain problem."""
    directory = tempfile.mkdtemp(prefix="relic-cpu-raw-simd-gemv-")
    try:
        simd.build(cache_dir=directory)
        scalar.build(cache_dir=directory)
    except RuntimeError as error:
        pytest.skip(f"the compiled host CPU GEMV cannot be built here: {error}")
    return directory


@pytest.fixture(scope="session")
def scalar_cache(cache):
    """The same source built with the portable scalar path forced."""
    try:
        simd.build(cache_dir=cache, extra_flags=FORCE_SCALAR)
    except RuntimeError as error:
        pytest.skip(f"the forced-scalar build of the same source failed: {error}")
    return cache


def finite_codes(shape, seed, low=0, high=256):
    """Random e4m3 bytes with the two NaN encodings removed.

    A NaN is a separate contract from the arithmetic, so the numerical tests stay
    on finite codes and the NaN tests construct their case explicitly.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(low, high, shape, generator=generator, dtype=torch.int32)
    values = torch.where((values & 0x7F) == 0x7F, values - 1, values)
    return values.to(torch.uint8)


def grid_bytes(rows, columns, seed, low=100, high=156):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(low, high, (rows, columns), generator=generator, dtype=torch.int32)
    return values.to(torch.uint8)


def run(cache, a_codes, a_scale, b_codes, b_scale, *, block_rows=BLOCK, threads=1, **kwargs):
    return simd.fp8_raw_simd_gemv(
        a_codes, a_scale, b_codes, b_scale,
        block_rows=block_rows, threads=threads, cache_dir=cache, **kwargs,
    )


def bits_equal(left, right):
    """FP32 bit-pattern equality, with NaN classified rather than compared by bits."""
    finite = torch.isfinite(left) & torch.isfinite(right)
    same = torch.equal(left[finite].view(torch.int32), right[finite].view(torch.int32))
    return same and torch.equal(torch.isnan(left), torch.isnan(right))


def reference(a_codes, a_scale, b_codes, b_scale, block_rows=BLOCK):
    return fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows, backend="torch")


# ---------------------------------------------------------------------------------------------
# The arithmetic, against the two existing implementations
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_rows", [BLOCK, 1])
@pytest.mark.parametrize("shape", [(64, 64), (256, 128), (1024, 256)])
def test_matches_the_scalar_kernel_and_the_torch_oracle(cache, block_rows, shape):
    n, k = shape
    a_codes = finite_codes((1, k), seed=k)
    b_codes = finite_codes((n, k), seed=n)
    a_scale = grid_bytes(1, k // BLOCK, seed=7)
    b_scale = grid_bytes((n + block_rows - 1) // block_rows, k // BLOCK, seed=11)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    expected = reference(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    old = scalar.fp8_block_gemv(
        a_codes, a_scale, b_codes, b_scale,
        block_rows=block_rows, cache_dir=cache,
    )
    assert actual.shape == (1, n)
    assert bits_equal(actual, expected)
    assert bits_equal(actual, old)


def test_every_code_product_is_reproduced(cache):
    """All 256 x 256 activation-code / weight-code products, one activation per call.

    K is two blocks so both halves of the truncation rule see a live operand; the
    remaining lanes are the zero code. 256 calls of 256 output rows is cheap and
    covers the whole product table rather than a sample of it.
    """
    codes = list(range(256))
    k = 64
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((len(codes) // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    b_codes = torch.zeros((len(codes), k), dtype=torch.uint8)
    for row, code in enumerate(codes):
        b_codes[row, 0] = code
        b_codes[row, 1] = code
    for activation in codes:
        a_codes = torch.zeros((1, k), dtype=torch.uint8)
        a_codes[0, 0] = activation
        a_codes[0, 1] = activation
        actual = run(cache, a_codes, a_scale, b_codes, b_scale)
        expected = reference(a_codes, a_scale, b_codes, b_scale)
        assert bits_equal(actual, expected), f"activation code {activation} diverged"


def test_cancellation_and_near_cancellation_are_exact(cache):
    """A block whose terms cancel to a tiny or zero total exercises both windows."""
    k = 128
    a_codes = finite_codes((1, k), seed=3)
    b_codes = finite_codes((512, k), seed=5)
    # A row of the negated codes, so the total cancels against the first row when
    # the weight row is the activation row itself.
    b_codes[0] = a_codes[0]
    b_codes[1] = (a_codes[0] ^ 0x80)  # sign-flipped codes
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((512 // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    expected = reference(a_codes, a_scale, b_codes, b_scale)
    assert bits_equal(actual, expected)
    # The sign-flipped row is the exact negation of the row above it.
    assert actual[0, 1] == -actual[0, 0]


@pytest.mark.parametrize("byte", [0, 1, 126, 127, 128, 253, 254])
def test_scale_endpoints_are_read_as_powers_of_two(cache, byte):
    """Byte 0 is the subnormal 2**-127 and byte 254 the largest finite grid value."""
    k, n = 64, 128
    a_codes = finite_codes((1, k), seed=byte + 1)
    b_codes = finite_codes((n, k), seed=byte + 2)
    a_scale = torch.full((1, k // BLOCK), byte, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), byte, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    expected = reference(a_codes, a_scale, b_codes, b_scale)
    assert bits_equal(actual, expected)


def test_an_all_zero_activation_row_produces_zero_columns(cache):
    """Every product is zero, so every column is a zero whatever the weight is."""
    k, n = 64, 64
    a_codes = torch.zeros((1, k), dtype=torch.uint8)
    b_codes = finite_codes((n, k), seed=22)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


def test_a_nan_weight_code_nan_classifies_only_its_column(cache):
    k, n = 64, 96
    a_codes = finite_codes((1, k), seed=31)
    b_codes = finite_codes((n, k), seed=32)
    b_codes[5, 0] = 0x7F  # the NaN code
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.isnan(actual[0, 5])
    assert torch.isfinite(actual[0, :5]).all()
    assert torch.isfinite(actual[0, 6:]).all()
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


def test_a_nan_activation_code_nan_classifies_every_column(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=41)
    a_codes[0, 7] = 0x7F
    b_codes = finite_codes((n, k), seed=42)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.isnan(actual).all()


# ---------------------------------------------------------------------------------------------
# The implementation shape: threads, tails, both paths
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 3, 7, 64, 256])
def test_the_result_does_not_depend_on_the_thread_count(cache, threads):
    k, n = 256, 1000
    a_codes = finite_codes((1, k), seed=51)
    b_codes = finite_codes((n, k), seed=52)
    a_scale = grid_bytes(1, k // BLOCK, seed=53)
    b_scale = grid_bytes(n, k // BLOCK, seed=54)  # per-row grid: N needs no multiple of 32

    baseline = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=1)
    for count in (threads,):
        actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=count)
        assert bits_equal(actual, baseline), f"threads={count} changed the result"


def test_a_short_final_chunk_and_a_non_multiple_of_32_output_are_exact(cache):
    """N = 1001 with a per-row grid: the last column, and the ragged thread chunk."""
    k, n = 96, 1001
    a_codes = finite_codes((1, k), seed=61)
    b_codes = finite_codes((n, k), seed=62)
    a_scale = grid_bytes(1, k // BLOCK, seed=63)
    b_scale = grid_bytes(n, k // BLOCK, seed=64)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=64)
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale, block_rows=1))


def test_the_forced_scalar_build_of_this_source_agrees_bit_for_bit(scalar_cache):
    """The fallback a non-NEON target compiles is the same kernel, not a variant."""
    k, n = 256, 512
    a_codes = finite_codes((1, k), seed=71)
    b_codes = finite_codes((n, k), seed=72)
    a_scale = grid_bytes(1, k // BLOCK, seed=73)
    b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=74)

    vector = run(scalar_cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    scalar_path = run(
        scalar_cache, a_codes, a_scale, b_codes, b_scale, threads=8, extra_flags=FORCE_SCALAR
    )
    assert bits_equal(vector, scalar_path)
    assert simd.vector_width(cache_dir=scalar_cache) in (1, 4)
    assert simd.vector_width(cache_dir=scalar_cache, extra_flags=FORCE_SCALAR) == 1


def test_the_reported_vector_width_matches_the_build(cache):
    width = simd.vector_width(cache_dir=cache)
    assert width in (1, 4)
    if hasattr(torch.backends, "cpu") and "aarch64" in (torch.backends.cpu.get_cpu_capability() or "").lower():
        assert width == 4
    assert simd.fp8_raw_simd_gemv_available(cache_dir=cache)


def test_out_storage_is_used_in_place_and_aliasing_is_refused(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=81)
    b_codes = finite_codes((n, k), seed=82)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    out = torch.empty((1, n), dtype=torch.float32)
    returned = run(cache, a_codes, a_scale, b_codes, b_scale, out=out)
    assert returned is out
    assert bits_equal(out, reference(a_codes, a_scale, b_codes, b_scale))


# ---------------------------------------------------------------------------------------------
# An independent exact-rational statement of the rule
# ---------------------------------------------------------------------------------------------


def _pow2(exponent: int) -> float:
    return float(math.ldexp(1.0, exponent))


def _code_value(byte: int) -> Fraction:
    sign = -1 if byte & 0x80 else 1
    field, mantissa = (byte >> 3) & 15, byte & 7
    if field == 0:
        return Fraction(sign * mantissa, 1 << 9)
    return Fraction(sign * (8 + mantissa), 1) * Fraction(_pow2(field - 10))


def _nominal(byte: int) -> int | None:
    if (byte & 0x7F) == 0:
        return None
    field = (byte >> 3) & 15
    return field - 7 if field >= 1 else -6


def _floor_log2(value: Fraction) -> int:
    """floor(log2(value)) for a positive rational, exactly."""
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    return exponent if Fraction(2) ** exponent <= value else exponent - 1


def _fraction_block(a_codes, b_codes) -> Fraction:
    """The documented rule, transcribed from the operator's docstring in exact rationals.

    The join is the part worth transcribing carefully: the second half's window is
    ``max(its own anchor, floor(log2|first half|))``, and the *running* first sum is
    re-cut onto that window before the two are added (``trunc(total_first *
    inverse_second)``). Adding the two half sums as they stand would put them on two
    different grids and lose exactly the half-step this test caught.
    """
    products = [Fraction(_code_value(a)) * Fraction(_code_value(b))
                for a, b in zip(a_codes, b_codes)]
    nominals = [_nominal(a) for a in a_codes]
    weight_nominals = [_nominal(b) for b in b_codes]

    def anchor(lo, hi):
        live = [n + w for n, w in zip(nominals[lo:hi], weight_nominals[lo:hi])
                if n is not None and w is not None]
        return max(live) if live else None

    def windowed(terms, exponent):
        """Sum of the terms each truncated toward zero onto ``2**(exponent-13)``.

        The cut applies to every product and the truncated integers are summed --
        that is the rule the integer 4-lane adds in the kernel exploit. Truncating
        the sum instead would be a different (and wrong) statement.
        """
        if exponent is None:
            return sum(terms, Fraction(0))
        step = Fraction(2) ** (exponent - 13)
        return sum((Fraction(math.trunc(term / step)) * step for term in terms),
                   Fraction(0))

    first = anchor(0, 16)
    first_sum = windowed(products[0:16], first)
    running = _floor_log2(abs(first_sum)) if first_sum != 0 else None
    second = anchor(16, 32)
    window = second if running is None else (running if second is None else max(second, running))
    second_sum = windowed(products[16:32], window)
    # The first half is re-cut onto the second half's window as a single value,
    # then added; the reference writes that as `trunc(total_first * inverse_second)`.
    total = second_sum + windowed([first_sum], window)
    if total == 0:
        return total
    return windowed([total], _floor_log2(abs(total)))


def test_an_independent_exact_rational_statement_of_the_rule_agrees(cache):
    """One K block, one sign of scale, compared as exact rationals then rounded once.

    This checks the *rule* the kernel implements, not the kernel's own arithmetic
    order: the scale multiplies are powers of two here, so the FP32 product is the
    exact rational scaled and the only rounding left is the final narrowing.
    """
    cases = 0
    for seed in range(6):
        a_codes = finite_codes((1, BLOCK), seed=100 + seed)
        b_codes = finite_codes((1, BLOCK), seed=200 + seed)
        expected = _fraction_block(a_codes[0].tolist(), b_codes[0].tolist())
        actual = run(
            cache, a_codes,
            torch.full((1, 1), 127, dtype=torch.uint8),
            b_codes,
            torch.full((1, 1), 127, dtype=torch.uint8),
            block_rows=1,
        )
        assert torch.isfinite(actual).all()
        assert Fraction(float(actual[0, 0])) == expected
        cases += 1
    assert cases == 6


def test_a_cancelling_block_is_zero_in_the_exact_statement_too(cache):
    """The same code in both halves, sign-flipped, is an exact zero block."""
    a_codes = torch.full((1, BLOCK), 0x38, dtype=torch.uint8)  # 1.0
    # 0x38 is exactly 1.0 and 0xB8 its negation: the sign bit, not a different
    # encoding of the same magnitude. (0x01 would be 2**-9.)
    b_codes = torch.full((1, BLOCK), 0x38, dtype=torch.uint8)
    b_codes[0, 16:] = 0xB8
    assert _fraction_block(a_codes[0].tolist(), b_codes[0].tolist()) == 0
    actual = run(
        cache, a_codes,
        torch.full((1, 1), 127, dtype=torch.uint8),
        b_codes,
        torch.full((1, 1), 127, dtype=torch.uint8),
        block_rows=1,
    )
    assert actual[0, 0] == 0.0


# ---------------------------------------------------------------------------------------------
# The public contract: refused inputs are refused by name
# ---------------------------------------------------------------------------------------------


def test_a_non_square_grid_needs_block_rows_one(cache):
    k, n = 64, 48
    a_codes = finite_codes((1, k), seed=91)
    b_codes = finite_codes((n, k), seed=92)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    per_row = torch.full((n, k // BLOCK), 127, dtype=torch.uint8)

    with pytest.raises(ValueError, match="weight scales must have shape"):
        run(cache, a_codes, a_scale, b_codes, per_row, block_rows=BLOCK)
    assert run(cache, a_codes, a_scale, b_codes, per_row, block_rows=1).shape == (1, n)


def test_bad_arguments_are_refused(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=101)
    b_codes = finite_codes((n, k), seed=102)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    with pytest.raises(ValueError, match="block_rows must be 1 or 32"):
        run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=16)
    with pytest.raises(ValueError, match="threads must be an integer"):
        run(cache, a_codes, a_scale, b_codes, b_scale, threads=0)
    with pytest.raises(ValueError, match="M = 1"):
        run(cache, torch.cat([a_codes, a_codes]), a_scale, b_codes, b_scale)
    with pytest.raises(ValueError, match="contiguous strided matrix"):
        run(cache, a_codes, a_scale, b_codes[:, :-1], b_scale)
    with pytest.raises(ValueError, match="same K, divisible by 32"):
        run(cache, a_codes, a_scale, b_codes[:, :-BLOCK].contiguous(), b_scale)


def test_scale_byte_255_is_refused_and_a_non_power_of_two_grid_too(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=111)
    b_codes = finite_codes((n, k), seed=112)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    bad = b_scale.clone()
    bad[0, 0] = 255
    with pytest.raises(ValueError, match="not a finite E8M0 grid value"):
        run(cache, a_codes, a_scale, b_codes, bad)

    fp32 = torch.full((n // BLOCK, k // BLOCK), 3.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="not an exact original E8M0 grid"):
        run(cache, a_codes, a_scale, b_codes, fp32)


def test_a_non_contiguous_or_gpu_operand_is_refused(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=121)
    b_codes = finite_codes((n, k), seed=122)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    wide = torch.zeros((n, 2 * k), dtype=torch.uint8)
    wide[:, ::2] = b_codes
    with pytest.raises(ValueError, match="contiguous strided matrix"):
        run(cache, a_codes, a_scale, wide[:, ::2], b_scale)