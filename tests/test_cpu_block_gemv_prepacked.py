"""Acceptance tests for the prepacked, column-vectorised exact host FP8 GEMV.

The claim under test is narrow and checkable: ``fp8_block_gemv_prepacked`` returns
**the same float32 bits** as the frozen v1 kernel ``cpu_block_gemv.fp8_block_gemv``
for the same operands, on every path this module can take -- the NEON path, the
portable scalar path, any thread count, both weight-scale grids and every column
tail. v1 is the oracle because it is the accepted implementation; an independent
exact-rational reference re-derives a small case from scratch so the test is not
only "two implementations agree".

Nothing here is a whole-model claim. These are the operator's own contracts, plus
the memory accounting and provenance behaviour the caller relies on:

* the prepacked form is exactly **4x** the stored code bytes plus a small scale and
  validity overhead, reported before it is allocated;
* the source tensors are provenance, not input -- the prepacked words are what the
  kernel reads, so mutating a source is *detected* rather than silently changing a
  result;
* NaN propagation is a class, not a payload: a NaN weight column and a NaN
  activation make the same columns NaN as v1 does.

Run from the repository root:

    python -m pytest tests/test_cpu_block_gemv_prepacked.py -q -rs

The big-geometry case at the released ``wkv`` shape (N = 25600, K = 6144) is part
of the file; it allocates roughly 800 MiB and takes a couple of seconds.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv as v1  # noqa: E402
from relic_core.kernels import cpu_block_gemv_prepacked as v2  # noqa: E402

BLOCK = 32
#: E4M3's NaN byte class, as the two implementations agree on it.
NAN_CODES = (0x7F, 0xFF)


def _host_toolchain_flags() -> list[str]:
    """Extra compiler flags this host's toolchain needs, if any.

    The operators take ``extra_flags``/``*_CXXFLAGS`` precisely so a host whose
    libc++ headers are not on the compiler's default search path can say so once.
    The local Apple Command Line Tools install here ships an empty libc++ include
    directory while the SDK it points at has the headers, so a plain compile cannot
    find ``<algorithm>``. Detected rather than assumed: the probe is a real compile,
    and a host where it succeeds gets no flags.
    """
    configured = os.environ.get("RELIC_CPU_GEMV_PREPACKED_CXXFLAGS", "").strip()
    if configured:
        return configured.split()
    if sys.platform != "darwin":
        return []
    probe = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False,
                                        dir=tempfile.gettempdir())
    probe.write("#include <algorithm>\nint main() { return 0; }\n")
    probe.close()
    try:
        plain = subprocess.run(
            [shutil.which("c++") or "c++", "-std=c++17", "-fsyntax-only", probe.name],
            capture_output=True,
        )
        if plain.returncode == 0:
            return []
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True
        ).stdout.strip()
        if not sdk:
            return []
        return ["-isysroot", sdk, "-isystem", f"{sdk}/usr/include/c++/v1"]
    finally:
        os.unlink(probe.name)


_FLAGS = _host_toolchain_flags()
# Both operators read their flags from the environment; setting them here (before
# any build) is the documented way to state a host's toolchain layout.
if _FLAGS:
    _flags_text = " ".join(_FLAGS)
    os.environ.setdefault("RELIC_CPU_GEMV_PREPACKED_CXXFLAGS", _flags_text)
    os.environ.setdefault("RELIC_CPU_GEMV_CXXFLAGS", _flags_text)
    os.environ.setdefault("RELIC_CPU_GEMV_PREPACKED_CXX", shutil.which("c++") or "c++")
    os.environ.setdefault("RELIC_CPU_GEMV_CXX", shutil.which("c++") or "c++")


@pytest.fixture(scope="module")
def cache() -> str:
    """One build directory for the whole module; the artifacts are content-addressed."""
    path = tempfile.mkdtemp(prefix="relic-cpu-gemv-prepacked-test-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _codes(n: int, k: int, *, seed: int, allow_nan: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    if not allow_nan:
        for nan_code in NAN_CODES:
            codes.masked_fill_(codes == nan_code, 0x38)
    return codes


def _row_codes(k: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    for nan_code in NAN_CODES:
        codes.masked_fill_(codes == nan_code, 0x38)
    return codes


def _scales(rows: int, blocks: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # 100..150 covers subnormal-scale codes (byte 0 is the only subnormal exponent,
    # so it is exercised separately) through ordinary STORED grid values.
    return torch.randint(100, 151, (rows, blocks), dtype=torch.uint8, generator=generator)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """All float32 bits equal, with NaN compared as a class and never as a payload."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    left = torch.isnan(a)
    right = torch.isnan(b)
    if not torch.equal(left, right):
        return False
    finite = ~left
    return torch.equal(_bits(a)[finite], _bits(b)[finite])


# ----------------------------------------------------------------------------------------------
# Memory accounting
# ----------------------------------------------------------------------------------------------


def test_prepacked_form_is_four_bytes_per_stored_code_byte():
    """The 4x figure is a contract, not an estimate: assert it exactly."""
    n, k = 128, 256
    costs = v2.prepacked_nbytes(n, k)
    assert costs["words"] == 4 * n * k, "the decode words are four bytes per code byte"
    assert costs["source"] == n * k + (n // BLOCK) * (k // BLOCK)
    assert costs["valid"] == n
    assert costs["scale_powers"] == (n // BLOCK) * (k // BLOCK) * 4
    assert costs["total"] == costs["words"] + costs["scale_powers"] + costs["valid"]
    # total is the words plus a small overhead, i.e. just over 4x the code bytes.
    assert costs["words"] < costs["total"] < costs["words"] * 1.02


def test_the_released_wkv_geometry_costs_the_documented_629_mib():
    """The number a caller has to budget for, computed rather than discovered."""
    costs = v2.prepacked_nbytes(25600, 6144)
    assert costs["words"] == 629_145_600
    # "source" is the stored weight bytes: codes plus the E8M0 scale grid.
    assert costs["source"] == 25600 * 6144 + (25600 // 32) * (6144 // 32)
    # 629,145,600 bytes of words is 600 MiB, the figure the module docstring quotes.
    assert costs["words"] == 600 * 1024 * 1024


def test_prepack_max_bytes_refuses_before_allocating(cache):
    n, k = 128, 256
    b_codes = _codes(n, k, seed=1)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=2)
    needed = v2.prepacked_nbytes(n, k)["total"]
    with pytest.raises(ValueError, match=rf"{needed}.*max_bytes is {needed - 1}"):
        v2.prepack_fp8_weight(b_codes, b_scale, max_bytes=needed - 1, cache_dir=cache)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, max_bytes=needed, cache_dir=cache)
    assert weight.bytes["total"] == needed


def test_reported_byte_accounting_matches_the_allocated_tensors(cache):
    n, k = 96, 128
    weight = v2.prepack_fp8_weight(_codes(n, k, seed=3), _scales(n // BLOCK, k // BLOCK, seed=4),
                                   cache_dir=cache)
    assert weight.bytes["words"] == weight.words.numel() * weight.words.element_size()
    assert weight.bytes["scale_powers"] == weight.scale_powers.numel() * 4
    assert weight.bytes["valid"] == n
    assert weight.bytes["total"] == v2.prepacked_nbytes(n, k)["total"]
    assert weight.words.dtype == torch.uint32 and weight.words.is_contiguous()
    assert not weight.words.requires_grad


# ----------------------------------------------------------------------------------------------
# Bit-exactness against the frozen v1 kernel
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_rows", [32, 1])
@pytest.mark.parametrize("threads", [1, 2, 8])
@pytest.mark.parametrize("variant", ["auto", "scalar"])
def test_every_float32_bit_matches_v1(cache, block_rows, threads, variant):
    n, k = 128, 256
    b_codes = _codes(n, k, seed=11)
    rows = n // BLOCK if block_rows == BLOCK else n
    b_scale = _scales(rows, k // BLOCK, seed=12)
    a_codes = _row_codes(k, seed=13)
    a_scale = _scales(1, k // BLOCK, seed=14)

    weight = v2.prepack_fp8_weight(b_codes, b_scale, block_rows=block_rows, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale,
                                 block_rows=block_rows, threads=1, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=threads,
                                         variant=variant, cache_dir=cache)
    assert _same(actual, expected), "prepacked GEMV disagrees with v1 bit for bit"


def test_the_result_does_not_depend_on_the_thread_count_or_variant(cache):
    n, k = 256, 512
    b_codes = _codes(n, k, seed=21)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=22)
    a_codes = _row_codes(k, seed=23)
    a_scale = _scales(1, k // BLOCK, seed=24)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    reference = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=1,
                                            variant="scalar", cache_dir=cache)
    for threads in (1, 3, 8):
        for variant in ("auto", "scalar"):
            actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=threads,
                                                 variant=variant, cache_dir=cache)
            assert _same(actual, reference)


def test_full_thread_count_is_bit_exact(cache):
    """The real decode uses every core the host has; the answer must not move."""
    threads = os.cpu_count() or 1
    n, k = 512, 256
    b_codes = _codes(n, k, seed=31)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=32)
    a_codes = _row_codes(k, seed=33)
    a_scale = _scales(1, k // BLOCK, seed=34)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, threads=threads,
                                 cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=threads,
                                         cache_dir=cache)
    assert _same(actual, expected)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 17, 31, 33, 63, 65, 127])
def test_per_row_grid_tails_match_v1(cache, n):
    """Every column count, including the ones the vector path cannot cover evenly."""
    k = 96
    b_codes = _codes(n, k, seed=41 + n)
    b_scale = _scales(n, k // BLOCK, seed=141 + n)
    a_codes = _row_codes(k, seed=241 + n)
    a_scale = _scales(1, k // BLOCK, seed=341 + n)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, block_rows=1, cache_dir=cache)
    for threads in (1, 4):
        expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, block_rows=1,
                                     threads=threads, cache_dir=cache)
        actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=threads,
                                             cache_dir=cache)
        assert _same(actual, expected), f"N = {n} disagreed"


def test_short_k_and_single_block_match_v1(cache):
    for k in (32, 64, 96):
        n = 64
        b_codes = _codes(n, k, seed=51 + k)
        b_scale = _scales(n // BLOCK, k // BLOCK, seed=61 + k)
        a_codes = _row_codes(k, seed=71 + k)
        a_scale = _scales(1, k // BLOCK, seed=81 + k)
        weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
        expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
        assert _same(v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache),
                     expected), f"K = {k} disagreed"


def test_every_code_class_including_signed_zeros_and_subnormals(cache):
    """Signed zeros, subnormals and the format's largest values, in weights and activation.

    The eight leading weight columns are single-code families, so a column exists
    for each interesting behaviour: both signed zeros, the smallest subnormals, the
    largest finite magnitudes, and an alternating-zero column. The activation row
    carries the same variety, including signed zeros in both halves -- the
    sentinel-anchor path -- and a subnormal in each half.
    """
    k, n = 64, 128
    b_codes = _codes(n, k, seed=91)
    families = (0x00, 0x80, 0x01, 0x81, 0x7E, 0xFE, 0x38, 0xB8)
    for column, byte in enumerate(families):
        b_codes[column, :] = byte
    b_codes[7, 0::2] = 0x00
    b_codes[7, 1::2] = 0x80
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=92)
    a_codes = torch.zeros((1, k), dtype=torch.uint8)
    a_codes[0, :16] = torch.tensor(
        [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
         0x0E, 0x8E, 0x38, 0xB8, 0x7E, 0xFE, 0x40, 0xC0],
        dtype=torch.uint8,
    )
    a_codes[0, 16:32] = torch.tensor(
        [0x00, 0x80, 0x00, 0x80, 0x02, 0x82, 0x03, 0x83,
         0x0F, 0x8F, 0x41, 0xC1, 0x7D, 0xFD, 0x3E, 0xBE],
        dtype=torch.uint8,
    )
    a_scale = _scales(1, k // BLOCK, seed=93)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert _same(actual, expected)
    # The single-code columns the families created are individually meaningful:
    assert float(actual[0, 0]) == 0.0 and float(actual[0, 1]) == 0.0, (
        "a signed-zero weight column is exactly zero, in both signs"
    )
    assert bool(torch.isfinite(actual).all())


def test_an_all_zero_block_hits_the_identity_window(cache):
    """A block of signed zeros has no anchor; both kernels take the (1, 1) window."""
    k = 64
    n = 64
    b_codes = torch.zeros((n, k), dtype=torch.uint8)
    b_codes[:, 1::2] = 0x80  # the other signed zero
    a_codes = torch.zeros((1, k), dtype=torch.uint8)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=101)
    a_scale = _scales(1, k // BLOCK, seed=102)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert _same(actual, expected)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_extreme_scale_bytes_are_exact(cache):
    """Byte 0 is a subnormal scale (2**-127) and 254 is the largest finite grid value."""
    n, k = 64, 64
    b_codes = _codes(n, k, seed=111)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=112)
    b_scale[0, 0] = 0
    b_scale[0, 1] = 254
    b_scale[1, 0] = 0
    a_codes = _row_codes(k, seed=113)
    a_scale = _scales(1, k // BLOCK, seed=114)
    a_scale[0, 0] = 0
    a_scale[0, 1] = 254
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert _same(actual, expected)
    # 2**127 x a non-zero total overflows binary32 in both kernels, which is the
    # reference's own behaviour rather than an error; the assertion is agreement.
    assert not bool(torch.isnan(actual).any())


# ----------------------------------------------------------------------------------------------
# Non-finite behaviour
# ----------------------------------------------------------------------------------------------


def test_a_nan_weight_column_is_nan_and_does_not_poison_others(cache):
    n, k = 64, 64
    b_codes = _codes(n, k, seed=121)
    b_codes[3, 5] = 0x7F  # NaN code in column 3
    b_codes[9, 0] = 0xFF  # negative NaN code in column 9
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=122)
    a_codes = _row_codes(k, seed=123)
    a_scale = _scales(1, k // BLOCK, seed=124)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    assert weight.valid_columns == n - 2, "the prepack resolves NaN columns once"

    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert _same(actual, expected)
    assert bool(torch.isnan(actual[0, 3])) and bool(torch.isnan(actual[0, 9]))
    others = [index for index in range(n) if index not in (3, 9)]
    assert bool(torch.isfinite(actual[0, others]).all())


def test_a_nan_activation_makes_every_column_nan(cache):
    n, k = 64, 64
    b_codes = _codes(n, k, seed=131)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=132)
    a_codes = _row_codes(k, seed=133)
    a_codes[0, 7] = 0x7F
    a_scale = _scales(1, k // BLOCK, seed=134)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert bool(torch.isnan(expected).all()) and bool(torch.isnan(actual).all())


# ----------------------------------------------------------------------------------------------
# Operand contract: the refusals are v1's, because the validators are imported from it
# ----------------------------------------------------------------------------------------------


def test_a_scale_byte_255_is_refused_by_name(cache):
    n, k = 64, 64
    b_codes = _codes(n, k, seed=141)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=142)
    b_scale[0, 0] = 255
    with pytest.raises(ValueError):
        v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)


def test_an_inexact_float_scale_is_refused(cache):
    """An FP32 grid must be exact powers of two; anything else is refused, not rounded."""
    n, k = 64, 64
    b_codes = _codes(n, k, seed=151)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 1.0, dtype=torch.float32)
    b_scale[0, 0] = 1.5
    with pytest.raises(ValueError):
        v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)


def test_exact_float32_scales_are_accepted_and_match_the_byte_form(cache):
    """The FP32 form the CPU quantizer returns is the same operand as the byte form."""
    n, k = 64, 128
    b_codes = _codes(n, k, seed=161)
    b_scale_bytes = _scales(n // BLOCK, k // BLOCK, seed=162)
    b_scale_float = torch.pow(
        torch.tensor(2.0, dtype=torch.float32),
        (b_scale_bytes.to(torch.int32) - 127).to(torch.float32),
    )
    a_codes = _row_codes(k, seed=163)
    a_scale = _scales(1, k // BLOCK, seed=164)
    weight_bytes = v2.prepack_fp8_weight(b_codes, b_scale_bytes, cache_dir=cache)
    weight_float = v2.prepack_fp8_weight(b_codes, b_scale_float, cache_dir=cache)
    assert torch.equal(weight_bytes.scale_powers, weight_float.scale_powers)
    assert _same(v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight_bytes, cache_dir=cache),
                 v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight_float, cache_dir=cache))


@pytest.mark.parametrize("block_rows", [0, 2, 16, 31, 33, 64])
def test_an_invalid_block_rows_is_refused(cache, block_rows):
    with pytest.raises(ValueError, match="block_rows must be 1 or 32"):
        v2.prepack_fp8_weight(_codes(64, 64, seed=171),
                              _scales(2, 2, seed=172), block_rows=block_rows, cache_dir=cache)


def test_a_square_grid_needs_n_divisible_by_32(cache):
    with pytest.raises(ValueError, match="divisible by 32"):
        v2.prepack_fp8_weight(_codes(40, 64, seed=181), _scales(1, 2, seed=182), cache_dir=cache)


def test_mismatched_scale_shape_is_refused(cache):
    b_codes = _codes(64, 128, seed=191)
    with pytest.raises(ValueError, match="weight scales must have shape"):
        v2.prepack_fp8_weight(b_codes, _scales(2, 2, seed=192), cache_dir=cache)


def test_a_multi_row_activation_is_refused_by_name(cache):
    """M = 1 is the contract; a prompt belongs on the batched block GEMM."""
    n, k = 64, 64
    weight = v2.prepack_fp8_weight(_codes(n, k, seed=201), _scales(n // BLOCK, k // BLOCK, seed=202),
                                   cache_dir=cache)
    a_codes = _row_codes(k, seed=203).repeat(2, 1)
    a_scale = _scales(1, k // BLOCK, seed=204)
    with pytest.raises(ValueError, match="M = 1 GEMV"):
        v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)


def test_a_width_mismatch_between_row_and_weight_is_refused(cache):
    weight = v2.prepack_fp8_weight(_codes(64, 64, seed=211), _scales(2, 2, seed=212),
                                   cache_dir=cache)
    with pytest.raises(ValueError, match="not read from the same checkpoint"):
        v2.fp8_block_gemv_prepacked(_row_codes(128, seed=213), _scales(1, 4, seed=214),
                                    weight, cache_dir=cache)


def test_bad_threads_and_variant_are_refused(cache):
    w = v2.prepack_fp8_weight(_codes(64, 64, seed=221), _scales(2, 2, seed=222), cache_dir=cache)
    a_codes, a_scale = _row_codes(64, seed=223), _scales(1, 2, seed=224)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="threads must be an integer >= 1"):
            v2.fp8_block_gemv_prepacked(a_codes, a_scale, w, threads=bad, cache_dir=cache)
    with pytest.raises(ValueError, match="variant must be one of"):
        v2.fp8_block_gemv_prepacked(a_codes, a_scale, w, variant="neon", cache_dir=cache)


def test_a_non_prepacked_weight_is_refused_by_type(cache):
    with pytest.raises(TypeError, match="prepack_fp8_weight returned"):
        v2.fp8_block_gemv_prepacked(_row_codes(64, seed=231), _scales(1, 2, seed=232),
                                    torch.zeros(4), cache_dir=cache)


def test_out_must_be_a_matching_contiguous_cpu_fp32_buffer(cache):
    n, k = 64, 64
    w = v2.prepack_fp8_weight(_codes(n, k, seed=241), _scales(n // BLOCK, k // BLOCK, seed=242),
                              cache_dir=cache)
    a_codes, a_scale = _row_codes(k, seed=243), _scales(1, k // BLOCK, seed=244)
    with pytest.raises(TypeError, match="expected torch.float32"):
        v2.fp8_block_gemv_prepacked(a_codes, a_scale, w, out=torch.empty((1, n), dtype=torch.float16),
                                    cache_dir=cache)
    with pytest.raises(ValueError, match="shape"):
        v2.fp8_block_gemv_prepacked(a_codes, a_scale, w, out=torch.empty((1, n + 1)),
                                    cache_dir=cache)
    with pytest.raises(ValueError, match="must not require grad"):
        v2.fp8_block_gemv_prepacked(
            a_codes, a_scale, w,
            out=torch.empty((1, n), requires_grad=True), cache_dir=cache)


def test_out_is_filled_and_returned(cache):
    n, k = 64, 64
    w = v2.prepack_fp8_weight(_codes(n, k, seed=251), _scales(n // BLOCK, k // BLOCK, seed=252),
                              cache_dir=cache)
    a_codes, a_scale = _row_codes(k, seed=253), _scales(1, k // BLOCK, seed=254)
    out = torch.full((1, n), 7.0)
    returned = v2.fp8_block_gemv_prepacked(a_codes, a_scale, w, out=out, cache_dir=cache)
    assert returned is out and not bool(torch.equal(out, torch.full((1, n), 7.0)))


# ----------------------------------------------------------------------------------------------
# Provenance: the prepacked form is self-contained, and a stale source is reported
# ----------------------------------------------------------------------------------------------


def test_mutating_the_source_after_prepack_is_detected_and_does_not_change_the_answer(cache):
    n, k = 64, 64
    b_codes = _codes(n, k, seed=261)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=262)
    a_codes, a_scale = _row_codes(k, seed=263), _scales(1, k // BLOCK, seed=264)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    before = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache).clone()
    weight.verify_sources()  # clean at this point

    b_codes[0, 0] ^= 0x01
    with pytest.raises(RuntimeError, match="was modified after"):
        weight.verify_sources()
    after = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    assert torch.equal(before, after), (
        "the kernel reads the prepacked words, so a mutated source cannot change an answer"
    )


def test_an_inference_tensor_source_is_reported_unverifiable(cache):
    """An inference tensor tracks no version; that is reported, not silently passed."""
    n, k = 64, 64
    with torch.inference_mode():
        b_codes = _codes(n, k, seed=271)
        b_scale = _scales(n // BLOCK, k // BLOCK, seed=272)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    versions = weight.source_versions()
    assert versions["codes"] is None or isinstance(versions["codes"], int)
    if versions["codes"] is None:
        with pytest.raises(RuntimeError, match="does not track a version counter"):
            weight.verify_sources()


def test_prepack_does_not_modify_the_source_tensors(cache):
    n, k = 64, 64
    b_codes = _codes(n, k, seed=281)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=282)
    codes_before = b_codes.clone()
    scales_before = b_scale.clone()
    v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    assert torch.equal(b_codes, codes_before) and torch.equal(b_scale, scales_before)


# ----------------------------------------------------------------------------------------------
# ABI and artifact identity
# ----------------------------------------------------------------------------------------------


def test_the_abi_is_version_two_and_its_artifact_is_named_and_bound_so(cache):
    assert v2.ABI_VERSION == 2
    path = v2.build(cache_dir=cache)
    assert "abi2" in path.name, "the artifact name has to carry the ABI version"
    library = v2.load(cache_dir=cache)
    library.relic_cpu_block_gemv_prepacked_abi_version.restype = ctypes.c_int32
    assert int(library.relic_cpu_block_gemv_prepacked_abi_version()) == v2.ABI_VERSION


def test_the_two_operators_never_share_an_artifact_or_a_fingerprint(cache):
    v2_path = v2.build(cache_dir=cache)
    v1_path = v1.build(cache_dir=cache)
    assert v2_path != v1_path
    assert "prepacked" in v2_path.name and "prepacked" not in v1_path.name


def test_the_fingerprint_covers_the_source_and_the_build_flags(cache):
    """A differently compiled kernel is a different kernel, so it must not reuse a cache entry."""
    baseline = v2.compiled_library_path(cache, extra_flags=[])
    assert v2.compiled_library_path(cache, extra_flags=[]) == baseline
    assert v2.compiled_library_path(cache, extra_flags=["-DFOO=1"]) != baseline
    assert v2._fingerprint([]) != v2._fingerprint(["-DFOO=1"])


def test_a_corrupt_artifact_at_the_expected_path_fails_loudly(cache):
    """A truncated or foreign file at the content-addressed path is not silently used."""
    import pathlib

    private = tempfile.mkdtemp(prefix="relic-cpu-gemv-corrupt-", dir=pathlib.Path(cache).parent)
    try:
        # The same flags on both calls: the artifact identity is source + flags, so a
        # corrupt file only has to defeat the build that would look for it.
        target = v2.compiled_library_path(private, extra_flags=[])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not a shared library")
        with pytest.raises(Exception) as raised:
            v2.load(cache_dir=private, extra_flags=[])
        assert not isinstance(raised.value, (SystemExit, KeyboardInterrupt))
    finally:
        shutil.rmtree(private, ignore_errors=True)


def test_the_operator_is_opt_in(cache, monkeypatch):
    """No cache directory means no build: a checkout that does not ask pays nothing."""
    for name in ("RELIC_CPU_GEMV_PREPACKED_CACHE", "RELIC_CPU_GEMV_CACHE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="opt-in"):
        v2.cache_directory(None)
    assert v2.cache_directory(cache) == Path(cache)


# ----------------------------------------------------------------------------------------------
# Independent exact-rational reference
# ----------------------------------------------------------------------------------------------


def _f32(value: Fraction) -> Fraction:
    """Round a rational to binary32 (nearest, ties to even) through ``struct``.

    Every value this reference produces is a dyadic rational with at most 24
    significant bits, so the intermediate ``float`` conversion is exact and the only
    rounding is ``struct``'s own binary32 one.
    """
    return Fraction(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _pow2(exponent: int) -> Fraction:
    return Fraction(2) ** exponent


def _trunc(value: Fraction) -> int:
    """trunc toward zero, as ``static_cast<int32_t>``/``std::trunc`` do here."""
    return int(value) if value >= 0 else -int(-value)


def _code_term(byte: int) -> tuple[int, int]:
    """``(significand, power)`` with sign folded in: value = significand * 2**power."""
    field = (byte >> 3) & 0xF
    mantissa = byte & 7
    sign = -1 if byte & 0x80 else 1
    if field == 0:
        return sign * mantissa, -9
    return sign * (8 + mantissa), field - 10


def _reference_column(a_codes, a_scale, b_codes, b_scale, column: int, block_rows: int) -> float:
    """One output column, from the reduction's definition, in exact rationals.

    This is deliberately a second derivation rather than a transcription of either
    kernel: it states the reference's rule (two 16-term halves, a truncating window
    per half, a running carry into the second window, one final renormalisation, then
    activation scale and weight scale in that order) over exact ``Fraction`` values,
    and lets :func:`_f32` supply the only rounding.
    """
    k = len(a_codes)
    blocks = k // BLOCK
    accumulator = Fraction(0)
    for block in range(blocks):
        base = block * BLOCK
        products = []
        for lane in range(BLOCK):
            a_significand, a_power = _code_term(int(a_codes[base + lane]))
            b_significand, b_power = _code_term(int(b_codes[column][base + lane]))
            # a_value * b_value, exactly: significands multiply, powers add.
            products.append(
                Fraction(a_significand * b_significand) * _pow2(a_power + b_power)
            )
        anchor_first = max(
            _activation_nominal(int(a_codes[base + lane]))
            + _weight_biased_nominal(int(b_codes[column][base + lane]))
            for lane in range(BLOCK // 2)
        ) - 1000
        inverse_first, step_first = _cut(anchor_first)
        whole_first = sum(
            _trunc(products[lane] * inverse_first) for lane in range(BLOCK // 2)
        )
        total_first = _f32(Fraction(whole_first) * step_first)
        running = _floor_exponent(total_first) if total_first != 0 else -1000000
        anchor_second = max(
            _activation_nominal(int(a_codes[base + lane]))
            + _weight_biased_nominal(int(b_codes[column][base + lane]))
            for lane in range(BLOCK // 2, BLOCK)
        ) - 1000
        inverse_second, step_second = _cut(max(anchor_second, running))
        whole_second = sum(
            _trunc(products[lane] * inverse_second) for lane in range(BLOCK // 2, BLOCK)
        )
        total = _f32(
            (Fraction(whole_second) + _trunc(total_first * inverse_second)) * step_second
        )
        if total != 0:
            step = _pow2(_floor_exponent(total) - 13)
            total = _trunc(total / step) * step
        activation_scale = _pow2(int(a_scale[block]) - 127)
        scale_row = column if block_rows == 1 else column // block_rows
        weight_scale = _pow2(int(b_scale[scale_row][block]) - 127)
        accumulator += _f32(_f32(total * activation_scale) * weight_scale)
    return float(accumulator)


def _activation_nominal(byte: int) -> int:
    if byte & 0x7F == 0:
        return -1000
    field = (byte >> 3) & 0xF
    return field - 7 if field >= 1 else -6


def _weight_biased_nominal(byte: int) -> int:
    nominal = _activation_nominal(byte)
    return nominal + 1000


def _cut(anchor: int) -> tuple[Fraction, Fraction]:
    if anchor < -500:
        return Fraction(1), Fraction(1)
    return _pow2(13 - anchor), _pow2(anchor - 13)


def _floor_exponent(value: Fraction) -> int:
    exponent = 0
    magnitude = abs(value)
    while magnitude >= 2:
        magnitude /= 2
        exponent += 1
    while magnitude < 1:
        magnitude *= 2
        exponent -= 1
    return exponent


def test_an_independent_exact_rational_reference_agrees(cache):
    """Re-derive a small case from the reduction's definition, not from either kernel."""
    n, k = 32, 64
    b_codes = _codes(n, k, seed=301)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=302)
    a_codes = _row_codes(k, seed=303)
    a_scale = _scales(1, k // BLOCK, seed=304)
    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, cache_dir=cache)
    for column in range(n):
        expected = _reference_column(a_codes[0].tolist(), a_scale[0].tolist(),
                                     b_codes.tolist(), b_scale.tolist(), column, BLOCK)
        assert struct.pack("<f", expected) == struct.pack("<f", float(actual[0, column])), (
            f"column {column}: exact reference {expected!r} != kernel {float(actual[0, column])!r}"
        )


# ----------------------------------------------------------------------------------------------
# The real geometry
# ----------------------------------------------------------------------------------------------


def test_the_released_wkv_geometry_is_bit_exact_and_reports_its_cost(cache):
    """N = 25600, K = 6144: the shape the host Engram projection actually runs.

    This is the case the byte accounting is about, so it is measured here rather
    than extrapolated -- the prepack allocates about 800 MiB and the comparison
    covers every output column.
    """
    n, k = 25600, 6144
    b_codes = _codes(n, k, seed=311)
    b_scale = _scales(n // BLOCK, k // BLOCK, seed=312)
    a_codes = _row_codes(k, seed=313)
    a_scale = _scales(1, k // BLOCK, seed=314)
    threads = os.cpu_count() or 1

    weight = v2.prepack_fp8_weight(b_codes, b_scale, cache_dir=cache)
    assert weight.bytes["words"] == 4 * b_codes.numel()
    assert weight.valid_columns == n
    expected = v1.fp8_block_gemv(a_codes, a_scale, b_codes, b_scale, threads=threads,
                                 cache_dir=cache)
    actual = v2.fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=threads,
                                         cache_dir=cache)
    assert _same(actual, expected)
    assert bool(torch.isfinite(actual).all())