"""Acceptance tests for the compiled host CPU FP8 block-32 GEMV.

The operator exists for the **host** Engram ``wkv`` projection on the Ascend path,
so its correctness claim is narrow and checkable: given the same raw e4m3 codes and
the same original E8M0 scale grids, it produces the bits the accepted Torch oracle
in ``block_gemm.py`` produces. That oracle is the reference here -- it is the same
reduction the runtime performs -- and the comparison is ``torch.equal`` on the
FP32 bit patterns, never a tolerance.

Nothing here needs a GPU, a checkpoint or a network. A C++ compiler does have to
be present, because the kernel is compiled from ``csrc/cpu_block_gemv.cpp`` into a
temporary directory; without one the whole module skips with the reason, and a
skip is not a pass.
"""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv as gemv  # noqa: E402
from relic_core.kernels.block_gemm import fp8_block_gemm  # noqa: E402

BLOCK = 32


@pytest.fixture(scope="session")
def cache():
    """A session-private build directory, or a skip that names the toolchain problem."""
    directory = tempfile.mkdtemp(prefix="relic-cpu-block-gemv-")
    try:
        gemv.build(cache_dir=directory)
    except RuntimeError as error:
        pytest.skip(f"the compiled host CPU GEMV cannot be built here: {error}")
    return directory


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


def reference(a_codes, a_scale, b_codes, b_scale, block_rows=BLOCK):
    return fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows, backend="torch")


def assert_same_bits(actual, expected, note=""):
    assert actual.shape == expected.shape, f"{note}: shape {actual.shape} != {expected.shape}"
    if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
        differing = int((actual.view(torch.int32) != expected.view(torch.int32)).sum())
        raise AssertionError(f"{note}: {differing} of {expected.numel()} outputs differ in bits")


# -- the arithmetic ----------------------------------------------------------------------------


def test_random_finite_codes_match_the_torch_oracle_bit_for_bit(cache):
    a = finite_codes((1, 256), 1)
    b = finite_codes((128, 256), 2)
    sa = grid_bytes(1, 8, 3)
    sb = grid_bytes(4, 8, 4)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "random"
    )


@pytest.mark.parametrize("byte", [0, 1, 126, 127, 128, 200, 254])
def test_every_scale_endpoint_is_an_exact_rescale(cache, byte):
    """The grid values that bracket the format: 2**-127, 1.0, 2**127 and neighbours.

    A power-of-two scale can only move an exponent, so these are where a kernel
    that built its powers by ``exp2`` approximation or that flushed subnormals
    would show up.
    """
    a = finite_codes((1, 128), 5)
    b = finite_codes((64, 128), 6)
    sa = torch.full((1, 4), byte, dtype=torch.uint8)
    sb = torch.full((2, 4), byte, dtype=torch.uint8)
    got = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    assert_same_bits(got, reference(a, sa, b, sb), f"scale byte {byte}")


def test_a_zero_activation_block_has_no_anchor_and_matches(cache):
    """An all-zero half is the sentinel branch: it cuts nothing instead of using a live anchor."""
    a = torch.zeros((1, 128), dtype=torch.uint8)
    b = finite_codes((64, 128), 7)
    sa = torch.full((1, 4), 127, dtype=torch.uint8)
    sb = torch.full((2, 4), 127, dtype=torch.uint8)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "zero activation"
    )


def test_negative_zero_codes_match(cache):
    """``0x80`` is the other zero encoding; it carries no nominal exponent either."""
    a = torch.full((1, 128), 0x80, dtype=torch.uint8)
    b = finite_codes((64, 128), 8)
    sa = torch.full((1, 4), 127, dtype=torch.uint8)
    sb = torch.full((2, 4), 127, dtype=torch.uint8)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "negative zero"
    )


def test_zero_weight_rows_match(cache):
    b = finite_codes((64, 128), 9)
    b[3] = 0
    b[17] = 0x80
    a = finite_codes((1, 128), 10)
    sa = grid_bytes(1, 4, 11)
    sb = grid_bytes(2, 4, 12)
    assert_same_bits(gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "zero rows")


def test_subnormal_e4m3_codes_match(cache):
    """Only the subnormal encodings, where the nominal exponent is floored at -6 rather than read.

    An e4m3 subnormal of magnitude below 2**-6 has a true exponent of -7 or lower,
    and every one of them contributes -6 to its half's anchor. A kernel that used
    the value's own exponent would land on a different window here and nowhere else.
    """
    a = finite_codes((1, 128), 13, low=0, high=8)
    b = finite_codes((64, 128), 14, low=0, high=8)
    sa = torch.full((1, 4), 120, dtype=torch.uint8)
    sb = torch.full((2, 4), 120, dtype=torch.uint8)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "subnormals"
    )


def test_the_second_half_carry_and_final_renormalisation_are_reproduced(cache):
    """Codes chosen so the halves' anchors differ and the sum carries past its window.

    The first half is small and the second large, which is the case where the
    running sum has to be rescaled onto the second window and the result then
    renormalised once more. Reproducing the first half's window alone is not enough.
    """
    a = torch.zeros(1, 128, dtype=torch.uint8)
    b = torch.zeros(64, 128, dtype=torch.uint8)
    # Small magnitudes in the first half of each K block, large in the second.
    a[0, 0:16] = 0x28  # 2**-5
    a[0, 16:32] = 0x78  # 960.0
    for block in range(4):
        b[:, block * BLOCK + 0:block * BLOCK + 16] = 0x30
        b[:, block * BLOCK + 16:block * BLOCK + 32] = 0x70
    sa = torch.full((1, 4), 127, dtype=torch.uint8)
    sb = torch.full((2, 4), 127, dtype=torch.uint8)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache), reference(a, sa, b, sb), "carry"
    )


def test_per_row_weight_grid_matches(cache):
    a = finite_codes((1, 128), 15)
    b = finite_codes((64, 128), 16)
    sa = grid_bytes(1, 4, 17)
    sb = grid_bytes(64, 4, 18)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, block_rows=1, cache_dir=cache),
        reference(a, sa, b, sb, block_rows=1),
        "per-row grid",
    )


def test_the_real_k_width_matches_across_many_k_blocks(cache):
    """K = 6144 is the Engram projection's width: 192 sequential K-block accumulations."""
    a = finite_codes((1, 6144), 19)
    b = finite_codes((512, 6144), 20)
    sa = grid_bytes(1, 192, 21)
    sb = grid_bytes(16, 192, 22)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, threads=2, cache_dir=cache),
        reference(a, sa, b, sb),
        "K=6144",
    )


# -- threads -----------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 8, 32])
def test_the_result_does_not_depend_on_the_thread_count(cache, threads):
    a = finite_codes((1, 256), 23)
    b = finite_codes((256, 256), 24)
    sa = grid_bytes(1, 8, 25)
    sb = grid_bytes(8, 8, 26)
    single = gemv.fp8_block_gemv(a, sa, b, sb, threads=1, cache_dir=cache)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, threads=threads, cache_dir=cache), single, f"{threads} threads"
    )


def test_more_threads_than_columns_is_allowed(cache):
    a = finite_codes((1, 64), 27)
    b = finite_codes((3, 64), 28)
    sa = grid_bytes(1, 2, 29)
    sb = grid_bytes(3, 2, 30)
    assert_same_bits(
        gemv.fp8_block_gemv(a, sa, b, sb, block_rows=1, threads=16, cache_dir=cache),
        reference(a, sa, b, sb, block_rows=1),
        "oversubscribed",
    )


# -- non-finite ---------------------------------------------------------------------------------


def test_a_nan_activation_code_makes_every_output_nan(cache):
    a = finite_codes((1, 128), 31)
    b = finite_codes((64, 128), 32)
    sa = grid_bytes(1, 4, 33)
    sb = grid_bytes(2, 4, 34)
    a[0, 5] = 0x7F
    got = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    expected = reference(a, sa, b, sb)
    assert bool(torch.isnan(expected).all())
    assert bool(torch.isnan(got).all())


def test_a_nan_weight_code_makes_only_its_column_nan(cache):
    a = finite_codes((1, 128), 35)
    b = finite_codes((64, 128), 36)
    sa = grid_bytes(1, 4, 37)
    sb = grid_bytes(2, 4, 38)
    b[63, 7] = 0xFF
    got = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    expected = reference(a, sa, b, sb)
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert bool(torch.isnan(got[0, 63]))
    assert int(torch.isnan(got).sum()) == 1


def test_scale_byte_255_is_refused_by_name_rather_than_becoming_an_infinity(cache):
    """The one raw byte that is not a finite grid value has no agreed meaning.

    ``2 ** 128`` does not fit binary32, so accepting the byte would mean answering
    with an infinity (the runtime's exponent arithmetic) or a NaN (the Torch
    oracle's table) depending on which reference was consulted. Neither is a
    number a checkpoint or quantizer produces, so the operator refuses it instead
    of picking a side, and the message names the byte.
    """
    a = finite_codes((1, 64), 39)
    b = finite_codes((32, 64), 40)
    good = torch.full((1, 2), 127, dtype=torch.uint8)
    bad = good.clone()
    bad[0, 1] = 255
    with pytest.raises(ValueError, match="0xFF"):
        gemv.fp8_block_gemv(a, bad, b, good, cache_dir=cache)
    with pytest.raises(ValueError, match="finite E8M0"):
        gemv.fp8_block_gemv(a, good, b, bad, cache_dir=cache)
    # The same grid through the FP32 path cannot even express 2**128, so the two
    # storage forms agree on refusing rather than disagreeing on a value.
    with pytest.raises(ValueError, match="not an exact original E8M0 grid"):
        gemv.fp8_block_gemv(a, good, b, torch.full((1, 2), float("inf")), cache_dir=cache)


def test_every_representable_finite_scale_byte_is_accepted(cache):
    """Byte 255 is the only refused one; 0 through 254 are all finite grids."""
    a = finite_codes((1, 64), 59)
    b = finite_codes((32, 64), 60)
    for byte in (0, 1, 126, 127, 128, 253, 254):
        sa = torch.full((1, 2), byte, dtype=torch.uint8)
        sb = torch.full((1, 2), byte, dtype=torch.uint8)
        got = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
        assert_same_bits(got, reference(a, sa, b, sb), f"scale byte {byte}")


# -- FP32 scale grids (what the CPU activation quantizer returns) -------------------------------


def fp32_grid_from_bytes(scale_bytes):
    """The exact FP32 value of each E8M0 byte, as the decoder would return it."""
    import math

    values = [math.ldexp(1.0, int(byte) - 127) for byte in scale_bytes.flatten().tolist()]
    return torch.tensor(values, dtype=torch.float32).reshape(scale_bytes.shape)


def test_an_exact_fp32_grid_produces_the_same_bits_as_its_bytes(cache):
    a = finite_codes((1, 256), 41)
    b = finite_codes((64, 256), 42)
    sa = grid_bytes(1, 8, 43, low=0, high=255)
    sb = grid_bytes(2, 8, 44, low=1, high=255)
    sa = torch.where(sa == 255, sa - 1, sa).to(torch.uint8)
    sb = torch.where(sb == 255, sb - 1, sb).to(torch.uint8)
    from_bytes = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    from_fp32 = gemv.fp8_block_gemv(
        a, fp32_grid_from_bytes(sa), b, fp32_grid_from_bytes(sb), cache_dir=cache
    )
    assert_same_bits(from_fp32, from_bytes, "fp32 grid")
    assert_same_bits(from_fp32, reference(a, sa, b, sb), "fp32 grid vs oracle")


def test_the_lowest_grid_value_is_a_binary32_subnormal_and_survives_the_conversion(cache):
    """``2 ** -127`` is the smallest original E8M0 scale and is a subnormal in FP32."""
    a = finite_codes((1, 64), 45)
    b = finite_codes((32, 64), 46)
    raw = torch.full((1, 2), 0, dtype=torch.uint8)
    sb_raw = torch.full((1, 2), 0, dtype=torch.uint8)
    fp32 = torch.tensor([[2.0**-127, 2.0**-127]], dtype=torch.float32)
    sb_fp32 = torch.tensor([[2.0**-127, 2.0**-127]], dtype=torch.float32)
    assert fp32.view(torch.int32)[0, 0].item() == 0x00400000
    assert_same_bits(
        gemv.fp8_block_gemv(a, fp32, b, sb_fp32, cache_dir=cache),
        gemv.fp8_block_gemv(a, raw, b, sb_raw, cache_dir=cache),
        "2**-127 grid",
    )


@pytest.mark.parametrize(
    "bad_value",
    [0.0, -1.0, 1.5, 3.0, float("inf"), float("nan"), 2.0**-128],
    ids=["zero", "negative", "1.5", "3.0", "inf", "nan", "2**-128"],
)
def test_an_fp32_scale_that_is_not_a_grid_value_is_refused_by_name(cache, bad_value):
    a = finite_codes((1, 64), 47)
    b = finite_codes((32, 64), 48)
    good = torch.full((1, 2), 1.0, dtype=torch.float32)
    bad = good.clone()
    bad[0, 1] = bad_value
    with pytest.raises(ValueError, match="not an exact original E8M0 grid"):
        gemv.fp8_block_gemv(a, bad, b, good, cache_dir=cache)
    with pytest.raises(ValueError, match="not an exact original E8M0 grid"):
        gemv.fp8_block_gemv(a, good, b, bad, cache_dir=cache)


# -- the interface contract ---------------------------------------------------------------------


def small_case(cache=None):
    a = finite_codes((1, 128), 49)
    b = finite_codes((64, 128), 50)
    sa = grid_bytes(1, 4, 51)
    sb = grid_bytes(2, 4, 52)
    return a, sa, b, sb


def test_a_batched_activation_is_refused_and_points_at_the_general_gemm(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(ValueError, match="M = 1"):
        gemv.fp8_block_gemv(torch.cat([a, a]), torch.cat([sa, sa]), b, sb, cache_dir=cache)


def test_unknown_block_rows_and_thread_counts_are_refused(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(ValueError, match="block_rows must be 1 or 32"):
        gemv.fp8_block_gemv(a, sa, b, sb, block_rows=16, cache_dir=cache)
    with pytest.raises(ValueError, match="threads must be"):
        gemv.fp8_block_gemv(a, sa, b, sb, threads=0, cache_dir=cache)
    with pytest.raises(ValueError, match="threads must be"):
        gemv.fp8_block_gemv(a, sa, b, sb, threads=True, cache_dir=cache)


def test_a_square_grid_needs_a_column_count_divisible_by_32(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(ValueError, match="divisible by 32"):
        gemv.fp8_block_gemv(a, sa, b[:50], grid_bytes(2, 4, 53), cache_dir=cache)


def test_shape_and_scale_grid_mismatches_are_refused(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(ValueError, match="same K"):
        gemv.fp8_block_gemv(a, sa, b[:, :64].contiguous(), sb, cache_dir=cache)
    with pytest.raises(ValueError, match="activation scales must have shape"):
        gemv.fp8_block_gemv(a, sa[:, :2], b, sb, cache_dir=cache)
    with pytest.raises(ValueError, match="weight scales must have shape"):
        gemv.fp8_block_gemv(a, sa, b, sb[:, :2].contiguous(), cache_dir=cache)


def test_a_non_contiguous_operand_is_refused(cache):
    a, sa, b, sb = small_case()
    wide = finite_codes((64, 256), 58)
    with pytest.raises(ValueError, match="contiguous"):
        gemv.fp8_block_gemv(a, sa, wide[:, :128], sb, cache_dir=cache)


def test_a_floating_scale_dtype_other_than_fp32_or_a_byte_view_is_refused(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(TypeError, match="activation scales"):
        gemv.fp8_block_gemv(a, sa.to(torch.float16), b, sb, cache_dir=cache)


def test_a_float_activation_is_refused_rather_than_silently_viewed(cache):
    a, sa, b, sb = small_case()
    with pytest.raises(TypeError, match="activation codes"):
        gemv.fp8_block_gemv(a.float(), sa, b, sb, cache_dir=cache)


def test_an_output_that_requires_grad_is_refused(cache):
    """Byte inputs cannot carry an autograd graph, but an FP32 output buffer can."""
    a, sa, b, sb = small_case()
    out = torch.empty((1, 64), dtype=torch.float32, requires_grad=True)
    with pytest.raises(ValueError, match="must not require grad"):
        gemv.fp8_block_gemv(a, sa, b, sb, out=out, cache_dir=cache)


def test_an_output_buffer_is_filled_in_place_and_validated(cache):
    a, sa, b, sb = small_case()
    out = torch.empty((1, 64), dtype=torch.float32)
    returned = gemv.fp8_block_gemv(a, sa, b, sb, out=out, cache_dir=cache)
    assert returned is out
    assert_same_bits(out, reference(a, sa, b, sb), "supplied out")
    with pytest.raises(TypeError, match="float32"):
        gemv.fp8_block_gemv(a, sa, b, sb, out=torch.empty((1, 64), dtype=torch.float16), cache_dir=cache)
    with pytest.raises(ValueError, match="shape"):
        gemv.fp8_block_gemv(a, sa, b, sb, out=torch.empty((1, 32)), cache_dir=cache)


def test_an_output_that_overlaps_an_input_is_refused(cache):
    a, sa, b, sb = small_case()
    overlapping = b[:64, :].view(torch.float32).reshape(1, -1)[:, :64]
    with pytest.raises(ValueError, match="overlap"):
        gemv.fp8_block_gemv(a, sa, b, sb, out=overlapping, cache_dir=cache)


# -- build and opt-in contract ------------------------------------------------------------------


def test_the_operator_is_opt_in_and_names_its_environment_variable(monkeypatch):
    monkeypatch.delenv(gemv.CACHE_ENV, raising=False)
    assert not gemv.fp8_block_gemv_available()
    with pytest.raises(RuntimeError, match=gemv.CACHE_ENV):
        gemv.cache_directory(None)


def test_the_artifact_name_is_content_addressed(cache):
    first = gemv.compiled_library_path(cache)
    assert first.is_file()
    # A different flag set is a different kernel and must not reuse the artifact.
    other = gemv.compiled_library_path(cache, extra_flags=["-DEXTRA=1"])
    assert other != first
    assert not other.is_file()


def test_building_twice_is_idempotent(cache):
    first = gemv.build(cache_dir=cache)
    second = gemv.build(cache_dir=cache)
    assert first == second and first.is_file()


def test_a_missing_compiler_is_reported_rather_than_falling_back():
    with pytest.raises(RuntimeError, match="not found"):
        gemv.build(cache_dir=tempfile.mkdtemp(), compiler="/nonexistent/compiler")


def test_a_named_compiler_is_honoured_instead_of_substituted(monkeypatch):
    monkeypatch.delenv(gemv.COMPILER_ENV, raising=False)
    with pytest.raises(RuntimeError, match="named by compiler= was not found"):
        gemv.build(cache_dir=tempfile.mkdtemp(), compiler="/nonexistent/compiler")
    monkeypatch.setenv(gemv.COMPILER_ENV, "/nonexistent/from-env")
    with pytest.raises(RuntimeError, match="named by RELIC_CPU_GEMV_CXX was not found"):
        gemv.build(cache_dir=tempfile.mkdtemp())


# -- the shape the Engram projection actually has -----------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RELIC_CPU_GEMV_FULL_SHAPE") != "1",
    reason="set RELIC_CPU_GEMV_FULL_SHAPE=1 to run the 25600x6144 case (a few seconds)",
)
def test_the_full_engram_projection_shape_runs_and_is_finite(cache):
    """The real ``wkv`` geometry: N = dim * (hc_mult + 1) = 25600, K = 6144, M = 1.

    Both Engram layers share this projection's shape, so this is the size the
    operator is for. It is opt-in because it reads 157 MiB of codes per call.
    """
    n, k = 25600, 6144
    a = finite_codes((1, k), 54)
    b = finite_codes((n, k), 55)
    sa = grid_bytes(1, k // BLOCK, 56)
    sb = grid_bytes(n // BLOCK, k // BLOCK, 57)
    single = gemv.fp8_block_gemv(a, sa, b, sb, threads=1, cache_dir=cache)
    assert single.shape == (1, n)
    assert bool(torch.isfinite(single).all())
    assert_same_bits(gemv.fp8_block_gemv(a, sa, b, sb, threads=4, cache_dir=cache), single, "full/4")
    assert_same_bits(gemv.fp8_block_gemv(a, sa, b, sb, threads=8, cache_dir=cache), single, "full/8")

# -- the storage forms the runtime actually hands over -------------------------------------------


def test_the_native_float8_dtypes_produce_the_same_bits_as_raw_bytes(cache):
    """The runtime holds activation codes as ``float8_e4m3fn`` and scale grids as
    ``float8_e8m0fnu``; checkpoints hand over the same bytes as ``uint8``.

    Both spellings are the same storage, so the operator has to treat them as one
    thing rather than casting either -- a numeric cast of an encoded byte is a
    different number, which is the bug this test exists to catch.
    """
    a = finite_codes((1, 256), 61)
    b = finite_codes((64, 256), 62)
    sa = grid_bytes(1, 8, 63, low=0, high=255)
    sb = grid_bytes(2, 8, 64, low=1, high=255)
    sa = torch.where(sa == 255, sa - 1, sa).to(torch.uint8)
    sb = torch.where(sb == 255, sb - 1, sb).to(torch.uint8)

    from_bytes = gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    from_float8 = gemv.fp8_block_gemv(
        a.view(torch.float8_e4m3fn),
        sa.view(torch.float8_e8m0fnu),
        b.view(torch.float8_e4m3fn),
        sb.view(torch.float8_e8m0fnu),
        cache_dir=cache,
    )
    assert_same_bits(from_float8, from_bytes, "float8 dtypes")
    assert_same_bits(from_bytes, reference(a, sa, b, sb), "float8 dtypes vs oracle")


def test_signed_int8_views_are_the_same_bytes(cache):
    a = finite_codes((1, 128), 65)
    b = finite_codes((64, 128), 66)
    sa = grid_bytes(1, 4, 67)
    sb = grid_bytes(2, 4, 68)
    assert_same_bits(
        gemv.fp8_block_gemv(a.view(torch.int8), sa.view(torch.int8), b.view(torch.int8),
                            sb.view(torch.int8), cache_dir=cache),
        gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache),
        "int8 views",
    )


def test_a_float8_scale_grid_still_refuses_byte_255(cache):
    """The refusal is about the byte, not about how the bytes were typed."""
    a = finite_codes((1, 64), 69)
    b = finite_codes((32, 64), 70)
    good = torch.full((1, 2), 127, dtype=torch.uint8)
    bad = good.clone()
    bad[0, 0] = 255
    with pytest.raises(ValueError, match="0xFF"):
        gemv.fp8_block_gemv(a, good, b, bad.view(torch.float8_e8m0fnu), cache_dir=cache)


def test_the_operands_are_not_modified(cache):
    """The operator writes only its output; an in-place scratch buffer would be a
    silent corruption of the checkpoint's mapping, which is read-only by contract."""
    a, sa, b, sb = small_case()
    copies = tuple(tensor.clone() for tensor in (a, sa, b, sb))
    gemv.fp8_block_gemv(a, sa, b, sb, cache_dir=cache)
    for original, copy in zip((a, sa, b, sb), copies):
        assert torch.equal(original, copy), "an input operand was modified"


def test_the_result_does_not_depend_on_torchs_thread_settings(cache):
    """``threads=`` is the operator's own knob, so torch's intraop pool must not move it.

    This is what lets the caller sweep 4/16/32/64 on the operator without
    reconfiguring torch for the rest of the process, and it is why the numbers from
    such a sweep are attributable to the operator alone.
    """
    a = finite_codes((1, 512), 71)
    b = finite_codes((64, 512), 72)
    sa = grid_bytes(1, 16, 73)
    sb = grid_bytes(2, 16, 74)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one_thread = gemv.fp8_block_gemv(a, sa, b, sb, threads=4, cache_dir=cache)
        torch.set_num_threads(max(2, min(8, os.cpu_count() or 2)))
        many_threads = gemv.fp8_block_gemv(a, sa, b, sb, threads=4, cache_dir=cache)
    finally:
        torch.set_num_threads(previous)
    assert_same_bits(many_threads, one_thread, "torch thread settings")
