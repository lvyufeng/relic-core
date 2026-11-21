"""Tests for the grouped-K exact M=1 decode block GEMV.

What each layer of this file can and cannot establish:

* **The fused kernels are not exercised without a card.** The Torch path delegates
  to the frozen prototype's CPU oracle, which itself delegates to the accepted
  block GEMM, so a CPU parity assertion proves the wrapper's operand
  normalisation and nothing about the Ascend kernels. It is still worth asserting:
  it is what catches a packed/int8/FP8 operand being validated or viewed wrongly.
* **The new indexing logic is tested without a device.** The grouped variant's
  whole novelty is which ``(block, column)`` pair one flat tile row owns, so that
  mapping is modelled independently in Python with :func:`_covered` and required
  to cover every pair exactly once, for ragged K, ragged N and a capped grid.
* **The source contract is tested by reading the source.** Every module constant a
  ``@jit`` body reads has to be ``tl.constexpr`` and stable across recompiles; the
  reduction has to stay 2-D; the index width has to be chosen before the offset
  products; and the arithmetic helpers have to be *imported* from the frozen
  modules rather than re-typed here.
* **Bit parity with the accepted operator and the frozen prototype is asserted on
  hardware**, together with an independent exact scalar oracle written in this
  file. That is the acceptance; it needs a leased card:

      RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 RELIC_DECODE_GEMV_TEST_BACKEND=triton \\
          python -m pytest tests/test_decode_grouped_gemv.py -q -rs

No tolerance is used anywhere: outputs are compared as bit patterns and NaN is
compared by classification, exactly as the accepted operator documents its own
non-finite cases.
"""
from __future__ import annotations

import ast
import math
import os
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import block_gemm, decode_block_gemv as prototype  # noqa: E402
from relic_core.kernels import decode_grouped_gemv as grouped  # noqa: E402
from relic_core.kernels.decode_grouped_gemv import (  # noqa: E402
    GROUP_K,
    MAX_PROGRAMS,
    MAX_TILE_ROWS,
    ROWS_N,
    VARIANTS,
    decode_grouped_gemv_scratch,
    fp4_int8_grouped_decode_gemv,
    fp4_packed_grouped_decode_gemv,
    fp8_grouped_decode_gemv,
)

KERNEL_MODULE = Path(grouped.__file__).resolve().parent / "_ascend_decode_grouped_kernels.py"
WRAPPER_MODULE = Path(grouped.__file__).resolve()

#: The exact doubled E2M1 codebook, indexed by nibble (E2M1 code 8 is negative
#: zero, which the integer form deliberately cannot carry).
DOUBLED = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _e4(code):
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = (math.ldexp(float(mantissa), -9) if exponent == 0
             else math.ldexp(float(8 + mantissa), exponent - 10))
    return math.copysign(value, -1.0 if code & 128 else 1.0)


def _a_nominal(code):
    return max(((code >> 3) & 15) - 7, -6)


def _b_value(code, fp4):
    return DOUBLED[code] * 0.5 if fp4 else _e4(code)


def _b_nominal(code, fp4):
    doubled = abs(DOUBLED[code])
    if doubled >= 8:
        return 2
    if doubled >= 4:
        return 1
    return 0 if doubled >= 2 else -1


def _cut(value, anchor):
    grid = 0 if anchor is None or anchor < -500 else anchor - 13
    return f32(math.trunc(f32(value * math.ldexp(1.0, -grid))) * math.ldexp(1.0, grid))


def _exponent(value):
    if value == 0:
        return None
    return ((struct.unpack("<i", struct.pack("<f", value))[0] >> 23) & 255) - 127


def _fragment(a_codes, b_codes, fp4):
    """Independent statement of one K-32 fragment, from the documented reduction.

    Written from the arithmetic rule rather than from the kernel's expression, so a
    kernel that dropped the two-half split, the carry into the second anchor or the
    final cut would disagree here even though it agreed with another transcription.
    """
    products, exponents = [], []
    for ac, bc in zip(a_codes, b_codes):
        value = f32(_e4(ac) * _b_value(bc, fp4))
        products.append(value)
        exponents.append(None if value == 0 else _a_nominal(ac) + _b_nominal(bc, fp4))
    first = max((v for v in exponents[:16] if v is not None), default=None)
    total = f32(sum(_cut(p, first) for p in products[:16]))
    running = _exponent(total)
    second_half = [v for v in exponents[16:] if v is not None]
    if running is not None:
        second_half.append(running)
    second = max(second_half, default=None)
    total = f32(_cut(total, second) + f32(sum(_cut(p, second) for p in products[16:])))
    return _cut(total, _exponent(total))


def _scalar_oracle(a_codes, a_scale, b_codes, b_scale, *, fp4, block_rows=1):
    """``(fragments, result)``: the scalar fragments and the strict K-order sum."""
    a, sa, b, sb = (tensor.tolist() for tensor in (a_codes, a_scale, b_codes, b_scale))
    n, k = len(b), len(a[0])
    blocks = k // 32
    fragments = [[0.0] * n for _ in range(blocks)]
    for block in range(blocks):
        for column in range(n):
            value = _fragment(a[0][block * 32:(block + 1) * 32],
                              b[column][block * 32:(block + 1) * 32], fp4)
            code_a, code_b = sa[0][block], sb[column // block_rows][block]
            scale_a = math.nan if code_a == 255 else math.ldexp(1.0, code_a - 127)
            scale_b = math.nan if code_b == 255 else math.ldexp(1.0, code_b - 127)
            fragments[block][column] = f32(f32(value * scale_a) * scale_b)
    result = []
    for column in range(n):
        value = 0.0
        for block in range(blocks):  # strictly increasing K, no reassociation
            value = f32(value + fragments[block][column])
        result.append(value)
    return fragments, torch.tensor([result], dtype=torch.float32)


def _nibbles_to_packed(nibbles):
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)


def _packed_operands(n, k, *, seed=17, scale_code=127):
    """One activation row, a packed FP4 weight and its original E8M0 grid.

    Activation codes avoid E4M3 NaN so a parity assertion is not comparing NaN to
    NaN; the non-finite case has its own test.
    """
    generator = torch.Generator().manual_seed(seed)
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    a_codes = torch.where((a_codes & 127) == 127,
                          (a_codes.to(torch.int32) - 1).to(torch.uint8), a_codes)
    a_scale = torch.full((1, k // 32), scale_code, dtype=torch.uint8)
    nibbles = torch.randint(0, 16, (n, k), dtype=torch.int64, generator=generator)
    b_scale = torch.full((n, k // 32), scale_code, dtype=torch.uint8)
    return a_codes, a_scale, _nibbles_to_packed(nibbles), b_scale, nibbles


def _int8_operands(n, k, *, seed=23):
    """The same weight as exact ``INT8 = 2 * E2M1`` plus the FP32 half-scale grid."""
    a_codes, a_scale, _, b_scale, nibbles = _packed_operands(n, k, seed=seed)
    doubled = torch.tensor(DOUBLED, dtype=torch.int8)[nibbles.to(torch.long)].to(torch.int8)
    half = torch.ldexp(torch.ones((n, k // 32), dtype=torch.float32),
                       b_scale.to(torch.int32) - 128)
    return a_codes, a_scale, doubled, half


def device():
    name = os.environ.get("RELIC_DECODE_GEMV_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        pytest.importorskip("torch_npu")
    return torch.device(name)


def backend():
    return os.environ.get(
        "RELIC_DECODE_GEMV_TEST_BACKEND",
        "torch" if device().type == "cpu" else "triton",
    )


def assert_bits(actual, expected, *, context=""):
    """Bit equality for finite values, NaN classification for non-finite ones."""
    assert actual.shape == expected.shape, (context, actual.shape, expected.shape)
    actual, expected = actual.cpu(), expected.cpu()
    nan_actual, nan_expected = torch.isnan(actual), torch.isnan(expected)
    assert torch.equal(nan_actual, nan_expected), f"{context}: NaN positions differ"
    finite = ~nan_expected
    if not bool(finite.any()):
        return
    differing = int((actual[finite].view(torch.int32) != expected[finite].view(torch.int32)).sum())
    assert differing == 0, f"{context}: {differing} of {int(finite.sum())} finite values differ"


def assert_finite_nonvacuous(expected, *, context=""):
    assert bool(torch.isfinite(expected).all()), (
        f"{context}: the fixture produced non-finite values, so the bit comparison would be vacuous"
    )


# ------------------------------------------------------------------------------------------------
# The new indexing logic: every (block, column) pair exactly once, nothing extra
# ------------------------------------------------------------------------------------------------


def _covered(n, blocks, *, variant, group_k, rows_n):
    """Model one launch in Python and return the covered pairs and duplicate count.

    This mirrors the kernels' index arithmetic -- flat tile row to
    ``(group, column)``, the per-program K-group stride, and the ragged masks -- so
    it can be checked without a device. It deliberately does not call any Triton
    code; it is a re-statement of the documented schedule.
    """
    column_chunks, _, group_stride = grouped._geometry(
        n, blocks, variant=variant, group_k=group_k, rows_n=rows_n
    )
    covered, duplicates = set(), 0
    for pid0 in range(column_chunks):
        if variant == "serial":
            for block in range(blocks):
                for within in range(rows_n):
                    column = pid0 * rows_n + within
                    if column < n:
                        covered.add((block, column))
            continue
        for pid1 in range(group_stride):
            for group_base in range(pid1 * group_k, blocks, group_stride * group_k):
                for flat_row in range(group_k * rows_n):
                    block = group_base + (flat_row // rows_n)
                    column = pid0 * rows_n + (flat_row % rows_n)
                    if block < blocks and column < n:
                        pair = (block, column)
                        if pair in covered:
                            duplicates += 1
                        covered.add(pair)
    return covered, duplicates


@pytest.mark.parametrize("rows_n", [8, 16, 32])
@pytest.mark.parametrize("group_k", [1, 2, 4])
@pytest.mark.parametrize("n,k", [(1, 32), (7, 96), (37, 160), (64, 256)])
def test_the_grouped_tile_covers_every_pair_exactly_once(n, k, group_k, rows_n):
    blocks = k // 32
    covered, duplicates = _covered(n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n)
    assert duplicates == 0
    assert covered == {(block, column) for block in range(blocks) for column in range(n)}


@pytest.mark.parametrize("rows_n", [8, 16, 32])
@pytest.mark.parametrize("n,k", [(1, 32), (7, 96), (37, 160)])
def test_the_serial_tile_covers_every_pair_exactly_once(n, k, rows_n):
    blocks = k // 32
    covered, duplicates = _covered(n, blocks, variant="serial", group_k=GROUP_K, rows_n=rows_n)
    assert duplicates == 0
    assert covered == {(block, column) for block in range(blocks) for column in range(n)}


def test_a_ragged_k_group_reads_a_defined_zero_and_is_masked_out():
    """K = 160 is five blocks; a GROUP_K = 8 tile has three rows past the end.

    The kernel masks those loads with ``other=0`` and masks their stores, so the
    last group must still cover exactly five blocks and no more. A mask that
    leaked would write a fragment for block 5..7 into a ``[5, N]`` scratch.
    """
    covered, duplicates = _covered(37, 5, variant="grouped", group_k=8, rows_n=16)
    assert duplicates == 0
    assert {block for block, _ in covered} == set(range(5))


def test_a_geometry_the_device_cannot_launch_is_refused_before_allocation():
    """The check has to run before a scratch allocation, or the CPU path would pass."""
    with pytest.raises(ValueError, match="total-program cap"):
        grouped._geometry(MAX_PROGRAMS * 64 + 1, 160, variant="grouped",
                          group_k=8, rows_n=64)
    with pytest.raises(ValueError, match="total-program cap"):
        grouped._geometry(MAX_PROGRAMS * 64 + 1, 160, variant="serial",
                          group_k=8, rows_n=64)


def test_a_capped_grid_strides_the_remaining_groups_without_missing_a_block():
    """When the K-group axis does not fit the cap, every program strides the rest.

    The stride has to be a whole number of groups, and the union of the strided
    block ranges has to be exactly ``range(blocks)`` -- a stride that skipped a
    block would leave a scratch row unwritten and pass 2 would still return a
    number. Checked on the index arithmetic rather than by materializing every
    ``(block, column)`` pair, which at this width would be millions of tuples.
    """
    rows_n, group_k = 8, 4
    n, blocks = 20_000, 160
    column_chunks, _, group_stride = grouped._geometry(
        n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n
    )
    assert column_chunks * group_stride <= MAX_PROGRAMS
    assert group_stride < math.ceil(blocks / group_k), "this fixture must actually cap the grid"
    step = group_stride * group_k
    reached = set()
    for pid1 in range(group_stride):
        for group_base in range(pid1 * group_k, blocks, step):
            for offset in range(group_k):
                block = group_base + offset
                if block < blocks:
                    reached.add(block)
    assert reached == set(range(blocks)), "the strided groups must cover every K block"

    # The column side must still tile the whole width exactly once.
    assert column_chunks * rows_n >= n
    assert (column_chunks - 1) * rows_n < n


def test_the_default_geometry_for_the_released_shapes():
    """Pin the program counts the schedule exists to shrink, so a regression shows up."""
    column_chunks, _, group_stride = grouped._geometry(
        2304, 160, variant="grouped", group_k=GROUP_K, rows_n=ROWS_N
    )
    assert (column_chunks, group_stride) == (144, 40)
    assert column_chunks * group_stride == 5760  # against 23040 one-fragment programs
    column_chunks, _, group_stride = grouped._geometry(
        5120, 72, variant="grouped", group_k=GROUP_K, rows_n=ROWS_N
    )
    assert (column_chunks, group_stride) == (320, 18)
    # The serial variant launches one program per column tile and loops K inside.
    assert grouped._geometry(2304, 160, variant="serial", group_k=GROUP_K,
                             rows_n=ROWS_N) == (144, 1, 1)


# ------------------------------------------------------------------------------------------------
# The CPU oracle path: accepted operator, frozen prototype, and this wrapper agree
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(32, 64), (7, 96), (64, 160)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_packed_cpu_path_agrees_with_the_frozen_prototype(n, k, variant):
    a, sa, packed, sb, _ = _packed_operands(n, k)
    expected = prototype.fp4_packed_decode_gemv(a, sa, packed, sb, backend="torch")
    assert_finite_nonvacuous(expected, context="packed cpu")
    actual = fp4_packed_grouped_decode_gemv(
        a, sa, packed, sb, variant=variant, backend="torch"
    )
    assert_bits(actual, expected, context=f"packed cpu {variant}")


@pytest.mark.parametrize("n,k", [(32, 64), (7, 96)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_int8_cpu_path_agrees_with_the_accepted_block_gemm(n, k, variant):
    a, sa, doubled, half = _int8_operands(n, k)
    expected = block_gemm.fp4_int8_block_gemm(a, sa, doubled, half, backend="torch")
    assert_finite_nonvacuous(expected, context="int8 cpu")
    actual = fp4_int8_grouped_decode_gemv(
        a, sa, doubled, half, variant=variant, backend="torch"
    )
    assert_bits(actual, expected, context=f"int8 cpu {variant}")


@pytest.mark.parametrize("block_rows", [1, 32])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_fp8_cpu_path_agrees_with_the_accepted_block_gemm(block_rows, variant):
    generator = torch.Generator().manual_seed(31)
    n, k = 48, 128
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    a_scale = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b_codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    b_scale = torch.full((math.ceil(n / block_rows), k // 32), 127, dtype=torch.uint8)
    expected = block_gemm.fp8_block_gemm(
        a_codes, a_scale, b_codes, b_scale, block_rows=block_rows, backend="torch"
    )
    actual = fp8_grouped_decode_gemv(
        a_codes, a_scale, b_codes, b_scale, block_rows=block_rows,
        variant=variant, backend="torch",
    )
    assert_bits(actual, expected, context=f"fp8 cpu {variant} block_rows={block_rows}")


def test_the_float_weight_scale_grid_is_accepted_for_fp8():
    """The Engram-style grid arrives as exact FP32 powers, not only as raw bytes."""
    generator = torch.Generator().manual_seed(41)
    n, k = 24, 96
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    a_scale = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b_codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    raw = torch.full((n, k // 32), 130, dtype=torch.uint8)
    float_grid = torch.ldexp(torch.ones(raw.shape, dtype=torch.float32),
                             torch.full(raw.shape, 130 - 127, dtype=torch.int32))
    by_bytes = fp8_grouped_decode_gemv(a_codes, a_scale, b_codes, raw,
                                       block_rows=1, backend="torch")
    by_float = fp8_grouped_decode_gemv(a_codes, a_scale, b_codes, float_grid,
                                      block_rows=1, backend="torch")
    assert_bits(by_float, by_bytes, context="fp8 float vs byte weight scale")


# ------------------------------------------------------------------------------------------------
# Refusals: the knobs are part of the contract
# ------------------------------------------------------------------------------------------------


def test_an_unknown_variant_is_refused_by_name():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="variant must be one of"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, variant="groupd", backend="torch")


@pytest.mark.parametrize("knob", ["group_k", "rows_n"])
def test_the_tile_sides_must_be_powers_of_two(knob):
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    arguments = {"group_k": GROUP_K, "rows_n": ROWS_N, knob: 6}
    with pytest.raises(ValueError, match="positive power of two"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, backend="torch", **arguments)


def test_a_tile_over_the_register_budget_is_refused_by_name():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="per-program tile budget"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, group_k=32, rows_n=64, backend="torch")


def test_every_sweep_the_root_will_run_is_accepted():
    """The tuning grid must not contain a combination the budget refuses."""
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    accepted = 0
    for group_k in (1, 2, 4):
        for rows_n in (4, 8, 16, 32):
            if group_k * rows_n > MAX_TILE_ROWS:
                continue  # e.g. 4 x 32 = 128, outside the measured budget
            fp4_packed_grouped_decode_gemv(
                a, sa, packed, sb, variant="grouped", group_k=group_k,
                rows_n=rows_n, backend="torch",
            )
            accepted += 1
    assert accepted >= 8, "the tuning grid should mostly be inside the budget"


def test_the_default_schedule_is_the_measured_ub_safe_configuration():
    """The defaults must be a configuration that actually compiles on the target.

    A 128-row tile was the default when the first tuning run failed to compile
    with "ub overflow, requires 2675968 bits while 1572864 bits available", so the
    defaults are pinned here against the budget rather than left to drift.
    """
    assert (GROUP_K, ROWS_N) == (4, 16)
    assert GROUP_K * ROWS_N <= MAX_TILE_ROWS
    assert MAX_TILE_ROWS == 64
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    # The default call must not need a geometry argument at all.
    fp4_packed_grouped_decode_gemv(a, sa, packed, sb, backend="torch")


def test_the_geometry_that_overflowed_the_unified_buffer_is_refused_by_name():
    """8 x 16 = 128 rows is the exact configuration that failed to compile.

    It has to be refused *before* the launch: a version that only failed inside
    the compiler would make the default unusable on the target and would not be
    catchable on a CPU-only host like this one.
    """
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="192 KiB unified buffer"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, group_k=8, rows_n=16,
                                       backend="torch")
    for group_k, rows_n in ((8, 16), (16, 8), (16, 16), (4, 32), (32, 8)):
        assert group_k * rows_n > MAX_TILE_ROWS
        with pytest.raises(ValueError, match="per-program tile budget"):
            fp4_packed_grouped_decode_gemv(a, sa, packed, sb, group_k=group_k,
                                           rows_n=rows_n, backend="torch")


def test_the_serial_variant_is_not_bounded_by_the_group_tile_budget():
    """Serial has no group tile, so a large group_k is inert rather than illegal.

    Bounding it there would refuse a harmless parameter and would also imply the
    serial kernel builds a tile it does not build.
    """
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    reference = fp4_packed_grouped_decode_gemv(a, sa, packed, sb, variant="serial",
                                              backend="torch")
    for group_k in (8, 16, 32):
        assert group_k * ROWS_N > MAX_TILE_ROWS
        actual = fp4_packed_grouped_decode_gemv(
            a, sa, packed, sb, variant="serial", group_k=group_k, backend="torch"
        )
        assert_bits(actual, reference, context=f"serial ignores group_k={group_k}")


def test_two_activation_rows_are_refused_by_name():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    two_rows = a.repeat(2, 1)
    with pytest.raises(ValueError, match="exactly one activation row"):
        fp4_packed_grouped_decode_gemv(two_rows, sa, packed, sb, backend="torch")


def test_a_wrong_k_is_refused():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="positive K divisible by 32"):
        fp4_packed_grouped_decode_gemv(a[:, :48], sa, packed, sb, backend="torch")


def test_packed_bytes_must_hold_two_codes_per_byte():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="two K codes per byte"):
        fp4_packed_grouped_decode_gemv(a, sa, packed[:, :-1].contiguous(), sb, backend="torch")


def test_the_half_scale_grid_must_match_the_weight():
    a, sa, doubled, half = _int8_operands(8, 64)
    with pytest.raises(ValueError, match="original half grid"):
        fp4_int8_grouped_decode_gemv(a, sa, doubled, half[:, :-1].contiguous(),
                                     backend="torch")


def test_a_cpu_tensor_cannot_ask_for_the_fused_backend():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="NPU tensors only"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, backend="triton")


def test_the_output_must_not_overlap_an_operand():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    n = packed.shape[0]
    # A [1, N] float32 view over the packed weight's storage is the same bytes.
    overlapping = packed.reshape(-1).view(torch.float32)[:n].reshape(1, n)
    with pytest.raises(ValueError, match="must not overlap"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, out=overlapping, backend="torch")


def test_the_scratch_pair_is_validated_even_on_the_cpu_path():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    wrong = (torch.empty((1, 8), dtype=torch.float32), torch.empty((2, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"\[K // 32, N\]"):
        fp4_packed_grouped_decode_gemv(a, sa, packed, sb, scratch=wrong, backend="torch")


def test_the_scratch_helper_matches_the_frozen_layout():
    fragments, invalid = decode_grouped_gemv_scratch(9, 128)
    assert tuple(fragments.shape) == (4, 9) and fragments.dtype == torch.float32
    assert tuple(invalid.shape) == (4, 9) and invalid.dtype == torch.uint8


def test_out_and_scratch_are_returned_in_place_on_the_cpu_path():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    out = torch.full((1, 8), -7.0, dtype=torch.float32)
    returned = fp4_packed_grouped_decode_gemv(a, sa, packed, sb, out=out, backend="torch")
    assert returned is out
    assert bool((out != -7.0).all()), "the supplied output must have been written"


# ------------------------------------------------------------------------------------------------
# Source contract: read the module, do not trust the docstrings
# ------------------------------------------------------------------------------------------------


def _tree():
    return ast.parse(KERNEL_MODULE.read_text(encoding="utf-8"))


def _constants(tree):
    """Module-level ``name = tl.constexpr(...)`` / ``name: tl.constexpr = ...``."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Call) and _dotted(value.func).endswith("tl.constexpr"):
            found[target.id] = ast.literal_eval(value.args[0])
    return found


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _jit_functions(tree):
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if _dotted(decorator).endswith("triton.jit"):
                yield node


def _free_names(function):
    bound = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return {node.id for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id not in bound}


def test_every_global_a_kernel_body_reads_is_constexpr_or_a_jit_function():
    tree = _tree()
    constants = set(_constants(tree))
    functions = {function.name for function in _jit_functions(tree)}
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports |= {alias.asname or alias.name for alias in node.names}
    allowed = constants | functions | imports | {"tl", "triton", "range", "float", "int", "abs"}
    problems = []
    for function in _jit_functions(tree):
        for name in sorted(_free_names(function)):
            if name not in allowed:
                problems.append(f"{function.name} reads {name!r}")
    assert not problems, (
        "the Triton front end rejects a plain module global read from a @jit body; "
        "annotate it as tl.constexpr: " + "; ".join(problems)
    )


def test_no_kernel_module_declares_a_plain_scalar_global():
    tree = _tree()
    constants = _constants(tree)
    plain = []
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target is not None and target not in constants:
            plain.append(target)
    assert not plain, f"these module scalars are not tl.constexpr: {plain}"


def test_every_constexpr_global_is_stable_across_recompiles():
    """A float NaN global makes Triton report the kernel as changed on recompile."""
    constants = _constants(_tree())
    unstable = [name for name, value in constants.items()
                if isinstance(value, float) and math.isnan(value)]
    assert not unstable, (
        "a NaN constexpr global compares unequal to itself, so Triton treats the kernel "
        f"as changed on its second call; build NaN from integer bits instead: {unstable}"
    )


def test_the_reduction_stays_2d_and_reduces_only_along_the_lane_axis():
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    for forbidden in (".reshape(", ".view(", "expand_dims", "unsqueeze", "broadcast_to",
                      "[None, :, None]", "[None, None, :]", "[:, :, None]"):
        assert forbidden not in source, (
            f"{forbidden!r} would make a 3-D operand, and this backend has no proven 3-D "
            "reduction; the (group, column, lane) cube is flattened up front"
        )
    tree = _tree()
    reductions = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and _dotted(node.func) in {"tl.sum", "tl.max"}]
    assert reductions, "the reduction should be visible to this test"
    for node in reductions:
        last = node.args[-1]
        assert isinstance(last, ast.Constant) and last.value == 1, (
            f"{_dotted(node.func)} must reduce axis 1 (the 32 K lanes) only"
        )


def test_the_frozen_reduction_helpers_are_imported_rather_than_recopied():
    """A re-typed helper is a second implementation that can drift."""
    tree = _tree()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported |= {alias.asname or alias.name for alias in node.names}
    required = {
        "_cut", "_decode_e4m3", "_floor_exp", "_grid_exponent", "_nominal_e4",
        "_original_scale_code", "_pow2", "_scale_e8m0",
    }
    assert required <= imported, f"missing frozen imports: {sorted(required - imported)}"
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    clash = required & defined
    assert not clash, f"these frozen helpers are re-implemented here: {sorted(clash)}"


def test_the_accumulation_pass_is_the_frozen_kernel_unchanged():
    """Pass 2 must not be re-implemented, or its strict K order could be changed."""
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    assert "decode_accumulate_kernel" in source
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {alias.asname or alias.name for alias in node.names}
    assert "decode_accumulate_kernel" in imported
    kernel_tree = _tree()
    defined = {node.name for node in kernel_tree.body if isinstance(node, ast.FunctionDef)}
    assert "decode_accumulate_kernel" not in defined, (
        "the second pass belongs to the frozen prototype; re-defining it here would let the "
        "strict increasing-K accumulation be changed silently"
    )


def test_the_serial_kernel_needs_no_k_scratch():
    """The whole point of the serial variant is that it skips the two-pass scratch."""
    tree = _tree()
    function = next(node for node in _jit_functions(tree)
                    if node.name == "decode_serial_gemv_kernel")
    parameters = [argument.arg for argument in function.args.args]
    assert "FRAG" not in parameters and "BAD" not in parameters
    assert parameters.count("OUT") == 1


def test_both_variants_share_one_arithmetic_path():
    """Grouped and serial must call the same reduction, so they cannot drift apart."""
    tree = _tree()
    defined = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for variant_function, helper in (("decode_grouped_fragment_kernel", "_fragment_rows"),
                                     ("decode_serial_gemv_kernel", "_fragment_single_row")):
        calls = {_dotted(node.func) for node in ast.walk(defined[variant_function])
                 if isinstance(node, ast.Call)}
        assert helper in calls, f"{variant_function} must call {helper}"
    # Both shape wrappers have to route through _reduce_two_halves and _reduce only there.
    for wrapper in ("_fragment_rows", "_fragment_single_row"):
        calls = {_dotted(node.func) for node in ast.walk(defined[wrapper])
                 if isinstance(node, ast.Call)}
        assert "_reduce_two_halves" in calls, f"{wrapper} must use the shared reduction"
    reduce_calls = sum(
        1 for node in ast.walk(defined["_reduce_two_halves"]) if isinstance(node, ast.Call)
        and _dotted(node.func) in {"tl.sum", "tl.max"}
    )
    assert reduce_calls == 4, "the two-half rule is two maxima and two sums; nothing else"


def test_index64_widens_before_any_offset_product():
    """``(...).to(int64)`` after the product would already have overflowed."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert ".to(tl.int64)" in source
    # Every widening must be applied to a bare name, never to a product.
    for line in source.splitlines():
        if ".to(tl.int64)" not in line:
            continue
        head = line.split(".to(tl.int64)")[0].rstrip()
        assert head.endswith(("columns", "block", "group_base", "row")), (
            f"a widening applied to something other than an index: {line.strip()}"
        )


def _packed_nibble_python(packed, lane):
    """The kernel's static-shift nibble selection, restated for all 256 bytes."""
    return packed & 15 if (lane & 1) == 0 else packed >> 4


def _e2m1_x2_python(nibble):
    """The kernel's comparison ladder, restated."""
    magnitude = nibble & 7
    absolute = (magnitude if magnitude < 4 else
                4 if magnitude == 4 else 6 if magnitude == 5 else 8 if magnitude == 6 else 12)
    return -absolute if (nibble & 8) != 0 else absolute


def test_the_static_shift_lut_is_the_exact_sixteen_code_codebook():
    """All sixteen codes, and the doubled values the frozen helper produces.

    This is the numerical contract the spelling change must not move: the frozen
    ``_e2m1_x2`` and this ladder have to agree on every code, including code 8
    (E2M1 negative zero -> integer zero) and code 15 (-12).
    """
    assert [_e2m1_x2_python(nibble) for nibble in range(16)] == list(DOUBLED)
    # And, for completeness, that the magnitudes are the documented ones.
    assert [_e2m1_x2_python(nibble) for nibble in range(8)] == [0, 1, 2, 3, 4, 6, 8, 12]


@pytest.mark.parametrize("lane_parity", [0, 1])
def test_every_packed_byte_decodes_to_the_documented_code(lane_parity):
    """All 256 byte values x both lane parities: the two nibbles are both reachable.

    A packed byte holds one low and one high nibble, and which one a lane reads is
    decided by its parity, so a select that returned the same nibble for both
    parities would still pass a single-parity check.
    """
    for byte in range(256):
        low, high = byte & 15, byte >> 4
        expected = DOUBLED[low] if lane_parity == 0 else DOUBLED[high]
        actual = _e2m1_x2_python(_packed_nibble_python(byte, lane_parity))
        assert actual == expected, f"byte {byte:#04x} lane parity {lane_parity}"


def _unparsed(name):
    """One kernel function as source with comments and docstrings removed."""
    tree = _tree()
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    body = [node for node in function.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_the_packed_decoders_use_no_per_lane_variable_shift():
    """A variable shift amount is the measured cost; only constant shifts may remain.

    ``packed >> ((lane & 1) * 4)`` and ``(2 + (m & 1)) << max((m >> 1) - 1, 0)``
    both compute the right answer and both lower badly on this backend. The
    replacement must contain no shift whose amount is not a literal.
    """
    tree = _tree()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_packed_nibble", "_e2m1_x2_lut"):
        assert name in functions, f"{name} must exist: it is the static-shift decoder"
    offenders = []
    for function in _jit_functions(tree):
        # A ``tl.constexpr`` parameter is a literal by the time the kernel is
        # specialized, so a shift by one is a compile-time shift and safe.
        constexpr_args = {
            argument.arg for argument in function.args.args
            if argument.annotation is not None
            and "constexpr" in ast.unparse(argument.annotation)
        }
        for node in ast.walk(function):
            if not isinstance(node, ast.BinOp):
                continue
            if not isinstance(node.op, (ast.LShift, ast.RShift)):
                continue
            amount = node.right
            if isinstance(amount, ast.Constant) and isinstance(amount.value, int):
                continue
            if isinstance(amount, ast.Name) and amount.id in constexpr_args:
                continue
            offenders.append(f"{function.name}: {ast.unparse(node)}")
    assert not offenders, (
        "these shifts have a runtime amount, which CANN lowers badly: " + "; ".join(offenders)
    )
    # The two constant-shift expressions the select is built from.
    select = _unparsed("_packed_nibble")
    assert "packed & 15" in select and "packed >> 4" in select
    assert "lanes & 1" in select
    # And no call to the variable-shift frozen spellings anywhere in a kernel body.
    for function in _jit_functions(tree):
        calls = {ast.unparse(node.func) for node in ast.walk(function)
                 if isinstance(node, ast.Call)}
        assert "_e2m1_x2" not in calls, f"{function.name} must use the LUT, not the shift helper"


def test_both_packed_paths_call_the_static_decoders():
    for kernel in ("decode_grouped_fragment_kernel", "decode_serial_gemv_kernel"):
        body = _unparsed(kernel)
        assert "_e2m1_x2_lut" in body, f"{kernel} must use the exact LUT ladder"
        assert "_packed_nibble" in body, f"{kernel} must use the static nibble select"


def test_the_frozen_unpack_helper_is_not_modified_or_redefined():
    """The proof helper stays as it is; this module may only spell it differently."""
    frozen = (KERNEL_MODULE.parent / "_ascend_unpack_kernels.py").read_text(encoding="utf-8")
    assert "def _e2m1_x2(" in frozen, "the frozen widening helper must still be defined"
    assert "def _e2m1_x2_lut(" not in frozen, "the experimental decoder must stay out of the frozen module"
    tree = _tree()
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_e2m1_x2" not in defined, "the frozen helper must not be redefined here"


def test_the_activation_tile_is_a_2d_gather_with_an_explicit_row_axis():
    """``block * 32 + lanes[None, :]`` broadcasts [TILE] against 32 and is rejected.

    This is the exact compile error the first hardware run hit: ``block`` is a
    ``[TILE]`` vector, so the row axis has to be named with ``[:, None]``.
    """
    body = _unparsed("decode_grouped_fragment_kernel")
    assert "block[:, None] * 32 + lanes[None, :]" in body, (
        "the A gather must index rows with an explicit [:, None]"
    )
    # The bare form must not survive anywhere in the kernel body.
    assert not any(
        "block * 32 + lanes[None, :]" in segment
        for segment in body.split("block[:, None] * 32 + lanes[None, :]")
    )


def _row_vectors_used_2d():
    """Lines of the unparsed kernels where a row vector feeds a 2-D offset."""
    offenders = []
    for kernel in ("decode_grouped_fragment_kernel", "decode_serial_gemv_kernel"):
        for line in _unparsed(kernel).splitlines():
            stripped = line.strip()
            if "tl.load(" not in stripped and "offsets =" not in stripped and "byte_offsets" not in stripped:
                continue
            if "[None, :]" in stripped and "[:, None]" not in stripped and "lanes" not in stripped:
                offenders.append(f"{kernel}: {stripped}")
    return offenders


def test_every_2d_offset_names_its_row_axis():
    """A [TILE] vector added to a [1, lane] broadcast is the compile error we hit."""
    assert not _row_vectors_used_2d()


def test_no_uint64_promoting_data_literal():
    """CANN vector casts cannot promote uint32 to uint64, so no literal may force it."""
    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            assert -2**31 <= node.value < 2**31, f"literal {node.value} does not fit int32"


def test_index64_is_chosen_from_the_full_reach():
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    assert "index64 = max(n * k, n * blocks)" in source, (
        "the index width must be chosen from the widest weight and scratch offset reach"
    )

# ------------------------------------------------------------------------------------------------
# Acceptance on hardware: bit parity with the accepted operator and the frozen prototype
# ------------------------------------------------------------------------------------------------

requires_device = pytest.mark.skipif(
    device().type == "cpu",
    reason="set RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 with a leased card to run the fused kernels",
)


def _to(tensor, target):
    return tensor.to(target)


@requires_device
@pytest.mark.parametrize("n,k", [(32, 64), (7, 96), (37, 160), (2304, 5120), (5120, 2304)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_packed_kernel_matches_the_accepted_operator_bit_for_bit(n, k, variant):
    """The primary failure this operator would have: a fragment reordered or rescaled."""
    target = device()
    a, sa, packed, sb, _ = _packed_operands(n, k)
    expected = prototype.fp4_packed_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target)
    )
    assert_finite_nonvacuous(expected, context="packed npu")
    actual = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        variant=variant,
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"packed npu {variant} {n}x{k}")


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_packed_kernel_matches_the_frozen_two_pass_prototype(variant):
    """Same schedule family, different granule: the answer must not move."""
    target = device()
    a, sa, packed, sb, _ = _packed_operands(48, 128, seed=53)
    expected = prototype.fp4_packed_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target)
    )
    actual = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        variant=variant,
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"packed vs prototype {variant}")


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_packed_kernel_matches_the_independent_scalar_oracle(variant):
    """The oracle is written from the reduction rule, not from the kernel's expression."""
    target = device()
    n, k = 9, 96
    a, sa, packed, sb, nibbles = _packed_operands(n, k, seed=61)
    weights = nibbles.to(torch.uint8)  # one code per K element, for the oracle
    fragments, expected = _scalar_oracle(a, sa, weights, sb, fp4=True)
    assert_finite_nonvacuous(expected, context="scalar oracle")
    actual = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        variant=variant, rows_n=8,
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"scalar oracle {variant}")


@requires_device
@pytest.mark.parametrize("group_k", [1, 2, 4])
def test_the_group_size_does_not_change_a_single_bit(group_k):
    """The sweep the profiling plan runs must be numerically inert."""
    target = device()
    a, sa, packed, sb, _ = _packed_operands(48, 160, seed=67)
    reference = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        group_k=1, rows_n=16,
    )
    actual = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        group_k=group_k, rows_n=16,
    )
    torch.npu.synchronize()
    assert_bits(actual, reference, context=f"group_k={group_k}")


@requires_device
@pytest.mark.parametrize("rows_n", [8, 16, 32])
def test_the_column_tile_does_not_change_a_single_bit(rows_n):
    target = device()
    a, sa, packed, sb, _ = _packed_operands(37, 160, seed=71)
    reference = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        rows_n=8,
    )
    actual = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        rows_n=rows_n,
    )
    torch.npu.synchronize()
    assert_bits(actual, reference, context=f"rows_n={rows_n}")


@requires_device
def test_grouped_group_k_one_reproduces_the_frozen_scratch_fragment_by_fragment():
    """At GROUP_K=1 the grouped pass-1 tile is the frozen tile, entry for entry.

    The scratch is the only place a fragment is visible, so this is what proves the
    grouped kernel did not merely produce the same final sum -- which a wrong
    fragment could still do after cancelling in pass 2.
    """
    target = device()
    n, k = 32, 128
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=73)
    frozen = prototype.decode_gemv_scratch(n, k, device=target)
    grouped_scratch = decode_grouped_gemv_scratch(n, k, device=target)
    prototype.fp4_packed_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        scratch=frozen,
    )
    fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        group_k=1, rows_n=16, scratch=grouped_scratch,
    )
    torch.npu.synchronize()
    finite = torch.isfinite(frozen[0])
    assert torch.equal(finite, torch.isfinite(grouped_scratch[0]))
    assert torch.equal(frozen[0][finite].view(torch.int32),
                       grouped_scratch[0][finite].view(torch.int32))
    assert torch.equal(frozen[1], grouped_scratch[1])


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_int8_kernel_matches_the_accepted_block_gemm(variant):
    target = device()
    a, sa, doubled, half = _int8_operands(48, 128, seed=79)
    expected = block_gemm.fp4_int8_block_gemm(
        _to(a, target), _to(sa, target), _to(doubled, target), _to(half, target)
    )
    actual = fp4_int8_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(doubled, target), _to(half, target),
        variant=variant,
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"int8 npu {variant}")


@requires_device
@pytest.mark.parametrize("block_rows", [1, 32])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_fp8_kernel_matches_the_accepted_block_gemm(block_rows, variant):
    target = device()
    generator = torch.Generator().manual_seed(83)
    n, k = 48, 128
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    a_scale = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b_codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    b_scale = torch.full((math.ceil(n / block_rows), k // 32), 127, dtype=torch.uint8)
    tensors = [_to(t, target) for t in (a_codes, a_scale, b_codes, b_scale)]
    expected = block_gemm.fp8_block_gemm(*tensors, block_rows=block_rows)
    actual = fp8_grouped_decode_gemv(
        *tensors, block_rows=block_rows, variant=variant
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"fp8 npu {variant} block_rows={block_rows}")


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_nan_activation_code_nans_the_whole_column(variant):
    """The accepted kernel's final ``where`` replaces a column with NaN, not one entry."""
    target = device()
    n, k = 16, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=89)
    a = a.clone()
    a[0, 32 + 5] = 0x7F  # E4M3 NaN, in the second K block
    tensors = [_to(t, target) for t in (a, sa, packed, sb)]
    actual = fp4_packed_grouped_decode_gemv(*tensors, variant=variant)
    torch.npu.synchronize()
    assert bool(torch.isnan(actual).all()), "a NaN activation code must NaN every column"


@requires_device
@pytest.mark.parametrize("code", [0, 1, 126, 253, 254])
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_finite_scale_endpoint_is_identical(code, variant):
    target = device()
    n, k = 8, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=97, scale_code=code)
    tensors = [_to(t, target) for t in (a, sa, packed, sb)]
    expected = prototype.fp4_packed_decode_gemv(*tensors)
    actual = fp4_packed_grouped_decode_gemv(*tensors, variant=variant)
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"scale {code} {variant}")


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_an_e8m0_scale_of_255_nans_like_the_accepted_operator(variant):
    target = device()
    n, k = 8, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=101, scale_code=255)
    tensors = [_to(t, target) for t in (a, sa, packed, sb)]
    expected = prototype.fp4_packed_decode_gemv(*tensors)
    actual = fp4_packed_grouped_decode_gemv(*tensors, variant=variant)
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"scale 255 {variant}")


@requires_device
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_cancelling_block_sequence_keeps_the_strict_k_order(variant):
    """Alternating large signs is where a reassociated sum would show up."""
    target = device()
    n, k = 8, 96
    generator = torch.Generator().manual_seed(103)
    a = torch.full((1, k), 0x38, dtype=torch.uint8)  # E4M3 1.0
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    signs = torch.tensor([0, 8], dtype=torch.int64).repeat_interleave(k // 2)[:k]
    nibbles = torch.where((torch.arange(k) // 32) % 2 == 0,
                          torch.full((k,), 7, dtype=torch.int64), signs)
    weights = nibbles.unsqueeze(0).repeat(n, 1)
    packed = _nibbles_to_packed(weights)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    tensors = [_to(t, target) for t in (a, sa, packed, sb)]
    expected = prototype.fp4_packed_decode_gemv(*tensors)
    actual = fp4_packed_grouped_decode_gemv(*tensors, variant=variant)
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"cancelling {variant}")
    assert generator is not None


@requires_device
def test_the_serial_variant_writes_the_supplied_output_in_place():
    target = device()
    n, k = 8, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=107)
    out = torch.full((1, n), -7.0, dtype=torch.float32, device=target)
    returned = fp4_packed_grouped_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        variant="serial", out=out,
    )
    torch.npu.synchronize()
    assert returned is out
    assert bool((out != -7.0).all())
