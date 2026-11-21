"""Tests for the prefetch-access grouped decode GEMV.

What each layer of this file can and cannot establish:

* **The fused kernel is not exercised without a card.** The Torch backend delegates
  to the frozen prototype's CPU oracle, which itself delegates to the accepted
  block GEMM, so a CPU parity assertion proves this wrapper's operand
  normalisation, dispatch and scratch plumbing -- and nothing about the Ascend
  kernel. The kernel's parity is asserted only by the leased-hardware test:

      RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 RELIC_DECODE_GEMV_TEST_BACKEND=triton \\
          python -m pytest tests/test_decode_prefetch_gemv.py -q -rs

* **The new memory access is modelled independently in Python.** The wide-row
  load, the column-major flat-row mapping, the gather that duplicates each byte
  into two lanes and the nibble select are re-stated here from the documented
  contract, then required to cover every ``(block, column)`` pair and every
  ``(column, block, byte)`` exactly once -- including ragged K and N. The gather
  model is given teeth by checking that a different lane-to-byte pairing is
  rejected and changes values.

* **The source contract is read, not trusted.** The raw weight tile has to stay
  ``uint8`` until after the gather, every ``reshape``/``broadcast_to`` has to stay
  two-dimensional, the index width has to be chosen before the offset products,
  the nibble select has to be the frozen static-shift helper, and the arithmetic
  helpers have to be *imported* from the frozen modules rather than re-typed.

No tolerance is used anywhere: finite values are compared as bit patterns and
non-finite ones by NaN classification, matching the accepted operator's own
contract. ``torch.equal`` is deliberately not used, because it reports two NaN
outputs as different.
"""
from __future__ import annotations

import ast
import math
import os
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import decode_block_gemv as prototype  # noqa: E402
from relic_core.kernels import decode_grouped_gemv as grouped  # noqa: E402
from relic_core.kernels import decode_prefetch_gemv as prefetch  # noqa: E402
from relic_core.kernels.decode_prefetch_gemv import (  # noqa: E402
    GROUP_K,
    MAX_TILE_ROWS,
    ROWS_N,
    decode_prefetch_gemv_scratch,
    fp4_packed_prefetch_decode_gemv,
)

KERNEL_MODULE = (
    Path(prefetch.__file__).resolve().parent / "_ascend_decode_prefetch_kernels.py"
)
WRAPPER_MODULE = Path(prefetch.__file__).resolve()

#: The E2M1 magnitudes, from the format rather than from any kernel table; the
#: exact ``INT8 = 2 * E2M1`` codebook is derived from them below. E2M1 code 8 is
#: negative zero, whose integer form is zero.
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def doubled(code):
    """Exact ``2 * E2M1(code)`` as an int; the sign comes from bit 3."""
    magnitude = int(MAGNITUDES[code & 7] * 2)
    return -magnitude if code & 8 else magnitude


# ------------------------------------------------------------------------------------------------
# Fixtures and bit comparison
# ------------------------------------------------------------------------------------------------


def _packed_operands(n, k, *, seed=17, scale_code=127):
    """One activation row, a packed FP4 weight and its original E8M0 grid."""
    generator = torch.Generator().manual_seed(seed)
    a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, generator=generator)
    # E4M3's NaN encodings are the two bytes whose low seven bits are all set.
    a_codes = torch.where((a_codes & 127) == 127,
                          (a_codes.to(torch.int32) - 1).to(torch.uint8), a_codes)
    a_scale = torch.full((1, k // 32), scale_code, dtype=torch.uint8)
    nibbles = torch.randint(0, 16, (n, k), dtype=torch.int64, generator=generator)
    b_scale = torch.full((n, k // 32), scale_code, dtype=torch.uint8)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return a_codes, a_scale, packed, b_scale, nibbles


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
# The new memory access, modelled independently: mapping, byte coverage, gather
# ------------------------------------------------------------------------------------------------


def _plan(n, blocks, group_k, rows_n):
    """Model one launch and return its covered ``(block, column)`` pairs.

    This re-states the kernel's column-major flat-row mapping and its ragged
    masks from the documented contract, and it calls no Triton code. The grid
    comes from the same :func:`decode_grouped_gemv._geometry` the wrapper
    launches, so a stride error shows up here as a missing or duplicated pair.
    """
    column_chunks, _, group_stride = grouped._geometry(
        n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n
    )
    shift = group_k.bit_length() - 1
    seen, duplicates = set(), 0
    for pid0 in range(column_chunks):
        for pid1 in range(group_stride):
            for group_base in range(pid1 * group_k, blocks, group_stride * group_k):
                for flat_row in range(group_k * rows_n):
                    column = pid0 * rows_n + (flat_row >> shift)
                    block = group_base + (flat_row & (group_k - 1))
                    if column < n and block < blocks:
                        if (block, column) in seen:
                            duplicates += 1
                        seen.add((block, column))
    return seen, duplicates


def _tile_geometry(n, k, blocks, group_k, rows_n, pid0, pid1):
    """``(columns, group_bases)`` one program covers, as the kernel computes them."""
    column_chunks, _, group_stride = grouped._geometry(
        n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n
    )
    assert 0 <= pid0 < column_chunks and 0 <= pid1 < group_stride
    columns = [pid0 * rows_n + within for within in range(rows_n)]
    group_bases = list(range(pid1 * group_k, blocks, group_stride * group_k))
    return columns, group_bases


def _wide_byte_tile(group_k, group_base):
    """Model the raw-byte load: the contiguous K-byte range one wide row covers.

    Returns ``(indices, slices)``: the whole range, and the same range cut into the
    16-byte slice each K block of the group owns -- which is what the reshape to
    ``[TILE, 16]`` produces, one tile row per ``(column, group)`` pair.
    """
    start = group_base * 16
    indices = [start + offset for offset in range(group_k * 16)]
    slices = [indices[block * 16:(block + 1) * 16] for block in range(group_k)]
    return indices, slices


def _gather_index():
    """The kernel's gather index: lane ``j`` reads byte ``j >> 1``."""
    return [lane >> 1 for lane in range(32)]


def _nibble(byte, lane):
    """The frozen static select: low nibble on an even lane, high on an odd one."""
    return byte & 15 if lane % 2 == 0 else byte >> 4


def _lane_addresses(index):
    """``(byte_index, nibble_parity)`` each lane reads under a given index."""
    return [(index[lane], lane & 1) for lane in range(32)]


@pytest.mark.parametrize("rows_n", [4, 8, 16, 32])
@pytest.mark.parametrize("group_k", [1, 2, 4])
@pytest.mark.parametrize("n,k", [(1, 32), (7, 96), (37, 160), (64, 256)])
def test_the_prefetch_tile_covers_every_pair_exactly_once(n, k, group_k, rows_n):
    blocks = k // 32
    covered, duplicates = _plan(n, blocks, group_k, rows_n)
    assert duplicates == 0
    assert covered == {(block, column) for block in range(blocks) for column in range(n)}


def test_a_ragged_k_group_is_masked_to_the_real_blocks():
    """K = 160 is five blocks; a GROUP_K = 4 tile covers one block in its last group."""
    covered, duplicates = _plan(37, 5, 4, 16)
    assert duplicates == 0
    assert {block for block, _ in covered} == set(range(5))
    # The wide load must drop the bytes of a block that does not exist, or it
    # would read into the next row of the packed weight.
    half = 160 // 2
    _, slices = _wide_byte_tile(4, 4)
    assert all(index < half for index in slices[0]), "block 4 is the last real block"
    assert [index for index in slices[1] if index < half] == [], "block 5 does not exist"


def test_the_strided_grid_still_covers_every_block_and_column():
    """A capped K-group axis must not leave a scratch row unwritten."""
    rows_n, group_k = 8, 4
    n, blocks = 20_000, 160
    column_chunks, _, group_stride = grouped._geometry(
        n, blocks, variant="grouped", group_k=group_k, rows_n=rows_n
    )
    assert column_chunks * group_stride <= grouped.MAX_PROGRAMS
    assert group_stride < math.ceil(blocks / group_k), "this fixture must actually cap the grid"
    step = group_stride * group_k
    reached = set()
    for pid1 in range(group_stride):
        for group_base in range(pid1 * group_k, blocks, step):
            for offset in range(group_k):
                if group_base + offset < blocks:
                    reached.add(group_base + offset)
    assert reached == set(range(blocks))
    assert column_chunks * rows_n >= n and (column_chunks - 1) * rows_n < n
    # And the promoted flat row really is `column * GROUP_K + group`.
    k = blocks * 32
    shift = group_k.bit_length() - 1
    for pid0 in (0, column_chunks - 1):
        columns, group_bases = _tile_geometry(n, k, blocks, group_k, rows_n, pid0, 0)
        group_base = group_bases[0]
        for flat_row in range(group_k * rows_n):
            assert flat_row >> shift == flat_row // group_k
            assert flat_row & (group_k - 1) == flat_row % group_k
            assert columns[flat_row >> shift] == pid0 * rows_n + flat_row // group_k
            assert group_base + (flat_row & (group_k - 1)) - group_base == flat_row % group_k


@pytest.mark.parametrize("group_k", [1, 2, 4])
def test_every_packed_byte_is_read_once_and_feeds_exactly_two_lanes(group_k):
    """Byte coverage of one program: no byte read twice, none skipped, both nibbles used."""
    k, rows_n = 256, 8
    blocks = k // 32
    shift = group_k.bit_length() - 1
    columns, group_bases = _tile_geometry(64, k, blocks, group_k, rows_n, 0, 0)
    index = _gather_index()
    for group_base in group_bases:
        indices, slices = _wide_byte_tile(group_k, group_base)
        assert len(set(indices)) == len(indices), "the wide load must not repeat a byte"
        assert indices == list(range(indices[0], indices[-1] + 1))
        expected = {(column, group_base + block, byte)
                    for column in columns for block in range(group_k)
                    for byte in slices[block]}
        seen, lane_use = set(), [0] * 32
        for flat_row in range(group_k * rows_n):
            column = columns[flat_row >> shift]
            group = flat_row & (group_k - 1)
            for lane in range(32):
                byte = slices[group][index[lane]]
                address = (column, group_base + group, byte)
                assert address in expected
                seen.add(address)
                lane_use[lane] += 1
        assert seen == expected, "every (column, block, byte) of the group must be read"
        assert len(seen) == len(columns) * group_k * 16
        assert lane_use == [group_k * rows_n] * 32, "each lane is used once per tile row"


def test_the_gather_duplicates_adjacent_bytes_and_the_select_takes_one_nibble_each():
    """For every packed byte value, two lanes read it and each takes its own nibble."""
    index = _gather_index()
    assert index == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                     8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
    for byte in range(256):
        values = [doubled(_nibble(byte, lane)) for lane in range(32)]
        low, high = doubled(byte & 15), doubled(byte >> 4)
        assert values[0::2] == [low] * 16, byte
        assert values[1::2] == [high] * 16, byte
        # The two lanes of one byte are the two K codes the packed byte carries,
        # and together they are the byte's own pair.
        assert (values[0], values[1]) == (low, high)


def test_the_byte_coverage_model_has_teeth():
    """A pairing that reads the right bytes in the wrong order must be rejected."""
    index = _gather_index()
    assert sorted(_lane_addresses(index)) == sorted(
        (byte, parity) for byte in range(16) for parity in (0, 1))
    wrong = [lane & 15 for lane in range(32)]
    assert wrong != index
    assert sorted(_lane_addresses(wrong)) != sorted(_lane_addresses(index)), (
        "the coverage model must not accept a lane-to-byte pairing that reads each byte "
        "once but pairs it with the wrong nibble"
    )
    # The values differ too, on a row of distinct bytes: lane 1 is the high nibble
    # of byte 0 under the contract, and the high nibble of byte 1 under `wrong`.
    distinct = [17 * byte for byte in range(16)]  # byte b has low nibble b, high nibble b
    assert doubled(distinct[index[1]] >> 4) == 0
    assert doubled(distinct[wrong[1]] >> 4) == 1


def test_the_int8_bitcast_view_preserves_all_256_packed_byte_values():
    """A bitcast is a same-width view, so 128..255 must come back through int8 and 0xff."""
    for byte in range(256):
        as_int8 = byte if byte < 128 else byte - 256  # the bit pattern int8 sees
        restored = as_int8 & 255                      # widen to int32, then mask
        assert restored == byte
        for lane in range(32):
            assert _nibble(restored, lane) == _nibble(byte, lane)
    # Teeth: the frozen select's odd lane shifts with no trailing mask, so the
    # byte must not still be sign-extended when it gets there.
    assert (-1 >> 4) == -1 and (-1 & 255) >> 4 == 15
    assert _nibble(-1, 1) != 15, "an unmasked sign-extended byte would decode wrongly"


# ------------------------------------------------------------------------------------------------
# The CPU backend: operand normalisation and dispatch, not kernel proof
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(32, 64), (7, 96), (64, 160)])
def test_the_packed_cpu_path_agrees_with_the_frozen_prototype(n, k):
    a, sa, packed, sb, _ = _packed_operands(n, k)
    expected = prototype.fp4_packed_decode_gemv(a, sa, packed, sb, backend="torch")
    assert_finite_nonvacuous(expected, context="packed cpu")
    actual = fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, backend="torch")
    assert_bits(actual, expected, context="packed cpu")


def test_a_nan_activation_code_turns_the_column_into_nan_like_the_accepted_operator():
    """E2M1 has no NaN code, so the non-finite case has to come from the activation."""
    n, k = 16, 64
    _, a_scale, packed, sb, _ = _packed_operands(n, k)
    nan_a = torch.full((1, k), 0x7F, dtype=torch.uint8)
    actual = fp4_packed_prefetch_decode_gemv(nan_a, a_scale, packed, sb, backend="torch")
    expected = prototype.fp4_packed_decode_gemv(nan_a, a_scale, packed, sb, backend="torch")
    assert bool(torch.isnan(actual).all()), "an E4M3-NaN activation code invalidates the column"
    assert_bits(actual, expected, context="nan activation")


@pytest.mark.parametrize("group_k,rows_n", [(4, 16), (2, 8), (1, 32), (4, 8)])
def test_the_cpu_result_does_not_depend_on_the_schedule(group_k, rows_n):
    """The schedule permutes which program writes a fragment, never the fragment."""
    n, k = 37, 160
    a, sa, packed, sb, _ = _packed_operands(n, k)
    reference = fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, backend="torch")
    actual = fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, group_k=group_k,
                                            rows_n=rows_n, backend="torch")
    assert_bits(actual, reference, context=f"schedule {group_k}x{rows_n}")


def test_the_scratch_helper_matches_the_frozen_layout():
    fragments, invalid = decode_prefetch_gemv_scratch(9, 128)
    assert tuple(fragments.shape) == (4, 9) and fragments.dtype == torch.float32
    assert tuple(invalid.shape) == (4, 9) and invalid.dtype == torch.uint8


def test_out_and_scratch_are_returned_in_place_on_the_cpu_path():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    out = torch.full((1, 8), -7.0, dtype=torch.float32)
    scratch = decode_prefetch_gemv_scratch(8, 64)
    returned = fp4_packed_prefetch_decode_gemv(
        a, sa, packed, sb, out=out, scratch=scratch, backend="torch"
    )
    assert returned is out
    assert bool((out != -7.0).all()), "the supplied output must have been written"


# ------------------------------------------------------------------------------------------------
# Refusals: the knobs are part of the contract
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("knob", ["group_k", "rows_n"])
def test_the_tile_sides_must_be_powers_of_two(knob):
    """The wide row length `GROUP_K * 16` goes through tl.arange, so it must be one too."""
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    arguments = {"group_k": GROUP_K, "rows_n": ROWS_N, knob: 6}
    with pytest.raises(ValueError, match="positive power of two"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, backend="torch", **arguments)


def test_a_tile_over_the_register_budget_is_refused_by_name():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="192 KiB unified buffer"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, group_k=8, rows_n=16,
                                       backend="torch")
    for group_k, rows_n in ((8, 16), (16, 8), (16, 16), (4, 32), (32, 8)):
        assert group_k * rows_n > MAX_TILE_ROWS
        with pytest.raises(ValueError, match="per-program tile budget"):
            fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, group_k=group_k,
                                            rows_n=rows_n, backend="torch")


def test_the_default_schedule_is_the_measured_ub_safe_configuration():
    assert (GROUP_K, ROWS_N) == (4, 16)
    assert GROUP_K * ROWS_N <= MAX_TILE_ROWS
    assert MAX_TILE_ROWS == 64
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, backend="torch")


def test_every_sweep_the_root_will_run_is_accepted():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    accepted = 0
    for group_k in (1, 2, 4):
        for rows_n in (4, 8, 16, 32):
            if group_k * rows_n > MAX_TILE_ROWS:
                continue
            fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, group_k=group_k,
                                            rows_n=rows_n, backend="torch")
            accepted += 1
    assert accepted >= 10, "the tuning grid should mostly be inside the budget"


def test_the_grid_bound_is_the_frozen_one_and_still_refuses_by_name():
    """The wrapper must not re-derive the program cap; the frozen check decides."""
    assert prefetch._geometry is grouped._geometry
    assert grouped.MAX_PROGRAMS == 65535
    with pytest.raises(ValueError, match="total-program cap"):
        prefetch._geometry(grouped.MAX_PROGRAMS * 64 + 1, 160, variant="grouped",
                           group_k=8, rows_n=64)


def test_two_activation_rows_are_refused_by_name():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="exactly one activation row"):
        fp4_packed_prefetch_decode_gemv(a.repeat(2, 1), sa, packed, sb, backend="torch")


def test_a_wrong_k_is_refused():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="positive K divisible by 32"):
        fp4_packed_prefetch_decode_gemv(a[:, :48], sa, packed, sb, backend="torch")


def test_packed_bytes_must_hold_two_codes_per_byte():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="two K codes per byte"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed[:, :-1].contiguous(), sb,
                                       backend="torch")


def test_a_cpu_tensor_cannot_ask_for_the_fused_backend():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    with pytest.raises(ValueError, match="NPU tensors only"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, backend="triton")


def test_the_output_must_not_overlap_an_operand():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    n = packed.shape[0]
    overlapping = packed.reshape(-1).view(torch.float32)[:n].reshape(1, n)
    with pytest.raises(ValueError, match="must not overlap"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, out=overlapping, backend="torch")


def test_the_scratch_pair_is_validated_even_on_the_cpu_path():
    a, sa, packed, sb, _ = _packed_operands(8, 64)
    wrong = (torch.empty((1, 8), dtype=torch.float32), torch.empty((2, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"\[K // 32, N\]"):
        fp4_packed_prefetch_decode_gemv(a, sa, packed, sb, scratch=wrong, backend="torch")


# ------------------------------------------------------------------------------------------------
# The unvalidated dense tile: refused by name, never substituted silently
# ------------------------------------------------------------------------------------------------

DENSE_ENTRIES = ("fp8_prefetch_decode_gemv", "fp4_int8_prefetch_decode_gemv")


@pytest.mark.parametrize("entry", DENSE_ENTRIES)
def test_the_unvalidated_dense_entry_points_refuse_by_name(entry):
    """The dense tile mismatched the accepted FP8 operator; it must not return a value.

    At ``1280 x 5120`` the FP8 form differed from the accepted operator on 71
    finite FP32 values while the grouped FP8 kernel matched, so the wide-row dense
    load/reshape is the suspect. Only the packed FP4 path is validated, so these
    two entry points raise instead of returning a value that could differ -- and
    the message has to name the operators to use instead, because a bare error
    would invite a caller to retry blindly.
    """
    function = getattr(prefetch, entry)
    with pytest.raises(RuntimeError) as caught:
        function(torch.zeros(1, 64, dtype=torch.uint8), torch.zeros(1, 2, dtype=torch.uint8),
                 torch.zeros(4, 64, dtype=torch.uint8), torch.zeros(1, 2, dtype=torch.uint8))
    message = str(caught.value)
    assert "not validated on hardware" in message
    assert "71 finite FP32 values" in message
    assert "Use block_gemm." in message and "or decode_grouped_gemv." in message


def test_the_refusal_is_unconditional_and_not_a_silent_fallback():
    """Even a fully valid dense operand is refused: there is no path that returns numbers."""
    for entry in DENSE_ENTRIES:
        function = getattr(prefetch, entry)
        for backend in ("torch", "auto", "triton"):
            with pytest.raises(RuntimeError, match="not validated on hardware"):
                function(torch.zeros(1, 64, dtype=torch.uint8),
                         torch.zeros(1, 2, dtype=torch.uint8),
                         torch.zeros(4, 64, dtype=torch.uint8),
                         torch.zeros(1, 2, dtype=torch.uint8), backend=backend)


def test_only_the_packed_fp4_entry_point_can_reach_the_kernel():
    """The launch path must be unreachable from the refused entries."""
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in DENSE_ENTRIES:
            continue
        calls = [_dotted(inner.func) for inner in ast.walk(node)
                 if isinstance(inner, ast.Call)]
        assert "_refuse_unvalidated" in calls, f"{node.name} must refuse immediately"
        assert "_run" not in calls, f"{node.name} must not reach the schedule"


def test_the_packed_entry_point_wires_the_fp4_codebook_flags():
    """The CPU oracle branches on `packed` first, so an inverted `fp4` flag is invisible there."""
    tree = ast.parse(WRAPPER_MODULE.read_text(encoding="utf-8"))
    found = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and _dotted(node.func) == "_run":
                keywords = {}
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    try:
                        keywords[keyword.arg] = ast.literal_eval(keyword.value)
                    except ValueError:
                        keywords[keyword.arg] = "<expression>"
                found[function.name] = {name: keywords.get(name) for name in ("fp4", "packed")}
    assert found == {"fp4_packed_prefetch_decode_gemv": {"fp4": True, "packed": True}}, found


# ------------------------------------------------------------------------------------------------
# Source contract: read the module, do not trust the docstrings
# ------------------------------------------------------------------------------------------------


def _tree():
    return ast.parse(KERNEL_MODULE.read_text(encoding="utf-8"))


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _tail(node):
    """Last component of a call target: ``to`` for ``x.to(...)``, ``gather`` for ``tl.gather``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _parents(node):
    """Map every child node to its parent, for following an operand's chain outward."""
    parents = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _jit_functions(tree):
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if _dotted(decorator).endswith("triton.jit"):
                yield node


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
        if (isinstance(target, ast.Name) and isinstance(value, ast.Call)
                and _dotted(value.func).endswith("tl.constexpr")):
            found[target.id] = ast.literal_eval(value.args[0])
    return found


def _free_names(function):
    bound = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return {node.id for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id not in bound}


def _imported(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {alias.asname or alias.name for alias in node.names}
    return names


def test_every_global_a_kernel_body_reads_is_constexpr_or_a_jit_function():
    tree = _tree()
    allowed = (_imported(tree) | set(_constants(tree))
               | {function.name for function in _jit_functions(tree)}
               | {"tl", "triton", "range", "float", "int", "abs"})
    problems = []
    for function in _jit_functions(tree):
        for name in sorted(_free_names(function)):
            if name not in allowed:
                problems.append(f"{function.name} reads {name!r}")
    assert not problems, (
        "the Triton front end rejects a plain module global read from a @jit body; "
        "annotate it as tl.constexpr: " + "; ".join(problems)
    )


def test_the_kernel_module_declares_only_integer_constexpr_globals():
    """A float NaN global would read as 'changed since compiled' on the second call."""
    constants = _constants(_tree())
    unstable = [name for name, value in constants.items() if isinstance(value, float)]
    assert not unstable, f"a float global is unstable across recompiles: {unstable}"


def test_the_new_schedule_defines_one_kernel_and_no_second_arithmetic():
    """Every arithmetic statement has to come from the frozen modules."""
    tree = _tree()
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions == ["decode_prefetch_fragment_kernel"], functions
    frozen = {"_cut", "_decode_e4m3", "_floor_exp", "_grid_exponent", "_nominal_e4",
              "_original_scale_code", "_pow2", "_scale_e8m0", "_e2m1_x2_lut",
              "_packed_nibble", "_fragment_rows", "_fragment_single_row",
              "_reduce_two_halves", "_fp4_value", "_fp4_nominal"}
    assert not (frozen & set(functions)), "a frozen helper must not be re-typed here"


def test_the_frozen_arithmetic_helpers_are_imported_from_their_own_modules():
    tree = _tree()
    grouped_module, from_block_gemm = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names = {alias.asname or alias.name for alias in node.names}
            if node.module.endswith("_ascend_decode_grouped_kernels"):
                grouped_module |= names
            elif node.module.endswith("_ascend_block_gemm_kernels"):
                from_block_gemm |= names
    assert {"_fragment_rows", "_packed_nibble", "_e2m1_x2_lut"} <= grouped_module
    assert "_original_scale_code" in from_block_gemm


def test_the_raw_weight_tile_reaches_the_gather_as_a_bitcast_int8():
    """The backend's gather takes int8 and refuses uint8; the bits must not move.

    A *bitcast* is a same-width re-interpretation, so every byte keeps its bits and
    the widen after the gather can restore the unsigned value. A numeric cast here
    would be a different (and wrong) statement, and a 4-byte widening before the
    gather would quadruple the tile in the unified buffer.
    """
    function = next(_jit_functions(_tree()))
    gathers = [node for node in ast.walk(function)
               if isinstance(node, ast.Call) and _dotted(node.func) == "tl.gather"]
    assert len(gathers) == 1, "one local gather is the whole expansion"
    source = gathers[0].args[0]
    assert isinstance(source, ast.Call) and _tail(source.func) == "to", (
        "the gather source must be the explicit bitcast view of the raw tile"
    )
    assert isinstance(source.func.value, ast.Name) and source.func.value.id == "pairs"
    assert _dotted(source.args[0]) == "tl.int8", (
        "int8 is one byte wide, so the bitcast cannot change a byte's bits"
    )
    bitcast = [keyword for keyword in source.keywords if keyword.arg == "bitcast"]
    assert bitcast and isinstance(bitcast[0].value, ast.Constant) and bitcast[0].value.value is True, (
        "without bitcast=True this would be a numeric conversion, not a same-width view"
    )
    # The reshaped tile itself must not be converted on the way to the view.
    for node in ast.walk(function):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "pairs"):
            conversions = [inner for inner in ast.walk(node.value)
                           if isinstance(inner, ast.Call) and _tail(inner.func) == "to"]
            assert not conversions, "pairs must stay the reshape of the loaded bytes"


def test_the_gathered_byte_is_widened_and_masked_only_after_the_gather():
    """Widen and `& 0xff` after the gather, or a sign-extended byte breaks the select."""
    function = next(_jit_functions(_tree()))
    parents = _parents(function)
    gather = next(node for node in ast.walk(function)
                  if isinstance(node, ast.Call) and _dotted(node.func) == "tl.gather")
    widen = parents[gather]
    assert isinstance(widen, ast.Attribute) and widen.attr == "to"
    widen_call = parents[widen]
    assert isinstance(widen_call, ast.Call)
    assert _dotted(widen_call.args[0]) == "tl.int32"
    masked = parents[widen_call]
    assert isinstance(masked, ast.BinOp) and isinstance(masked.op, ast.BitAnd), (
        "the widened byte must be masked back to 0..255 before the nibble select"
    )
    assert isinstance(masked.right, ast.Constant) and masked.right.value == 255
    assignment = parents[masked]
    assert isinstance(assignment, ast.Assign) and isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "gathered"
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert "tl.gather(byte_tile" not in source, "a uint8 gather is what the backend refused"

def test_the_gather_expands_adjacent_bytes_with_a_constant_index():
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert "pair_lanes = tl.arange(0, 32) >> 1" in source
    assert "gather_index = tl.broadcast_to(pair_lanes[None, :], (TILE, 32))" in source
    tree = _tree()
    function = next(_jit_functions(tree))
    gathers = [node for node in ast.walk(function)
               if isinstance(node, ast.Call) and _dotted(node.func) == "tl.gather"]
    assert len(gathers) == 1, "one local gather is the whole expansion"
    gather = gathers[0]
    assert _dotted(gather.args[0].func) == "pairs.to", "the source is pairs, viewed as int8"
    assert isinstance(gather.args[2], ast.Constant) and gather.args[2].value == 1
    # A gather index that is a runtime value would make the pairing dynamic.
    assert isinstance(gather.args[1], ast.Name) and gather.args[1].id == "gather_index"


def test_no_variable_nibble_shift_is_computed_here():
    """The nibble split has to stay the frozen constant-shift helper."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    for forbidden in ("& 15", ">> 4", "& 7"):
        assert forbidden not in source, (
            f"{forbidden!r} would be a local nibble split; use the frozen _packed_nibble"
        )
    assert "_packed_nibble" in _imported(_tree())


def test_the_weight_tile_is_addressed_by_tile_column_not_by_fragment_row():
    """The point of the module: one wide row per column, not one short row per fragment."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert "tile_columns[:, None] * (K // 2)" in source
    assert "tile_columns[:, None] * K" in source
    assert re.search(r"(?<!tile_)columns\[:, None\]", source) is None, (
        "the packed/dense tile must be loaded per tile column, not per fragment row"
    )


def test_the_tile_stays_two_dimensional():
    tree = _tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name not in {"tl.reshape", "tl.broadcast_to"}:
            continue
        shape = node.args[1]
        assert isinstance(shape, ast.Tuple) and len(shape.elts) == 2, (
            f"{name} must build a 2-D shape; the backend has no proven 3-D reduction"
        )
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    for forbidden in ("[None, :, None]", "[None, None, :]", "[:, :, None]", "[None, :, :]"):
        assert forbidden not in source, f"{forbidden!r} would make a 3-D operand"


def test_no_local_reduction_is_added():
    """All reductions stay inside the imported fragment, so none can be re-assured."""
    tree = _tree()
    reductions = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and _dotted(node.func) in {"tl.sum", "tl.max"}]
    assert not reductions, "the two-half reduction is imported; do not re-state it here"


def test_index64_widens_before_any_offset_product():
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert ".to(tl.int64)" in source
    for line in source.splitlines():
        if ".to(tl.int64)" not in line:
            continue
        head = line.split(".to(tl.int64)")[0].rstrip()
        assert head.endswith(("columns", "tile_columns", "group_base")), (
            f"widening is applied to {head!r}; it must be a bare index name, because a "
            "product computed in int32 before the cast has already overflowed"
        )


def test_the_accumulation_pass_is_the_frozen_kernel_unchanged():
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    assert "decode_accumulate_kernel" in _imported(ast.parse(source))
    kernels = _tree()
    defined = {node.name for node in kernels.body if isinstance(node, ast.FunctionDef)}
    assert "decode_accumulate_kernel" not in defined, (
        "pass 2 belongs to the frozen prototype; re-defining it here would let the strict "
        "increasing-K accumulation be changed silently"
    )
    assert "def decode_accumulate_kernel" not in source


def test_the_wrapper_reuses_the_frozen_scratch_and_geometry():
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    imported = _imported(ast.parse(source))
    assert {"decode_gemv_scratch", "_torch_reference", "_prepare_scratch"} <= imported
    assert {"_geometry", "MAX_TILE_ROWS", "_power_of_two"} <= imported
    assert "torch.empty" not in source, "the scratch and the output are the frozen helpers'"

# ------------------------------------------------------------------------------------------------
# Acceptance on hardware: bit parity with the frozen prototype and the accepted operator
#
# The prototype reads the same weight one fragment (16 bytes) at a time and the
# accepted block GEMM decodes it in a different kernel family entirely, so parity
# against both is real evidence that the wide-row access changed no value.
# ------------------------------------------------------------------------------------------------

requires_device = pytest.mark.skipif(
    device().type == "cpu",
    reason="set RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 with a leased card to run the fused kernel",
)


def _to(tensor, target):
    return tensor.to(target)


@requires_device
@pytest.mark.parametrize("n,k", [(32, 64), (7, 96), (37, 160), (2304, 5120), (5120, 2304)])
def test_the_packed_kernel_matches_the_frozen_prototype_bit_for_bit(n, k):
    """The primary failure this schedule could have: a byte gathered to the wrong lane."""
    target = device()
    a, sa, packed, sb, _ = _packed_operands(n, k)
    expected = prototype.fp4_packed_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target)
    )
    assert_finite_nonvacuous(expected, context="packed npu")
    actual = fp4_packed_prefetch_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        backend=backend(),
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context=f"packed npu {n}x{k}")


@requires_device
@pytest.mark.parametrize("group_k", [1, 2, 4])
@pytest.mark.parametrize("rows_n", [8, 16])
def test_the_wide_row_schedule_is_numerically_inert_on_the_card(group_k, rows_n):
    """The sweep the profiling plan runs must not move a single bit."""
    if group_k * rows_n > MAX_TILE_ROWS:
        pytest.skip("outside the measured per-program tile budget")
    target = device()
    a, sa, packed, sb, _ = _packed_operands(96, 160, seed=67)
    tensors = [_to(tensor, target) for tensor in (a, sa, packed, sb)]
    reference = fp4_packed_prefetch_decode_gemv(*tensors, group_k=1, rows_n=16,
                                                backend=backend())
    actual = fp4_packed_prefetch_decode_gemv(*tensors, group_k=group_k, rows_n=rows_n,
                                             backend=backend())
    torch.npu.synchronize()
    assert_bits(actual, reference, context=f"group_k={group_k} rows_n={rows_n}")


@requires_device
def test_a_ragged_weight_does_not_read_past_the_last_block():
    """K = 96 is three blocks, so a GROUP_K = 4 tile's last group is one real block."""
    target = device()
    a, sa, packed, sb, _ = _packed_operands(37, 96, seed=71)
    expected = prototype.fp4_packed_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target)
    )
    actual = fp4_packed_prefetch_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        group_k=4, rows_n=16, backend=backend(),
    )
    torch.npu.synchronize()
    assert_bits(actual, expected, context="ragged K npu")


@requires_device
def test_an_e8m0_scale_of_255_nans_like_the_accepted_operator():
    target = device()
    n, k = 8, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=101, scale_code=255)
    tensors = [_to(tensor, target) for tensor in (a, sa, packed, sb)]
    expected = prototype.fp4_packed_decode_gemv(*tensors)
    actual = fp4_packed_prefetch_decode_gemv(*tensors, backend=backend())
    torch.npu.synchronize()
    assert_bits(actual, expected, context="scale 255 npu")


@requires_device
def test_a_cancelling_block_sequence_keeps_the_strict_k_order():
    """Alternating large signs is where a reassociated sum would show up."""
    target = device()
    n, k = 8, 96
    a = torch.full((1, k), 0x38, dtype=torch.uint8)  # E4M3 1.0
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    signs = torch.tensor([0, 8], dtype=torch.int64).repeat_interleave(k // 2)[:k]
    nibbles = torch.where((torch.arange(k) // 32) % 2 == 0,
                          torch.full((k,), 7, dtype=torch.int64), signs)
    weights = nibbles.unsqueeze(0).repeat(n, 1)
    packed = (weights[:, 0::2] | (weights[:, 1::2] << 4)).to(torch.uint8)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    tensors = [_to(tensor, target) for tensor in (a, sa, packed, sb)]
    expected = prototype.fp4_packed_decode_gemv(*tensors)
    actual = fp4_packed_prefetch_decode_gemv(*tensors, backend=backend())
    torch.npu.synchronize()
    assert_bits(actual, expected, context="cancelling npu")


@requires_device
def test_the_device_path_writes_the_supplied_output_and_reuses_its_scratch():
    target = device()
    n, k = 8, 64
    a, sa, packed, sb, _ = _packed_operands(n, k, seed=107)
    out = torch.full((1, n), -7.0, dtype=torch.float32, device=target)
    scratch = decode_prefetch_gemv_scratch(n, k, device=target)
    returned = fp4_packed_prefetch_decode_gemv(
        _to(a, target), _to(sa, target), _to(packed, target), _to(sb, target),
        out=out, scratch=scratch, backend=backend(),
    )
    torch.npu.synchronize()
    assert returned is out
    assert bool((out != -7.0).all())
    assert bool(scratch[0].any() or scratch[1].any()), "the fragment scratch must have been written"


def test_index64_is_decided_from_the_padded_reach_not_the_logical_one():
    """A masked tail lane forms its address before the mask suppresses the access."""
    assert prefetch._needs_index64(2304, 5120, 160, 16, 4) is False
    assert prefetch._needs_index64(5120, 2304, 72, 16, 4) is False
    # `n * k` stays below 2**31, but the last tile row crosses it.
    n, k, blocks, rows_n, group_k = 2**26 - 1, 32, 1, 16, 4
    assert n * k < 2**31
    assert n * k + (rows_n - 1) * k >= 2**31, "this fixture must cross the cap only when padded"
    assert prefetch._needs_index64(n, k, blocks, rows_n, group_k) is True
    # And the launch must ask that helper rather than re-deriving the reach.
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    assert "index64 = _needs_index64(n, k, blocks, rows_n, group_k)" in source
    assert "index64 = max(n * k" not in source, (
        "the logical reach would leave the last tile's dead lanes unprotected"
    )
