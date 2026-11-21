"""Tests for the tile-major resident packed-FP4 decode GEMV.

What each layer can and cannot establish:

* **The permutation is exact.** ``pack_tile_major`` is only a byte permutation, so
  the inverse must return the *inputs* byte for byte, for the two released expert
  shapes and for a ragged ``N`` that forces padding. This is asserted on the bytes,
  never on values, and the layout is re-derived here from the documented contract
  with explicit indexing rather than by re-running the same ``permute``.
* **The CPU path is the accepted operator.** ``backend="torch"`` inverts the
  permutation and runs the frozen widening/half-scale/``fp4_int8_block_gemm``
  chain, so CPU parity against the frozen prototype proves this wrapper's operand
  normalisation, permutation and dispatch -- **and nothing about the Ascend
  kernel**. The kernel's arithmetic is the frozen fragment by import; only the
  leased-hardware test at the bottom asserts bit parity for it::

      RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 RELIC_DECODE_GEMV_TEST_BACKEND=triton \\
          python -m pytest tests/test_tile_major_packed_gemv.py -q -rs

* **The access contract is read, not trusted.** The weight tile has to be one flat
  contiguous ``GROUP_K * 256``-byte load with no per-row stride term, the raw tile
  has to stay one byte until after the gather, the gather has to go through the
  ``int8`` bitcast with the ``0xff`` widen *after* it, the arithmetic helpers have
  to be imported from the frozen modules rather than re-typed, and the index width
  has to be chosen before the offset products.

No tolerance is used: finite values are compared as bit patterns and NaN outputs
by classification, matching the accepted operator's own contract.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import decode_block_gemv as prototype  # noqa: E402
from relic_core.kernels import tile_major_packed_gemv as tile  # noqa: E402
from relic_core.kernels.tile_major_packed_gemv import (  # noqa: E402
    ACCUMULATE_COLUMNS,
    GROUP_K,
    IDENTITY_SCALE,
    NTILE_ROWS,
    fp4_packed_tile_major_decode_gemv,
    pack_tile_major,
    tile_major_packed_reference,
    tile_major_packed_scratch,
    unpack_tile_major,
)

WRAPPER_MODULE = Path(tile.__file__).resolve()
KERNEL_MODULE = WRAPPER_MODULE.parent / "_ascend_tile_major_packed_kernels.py"

#: The two released expert projections, as the checkpoint stores them packed:
#: ``(N, K)`` with ``N`` a multiple of the 16-row tile.
W1 = (2304, 5120)
W2 = (5120, 2304)
RELEASED = (W1, W2)


def _rand_bytes(shape, generator, low=0, high=256):
    return torch.randint(low, high, shape, generator=generator, dtype=torch.int64).to(torch.uint8)


def _operands(n, k, generator, *, scale_low=100, scale_high=150):
    a = _rand_bytes((1, k), generator)
    sa = _rand_bytes((1, k // 32), generator, scale_low, scale_high)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator, scale_low, scale_high)
    return a, sa, b, sb


def _finite_codes(codes):
    """The same bytes with every E4M3 NaN code replaced by zero.

    A random byte row carries an E4M3 NaN code (``byte & 127 == 127``) about once per
    128 bytes, and the accepted operator poisons a whole column on any one of them, so
    a test that is *about* weights has to start from finite activation codes.
    """
    poisoned = (codes & 0x7F) == 0x7F
    return torch.where(poisoned, torch.zeros_like(codes), codes)


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def _same_bits(left, right):
    """Bit equality over the whole tensor, with NaN bit-as-is rather than NaN != NaN."""
    return torch.equal(_bits(left), _bits(right))


# ------------------------------------------------------------------------------------------------
# The permutation: byte identity, coverage, padding, size
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", RELEASED)
def test_the_permutation_round_trips_the_released_expert_shapes(n, k):
    """``unpack(pack(b, sb))`` must return both inputs byte for byte."""
    generator = torch.Generator().manual_seed(101)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    back_b, back_sb = unpack_tile_major(b_tile, s_tile)
    assert torch.equal(back_b, b)
    assert torch.equal(back_sb, sb)


@pytest.mark.parametrize("n,k", RELEASED)
def test_the_bank_costs_the_same_bytes_as_the_logical_weight(n, k):
    """One bank, not two: the permutation must not expand the expert's footprint."""
    generator = torch.Generator().manual_seed(102)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    assert b_tile.numel() + s_tile.numel() == b.numel() + sb.numel()
    assert b_tile.numel() == n * k // 2
    assert s_tile.numel() == n * k // 32
    assert b_tile.dtype is torch.uint8 and s_tile.dtype is torch.uint8
    assert b_tile.is_contiguous() and s_tile.is_contiguous()


def test_every_packed_code_byte_and_every_scale_byte_survives_the_permutation():
    """All 256 byte values in both banks, compared as multisets, not as a sum."""
    n, k = 128, 96
    b = torch.arange(n * (k // 2), dtype=torch.int64).remainder(256).to(torch.uint8)
    b = b.reshape(n, k // 2)
    sb = torch.arange(n * (k // 32), dtype=torch.int64).remainder(256).to(torch.uint8)
    sb = sb.reshape(n, k // 32)
    assert b.shape == (n, k // 2) and sb.shape == (n, k // 32)
    assert sorted(b.unique().tolist()) == list(range(256))
    assert sorted(sb.unique().tolist()) == list(range(256))

    b_tile, s_tile = pack_tile_major(b, sb)
    back_b, back_sb = unpack_tile_major(b_tile, s_tile)
    assert torch.equal(back_b, b) and torch.equal(back_sb, sb)
    # And the banks hold exactly the inputs' byte multisets.
    for packed, flat in ((b_tile, b), (s_tile, sb)):
        assert torch.equal(packed.flatten().sort().values, flat.flatten().sort().values)


def test_the_layout_is_re_derived_independently_rather_than_by_the_same_permute():
    """The contract, checked by explicit indexing: ``b_tile[t,b,r,c]``/``s_tile[t,b,r]``.

    Re-running ``permute`` would only prove the permute is self-consistent. This
    states the mapping in the module's own words -- tile ``t`` holds logical rows
    ``t * 16 .. t * 16 + 15``, its K block ``b`` holds that row's bytes
    ``b * 16 .. b * 16 + 15``, and the scale is that row's byte ``b`` -- and checks
    it at coordinates drawn across the whole bank.
    """
    n, k = 48, 160  # 3 tiles, 5 blocks
    generator = torch.Generator().manual_seed(103)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    ntiles, blocks = b_tile.shape[0], b_tile.shape[1]
    assert (ntiles, blocks) == (3, 5)

    for t in (0, 1, ntiles - 1):
        for block in (0, 1, blocks - 1):
            for r in (0, 1, 15):
                logical = t * 16 + r
                assert torch.equal(b_tile[t, block, r], b[logical, block * 16 : (block + 1) * 16])
                assert int(s_tile[t, block, r]) == int(sb[logical, block])
    # Nothing else is reachable: the two banks hold only these coordinates.
    assert b_tile.shape[2:] == (16, 16) and s_tile.shape[2:] == (16,)
    assert ntiles * 16 == n and blocks * 32 == k


def test_a_padded_bank_is_zero_codes_and_identity_scales_in_its_extra_rows():
    """``N`` not a multiple of 16 pads with code 0 bytes and scale 127, masked out."""
    n, k = 20, 96  # 2 tiles, 3 blocks; 12 padding rows
    generator = torch.Generator().manual_seed(104)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    assert b_tile.shape == (2, 3, 16, 16)
    assert s_tile.shape == (2, 3, 16)

    # Rows 20..31 of the padded bank are tile 1's rows 4..15.
    assert not torch.any(b_tile[1, :, 4:, :])
    assert torch.all(s_tile[1, :, 4:] == IDENTITY_SCALE)
    # The real rows are untouched, and every padding byte is exactly the documented value.
    assert torch.equal(b_tile[1, :, :4, :], b[16:20].reshape(4, 3, 16).permute(1, 0, 2))
    # The real rows still round trip exactly, and the trim returns only those.
    back_b, back_sb = unpack_tile_major(b_tile, s_tile, n_output=n)
    assert tuple(back_b.shape) == (n, k // 2) and tuple(back_sb.shape) == (n, k // 32)
    assert torch.equal(back_b, b) and torch.equal(back_sb, sb)


def test_the_untrimmed_inverse_reports_the_physical_row_count():
    """Without ``n_output`` the inverse returns the padded extent, which is ``NTILE_ROWS``-aligned."""
    n, k = 20, 96
    generator = torch.Generator().manual_seed(105)
    b = _rand_bytes((n, k // 2), generator)
    sb = _rand_bytes((n, k // 32), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    b_pad, sb_pad = unpack_tile_major(b_tile, s_tile)
    assert b_pad.shape[0] == 32 and sb_pad.shape[0] == 32
    assert b_pad.shape[0] % NTILE_ROWS == 0
    assert torch.equal(b_pad[:n], b) and torch.equal(sb_pad[:n], sb)


def test_the_pack_accepts_an_int8_container_and_keeps_its_bits():
    """The banks are byte containers: an ``int8`` input is a view, not a conversion."""
    generator = torch.Generator().manual_seed(106)
    b = _rand_bytes((32, 16), generator)   # K = 32, one K block
    sb = _rand_bytes((32, 1), generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    b_tile_i8, s_tile_i8 = pack_tile_major(b.view(torch.int8), sb.view(torch.int8))
    assert b_tile_i8.view(torch.uint8).equal(b_tile)
    assert s_tile_i8.view(torch.uint8).equal(s_tile)


# ------------------------------------------------------------------------------------------------
# The CPU path is the accepted operator
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(64, 128), (48, 96), (32, 64), (16, 64)])
def test_the_tile_major_cpu_path_matches_the_frozen_prototype_bit_for_bit(n, k):
    """The wrapper must not introduce a second reduction: bit-for-bit, no tolerance."""
    generator = torch.Generator().manual_seed(107)
    a, sa, b, sb = _operands(n, k, generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    expected = prototype.fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    actual = fp4_packed_tile_major_decode_gemv(a, sa, b_tile, s_tile, backend="torch")
    assert actual.shape == (1, n) and actual.dtype is torch.float32
    assert _same_bits(actual, expected)


def test_the_reference_helper_and_the_entry_point_agree():
    """The documented reference seam and the dispatching entry point are one operator."""
    generator = torch.Generator().manual_seed(108)
    n, k = 32, 96
    a, sa, b, sb = _operands(n, k, generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    seam = tile_major_packed_reference(a, sa, b_tile, s_tile)
    entry = fp4_packed_tile_major_decode_gemv(a, sa, b_tile, s_tile, backend="torch")
    assert _same_bits(seam, entry)


def test_scale_zero_scale_nan_zero_weights_and_cancelling_blocks_all_match_the_prototype():
    """The accepted operator's edge cases, reached through the permutation.

    Scale byte 0 is ``2**-127`` (a real, tiny scale), 255 is E8M0 NaN, an all-zero
    weight block is all code 0, and a block whose codes are the sign-flipped copies
    of another's makes the K chain cancel -- none of which the permutation may
    disturb, because it does not touch a value.
    """
    n, k = 32, 128
    generator = torch.Generator().manual_seed(109)
    a, sa, b, sb = _operands(n, k, generator, scale_low=0, scale_high=256)
    a = _finite_codes(a)

    # Row 0: zero weights. Row 1: scale 0 everywhere. Row 2: scale 255 (NaN).
    b[0] = 0
    sb[1] = 0
    sb[2] = 255
    # Row 3: block 1 is the sign-flipped copy of block 0 (nibble bit 3 negates E2M1).
    block0 = b[3, 0:16].clone()
    b[3, 16:32] = block0 ^ 0x88
    b_tile, s_tile = pack_tile_major(b, sb)
    expected = prototype.fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    actual = fp4_packed_tile_major_decode_gemv(a, sa, b_tile, s_tile, backend="torch")
    assert _same_bits(actual, expected)
    # Row 2's weight scale is E8M0 NaN, so that column is NaN in the accepted
    # operator and must be NaN here too; the zero-weight row stays finite, so the
    # comparison is not vacuous on finite values either.
    assert torch.isnan(expected[0, 2]) and torch.isnan(actual[0, 2])
    assert torch.isfinite(expected[0, 0]) and torch.isfinite(actual[0, 0])


def test_a_nan_activation_code_poisons_every_column_exactly_as_the_prototype_does():
    """The invalid flag is OR'd across K blocks, so one NaN code NaNs the whole row.

    That is the accepted operator's behaviour, not a defect introduced here, and the
    permutation must not change which columns it reaches.
    """
    n, k = 32, 128
    generator = torch.Generator().manual_seed(115)
    a, sa, b, sb = _operands(n, k, generator)
    for offset in (0, 96):  # first and last K block of the row
        poisoned = a.clone()
        poisoned[0, offset] = 0x7F  # E4M3 byte with (b & 127) == 127 is NaN
        b_tile, s_tile = pack_tile_major(b, sb)
        expected = prototype.fp4_packed_decode_gemv(poisoned, sa, b, sb, backend="torch")
        actual = fp4_packed_tile_major_decode_gemv(poisoned, sa, b_tile, s_tile, backend="torch")
        assert torch.isnan(expected).all(), "the accepted operator poisons every column"
        assert _same_bits(actual, expected)


def test_the_inverse_names_a_multi_row_prefill_seam():
    """A prefill reuses the accepted dense path through the exact inverse bytes.

    The values are compared against the frozen prototype run on the *logical*
    matrix, which is the statement a prefill consumer needs: the inverse is not an
    approximation of the checkpoint's bytes.
    """
    n, k = 32, 96
    generator = torch.Generator().manual_seed(110)
    a, sa, b, sb = _operands(n, k, generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    back_b, back_sb = unpack_tile_major(b_tile, s_tile)
    assert torch.equal(back_b, b) and torch.equal(back_sb, sb)
    assert _same_bits(
        prototype.fp4_packed_decode_gemv(a, sa, back_b, back_sb, backend="torch"),
        prototype.fp4_packed_decode_gemv(a, sa, b, sb, backend="torch"),
    )


def test_the_scratch_is_the_frozen_two_pass_pair():
    fragments, invalid = tile_major_packed_scratch(32, 96, device="cpu")
    assert fragments.shape == (3, 32) and fragments.dtype is torch.float32
    assert invalid.shape == (3, 32) and invalid.dtype is torch.uint8


def test_a_caller_supplied_out_and_scratch_are_used_in_place():
    n, k = 32, 96
    generator = torch.Generator().manual_seed(111)
    a, sa, b, sb = _operands(n, k, generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    out = torch.empty((1, n), dtype=torch.float32)
    scratch = tile_major_packed_scratch(n, k)
    returned = fp4_packed_tile_major_decode_gemv(
        a, sa, b_tile, s_tile, backend="torch", out=out, scratch=scratch
    )
    assert returned is out
    assert _same_bits(out, prototype.fp4_packed_decode_gemv(a, sa, b, sb, backend="torch"))


# ------------------------------------------------------------------------------------------------
# Refusals: every one names what was wrong, and none of them silently rewrites
# ------------------------------------------------------------------------------------------------


def _small():
    generator = torch.Generator().manual_seed(112)
    n, k = 32, 96
    a, sa, b, sb = _operands(n, k, generator)
    b_tile, s_tile = pack_tile_major(b, sb)
    return a, sa, b_tile, s_tile


def test_a_wrong_bank_rank_or_tile_geometry_is_refused():
    a, sa, b_tile, s_tile = _small()
    with pytest.raises(ValueError, match="4-D tile tensor"):
        fp4_packed_tile_major_decode_gemv(a, sa, b_tile.reshape(2, 3, 256), s_tile, backend="torch")
    with pytest.raises(ValueError, match=r"16, 16\]"):
        fp4_packed_tile_major_decode_gemv(
            a, sa, torch.zeros((2, 3, 16, 8), dtype=torch.uint8), s_tile, backend="torch"
        )
    with pytest.raises(ValueError, match=r"\[NTILES, K // 32, 16\]"):
        fp4_packed_tile_major_decode_gemv(
            a, sa, b_tile, torch.zeros((2, 3, 8), dtype=torch.uint8), backend="torch"
        )


def test_a_non_contiguous_bank_is_refused_rather_than_copied():
    a, sa, b_tile, s_tile = _small()
    transposed = b_tile.transpose(0, 1)
    assert not transposed.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        fp4_packed_tile_major_decode_gemv(a, sa, transposed, s_tile, backend="torch")


def test_a_floating_bank_is_refused():
    a, sa, b_tile, s_tile = _small()
    with pytest.raises(TypeError, match="dtype"):
        fp4_packed_tile_major_decode_gemv(
            a, sa, b_tile.to(torch.float32), s_tile, backend="torch"
        )


def test_a_k_mismatch_between_the_bank_and_the_activation_is_refused():
    a, sa, b_tile, s_tile = _small()
    short_a = a[:, :64].contiguous()
    short_sa = sa[:, :2].contiguous()
    with pytest.raises(ValueError, match="the tile-major bank holds K"):
        fp4_packed_tile_major_decode_gemv(short_a, short_sa, b_tile, s_tile, backend="torch")


@pytest.mark.parametrize("n_output,error", [(0, ValueError), (33, ValueError), (-1, ValueError),
                                            (1.0, TypeError), (True, TypeError)])
def test_an_invalid_n_output_is_refused(n_output, error):
    a, sa, b_tile, s_tile = _small()
    with pytest.raises(error):
        fp4_packed_tile_major_decode_gemv(
            a, sa, b_tile, s_tile, backend="torch", n_output=n_output
        )


def test_a_trimmed_n_output_changes_the_output_width_only():
    a, sa, b_tile, s_tile = _small()
    physical = fp4_packed_tile_major_decode_gemv(a, sa, b_tile, s_tile, backend="torch")
    trimmed = fp4_packed_tile_major_decode_gemv(
        a, sa, b_tile, s_tile, backend="torch", n_output=20
    )
    assert trimmed.shape == (1, 20)
    assert torch.equal(_bits(trimmed), _bits(physical)[:, :20])


@pytest.mark.parametrize("group_k", [0, 3, 8, True])
def test_an_unsupported_group_k_is_refused(group_k):
    a, sa, b_tile, s_tile = _small()
    with pytest.raises((ValueError, TypeError), match="group_k"):
        fp4_packed_tile_major_decode_gemv(
            a, sa, b_tile, s_tile, backend="torch", group_k=group_k
        )


def test_an_overlapping_out_is_refused():
    a, sa, b_tile, s_tile = _small()
    overlapping = b_tile.view(torch.float32).reshape(-1, 32)[:1]
    assert overlapping.shape == (1, 32)
    with pytest.raises(ValueError, match="overlap"):
        fp4_packed_tile_major_decode_gemv(
            a, sa, b_tile, s_tile, backend="torch", out=overlapping
        )


def test_the_fused_backend_refuses_a_cpu_tensor_by_name():
    """No silent fallback: asking for the fused kernel on CPU is an error."""
    a, sa, b_tile, s_tile = _small()
    with pytest.raises(ValueError, match="NPU tensors only"):
        fp4_packed_tile_major_decode_gemv(a, sa, b_tile, s_tile, backend="triton")


def test_a_pack_of_ragged_rows_is_refused():
    generator = torch.Generator().manual_seed(113)
    b = _rand_bytes((16, 24), generator)  # 24 packed bytes = 1.5 K blocks
    sb = _rand_bytes((16, 1), generator)
    with pytest.raises(ValueError, match="not a whole number"):
        pack_tile_major(b, sb)


# ------------------------------------------------------------------------------------------------
# The source contract: read, not trusted
# ------------------------------------------------------------------------------------------------


def _kernel_function(name):
    tree = ast.parse(KERNEL_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {KERNEL_MODULE.name}")


def _assigned_names(body):
    return {
        node.targets[0].id
        for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def test_the_kernel_imports_the_accepted_arithmetic_instead_of_re_typing_it():
    """The fragment, the nibble select and the codebook must be the frozen ones."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "_ascend_decode_grouped_kernels"
        for alias in node.names
    }
    assert {"_fragment_rows", "_packed_nibble", "_e2m1_x2_lut"} <= imported
    # And none of them is re-implemented here.
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert defined == {"tile_major_packed_fragment_kernel"}
    body = _kernel_function("tile_major_packed_fragment_kernel").body
    calls = {
        node.func.id
        for node in ast.walk(_kernel_function("tile_major_packed_fragment_kernel"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"_fragment_rows", "_packed_nibble", "_e2m1_x2_lut"} <= calls
    assert _assigned_names(body) >= {"flat", "within", "group_of_row"}


def test_the_weight_tile_is_one_flat_contiguous_run_with_no_row_stride_term():
    """The point of the layout: the tile address must not carry a per-row stride.

    The load's index expression is required to be ``<scalar base> + <1-D arange>``
    where the arange covers ``GROUP_K * 256`` bytes, and to contain no subscripted
    per-row tensor -- a ``[... , None]`` row term is exactly the discrete access
    this layout exists to remove.
    """
    node = _kernel_function("tile_major_packed_fragment_kernel")
    loads = [call for call in ast.walk(node) if isinstance(call, ast.Call)
             and isinstance(call.func, ast.Attribute) and call.func.attr == "load"]
    # The tile load is the one whose index is `BT + ...`.
    def base_name(expression):
        while isinstance(expression, ast.BinOp):
            expression = expression.left
        return expression.id if isinstance(expression, ast.Name) else None

    tile_loads = [call for call in loads if base_name(call.args[0]) == "BT"]
    assert len(tile_loads) == 1, "the kernel must read the resident tile at exactly one site"
    index = tile_loads[0].args[0]
    assert not any(isinstance(sub, ast.Subscript) for sub in ast.walk(index)), (
        "the tile index must be a scalar base plus a flat arange, not a per-row gather"
    )
    source = ast.unparse(index)
    assert "group_bytes" in source
    assert "tile_base" in source

    # `group_bytes` is the flat `arange(0, GROUP_K * 256)` and is only ever added.
    assignments = {
        target.id: value
        for node_ in ast.walk(node)
        if isinstance(node_, ast.Assign)
        and len(node_.targets) == 1 and isinstance(node_.targets[0], ast.Name)
        for target, value in [(node_.targets[0], node_.value)]
    }
    arange = assignments["group_bytes"]
    assert isinstance(arange, ast.Call) and arange.func.attr == "arange"
    upper = ast.unparse(arange.args[1])
    assert "GROUP_K * 256" in upper


def test_the_raw_tile_stays_one_byte_until_after_the_gather():
    """``uint8`` reaches the backend's gather only as an ``int8`` bitcast, widened after.

    The measured refusal is "Expected dtype .../int8, but got uint8", so the order of
    the bitcast, the gather and the ``& 255`` widen is the contract -- widening first
    would also quadruple the tile in the unified buffer.
    """
    node = _kernel_function("tile_major_packed_fragment_kernel")
    body_source = ast.unparse(node)
    gather_at = body_source.index("tl.gather")
    gather_call = body_source[gather_at : body_source.index("& 255", gather_at)]
    assert "pairs.to(tl.int8, bitcast=True)" in gather_call, (
        "the raw tile must reach the gather through the same-width int8 bitcast"
    )
    assert "& 255" in body_source[gather_at:], "the unsigned widen must follow the gather"
    # The reshape is two-dimensional, which is the shape this backend proves.
    assert "tl.reshape(raw, (GROUP_K * 16, 16))" in body_source


def test_the_index_width_is_chosen_before_the_offset_products():
    """Both offset bases are widened under ``INDEX64`` before any product."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    body = ast.unparse(_kernel_function("tile_major_packed_fragment_kernel"))
    assert "tile = tile.to(tl.int64)" in body
    assert "columns = columns.to(tl.int64)" in body
    assert "group_base = group_base.to(tl.int64)" in body
    # The widening statements come before the products that use them.
    assert body.index("tile = tile.to(tl.int64)") < body.index("tile_base = ")
    assert body.index("tile = tile.to(tl.int64)") < body.index("scale_base = ")
    assert body.index("group_base = group_base.to(tl.int64)") < body.index("tile_base = ")
    assert body.index("group_base = group_base.to(tl.int64)") < body.index("scale_base = ")
    # ... and nothing is cast to int64 after being multiplied.
    assert ").to(tl.int64)" not in body.replace("tile = tile.to(tl.int64)", "").replace(
        "columns = columns.to(tl.int64)", ""
    ).replace("group_base = group_base.to(tl.int64)", "")
    assert "INDEX64" in source


_SHAPE_CALLS = {"arange", "reshape", "broadcast_to", "full", "zeros", "ones", "empty"}
_CONSTEXPR_ANNOTATION = "constexpr"


def _dynamic_shape_names(source):
    """Names a ``@jit`` body uses in a shape/``arange`` position that are not constexpr.

    Triton requires ``arange``'s arguments and every ``reshape``/``broadcast_to``
    shape entry to be compile-time constants. A kernel argument annotated
    ``tl.constexpr`` is one; a plain local is not, however constant its own
    expression looks -- which is exactly how the first real compile of this kernel
    failed, with ``ValueError("arange's arguments must be of type tl.constexpr")`` at
    ``tl.arange(0, tile_rows)`` for ``tile_rows = GROUP_K * 16``.

    Returns the offending names, so a test can require an empty set and a negative
    control can require the set to be non-empty.
    """
    tree = ast.parse(source)
    offenders = set()
    for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        constexpr_parameters = {
            argument.arg
            for argument in function.args.args + function.args.kwonlyargs
            if argument.annotation is not None
            and _CONSTEXPR_ANNOTATION in ast.unparse(argument.annotation)
        }
        constexpr_locals = {
            node.target.id
            for node in ast.walk(function)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _CONSTEXPR_ANNOTATION in ast.unparse(node.annotation)
        }
        plain_locals = {
            node.targets[0].id
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for call in [node for node in ast.walk(function) if isinstance(node, ast.Call)]:
            if not isinstance(call.func, ast.Attribute) or call.func.attr not in _SHAPE_CALLS:
                continue
            # `arange` needs every one of start/end/step constant; `reshape` and
            # `broadcast_to` take their shape as the second positional argument.
            shape_arguments = call.args if call.func.attr == "arange" else call.args[1:2]
            for argument in shape_arguments:
                for name in ast.walk(argument):
                    if isinstance(name, ast.Name) and name.id in plain_locals:
                        offenders.add(name.id)
        offenders -= constexpr_parameters | constexpr_locals
    return offenders


def test_every_shape_and_arange_argument_is_a_constexpr_expression():
    """The failure that actually happened: a plain local in a shape position.

    Checked on the real kernel source, and the negative control below shows the check
    rejects the exact spelling that broke the first hardware compile.
    """
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    offenders = _dynamic_shape_names(source)
    assert offenders == set(), (
        "a plain local in a shape/arange position is not tl.constexpr and fails to "
        f"compile before any arithmetic runs: {sorted(offenders)}"
    )
    body = ast.unparse(_kernel_function("tile_major_packed_fragment_kernel"))
    # The three shapes are written as the constexpr product, the same spelling the
    # frozen prefetch kernel already compiles on this backend.
    assert "tl.arange(0, GROUP_K * 16)" in body
    assert "tl.reshape(raw, (GROUP_K * 16, 16))" in body
    assert "tl.broadcast_to(pair_lanes[None, :], (GROUP_K * 16, 32))" in body
    assert "tile_rows" not in body


@pytest.mark.parametrize("position,before,after", [
    # The exact spelling that failed the first hardware compile.
    ("arange", "    flat = tl.arange(0, GROUP_K * 16)",
     "    tile_rows = GROUP_K * 16\n    flat = tl.arange(0, tile_rows)"),
    # The next two shape positions a bare local could hide in.
    ("reshape", "tl.reshape(raw, (GROUP_K * 16, 16))", "tl.reshape(raw, (tile_rows, 16))"),
    ("broadcast_to", "tl.broadcast_to(pair_lanes[None, :], (GROUP_K * 16, 32))",
     "tl.broadcast_to(pair_lanes[None, :], (tile_rows, 32))"),
])
def test_the_shape_check_has_teeth_on_every_position_that_broke_or_could_break(position,
                                                                              before, after):
    """Negative control: re-introducing a bare shape local must be caught.

    The first compile died at the ``arange``; a fix applied only there would have
    moved the same error to ``reshape`` or ``broadcast_to``, so all three are
    required to be detected.
    """
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert before in source, f"the negative control must target real source ({position})"
    broken = source.replace(before, after)
    if before.startswith("tl."):
        # The reshape/broadcast cases need the bare local to exist at all.
        broken = broken.replace(
            "    flat = tl.arange(0, GROUP_K * 16)",
            "    tile_rows = GROUP_K * 16\n    flat = tl.arange(0, GROUP_K * 16)",
        )
    assert broken != source, "the negative control must actually change the source"
    assert _dynamic_shape_names(broken) == {"tile_rows"}, position
    # ... and the real source is clean under the same check.
    assert _dynamic_shape_names(source) == set()


def test_the_kernel_module_has_no_plain_global_scalar_a_jit_body_reads():
    """The cached-global failure mode: every module-level name must be an import or a jit function.

    A plain module scalar read from a ``@jit`` body is refused by the front end, and an
    equal-to-itself check on a *float* NaN global fails the second compilation. This
    module needs neither, because all of its constants are kernel arguments.
    """
    tree = ast.parse(KERNEL_MODULE.read_text(encoding="utf-8"))
    plain = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            plain.append(ast.unparse(node))
        elif isinstance(node, ast.AnnAssign):
            plain.append(ast.unparse(node))
    assert plain == [], f"module-level value assignments would be JIT globals: {plain}"


def test_the_two_pass_arrangement_is_the_frozen_accumulator():
    """Pass 2 must be the frozen kernel imported, not an equivalent re-typed chain."""
    source = WRAPPER_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("_ascend_decode_block_gemv_kernels", "decode_accumulate_kernel") in imports
    assert "_ascend_tile_major_packed_kernels" in {module for module, _ in imports}
    # The accumulate launch passes the frozen kernel's own parameters.
    launch = ast.unparse(
        next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name == "_launch")
    )
    assert "decode_accumulate_kernel[" in launch
    assert "accumulate_columns" in launch


def test_the_grid_cap_is_checked_before_any_allocation_or_device_work():
    """A geometry this decoder cannot launch must fail on the CPU path too."""
    tree = ast.parse(WRAPPER_MODULE.read_text(encoding="utf-8"))
    entry = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "fp4_packed_tile_major_decode_gemv"
    )
    order = []
    for node in ast.walk(entry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            order.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            order.append(node.func.attr)
    geometry = order.index("_geometry")
    assert geometry < order.index("_prepare_scratch")
    assert geometry < order.index("_npu_launch") if "_npu_launch" in order else True
    assert prototype.MAX_PROGRAMS == tile.MAX_PROGRAMS
    assert ACCUMULATE_COLUMNS == prototype.ACCUMULATE_COLUMNS
    # The cap is genuinely enforced, not decorative.
    assert tile._geometry(n=16 * 4, blocks=4, group_k=4, accumulate_columns=64)[1] >= 1
    with pytest.raises(ValueError, match="total-program cap"):
        tile._geometry(n=16 * 70000, blocks=2, group_k=1, accumulate_columns=64)


@pytest.mark.parametrize("n,k,expected_group_stride", [
    (5120, 2304, 18),   # w1: 320 tiles, 72 blocks, 18 groups -> 5760 programs
    (2304, 5120, 40),   # w2: 144 tiles, 160 blocks, 40 groups -> 5760 programs
])
def _schedule_cells(n, blocks, group_k, accumulate_columns, *, max_programs=None):
    """Model the kernel's grid and loop in plain Python and count each written cell.

    The kernel's own statement is that every ``(block, column)`` pair in range is
    written by exactly one program, with the tail of a ragged final group masked out.
    This reproduces that mapping from the documented decomposition -- program
    ``(tile, group stride position)``, group ``g``, flat row ``j`` of the group tile
    being block ``g * 16 + (j >> 4)`` and tile column ``j & 15`` -- and counts.
    """
    if max_programs is not None:
        monkey = pytest.MonkeyPatch()
        monkey.setattr(tile, "MAX_PROGRAMS", max_programs)
        try:
            geometry = tile._geometry(n, blocks, group_k, accumulate_columns)
        finally:
            monkey.undo()
    else:
        geometry = tile._geometry(n, blocks, group_k, accumulate_columns)
    ntiles, group_stride, _ = geometry
    counts = {}
    for pid0 in range(ntiles):
        for pid1 in range(group_stride):
            for group_base in range(pid1 * group_k, blocks, group_stride * group_k):
                for j in range(group_k * 16):
                    block = group_base + (j >> 4)
                    column = pid0 * 16 + (j & 15)
                    if block < blocks and column < n:
                        counts[(block, column)] = counts.get((block, column), 0) + 1
    return counts


@pytest.mark.parametrize("n,k,group_k", [
    (20, 96, 4),    # ragged rows plus a ragged final group (3 blocks, group of 4)
    (32, 96, 1),    # one block per group
    (48, 160, 2),   # even group over a 5-block projection
    (16, 32, 4),    # a single tile, fewer blocks than one group
])
def test_every_block_and_column_is_written_exactly_once(n, k, group_k):
    blocks = k // 32
    counts = _schedule_cells(n, blocks, group_k, ACCUMULATE_COLUMNS)
    expected = {(block, column) for block in range(blocks) for column in range(n)}
    assert set(counts) == expected, "the schedule must cover every in-range cell"
    assert set(counts.values()) == {1}, "no cell may be written twice"


@pytest.mark.parametrize("n,k,group_k,max_programs", [
    (32, 256, 1, 4),    # 2 tiles, 8 blocks, 2 group programs -> striding, exact groups
    (32, 352, 4, 4),    # 2 tiles, 11 blocks, groups of 4 -> striding AND a ragged tail
    (144, 320, 4, 65535),  # the real (144, 320) hardware case: 9 tiles, 10 blocks
])
def test_the_group_stride_covers_every_block_when_the_grid_cap_forces_striding(
        n, k, group_k, max_programs):
    """A cap too small for one program per group must still cover every cell once.

    The third case is the shape the first hardware attempt actually used, where the
    grid is ``(9, 3)``: striding over groups of four with a two-block tail masked.
    """
    blocks = k // 32
    geometry = _schedule_cells(n, blocks, group_k, ACCUMULATE_COLUMNS,
                               max_programs=max_programs)
    assert set(geometry) == {(block, column) for block in range(blocks) for column in range(n)}
    assert set(geometry.values()) == {1}


@pytest.mark.parametrize("n,k,expected_group_stride", [
    (5120, 2304, 18),   # w1: 320 tiles, 72 blocks, 18 groups -> 5760 programs
    (2304, 5120, 40),   # w2: 144 tiles, 160 blocks, 40 groups -> 5760 programs
])
def test_the_released_geometries_launch_inside_the_device_cap(n, k, expected_group_stride):
    ntiles = math.ceil(n / NTILE_ROWS)
    blocks = k // 32
    tiles, stride, column_grid = tile._geometry(
        n=n, blocks=blocks, group_k=GROUP_K, accumulate_columns=ACCUMULATE_COLUMNS
    )
    assert tiles == ntiles
    assert stride == expected_group_stride
    assert tiles * stride <= prototype.MAX_PROGRAMS
    assert column_grid * ACCUMULATE_COLUMNS >= n


# ------------------------------------------------------------------------------------------------
# The fused kernel, on a leased card
# ------------------------------------------------------------------------------------------------

DEVICE = __import__("os").environ.get("RELIC_DECODE_GEMV_TEST_DEVICE", "").strip()
BACKEND = __import__("os").environ.get("RELIC_DECODE_GEMV_TEST_BACKEND", "").strip()


@pytest.mark.skipif(
    not DEVICE.startswith("npu") or BACKEND != "triton",
    reason="set RELIC_DECODE_GEMV_TEST_DEVICE=npu:0 and RELIC_DECODE_GEMV_TEST_BACKEND=triton",
)
@pytest.mark.parametrize("n,k", [(256, 256), (144, 320), (128, 192)])
def test_the_fused_tile_major_kernel_matches_the_accepted_operator_bit_for_bit(n, k):
    """The one claim only hardware can make: the new access reads the same values.

    The oracle is the frozen prototype's fused kernel on the *logical* checkpoint
    bytes, not its CPU path, so this compares two device runs over the same data.
    """
    import torch_npu  # noqa: F401

    device = torch.device(DEVICE)
    generator = torch.Generator().manual_seed(114)
    a, sa, b, sb = _operands(n, k, generator, scale_low=0, scale_high=256)
    # Make the weights exercise the whole byte range deliberately.
    b = torch.arange(b.numel(), dtype=torch.int64).remainder(256).to(torch.uint8).reshape(b.shape)
    b_tile, s_tile = pack_tile_major(b, sb)

    expected = prototype.fp4_packed_decode_gemv(
        a.to(device), sa.to(device), b.to(device), sb.to(device), backend="triton"
    )
    actual = fp4_packed_tile_major_decode_gemv(
        a.to(device), sa.to(device), b_tile.to(device), s_tile.to(device), backend="triton"
    )
    torch.npu.synchronize(device)
    assert _same_bits(actual.cpu(), expected.cpu())