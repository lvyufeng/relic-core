"""Contract tests for the scale-aware LUT binding; metadata only, never launched.

The table oracles here are deliberately independent of the module's own arithmetic:
the unit bits are re-derived from the FP4 *value* semantics and the repeat section is
re-derived by decomposing the Cube-order byte coordinates, so a mistake in either
transcription shows up as a disagreement rather than as a matching bug.
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import ascendc_scale_lut_tile_cube as lut

SLOT_COUNT = 12
LUT_WORDS = SLOT_COUNT * 256
POISON = 0x7FC07FC0

#: The exact FP4 E2M1 magnitude set the original helper encodes.
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _reference_unit_bits(code, scale):
    """Independent oracle: build the value, then read its exact BF16 bits.

    ``OriginalFp4Bits`` adds a scale delta and a mantissa bit; this instead forms
    ``magnitude * 2 ** (scale - 127)`` as a Python float, which is exact for every
    in-range code, and then takes the high half of the fp32 pattern.
    """
    if scale < 2 or scale > 252:
        return 0x7FC0
    magnitude = MAGNITUDES[code & 7]
    sign = 0x8000 if code & 8 else 0
    if magnitude == 0.0:
        return 0  # both signed zeros fold to +0, matching the accepted helper
    value = magnitude * 2.0 ** (scale - 127)
    return sign | (struct.unpack("<I", struct.pack("<f", value))[0] >> 16)


def _words(rows=32):
    return (lut.scale_lut_table_cpu(rows).long() & 0xFFFFFFFF).tolist()


# ------------------------------------------------------------------ unit reference


def test_the_unit_table_matches_an_independent_value_semantics_oracle():
    """16 codes x 11 scales, compared against the FP4 value re-derived from scratch."""
    words = _words()
    for slot in range(SLOT_COUNT - 1):
        scale = lut.SCALE_LUT_MIN_SCALE + slot
        for byte in range(256):
            low = _reference_unit_bits(byte & 15, scale)
            high = _reference_unit_bits(byte >> 4, scale)
            assert words[slot * 256 + byte] == low | (high << 16), (slot, byte)


def test_the_helper_matches_the_accepted_scale_127_unit_table():
    """The accepted consumer tabulates scale 127; this prototype must reproduce it bit for bit."""
    units = (0, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0)
    expected = list(units) + [unit | 0x8000 if unit else 0 for unit in units]
    assert [lut.original_fp4_bits(code, 127) for code in range(16)] == expected


def test_a_zero_magnitude_folds_both_signed_zeros():
    for scale in range(lut.SCALE_LUT_MIN_SCALE, lut.SCALE_LUT_MAX_SCALE + 1):
        assert lut.original_fp4_bits(0, scale) == 0
        assert lut.original_fp4_bits(8, scale) == 0


def test_out_of_range_scales_are_nan_units():
    for scale in (0, 1, 2 - 1, 253, 254, 255):
        assert lut.original_fp4_bits(7, scale) == 0x7FC0
    assert lut.original_fp4_bits(7, 2) != 0x7FC0
    assert lut.original_fp4_bits(7, 252) != 0x7FC0


def test_every_pair_word_combines_the_low_and_high_nibble_independently():
    words = _words()
    for slot in range(SLOT_COUNT - 1):
        scale = lut.SCALE_LUT_MIN_SCALE + slot
        for byte in range(256):
            word = words[slot * 256 + byte]
            assert word & 0xFFFF == lut.original_fp4_bits(byte & 15, scale)
            assert word >> 16 == lut.original_fp4_bits(byte >> 4, scale)


# ------------------------------------------------------------------- poison safety


def test_the_last_slot_is_two_nans_for_every_byte():
    words = _words()
    base = (SLOT_COUNT - 1) * 256
    assert all(word == POISON for word in words[base:base + 256])


def test_every_unmeasured_scale_byte_maps_to_the_poison_slot():
    """The native map is total, so an unmeasured scale poisons instead of reinterpreting."""
    words = _words()
    map_base = LUT_WORDS
    poison_base = (SLOT_COUNT - 1) * 256 * 4
    for raw in range(256):
        mapped = words[map_base + raw]
        if lut.SCALE_LUT_MIN_SCALE <= raw <= lut.SCALE_LUT_MAX_SCALE:
            assert mapped == (raw - lut.SCALE_LUT_MIN_SCALE) * 256 * 4, raw
        else:
            assert mapped == poison_base, raw
            assert words[mapped // 4] == POISON


def test_the_measured_range_is_exactly_the_eleven_observed_values():
    assert (lut.SCALE_LUT_MIN_SCALE, lut.SCALE_LUT_MAX_SCALE) == (117, 127)
    assert lut.SCALE_LUT_MAX_SCALE - lut.SCALE_LUT_MIN_SCALE + 1 == 11
    assert lut.SCALE_LUT_SLOT_COUNT == 12


# --------------------------------------------------------------------- pair repeat


def _reference_repeat_slot(pair_index, rows):
    """Decompose the Cube-order byte coordinate and derive its scale slot directly.

    Byte order per tile is ``[k16(8), n16(rows//16), row(16), kpair(8)]``; the scale of
    a decoded element is ``(k16 // 2) * rows + n16 * 16 + row``. The n16 width follows
    the output tile, so the same oracle covers N32 and N64.
    """
    n16_count = rows // 16
    block = 8 * 16 * n16_count
    row = (pair_index // 8) % 16
    n16 = (pair_index // 128) % n16_count
    k16 = pair_index // block
    return (k16 // 2) * rows + n16 * 16 + row


@pytest.mark.parametrize("rows", [32, 64])
def test_the_repeat_section_agrees_with_the_physical_coordinate_oracle(rows):
    words = _words(rows)
    repeat_base = LUT_WORDS + 256
    pairs = rows * 128 // 2
    assert len(words) - repeat_base == pairs
    for pair_index in range(pairs):
        assert words[repeat_base + pair_index] == _reference_repeat_slot(pair_index, rows) * 4, pair_index


@pytest.mark.parametrize("rows", [32, 64])
def test_every_repeat_offset_lands_inside_the_256_word_scale_map(rows):
    """The repeat indexes the scale map, so a bad offset would read outside the table."""
    words = _words(rows)
    repeat = words[LUT_WORDS + 256:]
    assert set(repeat) == {slot * 4 for slot in range(rows * 4)}
    assert max(repeat) == (rows * 4 - 1) * 4 <= 255 * 4
    for offset in repeat:
        assert 0 <= words[LUT_WORDS + offset // 4] < LUT_WORDS * 4


# ------------------------------------------------------------------ extents/budgets


def test_the_table_extents_are_the_measured_ones():
    assert lut.scale_lut_table_words(32) == 12 * 256 + 256 + 32 * 128 // 2 == 5376
    assert lut.scale_lut_table_words(64) == 12 * 256 + 256 + 64 * 128 // 2 == 7424
    assert len(_words(32)) == 5376 and len(_words(64)) == 7424
    # The table is 12 KiB of slots, nothing else grows with the expert count.
    assert SLOT_COUNT * 256 * 4 == 12 * 1024


def test_the_n32_n64_stage_budgets():
    assert lut.scale_lut_stage_shape(24, 32) == (24, 2, 4096)
    assert lut.scale_lut_stage_shape(24, 64) == (24, 2, 8192)
    assert 24 * 2 * 4096 * 2 == 393216
    assert 24 * 2 * 8192 * 2 == 786432
    for bad in (16, 48, True, 0):
        with pytest.raises(ValueError, match="N32 or N64"):
            lut.scale_lut_stage_shape(24, bad)


def test_the_table_is_int32_and_rejects_an_unknown_row_count():
    table = lut.scale_lut_table_cpu(32)
    assert table.dtype == torch.int32 and tuple(table.shape) == (5376,)
    for bad in (16, 48, True):
        with pytest.raises(ValueError, match="N32 or N64"):
            lut.scale_lut_table_cpu(bad)


# --------------------------------------------------------------- library metadata


class _Query:
    def __init__(self, value):
        self.value = value
        self.restype = None
        self.argtypes = None

    def __call__(self):
        return self.value


SYMBOLS = ("k_blocks_per_tile", "output_rows", "layout_id", "table_words", "scale_min",
           "scale_max", "prepared_tables", "unsigned_byte_cast", "launch_tables")


def _consumer(tmp_path, monkeypatch, *, rows=32, group_k=4, layout=None, table_words=None,
              scale_min=117, scale_max=127, prepared=1, unsigned=1, block_dim=24,
              unsigned_name="unsigned_byte_cast", lut_slots=None, poison_slot=None,
              stage_elements=None,
              drop=(), sha=None, payload=b"scale-lut-tile-cube-prototype"):
    path = tmp_path / "scale_lut_tile_cube.so"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    values = {"k_blocks_per_tile": group_k, "output_rows": rows,
              "layout_id": lut.SCALE_LUT_FORMATS[rows][1] if layout is None and rows in lut.SCALE_LUT_FORMATS else layout,
              "table_words": lut.scale_lut_table_words(rows) if table_words is None and rows in lut.SCALE_LUT_FORMATS else table_words,
              "scale_min": scale_min, "scale_max": scale_max,
              "prepared_tables": prepared, unsigned_name: unsigned, "launch_tables": None}
    if lut_slots is not None:
        values["lut_slots"] = lut_slots
    if poison_slot is not None:
        values["poison_slot"] = poison_slot
    if stage_elements is not None:
        values["stage_elements"] = stage_elements
    symbols = {f"relic_scale_lut_tile_cube_{name}": _Query(value)
               for name, value in values.items() if name not in drop}
    monkeypatch.setattr(lut.ctypes, "CDLL", lambda _: SimpleNamespace(**symbols))
    return lut.AscendCScaleLUTTileCubeConsumer(path, sha256=sha or digest, block_dim=block_dim)


def test_a_verified_n32_library_exposes_the_measured_contract(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    assert consumer.weight_format == "relic.fp4.cube-n32-k128.v1"
    assert consumer.cube_packed is True and consumer.layout_id == 1
    assert consumer.group_k == 4 and consumer.output_rows == 32
    assert (consumer.scale_min, consumer.scale_max) == (117, 127)
    assert consumer.table_words == 5376
    assert consumer.prepared_decode_tables is True and consumer.unsigned_byte_cast is True
    assert consumer.stage_shape == (24, 2, 4096) and consumer.stage_nbytes == 393216
    assert consumer.ring_elements == 4096


def test_a_verified_n64_library_uses_the_second_layout_id(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch, rows=64)
    assert consumer.weight_format == "relic.fp4.cube-n64-k128.v1"
    assert consumer.layout_id == 2
    assert consumer.table_words == 7424
    assert consumer.stage_shape == (24, 2, 8192) and consumer.stage_nbytes == 786432


def test_the_scale_launch_binds_seven_pointers_and_seven_uint32(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    assert len(consumer._launch_tables.argtypes) == 16
    assert consumer._launch_tables.argtypes[2:9] == [lut.ctypes.c_void_p] * 7


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_scale_query_is_mandatory(tmp_path, monkeypatch, symbol):
    with pytest.raises(ValueError, match="complete checked table-launch ABI"):
        _consumer(tmp_path, monkeypatch, drop=(symbol,))


def test_the_checksum_must_match(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exact verified library checksum"):
        _consumer(tmp_path, monkeypatch, sha="0" * 64)


@pytest.mark.parametrize("block_dim", [0, 25, -1, True])
def test_block_dim_is_bounded(tmp_path, monkeypatch, block_dim):
    with pytest.raises(ValueError, match="block_dim must be within 1..24"):
        _consumer(tmp_path, monkeypatch, block_dim=block_dim)


@pytest.mark.parametrize("rows", [16, 48, 128])
def test_only_n32_and_n64_rows_are_accepted(tmp_path, monkeypatch, rows):
    with pytest.raises(ValueError, match="N32 or N64 output rows"):
        _consumer(tmp_path, monkeypatch, rows=rows, layout=1, table_words=5376)


def test_a_layout_id_that_disagrees_with_the_rows_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="layout id does not match"):
        _consumer(tmp_path, monkeypatch, rows=64, layout=1)


@pytest.mark.parametrize("table_words", [4608, 6656, 5375, 7424])
def test_the_table_extent_must_match_the_declared_rows(tmp_path, monkeypatch, table_words):
    with pytest.raises(ValueError, match="table extent differs"):
        _consumer(tmp_path, monkeypatch, rows=32, table_words=table_words)


@pytest.mark.parametrize("scale_min,scale_max", [(116, 127), (117, 128), (0, 255), (117, 126)])
def test_the_native_scale_range_must_be_the_measured_one(tmp_path, monkeypatch, scale_min, scale_max):
    with pytest.raises(ValueError, match="exactly the measured scale range"):
        _consumer(tmp_path, monkeypatch, scale_min=scale_min, scale_max=scale_max)


@pytest.mark.parametrize("prepared,unsigned", [(0, 1), (1, 0), (2, 1), (1, 2)])
def test_prepared_and_unsigned_policies_are_fail_closed(tmp_path, monkeypatch, prepared, unsigned):
    with pytest.raises(ValueError, match="requires its immutable prepared tables|exact unsigned A2 byte path"):
        _consumer(tmp_path, monkeypatch, prepared=prepared, unsigned=unsigned)


def test_describe_states_the_no_metadata_no_expansion_contract(tmp_path, monkeypatch):
    described = _consumer(tmp_path, monkeypatch).describe()
    assert described["measured_scale_range"] == [117, 127]
    assert described["exact_scale_slots"] == 11 and described["poison_scale_slot"] == 11
    assert described["scale_LUT_bytes"] == 12288
    assert described["expert_metadata_added"] is False
    assert described["expanded_weight_scratch_bytes"] == 0
    assert described["original_scale_bytes_retained"] is True
    assert described["out_of_range_scale_poisons_instead_of_reinterpreting"] is True
    assert described["requires_measured_scale_range_certificate"] is True
    assert described["decode_table_bytes_per_device"] == 5376 * 4


def test_device_metadata_must_be_prepared_on_an_npu(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit NPU device"):
        _consumer(tmp_path, monkeypatch).prepare_device_metadata(torch.device("cpu"))


# ------------------------------------------------------------------- call contract


def _operands(rows=32, groups=2, blocks=4):
    """tiles*16 == rows: 2 original N16 banks for an N32 tile, 4 for an N64 tile."""
    tiles = rows // 16
    codes = torch.zeros((1, tiles, blocks, 16, 16), dtype=torch.uint8)
    scales = torch.zeros((1, tiles, blocks, 16), dtype=torch.uint8)
    x = torch.zeros((1, blocks * 32), dtype=torch.bfloat16)
    selected = torch.zeros((groups,), dtype=torch.int32)
    return x, codes, scales, selected


def _call(consumer, x, codes, scales, selected, **kwargs):
    kwargs.setdefault("world", 1)
    kwargs.setdefault("certified_bf16_operands", True)
    kwargs.setdefault("certified_scale_lut_operands", True)
    return consumer(x, codes, scales, selected, **kwargs)


def test_a_cpu_call_stops_at_the_explicit_npu_refusal(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="requires NPU and load-time exact BF16 certification"):
        _call(consumer, x, codes, scales, selected)


def test_the_measured_scale_certificate_is_required(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="explicit measured scale-range certificate"):
        _call(consumer, x, codes, scales, selected, certified_scale_lut_operands=False)


def test_a_partial_extent_is_refused(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="complete physical output extent"):
        _call(consumer, x, codes, scales, selected, n_output=16)


def test_a_non_divisible_k_grid_is_refused(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands(blocks=5)
    with pytest.raises(ValueError, match="divisible by the native tile grouping"):
        _call(consumer, x, codes, scales, selected)


def test_the_ring_scratch_shape_is_explicit(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="ring/output scratch metadata is inconsistent"):
        _call(consumer, x, codes, scales, selected,
              scratch=torch.empty((24, 2, 2048), dtype=torch.bfloat16))


def test_the_accumulator_metadata_is_checked(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="ring/output scratch metadata is inconsistent"):
        _call(consumer, x, codes, scales, selected,
              accumulation=torch.empty((2, 32), dtype=torch.bfloat16))


def test_ring_and_accumulator_that_share_storage_are_refused(tmp_path, monkeypatch):
    """The ring and the FP32 partial must never be views of one allocation."""
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    backing = torch.empty(24 * 2 * 4096, dtype=torch.bfloat16)
    stage = backing.view(24, 2, 4096)
    accumulation = backing[:128].view(torch.float32).view(2, 32)
    with pytest.raises(ValueError, match="must not alias input banks or each other"):
        _call(consumer, x, codes, scales, selected, scratch=stage, accumulation=accumulation)


def test_an_unsupported_output_dtype_is_refused(tmp_path, monkeypatch):
    consumer = _consumer(tmp_path, monkeypatch)
    x, codes, scales, selected = _operands()
    with pytest.raises(ValueError, match="BF16 or an explicitly requested FP32 TP partial"):
        _call(consumer, x, codes, scales, selected, output_dtype=torch.float16)

def test_the_canonical_unsigned_declaration_is_required(tmp_path, monkeypatch):
    """Only ``unsigned_byte_cast`` is accepted: the native exports that spelling alone.

    Accepting a legacy alias as well would let a library without a canonical
    declaration slip past the mandatory-set check.
    """
    consumer = _consumer(tmp_path, monkeypatch)
    assert consumer.unsigned_byte_cast is True
    with pytest.raises(ValueError, match="complete checked table-launch ABI"):
        _consumer(tmp_path, monkeypatch, drop=("unsigned_byte_cast",))


def test_a_legacy_unsigned_alias_alone_is_not_enough(tmp_path, monkeypatch):
    """A library exporting only the old spelling has not declared the canonical ABI."""
    with pytest.raises(ValueError, match="complete checked table-launch ABI"):
        _consumer(tmp_path, monkeypatch, unsigned_name="unsigned_cast")


@pytest.mark.parametrize("lut_slots", [11, 13])
def test_the_declared_lut_slot_count_must_match_the_table(tmp_path, monkeypatch, lut_slots):
    with pytest.raises(ValueError, match="LUT slot count differs"):
        _consumer(tmp_path, monkeypatch, lut_slots=lut_slots)


@pytest.mark.parametrize("poison_slot", [10, 12])
def test_the_declared_poison_slot_must_match_the_table(tmp_path, monkeypatch, poison_slot):
    with pytest.raises(ValueError, match="poison slot differs"):
        _consumer(tmp_path, monkeypatch, poison_slot=poison_slot)


# ------------------------------------------------------- region offsets / panel trap


@pytest.mark.parametrize("rows", [32, 64])
def test_the_three_regions_sit_at_the_declared_word_offsets(rows):
    """A shuffled region would decode plausible-but-wrong weights, so pin the seams."""
    words = _words(rows)
    lut_end = 12 * 256
    map_end = lut_end + 256
    assert len(words) == lut.scale_lut_table_words(rows) == map_end + rows * 64
    # seam: last poison LUT word | first scale-map word, which is raw byte 0 and
    # therefore out of the measured window -> poison base
    assert words[lut_end - 1] == POISON
    assert words[lut_end] == 11 * 1024
    # ... and the measured window starts at its own offset, raw 117 -> slot 0 base
    assert words[lut_end + lut.SCALE_LUT_MIN_SCALE] == 0
    assert words[lut_end + lut.SCALE_LUT_MAX_SCALE] == 10 * 1024
    # seam: last scale-map word (raw 255 -> poison base) | first repeat word
    assert words[map_end - 1] == 11 * 1024
    assert words[map_end] == 0
    # and every scale-map entry is a whole number of slot words
    assert all(words[lut_end + raw] % 1024 == 0 for raw in range(256))


def test_the_repeat_index_is_the_panel_nrow_not_a_physical_row():
    """``(p``8) % N`` is ``n16 * 16 + row``, not the row alone.

    Within one K16 fragment the pair order is ``[n16(2), row(16), kpair(8)]``, so a
    row-major reading of the fragment (``byte_in_block // 16``) disagrees at every
    panel boundary. This pins the difference so a future "simplification" cannot
    silently adopt the row-major reading.
    """
    rows = 32
    words = _words(rows)
    repeat = words[LUT_WORDS + 256:]
    # p = 128 is fragment 0, panel 1, row 0, pair 0: the nrow form gives 16, a
    # row-major reading of the fragment would give 8.
    assert repeat[128] == 16 * 4
    assert (128 % 512) // 16 * 4 != repeat[128]
    # every panel of 128 pairs shifts the slot by exactly 16 = one panel of rows.
    # A K128 tile has K/16 == 8 fragments of 256 pairs each at N32 (two panels each),
    # so the fragment count follows the K extent, not the row count.
    fragments = 128 // 16
    assert len(repeat) == fragments * 256
    for fragment in range(fragments):
        for pair in range(8):
            assert repeat[fragment * 256 + 128 + pair] == repeat[fragment * 256 + pair] + 16 * 4


@pytest.mark.parametrize("stage_elements", [4096, 8192, 2048])
def test_a_declared_per_slot_stage_extent_must_match_the_rows(tmp_path, monkeypatch, stage_elements):
    """The ring last dimension is rows*128; a kernel assuming another one would overrun it."""
    expected = _consumer(tmp_path, monkeypatch, rows=32).ring_elements
    if stage_elements == expected:
        consumer = _consumer(tmp_path, monkeypatch, rows=32, stage_elements=stage_elements)
        assert consumer.stage_shape[-1] == stage_elements
    else:
        with pytest.raises(ValueError, match="per-slot stage extent differs"):
            _consumer(tmp_path, monkeypatch, rows=32, stage_elements=stage_elements)


# ------------------------------------------------------- native source contract
#
# Static only: these read the sibling native source as text. They pin the ABI and the
# two failure modes root flagged -- the tile count (raw source banks in, divided by
# N_TILES in the kernel) and the raw/scale UB reuse fence -- so a regression fails
# here instead of reading past a bank on a device.

NATIVE_CPP = (Path(__file__).resolve().parents[1]
              / "relic_core" / "csrc" / "ascend" / "scale_lut_tile_cube" / "scale_lut_tile_cube.cpp")
_CONSTANT = re.compile(r"constexpr\s+uint32_t\s+([A-Z][A-Z0-9_]*)\s*=\s*([^;]+);")
_ARITHMETIC = re.compile(r"^[A-Za-z0-9_ ()*/+-]+$")
_MACRO = re.compile(r"#define\s+([A-Z][A-Z0-9_]*)\s+(\d+)")


def _native_text():
    if not NATIVE_CPP.is_file():
        pytest.skip("the scale-aware LUT native source is not present in this checkout")
    return NATIVE_CPP.read_text()


def _native_constants(text):
    """Resolve the constants that are pure arithmetic; leave the rest unresolved.

    The output-row count arrives as a preprocessor macro (``RELIC_LUT_OUTPUT_ROWS``),
    so the plain integer defines are seeded first; without that the whole geometry
    chain (K, BYTES, ELEMENTS, TABLE_WORDS) would stay unresolved.
    """
    resolved, pending = {name: int(value) for name, value in _MACRO.findall(text)}, []
    for name, expression in _CONSTANT.findall(text):
        expression = expression.split("//")[0].strip()
        # C integer suffixes are not Python literals (e.g. 0x7fc07fc0u).
        pending.append((name, re.sub(r"([0-9A-Fa-f])([uUlL]+)$", r"\1", expression)))
    for _ in range(len(pending)):
        progressed = False
        for name, expression in list(pending):
            if not _ARITHMETIC.match(expression):
                continue
            try:
                resolved[name] = eval(expression, {"__builtins__": {}}, dict(resolved))  # noqa: S307
            except (NameError, SyntaxError):
                continue
            pending.remove((name, expression))
            progressed = True
        if not progressed:
            break
    return resolved


def test_the_native_lut_geometry_matches_the_prepared_table():
    text = _native_text()
    constants = _native_constants(text)
    assert constants["SCALE_MIN"] == 117 and constants["SCALE_MAX"] == 127
    assert constants["LUT_SLOTS"] == 12 and constants["LUT_WORDS_PER_SLOT"] == 256
    assert constants["POISON_SLOT"] == 11
    assert constants["TABLE_CODE"] == 0
    assert constants["TABLE_SMAP"] == constants["LUT_SLOTS"] * constants["LUT_WORDS_PER_SLOT"] == 3072
    assert constants["TABLE_REPEAT"] == constants["TABLE_SMAP"] + 256 == 3328
    assert constants["TABLE_WORDS"] == constants["TABLE_REPEAT"] + constants["BYTES"]
    # and the table my binding prepares has exactly that extent for both row counts
    assert constants["TABLE_WORDS"] + constants["BYTES"] == lut.scale_lut_table_words(32) + 2048
    assert re.search(r"POISON_PAIR\s*=\s*0x7fc07fc0", text)
    assert re.search(r"LAYOUT_ID\s*=\s*\(N == 32\)\s*\?\s*1\s*:\s*2", text)


def test_the_native_abi_takes_raw_source_tiles_and_divides_by_n_tiles():
    """Root P1: the accepted ABI passes n/16 source banks; storing it raw walks 2x/4x."""
    text = _native_text()
    assignments = re.findall(r"tiles_\s*=\s*tiles\s*/\s*N_TILES\s*;", text)
    assert len(assignments) == 2, "both the cube and tile Init sites must derive the tile count"
    assert len(re.findall(r"sourceTiles_\s*=\s*tiles\s*;", text)) == 2
    assert "tiles_ = tiles;" not in text, "the raw source-bank count must never become the tile count"


def test_the_native_raw_ub_reuse_fence_is_v_mte2_at_the_loop_end():
    """Root P1: MTE2_V orders data ready; only V_MTE2 orders the next write after V read."""
    text = _native_text()
    ready = [m.start() for m in re.finditer(r"SetFlag<HardEvent::MTE2_V>", text)]
    reuse = [m.start() for m in re.finditer(r"SetFlag<HardEvent::V_MTE2>", text)]
    assert ready, "the data-ready MTE2_V fence must still exist"
    assert reuse, "the raw/scale UB reuse fence must be V_MTE2"
    # the reuse fence must follow the decode that last reads raw_/scaleBytes_
    decode = text.rindex("Gather(scaled_")
    assert reuse[0] > decode, "V_MTE2 must come after the last vector read of the raw buffers"
    # and no MTE2_V may be the final fence before the loop closes
    assert max(ready) < max(reuse)


def test_the_native_exports_only_the_prepared_table_launch():
    text = _native_text()
    assert "relic_scale_lut_tile_cube_launch(" not in text, "there must be no non-table launch symbol"
    signature = re.search(r'extern "C" void relic_scale_lut_tile_cube_launch_tables\((.*?)\)\s*\n\{', text, re.S)
    assert signature is not None
    parameters = [part.strip() for part in signature.group(1).replace("\n", " ").split(",")]
    assert len(parameters) == 16, parameters
    assert parameters[:2] == ["uint32_t blockDim", "void *stream"]
    assert all(part.startswith("void *") for part in parameters[2:9])
    assert all(part.startswith("uint32_t") for part in parameters[9:])


def test_the_native_byte_index_path_is_the_unsigned_a2_cast_only():
    text = _native_text()
    assert "ReinterpretCast<int8_t>" not in text
    assert not re.search(r"Adds\(\s*(codeIdx_|scaleIdx_)\s*,", text)
    assert re.search(r"Cast\(\s*codeHalf_\s*,\s*raw_\s*,", text)
    assert re.search(r"Cast\(\s*scaleHalf_\s*,\s*scaleBytes_\s*,", text)
    # canonical spelling only, and it must declare 1
    assert re.search(r"relic_scale_lut_tile_cube_unsigned_byte_cast\(\)\s*\{[^}]*return 1;", text, re.S)
    assert "relic_scale_lut_tile_cube_unsigned_cast(" not in text


def test_the_native_declares_the_measured_window_and_per_slot_stage():
    text = _native_text()
    assert re.search(r"relic_scale_lut_tile_cube_scale_min\(\)\s*\{[^}]*return SCALE_MIN;", text, re.S)
    assert re.search(r"relic_scale_lut_tile_cube_scale_max\(\)\s*\{[^}]*return SCALE_MAX;", text, re.S)
    assert re.search(r"relic_scale_lut_tile_cube_stage_elements\(\)\s*\{[^}]*return ELEMENTS;", text, re.S)
    assert re.search(r"relic_scale_lut_tile_cube_lut_slots\(\)\s*\{[^}]*return LUT_SLOTS;", text, re.S)
    assert re.search(r"relic_scale_lut_tile_cube_poison_slot\(\)\s*\{[^}]*return POISON_SLOT;", text, re.S)


#: Identifiers that belong to the accepted kernel's other build flags and decoders.
#: This prototype has none of them: it is one straight-line LUT path, so any hit means
#: a stale copy of the old decoder leaked in and there is a second code path to review.
_ABSENT_FROM_THE_LUT_KERNEL = (
    "PRELOAD_A", "preload_", "fullK_", "DIRECT_BITS", "DecodeDirectBits", "CUBE_PACKED",
    "UNSIGNED_BYTE_CAST", "PREPARED_TABLES", "nzIndexes_", "values_", "scaleFactors_",
    "expandedFactors_", "bitMask", "bitPoison", "unitPairs_", "lookup_",
    "repeatScaleIndexes_", "L1_BYTES",
)


def test_the_native_has_no_leftover_code_path_from_the_accepted_kernel():
    """A leaked build flag would mean a second decoder path the tests do not cover."""
    text = _native_text()
    leaked = [name for name in _ABSENT_FROM_THE_LUT_KERNEL if name in text]
    assert not leaked, f"stale accepted-kernel identifiers leaked into the LUT kernel: {leaked}"


def test_the_slot_stride_is_pinned_from_literals_not_from_sizeof():
    """``LUT_BYTES_PER_SLOT`` uses ``sizeof`` and is deliberately not resolved as text.

    The stride is still worth pinning, so it is stated here from the literals: the LUT
    is 12 slots, each 256 words of 4 bytes, i.e. 1024 bytes per slot and 12288 B total.
    """
    text = _native_text()
    assert "LUT_BYTES_PER_SLOT" in text, "the native stride constant must exist"
    assert re.search(r"LUT_WORDS_PER_SLOT \* sizeof\(uint32_t\)", text)
    # The four region constants must resolve from pure integer literals; that the
    # resolver returns all of them is the assertion (it skips any sizeof expression).
    constants = _native_constants(text)
    for region in ("TABLE_CODE", "TABLE_SMAP", "TABLE_REPEAT", "TABLE_WORDS"):
        assert region in constants, region
    assert constants["LUT_WORDS_PER_SLOT"] * 4 == 1024
    assert constants["LUT_SLOTS"] * constants["LUT_WORDS_PER_SLOT"] * 4 == 12288
    assert constants["TABLE_SMAP"] == constants["LUT_SLOTS"] * constants["LUT_WORDS_PER_SLOT"]


def test_the_activation_l1_holds_one_slot_and_is_not_indexed_by_group():
    """Single-A-slot invariant for the shared=False path.

    ``sharedA_`` is one ``M * K`` A1 slot and the MTE2 ``DataCopy`` already selects the
    group's activation with ``xBase``. If the MTE1 ``LoadData`` re-applies a per-group
    offset (``row * M * K`` with ``row = shared_ ? 0 : group``) it reads past that one
    slot as soon as ``groups >= 2`` and ``shared`` is false -- which is a wrong-answer /
    out-of-bounds read rather than a crash, so it is pinned here.

    THIS IS A CORPUS WARNING, not just a source rule. The hardware gate that found it
    passed K128 ``shared=true`` and dynamic ``shared=true``; it only failed at
    ``shared=false`` with three active groups, where the N32 BF16/FP32 difference was
    exactly 256 = 2 groups x 128. A probe corpus of shared-only cases is therefore
    structurally incapable of catching this, so do not trim the ``shared=false``
    multi-group case as redundant with the shared ones.

    The static source must have exactly one A1 declaration, and the A ``LoadData`` must
    take ``sharedA_`` with no index expression. This guards the A1 slot count, which is
    a different invariant from the bank-stride (``tiles``) and raw/scale fence pins.
    """
    text = _native_text()
    declarations = re.findall(r"sharedA_\s*=\s*LocalTensor<bfloat16_t>\(\s*TPosition::A1,\s*([^,]+),\s*([^)]+)\)", text)
    assert len(declarations) == 1, f"expected exactly one A1 activation slot, found {declarations}"
    offset, length = (expression.strip() for expression in declarations[0])
    assert offset == "L1_A_OFFSET" and length == "M * K"
    # the A L1 buffer sits after the two B ring slots: there is no second A slot
    assert re.search(r"L1_A_OFFSET\s*=\s*2 \* L1_B_SLOT_BYTES", text)

    # (1) exactly one LoadData reads the activation, and it is unindexed
    assert text.count("LoadData(a_, sharedA_, load);") == 1, "the A1 load must be unconditional"
    a_loads = re.findall(r"LoadData\(\s*a_,\s*([^,]+),", text)
    assert len(a_loads) == 1, f"expected one A LoadData, found {a_loads}"
    assert a_loads[0].strip() == "sharedA_", (
        "the A LoadData must take the single slot directly; an indexed source such as "
        "sharedA_[row * M * K] reads past the one A1 slot when shared is false")
    # (2) no subscript on the A1 tensor anywhere: a subscript here is the bug itself
    assert "sharedA_[" not in text, "the A1 tensor must never be subscripted"
    # (3) preload-style row strides must not survive in a kernel with no preload mode
    assert "row * M * K" not in text and "row) * M * K" not in text
    assert "fullK_" not in text and "* fullK_" not in text
    assert not re.search(r"const uint32_t row\b", text), "the per-group A1 row helper must be gone"
    # (4) ... while shared_ still selects the GM activation row, so the flag is not dead
    xbase = re.search(r"const uint64_t xBase = shared_ \? 0 : static_cast<uint64_t>\(group\) \* blocks_ \* K;", text)
    assert xbase is not None, "shared_ must keep selecting the GM activation row through xBase"
    # the MTE2 source is still the group-correct base, so removing the index loses nothing
    assert re.search(r"DataCopy\(\s*sharedA_,\s*activation_\[\s*xBase \+ block \* K\s*\]", text)
    assert re.search(r"const uint64_t xBase = shared_ \? 0 : static_cast<uint64_t>\(group\) \* blocks_ \* K;", text)
    # the B ring LoadData keeps its slot index: that one is legitimate
    assert re.search(r"LoadData\(\s*b_,\s*sharedB_\[", text)
