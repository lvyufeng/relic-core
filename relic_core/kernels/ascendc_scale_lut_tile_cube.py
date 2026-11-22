"""Explicit binding for the scale-aware LUT Cube consumer prototype.

The accepted compact consumer decodes a packed FP4 byte into two FP8-derived unit
values and then multiplies each decoded element by its per-32-block E8M0 factor
before the exact BF16 narrowing. That multiply is exact but it is also one gather,
one vector multiply and one narrowing per element.

This prototype observes that the E8M0 scale bytes in the real checkpoint are not a
broad distribution: every one of the 46080 scaled matrices uses a value in
``117..127`` -- eleven distinct factors. So the exact ``OriginalFp4Bits`` result can
be *tabulated* per (scale value, packed byte) pair offline and the device then only
needs table gathers:

    value_pair = scaleLut[(byte << 2) + scaleSlotBase]

with two BF16 units per word, one for the low nibble code and one for the high
nibble code. The table is 12 slots x 256 words = 12288 B: eleven exact factors plus
one poison slot that yields two NaNs, which is where every scale byte outside
``117..127`` is mapped. Nothing about the checkpoint changes: the original Cube-order
N32/K128 packed bytes and the original E8M0 scale bytes are read verbatim, no expert
metadata is added and no weight matrix is expanded.

The module only binds. Nothing is built, installed or registered on import, and no
global provider is replaced.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from threading import RLock

import torch

from .ascend_unpack import _npu_launch
from .online_tile_cube import _validate

__all__ = ["SCALE_LUT_MIN_SCALE", "SCALE_LUT_MAX_SCALE", "SCALE_LUT_SLOT_COUNT",
           "SCALE_LUT_POISON_WORD", "SCALE_LUT_FORMATS", "original_fp4_bits",
           "scale_lut_table_words", "scale_lut_table_cpu", "scale_lut_stage_shape",
           "AscendCScaleLUTTileCubeConsumer"]

#: The eleven distinct E8M0 scale bytes measured across every checkpoint matrix.
SCALE_LUT_MIN_SCALE = 117
SCALE_LUT_MAX_SCALE = 127
#: Eleven exact slots plus the poison slot every out-of-range byte maps to.
SCALE_LUT_SLOT_COUNT = SCALE_LUT_MAX_SCALE - SCALE_LUT_MIN_SCALE + 2
#: Two NaN BF16 units; an out-of-range scale poisons instead of reinterpreting.
SCALE_LUT_POISON_WORD = 0x7FC07FC0
#: Source format -> native layout id. The table geometry follows the output rows.
SCALE_LUT_FORMATS = {32: ("relic.fp4.cube-n32-k128.v1", 1),
                     64: ("relic.fp4.cube-n64-k128.v1", 2)}
#: One original K32 block group per Cube K tile, as in the accepted consumer.
GROUP_K = 4
SLOTS = 2
K_BLOCK = 32


def original_fp4_bits(code, scale):
    """Exact port of the accepted native ``OriginalFp4Bits`` helper.

    Constructs the original scaled BF16 bits without floating weight arithmetic:
    a zero magnitude folds both signed zeros to ``+0`` and an out-of-range scale is
    a NaN unit rather than a reinterpreted payload.
    """
    if scale < 2 or scale > 252:
        return 0x7FC0
    magnitude = code & 7
    if magnitude == 0:
        return 0
    delta = 2 if magnitude >= 6 else (1 if magnitude >= 4 else (0 if magnitude >= 2 else -1))
    mantissa = 0x40 if magnitude in (3, 5, 7) else 0
    return ((code & 8) << 12) | ((scale + delta) << 7) | mantissa


def _check_rows(rows):
    if isinstance(rows, bool) or not isinstance(rows, int) or rows not in SCALE_LUT_FORMATS:
        raise ValueError("the scale-aware LUT prototype is defined for N32 or N64 output rows")
    return rows


def scale_lut_table_words(rows):
    """Words in the prepared table: 12 slots, the scale map, then the pair repeat."""
    rows = _check_rows(rows)
    return SCALE_LUT_SLOT_COUNT * 256 + 256 + rows * 128 // 2


def scale_lut_stage_shape(block_dim, rows=32):
    """BF16 ring shared by the single projection: ``[block_dim, 2 slots, rows*128]``."""
    rows = _check_rows(rows)
    if isinstance(block_dim, bool) or not isinstance(block_dim, int) or not 1 <= block_dim <= 24:
        raise ValueError("the scale-aware LUT stage requires the bounded block_dim 1..24")
    return (block_dim, SLOTS, rows * 128)


def scale_lut_pair_word(scale, code_byte):
    """One table word: the low-nibble unit in the low half, the high-nibble in the high."""
    low = original_fp4_bits(code_byte & 15, scale)
    high = original_fp4_bits(code_byte >> 4, scale)
    return low | (high << 16)


def scale_lut_table_cpu(rows=32):
    """Exact small metadata only; no checkpoint weight or routing value is read.

    Three sections, all word addressed:

    ``[12 * 256]``  paired-BF16 unit words, slot ``s`` holding scale ``117 + s`` and
                    slot 11 the poison pair;
    ``[256]``       raw scale byte -> byte offset of its slot inside the LUT, with
                    every byte outside ``117..127`` mapping to the poison slot;
    ``[rows*64]``   one byte offset per packed code byte, selecting its scale word
                    from the original Cube-order tile coordinates.
    """
    rows = _check_rows(rows)
    payload = []
    for slot in range(SCALE_LUT_SLOT_COUNT):
        if slot == SCALE_LUT_SLOT_COUNT - 1:
            payload.extend([SCALE_LUT_POISON_WORD] * 256)
            continue
        scale = SCALE_LUT_MIN_SCALE + slot
        payload.extend(scale_lut_pair_word(scale, code_byte) for code_byte in range(256))
    payload.extend(((raw - SCALE_LUT_MIN_SCALE) * 256 * 4
                    if SCALE_LUT_MIN_SCALE <= raw <= SCALE_LUT_MAX_SCALE
                    else (SCALE_LUT_SLOT_COUNT - 1) * 256 * 4)
                   for raw in range(256))
    payload.extend(((pair_index // (rows * 16)) * rows + (pair_index // 8) % rows) * 4
                   for pair_index in range(rows * 128 // 2))
    if len(payload) != scale_lut_table_words(rows):
        raise RuntimeError("scale-aware LUT table layout differs from the native ABI")
    return torch.tensor([((word + 2**31) % 2**32) - 2**31 for word in payload], dtype=torch.int32)


class AscendCScaleLUTTileCubeConsumer:
    """One explicit native library owner; original Cube-order packed bytes only."""

    def __init__(self, library_path, *, sha256: str, block_dim=24):
        path = Path(library_path).resolve(strict=True)
        if not path.is_file() or len(sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError("native scale-aware LUT consumer needs the exact verified library checksum")
        if isinstance(block_dim, bool) or not isinstance(block_dim, int) or not 1 <= block_dim <= 24:
            raise ValueError("Ascend910B prototype block_dim must be within 1..24")
        self.path = path
        self.sha256 = sha256
        self.block_dim = block_dim
        self._library = ctypes.CDLL(str(path))
        for symbol in ("relic_scale_lut_tile_cube_k_blocks_per_tile",
                       "relic_scale_lut_tile_cube_output_rows",
                       "relic_scale_lut_tile_cube_layout_id",
                       "relic_scale_lut_tile_cube_table_words",
                       "relic_scale_lut_tile_cube_scale_min",
                       "relic_scale_lut_tile_cube_scale_max",
                       "relic_scale_lut_tile_cube_prepared_tables",
                       "relic_scale_lut_tile_cube_unsigned_byte_cast",
                       "relic_scale_lut_tile_cube_launch_tables"):
            if getattr(self._library, symbol, None) is None:
                raise ValueError("native scale-aware LUT consumer requires the complete checked table-launch ABI")
        self.group_k = self._query("relic_scale_lut_tile_cube_k_blocks_per_tile")
        if self.group_k != GROUP_K:
            raise ValueError("the scale-aware LUT consumer must fold exactly four original K32 blocks")
        self.output_rows = self._query("relic_scale_lut_tile_cube_output_rows")
        if self.output_rows not in SCALE_LUT_FORMATS:
            raise ValueError("the scale-aware LUT consumer must declare N32 or N64 output rows")
        self.weight_format, expected_layout = SCALE_LUT_FORMATS[self.output_rows]
        self.layout_id = self._query("relic_scale_lut_tile_cube_layout_id")
        if self.layout_id != expected_layout:
            raise ValueError("the declared layout id does not match the declared output rows")
        # Compact-only by construction, so the format helper can compare a boolean.
        self.cube_packed = True
        self.table_words = self._query("relic_scale_lut_tile_cube_table_words")
        if self.table_words != scale_lut_table_words(self.output_rows):
            raise ValueError("native scale-aware LUT table extent differs from the expected ABI")
        scale_min = self._query("relic_scale_lut_tile_cube_scale_min")
        scale_max = self._query("relic_scale_lut_tile_cube_scale_max")
        if (scale_min, scale_max) != (SCALE_LUT_MIN_SCALE, SCALE_LUT_MAX_SCALE):
            raise ValueError("the native table must cover exactly the measured scale range")
        if self._query("relic_scale_lut_tile_cube_prepared_tables") != 1:
            raise ValueError("the scale-aware LUT consumer requires its immutable prepared tables")
        self.prepared_decode_tables = True
        if self._query("relic_scale_lut_tile_cube_unsigned_byte_cast") != 1:
            raise ValueError("the scale-aware LUT prototype requires the exact unsigned A2 byte path")
        self.unsigned_byte_cast = True
        # The LUT geometry is declared too, so a table built by one revision cannot be
        # consumed by a kernel that assumes another slot count.
        lut_slots = self._optional_query("relic_scale_lut_tile_cube_lut_slots")
        if lut_slots is not None and lut_slots != SCALE_LUT_SLOT_COUNT:
            raise ValueError("native LUT slot count differs from the measured table")
        poison_slot = self._optional_query("relic_scale_lut_tile_cube_poison_slot")
        if poison_slot is not None and poison_slot != SCALE_LUT_SLOT_COUNT - 1:
            raise ValueError("native poison slot differs from the measured table")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.ring_elements = self.output_rows * 128
        # The native declares its own per-slot ring extent; pin it so a kernel that
        # assumes another ELEMENTS cannot silently index the stage out of bounds.
        stage_elements = self._optional_query("relic_scale_lut_tile_cube_stage_elements")
        if stage_elements is not None and stage_elements != self.ring_elements:
            raise ValueError("native per-slot stage extent differs from the declared output rows")
        self.stage_shape = scale_lut_stage_shape(block_dim, self.output_rows)
        self.stage_nbytes = self.stage_shape[0] * self.stage_shape[1] * self.stage_shape[2] * 2
        self._launch_tables = self._library.relic_scale_lut_tile_cube_launch_tables
        self._launch_tables.restype = None
        self._launch_tables.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                        *([ctypes.c_void_p] * 7), *([ctypes.c_uint32] * 7)]
        self._decode_tables = {}
        self._decode_table_versions = {}
        self._cpu_decode_table = None
        self._table_lock = RLock()

    def _query(self, name):
        function = getattr(self._library, name)
        function.restype, function.argtypes = ctypes.c_uint32, []
        return function()

    def _optional_query(self, name):
        function = getattr(self._library, name, None)
        if function is None:
            return None
        function.restype, function.argtypes = ctypes.c_uint32, []
        return function()

    @torch.no_grad()
    def prepare_device_metadata(self, device):
        """Prepare the immutable scale-aware table once per device, before capture."""
        device = torch.device(device)
        if device.type != "npu":
            raise ValueError("native scale-aware LUT metadata must be prepared on the explicit NPU device")
        if device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        with self._table_lock:
            table = self._decode_tables.get(device)
            if table is not None:
                try:
                    version = table._version
                except RuntimeError:
                    version = None
                if (table.shape != (self.table_words,) or table.dtype != torch.int32
                        or table.device != device or not table.is_contiguous()
                        or version != self._decode_table_versions[device]):
                    raise RuntimeError("an immutable native scale-aware LUT table changed after preparation")
                return
            capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise RuntimeError("prepare native scale-aware LUT metadata before entering NPU graph capture")
            if self._cpu_decode_table is None:
                self._cpu_decode_table = scale_lut_table_cpu(self.output_rows)
            table = self._cpu_decode_table.to(device=device, non_blocking=False)
            try:
                version = table._version
            except RuntimeError:
                version = None
            self._decode_table_versions[device] = version
            self._decode_tables[device] = table

    @torch.no_grad()
    def __call__(self, x, b_tile, s_tile, selected, *, world, n_output=None,
                 certified_bf16_operands=False, certified_scale_lut_operands=False,
                 scratch=None, accumulation=None, output_dtype=torch.bfloat16):
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("native output must be BF16 or an explicitly requested FP32 TP partial")
        owned, tiles, blocks, groups, n = _validate(x, b_tile, s_tile, selected,
                                                    world=world, n_output=n_output, group_k=1, out=None)
        if n != tiles * 16:
            raise ValueError("the scale-aware LUT prototype requires the complete physical output extent")
        if n % self.output_rows:
            raise ValueError("scale-aware output rows must preserve a complete bounded tile")
        if blocks % self.group_k:
            raise ValueError("the original K32 grid must be divisible by the native tile grouping")
        if any(value >= 2**32 for value in (owned, tiles, blocks, blocks * 32, groups, world, n)):
            raise ValueError("native geometry must fit the uint32 C ABI without truncation")
        # The native scale map is total: any byte outside 117..127 becomes the poison
        # slot, so an unmeasured scale would silently become NaN. The caller has to
        # certify the real range once at load, before the bank is installed.
        if certified_scale_lut_operands is not True:
            raise ValueError("the scale-aware LUT consumer requires an explicit measured scale-range certificate")
        output = accumulation if accumulation is not None else torch.empty((groups, n), dtype=torch.float32,
                                                                          device=x.device)
        stage = scratch if scratch is not None else torch.empty(self.stage_shape, dtype=torch.bfloat16,
                                                                device=x.device)
        for tensor, shape, dtype in ((output, (groups, n), torch.float32),
                                     (stage, self.stage_shape, torch.bfloat16)):
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != x.device
                    or not tensor.is_contiguous() or tensor.requires_grad):
                raise ValueError("native ring/output scratch metadata is inconsistent")
        if len({t.untyped_storage()._cdata
                for t in (x, b_tile, s_tile, selected, output, stage)}) != 6:
            raise ValueError("native ring/output scratch must not alias input banks or each other")
        if x.device.type != "npu" or certified_bf16_operands is not True:
            raise ValueError("the native consumer requires NPU and load-time exact BF16 certification")
        with _npu_launch(x):
            stream = torch.npu.current_stream(x.device).npu_stream
            self.prepare_device_metadata(x.device)
            table = self._decode_tables[x.device]
            self._launch_tables(self.block_dim, stream, x.data_ptr(), b_tile.data_ptr(),
                                s_tile.data_ptr(), selected.data_ptr(), output.data_ptr(),
                                stage.data_ptr(), table.data_ptr(),
                                tiles, blocks, groups, owned, world, n, int(x.shape[0] == 1))
        return output if output_dtype == torch.float32 else output.to(torch.bfloat16)

    def describe(self):
        return {"library": str(self.path), "sha256": self.sha256,
                "weight_format": self.weight_format,
                "cube_packed_compact_banks": self.cube_packed,
                "native_layout_id": self.layout_id,
                "original_K32_blocks_per_Cube_tile": self.group_k,
                "native_output_rows_per_tile": self.output_rows,
                "measured_scale_range": [self.scale_min, self.scale_max],
                "exact_scale_slots": SCALE_LUT_SLOT_COUNT - 1,
                "poison_scale_slot": SCALE_LUT_SLOT_COUNT - 1,
                "out_of_range_scale_poisons_instead_of_reinterpreting": True,
                "scale_LUT_bytes": SCALE_LUT_SLOT_COUNT * 256 * 4,
                "table_words": self.table_words,
                "decode_table_bytes_per_device": self.table_words * 4,
                "decode_table_devices_prepared": [str(device) for device in self._decode_tables],
                "decode_tables_contain_no_checkpoint_weight_or_routing_values": True,
                "expert_metadata_added": False,
                "expanded_weight_scratch_bytes": 0,
                "original_scale_bytes_retained": True,
                "single_vector_multiply_removed_by_tabulation": True,
                "requires_measured_scale_range_certificate": True,
                "ring_shape": list(self.stage_shape),
                "ring_bytes_total": self.stage_nbytes,
                "exact_A2_unsigned_byte_to_half_indices": self.unsigned_byte_cast,
                "immutable_decode_tables_prepared_once_per_device": True}