"""Tests for the vectorized exact FP4 byte widening.

Three layers, deliberately separable:

* **Portable, no devices.** The kernel's own integer helpers are re-executed as
  plain Python integers straight from the kernel source and held to statements
  computed here from the value table, never from the helper being tested.
* **Structure.** The arithmetic kernel must contain exactly one data load and no
  gather, and the LUT kernel must gather a table it builds on chip instead of
  loading one from memory; that property is the point of the module, so it is
  asserted rather than described.
* **CPU API**, then **NPU, opt-in** through ``RELIC_VECTOR_FP4_TEST_DEVICE``.
  The CPU path delegates to the frozen widening, and the NPU cases compare the new
  kernels against that same frozen result for every one of the 256 byte patterns.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import unpack, vector_fp4_decode  # noqa: E402

KERNELS = Path(__file__).resolve().parents[1] / "relic_core/kernels"
SOURCE = KERNELS / "_ascend_vector_fp4_decode_kernels.py"
NPU_DEVICE = __import__("os").environ.get("RELIC_VECTOR_FP4_TEST_DEVICE", "")

TABLE = vector_fp4_decode.X2_TABLE


def expected_pair(byte: int) -> int:
    """The pair bits stated from the table, independent of the kernel helpers."""
    return (TABLE[byte & 15] & 255) | ((TABLE[byte >> 4] & 255) << 8)


# -- portable: re-execute the kernel's integer helpers as plain Python ints -------


class Scalar:
    """Minimal int32-shaped scalar the helpers' arithmetic is closed over."""

    def __init__(self, value):
        self.value = int(value.value if isinstance(value, Scalar) else value)

    def to(self, dtype):
        """Model the kernel's named casts: ``uint16`` masks, ``int8`` sign-extends."""
        if dtype == "uint16":
            return Scalar(self.value & 0xFFFF)
        if dtype == "int8":
            return Scalar(((self.value + 0x80) & 0xFF) - 0x80)
        return Scalar(self.value)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def __neg__(self):
        return Scalar(-self.value)

    def __and__(self, other):
        return Scalar(self.value & int(Scalar(other)))

    __rand__ = __and__

    def __or__(self, other):
        return Scalar(self.value | int(Scalar(other)))

    __ror__ = __or__

    def __add__(self, other):
        return Scalar(self.value + int(Scalar(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return Scalar(self.value - int(Scalar(other)))

    __rsub__ = __sub__

    def __lshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "a selected branch shifted by an undefined amount"
        return Scalar(self.value << count)

    def __rshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "a selected branch shifted by an undefined amount"
        return Scalar(self.value >> count)

    def __lt__(self, other):
        return Scalar(self.value < int(Scalar(other)))

    def __ge__(self, other):
        return Scalar(self.value >= int(Scalar(other)))

    def __eq__(self, other):
        return Scalar(self.value == int(Scalar(other)))

    def __ne__(self, other):
        return Scalar(self.value != int(Scalar(other)))


def _function(name: str):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def scalar_helper(name: str, *dependencies: str):
    """Compile one helper plus its jit dependencies into plain-integer callables."""
    tl = type("tl", (), {
        "where": staticmethod(lambda condition, yes, no: Scalar(yes if bool(condition) else no)),
        "int8": "int8",
        "uint16": "uint16",
        "int32": "int32",
    })
    namespace = {"tl": tl}
    for dependency in dependencies:
        exec(compile(_function(dependency), str(SOURCE), "exec"), namespace)
    exec(compile(_function(name), str(SOURCE), "exec"), namespace)
    return namespace[name]


code_x2 = scalar_helper("_code_x2")
pair_bits = scalar_helper("_pair", "_code_x2")


def test_the_stated_table_is_the_frozen_widenings_table():
    assert tuple(TABLE) == tuple(unpack._FP4_X2)
    assert len(TABLE) == 16


def test_every_code_maps_to_twice_its_e2m1_magnitude():
    """Independent statement: E2M1 is 1/2/1, so x2 magnitudes are 0,.5,1,1.5.. doubled."""
    assert [int(code_x2(code)) for code in range(16)] == [0, 1, 2, 3, 4, 6, 8, 12,
                                                          0, -1, -2, -3, -4, -6, -8, -12]


def test_every_byte_pair_matches_the_table_including_both_zero_codes():
    for byte in range(256):
        assert int(pair_bits(byte)) == expected_pair(byte), hex(byte)
    # The two negative-zero codes fold to integer zero in either nibble position.
    assert int(pair_bits(0x00)) == 0
    assert int(pair_bits(0x88)) == 0
    assert int(pair_bits(0xFF)) == expected_pair(0xFF)
    # Code 15 is -12, so the pair is the two's-complement bytes of -12 twice.
    assert int(pair_bits(0xFF)) == (0xF4 | (0xF4 << 8))
    assert int(pair_bits(0x78)) == expected_pair(0x78)


# -- structure: the lookup really left the kernels -------------------------------


def _kernel_calls(name: str) -> list[str]:
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return [ast.unparse(call.func) for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)]


def test_the_arithmetic_kernel_has_one_data_load_and_no_gather():
    calls = _kernel_calls("fp4_vector_arithmetic_kernel")
    assert calls.count("tl.load") == 1, calls
    assert "tl.gather" not in calls, calls
    assert "tl.store" in calls


def test_the_lut_kernel_gathers_a_table_it_builds_instead_of_loading_one():
    calls = _kernel_calls("fp4_vector_lut_kernel")
    assert calls.count("tl.load") == 1, calls
    assert calls.count("tl.gather") == 2, calls
    assert "tl.store" in calls


def test_no_kernel_takes_a_lookup_table_pointer():
    tree = ast.parse(SOURCE.read_text())
    for name in ("fp4_vector_arithmetic_kernel", "fp4_vector_lut_kernel"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        arguments = [argument.arg for argument in node.args.args[:2]]
        assert arguments == ["A", "OUT"], (name, arguments)


def test_the_default_widening_module_is_not_rewired():
    """The frozen path keeps its own file and function; this module only adds."""
    assert "vector_fp4_decode" not in Path(unpack.__file__).read_text()
    assert vector_fp4_decode.unpack_fp4_vector is not unpack.unpack_fp4_to_int8


# -- CPU: shared validation, the frozen result, and the refusals ------------------


def all_patterns() -> torch.Tensor:
    return torch.arange(256, dtype=torch.int64).to(torch.uint8)


def test_the_cpu_backend_produces_the_frozen_bytes_for_every_pattern():
    packed = all_patterns()
    actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="torch")
    frozen = unpack.unpack_fp4_to_int8(packed, backend="torch")
    assert actual.dtype == torch.int8 and frozen.dtype == torch.int8
    assert actual.tolist() == frozen.tolist()
    flat = actual.view(torch.uint8).tolist()
    for byte in range(256):
        assert ((flat[2 * byte + 1] << 8) | flat[2 * byte]) == expected_pair(byte), hex(byte)


def test_shapes_are_the_last_dimension_doubled_and_the_low_nibble_comes_first():
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="torch")
    assert tuple(actual.shape) == (1, 4)
    assert actual.tolist() == [[1, 2, 3, 4]]
    flat = vector_fp4_decode.unpack_fp4_vector(packed.reshape(-1), backend="torch")
    assert tuple(flat.shape) == (4,)


def test_the_explicit_tile_choices_still_validate_on_host_tensors():
    packed = all_patterns()
    for tile in vector_fp4_decode._TILES:
        for mode in ("auto", "arithmetic", "lut"):
            actual = vector_fp4_decode.unpack_fp4_vector(
                packed, backend="torch", mode=mode, tile=tile)
            assert actual.tolist() == unpack.unpack_fp4_to_int8(packed, backend="torch").tolist()


def test_the_callers_output_buffer_is_used_and_returned():
    packed = all_patterns()
    out = torch.empty(512, dtype=torch.int8)
    actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="torch", out=out)
    assert actual is out
    assert out.tolist() == unpack.unpack_fp4_to_int8(packed, backend="torch").tolist()


def test_an_empty_input_returns_an_empty_result_without_a_launch():
    for shape in ((0,), (0, 4), (3, 0)):
        packed = torch.empty(shape, dtype=torch.uint8)
        actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="torch")
        assert tuple(actual.shape) == (*shape[:-1], shape[-1] * 2)
        assert actual.numel() == 0


@pytest.mark.parametrize("call, error", [
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="torch", mode="bogus"), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="torch", tile=1000), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="torch", tile=True), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="bogus"), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p.to(torch.int32)), TypeError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p.reshape(16, 16)[:, ::2]), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="torch",
                                                   out=torch.empty(256, dtype=torch.int16)), TypeError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="torch",
                                                   out=torch.empty(128, dtype=torch.int8)), ValueError),
    (lambda p: vector_fp4_decode.unpack_fp4_vector(p, backend="triton"), ValueError),
])
def test_bad_calls_are_refused_by_name(call, error):
    with pytest.raises(error):
        call(all_patterns())


def test_an_output_that_shares_storage_with_the_input_is_refused():
    buffer = torch.empty(512, dtype=torch.int8)
    packed = buffer[:256].view(torch.uint8)
    with pytest.raises(ValueError):
        vector_fp4_decode.unpack_fp4_vector(packed, backend="torch", out=buffer)


def test_a_zero_rank_input_is_refused():
    with pytest.raises(ValueError):
        vector_fp4_decode.unpack_fp4_vector(torch.tensor(7, dtype=torch.uint8), backend="torch")


def test_the_gathered_tile_keeps_its_inner_width_wide_enough():
    for tile in vector_fp4_decode._TILES:
        rows = vector_fp4_decode._lut_rows(tile)
        assert 16 <= rows <= 64 and tile % rows == 0
        assert tile // rows >= 16, (tile, rows)


def test_the_pointer_width_is_decided_before_any_launch():
    assert vector_fp4_decode._index64(1, 1024) is False
    assert vector_fp4_decode._index64(2**30, 1024) is True
    assert vector_fp4_decode._index64(2**31 // 2 - 1024, 1024) is False
    # One byte short of the boundary pads up to 2**30 bytes, so the doubled
    # element index reaches exactly 2**31.
    assert vector_fp4_decode._index64(2**31 // 2 - 1, 1024) is True


# -- NPU: the new kernels against the frozen widening ----------------------------


needs_device = pytest.mark.skipif(
    not NPU_DEVICE, reason="set RELIC_VECTOR_FP4_TEST_DEVICE=npu:0 on a leased card")


def device_input(count: int, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (count,), generator=generator, dtype=torch.int64).to(torch.uint8)


@needs_device
@pytest.mark.parametrize("mode", ["arithmetic", "lut"])
def test_every_byte_pattern_is_the_frozen_widening_on_the_device(mode):
    torch_npu = pytest.importorskip("torch_npu")  # noqa: F841
    device = torch.device(NPU_DEVICE)
    packed = all_patterns().to(device)
    actual = vector_fp4_decode.unpack_fp4_vector(packed, mode=mode, backend="triton")
    expected = unpack.unpack_fp4_to_int8(packed, backend="torch")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@needs_device
@pytest.mark.parametrize("tile", [512, 1024, 2048, 4096])
def test_tiles_and_a_masked_tail_agree_with_each_other(tile):
    pytest.importorskip("torch_npu")
    device = torch.device(NPU_DEVICE)
    packed = device_input(tile * 3 + 7).to(device)
    first = vector_fp4_decode.unpack_fp4_vector(packed, backend="triton", tile=tile)
    second = vector_fp4_decode.unpack_fp4_vector(packed, backend="triton", mode="lut", tile=tile)
    expected = unpack.unpack_fp4_to_int8(packed, backend="torch")
    assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(second.view(torch.uint8), expected.view(torch.uint8))


@needs_device
def test_a_two_dimensional_weight_keeps_its_rows_and_order():
    pytest.importorskip("torch_npu")
    device = torch.device(NPU_DEVICE)
    rows, columns = 17, 65
    packed = device_input(rows * columns, seed=29).reshape(rows, columns).to(device)
    actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="triton")
    assert tuple(actual.shape) == (rows, columns * 2)
    expected = unpack.unpack_fp4_to_int8(packed, backend="torch")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@needs_device
def test_a_reused_output_buffer_is_written_exactly_once():
    pytest.importorskip("torch_npu")
    device = torch.device(NPU_DEVICE)
    packed = device_input(4096, seed=37).to(device)
    out = torch.empty(8192, dtype=torch.int8, device=device)
    for _ in range(3):
        actual = vector_fp4_decode.unpack_fp4_vector(packed, backend="triton", out=out)
        assert actual is out
    expected = unpack.unpack_fp4_to_int8(packed, backend="torch")
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))