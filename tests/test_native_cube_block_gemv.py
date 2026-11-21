"""Tests for the opt-in native Cube block GEMV prototype.

Three layers, deliberately separable:

* **Portable, no torch devices.** The three bit helpers are re-executed as plain
  Python integers straight from the kernel source and compared against independent
  exact statements: an E4M3 code's E2M1/E4M3 value widened to BF16 bits is computed
  from the codebook and from ``struct``, never from the helper being tested.
* **CPU API.** The validators, refusals, the natural oracles and the merged-scale
  certification, with ``backend="torch"``.
* **NPU, opt-in.** ``RELIC_NATIVE_CUBE_TEST_DEVICE=npu:0`` runs the fused kernel and
  reports how far its changed reduction is from the natural FP32 oracle. These are
  difference measurements, not an equality claim: the whole point of the operator is
  that the reduction changed, so the tests assert closeness and NaN classification,
  never bit equality with the accepted window-13 rule.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import native_cube_block_gemv as native  # noqa: E402
from relic_core.kernels.block_gemm import _original_float_scale  # noqa: E402
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8  # noqa: E402

KERNELS = Path(__file__).resolve().parents[1] / "relic_core/kernels"
HELPERS = KERNELS / "_ascend_native_cube_block_kernels.py"
FROZEN = KERNELS / "_ascend_block_gemm_kernels.py"
NPU_DEVICE = __import__("os").environ.get("RELIC_NATIVE_CUBE_TEST_DEVICE", "")

#: Tile configuration for the gated merged-scale tests. That kernel holds wider
#: integer tiles than the per-K-32 kernel, and at BK=256/BN=64 the CANN unified
#: buffer allocation asked for 6,825,984 bits against 1,572,864 available on the
#: 910B3 build, so the probe configuration is the 8x smaller BK=128/BN=32 that the
#: public keyword arguments already accept. This changes no default.
SCALED_TILES = {"block_k": 128, "block_n": 32}


# -- portable: re-execute the kernel's own integer bit helpers ----------------------


class Scalar:
    """Minimal int32-shaped scalar the helpers' arithmetic is closed over."""

    def __init__(self, value):
        self.value = int(value.value if isinstance(value, Scalar) else value)

    def to(self, dtype):
        """Model the kernel's named casts: ``uint16`` masks, ``int16`` sign-extends."""
        if dtype == "uint16":
            return Scalar(self.value & 0xFFFF)
        if dtype == "int16":
            return Scalar(((self.value + 0x8000) & 0xFFFF) - 0x8000)
        return Scalar(self.value)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def __and__(self, other):
        return Scalar(self.value & int(Scalar(other)))

    __rand__ = __and__

    def __or__(self, other):
        return Scalar(self.value | int(Scalar(other)))

    __ror__ = __or__

    def __invert__(self):
        return Scalar(-1 - self.value)

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

    def __le__(self, other):
        return Scalar(self.value <= int(Scalar(other)))

    def __gt__(self, other):
        return Scalar(self.value > int(Scalar(other)))

    def __ge__(self, other):
        return Scalar(self.value >= int(Scalar(other)))

    def __eq__(self, other):
        return Scalar(self.value == int(Scalar(other)))

    def __ne__(self, other):
        return Scalar(self.value != int(Scalar(other)))


def _function(path: Path, name: str):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def scalar_helper(name: str, *dependencies: tuple[Path, str]):
    """Compile one helper plus its jit dependencies into plain-integer callables."""
    tl = type("tl", (), {
        "where": staticmethod(lambda c, x, y: Scalar(x if bool(c) else y)),
        "maximum": staticmethod(lambda x, y: Scalar(max(int(Scalar(x)), int(Scalar(y))))),
        "minimum": staticmethod(lambda x, y: Scalar(min(int(Scalar(x)), int(Scalar(y))))),
        "uint16": "uint16",
        "int16": "int16",
        "int32": "int32",
    })
    namespace = {"tl": tl}
    for path, function in dependencies:
        exec(compile(_function(path, function), str(path), "exec"), namespace)
    exec(compile(_function(HELPERS, name), str(HELPERS), "exec"), namespace)
    return namespace[name]


def e4m3_value(code: int) -> float:
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    if exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -magnitude if code & 128 else magnitude


def bf16_bits(value: float) -> int:
    """Independent BF16 bit pattern of an exactly representable value."""
    if math.isnan(value):
        return 0x7FC0
    if value == 0.0:
        return 0x8000 if math.copysign(1.0, value) < 0 else 0
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    assert (bits & 0xFFFF) == 0, "test value is not exactly representable in BF16"
    return bits >> 16


def test_every_e4m3_code_becomes_its_exact_bf16_bits():
    helper = scalar_helper("_e4_unit_bf16_bits")
    for code in range(256):
        unit = e4m3_value(code)
        # Both E4M3 NaN codes stay NaN in BF16 and keep their sign bit.
        expected = (0x7FC0 | ((code & 128) << 8)) if math.isnan(unit) else bf16_bits(unit)
        assert int(helper(Scalar(code))) & 0xFFFF == expected, f"E4M3 code {code:#04x}"


def test_every_e2m1_magnitude_becomes_its_exact_bf16_bits():
    helper = scalar_helper("_e2m1_unit_bf16_bits")
    expected = (0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0)
    assert [int(helper(Scalar(magnitude))) & 0xFFFF for magnitude in range(8)] == list(expected)


def scaled_reference(unit_bits: int, scale: int, zero: bool) -> int:
    """Independent statement of the merged-scale contract."""
    sign = unit_bits & 0x8000
    if scale == 255:  # E8M0 NaN
        return 0x7FC0 | sign
    if zero:
        return sign
    shifted = ((unit_bits >> 7) & 255) + scale - 127
    if not 1 <= shifted <= 254:
        return 0x7FC0 | sign  # not a normal BF16 value: refuse, do not round
    return sign | (shifted << 7) | (unit_bits & 127)


def test_every_scaled_lane_matches_the_exact_contract():
    e4 = scalar_helper("_e4_unit_bf16_bits", (FROZEN, "_ilog2"))
    e2 = scalar_helper("_e2m1_unit_bf16_bits")
    scaled = scalar_helper("_scaled_bf16_bits")
    units = [int(e4(Scalar(code))) & 0xFFFF for code in range(256)]
    units += [int(e2(Scalar(magnitude))) & 0xFFFF for magnitude in range(8)]
    for unit_bits in set(units):
        zero = (unit_bits & 0x7FFF) == 0
        for scale in (0, 1, 2, 3, 126, 127, 128, 129, 253, 254, 255):
            expected = scaled_reference(unit_bits, scale, zero)
            actual = int(scaled(Scalar(unit_bits), Scalar(scale), Scalar(int(zero)))) & 0xFFFF
            assert actual == expected, f"unit {unit_bits:#06x} scale {scale}"


def test_a_zero_lane_keeps_its_sign_and_a_nan_scale_still_wins():
    scaled = scalar_helper("_scaled_bf16_bits")
    assert int(scaled(Scalar(0x0000), Scalar(127), Scalar(1))) & 0xFFFF == 0x0000
    assert int(scaled(Scalar(0x8000), Scalar(0), Scalar(1))) & 0xFFFF == 0x8000
    assert int(scaled(Scalar(0x8000), Scalar(255), Scalar(1))) & 0xFFFF == 0xFFC0


def test_underflow_and_overflow_lanes_become_nan_not_a_rounded_value():
    scaled = scalar_helper("_scaled_bf16_bits")
    normal = 0x3F80  # 1.0
    # Byte 0 is 2**-127: below the smallest BF16 normal, so it must be refused.
    assert int(scaled(Scalar(normal), Scalar(0), Scalar(0))) & 0x7FFF == 0x7FC0
    # Byte 1 is 2**-126: exactly the smallest normal, so it is representable.
    assert int(scaled(Scalar(normal), Scalar(1), Scalar(0))) & 0xFFFF == 0x0080
    assert int(scaled(Scalar(normal), Scalar(254), Scalar(0))) & 0x7FFF != 0x7FC0
    assert int(scaled(Scalar(normal), Scalar(255), Scalar(0))) & 0x7FFF == 0x7FC0


def literal_left_bitwise_offenders(source: str) -> list[int]:
    """Line numbers of ``literal <bitwise> tensor`` expressions in JIT source.

    This frontend implements the reflected bitwise dunders on tensors only, so a
    Python int on the left of ``|``/``&``/``^`` raises ``AttributeError: 'int' object
    has no attribute 'type'`` *before* the kernel exists. An int on both sides is
    static arithmetic and is allowed; a tensor on the left always works.
    """
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        if not isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor)):
            continue
        if not (isinstance(node.left, ast.Constant) and isinstance(node.left.value, int)):
            continue
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            continue  # pure int arithmetic, promoted at trace time
        offenders.append(node.lineno)
    return offenders


def test_no_jit_source_puts_an_int_literal_left_of_a_bitwise_op():
    """Guard the first NPU compile failure: ``0x7FC0 | sign`` never reaches the JIT."""
    for path in (HELPERS, FROZEN):
        offenders = literal_left_bitwise_offenders(Path(path).read_text())
        assert offenders == [], f"{path} has int-literal-left bitwise ops at {offenders}"


def test_merged_scale_path_keeps_its_bit_patterns_16_bit():
    """Pin the dtype boundary the 910B3 unified-buffer budget forced.

    At BK=128/BN=32 the int32 merged-scale kernel asked for 1,690,624 bits against
    1,572,864 available, so the bit patterns that feed the Cube dot must stay 16-bit
    (the arithmetic inside the helper is still int32, which is the hardware-validated
    idiom). This test fails if a later edit widens them back.
    """
    source = HELPERS.read_text()
    scaled_bits = source[source.index("def _scaled_bf16_bits"):
                         source.index("def _bfloat16_tile")]
    assert ".to(tl.uint16)" in scaled_bits, "the merged-scale helper must return 16-bit bits"
    kernel = source[source.index("def native_cube_scaled_bf16_kernel"):]
    assert ".to(tl.uint16)" in kernel and ".to(tl.int16)" in kernel
    # The per-K-32 kernel already compiled and is benchmarked: it must NOT have
    # adopted the 16-bit boundary, so its arithmetic stays exactly as validated.
    block32 = source[source.index("def native_cube_block32_kernel"):
                     source.index("def native_cube_scaled_bf16_kernel")]
    assert "to(tl.uint16)" not in block32 and "to(tl.int16)" not in block32


def test_the_guard_itself_catches_the_reported_pattern():
    """Negative control: the guard must fire on the exact failure and pass the fix."""
    assert literal_left_bitwise_offenders("x = tl.where(c, 0x7FC0 | sign, y)") == [1]
    assert literal_left_bitwise_offenders("x = tl.where(c, sign | 0x7FC0, y)") == []
    assert literal_left_bitwise_offenders("x = 0x7FC0 | 0x8000") == []  # static ints
    assert literal_left_bitwise_offenders("x = a & 0x40") == []  # tensor on the left


# -- CPU API: contract, oracles, certification --------------------------------------


def small_operands(k=64, n=5, seed=0, scales=None):
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.int32).to(torch.uint8)
    blocks = torch.tensor(scales or [[130, 131], [129, 132], [128, 128], [133, 129], [127, 130]],
                          dtype=torch.uint8)[:n]
    a = torch.randint(0, 240, (k,), generator=generator, dtype=torch.int32).to(torch.uint8)
    a = a.masked_fill((a & 127) == 127, 0x38).to(torch.uint8)
    sa = torch.full((k // 32,), 127, dtype=torch.uint8)
    return a, sa, packed, blocks


def exact_reference(values, a_codes, a_scale, scale_bytes, *, block_rows=1):
    """The documented natural order: per-K-32 FP32 sum, then separate SA and SB."""
    k = a_codes.numel()
    a_unit = native._unit_values(a_codes)
    sa = native._e8m0_values(a_scale)
    sb = native._e8m0_values(scale_bytes.repeat_interleave(block_rows, dim=0)[: values.shape[0]])
    out = torch.zeros(values.shape[0], dtype=torch.float32)
    for block in range(k // 32):
        span = slice(block * 32, (block + 1) * 32)
        partial = torch.sum(a_unit[span][None, :] * values[:, span], dim=-1)
        out = out + (partial * sa[block]) * sb[:, block]
    return out


def test_packed_and_int8_modes_agree_and_match_the_exact_reference():
    a, sa, packed, blocks = small_operands()
    packed_out = native.native_cube_fp4_gemv(a, sa, packed, blocks, backend="torch")
    q = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(blocks, backend="torch")
    int8_out = native.native_cube_fp4_gemv(a, sa, q, half, b_format="int8_x2", backend="torch")
    assert tuple(packed_out.shape) == (1, packed.shape[0])
    assert packed_out.dtype == torch.float32
    expected = exact_reference(native._packed_values(packed), a, sa, blocks)
    assert torch.equal(packed_out[0], expected)
    assert torch.equal(int8_out[0], expected)


def test_fp8_square_and_row_grids_match_the_exact_reference():
    a, sa, _, blocks = small_operands()
    n, k = 5, a.numel()
    codes = torch.randint(0, 240, (n, k), generator=torch.Generator().manual_seed(3),
                          dtype=torch.int32).to(torch.uint8)
    codes = codes.masked_fill((codes & 127) == 127, 0x38).to(torch.uint8)
    square = torch.tensor([[127, 128]], dtype=torch.uint8)
    row = blocks
    values = native._unit_values(codes)
    square_out = native.native_cube_fp8_gemv(a, sa, codes, square, block_rows=32, backend="torch")
    row_out = native.native_cube_fp8_gemv(a, sa, codes, row, block_rows=1, backend="torch")
    assert torch.equal(square_out[0], exact_reference(values, a, sa, square, block_rows=32))
    assert torch.equal(row_out[0], exact_reference(values, a, sa, row))


def test_scaled_mode_needs_certification_and_then_matches_its_own_oracle():
    a, sa, packed, blocks = small_operands()
    with pytest.raises(ValueError, match="scaled_bf16_safe=True"):
        native.native_cube_fp4_gemv(a, sa, packed, blocks, mode="scaled_bf16", backend="torch")
    assert native.scaled_bf16_weight_representable(packed, blocks, fp4=True)
    assert native.scaled_bf16_activation_representable(a, sa)
    out = native.native_cube_fp4_gemv(a, sa, packed, blocks, mode="scaled_bf16", backend="torch",
                                      scaled_bf16_safe=True)
    values = native._packed_values(packed)
    a_value = native._unit_values(a) * native._e8m0_values(sa).repeat_interleave(32)
    b_value = values * native._e8m0_values(blocks).repeat_interleave(32, dim=1)
    assert torch.equal(out[0], torch.sum(b_value * a_value[None, :], dim=-1))


def test_certification_refuses_an_underflowing_scale_and_the_nan_byte():
    a, sa, packed, blocks = small_operands()
    assert native.scaled_bf16_weight_representable(packed, blocks, fp4=True)
    underflow = blocks.clone()
    underflow[0, 0] = 1  # 2**-126 on a nonzero lane: subnormal in BF16
    assert not native.scaled_bf16_weight_representable(packed, underflow, fp4=True)
    nan_scale = blocks.clone()
    nan_scale[1, 1] = 255
    assert not native.scaled_bf16_weight_representable(packed, nan_scale, fp4=True)
    nan_code = a.clone()
    nan_code[0] = 0x7F
    assert not native.scaled_bf16_activation_representable(nan_code, sa)
    assert native.scaled_bf16_activation_representable(a, sa)


def test_a_safe_fp4_store_is_certified_through_both_container_formats():
    """The packed and INT8-times-two views of one store must certify identically."""
    a, sa, packed, blocks = small_operands()
    widened = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(blocks, backend="torch")
    assert native.scaled_bf16_weight_representable(packed, blocks, fp4=True)
    assert native.scaled_bf16_weight_representable(widened, half, fp4=True, b_format="int8_x2")
    assert native.native_cube_fp4_gemv(a, sa, widened, half, b_format="int8_x2",
                                       mode="scaled_bf16", backend="torch",
                                       scaled_bf16_safe=True).shape == (1, packed.shape[0])


def test_the_natural_oracle_keeps_the_two_rescales_separate():
    """One K-32 block: the oracle must be exactly ``(sum * SA) * SB``, not fused."""
    k, n = 32, 3
    a = torch.tensor([0x38] * k, dtype=torch.uint8)  # every activation value 1.0
    sa = torch.tensor([127], dtype=torch.uint8)
    q = torch.tensor([[1, 2, 3], [4, 6, 8], [12, 0, 0]], dtype=torch.int8)
    q = torch.cat([q, torch.zeros(n, k - 3, dtype=torch.int8)], dim=1)
    half = torch.tensor([[4.0], [2.0], [1.0]], dtype=torch.float32)  # 2x the full scale
    out = native.native_cube_fp4_gemv(a, sa, q, half, b_format="int8_x2", backend="torch")
    values = q.to(torch.float32) * 0.5
    full = _original_float_scale(half, half=True)
    expected = (torch.sum(values * torch.ones(1, k), dim=-1) * 1.0) * full[:, 0]
    assert torch.equal(out[0], expected)


def test_output_buffer_is_used_returned_and_never_aliases():
    a, sa, packed, blocks = small_operands()
    out = torch.empty((1, packed.shape[0]), dtype=torch.float32)
    result = native.native_cube_fp4_gemv(a, sa, packed, blocks, backend="torch", out=out)
    assert result is out
    with pytest.raises(ValueError, match="shape"):
        native.native_cube_fp4_gemv(a, sa, packed, blocks, backend="torch",
                                    out=torch.empty((1, packed.shape[0] - 1)))
    # An out buffer that overlaps an input must be refused, not written through.
    shared = torch.empty((1, blocks.shape[0]), dtype=torch.float32)
    overlapping = shared.view(torch.uint8).reshape(-1)[: blocks.numel()].reshape(blocks.shape)
    with pytest.raises(ValueError, match="alias"):
        native.native_cube_fp4_gemv(a, sa, packed, overlapping, backend="torch", out=shared)


@pytest.mark.parametrize("call, error", [
    (lambda a, sa, p, b: native.native_cube_fp4_gemv([[0]], sa, p, b), TypeError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a.float(), sa, p, b), TypeError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(torch.stack([a, a]), sa, p, b), ValueError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa[:1], p, b), ValueError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa, p, b, mode="bogus"), ValueError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa, p, b, block_n=16), ValueError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa, p, b, b_format="other"), ValueError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa, p.float(), b, b_format="int8_x2"), TypeError),
    (lambda a, sa, p, b: native.native_cube_fp4_gemv(a, sa, p, b, scaled_bf16_safe=True), ValueError),
])
def test_bad_operands_are_refused_by_name(call, error):
    a, sa, packed, blocks = small_operands()
    with pytest.raises(error):
        call(a, sa, packed, blocks)


def test_index_width_and_grid_are_decided_before_any_product():
    assert native._index64(1, 7168, False) is False
    assert native._index64(2**31 // 32, 32, False) is True
    assert native._index64(2**31, 64, True) is True
    assert native.MAX_PROGRAMS == 65535


def test_contract_attributes_name_the_two_changed_rules():
    assert native.NUMERICAL_CONTRACT == "native_cube_no_reference_window"
    assert native.SCALED_BF16_CONTRACT == "native_cube_scaled_bf16_pre_mma_scaling"
    assert native.SUPPORTED_BLOCK_N == (32, 64)
    assert native.SUPPORTED_BLOCK_K == (128, 256)


# -- NPU: the fused kernel's changed reduction, measured against the natural oracle --


needs_device = pytest.mark.skipif(
    not NPU_DEVICE, reason="set RELIC_NATIVE_CUBE_TEST_DEVICE=npu:0 on a leased card")


def device_operands(k=256, n=64, seed=1):
    a, sa, packed, blocks = small_operands(k=k, n=n, seed=seed,
                                           scales=[[128 + (i % 3)] * (k // 32) for i in range(n)])
    return a.to(NPU_DEVICE), sa.to(NPU_DEVICE), packed.to(NPU_DEVICE), blocks.to(NPU_DEVICE)


def diff(actual, expected) -> dict:
    delta = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().clamp_min(1e-30)
    return {"max_abs": float(delta.max()), "max_rel": float((delta / scale).max()),
            "median_abs": float(delta.median()), "finite": bool(torch.isfinite(actual).all())}


@needs_device
@pytest.mark.parametrize("block_n", [32, 64])
def test_npu_block32_fp4_stays_close_to_the_natural_oracle(block_n):
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    cpu_a, cpu_sa, cpu_p, cpu_b = (t.to("cpu") for t in (a, sa, packed, blocks))
    expected = native.native_cube_fp4_gemv(cpu_a, cpu_sa, cpu_p, cpu_b, backend="torch")
    actual = native.native_cube_fp4_gemv(a, sa, packed, blocks, block_n=block_n)
    stats = diff(actual.cpu()[0], expected[0])
    assert stats["finite"] and stats["max_rel"] < 1e-2, stats


@needs_device
def test_npu_packed_and_int8_modes_produce_the_same_result():
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    packed_out = native.native_cube_fp4_gemv(a, sa, packed, blocks)
    q = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(blocks, backend="torch")
    int8_out = native.native_cube_fp4_gemv(a, sa, q.to(NPU_DEVICE), half.to(NPU_DEVICE),
                                          b_format="int8_x2")
    assert torch.equal(packed_out.cpu(), int8_out.cpu())


@needs_device
def test_npu_fp8_block32_stays_close_to_the_natural_oracle():
    import torch_npu  # noqa: F401
    k, n = 256, 64
    a, sa, _, _ = small_operands(k=k, n=n, seed=4)
    codes = torch.randint(0, 240, (n, k), generator=torch.Generator().manual_seed(5),
                          dtype=torch.int32).to(torch.uint8)
    codes = codes.masked_fill((codes & 127) == 127, 0x38).to(torch.uint8)
    grid = torch.full((n, k // 32), 127, dtype=torch.uint8)
    expected = native.native_cube_fp8_gemv(a, sa, codes, grid, block_rows=1, backend="torch")
    actual = native.native_cube_fp8_gemv(a.to(NPU_DEVICE), sa.to(NPU_DEVICE), codes.to(NPU_DEVICE),
                                         grid.to(NPU_DEVICE), block_rows=1)
    stats = diff(actual.cpu()[0], expected[0])
    assert stats["finite"] and stats["max_rel"] < 1e-2, stats


@needs_device
def test_npu_scaled_bf16_mode_is_exact_in_its_operands_and_close_to_its_oracle():
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    cpu = tuple(t.to("cpu") for t in (a, sa, packed, blocks))
    assert native.scaled_bf16_weight_representable(cpu[2], cpu[3], fp4=True)
    assert native.scaled_bf16_activation_representable(cpu[0], cpu[1])
    expected = native.native_cube_fp4_gemv(*cpu, mode="scaled_bf16", backend="torch",
                                           scaled_bf16_safe=True)
    actual = native.native_cube_fp4_gemv(a, sa, packed, blocks, mode="scaled_bf16",
                                        scaled_bf16_safe=True, **SCALED_TILES)
    stats = diff(actual.cpu()[0], expected[0])
    assert stats["finite"] and stats["max_rel"] < 1e-2, stats


@needs_device
def test_npu_nan_classification_follows_the_original_bytes():
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    nan_a = a.clone()
    nan_a[0] = 0x7F  # E4M3 NaN activation code
    out = native.native_cube_fp4_gemv(nan_a, sa, packed, blocks)
    assert bool(torch.isnan(out).all()), "a NaN activation code must invalidate every column"
    nan_b = blocks.clone()
    nan_b[3, 0] = 255  # E8M0 NaN weight scale
    out = native.native_cube_fp4_gemv(a, sa, packed, nan_b)
    assert bool(torch.isnan(out[0, 3]))
    assert bool(torch.isfinite(out[0, :3]).all())


@needs_device
def test_npu_scaled_mode_keeps_nan_codes_nan():
    """A NaN code must stay NaN even where the exponent add would otherwise land."""
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    safe = torch.full_like(blocks, 128)  # 2**1: representable for every codebook value
    nan_a = a.clone()
    nan_a[0] = 0x7F  # E4M3 NaN activation code
    out = native.native_cube_fp4_gemv(nan_a, sa, packed, safe, mode="scaled_bf16",
                                      scaled_bf16_safe=True, **SCALED_TILES)
    assert bool(torch.isnan(out).all())
    nan_b = blocks.clone()
    nan_b[:, 0] = 255  # E8M0 NaN in the first K block poisons every column
    out = native.native_cube_fp4_gemv(a, sa, packed, nan_b, mode="scaled_bf16",
                                      scaled_bf16_safe=True, **SCALED_TILES)
    assert bool(torch.isnan(out).all())


@needs_device
def test_npu_block_n_tiles_agree_with_each_other():
    import torch_npu  # noqa: F401
    a, sa, packed, blocks = device_operands()
    wide = native.native_cube_fp4_gemv(a, sa, packed, blocks, block_n=64)
    narrow = native.native_cube_fp4_gemv(a, sa, packed, blocks, block_n=32)
    assert torch.equal(wide.cpu(), narrow.cpu())