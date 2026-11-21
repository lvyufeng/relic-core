"""The fused FP8 activation quantize+decode pass, checked without an accelerator.

The claim under test is narrow and exact: one kernel that encodes the original E4M3
code byte and E8M0 scale byte and then decodes that same byte pair must produce the
**same BF16 bits** as ``quantize_block_fp8`` followed by
``unpack_fp8_to_bf16(block_shape=(1, 32))``. Bit equality, not a tolerance.

Three independent layers check it here, none of which needs Triton:

* ``backend="torch"`` is the composed frozen reference; it is compared against the
  two operators called directly, which pins the wrapper's shape handling.
* The fused kernel's *body* is emulated in Python in the same order, with the two
  float steps done in FP32 (torch) and **every integer decision executed from the
  real ``@triton.jit`` source** by re-executing the frozen helper bodies on scalars
  (the idiom ``tests/test_block_fp8_quant_math.py`` uses). The emulation is then held
  to the frozen composition over ties, scale boundaries, the amax floor, zero and
  signed zero, subnormals, the ``+-448`` clamp and non-finite blocks. Because the
  helpers are the real ones, a mis-ordered or mis-scaled decode fails rather than
  agreeing with itself.
* AST checks pin the parts a CPU cannot execute: no division and no ``log2``/``exp2``
  in the encode path, the released constants passed as constexpr arguments, the
  grid bounded at the device cap with striding, the pointer width decided before any
  offset product, the two code buffers written only under ``WRITE_CODES``, and the
  decode consuming the produced scale *byte*.

Device compilation and hardware timing are separate claims, and belong to the lead's
run on a leased card: this file says nothing about either.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import block_fp8_quant, unpack
from relic_core.kernels import fused_fp8_activation_decode as fused

KERNEL_SOURCE = (Path(__file__).resolve().parents[1]
                 / "relic_core/kernels/_ascend_fused_fp8_activation_decode_kernels.py")
QUANT_SOURCE = (Path(__file__).resolve().parents[1]
                / "relic_core/kernels/_ascend_fp8_quant_kernels.py")
UNPACK_SOURCE = (Path(__file__).resolve().parents[1]
                 / "relic_core/kernels/_ascend_unpack_kernels.py")

BLOCK = block_fp8_quant.BLOCK
FLOOR = fused.FLOOR
INV_FP8_MAX = fused.INV_FP8_MAX


# -- executing the frozen helper bodies on integer scalars ---------------------


class Bits:
    """The subset of Triton integer-vector semantics the bit helpers rely on.

    Comparisons return a ``Bits`` rather than a ``bool`` because the decode helper
    ORs two of them together before converting the result, which a Python ``bool``
    cannot carry. ``tl.where`` still only ever reads them through ``__bool__``.
    """

    def __init__(self, value):
        self.value = _wrap32(int(value.value if isinstance(value, Bits) else value))

    def to(self, dtype):
        return self

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def _binary(self, other, operation):
        return Bits(operation(self.value, _int(other)))

    def __neg__(self):
        return Bits(-self.value)

    def __pos__(self):
        return Bits(self.value)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return Bits(_int(other)) - self

    def __and__(self, other):
        return self._binary(other, lambda a, b: a & b)

    __rand__ = __and__

    def __or__(self, other):
        return self._binary(other, lambda a, b: a | b)

    __ror__ = __or__

    def __lshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Bits(self.value << count)

    def __rlshift__(self, other):
        return Bits(_int(other)) << self

    def __rshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Bits(self.value >> count)

    def __rrshift__(self, other):
        return Bits(_int(other)) >> self

    def __eq__(self, other):
        return Bits(1 if self.value == _int(other) else 0)

    def __ne__(self, other):
        return Bits(1 if self.value != _int(other) else 0)

    def __lt__(self, other):
        return Bits(1 if self.value < _int(other) else 0)

    def __le__(self, other):
        return Bits(1 if self.value <= _int(other) else 0)

    def __gt__(self, other):
        return Bits(1 if self.value > _int(other) else 0)

    def __ge__(self, other):
        return Bits(1 if self.value >= _int(other) else 0)

    __hash__ = None


def _wrap32(value):
    return (int(value) + 2**31) % 2**32 - 2**31


def _int(value):
    return int(value.value if isinstance(value, Bits) else value)


def scalar_functions(source: Path) -> dict:
    """Every top-level ``@triton.jit`` helper in ``source``, as callable Python.

    Only the helper under test is ever called; the device kernels in the same
    namespace are inert because nothing invokes them.
    """
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    triton = SimpleNamespace(
        where=lambda condition, x, y: Bits(x if bool(condition) else y),
        minimum=lambda x, y: Bits(min(_int(x), _int(y))),
        maximum=lambda x, y: Bits(max(_int(x), _int(y))),
        uint16="uint16", int32="int32", float32="float32",
    )
    namespace = {"tl": triton}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


QUANT_HELPERS = scalar_functions(QUANT_SOURCE)
UNPACK_HELPERS = scalar_functions(UNPACK_SOURCE)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


# -- the fused kernel body, emulated -------------------------------------------


def emulate_fused(x: torch.Tensor):
    """The kernel body in Python: FP32 float steps, frozen integer decisions.

    Returns ``(bf16_bit_patterns, codes, scales)`` for a ``[rows, width]`` input.
    Every line mirrors the kernel in ``_ascend_fused_fp8_activation_decode_kernels``;
    the two "float" steps are the same IEEE FP32 operations Triton would emit and
    the three integer decisions are the frozen helper bodies themselves.
    """
    width = int(x.shape[-1])
    flat = x.reshape(-1, width).to(torch.float32)
    rows = int(flat.shape[0])
    blocks = width // BLOCK
    scale_exponent = QUANT_HELPERS["_scale_exponent_from_bits"]
    pow2_bits = QUANT_HELPERS["_pow2_bits"]
    e4m3_byte = QUANT_HELPERS["_e4m3_byte_from_bits"]
    nonfinite_bits = QUANT_HELPERS["_nonfinite_bits"]
    decode_bits = UNPACK_HELPERS["_e4m3_e8m0_bf16_bits"]
    floor = torch.tensor(FLOOR, dtype=torch.float32)
    inverse_max = torch.tensor(INV_FP8_MAX, dtype=torch.float32)
    patterns = torch.empty((rows, width), dtype=torch.int32)
    codes = torch.empty((rows, width), dtype=torch.uint8)
    scales = torch.empty((rows, blocks), dtype=torch.uint8)
    for row in range(rows):
        for block in range(blocks):
            values = flat[row, block * BLOCK:(block + 1) * BLOCK]
            magnitude = values.view(torch.int32) & 0x7FFFFFFF
            block_nonfinite = max(int(nonfinite_bits(Bits(int(one))))
                                  for one in magnitude) != 0
            amax = f32_value(int(magnitude.max()))
            scaled = torch.maximum(torch.tensor(amax, dtype=torch.float32), floor) * inverse_max
            exponent = int(scale_exponent(Bits(_wrap32(f32_bits(float(scaled))))))
            inverse = f32_value(int(pow2_bits(Bits(-exponent))))
            normalized = torch.clamp(values * torch.tensor(inverse, dtype=torch.float32),
                                     -448.0, 448.0)
            block_codes = [int(e4m3_byte(Bits(_wrap32(int(one)))))
                           for one in normalized.view(torch.int32)]
            block_scale = exponent + 127
            if block_nonfinite:
                block_codes = [0x7F] * BLOCK
                block_scale = 255
            for lane, code in enumerate(block_codes):
                patterns[row, block * BLOCK + lane] = int(
                    decode_bits(Bits(code), Bits(block_scale)))
            codes[row, block * BLOCK:(block + 1) * BLOCK] = torch.tensor(
                block_codes, dtype=torch.uint8)
            scales[row, block] = block_scale
    return patterns, codes, scales


def reference_bits(x: torch.Tensor):
    """The two frozen operators composed: the contract the fusion must meet."""
    width = int(x.shape[-1])
    flat = x.reshape(-1, width)
    codes, scales = block_fp8_quant.quantize_block_fp8(flat, backend="torch")
    decoded = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1, BLOCK),
                                        backend="torch")
    return decoded, codes, scales


def pattern_row(patterns: torch.Tensor) -> list:
    return [int(one) & 0xFFFF for one in patterns.reshape(-1).tolist()]


def decoded_row(decoded: torch.Tensor) -> list:
    return [int(one) & 0xFFFF for one in decoded.reshape(-1).view(torch.int16).tolist()]


# -- value batteries -----------------------------------------------------------


def _block(rows: int, blocks: int, values) -> torch.Tensor:
    """A ``[rows, blocks * 32]`` FP32 matrix filled from ``values``, repeated to fit."""
    template = torch.tensor(list(values), dtype=torch.float32)
    assert template.numel() >= BLOCK, "a battery block is at least one 32-wide block"
    needed = rows * blocks * BLOCK
    return template.repeat(-(-needed // template.numel()))[:needed].reshape(
        rows, blocks * BLOCK)


def octave_battery() -> torch.Tensor:
    """Magnitudes across every E4M3 octave, including clamp, tie and subnormal edges."""
    values = []
    for exponent in (0, -1, -2, -3, -4, -6, -7, -8, -9, -10, -12, -20, -126, 1, 5, 127):
        for step in (-1, 0, 1):
            magnitude = math.ldexp(1.0, exponent) * (1 + step / 16)
            values.extend((magnitude, -magnitude))
    values.extend((448.0, -448.0, 449.0, -449.0, 1e-4, -1e-4, 0.0, -0.0,
                   2**-9, 2**-10, 2**-11, 2**-149, 3.4e38, -3.4e38))
    return _block(1, 2, values)


def tie_battery() -> torch.Tensor:
    """Values landing exactly on E4M3 rounding ties for several block scales.

    A tie needs the retained significand to be odd at the discarded bit; sweeping the
    odd multiples of the half-quantum of each octave walks them without assuming one
    scale.
    """
    values = []
    for exponent in range(-8, 8):
        unit = math.ldexp(1.0, exponent)
        for numerator in (1, 3, 5, 7, 9, 11, 13, 15):
            for nudge in (0.0, 1e-7, -1e-7):
                values.append(unit * (1 + numerator / 16 + nudge))
    while len(values) < BLOCK:
        values.append(1.0)
    return _block(1, 3, values)


def tiny_battery() -> torch.Tensor:
    """A block whose amax sits below the released ``1e-4`` floor."""
    values = [0.0] * BLOCK
    for index in range(BLOCK):
        values[index] = math.ldexp(1.0, -30 - index)
    return _block(2, 1, values)


def scale_sweep_battery() -> torch.Tensor:
    """One block per octave, so the scale-byte decision is swept across its range.

    Each block's amax is exactly ``2**e``, which puts the ceil-power-of-two scale on
    an exact power-of-two boundary for that octave -- the case where an off-by-one
    in the ceil or in the ``+127`` bias is visible in both the code bytes and the
    decoded values.
    """
    blocks = []
    for exponent in (-40, -30, -22, -14, -10, -1, 0, 1, 3, 8, 20, 60, 120, 127):
        top = math.ldexp(1.0, exponent)
        blocks.extend(top * (1 - lane / 64) for lane in range(BLOCK))
    return _block(len(blocks) // BLOCK, 1, blocks)


def nonfinite_battery() -> torch.Tensor:
    """One infinity block and one NaN block, plus a finite neighbour."""
    infinity = [1.0] * BLOCK
    infinity[3] = float("inf")
    nan = [1.0] * BLOCK
    nan[7] = float("nan")
    return torch.cat((_block(1, 1, infinity), _block(1, 1, nan),
                      _block(1, 1, [0.5] * BLOCK)), dim=0)


def negative_zero_battery() -> torch.Tensor:
    negative = [-0.0] * BLOCK
    negative[0] = 1.0
    return _block(1, 1, negative)


def random_battery(dtype: torch.dtype, seed: int, rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, blocks * BLOCK, generator=generator) * 8
    return values.to(dtype)


BATTERIES = {
    "octaves": octave_battery,
    "ties": tie_battery,
    "tiny": tiny_battery,
    "scale_sweep": scale_sweep_battery,
    "nonfinite": nonfinite_battery,
    "negative_zero": negative_zero_battery,
    "random_bf16": lambda: random_battery(torch.bfloat16, 2001, 4, 3),
    "random_fp32": lambda: random_battery(torch.float32, 2002, 3, 5),
    "wide_random_bf16": lambda: random_battery(torch.bfloat16, 2003, 2, 16),
}


def test_the_batteries_are_not_degenerate():
    """Anti-vacuity: the comparison above must be exercised over many code bytes.

    A battery set that decoded to one or two patterns would agree with any
    implementation, including a wrong one.
    """
    codes_seen, scales_seen = set(), set()
    for make in BATTERIES.values():
        _, codes, scales = emulate_fused(make())
        codes_seen |= set(codes.reshape(-1).tolist())
        scales_seen |= set(scales.reshape(-1).tolist())
    assert len(codes_seen) >= 48, f"only {len(codes_seen)} distinct code bytes were exercised"
    assert len(scales_seen) >= 10, f"only {len(scales_seen)} distinct scale bytes were exercised"


def test_the_bit_comparison_has_teeth_on_the_decoded_scale():
    """Negative control: decoding the same bytes one octave down must be rejected."""
    x = BATTERIES["octaves"]()
    patterns, codes, scales = emulate_fused(x)
    decoded = decoded_row(reference_bits(x)[0])
    assert pattern_row(patterns) == decoded
    blocks = int(x.shape[-1]) // BLOCK
    wrong = []
    for row in range(int(x.shape[0])):
        for block in range(blocks):
            scale = int(scales[row, block]) - 1
            for lane in range(BLOCK):
                code = int(codes[row, block * BLOCK + lane])
                wrong.append(int(UNPACK_HELPERS["_e4m3_e8m0_bf16_bits"](
                    Bits(code), Bits(scale))) & 0xFFFF)
    assert wrong != decoded, "the oracle accepted a one-octave scale error"


# -- the token of the contract: identical bits ---------------------------------


@pytest.mark.parametrize("name", sorted(BATTERIES))
def test_the_fused_body_matches_the_two_operator_composition_bit_for_bit(name):
    """Encode-then-decode in registers equals encode, store, reload, decode."""
    x = BATTERIES[name]()
    patterns, codes, scales = emulate_fused(x)
    decoded, expected_codes, expected_scales = reference_bits(x)
    assert pattern_row(patterns) == decoded_row(decoded)
    assert codes.reshape(-1).tolist() == expected_codes.reshape(-1).tolist()
    assert scales.reshape(-1).tolist() == expected_scales.reshape(-1).tolist()


@pytest.mark.parametrize("name", sorted(BATTERIES))
def test_the_torch_backend_is_the_two_frozen_operators(name):
    """``backend="torch"`` is the composition itself, over any input rank."""
    x = BATTERIES[name]()
    decoded, codes, scales = fused.quantize_decode_fp8(
        x, backend="torch", return_codes=True)
    expected, expected_codes, expected_scales = reference_bits(x)
    assert decoded.dtype == torch.bfloat16 and decoded.shape == x.shape
    assert decoded_row(decoded) == decoded_row(expected)
    assert codes.reshape(-1).tolist() == expected_codes.reshape(-1).tolist()
    assert scales.reshape(-1).tolist() == expected_scales.reshape(-1).tolist()


def test_the_returned_codes_are_the_original_quantizer_bytes():
    """``return_codes`` is the frozen quantizer's output, not a second encoding."""
    x = random_battery(torch.bfloat16, 2004, 3, 4)
    _, codes, scales = fused.quantize_decode_fp8(x, backend="torch", return_codes=True)
    expected_codes, expected_scales = block_fp8_quant.quantize_block_fp8(
        x.reshape(-1, x.shape[-1]), backend="torch")
    assert (codes.view(torch.uint8).view(torch.int16)
            == expected_codes.reshape(codes.shape).view(torch.int16)).all()
    assert scales.shape == (*x.shape[:-1], x.shape[-1] // BLOCK)
    assert scales.reshape(-1).tolist() == expected_scales.reshape(-1).tolist()


@pytest.mark.parametrize("shape", [(BLOCK,), (2, BLOCK), (2, 3, 2 * BLOCK), (0, BLOCK)])
def test_arbitrary_rank_and_empty_input_keep_the_shape_contract(shape):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    decoded, codes, scales = fused.quantize_decode_fp8(x, backend="torch",
                                                       return_codes=True)
    assert decoded.shape == shape
    assert codes.shape == shape
    assert scales.shape == (*shape[:-1], shape[-1] // BLOCK)
    if x.numel() == 0:
        assert decoded.numel() == 0


def test_nonfinite_blocks_keep_the_frozen_nan_contract():
    """Both the code bytes and the decoded BF16 stay NaN where the frozen pair is."""
    x = nonfinite_battery()
    decoded, codes, scales = fused.quantize_decode_fp8(x, backend="torch",
                                                       return_codes=True)
    assert bool(torch.isnan(decoded[:2]).all()), "an infinity or NaN block decodes to NaN"
    assert codes[0].tolist() == [0x7F] * BLOCK and codes[1].tolist() == [0x7F] * BLOCK
    assert scales[0].item() == 255 and scales[1].item() == 255
    assert bool(torch.isfinite(decoded[2]).all())


def test_a_negative_zero_keeps_its_sign_through_the_decode():
    x = negative_zero_battery()
    decoded, codes, _ = fused.quantize_decode_fp8(x, backend="torch", return_codes=True)
    assert int(codes[0, 1]) == 0x80, "the released encoder keeps the sign of a zero"
    assert bool(torch.signbit(decoded[0, 1])), "and the decode keeps it too"


def test_out_is_used_and_returned_and_must_not_alias():
    x = random_battery(torch.bfloat16, 2005, 2, 2)
    out = torch.empty_like(x)
    returned = fused.quantize_decode_fp8(x, backend="torch", out=out)
    assert returned is out
    assert decoded_row(returned) == decoded_row(reference_bits(x)[0])
    with pytest.raises(ValueError, match="must not alias"):
        fused.quantize_decode_fp8(x, backend="torch", out=x)


@pytest.mark.parametrize("call,error,match", [
    (lambda x: fused.quantize_decode_fp8(x[:, ::2], backend="torch"),
     ValueError, "contiguous"),
    (lambda x: fused.quantize_decode_fp8(x.to(torch.float16), backend="torch"),
     TypeError, "BF16 or FP32"),
    (lambda x: fused.quantize_decode_fp8(x[:, :-1], backend="torch"),
     ValueError, "multiple of 32"),
    (lambda x: fused.quantize_decode_fp8(torch.tensor(1.0), backend="torch"),
     ValueError, "rank >= 1"),
    (lambda x: fused.quantize_decode_fp8(x, backend="bogus"),
     ValueError, "backend must be"),
    (lambda x: fused.quantize_decode_fp8(x, backend="triton"),
     ValueError, "requires an NPU input"),
    (lambda x: fused.quantize_decode_fp8(x, backend="torch", return_codes=1),
     TypeError, "return_codes must be a bool"),
    (lambda x: fused.quantize_decode_fp8(x, backend="torch", out=torch.empty_like(x).float()),
     TypeError, "bfloat16"),
    (lambda x: fused.quantize_decode_fp8(
        x, backend="torch", out=torch.empty(2, 1, dtype=torch.bfloat16)),
     ValueError, "must have shape"),
])
def test_bad_arguments_are_refused_by_name(call, error, match):
    x = torch.zeros(2, 2 * BLOCK, dtype=torch.bfloat16)
    with pytest.raises(error, match=match):
        call(x)


# -- AST contracts the CPU cannot execute --------------------------------------


def kernel_tree() -> ast.Module:
    return ast.parse(KERNEL_SOURCE.read_text())


def test_the_frozen_constants_are_the_released_numbers():
    assert BLOCK == 32
    assert fused.BLOCK is block_fp8_quant.BLOCK
    assert f32_value(f32_bits(FLOOR)) == 1e-4 or FLOOR == f32_value(f32_bits(1e-4))
    assert FLOOR == f32_value(f32_bits(1e-4))
    assert INV_FP8_MAX == f32_value(f32_bits(1.0 / 448.0))


def test_the_encode_and_decode_math_is_imported_not_retyped():
    """Every non-kernel helper must come from the frozen modules by name."""
    imports = {}
    for node in kernel_tree().body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.name] = node.module
    assert imports["_scale_exponent_from_bits"] == "_ascend_fp8_quant_kernels"
    assert imports["_e4m3_byte_from_bits"] == "_ascend_fp8_quant_kernels"
    assert imports["_pow2_bits"] == "_ascend_fp8_quant_kernels"
    assert imports["_nonfinite_bits"] == "_ascend_fp8_quant_kernels"
    assert imports["_e4m3_e8m0_bf16_bits"] == "_ascend_unpack_kernels"
    defined = [node.name for node in kernel_tree().body
               if isinstance(node, ast.FunctionDef)]
    assert defined == ["fused_fp8_activation_decode_kernel"], \
        "the kernel module defines no second copy of the frozen arithmetic"


def test_the_encode_path_has_no_division_and_no_float_logarithm():
    tree = kernel_tree()
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Div)], \
        "the released scale and encode are integer bit arithmetic, never a division"
    source = KERNEL_SOURCE.read_text()
    for forbidden in ("log2(", "exp2(", "frexp(", "float8_e4m3fn", "float8_e5m2"):
        assert forbidden not in source, f"{forbidden} has no place in the encode path"


def test_the_released_constants_arrive_as_constexpr_kernel_arguments():
    kernel = next(node for node in kernel_tree().body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "fused_fp8_activation_decode_kernel")
    constexpr = {}
    for argument in kernel.args.args:
        annotation = argument.annotation
        constexpr[argument.arg] = (isinstance(annotation, ast.Attribute)
                                  and annotation.attr == "constexpr")
    for name in ("NUM_BLOCKS", "BLOCK", "BLOCKS", "FLOOR", "INV_FP8_MAX",
                 "WRITE_CODES", "INDEX64"):
        assert constexpr.get(name), f"{name} must be a constexpr kernel argument"


def test_the_grid_is_bounded_and_the_kernel_strides_the_remainder():
    wrapper = (Path(__file__).resolve().parents[1]
               / "relic_core/kernels/fused_fp8_activation_decode.py").read_text()
    assert "MAX_PROGRAMS = 65535" in wrapper
    assert "min(MAX_PROGRAMS" in wrapper, "the launch grid must be capped"
    kernel = KERNEL_SOURCE.read_text()
    assert "tl.num_programs(0)" in kernel, \
        "a capped grid is only correct if the kernel strides the tiles it did not get"
    assert "for program in range(tl.program_id(0)" in kernel


def test_the_pointer_width_is_decided_before_any_offset_product():
    wrapper = (Path(__file__).resolve().parents[1]
               / "relic_core/kernels/fused_fp8_activation_decode.py").read_text()
    padded = wrapper.index("padded = ")
    launch = wrapper.index("fused_fp8_activation_decode_kernel[")
    assert padded < launch, "the padded reach must be computed before the launch"
    assert "padded >= 2**31" in wrapper
    kernel = KERNEL_SOURCE.read_text()
    # INDEX64 widens the program id before the block index is formed from it.
    assert kernel.index("if INDEX64:") < kernel.index("block_ids = first_block")


def test_the_code_buffers_are_written_only_under_write_codes():
    kernel = next(node for node in kernel_tree().body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "fused_fp8_activation_decode_kernel")
    guarded = [node for node in ast.walk(kernel)
               if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "WRITE_CODES"]
    assert len(guarded) == 1
    stores = [node for node in ast.walk(guarded[0]) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "store"]
    assert len(stores) == 2, "both the code and the scale byte are gated together"


def test_the_decode_consumes_the_produced_byte_pair():
    """The fused decode must read the scale *byte*, not the raw exponent."""
    kernel = next(node for node in kernel_tree().body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "fused_fp8_activation_decode_kernel")
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_e4m3_e8m0_bf16_bits"]
    assert len(calls) == 1
    code_argument, scale_argument = calls[0].args
    assert isinstance(code_argument, ast.Name) and code_argument.id == "codes"
    assert isinstance(scale_argument, ast.Subscript), \
        "the decode takes the per-block scale byte, broadcast over the block"
    assert isinstance(scale_argument.value, ast.Name)
    assert scale_argument.value.id == "scale_bytes"
    # And the non-finite override happens before the decode, not after it.
    source = KERNEL_SOURCE.read_text()
    assert source.index("codes = tl.where(nonfinite") < source.index(
        "bits = _e4m3_e8m0_bf16_bits")