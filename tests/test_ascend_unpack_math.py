"""Exhaustive scalar semantics tests for the fused unpack bit expressions.

These execute the project's own small Triton function bodies with integer scalar
operations, not Triton or an NPU. Device compilation and hardware correctness
must be validated separately by test_online_unpack.py with an explicit backend.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import struct
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_unpack_kernels.py"


class Scalar:
    def __init__(self, value):
        self.value = int(value.value if isinstance(value, Scalar) else value)

    def to(self, dtype):
        return Scalar(self.value & 0xFFFF if dtype == "uint16" else self.value)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def __add__(self, other):
        return Scalar(self.value + int(Scalar(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return Scalar(self.value - int(Scalar(other)))

    def __rsub__(self, other):
        return Scalar(int(Scalar(other)) - self.value)

    def __neg__(self):
        return Scalar(-self.value)

    def __and__(self, other):
        return Scalar(self.value & int(Scalar(other)))

    def __or__(self, other):
        return Scalar(self.value | int(Scalar(other)))

    def __lshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Scalar(self.value << count)

    def __rlshift__(self, other):
        return Scalar(other) << self

    def __rshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
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


def scalar_function(name):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    tl = SimpleNamespace(where=lambda c, x, y: Scalar(x if bool(c) else y),
                         maximum=lambda x, y: Scalar(max(int(Scalar(x)), int(Scalar(y)))),
                         minimum=lambda x, y: Scalar(min(int(Scalar(x)), int(Scalar(y)))),
                         uint16="uint16", int32="int32")
    namespace = {"tl": tl}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


def fp8_value(code):
    magnitude = code & 127
    if magnitude == 127:
        return math.nan
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(float(8 + mantissa), exponent - 10)
    return -value if code & 128 else value


def fp32_to_bf16_bits(value):
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        bits = 0xFF800000 if value < 0 else 0x7F800000
    if (bits & 0x7F800000) == 0x7F800000:
        return (bits >> 16) | (0x40 if bits & 0x7FFFFF else 0)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def test_all_fp4_codes_are_exact_integer_widening():
    function = scalar_function("_e2m1_x2")
    expected = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
    assert [int(function(Scalar(code))) for code in range(16)] == list(expected)


def test_all_e4m3_e8m0_pairs_match_independent_fp32_bf16_oracle():
    function = scalar_function("_e4m3_e8m0_bf16_bits")
    for code in range(256):
        for scale in range(256):
            actual = int(function(Scalar(code), Scalar(scale)))
            if (code & 127) == 127 or scale == 255:
                assert (actual & 0x7F80) == 0x7F80 and (actual & 0x7F) != 0, (code, scale, actual)
            else:
                expected = fp32_to_bf16_bits(fp8_value(code) * math.ldexp(1.0, scale - 127))
                assert actual == expected, (code, scale, hex(actual), hex(expected))


def test_zero_sign_is_retained_for_every_finite_scale():
    function = scalar_function("_e4m3_e8m0_bf16_bits")
    for scale in range(255):
        assert int(function(Scalar(0), Scalar(scale))) == 0
        assert int(function(Scalar(128), Scalar(scale))) == 0x8000
