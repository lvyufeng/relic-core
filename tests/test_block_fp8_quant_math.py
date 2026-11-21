"""Exact scalar semantics tests for the fused block-32 FP8 quantizer bit math.

These execute this project's own Triton function bodies with integer scalars --
no Triton, no torch and no accelerator -- and compare them against an independent
oracle written in exact rational arithmetic. Device compilation and hardware
behaviour are separate claims: tests/test_block_fp8_quant.py covers the operator
on a real backend.

The two decisions that must not drift are the ceil-power-of-two block scale and
the E4M3 round-to-nearest-even encode. Both are bit manipulation in the released
kernel, so both are reproduced here by re-executing our own source rather than by
re-deriving them, and the oracle is deliberately a different method: it works on
``Fraction`` values and on the enumerated set of representable E4M3 magnitudes,
so a shared mistake in the shift arithmetic cannot hide a wrong result.
"""
from __future__ import annotations

import ast
import bisect
import math
from fractions import Fraction
from pathlib import Path
import random
import struct
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_fp8_quant_kernels.py"

# The released contract: 32 values per scale, an FP32 amax floored at 1e-4, and
# clamp to the E4M3 maximum. The floor is what bounds the exponent from below.
BLOCK = 32
FP8_MAX = 448.0
FLOOR = struct.unpack("<f", struct.pack("<f", 1e-4))[0]
INV_FP8_MAX = struct.unpack("<f", struct.pack("<f", 1.0 / 448.0))[0]


class Int32:
    """The subset of Triton int32 tensor semantics the helpers rely on."""

    def __init__(self, value):
        self.value = _wrap32(int(value.value if isinstance(value, Int32) else value))

    def to(self, dtype):
        return self

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def __add__(self, other):
        return Int32(self.value + _int(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Int32(self.value - _int(other))

    def __rsub__(self, other):
        return Int32(_int(other) - self.value)

    def __and__(self, other):
        return Int32(self.value & _int(other))

    __rand__ = __and__

    def __or__(self, other):
        return Int32(self.value | _int(other))

    __ror__ = __or__

    def __lshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Int32(self.value << count)

    def __rlshift__(self, other):
        return Int32(_int(other)) << self

    def __rshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Int32(self.value >> count)

    def __eq__(self, other):
        return self.value == _int(other)

    def __ne__(self, other):
        return self.value != _int(other)

    def __lt__(self, other):
        return self.value < _int(other)

    def __le__(self, other):
        return self.value <= _int(other)

    def __gt__(self, other):
        return self.value > _int(other)

    def __ge__(self, other):
        return self.value >= _int(other)

    __hash__ = None


def _wrap32(value):
    return (value + 2**31) % 2**32 - 2**31


def _int(value):
    return int(value.value if isinstance(value, Int32) else value)


def scalar_function(name):
    """Load one ``@triton.jit`` helper from the kernel source as a Python function.

    Every top-level helper is defined in the same namespace, because the encode
    path calls the rounding helper and the point of these tests is to execute the
    real bodies rather than copies. Only the helper under test is ever called, so
    the device-only kernel body is inert here.
    """
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert any(node.name == name for node in functions), f"{name} is missing from {SOURCE.name}"
    for function in functions:
        function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    tl = SimpleNamespace(where=lambda condition, x, y: Int32(x if bool(condition) else y),
                         minimum=lambda x, y: Int32(min(_int(x), _int(y))),
                         maximum=lambda x, y: Int32(max(_int(x), _int(y))),
                         uint8="uint8", int32="int32", float32="float32")
    namespace = {"tl": tl}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


def signed32(bits):
    """The int32 register value of a 32-bit float32 pattern."""
    return _wrap32(bits)


def f32_value(bits):
    """The exact Python float (a subset of double) spelled by a float32 pattern."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    return exponent


def _round_half_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def ceil_log2_exact(value: Fraction) -> int:
    """``ceil(log2(value))`` for a positive value, exactly."""
    exponent = _floor_log2(value)
    return exponent if Fraction(2) ** exponent == value else exponent + 1


def f32_round_bits(value: Fraction) -> int:
    """Round an exact rational to the nearest float32 (ties to even), as bits."""
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    if exponent < -126:
        quantum = Fraction(1, 2**149)
        mantissa = _round_half_even(magnitude / quantum)
        if mantissa == 0:
            return sign
        if mantissa >= 2**23:  # rounded up into the smallest normal
            return sign | (1 << 23)
        return sign | mantissa
    quantum = Fraction(2) ** (exponent - 23)
    mantissa = _round_half_even(magnitude / quantum)
    if mantissa == 2**24:
        exponent += 1
        mantissa = 2**23
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (mantissa - 2**23)


def _scaled_bits(value: float, multiplier) -> int:
    """FP32 bits of ``value * multiplier`` keeping the sign of a zero input.

    ``Fraction(-0.0) == Fraction(0)``, so an exact-rational pipeline would drop
    the sign of a negative zero -- the same sign the released kernel keeps in the
    code byte. The scale side of the product is always positive, so the input's
    sign bit is the product's sign bit.
    """
    magnitude = f32_round_bits(Fraction(abs(value)) * Fraction(multiplier))
    return (magnitude | 0x80000000) if f32_bits(value) >> 31 else magnitude


def _e4m3_magnitude(code: int) -> Fraction:
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 0:
        return Fraction(mantissa, 512)
    return Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)


# Codes 0..126 are the finite non-negative E4M3FN magnitudes, ordered; code 127
# is the only NaN and is therefore outside the finite encode path.
_E4M3_TABLE = tuple((_e4m3_magnitude(code), code) for code in range(127))
_E4M3_VALUES = tuple(value for value, _ in _E4M3_TABLE)


def e4m3_rne(magnitude: Fraction) -> int:
    """Exact E4M3FN round-to-nearest-even of a non-negative magnitude.

    Rounding is expressed as a nearest-neighbour search in the enumerated set of
    representable values, with ties broken towards an even mantissa field, which
    is the IEEE definition and not the shift arithmetic under test.
    """
    if magnitude <= 0:
        return 0
    index = bisect.bisect_left(_E4M3_VALUES, magnitude)
    if index == len(_E4M3_VALUES):
        return 0x7E
    if _E4M3_VALUES[index] == magnitude:
        return _E4M3_TABLE[index][1]
    lower, upper = _E4M3_TABLE[index - 1], _E4M3_TABLE[index]
    below, above = magnitude - lower[0], upper[0] - magnitude
    if below < above:
        return lower[1]
    if above < below:
        return upper[1]
    return lower[1] if (lower[1] & 7) % 2 == 0 else upper[1]


def e4m3_byte(bits: int) -> int:
    """Exact encode of a finite-or-special float32 pattern (saturating)."""
    value = f32_value(bits)
    sign = 0x80 if bits >> 31 else 0
    if math.isnan(value) or math.isinf(value):
        return sign | 0x7E  # the kernel replaces non-finite blocks before encoding
    return sign | e4m3_rne(Fraction(abs(value)))


def test_scale_exponent_is_exact_ceil_log2_for_normal_fp32():
    function = scalar_function("_scale_exponent_from_bits")
    mantissas = (0x000000, 0x000001, 0x000100, 0x400000, 0x7FFFFE, 0x7FFFFF)
    for exponent in range(1, 255):
        for mantissa in mantissas:
            bits = (exponent << 23) | mantissa
            actual = int(function(Int32(signed32(bits))))
            expected = ceil_log2_exact(Fraction(f32_value(bits)))
            assert actual == expected, (hex(bits), actual, expected)


def test_scale_exponent_stays_inside_the_released_floor_and_fp32_range():
    function = scalar_function("_scale_exponent_from_bits")
    amax_values = [FLOOR, 1e-4 * 1.0000001, 1e-3, 0.5, 1.0, 1.5, FP8_MAX,
                   f32_value(0x7F7FFFFF), 2.0**-126, 2.0**6, 2.0**20, 1e30]
    # The released kernel floors amax at 1e-4 before the reciprocal multiply, so
    # even a finite amax far below the floor produces a normal FP32 product: the
    # exponent field is never zero and the reciprocal is always a normal power.
    for amax in amax_values:
        floored = max(amax, FLOOR)
        scaled = f32_round_bits(Fraction(floored) * Fraction(INV_FP8_MAX))
        exponent = int(function(Int32(signed32(scaled))))
        assert exponent == ceil_log2_exact(Fraction(f32_value(scaled)))
        assert -22 <= exponent <= 120, (amax, exponent)
        assert 0 <= exponent + 127 <= 254, "the E8M0 byte must stay a finite power"
        assert 1 <= 127 - exponent <= 149, "the exact reciprocal must be normal"


def test_pow2_bits_are_exact_powers_of_two():
    function = scalar_function("_pow2_bits")
    for exponent in range(-22, 121):
        bits = int(function(Int32(exponent)))
        assert f32_value(bits) == math.ldexp(1.0, exponent), (exponent, hex(bits))


def test_round_half_even_shift_matches_exact_rounding_on_boundaries():
    function = scalar_function("_round_half_even_shift")

    def oracle(value, shift):
        quotient, remainder = divmod(value, 1 << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
        return quotient

    for shift in range(1, 26):
        half = 1 << shift
        neighbours = set()
        for retained in (0, 1, 2, 3, 7, 8, 15, 16, 255):
            base = retained * half
            neighbours.update(base + delta for delta in (-1, 0, 1, half // 2 - 1, half // 2, half // 2 + 1))
        for value in sorted(neighbours):
            if not 0 <= value < 2**24:
                continue
            assert int(function(Int32(value), Int32(shift))) == oracle(value, shift), (value, shift)


def test_round_half_even_shift_matches_exact_rounding_on_random_values():
    function = scalar_function("_round_half_even_shift")
    generator = random.Random(20261117)
    for _ in range(20000):
        shift = generator.randrange(1, 26)
        value = generator.randrange(0, 2**24)
        quotient, remainder = divmod(value, 1 << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
        assert int(function(Int32(value), Int32(shift))) == quotient, (value, shift)


def test_encode_matches_exact_oracle_on_every_representable_magnitude():
    function = scalar_function("_e4m3_byte_from_bits")
    for magnitude, code in _E4M3_TABLE:
        bits = f32_bits(float(magnitude))
        assert Fraction(f32_value(bits)) == magnitude, "the test value must be exact in FP32"
        assert int(function(Int32(signed32(bits)))) == code, (hex(bits), code)


def test_encode_rounds_every_midpoint_to_the_even_mantissa():
    function = scalar_function("_e4m3_byte_from_bits")
    for (lower, low_code), (upper, high_code) in zip(_E4M3_TABLE, _E4M3_TABLE[1:]):
        midpoint = (lower + upper) / 2
        bits = f32_round_bits(midpoint)
        assert Fraction(f32_value(bits)) == midpoint, "midpoints of E4M3 values are exact in FP32"
        expected = low_code if (low_code & 7) % 2 == 0 else high_code
        assert int(function(Int32(signed32(bits)))) == expected, (hex(bits), low_code, high_code)


def test_encode_rounds_each_midpoint_neighbour_away_from_the_tie():
    function = scalar_function("_e4m3_byte_from_bits")
    for (lower, low_code), (upper, high_code) in zip(_E4M3_TABLE, _E4M3_TABLE[1:]):
        bits = f32_round_bits((lower + upper) / 2)
        below, above = bits - 1, bits + 1
        assert int(function(Int32(signed32(below)))) == low_code, hex(below)
        assert int(function(Int32(signed32(above)))) == high_code, hex(above)


def test_encode_matches_exact_oracle_on_sampled_float32_patterns():
    function = scalar_function("_e4m3_byte_from_bits")
    generator = random.Random(448)
    checked = 0
    while checked < 6000:
        bits = generator.getrandbits(32)
        value = f32_value(bits)
        if not math.isfinite(value) or abs(value) > FP8_MAX:
            continue
        assert int(function(Int32(signed32(bits)))) == e4m3_byte(bits), hex(bits)
        checked += 1


def test_encode_matches_exact_oracle_on_fp32_subnormals():
    function = scalar_function("_e4m3_byte_from_bits")
    generator = random.Random(512)
    patterns = list(range(1, 256)) + [generator.randrange(1, 1 << 23) for _ in range(2000)]
    for mantissa in patterns:
        for bits in (mantissa, (1 << 31) | mantissa):
            assert int(function(Int32(signed32(bits)))) == e4m3_byte(bits), hex(bits)


def test_encode_saturates_at_the_e4m3_maximum_and_keeps_every_signed_zero():
    function = scalar_function("_e4m3_byte_from_bits")
    assert int(function(Int32(signed32(f32_bits(FP8_MAX))))) == 0x7E
    assert int(function(Int32(signed32(f32_bits(-FP8_MAX))))) == 0xFE
    for value, expected in ((FP8_MAX * 1.0000001, 0x7E), (1e5, 0x7E), (-1e5, 0xFE)):
        bits = f32_round_bits(Fraction(value))
        assert int(function(Int32(signed32(bits)))) == expected, hex(bits)
    # Positive and negative zero must both survive; the amax floor keeps the
    # scale positive, so a finite block can carry that sign into the code byte.
    for bits, expected in ((0x00000000, 0x00), (0x80000000, 0x80)):
        assert int(function(Int32(signed32(bits)))) == expected
    # The kernel replaces non-finite blocks before encoding; the helper itself is
    # defined as a saturating encode rather than producing the NaN code.
    for bits in (0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001):
        assert int(function(Int32(signed32(bits)))) == e4m3_byte(bits)


def test_nonfinite_test_selects_infinity_and_nan_only():
    function = scalar_function("_nonfinite_bits")
    for bits in (0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0xFFFFFFFF):
        assert int(function(Int32(signed32(bits)))) == 1, hex(bits)
    for bits in (0x00000000, 0x80000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x00800000,
                 0x3F800000, 0x43700000):
        assert int(function(Int32(signed32(bits)))) == 0, hex(bits)


def test_released_block_algorithm_composes_to_the_exact_oracle():
    """Run the kernel's own helpers in the kernel's order and compare end to end.

    The arithmetic is checked in the tests above one function at a time; this one
    checks that composing them the way ``block_fp8_quant_kernel`` does -- ceil
    power-of-two scale from the FP32 reciprocal product, exact reciprocal
    multiply, clamp, encode -- reproduces the released algorithm exactly.
    """
    scale_exponent = scalar_function("_scale_exponent_from_bits")
    pow2 = scalar_function("_pow2_bits")
    encode = scalar_function("_e4m3_byte_from_bits")

    def kernel_block(row):
        amax = max((abs(value) for value in row), default=0.0)
        scaled_bits = f32_round_bits(Fraction(max(amax, FLOOR)) * Fraction(INV_FP8_MAX))
        exponent = int(scale_exponent(Int32(signed32(scaled_bits))))
        inverse = f32_value(int(pow2(Int32(-exponent))))
        assert inverse == math.ldexp(1.0, -exponent), "the reciprocal must be exact"
        codes = []
        for value in row:
            codes.append(int(encode(Int32(signed32(_scaled_bits(value, inverse))))))
        return codes, exponent + 127

    def exact_block(row):
        amax = Fraction(max((abs(value) for value in row), default=0.0))
        scaled = Fraction(f32_value(f32_round_bits(Fraction(max(amax, FLOOR))
                                                  * Fraction(INV_FP8_MAX))))
        exponent = ceil_log2_exact(scaled)
        codes = []
        for value in row:
            product_bits = _scaled_bits(value, Fraction(2) ** -exponent)
            clamped = min(max(f32_value(product_bits), -FP8_MAX), FP8_MAX)
            sign = 0x80 if product_bits >> 31 else 0
            codes.append(sign | e4m3_rne(Fraction(abs(clamped))))
        return codes, exponent + 127

    generator = random.Random(20261117)
    rows = [
        [0.0] * BLOCK,
        [-0.0] * BLOCK,
        [1.0] * BLOCK,
        [FP8_MAX] + [0.0] * (BLOCK - 1),
        [-FP8_MAX] + [0.0] * (BLOCK - 1),
        [2.0**-10] + [0.0] * (BLOCK - 1),
        [1.5 * 2.0**-9] + [0.0] * (BLOCK - 1),
        [1e30] + [-1e-30] * (BLOCK - 1),
        [FLOOR] * BLOCK,
        [2.0**-126] + [0.0] * (BLOCK - 1),
    ]
    for exponent in (-140, -60, -22, -9, -3, 0, 1, 6, 20, 40, 100, 120):
        for _ in range(4):
            rows.append([generator.uniform(-1.0, 1.0) * 2.0**exponent for _ in range(BLOCK)])
    for row in rows:
        # Round the generated values through FP32, as the operator's input would be.
        row = [f32_value(f32_round_bits(Fraction(value))) for value in row]
        assert kernel_block(row) == exact_block(row), row[:2]
        codes, scale_byte = exact_block(row)
        assert scale_byte != 255, "a finite block must never get the NaN scale byte"
        assert all((code & 0x7F) <= 0x7E for code in codes), "0x7F is NaN-only"


def test_encode_never_emits_the_nan_code_for_finite_inputs():
    encode = scalar_function("_e4m3_byte_from_bits")
    generator = random.Random(7)
    for _ in range(5000):
        bits = generator.getrandbits(31)
        value = f32_value(bits)
        if not math.isfinite(value) or abs(value) > FP8_MAX:
            continue
        assert (int(encode(Int32(signed32(bits)))) & 0x7F) <= 0x7E, hex(bits)


def test_kernel_source_has_no_division_or_approximate_math():
    """Guard the two invariants the released kernel implements with bit tricks."""
    tree = ast.parse(SOURCE.read_text())
    forbidden = {"log", "log2", "log10", "exp", "exp2", "frexp", "ldexp", "pow", "sqrt", "ceil"}
    divisions = 0
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            divisions += 1
        if isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = {alias.name.split(".")[0] for alias in node.names}
            assert not imported & forbidden, imported & forbidden
    assert divisions == 0, "the scale and encode paths must be integer bit arithmetic"
    assert not seen & forbidden, seen & forbidden
    bitcasts = sum(1 for node in ast.walk(tree)
                   if isinstance(node, ast.keyword) and node.arg == "bitcast")
    assert bitcasts >= 3, "the FP32 bit patterns must be reinterpreted, not converted"