"""Exact scalar semantics tests for the fused block-wise FP4 quantizer bit math.

These execute this project's own Triton function bodies -- the same bodies the
Ascend kernel compiles -- on plain Python integer and float scalars: no Triton, no
torch, no accelerator. They are what makes the device kernel trustworthy before it
has ever run on a device, and the oracle they are compared against is written in
exact rational arithmetic and an enumerated codebook, so no rounding decision is
shared between the two sides.

The three decisions that must not drift, because the released kernel makes all
three with IEEE bit manipulation rather than with a math library:

* the **E8M0** scale is ``2**ceil(log2(amax * fp32(1/6)))``. The released helper is
  named ``fast_round_scale``, but it ceils the exponent field; a nearest rounding
  of the logarithm would put every scale in the wrong power of two half the time.
* the **E4M3** scale is ``fp32(e4m3(amax / 6.0))``, i.e. a rounding of the ratio,
  and the normalizing division then uses that *rounded* scale.
* the E2M1 encode rounds to nearest with ties to the even code, and the two zero
  codes keep the sign of the value.

Hardware compilation and real-device behaviour are separate claims:
``tests/test_block_fp4_quant.py`` covers the operator on a real backend.
"""
from __future__ import annotations

import ast
import math
import struct
from fractions import Fraction
from pathlib import Path
import random

SOURCES = (
    Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_fp8_quant_kernels.py",
    Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_fp4_quant_kernels.py",
)

#: The released contract, as constants independent of the modules under test.
FP4_MAX = 6.0
INV_FP4_MAX = struct.unpack("<f", struct.pack("<f", 1.0 / 6.0))[0]
FLOORS = {"e8m0": 6.0 * 2.0**-126, "e4m3": 6.0 * 2.0**-9}
E8M0_NAN = 255
E4M3_NAN = 0x7F
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# --- scalar semantics for the executed Triton bodies --------------------------

def _wrap32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


class Int:
    """The Triton ``int32`` scalar subset the helpers use."""

    def __init__(self, value):
        self.value = _wrap32(int(value.value if isinstance(value, Int) else value))

    def to(self, dtype, bitcast=False):
        if bitcast:
            assert dtype is tl.float32, "only the int32/float32 bitcast is used"
            return Float(_value(self.value))
        return Float(float(self.value)) if dtype is tl.float32 else self

    def __int__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    def __add__(self, other):
        return Int(self.value + _int(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Int(self.value - _int(other))

    def __rsub__(self, other):
        return Int(_int(other) - self.value)

    def __and__(self, other):
        return Int(self.value & _int(other))

    __rand__ = __and__

    def __or__(self, other):
        return Int(self.value | _int(other))

    __ror__ = __or__

    def __lshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Int(self.value << count)

    def __rlshift__(self, other):
        return Int(_int(other) << self.value)

    def __rshift__(self, other):
        count = _int(other)
        assert 0 <= count < 32, "undefined shift in a selected or unselected branch"
        return Int(self.value >> count)

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


class Float:
    """The Triton ``float32`` scalar subset the helpers use."""

    def __init__(self, value):
        self.value = float(value.value if isinstance(value, Float) else value)

    def to(self, dtype, bitcast=False):
        if bitcast:
            assert dtype is tl.int32, "only the int32/float32 bitcast is used"
            return Int(_bits(self.value))
        assert dtype is tl.float32
        return self

    def __float__(self):
        return self.value

    def __add__(self, other):
        return Float(self.value + _float(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Float(self.value - _float(other))

    def __rsub__(self, other):
        return Float(_float(other) - self.value)

    def __mul__(self, other):
        return Float(self.value * _float(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Float(_f32(self.value / _float(other)))

    def __neg__(self):
        return Float(-self.value)

    def __lt__(self, other):
        return self.value < _float(other)

    def __le__(self, other):
        return self.value <= _float(other)

    def __gt__(self, other):
        return self.value > _float(other)

    def __ge__(self, other):
        return self.value >= _float(other)


def _int(value) -> int:
    return value.value if isinstance(value, Int) else int(value)


def _float(value) -> float:
    return value.value if isinstance(value, Float) else float(value)


def _f32(value: float) -> float:
    """Round a Python float to float32, staying a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _coerce(result, *operands):
    """Give a bare Python result the wrapper type the expression itself carries.

    ``tl.where(cond, -9, exponent - 10)`` mixes a Python literal with an int32
    value, exactly as the Triton source does; the scalar semantics have to return
    an ``Int`` there, because the next call is ``.to(...)``.
    """
    if isinstance(result, (Int, Float)):
        return result
    for operand in operands:
        if isinstance(operand, Float):
            return Float(result)
        if isinstance(operand, Int):
            return Int(result)
    return result


class _TritonShim:
    """The handful of ``tl`` names the executed helpers reference."""

    float32 = "float32"
    int32 = "int32"
    uint16 = "uint16"
    bfloat16 = "bfloat16"

    @staticmethod
    def constexpr(value):
        """A constexpr global is just its value once the front end unwraps it."""
        return value

    @staticmethod
    def cast(value, dtype, bitcast=False):
        """Only the 16-bit unsigned-to-BF16 bitcast is used here.

        A BF16 scalar has no natural Python representation, so the harness carries it
        as its 16-bit pattern: the helper under test only builds patterns, and the
        assertions are about patterns.
        """
        assert bitcast and dtype is _TritonShim.bfloat16
        return Int(value)

    @staticmethod
    def where(condition, first, second):
        return _coerce(first if bool(condition) else second, first, second)

    @staticmethod
    def maximum(first, second):
        return _coerce(first if _compare(first, second, ">=") else second, first, second)

    @staticmethod
    def minimum(first, second):
        return _coerce(first if _compare(first, second, "<=") else second, first, second)

    @staticmethod
    def max(value, axis=None):
        assert axis == 1, "only the block reduction over axis 1 is used"
        return value

    @staticmethod
    def load(*args, **kwargs):  # pragma: no cover - kernels are not called here
        raise AssertionError("the math tests call scalar helpers, not kernels")

    store = load
    program_id = load
    arange = load


def _compare(first, second, operator: str) -> bool:
    """Compare two scalar-semantics values without narrowing either of them.

    Truncating a float floor to ``int`` here would have turned the E8M0 floor
    (``6 * 2**-126``) into zero and silently disarmed it.
    """
    left = first.value if isinstance(first, (Int, Float)) else first
    right = second.value if isinstance(second, (Int, Float)) else second
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == "<":
        return left < right
    if operator == ">":
        return left > right
    raise AssertionError(operator)


tl = _TritonShim()


def _constant_value(node):
    """Evaluate the tiny literal grammar the kernel module uses for constants.

    Deliberately whitelisted rather than ``eval``-ed: a literal, ``float("...")``,
    and a ``tl.constexpr(...)`` wrapper around one of those. Anything else is not a
    constant and is skipped, so the harness never executes module code.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Call) and len(node.args) == 1:
        function = node.func
        if isinstance(function, ast.Name) and function.id == "float":
            # ``float("nan")`` passes a string literal, so this branch converts its
            # argument directly instead of going through the numeric path.
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, (str, int, float)):
                return float(argument.value)
            return float(_constant_value(argument))
        if isinstance(function, ast.Attribute) and function.attr == "constexpr":
            return _constant_value(node.args[0])
    raise ValueError("not a scalar constant")


def _executed(namespace: dict) -> dict:
    """Execute the module-level integer constants and the function defs of both files.

    Imports, kernel bodies' decorators and everything else are skipped: the helpers
    are pure scalar expressions, and executing their real bodies is the point.
    """
    for source in SOURCES:
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                # Module-level scalar constants are part of the helper contract: the
                # midpoint bit patterns, the E4M3 maximum and the NaN value. They are
                # declared ``tl.constexpr(...)`` because the Triton front end refuses to
                # read a plain global from a kernel body, so the collector has to see
                # through that annotation, and through ``float("nan")``.
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        value = _constant_value(node.value)
                    except ValueError:
                        continue
                    namespace[node.targets[0].id] = value
            elif isinstance(node, ast.FunctionDef) and node.decorator_list:
                node.decorator_list = []
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(ast.fix_missing_locations(module), str(source), "exec"),
                     namespace)
    return namespace


NAMESPACE = _executed({"tl": tl, "float": float})


def _code(value: float) -> int:
    """Run the kernel's own E2M1 encoder on a float32 value."""
    return int(NAMESPACE["_e2m1_code"](Int(_bits(_f32(value)))))


def _code_value(code: int) -> float:
    """Run the kernel's own E2M1 decoder on a code."""
    return float(NAMESPACE["_e2m1_value"](Int(code)))


def _scale(amax: float, scale_format: str) -> tuple[int, float]:
    """Run the kernel's own block-scale decision on a float32 amax."""
    floor = FLOORS[scale_format]
    byte, value = NAMESPACE["_block_scale"](
        Float(_f32(amax)), floor, INV_FP4_MAX, scale_format == "e4m3")
    return int(byte), float(value)


def _e4m3_value(byte: int) -> float:
    return float(NAMESPACE["_e4m3_byte_value"](Int(byte)))


def _pattern(value: int) -> int:
    """A bit pattern as an unsigned 32-bit number.

    The helpers work on signed int32 tensors, so the sign bit comes back as a
    negative Python int; comparing patterns rather than signed values is what these
    assertions are about.
    """
    return value & 0xFFFFFFFF


def _zero_sign_f32(code: int, raw_bits: int) -> int:
    """Run the kernel's FP32 zero-sign helper on one code, returning the pattern."""
    return _pattern(int(NAMESPACE["_with_zero_sign_f32"](Int(code), Int(raw_bits))))


def _signed_zero_bf16(code: int) -> int:
    """Run the kernel's BF16 signed-zero builder on one code, returning the pattern."""
    return _pattern(int(NAMESPACE["_signed_zero_bf16"](Int(code))))


# --- exact oracle -------------------------------------------------------------

def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    return exponent


def _ceil_log2(value: Fraction) -> int:
    exponent = _floor_log2(value)
    return exponent if Fraction(2) ** exponent == value else exponent + 1


def _round_half_even(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _f32_round(value: Fraction) -> float:
    """The nearest float32 to an exact rational, ties to even."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    if exponent > 127:
        return math.copysign(math.inf, sign)
    if exponent >= -126:
        step = Fraction(2) ** (exponent - 23)
        significand = _round_half_even(magnitude / step)
        if significand == 2**24:
            significand //= 2
            exponent += 1
            if exponent > 127:
                return math.copysign(math.inf, sign)
        bits = (exponent + 127) << 23 | (significand - 2**23)
        return sign * _value(bits)
    significand = _round_half_even(magnitude / Fraction(2) ** -149)
    if significand >= 2**23:
        return sign * _value(0x00800000)
    return sign * _value(significand)


def _e4m3_round(value: Fraction) -> int:
    """The E4M3 byte an exact rational rounds to (RNE, saturating at 448)."""
    sign = 0x80 if value < 0 else 0
    magnitude = min(abs(value), Fraction(448))
    if magnitude == 0:
        return sign
    exponent = _floor_log2(magnitude)
    if exponent >= -6:
        step = Fraction(2) ** (exponent - 3)
        significand = _round_half_even(magnitude / step)
        if significand == 16:
            significand //= 2
            exponent += 1
        return sign | ((exponent + 7) << 3) | (significand - 8)
    significand = _round_half_even(magnitude / Fraction(2) ** -9)
    if significand >= 8:
        return sign | (1 << 3)
    return sign | significand


def _e4m3_value_oracle(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    if (byte & 0x7F) == 0x7F:
        return math.nan
    if exponent == 0:
        return sign * math.ldexp(float(mantissa), -9)
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def _e2m1_index(value: Fraction) -> int:
    """Magnitude code 0..7 an exact rational rounds to, ties to the even code."""
    if value <= 0:
        return 0
    if value > Fraction(6):
        value = Fraction(6)
    upper = 1
    while upper + 1 < len(MAGNITUDES) and Fraction(MAGNITUDES[upper]) < value:
        upper += 1
    if Fraction(MAGNITUDES[upper]) < value:
        return len(MAGNITUDES) - 1
    below = Fraction(MAGNITUDES[upper - 1])
    above = Fraction(MAGNITUDES[upper])
    if value - below < above - value:
        return upper - 1
    if value - below > above - value:
        return upper
    return upper if upper % 2 == 0 else upper - 1


def _oracle_scale(amax: float, scale_format: str) -> tuple[int, float]:
    """The released scale decision for one block amax, in exact arithmetic."""
    floored = max(_f32(amax), FLOORS[scale_format])
    if scale_format == "e8m0":
        product = _f32_round(Fraction(floored) * Fraction(INV_FP4_MAX))
        exponent = _ceil_log2(Fraction(product))
        return exponent + 127, math.ldexp(1.0, exponent)
    ratio = _f32_round(Fraction(floored) / Fraction(FP4_MAX))
    byte = _e4m3_round(Fraction(ratio))
    return byte, _e4m3_value_oracle(byte)


def _oracle_code(value: float, scale: float) -> int:
    """The released E2M1 code of one value under a given scale, ties to even."""
    normalized = _f32_round(Fraction(_f32(value)) / Fraction(scale))
    normalized = min(max(normalized, -FP4_MAX), FP4_MAX)
    index = _e2m1_index(Fraction(abs(normalized)))
    # A rational has no signed zero, and the scale is always positive here, so the
    # sign is the input's: IEEE division preserves the sign of a zero.
    return index + (8 if math.copysign(1.0, value) < 0 else 0)


# --- the encoder --------------------------------------------------------------

def test_every_codebook_value_encodes_to_its_own_code():
    for index, magnitude in enumerate(MAGNITUDES):
        assert _code(magnitude) == index, magnitude
        assert _code(-magnitude) == index + 8, magnitude


def test_ties_go_to_the_even_code_from_both_sides():
    for midpoint in MIDPOINTS:
        above = _value(_bits(midpoint) + 1)
        below = _value(_bits(midpoint) - 1)
        exact = _code(midpoint)
        assert exact == _oracle_code(midpoint, 1.0), midpoint
        assert exact % 2 == 0, f"the tie at {midpoint} must pick the even code"
        # A tie is between two *adjacent* codes, and the even one was taken. One
        # ulp above must be the upper neighbour and one ulp below the lower one --
        # unless the even code already was that neighbour, in which case it stays.
        assert _code(above) == _oracle_code(above, 1.0), midpoint
        assert _code(below) == _oracle_code(below, 1.0), midpoint
        assert _code(above) - _code(below) == 1, f"{midpoint} must sit between two codes"
        assert exact in (_code(above), _code(below)), midpoint
        assert _code(below) <= exact <= _code(above), midpoint
        # The negation must be symmetric: same magnitude, sign bit added.
        assert _code(-midpoint) == exact + 8


def test_one_ulp_around_every_midpoint_matches_the_exact_oracle():
    for midpoint in MIDPOINTS:
        base = _bits(midpoint)
        for delta in range(-4, 5):
            value = _value(base + delta)
            assert _code(value) == _oracle_code(value, 1.0), (midpoint, delta)


def test_random_fp32_magnitudes_in_range_match_the_exact_oracle():
    generator = random.Random(20261117)
    for _ in range(4000):
        value = generator.uniform(0.0, 6.0)
        value = _f32(value)
        assert _code(value) == _oracle_code(value, 1.0), value
        assert _code(-value) == _oracle_code(-value, 1.0), value


def test_every_exponent_boundary_across_the_range_is_exact():
    """Sweep the FP32 patterns around each binade edge, one and two ulps apart."""
    for exponent in range(-128, 4):
        power = math.ldexp(1.0, exponent)
        if not math.isfinite(power) or power > 6.0:
            continue
        for delta in (-2, -1, 0, 1, 2):
            value = power if delta == 0 else _value(max(_bits(power) + delta, 0))
            if value > 6.0 or not math.isfinite(value):
                continue
            assert _code(value) == _oracle_code(value, 1.0), value


def test_negative_zero_keeps_its_sign_and_never_becomes_positive_zero():
    assert _code(0.0) == 0
    assert _code(-0.0) == 8
    assert math.copysign(1.0, _code_value(0)) > 0
    assert math.copysign(1.0, _code_value(8)) < 0
    assert _code_value(8) == 0.0


def test_kernel_decoder_returns_the_codebook_values():
    for code in range(16):
        expected = MAGNITUDES[code & 7] * (-1.0 if code & 8 else 1.0)
        actual = _code_value(code)
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_zero_sign_helper_restores_the_signed_zero_of_a_zero_code():
    """The backend cannot be trusted to compute a zero's sign; this pins the fix.

    The device run showed the FP32 inplace result ``0`` where the reference gives
    ``0x8000`` in BF16 for code 8, in both scale formats, while every non-zero value
    matched exactly. The sign is therefore written into the bits from ``codes & 8``.
    """
    # Only codes 0 and 8 are zero magnitudes; 9..15 are negative non-zero values and
    # must not be touched (the next test covers that).
    assert _zero_sign_f32(0, 0) == 0x00000000
    assert _zero_sign_f32(8, 0) == 0x80000000
    # OR-ing the sign bit into an already negative zero must not change it.
    assert _zero_sign_f32(0, 0x80000000) == 0x80000000
    assert _zero_sign_f32(8, 0x80000000) == 0x80000000


def test_zero_sign_helper_leaves_non_zero_magnitudes_alone():
    """Only zero codes are rewritten, so no value can be corrupted by this fix."""
    patterns = (0x00000000, 0x80000000, 0x3FC00000, 0xBFC00000,
                0x00000001, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF)
    for code in range(16):
        for pattern in patterns:
            if code & 7:
                assert _zero_sign_f32(code, pattern) == pattern, (code, pattern)


def test_zero_sign_helper_never_rewrites_a_nan_or_infinity():
    """A non-finite block keeps its NaN instead of receiving a zero's sign.

    A non-finite block quantizes to zero codes and a NaN scale, so it satisfies the
    zero-magnitude test; the exponent check is what keeps it from being turned into
    a signed zero. This is the case the lead called out.
    """
    non_finite = (0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000, 0x7F800001)
    for code in range(16):
        for pattern in non_finite:
            assert _zero_sign_f32(code, pattern) == pattern, (code, pattern)
    # The BF16 stage selects an explicit zero instead of rewriting bits, and it is
    # gated by the same exponent test on the FP32 result, checked above; here the
    # builder itself must still produce the right pattern for the zero codes only.
    assert _signed_zero_bf16(0) == 0x0000
    assert _signed_zero_bf16(8) == 0x8000


def test_bf16_signed_zero_builder_uses_the_bf16_sign_bit():
    """The builder produces exactly the 16-bit BF16 patterns for +-0."""
    for code in range(16):
        expected = 0x8000 if code & 8 else 0x0000
        assert _signed_zero_bf16(code) == expected, code
    # Only the codes', sign matters: no magnitude bits may leak into the pattern.
    for code in (0, 7, 8, 15):
        assert _signed_zero_bf16(code) in (0x0000, 0x8000), code


def test_zero_sign_helpers_exist_in_the_kernel_module():
    """Guard the wiring: both helpers must be the module's own functions."""
    for name in ("_with_zero_sign_f32", "_signed_zero_bf16"):
        assert name in NAMESPACE, name


# --- the scale decisions ------------------------------------------------------

def test_e8m0_scale_is_the_ceiling_power_of_two_not_a_rounded_log():
    """A nearest rounding of the logarithm would disagree on most of these."""
    disagreements = 0
    for value in (1.0, 1.5, 2.0, 3.0, 6.0, 7.0, 8.0, 0.5, 0.75, 0.9, 1.1, 5.9,
                  2.0000001, 100.0, 1000.0, 2.0**20, 1e-3, 6.0 * 2.0**-126,
                  6.0 * 2.0**-9, 1e-38):
        byte, scale = _scale(value, "e8m0")
        expected_byte, expected_scale = _oracle_scale(value, "e8m0")
        assert byte == expected_byte, value
        assert scale == expected_scale, value
        # Above the floor the selected scale always covers the block, and never by
        # more than a factor of two: that is exactly what a ceiling of the log does.
        if value >= FLOORS["e8m0"]:
            ratio = Fraction(value) / Fraction(FP4_MAX)
            assert expected_scale >= ratio
            assert Fraction(expected_scale) < 2 * ratio
        if value > 0 and math.log2(Fraction(value) / Fraction(FP4_MAX)) % 1:
            # Where the logarithm is not an integer, ceil and round differ for
            # every mantissa whose fractional part is below one half.
            if (math.log2(Fraction(value) / Fraction(FP4_MAX))) % 1 > 0.5:
                disagreements += 1
    assert disagreements > 0, "the sample must contain a case that separates ceil from round"


def test_e8m0_scale_sweeps_every_octave_of_the_amax_range():
    generator = random.Random(4)
    exponent = -126
    while exponent <= 120:
        for _ in range(3):
            amax = math.ldexp(generator.uniform(1.0, 2.0), exponent)
            amax = _f32(amax)
            assert _scale(amax, "e8m0") == _oracle_scale(amax, "e8m0"), amax
        exponent += 7


def test_e8m0_floor_keeps_a_zero_block_at_the_smallest_power_of_two():
    byte, scale = _scale(0.0, "e8m0")
    assert byte == 1 and scale == 2.0**-126
    assert _scale(FLOORS["e8m0"], "e8m0") == (1, 2.0**-126)
    # Just below one more octave the ceiling still lands on the same scale.
    assert _scale(2.0**-126 / Fraction(6, 5), "e8m0")[1] >= 2.0**-126


def test_e4m3_scale_is_the_rounded_ratio_and_saturates_at_448():
    for value in (0.0, 1e-8, 0.25, 1.0, 6.0, 2688.0, 2689.0, 1e10, 3.4e38):
        byte, scale = _scale(value, "e4m3")
        expected_byte, expected_scale = _oracle_scale(value, "e4m3")
        assert byte == expected_byte, value
        if math.isnan(expected_scale):
            assert math.isnan(scale)
        else:
            assert scale == expected_scale, value
    assert _scale(0.0, "e4m3") == (1, 2.0**-9), "the all-zero block keeps a nonzero scale"
    assert _scale(3.4e38, "e4m3")[0] == 0x7E, "an out-of-range ratio saturates, not NaN"


def test_e4m3_scale_saturates_for_the_amax_range_that_overflows_the_ratio():
    """The encode input must be clamped, not passed through as a 480-ish ratio.

    The shared E4M3 encoder's contract is an input already within the format; a raw
    ratio in ``[448, 512)`` encodes to the NaN byte instead of saturating. These are
    the amax values where an unclamped ratio would land exactly there, so this test
    fails by name if the clamp is ever dropped.
    """
    for amax in (2688.0, 2784.0, 2816.0, 2880.0, 3000.0, 10000.0):
        byte, scale = _scale(amax, "e4m3")
        assert byte == 0x7E, f"amax {amax} must saturate the scale to 0x7E"
        assert scale == 448.0, amax
        assert byte != E4M3_NAN, "a saturated scale is not the NaN byte"
    # Just below the boundary the ratio is still inside the format and rounds up.
    assert _scale(2688.0, "e4m3") == (0x7E, 448.0)
    assert _scale(2687.0, "e4m3")[0] < 0x7E or _scale(2687.0, "e4m3")[1] == 448.0


def test_composed_block_with_a_saturating_amax_keeps_the_clamped_codes():
    """A mixed block whose amax saturates the scale still encodes every lane."""
    values = [2880.0, 448.0, 224.0, 0.01171875, -2880.0, -1.0, 0.0, -0.0] * 4
    values = [_f32(value) for value in values]
    codes, byte, scale = _kernel_block(values, "e4m3")
    assert byte == 0x7E and scale == 448.0
    expected_byte, expected_scale = _oracle_scale(max(abs(v) for v in values), "e4m3")
    assert (byte, scale) == (expected_byte, expected_scale)
    for value, code in zip(values, codes):
        assert code == _oracle_code(value, expected_scale), (value, code)
    # 2880/448 clamps to 6.0 -> the largest code; 448/448 == 1.0 -> code 2;
    # 224/448 == 0.5 -> code 1; the small lane rounds to a signed zero.
    assert codes[0] == 7 and codes[1] == 2 and codes[2] == 1
    assert codes[3] == 0 and codes[4] == 15


def test_e4m3_scale_decoder_matches_the_oracle_for_every_byte():
    for byte in range(256):
        actual = _e4m3_value(byte)
        expected = _e4m3_value_oracle(byte)
        if math.isnan(expected):
            assert math.isnan(actual), byte
        else:
            assert actual == expected, byte
    assert math.isnan(_e4m3_value(E4M3_NAN))
    assert math.isnan(_e4m3_value(0xFF))
    # The kernel substitutes the quiet-NaN pattern rather than a float NaN constant,
    # so pin the pattern itself: a signalling or differently-shaped NaN would still
    # be "a NaN" but would not be the bit pattern this operator documents.
    assert _bits(_e4m3_value(E4M3_NAN)) == 0x7FC00000
    assert _bits(_e4m3_value(0xFF)) == 0xFFC00000, "a NaN byte keeps its sign bit"


# --- the composed operator ----------------------------------------------------

def _kernel_block(values: list[float], scale_format: str) -> tuple[list[int], int, float]:
    """Compose the kernel's own helpers the way its kernel body does."""
    amax = max(abs(value) for value in values)
    byte, scale = _scale(amax, scale_format)
    codes = []
    for value in values:
        normalized = value / scale if scale_format == "e4m3" else value * (1.0 / scale)
        normalized = min(max(normalized, -FP4_MAX), FP4_MAX)
        codes.append(_code(normalized))
    return codes, byte, scale


def test_composed_blocks_match_the_exact_oracle():
    """Random blocks, including ones whose values sit exactly on a tie."""
    generator = random.Random(11)
    for scale_format in ("e8m0", "e4m3"):
        for _ in range(200):
            values = [generator.uniform(-8.0, 8.0) for _ in range(32)]
            values = [_f32(value) for value in values]
            codes, byte, _ = _kernel_block(values, scale_format)
            expected_byte, expected_scale = _oracle_scale(max(abs(v) for v in values),
                                                          scale_format)
            assert byte == expected_byte, values
            for value, code in zip(values, codes):
                assert code == _oracle_code(value, expected_scale), (value, code)


def test_composed_blocks_match_on_the_exact_tie_values():
    """Every midpoint and codebook value, with an amax that fixes the scale at 1."""
    for scale_format in ("e8m0", "e4m3"):
        for value in MAGNITUDES + MIDPOINTS + tuple(-m for m in MAGNITUDES + MIDPOINTS):
            block = [6.0] + [_f32(value)] * 31
            codes, byte, scale = _kernel_block(block, scale_format)
            expected_byte, expected_scale = _oracle_scale(6.0, scale_format)
            assert byte == expected_byte, value
            assert codes[0] == _oracle_code(6.0, expected_scale)
            assert codes[1] == _oracle_code(_f32(value), expected_scale), (scale_format, value)


def test_negative_zero_survives_the_composed_operator():
    for scale_format in ("e8m0", "e4m3"):
        codes, _, _ = _kernel_block([-0.0] * 32, scale_format)
        assert codes == [8] * 32, f"{scale_format} lost the sign of zero"
        codes, _, _ = _kernel_block([0.0] * 32, scale_format)
        assert codes == [0] * 32


def _is_jit(node) -> bool:
    """Whether a function definition carries a ``*jit`` decorator."""
    return any(getattr(dec, "attr", getattr(dec, "id", "")) == "jit"
               for dec in node.decorator_list)


def _free_names(function):
    """Names a function body reads that it does not bind itself.

    The decorator list is excluded: ``@triton.jit`` reads ``triton``, which is the
    decoration itself and not something the kernel body needs at trace time.
    """
    bound = {argument.arg for argument in function.args.args}
    bound |= {argument.arg for argument in function.args.kwonlyargs}
    nodes = [node for statement in function.body for node in ast.walk(statement)]
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
    loaded = {node.id for node in nodes
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return loaded - bound


def _module_symbols():
    """Every kernel module's jit functions, constants, and imports, by source."""
    tables = {}
    for source in SOURCES:
        tree = ast.parse(source.read_text())
        table = {"jit": set(), "annotated": set(), "plain": set(), "imported": set()}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _is_jit(node):
                table["jit"].add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                # The other accepted spelling: ``_X: tl.constexpr = 5``. It produces
                # the same constexpr object, so the guard must not treat it as plain.
                annotation = node.annotation
                declared = getattr(annotation, "attr", getattr(annotation, "id", ""))
                if declared == "constexpr":
                    table["annotated"].add(node.target.id)
                elif isinstance(node.value, ast.Constant):
                    table["plain"].add(node.target.id)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
                if not isinstance(target, ast.Name):
                    continue
                if (isinstance(value, ast.Constant)
                        and isinstance(value.value, (int, float))):
                    table["plain"].add(target.id)
                elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                      and value.func.attr == "constexpr"):
                    table["annotated"].add(target.id)
            elif isinstance(node, ast.ImportFrom):
                table["imported"] |= {alias.asname or alias.name for alias in node.names}
        tables[source.name] = table
    return tables


def test_every_global_a_kernel_body_reads_is_constexpr_or_a_jit_function():
    """The Triton front end refuses a plain module global inside ``@jit``.

    It fails at compile time with "cannot access global variable ... within @jit",
    which is exactly how the first device run failed, and the escape hatch that
    would relax the rule is deliberately unused. This test walks the real sources,
    so adding a plain module constant to a kernel body fails here -- on a CPU, with
    no Triton installed -- instead of on a leased accelerator.
    """
    builtins = {"tl", "float", "int", "bool", "min", "max", "len", "range"}
    tables = _module_symbols()
    every_jit = set().union(*(table["jit"] for table in tables.values()))
    allowed = every_jit | builtins
    for source in SOURCES:
        table = tables[source.name]
        readable = allowed | table["annotated"] | table["imported"]
        for node in ast.parse(source.read_text()).body:
            if not (isinstance(node, ast.FunctionDef) and _is_jit(node)):
                continue
            for name in _free_names(node):
                assert name not in table["plain"], (
                    f"{source.name}:{node.name} reads the plain module global {name!r}; "
                    "annotate it tl.constexpr or pass it in")
                assert name in readable, (
                    f"{source.name}:{node.name} reads the unknown global {name!r}")


def test_every_constexpr_global_is_stable_across_recompiles():
    """A cached global must compare equal to itself, or the second compile fails.

    Triton's JIT stores a compiled kernel's globals and re-checks them when the same
    kernel is asked for again; a value that does not compare equal to itself reads
    as "changed since the compiled kernel" and the call raises. ``nan`` is exactly
    such a value, which is why the two NaN sites in this operator select the integer
    pattern ``0x7FC00000`` and bitcast once instead of holding a float NaN constant.
    This test walks the declared constants, so reintroducing a NaN (or any other
    self-unequal) constexpr global fails here, on a CPU.
    """
    for source in SOURCES:
        for node in ast.parse(source.read_text()).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            value = node.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "constexpr"):
                continue
            try:
                constant = _constant_value(value)
            except ValueError:
                continue
            name = node.targets[0].id
            assert constant == constant, (
                f"{source.name} declares {name} as a constexpr whose value does not "
                "compare equal to itself; that breaks the JIT's cached-global check. "
                "Use an integer bit pattern and a single bitcast instead")
            assert isinstance(constant, (int, float)), (source.name, name, constant)


def test_no_kernel_module_declares_a_plain_scalar_global():
    """Any scalar global in these modules must carry the annotation upfront.

    Kept separate from the per-body check so an unused plain constant -- which the
    front end would reject the moment some future kernel body reads it -- is caught
    even while nothing reads it yet.
    """
    for source in SOURCES:
        table = _module_symbols()[source.name]
        assert not table["plain"], (
            f"{source.name} declares plain scalar globals {sorted(table['plain'])}; "
            "Triton only accepts tl.constexpr globals inside @jit")


def test_the_executed_bodies_are_the_ones_the_package_ships():
    """The executed bodies must come from the package sources, not a copy.

    Checked by AST rather than by importing: this file runs with no Torch and no
    Triton, and importing the kernel module pulls Triton in.
    """
    expected = {
        "_ascend_fp8_quant_kernels.py": {"_nonfinite_bits", "_scale_exponent_from_bits",
                                        "_pow2_bits", "_round_half_even_shift",
                                        "_e4m3_byte_from_bits"},
        "_ascend_fp4_quant_kernels.py": {"_e2m1_code", "_e2m1_value", "_block_scale",
                                        "_e4m3_byte_value", "_block_quantize"},
    }
    for source in SOURCES:
        names = {node.name for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
        assert expected[source.name] <= names, (source.name, expected[source.name] - names)
        assert expected[source.name] <= set(NAMESPACE), source.name