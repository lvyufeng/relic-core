"""The constant-shift fused FP8 activation pass, checked without an accelerator.

The claim under test is exact and narrow: the specialized kernel must emit the **same
E4M3 code bytes**, the **same E8M0 scale bytes** and the **same decoded BF16 bits** as
``quantize_block_fp8`` followed by ``unpack_fp8_to_bf16(block_shape=(1, 32))``, while
replacing the two lane-dependent shift branches with constant ladders.

Three independent layers, none of which needs Triton:

* **The two specializations are proved against the frozen helpers themselves.** The
  frozen and the new ``@triton.jit`` helper bodies are re-executed on scalars, so the
  comparison is between the real source of each, not between two retyped copies. The
  decode ladder is enumerated over *every* producible byte pair (all 256 codes against
  every scale byte from 105 to 255) and the encode ladder over every rounding decision
  class of the FP32 exponent/mantissa space, plus random FP32 patterns.
* **The kernel body is emulated in the kernel's own order** -- FP32 float steps in
  torch, integer decisions from the re-executed helpers -- and held to the composed
  frozen operators over octave, tie, floor, scale-sweep, non-finite, signed-zero and
  random batteries.
* **AST contracts pin what a CPU cannot execute**: no lane-dependent shift anywhere in
  the two new helpers, the frozen governing helpers imported rather than retyped, the
  constant-shift normal rounding only ever called at 20 bits, the grid bound, the
  pointer width decided before any offset product, and the two code buffers written
  only together.

The two algebraic facts the specialization rests on are proved here rather than
asserted: the released amax floor and ``fp32(1/448)`` make the smallest producible
scale byte ``105``, and below that byte the constant ladder genuinely differs from the
frozen decoder -- so the domain restriction is load bearing, not vacuous.

Device compilation and hardware timing are separate claims and belong to the lead's run
on a leased card: this file says nothing about either, and never reports an oracle pass
as a device result.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import struct

import pytest
import torch

from relic_core.kernels import block_fp8_quant, unpack
from relic_core.kernels import fast_fused_fp8_activation_decode as fast
from relic_core.kernels import fused_fp8_activation_decode as frozen_fused

KERNEL_SOURCE = (Path(__file__).resolve().parents[1]
                 / "relic_core/kernels/_ascend_fast_fused_fp8_activation_decode_kernels.py")
FROZEN_QUANT_SOURCE = (Path(__file__).resolve().parents[1]
                       / "relic_core/kernels/_ascend_fp8_quant_kernels.py")
FROZEN_UNPACK_SOURCE = (Path(__file__).resolve().parents[1]
                        / "relic_core/kernels/_ascend_unpack_kernels.py")
WRAPPER_SOURCE = (Path(__file__).resolve().parents[1]
                  / "relic_core/kernels/fast_fused_fp8_activation_decode.py")

BLOCK = block_fp8_quant.BLOCK
FLOOR = fast.FLOOR
INV_FP8_MAX = fast.INV_FP8_MAX
#: The smallest E8M0 byte the released encoder can produce; proved by a test below.
MIN_PRODUCED_SCALE = 105
#: BF16's smallest normal magnitude: no producible value may fall below it.
BF16_MIN_NORMAL_EXPONENT = -126


# -- executing @triton.jit helper bodies on scalars ----------------------------


def f32(value: float) -> float:
    """The FP32 rounding of a Python float, saturating to infinity on overflow.

    A device FP32 multiply overflows to infinity rather than trapping, so the emulation
    must do the same: the encode's ``|value| * 512`` is only ever selected for lanes
    below ``2**-6``, and a huge lane has to reach the mask as infinity, not as an
    exception, or the emulation would be stricter than the device.
    """
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        if math.isnan(value):
            return float("nan")
        return math.inf if value > 0 else -math.inf


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def wrap32(value: int) -> int:
    return (int(value) + 2**31) % 2**32 - 2**31


FLOAT32, INT32, UINT16 = "float32", "int32", "uint16"


class Num:
    """One scalar lane of the Triton integer/float subset the bit helpers use.

    ``tl.where`` only ever selects a pre-computed branch, so the emulation evaluates
    both branches exactly as the device does and reads the condition through
    ``__bool__``. Integer results wrap to 32 bits like the device's ``int32``; float
    results are rounded to FP32 after every operation, so a helper that relied on a
    wider intermediate would fail here rather than agree by accident.
    """

    def __init__(self, value):
        self.value = value.value if isinstance(value, Num) else value

    # -- conversions --
    def to(self, dtype, bitcast=False):
        if bitcast:
            if dtype == FLOAT32:
                return Num(bits_f32(int(self.value)))
            if dtype == INT32:
                return Num(wrap32(f32_bits(float(self.value))))
            raise AssertionError(f"unsupported bitcast width: {dtype}")
        if dtype == INT32:
            return Num(int(self.value))  # truncation toward zero, as the device does
        if dtype == FLOAT32:
            return Num(f32(float(self.value)))
        if dtype == UINT16:
            return Num(int(self.value) & 0xFFFF)
        raise AssertionError(f"unsupported cast target: {dtype}")

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)

    def __index__(self):
        return int(self.value)

    def __repr__(self):
        return f"Num({self.value!r})"

    def _apply(self, other, integer, floating):
        right = other.value if isinstance(other, Num) else other
        if isinstance(self.value, float) or isinstance(right, float):
            return Num(f32(floating(float(self.value), float(right))))
        return Num(wrap32(integer(int(self.value), int(right))))

    # -- arithmetic the helpers use --
    def __neg__(self):
        return Num(-self.value)

    def __pos__(self):
        return Num(+self.value)

    def __add__(self, other):
        return self._apply(other, lambda a, b: a + b, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._apply(other, lambda a, b: a - b, lambda a, b: a - b)

    def __rsub__(self, other):
        return Num(other).__sub__(self)

    def __mul__(self, other):
        return self._apply(other, lambda a, b: a * b, lambda a, b: a * b)

    __rmul__ = __mul__

    # -- bitwise, with the 32-bit wrap the device's int32 has --
    def __and__(self, other):
        return self._apply(other, lambda a, b: a & b, lambda a, b: a)

    __rand__ = __and__

    def __or__(self, other):
        return self._apply(other, lambda a, b: a | b, lambda a, b: a)

    __ror__ = __or__

    def __lshift__(self, other):
        count = int(other)
        assert 0 <= count < 32, "an undefined shift reached a selected branch"
        return Num(wrap32(int(self.value) << count))

    def __rlshift__(self, other):
        return Num(other) << self

    def __rshift__(self, other):
        count = int(other)
        assert 0 <= count < 32, "an undefined shift reached a selected branch"
        return Num(wrap32(int(self.value) >> count))

    def __rrshift__(self, other):
        return Num(other) >> self

    # -- comparisons --
    def __eq__(self, other):  # noqa: D105 - a lane predicate, not an identity test
        return Num(1 if self.value == _value(other) else 0)

    def __ne__(self, other):
        return Num(1 if self.value != _value(other) else 0)

    def __lt__(self, other):
        return Num(1 if self.value < _value(other) else 0)

    def __le__(self, other):
        return Num(1 if self.value <= _value(other) else 0)

    def __gt__(self, other):
        return Num(1 if self.value > _value(other) else 0)

    def __ge__(self, other):
        return Num(1 if self.value >= _value(other) else 0)

    __hash__ = None


def _value(one):
    return one.value if isinstance(one, Num) else one


def scalar_functions(*sources: Path) -> dict:
    """Every top-level ``@triton.jit`` helper in ``sources``, as callable Python.

    The bodies are the real ones; only the decorator and the ``tl`` namespace are
    stubbed. Passing several sources merges them into one namespace so a new helper
    that imports a frozen helper resolves to that frozen body.
    """
    functions = []
    for source in sources:
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
                functions.append(node)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    tl = _TritonStub()
    namespace = {"tl": tl}
    for source in sources:
        exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - test-local
    return namespace


class _TritonStub:
    """The ``tl`` surface the bit helpers use, over :class:`Num`."""

    float32, int32, uint16 = FLOAT32, INT32, UINT16

    @staticmethod
    def where(condition, when_true, when_false):
        return Num(when_true if bool(condition) else when_false)

    @staticmethod
    def minimum(first, second):
        return Num(min(_value(first), _value(second)))

    @staticmethod
    def maximum(first, second):
        return Num(max(_value(first), _value(second)))

    @staticmethod
    def cast(bits, dtype, bitcast=False):
        return Num(bits).to(dtype, bitcast=bitcast)


# The new helpers resolve ``_round_half_even_shift`` from the frozen module, exactly as
# the device module imports it; the frozen oracles are here for the same reason.
FAST_HELPERS = scalar_functions(KERNEL_SOURCE, FROZEN_QUANT_SOURCE)
FROZEN_HELPERS = scalar_functions(FROZEN_QUANT_SOURCE, FROZEN_UNPACK_SOURCE)


def fast_encode(bits: int) -> int:
    return int(FAST_HELPERS["_e4m3_byte_from_bits_fast"](Num(wrap32(bits))))


def frozen_encode(bits: int) -> int:
    return int(FROZEN_HELPERS["_e4m3_byte_from_bits"](Num(wrap32(bits))))


def fast_decode(code: int, scale: int) -> int:
    return int(FAST_HELPERS["_e4m3_e8m0_bf16_bits_fast"](Num(code), Num(scale))) & 0xFFFF


def frozen_decode(code: int, scale: int) -> int:
    return int(FROZEN_HELPERS["_e4m3_e8m0_bf16_bits"](Num(code), Num(scale))) & 0xFFFF


# -- the two proved specializations --------------------------------------------


def test_the_smallest_producible_scale_byte_is_105():
    """The decode ladder's only precondition, proved from the released constants.

    The 1e-4 floor makes ``fp32(1e-4)`` the smallest possible amax, so the smallest
    possible ``amax * fp32(1/448)`` is that product; its ceil-power-of-two exponent is
    ``-22`` and the emitted byte is ``105``. Both constants are the FP32 literals the
    frozen quantizer multiplies, so this is the released arithmetic and not a stand-in.
    """
    assert FLOOR == f32(1e-4) and FLOOR > 0.0
    assert INV_FP8_MAX == f32(1.0 / 448.0) and INV_FP8_MAX > 0.0
    smallest = f32(FLOOR * INV_FP8_MAX)
    exponent = ((f32_bits(smallest) >> 23) & 255) - 127 + int(
        (f32_bits(smallest) & 0x7FFFFF) != 0)
    assert exponent == -22, f"the smallest scale exponent is {exponent}, not -22"
    assert exponent + 127 == MIN_PRODUCED_SCALE
    # The product is monotone in amax, so the floor really is the minimiser: every
    # larger amax reports an exponent at or above this one.
    for step in range(1, 4000):
        amax = f32(FLOOR * (1.0 + step / 512.0))
        product = f32(amax * INV_FP8_MAX)
        reported = ((f32_bits(product) >> 23) & 255) - 127 + int(
            (f32_bits(product) & 0x7FFFFF) != 0)
        assert reported >= -22


def test_every_produced_value_is_a_bf16_normal_not_a_subnormal():
    """The second precondition: no conservative lane lands in BF16's subnormal range.

    Recomputed from the frozen decoder's own exponent terms rather than from the new
    ladder, so this proves the *contract* the ladder assumes.
    """
    smallest_magnitude_exponent = None
    for scale in range(MIN_PRODUCED_SCALE, 256):
        for code in range(256):
            exponent = (code >> 3) & 15
            mantissa = code & 7
            significand = mantissa if exponent == 0 else 8 + mantissa
            if significand == 0:
                continue
            # A NaN lane (the E4M3 NaN code, or any lane of a block whose scale byte is
            # the E8M0 NaN byte) has no magnitude to classify; the decoder overrides it
            # to NaN after the exponent arithmetic, so only finite lanes are proved.
            if (code & 127) == 127 or scale == 255:
                continue
            power = (-9 if exponent == 0 else exponent - 10) + scale - 127
            msb = 3 if significand >= 8 else (2 if significand >= 4 else (1 if significand >= 2 else 0))
            biased = power + msb + 127
            assert biased > 0, (code, scale, "would decode through BF16's subnormal path")
            if smallest_magnitude_exponent is None or biased < smallest_magnitude_exponent:
                smallest_magnitude_exponent = biased
    # biased == 1 is the smallest BF16 normal; anything below it would be subnormal.
    assert smallest_magnitude_exponent == 96, (
        f"the smallest producible exponent field is {smallest_magnitude_exponent}")
    assert 96 - 127 == -31 > BF16_MIN_NORMAL_EXPONENT


def test_the_decode_ladder_matches_the_frozen_decoder_on_every_produced_pair():
    """Exhaustive: all 256 code bytes against every scale byte the encoder can emit."""
    for scale in range(MIN_PRODUCED_SCALE, 256):
        for code in range(256):
            assert fast_decode(code, scale) == frozen_decode(code, scale), (code, scale)


def test_the_decode_ladder_is_not_a_relabelling_below_the_produced_domain():
    """Negative control: below byte 105 the ladder is genuinely a different function.

    Without this, ``scale >= 105`` could be decoration around a ladder that is actually
    valid everywhere, and the specialization would not be proved to need the proof.
    """
    differing = [(code, scale) for scale in range(MIN_PRODUCED_SCALE)
                 for code in range(256)
                 if fast_decode(code, scale) != frozen_decode(code, scale)]
    assert len(differing) >= 100, (
        "the ladder agreed below 105, so the domain restriction is untested")
    # And the disagreements are the BF16-subnormal cases the proof excludes.
    for code, scale in differing:
        exponent = (code >> 3) & 15
        mantissa = code & 7
        significand = mantissa if exponent == 0 else 8 + mantissa
        if significand == 0:
            continue
        power = (-9 if exponent == 0 else exponent - 10) + scale - 127
        msb = 3 if significand >= 8 else (2 if significand >= 4 else (1 if significand >= 2 else 0))
        assert power + msb + 127 <= 0, (code, scale, "disagreed outside the subnormal path")


def _significands(shift: int) -> list:
    """Every significand that is representative of a distinct rounding decision.

    The rounded quotient depends only on ``significand >> shift`` and on which side of
    the half-way point the discarded remainder falls, so one value per (quotient,
    remainder class) pair covers the branch exactly.
    """
    half = 1 << (shift - 1)
    values = []
    for quotient in range((0xFFFFFF >> shift) + 1):
        for remainder in {0, 1, half - 1, half, half + 1, (1 << shift) - 1}:
            significand = (quotient << shift) + remainder
            if 0x800000 <= significand <= 0xFFFFFF:
                values.append(significand)
    return values


def test_the_encode_subnormal_branch_matches_the_frozen_encoder_exhaustively():
    """Every subnormal-branch decision, over every exponent field that can select it.

    The frozen branch shifts by ``clamp(141 - exponent, 1, 25)``; that is 25 for every
    exponent field at or below 116, so the distinct shift widths below the E4M3 normal
    range are 21..25, plus the zero exponent field, whose lane is never zero-result
    from the shift but from the subnormal rule.
    """
    for exponent in range(0, 121):
        shift = min(max(141 - exponent, 1), 25)
        for significand in _significands(shift):
            for negative in (0, 1):
                bits = (negative << 31) | (exponent << 23) | (significand & 0x7FFFFF)
                assert fast_encode(bits) == frozen_encode(bits), hex(bits)


def test_the_encode_normal_branch_matches_the_frozen_encoder_exhaustively():
    """Every normal-branch decision: the constant-20 rounding against every exponent.

    Above the subnormal range the frozen encoder rounds by the constant 20 bits, so the
    decision is the top four significand bits, the discarded-remainder class and the
    exponent field -- all of which are enumerated here for every field that selects the
    branch.
    """
    for exponent in range(121, 256):
        for significand in _significands(20):
            for negative in (0, 1):
                bits = (negative << 31) | (exponent << 23) | (significand & 0x7FFFFF)
                assert fast_encode(bits) == frozen_encode(bits), hex(bits)


def test_the_encode_ladder_matches_the_frozen_encoder_on_random_fp32_patterns():
    generator = torch.Generator().manual_seed(97)
    patterns = torch.randint(0, 2**31, (60000,), generator=generator, dtype=torch.int64)
    for bits in patterns.tolist():
        for negative in (0, 1):
            pattern = (negative << 31) | bits
            assert fast_encode(pattern) == frozen_encode(pattern), hex(pattern)


def test_the_encode_ladder_matches_at_the_released_boundaries():
    """The exact values the released contract singles out."""
    special = [0.0, -0.0, 1e-4, -1e-4, 448.0, -448.0, 449.0, -449.0,
               2.0**-9, 2.0**-10, 2.0**-6, 2.0**-7, 3.4028234663852886e38,
               5.877471754111438e-39, 1.401298464324817e-45]
    special += [math.ldexp(1.0, e) for e in range(-30, 10)]
    special += [math.ldexp(1.0, e) * (1 + 1 / 16) for e in range(-14, 10)]
    special += [float("inf"), float("-inf"), float("nan")]
    for value in special:
        bits = f32_bits(value) if math.isfinite(value) else (
            0x7F800000 if value > 0 else 0xFF800000)
        if math.isnan(value):
            bits = 0x7FC00000
        assert fast_encode(bits) == frozen_encode(bits), (value, hex(bits))


def test_the_encode_tie_rule_is_load_bearing():
    """Negative control: a ``>=`` tie or a dropped odd-bit test changes real codes."""
    def mutant(bits):
        # The frozen and fast forms both round a tie to even; a mutant that always
        # rounds a tie up must disagree somewhere in the sweep above.
        sign = (wrap32(bits) >> 24) & 128
        exponent = (wrap32(bits) >> 23) & 255
        significand = (wrap32(bits) & 0x7FFFFF) | 0x800000
        value = abs(bits_f32(wrap32(bits) & 0x7FFFFFFF)) * 512.0
        lane = 0 < exponent <= 120
        scaled = f32(value) if lane else 0.0
        truncated = int(scaled)
        fraction = scaled - truncated
        increment = 1 if fraction >= 0.5 else 0
        subnormal = min(truncated + increment, 8)
        rounded = FROZEN_HELPERS["_round_half_even_shift"](Num(significand), Num(20))
        carry = int(rounded) >> 4
        biased = exponent - 120 + carry
        normal = 0x7E if biased > 15 else ((biased << 3) | (int(rounded) - 8 - (carry << 3)))
        return sign | (subnormal if exponent <= 120 else normal)

    differing = 0
    for exponent in (1, 100, 117, 118, 119, 120):
        shift = min(max(141 - exponent, 1), 25)
        half = 1 << (shift - 1)
        for quotient in range((0xFFFFFF >> shift) + 1):
            for remainder in (half - 1, half, half + 1):
                significand = (quotient << shift) + remainder
                if not 0x800000 <= significand <= 0xFFFFFF:
                    continue
                bits = (exponent << 23) | (significand & 0x7FFFFF)
                if mutant(bits) != frozen_encode(bits):
                    differing += 1
    assert differing > 0, "the mutant tie rule was indistinguishable from the real one"


# -- the kernel body, emulated in its own order --------------------------------


def emulate_fast(x: torch.Tensor):
    """The specialized kernel body in Python: FP32 steps, re-executed integer decisions.

    Every line mirrors ``fast_fused_fp8_activation_decode_kernel``. The float steps are
    the same IEEE FP32 operations Triton emits and the integer decisions are the new
    helper bodies themselves, so a mis-ordered or mis-scaled pass fails here.
    """
    width = int(x.shape[-1])
    flat = x.reshape(-1, width).to(torch.float32)
    rows = int(flat.shape[0])
    blocks = width // BLOCK
    scale_exponent = FROZEN_HELPERS["_scale_exponent_from_bits"]
    pow2_bits = FROZEN_HELPERS["_pow2_bits"]
    nonfinite_bits = FROZEN_HELPERS["_nonfinite_bits"]
    floor = torch.tensor(FLOOR, dtype=torch.float32)
    inverse_max = torch.tensor(INV_FP8_MAX, dtype=torch.float32)
    patterns = torch.empty((rows, width), dtype=torch.int32)
    codes = torch.empty((rows, width), dtype=torch.uint8)
    scales = torch.empty((rows, blocks), dtype=torch.uint8)
    for row in range(rows):
        for block in range(blocks):
            values = flat[row, block * BLOCK:(block + 1) * BLOCK]
            magnitude = values.view(torch.int32) & 0x7FFFFFFF
            block_nonfinite = max(int(nonfinite_bits(Num(int(one))))
                                  for one in magnitude) != 0
            amax = bits_f32(int(magnitude.max()))
            scaled = torch.maximum(torch.tensor(amax, dtype=torch.float32), floor) * inverse_max
            exponent = int(scale_exponent(Num(wrap32(f32_bits(float(scaled))))))
            inverse = bits_f32(int(pow2_bits(Num(-exponent))))
            normalized = torch.clamp(values * torch.tensor(inverse, dtype=torch.float32),
                                     -448.0, 448.0)
            block_codes = [fast_encode(int(one)) for one in normalized.view(torch.int32)]
            block_scale = exponent + 127
            if block_nonfinite:
                block_codes = [0x7F] * BLOCK
                block_scale = 255
            for lane, code in enumerate(block_codes):
                patterns[row, block * BLOCK + lane] = fast_decode(code, block_scale)
            codes[row, block * BLOCK:(block + 1) * BLOCK] = torch.tensor(
                block_codes, dtype=torch.uint8)
            scales[row, block] = block_scale
    return patterns, codes, scales


def reference(x: torch.Tensor):
    """The two frozen operators composed: the contract the specialization must meet."""
    width = int(x.shape[-1])
    flat = x.reshape(-1, width)
    codes, scales = block_fp8_quant.quantize_block_fp8(flat, backend="torch")
    decoded = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1, BLOCK),
                                        backend="torch")
    return decoded, codes, scales


def pattern_list(patterns: torch.Tensor) -> list:
    return [int(one) & 0xFFFF for one in patterns.reshape(-1).tolist()]


def decoded_bits(decoded: torch.Tensor) -> list:
    return [int(one) & 0xFFFF for one in decoded.reshape(-1).view(torch.int16).tolist()]


# -- value batteries -----------------------------------------------------------


def _block(rows: int, blocks: int, values) -> torch.Tensor:
    template = torch.tensor(list(values), dtype=torch.float32)
    assert template.numel() >= BLOCK, "a battery block is at least one 32-wide block"
    needed = rows * blocks * BLOCK
    return template.repeat(-(-needed // template.numel()))[:needed].reshape(
        rows, blocks * BLOCK)


def octave_battery() -> torch.Tensor:
    values = []
    for exponent in (0, -1, -2, -3, -4, -6, -7, -8, -9, -10, -12, -20, -30, -126,
                     1, 5, 40, 120, 127):
        for step in (-1, 0, 1):
            magnitude = math.ldexp(1.0, exponent) * (1 + step / 16)
            values.extend((magnitude, -magnitude))
    values.extend((448.0, -448.0, 449.0, -449.0, 1e-4, -1e-4, 0.0, -0.0,
                   2**-9, 2**-10, 2**-11, 2**-149, 3.4e38, -3.4e38))
    return _block(1, 2, values)


def tie_battery() -> torch.Tensor:
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
    """Blocks whose amax sits at or below the released floor, so byte 105 dominates."""
    values = [math.ldexp(1.0, -30 - index) for index in range(BLOCK)]
    at_floor = [1e-4] * BLOCK
    return torch.cat((_block(1, 1, values), _block(1, 1, at_floor)), dim=0)


def floor_battery() -> torch.Tensor:
    """Every lane exactly the floor, and one lane just under it."""
    at_floor = [FLOOR] * BLOCK
    under = [f32(FLOOR * 0.5)] * BLOCK
    zeros = [0.0] * BLOCK
    return torch.cat((_block(1, 1, at_floor), _block(1, 1, under), _block(1, 1, zeros)), dim=0)


def scale_sweep_battery() -> torch.Tensor:
    blocks = []
    for exponent in (-40, -30, -22, -14, -10, -1, 0, 1, 3, 8, 20, 60, 120, 127):
        top = math.ldexp(1.0, exponent)
        blocks.extend(top * (1 - lane / 64) for lane in range(BLOCK))
    return _block(len(blocks) // BLOCK, 1, blocks)


def nonfinite_battery() -> torch.Tensor:
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


def subnormal_e4m3_battery() -> torch.Tensor:
    """Blocks that reach every E4M3 subnormal code, not just the smallest.

    The block scale is ``2**ceil(log2(amax / 448))``, so an amax of exactly 448 puts the
    scale byte at the identity ``127`` and a lane of ``k * 2**-9`` then normalizes to
    ``k * 2**-9`` and rounds to subnormal code ``k``. Covering k = 1..7 exercises each
    rung of the constant subnormal ladder, which a battery of decreasing powers of two
    does not: those produce only one or two distinct codes.
    """
    values = [448.0]
    values.extend(k * 2.0**-9 for k in range(1, 8))
    values.extend(-k * 2.0**-9 for k in range(1, 8))
    values.extend(0.5 * 2.0**-9 for _ in range(4))  # a rounding tie, resolved to even
    values.extend(math.ldexp(1.0, -10 - index) for index in range(BLOCK - len(values)))
    return _block(2, 2, values)


def random_battery(dtype: torch.dtype, seed: int, rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, blocks * BLOCK, generator=generator) * 8).to(dtype)


BATTERIES = {
    "octaves": octave_battery,
    "ties": tie_battery,
    "tiny": tiny_battery,
    "floor": floor_battery,
    "scale_sweep": scale_sweep_battery,
    "nonfinite": nonfinite_battery,
    "negative_zero": negative_zero_battery,
    "e4m3_subnormals": subnormal_e4m3_battery,
    "random_bf16": lambda: random_battery(torch.bfloat16, 2101, 4, 3),
    "random_fp32": lambda: random_battery(torch.float32, 2102, 3, 5),
    "wide_random_bf16": lambda: random_battery(torch.bfloat16, 2103, 2, 16),
}


def test_the_batteries_are_not_degenerate():
    """Anti-vacuity: a pass that emitted one or two byte values would agree with anything."""
    codes_seen, scales_seen = set(), set()
    for make in BATTERIES.values():
        _, codes, scales = emulate_fast(make())
        codes_seen |= set(codes.reshape(-1).tolist())
        scales_seen |= set(scales.reshape(-1).tolist())
    assert len(codes_seen) >= 48, f"only {len(codes_seen)} distinct code bytes"
    assert len(scales_seen) >= 10, f"only {len(scales_seen)} distinct scale bytes"
    # The subnormal encode branch must actually be reached, or the specialization of
    # that branch is untested by the batteries.
    _, codes, _ = emulate_fast(BATTERIES["e4m3_subnormals"]())
    subnormal_codes = {code for code in codes.reshape(-1).tolist() if 0 < (code & 127) < 8}
    assert len(subnormal_codes) >= 4, (
        f"only {len(subnormal_codes)} E4M3 subnormal code bytes were exercised")


@pytest.mark.parametrize("name", sorted(BATTERIES))
def test_the_specialized_body_matches_the_two_operator_composition_bit_for_bit(name):
    """The whole pass: identical codes, identical scales, identical BF16 bits."""
    x = BATTERIES[name]()
    patterns, codes, scales = emulate_fast(x)
    decoded, expected_codes, expected_scales = reference(x)
    assert pattern_list(patterns) == decoded_bits(decoded)
    assert codes.reshape(-1).tolist() == expected_codes.reshape(-1).tolist()
    assert scales.reshape(-1).tolist() == expected_scales.reshape(-1).tolist()


def test_the_bit_comparison_has_teeth_on_the_decoded_scale():
    """Negative control: decoding the same bytes one octave down must be rejected."""
    x = BATTERIES["octaves"]()
    patterns, codes, scales = emulate_fast(x)
    assert pattern_list(patterns) == decoded_bits(reference(x)[0])
    blocks = int(x.shape[-1]) // BLOCK
    wrong = []
    for row in range(int(x.shape[0])):
        for block in range(blocks):
            scale = int(scales[row, block]) - 1
            for lane in range(BLOCK):
                wrong.append(fast_decode(int(codes[row, block * BLOCK + lane]), scale))
    assert wrong != pattern_list(patterns), "the oracle accepted a one-octave error"


def test_the_emulation_would_notice_a_missing_nonfinite_override():
    """Negative control: the NaN override is what makes those blocks NaN.

    Without it the infinity block would take a finite exponent byte and its lanes the
    saturated finite code, which decodes to large finite values instead. The frozen
    contract's NaN is therefore produced by the override and not by the code byte.
    """
    x = BATTERIES["nonfinite"]()
    patterns, codes, scales = emulate_fast(x)
    decoded = decoded_bits(reference(x)[0])
    assert pattern_list(patterns) == decoded
    for row in (0, 1):
        assert int(scales[row, 0]) == 255, "a non-finite block takes the NaN scale byte"
        assert codes[row, :BLOCK].tolist() == [0x7F] * BLOCK
        assert all((int(one) & 0x7F80) == 0x7F80 and (int(one) & 0x7F) != 0
                   for one in patterns[row].tolist()), "the block decodes to NaN"
    # The finite reading the override displaces: the saturated finite code under a real
    # exponent byte decodes to a large finite magnitude, not to NaN.
    displaced = [fast_decode(0x7E, 127) for _ in range(BLOCK)]
    assert all((one & 0x7F80) != 0x7F80 for one in displaced), (
        "the displaced reading must be finite for this control to mean anything")
    assert displaced != [int(one) for one in patterns[0].tolist()]


# -- the wrapper ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BATTERIES))
def test_the_torch_backend_is_the_frozen_composition(name):
    x = BATTERIES[name]()
    decoded, codes, scales = fast.quantize_decode_fp8_fast(
        x, backend="torch", return_codes=True)
    expected, expected_codes, expected_scales = reference(x)
    assert decoded.dtype == torch.bfloat16 and decoded.shape == x.shape
    assert decoded_bits(decoded) == decoded_bits(expected)
    assert codes.reshape(-1).tolist() == expected_codes.reshape(-1).tolist()
    assert scales.reshape(-1).tolist() == expected_scales.reshape(-1).tolist()


def test_the_wrapper_selects_the_same_oracle_as_the_frozen_wrapper():
    """The specialization shares the frozen validation, backend choice and oracle."""
    assert fast._torch_reference is frozen_fused._torch_reference
    assert fast._checked_input is frozen_fused._checked_input
    assert fast._checked_out is frozen_fused._checked_out
    assert fast._select_backend is frozen_fused._select_backend
    assert fast.BLOCK is block_fp8_quant.BLOCK
    assert fast.BLOCKS == frozen_fused.BLOCKS
    assert fast.MAX_PROGRAMS == frozen_fused.MAX_PROGRAMS
    assert fast.FLOOR == frozen_fused.FLOOR
    assert fast.INV_FP8_MAX == frozen_fused.INV_FP8_MAX
    x = random_battery(torch.bfloat16, 2104, 2, 2)
    ours = fast.quantize_decode_fp8_fast(x, backend="torch", return_codes=True)
    theirs = frozen_fused.quantize_decode_fp8(x, backend="torch", return_codes=True)
    assert decoded_bits(ours[0]) == decoded_bits(theirs[0])
    assert ours[1].reshape(-1).tolist() == theirs[1].reshape(-1).tolist()
    assert ours[2].reshape(-1).tolist() == theirs[2].reshape(-1).tolist()


def test_nonfinite_blocks_keep_the_frozen_nan_contract():
    x = nonfinite_battery()
    decoded, codes, scales = fast.quantize_decode_fp8_fast(
        x, backend="torch", return_codes=True)
    assert bool(torch.isnan(decoded[:2]).all())
    assert codes[0].tolist() == [0x7F] * BLOCK and codes[1].tolist() == [0x7F] * BLOCK
    assert scales[0].item() == 255 and scales[1].item() == 255
    assert bool(torch.isfinite(decoded[2]).all())


def test_a_negative_zero_keeps_its_sign_through_the_decode():
    x = negative_zero_battery()
    decoded, codes, _ = fast.quantize_decode_fp8_fast(x, backend="torch", return_codes=True)
    assert int(codes[0, 1]) == 0x80
    assert bool(torch.signbit(decoded[0, 1]))


@pytest.mark.parametrize("shape", [(BLOCK,), (2, BLOCK), (2, 3, 2 * BLOCK), (0, BLOCK)])
def test_arbitrary_rank_and_empty_input_keep_the_shape_contract(shape):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    decoded, codes, scales = fast.quantize_decode_fp8_fast(
        x, backend="torch", return_codes=True)
    assert decoded.shape == shape and codes.shape == shape
    assert scales.shape == (*shape[:-1], shape[-1] // BLOCK)
    if x.numel() == 0:
        assert decoded.numel() == 0


def test_out_is_used_and_returned_and_must_not_alias():
    x = random_battery(torch.bfloat16, 2105, 2, 2)
    out = torch.empty_like(x)
    returned = fast.quantize_decode_fp8_fast(x, backend="torch", out=out)
    assert returned is out
    assert decoded_bits(returned) == decoded_bits(reference(x)[0])
    with pytest.raises(ValueError, match="must not alias"):
        fast.quantize_decode_fp8_fast(x, backend="torch", out=x)


@pytest.mark.parametrize("call,error,match", [
    (lambda x: fast.quantize_decode_fp8_fast(x[:, ::2], backend="torch"),
     ValueError, "contiguous"),
    (lambda x: fast.quantize_decode_fp8_fast(x.to(torch.float16), backend="torch"),
     TypeError, "BF16 or FP32"),
    (lambda x: fast.quantize_decode_fp8_fast(x[:, :-1], backend="torch"),
     ValueError, "multiple of 32"),
    (lambda x: fast.quantize_decode_fp8_fast(torch.tensor(1.0), backend="torch"),
     ValueError, "rank >= 1"),
    (lambda x: fast.quantize_decode_fp8_fast(x, backend="bogus"),
     ValueError, "backend must be"),
    (lambda x: fast.quantize_decode_fp8_fast(x, backend="triton"),
     ValueError, "requires an NPU input"),
    (lambda x: fast.quantize_decode_fp8_fast(x, backend="torch", return_codes=1),
     TypeError, "return_codes must be a bool"),
    (lambda x: fast.quantize_decode_fp8_fast(
        x, backend="torch", out=torch.empty_like(x).float()),
     TypeError, "bfloat16"),
])
def test_bad_arguments_are_refused_by_name(call, error, match):
    x = torch.zeros(2, 2 * BLOCK, dtype=torch.bfloat16)
    with pytest.raises(error, match=match):
        call(x)


# -- AST contracts the CPU cannot execute --------------------------------------


def kernel_tree() -> ast.Module:
    return ast.parse(KERNEL_SOURCE.read_text())


def helper(name: str) -> ast.FunctionDef:
    for node in kernel_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in the specialized kernel module")


NEW_HELPERS = ("_e4m3_byte_from_bits_fast", "_e4m3_e8m0_bf16_bits_fast")


def test_no_lane_value_selects_a_shift_amount_in_the_new_helpers():
    """The whole point of the specialization, pinned on the source.

    Every shift in the two new helpers must have an integer literal on the right, so
    the backend emits a constant shift. A regression that reintroduced
    ``significand >> (141 - exponent)`` or ``<< (7 - msb)`` fails here.
    """
    offenders = []
    for name in NEW_HELPERS:
        for node in ast.walk(helper(name)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.LShift, ast.RShift)):
                if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int)):
                    offenders.append((name, ast.dump(node.op), ast.unparse(node)))
    assert offenders == [], f"lane-dependent shifts remain: {offenders}"


def test_the_new_helpers_do_not_touch_memory():
    """Integer-and-vector helpers only, the form this backend lowers reliably."""
    for name in NEW_HELPERS:
        attributes = {node.attr for node in ast.walk(helper(name))
                      if isinstance(node, ast.Attribute)}
        assert "load" not in attributes and "store" not in attributes, name
        assert "program_id" not in attributes and "num_programs" not in attributes, name


def test_the_frozen_governing_helpers_are_imported_not_retyped():
    imports = {}
    for node in kernel_tree().body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.name] = node.module
    for name in ("_scale_exponent_from_bits", "_pow2_bits", "_nonfinite_bits",
                 "_round_half_even_shift"):
        assert imports.get(name) == "_ascend_fp8_quant_kernels", name
    defined = [node.name for node in kernel_tree().body
               if isinstance(node, ast.FunctionDef)]
    assert defined == [*NEW_HELPERS, "fast_fused_fp8_activation_decode_kernel"], defined


def test_the_constant_shift_normal_rounding_is_only_called_at_twenty_bits():
    """The frozen normal encode is reused, and only with its own constant shift."""
    for node in ast.walk(kernel_tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_round_half_even_shift"):
            assert isinstance(node.args[1], ast.Constant) and node.args[1].value == 20, (
                "the frozen rounding helper must only be called with the constant 20")


def test_the_kernel_does_not_call_the_variable_shift_frozen_helpers():
    kernel = helper("fast_fused_fp8_activation_decode_kernel")
    called = {node.func.id for node in ast.walk(kernel)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for name in ("_e4m3_byte_from_bits", "_e4m3_e8m0_bf16_bits"):
        assert name not in called, f"{name} still has a variable shift"
    for name in NEW_HELPERS:
        assert name in called, name


def test_the_encode_and_decode_path_has_no_division_and_no_float_logarithm():
    """Integer bit arithmetic only: no division, no math-library logarithm, no FP8 cast.

    Checked on the AST rather than on the text, because the module docstring names the
    logarithm this pass replaces and a text scan would read that prose as code.
    """
    tree = kernel_tree()
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Div)], \
        "the released scale and encode are integer bit arithmetic, never a division"
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    for forbidden in ("log2", "exp2", "frexp", "log", "exp"):
        assert forbidden not in called, f"{forbidden} has no place in this path"
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("float8_e4m3fn", "float8_e5m2"):
        assert forbidden not in attributes, f"{forbidden} is the cast this path avoids"


def test_the_released_constants_arrive_as_constexpr_kernel_arguments():
    kernel = helper("fast_fused_fp8_activation_decode_kernel")
    constexpr = {}
    for argument in kernel.args.args:
        annotation = argument.annotation
        constexpr[argument.arg] = (isinstance(annotation, ast.Attribute)
                                   and annotation.attr == "constexpr")
    for name in ("NUM_BLOCKS", "BLOCK", "BLOCKS", "FLOOR", "INV_FP8_MAX",
                 "WRITE_CODES", "INDEX64"):
        assert constexpr.get(name), f"{name} must be a constexpr kernel argument"


def test_the_grid_is_bounded_and_the_kernel_strides_the_remainder():
    wrapper = WRAPPER_SOURCE.read_text()
    assert "MAX_PROGRAMS = " not in wrapper, "the cap is the frozen wrapper's constant"
    assert "min(MAX_PROGRAMS" in wrapper, "the launch grid must be capped"
    kernel = KERNEL_SOURCE.read_text()
    assert "tl.num_programs(0)" in kernel
    assert "for program in range(tl.program_id(0)" in kernel


def test_the_pointer_width_is_decided_before_any_offset_product():
    wrapper = WRAPPER_SOURCE.read_text()
    padded = wrapper.index("padded = ")
    launch = wrapper.index("fast_fused_fp8_activation_decode_kernel[")
    assert padded < launch, "the padded reach must be computed before the launch"
    assert "padded >= 2**31" in wrapper
    kernel = KERNEL_SOURCE.read_text()
    assert kernel.index("if INDEX64:") < kernel.index("block_ids = first_block")


def test_the_code_buffers_are_written_only_under_write_codes():
    kernel = helper("fast_fused_fp8_activation_decode_kernel")
    guarded = [node for node in ast.walk(kernel)
               if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "WRITE_CODES"]
    assert len(guarded) == 1
    stores = [node for node in ast.walk(guarded[0]) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "store"]
    assert len(stores) == 2, "both the code and the scale byte are gated together"


def test_the_decode_consumes_the_produced_byte_pair():
    """The fused decode must read the scale *byte*, not the raw exponent."""
    kernel = helper("fast_fused_fp8_activation_decode_kernel")
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_e4m3_e8m0_bf16_bits_fast"]
    assert len(calls) == 1
    code_argument, scale_argument = calls[0].args
    assert isinstance(code_argument, ast.Name) and code_argument.id == "codes"
    assert isinstance(scale_argument, ast.Subscript)
    assert isinstance(scale_argument.value, ast.Name)
    assert scale_argument.value.id == "scale_bytes"
    source = KERNEL_SOURCE.read_text()
    assert source.index("codes = tl.where(nonfinite") < source.index(
        "bits = _e4m3_e8m0_bf16_bits_fast")


def test_the_min_scale_precondition_is_stated_in_the_kernel_source():
    """The proof obligation the ladder relies on must be recorded where it is used."""
    source = KERNEL_SOURCE.read_text()
    assert "105" in source, "the smallest producible scale byte must be documented"
    assert "2**-31" in source or "2 ** -31" in source, (
        "the smallest producible magnitude must be documented")