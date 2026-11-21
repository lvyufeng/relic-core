"""An independent, torch-free oracle for the compiled host CPU block-32 GEMV.

Every other test of this operator compares it against the Torch implementation in
``block_gemm.py``. That is the right reference -- it is the reduction the runtime
performs -- but it cannot answer the question this file asks, which is whether the
arithmetic is *exact*. An oracle written against the same float kernels the
reference uses would agree with it even if both rounded.

So the oracle here is built from ``fractions.Fraction``: the operands are exact
dyadic rationals, the truncating cuts and the window arithmetic are exact, and the
only place a value is rounded is the one place the reference rounds as well -- the
per-K-block FP32 accumulation and the two scale multiplies. Nothing in this file
imports torch, and the kernel is called through the C ABI with plain ``ctypes``
buffers, so no tensor library sits between the bytes and the result.

The claims it pins:

* the reduction is exact when no rounding is asked for. With a single K block and
  unit scales there is no accumulation and no scaling, so the kernel's output has
  to be the exact rational fragment -- not merely close to it;
* a power-of-two scale is an exact rescale. An integer shift preserves the
  significand, so the result differs from the unscaled one only in exponent, and
  ``2 ** -127`` (a binary32 subnormal) must survive rather than be flushed to zero;
* the two-half window rule, the running-sum carry and the final renormalisation
  each land on their grid, checked against explicit hand-computed values rather
  than against the kernel;
* truncation is toward zero, so a negated block is exactly the negation of the
  positive one.
"""
from __future__ import annotations

import ctypes
import math
import struct
import tempfile
from fractions import Fraction

import pytest

from relic_core.kernels import cpu_block_gemv as gemv

BLOCK = 32
HALF = 16
WINDOW = 13
EMPTY = -1000


# -- exact building blocks ----------------------------------------------------------------------


def pow2(exponent):
    return Fraction(2) ** exponent


def e4_value(code):
    """The exact value of an e4m3 byte, or ``None`` for the two NaN encodings."""
    if code & 0x7F == 0x7F:
        return None
    field, mantissa = (code >> 3) & 15, code & 7
    magnitude = Fraction(mantissa, 512) if field == 0 else Fraction(8 + mantissa) * pow2(field - 10)
    return -magnitude if code & 0x80 else magnitude


def e4_nominal(code):
    """``max(true exponent, -6)``, or the sentinel that a zero operand contributes."""
    if code & 0x7F == 0:
        return EMPTY
    return max(((code >> 3) & 15) - 7, -6)


def floor_log2(value):
    """``floor(log2|value|)`` for a non-zero rational, computed exactly."""
    value = abs(value)
    estimate = value.numerator.bit_length() - value.denominator.bit_length()
    while value < pow2(estimate):
        estimate -= 1
    while value >= pow2(estimate + 1):
        estimate += 1
    return estimate


def truncate(value):
    """Truncation toward zero, exactly."""
    return Fraction(math.trunc(value))


def round_half_even(value):
    """Round a non-negative rational to the nearest integer, ties to even."""
    floor = math.floor(value)
    remainder = value - floor
    if remainder < Fraction(1, 2):
        return floor
    if remainder > Fraction(1, 2):
        return floor + 1
    return floor if floor % 2 == 0 else floor + 1


def to_f32(value):
    """The exact rational value of ``value`` rounded to binary32."""
    if value == 0:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    exponent = floor_log2(magnitude)
    if exponent > 127:
        return math.inf * sign
    if exponent < -126:
        subnormal = magnitude / pow2(-149)
        steps = round_half_even(subnormal)
        if steps == 0:
            return Fraction(0)
        if steps >= 1 << 23:
            return sign * pow2(-126)
        return sign * Fraction(steps) * pow2(-149)
    significand = magnitude / pow2(exponent)
    steps = round_half_even(significand * (1 << 23))
    if steps == 1 << 24:
        exponent += 1
        steps = 1 << 23
    if exponent > 127:
        return math.inf * sign
    return sign * Fraction(steps) * pow2(exponent - 23)


def e8m0_value(byte):
    """The exact value of a raw E8M0 scale byte."""
    return Fraction(1 << (byte - 127)) if byte >= 127 else Fraction(1, 1 << (127 - byte))


def bits_of(value):
    """The binary32 bit pattern of a value the oracle produced."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


# -- the reference reduction, in exact arithmetic -----------------------------------------------


def cut_term(value, anchor):
    """Truncate ``value`` onto ``2 ** (anchor - 13)``, in real units.

    The reference cuts by multiplying by the grid's inverse, truncating that
    quotient and multiplying the step back in, so the result is a value on the grid
    rather than a count of steps. ``anchor is None`` is the sentinel branch a half
    with no live operand takes: the grid degenerates to the identity instead of to
    an exponent derived from a product that was never there.
    """
    if anchor is None:
        return value
    step = pow2(anchor - WINDOW)
    return truncate(value / step) * step


def block_fragment(a_block, b_block):
    """The 32-wide block reduction: two halves, a running window, one renormalisation."""
    a_values = [e4_value(code) for code in a_block]
    b_values = [e4_value(code) for code in b_block]
    if any(value is None for value in a_values) or any(value is None for value in b_values):
        return None
    products = [a * b for a, b in zip(a_values, b_values)]
    nominals = [e4_nominal(a) + e4_nominal(b) for a, b in zip(a_block, b_block)]

    anchor_first = max(nominals[:HALF])
    first_half = sum((cut_term(p, None if anchor_first < -500 else anchor_first)
                      for p in products[:HALF]), Fraction(0))

    # The second half takes max(its own anchor, the running sum's exponent): the
    # running total is a peer of the second half's products, not a prefix of them.
    # A first half that summed to exactly zero contributes the empty exponent,
    # which never wins the maximum against a live anchor.
    running = floor_log2(first_half) if first_half != 0 else EMPTY
    window = max(max(nominals[HALF:]), running)
    resolved = None if window < -500 else window

    # Both the carried total and the second half's products are cut onto that one
    # grid and then added in real units.
    total = cut_term(first_half, resolved)
    total += sum((cut_term(p, resolved) for p in products[HALF:]), Fraction(0))

    if total == 0:
        return Fraction(0)
    # A separate, final renormalisation: the second half's sum can carry bits past
    # its own window, so the block result gets one more cut at its own exponent.
    return cut_term(total, floor_log2(total))


def mul_f32(left, right):
    if math.isinf(left) or math.isinf(right):
        return left * right
    return to_f32(left * right)


def add_f32(left, right):
    if math.isinf(left) or math.isinf(right):
        return left + right
    return to_f32(left + right)


def oracle(a_codes, a_scale, b_codes, b_scale, *, block_rows=1):
    """The expected output, computed one column at a time in increasing K order."""
    k = len(a_codes)
    blocks = k // BLOCK
    n = len(b_codes)
    out = []
    for column in range(n):
        scale_row = column if block_rows == 1 else column // block_rows
        accumulator = Fraction(0)
        poisoned = False
        for block in range(blocks):
            low = block * BLOCK
            fragment = block_fragment(a_codes[low:low + BLOCK], b_codes[column][low:low + BLOCK])
            if fragment is None:
                poisoned = True
                break
            # Activation scale first, weight scale second: two separately rounded
            # multiplies, in that order.
            scaled = mul_f32(mul_f32(fragment, e8m0_value(a_scale[block])),
                             e8m0_value(b_scale[scale_row * blocks + block]))
            accumulator = add_f32(accumulator, scaled)
        out.append(math.nan if poisoned else accumulator)
    return out


# -- driving the kernel directly, with no tensor library -----------------------------------------


def call_kernel(cache_dir, a_codes, a_scale, b_codes, b_scale, *, block_rows=1, threads=1):
    k = len(a_codes)
    n = len(b_codes)
    assert len(a_scale) == k // BLOCK
    library = gemv.load(cache_dir=cache_dir)
    a_buffer = (ctypes.c_uint8 * k)(*a_codes)
    a_scale_buffer = (ctypes.c_uint8 * len(a_scale))(*a_scale)
    b_buffer = (ctypes.c_uint8 * (n * k))(*[code for row in b_codes for code in row])
    b_scale_buffer = (ctypes.c_uint8 * len(b_scale))(*b_scale)
    out_buffer = (ctypes.c_float * n)()
    used = ctypes.c_int32(0)
    status = library.relic_cpu_block_gemv_fp8(
        ctypes.cast(a_buffer, ctypes.c_void_p),
        ctypes.cast(a_scale_buffer, ctypes.c_void_p),
        ctypes.cast(b_buffer, ctypes.c_void_p),
        ctypes.cast(b_scale_buffer, ctypes.c_void_p),
        ctypes.cast(out_buffer, ctypes.c_void_p),
        n,
        k,
        block_rows,
        threads,
        ctypes.byref(used),
    )
    assert status == 0, f"the kernel refused the call with status {status}"
    return list(out_buffer), used.value


@pytest.fixture(scope="session")
def cache():
    directory = tempfile.mkdtemp(prefix="relic-cpu-block-gemv-math-")
    try:
        gemv.build(cache_dir=directory)
    except RuntimeError as error:
        pytest.skip(f"the compiled host CPU GEMV cannot be built here: {error}")
    return directory


def assert_matches_oracle(cache, a_codes, a_scale, b_codes, b_scale, *, block_rows=1, note=""):
    got, _ = call_kernel(cache, a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    expected = oracle(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    assert len(got) == len(expected)
    for column, (actual, wanted) in enumerate(zip(got, expected)):
        actual_bits = struct.unpack("<I", struct.pack("<f", actual))[0]
        if isinstance(wanted, float) and math.isnan(wanted):
            assert math.isnan(actual), f"{note} column {column}: expected NaN, got {actual}"
            continue
        assert actual_bits == bits_of(wanted), (
            f"{note} column {column}: {struct.unpack('<f', struct.pack('<I', actual_bits))[0]!r} "
            f"!= {float(wanted)!r} (bits {actual_bits:#010x} != {bits_of(wanted):#010x})"
        )


# -- exactness, with nothing to round -----------------------------------------------------------


def test_a_single_block_with_unit_scales_needs_no_rounding_at_all(cache):
    """One K block and unit scales: the answer is the exact rational fragment.

    There is no accumulation to round and the scales are 1.0, so any deviation here
    is the kernel's own error rather than a rounding the reference also performs.
    A tolerance-based test could not distinguish the two; a bit comparison can.
    """
    cases = [
        # (activation code, weight codes) chosen to exercise each branch in turn
        (0x38, [0x38] * BLOCK),                     # plain in-window products
        (0x78, [0x78] * BLOCK),                     # large magnitudes
        (0x01, [0x01] * BLOCK),                     # subnormal operands
        (0x38, [0x38, 0xB8] * HALF),                # exact cancellation to zero
        (0x7E, [0x7E] * BLOCK),                     # largest finite value
        (0x38, list(range(1, 33))),                 # every subnormal field in one block
    ]
    for index, (activation, weights) in enumerate(cases):
        a_codes = [activation] * BLOCK
        b_codes = [list(weights)]
        scale = [127]
        expected = block_fragment(a_codes, b_codes[0])
        got, _ = call_kernel(cache, a_codes, scale, b_codes, scale)
        assert struct.pack("<f", got[0]) == struct.pack("<f", float(expected)), (
            f"case {index}: {got[0]!r} != {float(expected)!r} exactly"
        )


def test_the_fragment_never_needs_rounding_across_random_blocks(cache):
    """The exact fragment is always representable in binary32, for any code choice.

    This is the property the whole exactness argument rests on, so it is checked
    against the oracle's own rational result rather than assumed.
    """
    import random

    generator = random.Random(20261118)
    for _ in range(64):
        a_codes = [generator.randrange(256) for _ in range(BLOCK)]
        b_codes = [generator.randrange(256) for _ in range(BLOCK)]
        a_codes = [code - 1 if code & 0x7F == 0x7F else code for code in a_codes]
        b_codes = [code - 1 if code & 0x7F == 0x7F else code for code in b_codes]
        fragment = block_fragment(a_codes, b_codes)
        if fragment == 0:
            continue
        assert to_f32(fragment) == fragment, f"the exact fragment {float(fragment)!r} needs rounding"


# -- the window rule, against hand-computed values ----------------------------------------------


def test_the_two_halves_use_their_own_anchors(cache):
    """A small first half and a large second half, with the answer computed by hand.

    ``0x28`` is 0.25 and ``0x30`` is 0.5, so each first-half product is 0.125 and
    the sixteen of them sum to exactly 2.0 with an anchor of -3. ``0x78`` is 256
    and ``0x70`` is 128, so each second-half product is 32768 with an anchor of 15
    -- fourteen octaves away. A kernel that used one anchor for the whole block, or
    that carried the first half's window into the second, lands somewhere else.
    """
    a_codes = [0x28] * HALF + [0x78] * HALF
    b_codes = [0x30] * HALF + [0x70] * HALF
    # First half: 16 products of 2**-3 sum to exactly 2.0 on the 2**-16 grid.
    # Second half: 16 products of 2**15 with anchor 15; the running sum is 2.0, so
    # window = max(15, 1) = 15 and its grid is 2**2, on which 2**15 is exact.
    # The carried 2.0 is cut onto that grid and truncates to 0, so the block total
    # is 16 * 2**15 = 2**19, which the final cut leaves alone.
    assert block_fragment(a_codes, b_codes) == Fraction(2**19)
    got, _ = call_kernel(cache, a_codes, [127], [b_codes], [127])
    assert got[0] == 2.0**19


def test_a_large_first_half_carries_into_the_second_halfs_window(cache):
    """The case the running sum exists for: the second window must follow the sum.

    ``0x78`` products put the first half at 2**19 with a running exponent of 19,
    while the second half's own anchor is only -4. Taking the second half's anchor
    alone would return 1.0; following the running sum truncates its products to
    zero and returns 2**19, which is the reference's answer.
    """
    a_codes = [0x78] * HALF + [0x28] * HALF
    b_codes = [0x70] * HALF + [0x28] * HALF
    # First half: 16 products of 2**15 sum to 2**19, so the running exponent is 19.
    # Second half: the products are 2**-4 with an anchor of only -4, but the window
    # follows the running sum to 19, whose grid is 2**6 -- on which 2**-4 truncates
    # to zero. Taking the second half's own anchor would keep them and give 1.0.
    assert block_fragment(a_codes, b_codes) == Fraction(2**19)
    got, _ = call_kernel(cache, a_codes, [127], [b_codes], [127])
    assert got[0] == 2.0**19
    # The wrong window is not a rounding difference: it is a factor of 2**19.
    assert got[0] != 1.0


def test_a_cancelling_first_half_leaves_no_running_exponent(cache):
    """When the first half sums to zero, the second half keeps its own anchor.

    The running exponent is a sentinel in that case rather than a value read out of
    a zero, and the second half must not be dragged down by it. The first half is
    ``+x, -x`` repeated, which cancels exactly.
    """
    a_codes = [0x38] * BLOCK
    b_codes = ([0x38, 0xB8] * (HALF // 2)) + [0x70] * HALF
    # First half: +1, -1 repeated, so it sums to exactly zero and contributes the
    # empty exponent rather than a running one. Second half: 1.0 * 128.0 = 2**7
    # each on the 2**-6 grid, so the block total is 16 * 2**7 = 2**11.
    assert block_fragment(a_codes, b_codes) == Fraction(2**11)
    got, _ = call_kernel(cache, a_codes, [127], [b_codes], [127])
    assert got[0] == 2.0**11


def test_an_anchorless_block_cuts_nothing(cache):
    """An all-zero weight row has no anchor, so its products pass through untouched."""
    a_codes = [0x7E] * BLOCK
    b_codes = [0x00] * BLOCK
    assert block_fragment(a_codes, b_codes) == 0
    got, _ = call_kernel(cache, a_codes, [127], [b_codes], [127])
    assert got[0] == 0.0


# -- scale behaviour ----------------------------------------------------------------------------


def test_truncation_is_toward_zero_so_a_negated_block_is_the_negation(cache):
    """The cut truncates toward zero, which is not the same as flooring.

    Negating every activation code negates every product and every grid value, so
    the result has to be the exact negation. A kernel that floored instead would
    differ by one grid step on some blocks, and only on negative ones.
    """
    b_codes = [[0x40] * HALF + [0x58] * HALF]
    a_positive = [0x38] * HALF + [0x41] * HALF
    a_negative = [code ^ 0x80 for code in a_positive]
    positive, _ = call_kernel(cache, a_positive, [127], b_codes, [127])
    negative, _ = call_kernel(cache, a_negative, [127], b_codes, [127])
    assert negative[0] == -positive[0]


def test_the_lowest_grid_scale_is_an_exact_subnormal_rescale(cache):
    """``2 ** -127`` is the smallest original E8M0 scale and is a binary32 subnormal.

    Rescaling by it moves the exponent only, so the significand -- and therefore
    the answer's lower bits -- must be preserved. A kernel that flushed subnormals
    to zero would return 0 here; one that used a rounded ``exp2`` would lose the
    bottom bits of the result.
    """
    # 2**-6 * 2**-6 * 16 = 2**-7, and one 2**-127 rescale puts it at 2**-134: inside
    # the binary32 subnormal range, where a kernel that flushed would return zero.
    a_codes = [0x08] * BLOCK                  # 2**-6
    b_codes = [[0x08] * BLOCK]
    fragment = block_fragment(a_codes, b_codes[0])
    assert fragment == pow2(-7)
    expected = mul_f32(mul_f32(fragment, e8m0_value(0)), e8m0_value(127))
    got, _ = call_kernel(cache, a_codes, [0], b_codes, [127])
    assert struct.pack("<f", got[0]) == struct.pack("<f", float(expected))
    assert got[0] != 0.0 and math.isfinite(got[0])
    assert abs(got[0]) < 2.0**-126, "the result should be subnormal, so nothing was flushed"


def test_the_highest_finite_grid_scale_is_an_exact_rescale_until_it_overflows(cache):
    """``2 ** 127`` is the largest grid value; the products may overflow, exactly as the oracle does."""
    a_codes = [0x38] * BLOCK
    b_codes = [[0x40] * BLOCK]
    fragment = block_fragment(a_codes, b_codes[0])
    expected = mul_f32(mul_f32(fragment, e8m0_value(254)), e8m0_value(254))
    got, _ = call_kernel(cache, a_codes, [254], b_codes, [254])
    assert struct.pack("<f", got[0]) == struct.pack("<f", float(expected))


@pytest.mark.parametrize("byte", [0, 1, 63, 126, 127, 128, 190, 254])
def test_a_power_of_two_scale_only_moves_the_exponent(cache, byte):
    """Scaling by ``2 ** (byte - 127)`` must equal the unscaled result times that power exactly."""
    import random

    generator = random.Random(1000 + byte)
    a_codes = [generator.randrange(1, 0x7F) for _ in range(BLOCK)]
    b_codes = [[generator.randrange(1, 0x7F) for _ in range(BLOCK)]]
    plain, _ = call_kernel(cache, a_codes, [127], b_codes, [127])
    scaled, _ = call_kernel(cache, a_codes, [byte], b_codes, [byte])
    scale = e8m0_value(byte)
    expected = mul_f32(mul_f32(Fraction(plain[0]), scale), scale)
    assert struct.pack("<f", scaled[0]) == struct.pack("<f", float(expected))


# -- agreement with the exact oracle over whole columns -----------------------------------------


def test_random_columns_match_the_exact_rational_oracle(cache):
    import random

    generator = random.Random(424242)
    k, n = 128, 6
    a_codes = [generator.randrange(256) for _ in range(k)]
    a_codes = [code - 1 if code & 0x7F == 0x7F else code for code in a_codes]
    b_codes = [[generator.randrange(256) for _ in range(k)] for _ in range(n)]
    b_codes = [[code - 1 if code & 0x7F == 0x7F else code for code in row] for row in b_codes]
    a_scale = [generator.randrange(0, 255) for _ in range(k // BLOCK)]
    b_scale = [generator.randrange(0, 255) for _ in range(n * (k // BLOCK))]
    assert_matches_oracle(cache, a_codes, a_scale, b_codes, b_scale, note="random rational")


def test_the_full_accumulation_matches_over_many_k_blocks(cache):
    """K = 6144 in exact arithmetic: 192 sequential block accumulations, fully checked."""
    import random

    generator = random.Random(6144)
    k, n = 6144, 4
    a_codes = [generator.randrange(256) for _ in range(k)]
    a_codes = [code - 1 if code & 0x7F == 0x7F else code for code in a_codes]
    b_codes = [[generator.randrange(256) for _ in range(k)] for _ in range(n)]
    b_codes = [[code - 1 if code & 0x7F == 0x7F else code for code in row] for row in b_codes]
    a_scale = [generator.randrange(110, 145) for _ in range(k // BLOCK)]
    b_scale = [generator.randrange(110, 145) for _ in range(n * (k // BLOCK))]
    assert_matches_oracle(cache, a_codes, a_scale, b_codes, b_scale, note="K=6144")


def test_a_nan_code_makes_its_column_nan_in_the_oracle_too(cache):
    a_codes = [0x38] * BLOCK
    b_codes = [[0x40] * BLOCK, [0x7F] * BLOCK]
    assert_matches_oracle(cache, a_codes, [127], b_codes, [127, 127], note="nan column")


def test_the_square_scale_grid_matches_the_exact_oracle(cache):
    """``block_rows=32`` shares one scale across each 32-row band of the weight.

    The grid shapes differ from the per-row form the other tests use, and the two
    are easy to confuse at the API boundary, so the exact oracle checks the square
    layout too -- including that column 31 and column 32 pick different scale rows.
    """
    import random

    generator = random.Random(3232)
    k, n = 128, 64
    a_codes = [generator.randrange(256) for _ in range(k)]
    a_codes = [code - 1 if code & 0x7F == 0x7F else code for code in a_codes]
    b_codes = [[generator.randrange(256) for _ in range(k)] for _ in range(n)]
    b_codes = [[code - 1 if code & 0x7F == 0x7F else code for code in row] for row in b_codes]
    a_scale = [generator.randrange(110, 145) for _ in range(k // BLOCK)]
    # One scale per 32-row band and per K block, with a deliberately visible
    # difference between the two bands.
    b_scale = [110 + 7 * band + block for band in range(n // BLOCK) for block in range(k // BLOCK)]
    assert_matches_oracle(cache, a_codes, a_scale, b_codes, b_scale,
                          block_rows=BLOCK, note="square grid")
