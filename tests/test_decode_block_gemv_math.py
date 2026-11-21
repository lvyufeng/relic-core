"""Torch-free checks of the decode GEMV kernel sources and its fragment arithmetic.

Three separable claims are checked here, none of which needs a device:

* the new kernel module **re-uses** the frozen reduction helpers instead of
  re-deriving them, and its two passes are structurally what the docstrings say
  (no reduction over the K-block axis, no reassociation);
* the small new helpers -- the FP4 codebook validity/value/nominal split and the
  packed-nibble widening -- agree with the frozen codebook over *every* code;
* in the normal float32 range the per-fragment pipeline involves **no rounding**
  at all: it is reproduced exactly by rational arithmetic written independently
  in this file, so the two-half truncation/carry/renormalisation sequence is
  verified against a from-scratch statement rather than against itself.

The legacy out-of-range edge semantics (`_pow2` wrapping when an exponent leaves
the biased range, which `_cut` inherits) are *not* modelled rationally here; they
belong to the frozen helpers, are unchanged by the two-pass split, and are pinned
by ``tests/test_block_gemm_math.py`` plus the device equality test.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_decode_block_gemv_kernels.py"
GEMM = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_block_gemm_kernels.py"
UNPACK = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_unpack_kernels.py"

#: The exact doubled E2M1 codebook, indexed by nibble. This is the table
#: ``unpack_fp4_to_int8`` documents and implements; the packed decode path has to
#: produce the same INT8 values, and E2M1 code 8 is negative zero, which the
#: integer form deliberately cannot carry (it widens to 0).
DOUBLED = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


class Scalar:
    """The minimal ``tl``-expression shim the frozen math tests already use."""

    def __init__(self, value):
        self.value = value.value if isinstance(value, Scalar) else value

    def to(self, dtype, bitcast=False):
        if bitcast:
            if dtype in {"uint32", "int32"}:
                raw = bits(self.value)
                return Scalar(raw - 2**32 if dtype == "int32" and raw >= 2**31 else raw)
            if dtype == "float32":
                return Scalar(struct.unpack("<f", struct.pack("<I", int(self.value) & 0xFFFFFFFF))[0])
            raise AssertionError(dtype)
        if dtype == "float32":
            return Scalar(f32(self.value))
        if dtype == "uint32":
            return Scalar(int(self.value) & 0xFFFFFFFF)
        if dtype == "int32":
            value = int(self.value) & 0xFFFFFFFF
            return Scalar(value - 2**32 if value >= 2**31 else value)
        raise AssertionError(dtype)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __mul__(self, other):
        return Scalar(f32(self.value * Scalar(other).value))

    __rmul__ = __mul__

    def __add__(self, other):
        return Scalar(self.value + Scalar(other).value)

    __radd__ = __add__

    def __sub__(self, other):
        return Scalar(self.value - Scalar(other).value)

    def __rsub__(self, other):
        return Scalar(Scalar(other).value - self.value)

    def __neg__(self):
        return Scalar(-self.value)

    def __and__(self, other):
        return Scalar(int(self.value) & int(Scalar(other)))

    __rand__ = __and__

    def __or__(self, other):
        return Scalar(int(self.value) | int(Scalar(other)))

    __ror__ = __or__

    def __lshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "undefined shift, including unselected branch"
        return Scalar(int(self.value) << count)

    def __rshift__(self, other):
        count = int(Scalar(other))
        assert 0 <= count < 32, "undefined shift, including unselected branch"
        return Scalar(int(self.value) >> count)

    def __lt__(self, other):
        return Scalar(self.value < Scalar(other).value)

    def __le__(self, other):
        return Scalar(self.value <= Scalar(other).value)

    def __gt__(self, other):
        return Scalar(self.value > Scalar(other).value)

    def __ge__(self, other):
        return Scalar(self.value >= Scalar(other).value)

    def __eq__(self, other):
        return Scalar(self.value == Scalar(other).value)

    def __ne__(self, other):
        return Scalar(self.value != Scalar(other).value)


def tl_shim():
    return SimpleNamespace(
        where=lambda condition, x, y: Scalar(x if bool(condition) else y),
        maximum=lambda x, y: Scalar(max(Scalar(x).value, Scalar(y).value)),
        minimum=lambda x, y: Scalar(min(Scalar(x).value, Scalar(y).value)),
        abs=lambda x: Scalar(abs(Scalar(x).value)),
        uint32="uint32", int32="int32", float32="float32",
        constexpr=lambda value: value,  # a constexpr global is its value once unwrapped
    )


def _plain_int(node):
    """``-2000`` parses as ``UnaryOp(USub, Constant)``, not as a Constant."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _plain_int(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _constexpr_value(node):
    """The whitelisted scalar-constant grammar: a literal, its negation, ``tl.constexpr(x)``.

    The ``tl.constexpr(...)`` wrapper is the *only* accepted spelling for a module
    scalar in this package -- the Triton front end refuses a plain global inside a
    ``@jit`` body -- so the collector has to see through it to compare values.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "constexpr" and len(node.args) == 1:
        return _constexpr_value(node.args[0])
    return _plain_int(node)


def execute(paths, skip=()):
    """Exec the module-level helpers of each file (decorators stripped) as Python.

    Only ``FunctionDef`` nodes and plain integer constant assignments are taken,
    so a kernel or a call to ``triton.jit`` machinery cannot run; this executes
    the bit-expression bodies only.
    """
    namespace = {"tl": tl_shim()}
    for path in paths:
        tree = ast.parse(path.read_text())
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in skip:
                node.decorator_list = []
                body.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and _constexpr_value(node.value) is not None:
                node.value = ast.Constant(_constexpr_value(node.value))
                body.append(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"),
             namespace)
    return namespace


def helpers():
    return execute((KERNEL, GEMM, UNPACK), skip=("decode_fragment_kernel", "decode_accumulate_kernel",
                                                 "block_gemm_kernel"))


def kernel_body(name):
    """The parsed kernel, with the ``@triton.jit`` decorator removed.

    The decorator is a ``triton`` attribute reference, not something the body
    reads, so leaving it in would make every free-name scan report ``triton``.
    """
    tree = ast.parse(KERNEL.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    return function


# -- the packed widening and the FP4 codebook ---------------------------------------------


def test_packed_nibbles_reproduce_the_frozen_int8_codebook():
    """Both nibbles of every one of the 256 packed byte values decode exactly."""
    h = helpers()
    for byte in range(256):
        low = h["_e2m1_x2"](Scalar(byte & 15))
        high = h["_e2m1_x2"](Scalar((byte >> 4) & 15))
        assert int(low) == DOUBLED[byte & 15], (byte, int(low))
        assert int(high) == DOUBLED[(byte >> 4) & 15], (byte, int(high))


@pytest.mark.parametrize("code", range(16))
def test_fp4_value_is_the_exact_original_e2m1_value(code):
    h = helpers()
    x2 = Scalar(DOUBLED[code])
    magnitude = Scalar(abs(DOUBLED[code]))
    value = float(h["_fp4_value"](x2, magnitude))
    assert bits(value) == bits(DOUBLED[code] * 0.5), (code, value)
    assert h["_fp4_code_valid"](magnitude).value


@pytest.mark.parametrize("code", range(16))
def test_fp4_nominal_is_the_exponent_of_the_original_value(code):
    h = helpers()
    doubled = DOUBLED[code]
    nominal = int(h["_fp4_nominal"](Scalar(abs(doubled))))
    if doubled == 0:
        assert nominal == -1000, "a zero operand carries the frozen empty sentinel"
    else:
        # The nominal is floor(log2(value)) of the *original* E2M1 value, which is
        # the doubled code over two -- not of the doubled code itself.
        expected = math.frexp(abs(doubled) * 0.5)[1] - 1
        assert nominal == expected, (code, nominal, expected)


@pytest.mark.parametrize("magnitude", [5, 7, 9, 10, 11, 13, 200])
def test_an_out_of_codebook_int8_value_is_zero_but_still_flagged_invalid(magnitude):
    """Exactly the frozen split: zeroed value for the product, magnitude for the anchor."""
    h = helpers()
    value = float(h["_fp4_value"](Scalar(magnitude), Scalar(abs(magnitude))))
    assert bits(value) == bits(0.0)
    assert not h["_fp4_code_valid"](Scalar(abs(magnitude))).value
    # The nominal is magnitude-only in the frozen kernel; only the flag turns the
    # column into NaN, so the anchor may read a value for an invalid code.
    assert int(h["_fp4_nominal"](Scalar(abs(magnitude)))) in {-1, 0, 1, 2}


# -- the fragment is rounding-free: a rational statement of the same reduction ------------


def e4_exact(code):
    """The exact E4M3FN value of a byte as a rational, or ``None`` for NaN."""
    if code & 127 == 127:
        return None
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = (Fraction(mantissa, 2**9) if exponent == 0
             else Fraction(8 + mantissa) * Fraction(2) ** (exponent - 10))
    return -value if code & 128 else value


def a_nominal(code):
    return max(((code >> 3) & 15) - 7, -6)


def b_nominal(code, fp4):
    if fp4:
        doubled = abs(DOUBLED[code])
        if doubled == 0:
            return None
        if doubled >= 8:
            return 2
        if doubled >= 4:
            return 1
        return 0 if doubled >= 2 else -1
    return None if code & 127 == 0 else a_nominal(code)


def b_value(code, fp4):
    if fp4:
        return Fraction(DOUBLED[code], 2)
    return e4_exact(code)


def cut(value, anchor):
    """The frozen window truncation, stated on rationals.

    ``None`` is the kernel's sentinel path: ``_grid_exponent`` maps anything below
    -500 to a zero grid exponent, i.e. truncation to an integer.
    """
    grid = 0 if anchor is None or anchor < -500 else anchor - 13
    step = Fraction(2) ** grid
    return Fraction(math.trunc(value / step)) * step


def floor_exponent(value):
    """The frozen ``_floor_exp`` on a normal value: the exponent field, not a log."""
    if value == 0:
        return None
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    return exponent


def fragment_exact(a_codes, b_codes, fp4):
    """Independent rational statement of the documented fragment reduction."""
    products, exponents = [], []
    for ac, bc in zip(a_codes, b_codes):
        a = e4_exact(ac)
        b = b_value(bc, fp4)
        if a is None or b is None:
            return None
        product = a * b
        products.append(product)
        exponents.append(None if product == 0 else a_nominal(ac) + b_nominal(bc, fp4))
    first = max((v for v in exponents[:16] if v is not None), default=None)
    total = sum((cut(p, first) for p in products[:16]), Fraction(0))
    running = floor_exponent(total)
    second_half = [v for v in exponents[16:] if v is not None]
    if running is not None:
        second_half.append(running)
    second = max(second_half, default=None)
    total = cut(total, second) + sum((cut(p, second) for p in products[16:]), Fraction(0))
    return cut(total, floor_exponent(total))


def fragment_float(a_codes, b_codes, fp4):
    """The same reduction on float32 values, i.e. what the kernel actually computes."""
    fp4_values = tuple(Fraction(DOUBLED[code], 2) for code in range(16))
    products, exponents = [], []
    for ac, bc in zip(a_codes, b_codes):
        a = e4_exact(ac)
        b = fp4_values[bc] if fp4 else e4_exact(bc)
        if a is None or b is None:
            return None
        product = f32(float(a) * float(b))
        products.append(product)
        exponents.append(None if product == 0 else a_nominal(ac) + b_nominal(bc, fp4))
    first = max((v for v in exponents[:16] if v is not None), default=None)
    total = f32(sum((f32(p) for p in (float(x) for x in
                                      (cut_float(p, first) for p in products[:16]))), 0.0))
    running = None if total == 0 else struct.unpack("<i", struct.pack("<f", total))[0]
    running = None if running is None else ((running >> 23) & 255) - 127
    second_half = [v for v in exponents[16:] if v is not None]
    if running is not None:
        second_half.append(running)
    second = max(second_half, default=None)
    carry = cut_float(total, second)
    tail = sum((cut_float(p, second) for p in products[16:]), 0.0)
    total = f32(carry + f32(tail))
    exponent = None if total == 0 else ((struct.unpack("<i", struct.pack("<f", total))[0] >> 23) & 255) - 127
    return cut_float(total, exponent)


def cut_float(value, anchor):
    grid = 0 if anchor is None or anchor < -500 else anchor - 13
    scale = math.ldexp(1.0, -grid)
    return f32(math.trunc(f32(value * scale)) * math.ldexp(1.0, grid))


def test_the_fragment_reduction_is_rounding_free_over_every_a_and_b_code():
    """Every E4M3 activation code against every E2M1 and E4M3 weight code.

    A product of two <= 4-bit significands needs at most 8 bits, so no FP32
    rounding happens inside a fragment; the rational statement above must equal the
    float one exactly, which is what makes the two-half carry exact.
    """
    for fp4 in (True, False):
        codes = range(16) if fp4 else range(256)
        for ac in range(256):
            for bc in codes:
                a_codes = [ac] * 32
                b_codes = [bc] * 32
                exact = fragment_exact(a_codes, b_codes, fp4)
                actual = fragment_float(a_codes, b_codes, fp4)
                if exact is None:
                    assert actual is None, (fp4, ac, bc)
                    continue
                assert actual is not None, (fp4, ac, bc)
                assert Fraction(actual) == exact, (fp4, ac, bc, actual, float(exact))


def test_mixed_blocks_are_rounding_free_and_keep_the_half_seam():
    """Random full blocks, so the two halves and the carry all take part."""
    import random

    rng = random.Random(4021)
    for fp4 in (True, False):
        codes = 16 if fp4 else 256
        for _ in range(200):
            a_codes = [rng.randrange(256) for _ in range(32)]
            b_codes = [rng.randrange(codes) for _ in range(32)]
            if any(code & 127 == 127 for code in a_codes):
                continue
            exact = fragment_exact(a_codes, b_codes, fp4)
            actual = fragment_float(a_codes, b_codes, fp4)
            if exact is None:
                assert actual is None
                continue
            assert Fraction(actual) == exact, (fp4, a_codes, b_codes)


def test_a_zero_activation_block_still_carries_a_zero_fragment():
    exact = fragment_exact([0] * 32, [7] * 32, True)
    assert exact == 0
    assert fragment_float([0] * 32, [7] * 32, True) == 0.0


# -- the split itself: structural pins ----------------------------------------------------


def test_kernel_module_reuses_the_frozen_reduction_helpers():
    """The fragment must be the frozen fragment, so the helpers are *imported*."""
    tree = ast.parse(KERNEL.read_text())
    imported = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.name] = node.module
    required = ("_cut", "_decode_e4m3", "_floor_exp", "_grid_exponent", "_nominal_e4",
                "_original_scale_code", "_pow2", "_scale_e8m0")
    for name in required:
        assert name in imported, f"{name} must come from the frozen block GEMM module"
        assert imported[name].endswith("_ascend_block_gemm_kernels"), (name, imported[name])
    assert imported["_e2m1_x2"].endswith("_ascend_unpack_kernels")
    redefined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert redefined.isdisjoint(required), "the reduction helpers must not be re-derived here"


def _free_names(function):
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    declared = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            declared.add(node.arg)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    declared.add(target.id)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
    return names - declared


def _is_jit(node):
    """Whether a function definition carries a ``*jit`` decorator."""
    return any(getattr(dec, "attr", getattr(dec, "id", "")) == "jit"
               for dec in node.decorator_list)


def _module_symbols():
    """Classify a kernel module's globals the way the front end sees them.

    ``plain`` is the set the front end rejects inside a ``@jit`` body: a scalar
    declared without ``tl.constexpr``. Both accepted spellings of the annotation are
    collected into ``constexpr`` -- the call form ``_X = tl.constexpr(5)`` and the
    annotation form ``_X: tl.constexpr = 5``.
    """
    table = {"jit": set(), "constexpr": set(), "plain": set(), "imported": set()}
    tree = ast.parse(KERNEL.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _is_jit(node):
            table["jit"].add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared = getattr(node.annotation, "attr", getattr(node.annotation, "id", ""))
            if declared == "constexpr":
                table["constexpr"].add(node.target.id)
            elif isinstance(node.value, ast.Constant):
                table["plain"].add(node.target.id)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                table["plain"].add(target.id)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
                    and value.func.attr == "constexpr":
                table["constexpr"].add(target.id)
        elif isinstance(node, ast.ImportFrom):
            table["imported"] |= {alias.asname or alias.name for alias in node.names}
    return table


def test_every_global_a_kernel_body_reads_is_constexpr_or_a_jit_function():
    """The Triton front end refuses a plain module global inside ``@jit``.

    It fails at compile time with "cannot access global variable ... within @jit",
    which is exactly how the first device run of the FP4 quantizer in this tree
    failed, and the environment flag that would relax the rule is deliberately
    unused. This test walks the real source, so adding a plain module constant to a
    kernel body fails here -- on a CPU, with no Triton installed -- instead of on a
    leased accelerator.
    """
    builtins = {"tl", "float", "int", "bool", "min", "max", "len", "range"}
    table = _module_symbols()
    allowed = table["jit"] | table["constexpr"] | table["imported"] | builtins
    for name in ("decode_fragment_kernel", "decode_accumulate_kernel"):
        for free in _free_names(kernel_body(name)):
            assert free not in table["plain"], (
                f"{name} reads the plain module global {free!r}; annotate it "
                f"tl.constexpr or pass it in")
            assert free in allowed, f"{name} reads the unknown global {free!r}"


def test_no_kernel_module_declares_a_plain_scalar_global():
    """Any scalar global must carry the annotation upfront, read or not yet."""
    table = _module_symbols()
    assert not table["plain"], (
        f"plain scalar globals {sorted(table['plain'])}; Triton only accepts "
        f"tl.constexpr globals inside @jit")
    assert table["constexpr"] == {"_BELOW_ALL", "_EMPTY"}


def test_every_constexpr_global_is_stable_across_recompiles():
    """A cached global must compare equal to itself, or the second compile fails.

    Triton re-checks a compiled kernel's globals by equality when the same kernel is
    requested again, so a self-unequal value (``nan``) reads as "changed since
    compiled" and the call raises. Both sentinels must stay integers.
    """
    for node in ast.parse(KERNEL.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "constexpr"):
            continue
        constant = _constexpr_value(value)
        name = node.targets[0].id
        assert constant == constant, (
            f"{name} is a constexpr whose value does not compare equal to itself; "
            f"that breaks the JIT's cached-global check")
        assert isinstance(constant, int), (name, constant)


def test_no_uint64_promoting_data_literal():
    tree = ast.parse(KERNEL.read_text())
    assert not any(isinstance(node, ast.Constant) and isinstance(node.value, int)
                   and node.value >= 2**31 for node in ast.walk(tree))


def _is_int64(node):
    """``tl.int64`` parses as an Attribute; a bare ``int64`` as a Name."""
    if isinstance(node, ast.Attribute):
        return node.attr == "int64"
    return isinstance(node, ast.Name) and node.id == "int64"


def test_index64_widens_before_any_offset_product():
    """Cast-after-multiply would corrupt an address above 2**31, so pin the order.

    Two things are checked: every ``.to(int64)`` is applied to a *name* or a
    ``program_id``/``arange`` value and never to an assembled sum or product (a
    cast-last would be too late), and each loop body widens its block induction
    variable as its first statement so no statement can use the narrow one.
    """
    for name in ("decode_fragment_kernel", "decode_accumulate_kernel"):
        function = kernel_body(name)
        casts = [node for node in ast.walk(function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "to"
                 and node.args and _is_int64(node.args[0])]
        assert casts, f"{name} never widens an index"
        for cast in casts:
            widened_value = cast.func.value
            assert isinstance(widened_value, (ast.Name, ast.Call, ast.Constant)), (
                f"{name} casts an assembled expression to int64: {ast.dump(widened_value)}")
        loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
        block_loops = [node for node in loops if "KBLOCKS" in ast.dump(node.iter)]
        assert block_loops, f"{name} has no loop over the K blocks"
        for loop in block_loops:
            first = loop.body[0]
            assert isinstance(first, ast.If) and "INDEX64" in ast.dump(first.test), (
                f"{name} must widen its block index before using it")
            assert isinstance(first.body[0], ast.Assign)
            assert "int64" in ast.dump(first.body[0].value)


def test_the_accumulation_pass_never_reduces_over_the_k_axis():
    """The whole point: K blocks are added in order, so no reduction over them."""

    def reductions(function):
        found = []
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"sum", "dot", "cumsum", "reduce", "max", "min", "fma"}:
                    found.append(node.func.attr)
        return found

    function = kernel_body("decode_accumulate_kernel")
    assert reductions(function) == [], "the K accumulation must not reduce or fuse"
    adds = [node for node in ast.walk(function)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Add)]
    assert adds, "the accumulation must be an explicit add"
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    assert any("KBLOCKS" in ast.dump(node.iter) for node in loops), "no loop over the K blocks"


def test_the_fragment_pass_reduces_only_along_the_lane_axis():
    """The intra-fragment ``tl.sum`` is over the 32 lanes and must never touch K blocks.

    The accepted kernel sums the first/second 16-lane halves along axis 1; that is
    the fragment, and it is required. What must not appear is a reduction over the
    K-block axis -- that would be a split-K tree and a different number.
    """
    function = kernel_body("decode_fragment_kernel")
    sums = [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sum"]
    assert sums, "the two half-lane sums must still be there"
    for node in sums:
        assert len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
            and node.args[1].value == 1, "the only reduction axis is the lane axis (1)"
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"dot", "cumsum", "fma"}, node.func.attr
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    block_loops = [node for node in loops if "KBLOCKS" in ast.dump(node.iter)]
    assert len(block_loops) == 1
    # The K-block axis must stay a scalar: materialising it as a vector inside the
    # loop (an ``arange`` over blocks) would be the start of a K-axis reduction.
    for node in block_loops[0].body:
        called = {sub.func.attr for sub in ast.walk(node)
                  if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)}
        assert "arange" not in called, "the K-block axis must not become a vector"
        assert "cumsum" not in called
    program_ids = {node.args[0].value for node in ast.walk(function)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "program_id" and node.args
                   and isinstance(node.args[0], ast.Constant)}
    assert program_ids == {0, 1}, f"expected a two-axis grid, saw program_id {program_ids}"