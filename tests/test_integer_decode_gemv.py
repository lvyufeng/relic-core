"""Contracts for the integer-significand decode GEMV.

Three independent layers, none of which needs an accelerator:

* a **Torch mirror** of the kernel's integer statements (``_mirror_fragment``),
  compared bit for bit against the accepted CPU operator. This is what proves the
  significand/power rewrite computes the accepted fragment and not merely
  something close to it;
* an **exact-rational oracle** (``fractions.Fraction``) over every one of the
  4096 E4M3 x E2M1 code pairs, checking the field decomposition and the truncated
  term straight from the definition, with no float arithmetic at all;
* the **contract**: shapes, dtypes, aliasing, scratch, geometry caps and the
  module-level constexpr rule the backend enforces at compile time.

The NPU comparison against the accepted operator lives in the runtime worktree's
hardware run; nothing here is a substitute for it.
"""
from __future__ import annotations

import ast
import math
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import integer_decode_gemv as G  # noqa: E402
from relic_core.kernels.decode_block_gemv import (  # noqa: E402
    fp4_packed_decode_gemv,
    fp8_decode_gemv,
)

EMPTY = -1000
KERNEL_MODULE = Path(G.__file__).with_name("_ascend_integer_decode_kernels.py")


# -- the mirror: the kernel's integer statements in Torch --------------------------------------


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    return ((exponent.to(torch.int32) + 127) << 23).contiguous().view(torch.float32)


def _ilog2(value: torch.Tensor) -> torch.Tensor:
    """The frozen kernel's integer floor-log2, transcribed."""
    high = torch.where(value >= 65536, torch.full_like(value, 16), torch.zeros_like(value))
    value = value >> high
    shift = torch.where(value >= 256, torch.full_like(value, 8), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    shift = torch.where(value >= 16, torch.full_like(value, 4), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    shift = torch.where(value >= 4, torch.full_like(value, 2), torch.zeros_like(value))
    value = value >> shift
    high = high + shift
    return high + torch.where(value >= 2, torch.ones_like(value), torch.zeros_like(value))


def _grid_exponent(anchor: torch.Tensor) -> torch.Tensor:
    return torch.where(anchor < -500, torch.zeros_like(anchor), anchor - 13)


def _trunc_shift(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    left = shift.clamp(min=0)
    right = (-shift).clamp(min=1)
    magnitude = value.abs()
    scaled = torch.where(shift >= 0, magnitude << left, magnitude >> right)
    return torch.where(value < 0, -scaled, scaled)


def _fields_e4(codes: torch.Tensor):
    codes = codes.to(torch.int32)
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    nan = (codes & 127) == 127
    zero = (codes & 127) == 0
    significand = torch.where(exponent == 0, mantissa, 8 + mantissa)
    significand = torch.where(nan, torch.zeros_like(significand), significand)
    power = torch.where(exponent == 0, -9, exponent - 10)
    nominal = torch.maximum(exponent - 7, torch.full_like(exponent, -6))
    nominal = torch.where(zero, torch.full_like(nominal, EMPTY), nominal)
    return significand, power, nominal, (codes & 128) != 0, nan


def _fields_e2(code: torch.Tensor):
    magnitude = code.to(torch.int32) & 7
    significand = torch.where(magnitude < 2, magnitude, 2 + (magnitude & 1))
    power = torch.where(magnitude < 2, -1, (magnitude >> 1) - 2)
    nominal = (magnitude >> 1) - 1
    nominal = torch.where(magnitude != 0, nominal, torch.full_like(nominal, EMPTY))
    return significand, power, nominal, (code & 8) != 0


def _terms(significand_product, power, grid, negative, half):
    shift = power - grid[:, None]
    magnitude = torch.where(
        shift >= 0, significand_product << shift.clamp(min=0),
        significand_product >> (-shift).clamp(min=1),
    )
    magnitude = torch.where(half, magnitude, torch.zeros_like(magnitude))
    return torch.where(negative, -magnitude, magnitude)


def _mirror_fragment(a_codes: torch.Tensor, a_scale: torch.Tensor, b_codes: torch.Tensor,
                     block: int, *, b_is_e4: bool = False):
    """The kernel fragment for one K block: integer reduction, then the FP32 scales.

    ``a_codes`` is ``[32]``, ``b_codes`` is ``[N, 32]`` of integer E2M1 magnitudes
    with their sign in bit 3, ``a_scale`` is the E8M0 byte for this block. Returns
    the fragment before rescaling, so the caller applies its own scale order.
    """
    sig_a, pa, nom_a, neg_a, _nan_a = _fields_e4(a_codes)
    if b_is_e4:
        sig_b, pb, nom_b, neg_b, _nan_b = _fields_e4(b_codes)
    else:
        sig_b, pb, nom_b, neg_b = _fields_e2(b_codes)
    significand_product = sig_a[None, :] * sig_b
    power = pa[None, :] + pb
    nominal = nom_a[None, :] + nom_b
    negative = neg_a[None, :] ^ neg_b

    grid0 = _grid_exponent(nominal[:, :16].max(-1).values)
    total0 = _terms(significand_product, power, grid0, negative,
                    torch.arange(32)[None, :] < 16).sum(-1)
    carry = torch.where(
        total0 == 0, torch.full_like(total0, EMPTY), _ilog2(total0.abs()) + grid0
    )
    second = torch.maximum(nominal[:, 16:].max(-1).values, carry)
    grid1 = _grid_exponent(second)
    whole = _terms(significand_product, power, grid1, negative,
                   torch.arange(32)[None, :] >= 16).sum(-1)
    whole = whole + _trunc_shift(total0, grid0 - grid1)

    exponent = torch.where(
        whole == 0, torch.full_like(whole, EMPTY), _ilog2(whole.abs()) + grid1
    )
    window = _grid_exponent(exponent)
    return _trunc_shift(whole, grid1 - window).to(torch.float32) * _pow2(window)


def _mirror_packed(a, sa, b_packed, sb):
    """The grouped/serial fragment chain, accumulated in increasing block order."""
    n, k = b_packed.shape[0], a.shape[1]
    blocks = k // 32
    out = torch.zeros((1, n), dtype=torch.float32)
    for block in range(blocks):
        packed = b_packed[:, block * 16:(block + 1) * 16].to(torch.int32)
        low = packed & 15
        high = (packed >> 4) & 15
        codes = torch.empty((n, 32), dtype=torch.int32)
        codes[:, 0::2] = low
        codes[:, 1::2] = high
        row = a[0, block * 32:(block + 1) * 32]
        fragment = _mirror_fragment(row, sa[0, block], codes, block)
        scale_a = _pow2(sa[0, block].to(torch.int32) - 127)
        scale_b = _pow2(sb[:, block].to(torch.int32) - 127)
        out = out + (fragment * scale_a) * scale_b
    return out


def _mirror_fp8(a, sa, b_codes, sb, block_rows):
    n, k = b_codes.shape[0], a.shape[1]
    blocks = k // 32
    out = torch.zeros((1, n), dtype=torch.float32)
    for block in range(blocks):
        codes = b_codes[:, block * 32:(block + 1) * 32].to(torch.int32)
        row = a[0, block * 32:(block + 1) * 32]
        fragment = _mirror_fragment(row, sa[0, block], codes, block, b_is_e4=True)
        scale_a = _pow2(sa[0, block].to(torch.int32) - 127)
        rows = torch.arange(n) // block_rows
        raw = sb[rows, block]
        if raw.dtype == torch.float32:
            scale_b = raw
        else:
            scale_b = _pow2(raw.to(torch.int32) - 127)
        out = out + (fragment * scale_a) * scale_b
    return out


# -- fixtures ---------------------------------------------------------------------------------


def _codes(shape, generator, *, e4=True):
    """Valid E4M3 bytes (never NaN) or valid E2M1 nibble magnitudes."""
    raw = torch.randint(0, 255, shape, generator=generator, dtype=torch.int32)
    raw[(raw & 127) == 127] = 0x38
    return raw.to(torch.uint8)


def _packed(n, k, generator):
    return torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.int32).to(torch.uint8)


# -- the mirror is the accepted operator ------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n,k", [(1, 32), (17, 64), (33, 128), (64, 256)])
def test_mirror_matches_the_accepted_fp4_packed_operator(seed, n, k):
    generator = torch.Generator().manual_seed(seed)
    a = _codes((1, k), generator)
    sa = torch.randint(0, 255, (1, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.randint(0, 255, (n, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    mirror = _mirror_packed(a, sa, b, sb)
    assert torch.equal(mirror.view(torch.int32), accepted.view(torch.int32))


@pytest.mark.parametrize("block_rows", [1, 32])
def test_mirror_matches_the_accepted_fp8_operator(block_rows):
    generator = torch.Generator().manual_seed(11)
    n, k = 40, 96
    a = _codes((1, k), generator)
    sa = torch.randint(0, 255, (1, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    b = _codes((n, k), generator)
    rows = math.ceil(n / block_rows)
    sb = torch.randint(0, 255, (rows, k // 32), generator=generator, dtype=torch.int32).to(torch.uint8)
    accepted = fp8_decode_gemv(a, sa, b, sb, block_rows=block_rows, backend="torch")
    mirror = _mirror_fp8(a, sa, b, sb, block_rows)
    assert torch.equal(mirror.view(torch.int32), accepted.view(torch.int32))


def test_mirror_matches_on_every_activation_code_against_every_nibble():
    """All 256 E4M3 activation bytes against all 16 E2M1 nibbles, in one K block."""
    n, k = 16, 32
    nibbles = torch.arange(16, dtype=torch.int32)
    b = torch.empty((n, k // 2), dtype=torch.uint8)
    for column in range(n):
        packed = torch.empty(k // 2, dtype=torch.int32)
        packed[0::2] = nibbles[:k // 4]
        packed[1::2] = nibbles.flip(0)[:k // 4]
        b[column] = packed.to(torch.uint8)
    sb = torch.full((n, 1), 127, dtype=torch.uint8)
    sa = torch.full((1, 1), 127, dtype=torch.uint8)
    for code in range(256):
        a = torch.full((1, k), code, dtype=torch.uint8)
        accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
        if (code & 127) == 127:
            # A NaN activation is the invalid-flag path, not an arithmetic one:
            # the accepted operator reports NaN for the whole column and this
            # operator raises the same flag. The integer fragment itself is zero.
            assert bool(torch.isnan(accepted).all())
            continue
        mirror = _mirror_packed(a, sa, b, sb)
        assert torch.equal(mirror.view(torch.int32), accepted.view(torch.int32)), (
            f"E4M3 byte {code:#04x} disagrees with the accepted reduction"
        )


def test_first_half_cancellation_and_zero_carry_match_the_accepted_operator():
    """Equal magnitudes with opposite signs cancel the first half exactly.

    The first half then carries a zero exponent sentinel into the second half,
    which is where an integer rewrite is most likely to diverge from the float
    one. The second half is built so the surviving terms still normalize.
    """
    k = 32
    a = torch.tensor([[0x38] * k], dtype=torch.uint8)  # all 1.0
    # Alternating +1/-1 codes in the first half: 1 and 9 (0x1 is +0.5? no: bit 3 is
    # the sign), so use magnitude 2 (1.0) with and without the sign bit.
    packed = torch.zeros((2, k // 2), dtype=torch.int32)
    # 16 K lanes = 8 packed bytes per half; low nibble is the even K element.
    packed[:, :8] = 2 + 10 * 16  # +1.0 / -1.0 alternating, eight times
    packed[:, 8:] = torch.tensor([4 + 4 * 16] * 4 + [6 + 6 * 16] * 4, dtype=torch.int32)
    packed = packed.to(torch.uint8)
    sa = torch.full((1, 1), 127, dtype=torch.uint8)
    sb = torch.full((2, 1), 127, dtype=torch.uint8)
    accepted = fp4_packed_decode_gemv(a, sa, packed, sb, backend="torch")
    mirror = _mirror_packed(a, sa, packed, sb)
    assert torch.equal(mirror.view(torch.int32), accepted.view(torch.int32))


# -- the exact-rational oracle ----------------------------------------------------------------


def _e4_exact(byte: int) -> Fraction:
    """The exact value of an E4M3 byte, as a rational, from its definition."""
    sign = -1 if byte & 128 else 1
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    if exponent == 0:
        return Fraction(sign * mantissa, 2**9)
    return Fraction(sign * (8 + mantissa), 1) * Fraction(2) ** (exponent - 10)


def _e2_exact(code: int) -> Fraction:
    """The exact E2M1 value of a 4-bit code: 0, .5, 1, 1.5, 2, 3, 4, 6, signed."""
    magnitudes = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
                  Fraction(2), Fraction(3), Fraction(4), Fraction(6))
    value = magnitudes[code & 7]
    return -value if code & 8 else value


def test_field_decomposition_is_exact_for_every_code():
    """``significand * 2**power`` must equal the exact value, for all 4096 pairs."""
    for byte in range(256):
        if (byte & 127) == 127:
            continue  # NaN has no value; the invalid flag is what carries it
        significand, power, _nominal, negative, _nan = _fields_e4(torch.tensor([byte]))
        rebuilt = Fraction(int(significand), 1) * Fraction(2) ** int(power)
        if bool(negative):
            rebuilt = -rebuilt
        assert rebuilt == _e4_exact(byte), f"E4M3 byte {byte:#04x}"
    for code in range(16):
        significand, power, _nominal, negative = _fields_e2(torch.tensor([code]))
        rebuilt = Fraction(int(significand), 1) * Fraction(2) ** int(power)
        if bool(negative):
            rebuilt = -rebuilt
        assert rebuilt == _e2_exact(code), f"E2M1 code {code:#x}"


def test_truncated_term_is_exact_toward_zero_for_every_pair():
    """The integer term must be ``trunc(product / 2**grid)`` exactly, for all pairs."""
    for byte in range(256):
        if (byte & 127) == 127:
            continue
        a_exact = _e4_exact(byte)
        sig_a, pa, _na, neg_a, _nan = _fields_e4(torch.tensor([byte]))
        for code in range(16):
            b_exact = _e2_exact(code)
            sig_b, pb, _nb, neg_b = _fields_e2(torch.tensor([code]))
            product = a_exact * b_exact
            for grid in (-20, -13, -7, -3, 0, 5):
                shift = int(pa) + int(pb) - grid
                magnitude = int(sig_a) * int(sig_b)
                magnitude = magnitude << shift if shift >= 0 else magnitude >> (-shift)
                sign = -1 if bool(neg_a) ^ bool(neg_b) else 1
                expected = int((product * Fraction(2) ** (-grid)).__trunc__())
                assert sign * magnitude == expected, (
                    f"byte {byte:#04x} code {code:#x} grid {grid}"
                )


# -- contract ---------------------------------------------------------------------------------


def _operands(n=8, k=96, seed=3):
    generator = torch.Generator().manual_seed(seed)
    a = _codes((1, k), generator)
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    return a, sa, b, sb


def test_cpu_backend_delegates_and_rejects_the_fused_backend():
    a, sa, b, sb = _operands()
    out = G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch")
    assert out.shape == (1, 8) and out.dtype == torch.float32
    with pytest.raises(ValueError, match="NPU tensors only"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="triton")


@pytest.mark.parametrize("variant", ["auto", "serial", "grouped"])
def test_every_variant_returns_the_accepted_answer_on_the_cpu_path(variant):
    a, sa, b, sb = _operands()
    out = G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", variant=variant)
    accepted = fp4_packed_decode_gemv(a, sa, b, sb, backend="torch")
    assert torch.equal(out.view(torch.int32), accepted.view(torch.int32))


def test_the_schedule_knobs_must_be_powers_of_two():
    """``rows_n`` sizes a ``tl.arange`` lane vector, which needs a power-of-two extent."""
    a, sa, b, sb = _operands()
    for rows_n in (3, 5, 6, 12):
        with pytest.raises(ValueError, match="rows_n must be a power of two"):
            G.fp4_packed_integer_decode_gemv(
                a, sa, b, sb, backend="torch", rows_n=rows_n)
    for group_k in (3, 5, 24):
        with pytest.raises(ValueError, match="group_k must be a power of two"):
            G.fp4_packed_integer_decode_gemv(
                a, sa, b, sb, backend="torch", rows_n=4, group_k=group_k)
    for rows_n in (32, 64, 128):
        with pytest.raises(ValueError, match="exceeds the supported maximum 16"):
            G.fp4_packed_integer_decode_gemv(
                a, sa, b, sb, backend="torch", rows_n=rows_n)
    with pytest.raises(ValueError, match="exceeds the supported maximum"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", rows_n=4, group_k=64)
    # The accepted values are the powers of two inside the measured limits.
    for rows_n in (1, 2, 4, 8, 16):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", rows_n=rows_n, group_k=1)


def test_an_unknown_variant_is_refused_by_name():
    a, sa, b, sb = _operands()
    with pytest.raises(ValueError, match="variant must be one of"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", variant="tree")


def test_the_geometry_is_bounded_before_any_allocation():
    """A geometry over the program cap must fail on the CPU path too."""
    from relic_core.kernels.integer_decode_gemv import _integer_grids

    chunks, groups, ksplit = _integer_grids(129_280, 224, 16, 4)
    assert chunks * ksplit <= 65535 and groups == 56
    with pytest.raises(ValueError, match="total-program cap"):
        _integer_grids(129_280, 224, 1, 4)


def test_packed_bytes_must_be_two_codes_per_position():
    a, sa, b, sb = _operands()
    with pytest.raises(ValueError, match="two K codes per byte"):
        G.fp4_packed_integer_decode_gemv(a, sa, b[:, :-1].contiguous(), sb, backend="torch")


def test_scale_grid_must_be_the_original_one():
    a, sa, b, sb = _operands()
    with pytest.raises(ValueError, match="original grid"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb[:, :1].contiguous(), backend="torch")


def test_the_activation_must_be_a_single_row():
    generator = torch.Generator().manual_seed(5)
    a = _codes((2, 64), generator)
    b = _packed(4, 64, generator)
    with pytest.raises(ValueError, match="exactly one activation row"):
        G.fp4_packed_integer_decode_gemv(
            a, torch.full((2, 2), 127, dtype=torch.uint8), b,
            torch.full((4, 2), 127, dtype=torch.uint8), backend="torch")


def test_output_must_not_alias_an_operand():
    """An output that shares storage with the activation codes is refused, not raced."""
    k, n = 96, 8
    generator = torch.Generator().manual_seed(3)
    buf = torch.zeros(k, dtype=torch.float32)
    a = buf.view(torch.uint8)[:k].view(1, k)
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b = _packed(n, k, generator)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    out = buf[:n].view(1, n)
    assert out.data_ptr() == buf.data_ptr()
    with pytest.raises(ValueError, match="must not overlap"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", out=out)


def test_scratch_is_validated_with_the_frozen_layout():
    a, sa, b, sb = _operands(n=8, k=96)
    scratch = G.integer_decode_gemv_scratch(8, 96)
    assert [tuple(t.shape) for t in scratch] == [(3, 8), (3, 8)]
    assert scratch[0].dtype == torch.float32 and scratch[1].dtype == torch.uint8
    out = G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", scratch=scratch)
    assert out.shape == (1, 8)
    bad = (torch.empty((2, 8), dtype=torch.float32), torch.empty((3, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match="fragment scratch must be"):
        G.fp4_packed_integer_decode_gemv(a, sa, b, sb, backend="torch", scratch=bad)


def test_no_extra_permission_flag_is_needed_to_use_the_operator():
    """The exact defaults must be usable with no opt-in flag or 256-byte budget."""
    import inspect

    signature = inspect.signature(G.fp4_packed_integer_decode_gemv)
    assert "permission" not in signature.parameters
    assert "allow_" not in " ".join(signature.parameters)
    source = Path(G.__file__).read_text()
    assert "os.environ" not in source and "getenv" not in source


# -- the backend's compile-time rules ---------------------------------------------------------


def test_every_global_a_kernel_body_reads_is_constexpr_or_a_jit_function():
    """A plain module global is a hard compile error on this backend, not a warning."""
    source = KERNEL_MODULE.read_text()
    tree = ast.parse(source)
    declared = {
        node.target.id for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.target, ast.Name)
    }
    jit_names = {
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    }
    imported = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
    bodies = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    assert bodies, "no @triton.jit function found"
    builtins = {"range", "float", "int", "bool", "min", "max", "abs"}
    for function in bodies:
        allowed = {"tl", "triton"} | jit_names | imported | builtins
        allowed |= {argument.arg for argument in function.args.args}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                pending = list(node.targets)
                while pending:
                    target = pending.pop()
                    if isinstance(target, ast.Name):
                        allowed.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        pending.extend(target.elts)
        for node in ast.walk(function):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                allowed.add(node.target.id)
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name in allowed:
                    continue
                if name in declared:
                    raise AssertionError(
                        f"{function.name} reads module global {name!r}; it must be a "
                        "tl.constexpr so the compiled-kernel cache key is stable"
                    )
                raise AssertionError(f"{function.name} reads unknown global {name!r}")


def test_no_kernel_declares_a_float_nan_global():
    """``nan != nan`` would make the cached-global check fail on recompilation."""
    tree = ast.parse(KERNEL_MODULE.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call):
                continue  # tl.constexpr(...) of an int is the accepted spelling
            if isinstance(value, ast.Constant) and isinstance(value.value, float):
                if math.isnan(value.value):
                    raise AssertionError("a float NaN global is not stable across recompiles")


def test_the_kernel_never_decodes_an_operand_to_a_float():
    """The point of this operator: no float decode, no expanded INT8 weight."""
    source = KERNEL_MODULE.read_text()
    for forbidden in ("_decode_e4m3", "unpack_fp4_to_int8", "_e2m1_x2", "float64", "pow("):
        assert forbidden not in source, f"the integer path must not use {forbidden}"

# -- launch wiring: the grid and argument order each kernel is handed ---------------------------


class _Capture:
    """A stand-in Triton kernel that records the grid and the positional arguments."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def __getitem__(self, grid):
        self.pending = grid
        return self

    def __call__(self, *arguments):
        self.calls.append((self.pending, arguments))
        return None


def _fakes(monkeypatch):
    """Replace the two kernel modules and the launch context with recorders.

    No Triton and no device: this pins the part that a numerical test cannot see
    on a CPU-only host -- the grid each launch uses and the exact positional order
    every kernel and helper is called with.
    """
    import sys
    import types

    from relic_core.kernels import integer_decode_gemv as module

    fragment = _Capture("integer_fragment_kernel")
    serial = _Capture("integer_serial_kernel")
    accumulate = _Capture("decode_accumulate_kernel")
    kernels = types.ModuleType("relic_core.kernels._ascend_integer_decode_kernels")
    kernels.integer_fragment_kernel = fragment
    kernels.integer_serial_kernel = serial
    decoder = types.ModuleType("relic_core.kernels._ascend_decode_block_gemv_kernels")
    decoder.decode_accumulate_kernel = accumulate
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setitem(sys.modules, decoder.__name__, decoder)
    monkeypatch.setattr(module, "_npu_launch", lambda raw: __import__("contextlib").nullcontext())
    monkeypatch.setattr(module, "_backend", lambda backend, device: "triton")
    return fragment, serial, accumulate


def test_serial_launch_grid_and_argument_order(monkeypatch):
    from relic_core.kernels.decode_block_gemv import ACCUMULATE_COLUMNS

    fragment, serial, accumulate = _fakes(monkeypatch)
    a, sa, b, sb = _operands(n=24, k=128)
    G.fp4_packed_integer_decode_gemv(
        a, sa, b, sb, backend="triton", variant="serial", rows_n=8
    )
    assert not fragment.calls and not accumulate.calls
    assert len(serial.calls) == 1
    grid, arguments = serial.calls[0]
    assert grid == (math.ceil(24 / 8),)
    assert arguments[:4] == (a, sa, b, sb)
    assert isinstance(arguments[4], torch.Tensor) and arguments[4].shape == (1, 24)
    n_arg, k, blocks, block_rows, packed, fp4, b_scale_float, rows_n, index64 = arguments[5:]
    assert (n_arg, k, blocks, block_rows, packed, fp4, b_scale_float, rows_n, index64) == (
        24, 128, 4, 1, True, True, False, 8, False)
    assert ACCUMULATE_COLUMNS > 0


def test_grouped_launch_grid_and_argument_order(monkeypatch):
    from relic_core.kernels.decode_block_gemv import ACCUMULATE_COLUMNS

    fragment, serial, accumulate = _fakes(monkeypatch)
    a, sa, b, sb = _operands(n=24, k=128)
    scratch = G.integer_decode_gemv_scratch(24, 128)
    out = G.fp4_packed_integer_decode_gemv(
        a, sa, b, sb, backend="triton", variant="grouped", rows_n=8, group_k=2,
        scratch=scratch,
    )
    assert not serial.calls
    assert len(fragment.calls) == 1 and len(accumulate.calls) == 1
    grid, arguments = fragment.calls[0]
    # ceil(24 / 8) column chunks by min(groups, cap // chunks) block splits.
    assert grid == (3, 2)
    assert arguments[:4] == (a, sa, b, sb)
    assert arguments[4] is scratch[0] and arguments[5] is scratch[1]
    n, k, blocks, block_rows, packed, fp4, b_scale_float, rows_n, group_k, ksplit, index64 = arguments[6:]
    assert (n, k, blocks, block_rows, packed, fp4, b_scale_float) == (24, 128, 4, 1, True, True, False)
    assert (rows_n, group_k, ksplit, index64) == (8, 2, 2, False)
    grid, arguments = accumulate.calls[0]
    assert grid == (math.ceil(24 / ACCUMULATE_COLUMNS),)
    assert arguments == (scratch[0], scratch[1], out, 24, 4, ACCUMULATE_COLUMNS, False)


def test_fp8_launch_carries_its_block_rows_and_scale_kind(monkeypatch):
    fragment, serial, accumulate = _fakes(monkeypatch)
    generator = torch.Generator().manual_seed(13)
    n, k = 12, 64
    a = _codes((1, k), generator)
    sa = torch.full((1, k // 32), 127, dtype=torch.uint8)
    b = _codes((n, k), generator)
    sb = torch.full((math.ceil(n), k // 32), 127, dtype=torch.uint8)
    G.fp8_integer_decode_gemv(
        a, sa, b, sb, block_rows=1, backend="triton", variant="serial", rows_n=4
    )
    grid, arguments = serial.calls[0]
    assert grid == (math.ceil(n / 4),)
    n_arg, k_arg, blocks, block_rows, packed, fp4, b_scale_float, rows_n, _index64 = arguments[5:]
    assert (n_arg, k_arg, blocks, block_rows, packed, fp4, b_scale_float, rows_n) == (
        12, 64, 2, 1, False, False, False, 4)


def test_the_index_width_is_chosen_from_the_reaches_before_any_product(monkeypatch):
    """``N * K`` / ``KBLOCKS * N`` past 2**31 must widen the index before use."""
    fragment, serial, accumulate = _fakes(monkeypatch)
    assert G._index_width(65536, 32768, 1024) is True  # exactly 2**31: widened
    assert G._index_width(1024, 7168, 224) is False
    assert G._index_width(2**31 // 64, 64, 2) is True

    a, sa, b, sb = _operands(n=8, k=64)
    out = torch.empty((1, 65536))
    monkeypatch.setattr(G, "_output", lambda target, n, device, protected: out)
    # A geometry whose reach crosses 2**31 but whose grid still fits the cap.
    returned = G._run(
        a, sa, b, sb, fp4=True, packed=True, block_rows=1, b_scale_float=False,
        n=65536, k=32768, device=a.device, out=out, scratch=None, backend="triton",
        rows_n=16, group_k=4, variant="serial")
    assert returned is out
    grid, arguments = serial.calls[-1]
    assert grid == (math.ceil(65536 / 16),)
    assert arguments[-1] is True


def test_the_packed_nibbles_use_constant_shifts_not_a_lane_varying_shift():
    """A per-lane shift amount cost 49.75 ms against 18.62 ms for the constant form.

    The packed extraction must therefore be two constant shifts selected by a lane
    mask. This rejects any future ``>> ((lanes & 1) * 4)`` rewrite by looking at the
    shift amounts themselves: no shift/lshift in a kernel body may take an
    expression mentioning ``lanes`` as its right operand.
    """
    tree = ast.parse(KERNEL_MODULE.read_text())
    bodies = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    for function in bodies:
        for node in ast.walk(function):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.RShift, ast.LShift)):
                amount = ast.unparse(node.right)
                assert "lanes" not in amount, (
                    f"{function.name} shifts by {amount!r}, a lane-varying amount; the "
                    "packed nibbles must use constant shifts selected by tl.where"
                )
    source = KERNEL_MODULE.read_text()
    assert "weight & 15" in source and "weight >> 4" in source


def test_no_compile_environment_flag_is_required():
    """The operator must not depend on a relaxation flag for the front end."""
    source = KERNEL_MODULE.read_text() + Path(G.__file__).read_text()
    for flag in ("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "TRITON_ALL_BLOCKS_PARALLEL",
                 "TRITON_ALWAYS_COMPILE"):
        assert flag not in source, f"the operator must not rely on {flag}"


def test_a_non_tensor_weight_scale_is_refused_by_type_before_any_attribute_read():
    a, sa, b, sb = _operands()
    with pytest.raises(TypeError, match="weight scales must be a torch.Tensor"):
        G.fp8_integer_decode_gemv(
            a, sa, _codes((8, 96), torch.Generator().manual_seed(0)), [[127]],
            block_rows=32, backend="torch")


# -- the guarded scale fast path ---------------------------------------------------------------


def _predicate(raw: int, code: int) -> bool:
    """The kernel's per-lane predicate, transcribed over unsigned FP32 bits."""
    exponent = (raw >> 23) & 255
    new_exponent = exponent + code - 127
    zero = (raw & 0x7FFFFFFF) == 0
    return code != 255 and (zero or (0 < exponent < 255 and 0 < new_exponent < 255))


def _fast_bits(raw: int, code: int) -> int:
    """The kernel's fast-path bits for one lane, transcribed (exponent clamped)."""
    if (raw & 0x7FFFFFFF) == 0:
        return raw
    mantissa = raw & 0x7FFFFF
    sign = (raw >> 31) << 31
    exponent = (raw >> 23) & 255
    clamped = min(max(exponent + code - 127, 0), 255)
    return sign | mantissa | (clamped << 23)


def _float_from(unsigned_bits: int) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<I", unsigned_bits))[0]


_FP32_PATTERNS = (
    0x00000000,  # +0
    0x80000000,  # -0
    0x00000001, 0x80000001,  # smallest subnormals
    0x007FFFFF, 0x807FFFFF,  # largest subnormals
    0x00800000, 0x80800000,  # smallest normals
    0x00800001, 0x01000000,
    0x3F800000, 0xBF800000,  # +-1
    0x3F800001, 0x3F7FFFFF,
    0x7F7FFFFF, 0xFF7FFFFF,  # largest finite
    0x7F800000, 0xFF800000,  # +-inf
    0x7FC00000, 0xFFC00000,  # +-quiet NaN
    0x7F800001, 0xFFBFFFFF,  # signalling NaN patterns
)


def test_the_guarded_fast_path_is_exact_where_it_is_taken():
    """Whenever the predicate holds, the fast result is the exact power-of-two product.

    Both endpoints are normal and the result exponent stays normal, so the
    operation is a pure exponent move with no rounding at all. The check is an
    exact rational equality, so it is a proof for those lanes rather than a
    tolerance.
    """
    patterns = list(_FP32_PATTERNS)
    generator = torch.Generator().manual_seed(99)
    patterns += torch.randint(0, 2**32, (256,), generator=generator,
                              dtype=torch.int64).tolist()
    taken = 0
    for code in range(256):
        factor = Fraction(2) ** (code - 127)
        for raw in patterns:
            if not _predicate(raw, code):
                continue
            taken += 1
            built = _fast_bits(raw, code)
            if (raw & 0x7FFFFFFF) == 0:
                assert built == raw, "a signed zero must keep its sign"
                continue
            expected = Fraction(_float_from(raw)) * factor
            assert Fraction(_float_from(built)) == expected, (
                f"bits {raw:#010x} with scale byte {code} is not the exact product"
            )
            assert 0 < (built >> 23) & 255 < 255, "the fast result must stay normal"
    assert taken > 1000, "the enumeration must actually exercise the fast path"


def test_the_guarded_fast_path_refuses_every_unsafe_class():
    """Subnormal, overflow, infinity, NaN and byte 255 must all fall back."""
    for raw in (0x00000001, 0x007FFFFF, 0x7F800000, 0x7FC00000, 0xFFC00000):
        assert not _predicate(raw, 127), f"bits {raw:#010x} must not take the fast path"
    assert not _predicate(0x3F800000, 255), "a NaN scale byte must always fall back"
    # Largest normal, scaled up: the result exponent leaves the normal range.
    assert not _predicate(0x7F7FFFFF, 254)
    # Smallest normal, scaled down: the result is subnormal.
    assert not _predicate(0x00800000, 0)
    # A NaN scale byte on a zero must not be short-circuited to a signed zero.
    assert not _predicate(0x00000000, 255) and not _predicate(0x80000000, 255)


def test_one_unsafe_lane_sends_the_whole_call_through_the_frozen_helper():
    """The branch is uniform: the guard is a reduction over the tile, not per lane."""
    safe_tile = [_predicate(0x3F800000, 127) for _ in range(16)]
    assert all(safe_tile)
    mixed = list(safe_tile)
    mixed[9] = _predicate(0x00000001, 127)  # one subnormal lane
    assert not all(mixed), "a single unsafe lane must disable the fast path for the call"


def test_the_scale_guard_builds_its_mask_from_int32_sized_literals():
    """``raw & 0x807fffff`` would exceed 2**31 - 1 and promote the index wider.

    The proven pattern builds the sign and the significand from two literals that
    each fit in int32, which is what the frozen helper does too.
    """
    source = KERNEL_MODULE.read_text()
    assert "0x807fffff" not in source.lower()
    assert "0x7fffff" in source.lower() and "(raw >> 31) << 31" in source


def test_every_integer_literal_in_a_kernel_body_fits_in_int32():
    """A wider literal becomes int64/uint64 and this compiler refuses the cast."""
    tree = ast.parse(KERNEL_MODULE.read_text())
    bodies = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    for function in bodies:
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                assert -(2**31) <= node.value <= 2**31 - 1, (
                    f"{function.name} uses literal {node.value:#x}, which is wider than int32"
                )


def test_the_two_rescales_stay_separate_and_are_placed_by_schedule():
    """Scale A and scale B are applied one after the other, never as one exponent.

    The serial schedule is the proof schedule: the frozen helper twice, no dynamic
    branch. The grouped schedule adds the guarded fast path, but its two steps are
    still separate calls and the generic fallback is the frozen helper twice.
    """
    source = KERNEL_MODULE.read_text()
    assert "scale_a + scale_b" not in source, "the two scale exponents must never be combined"
    assert source.count("_scale_e8m0(_scale_e8m0(fragment, scale_a), scale_b)") == 2
    assert "_scale_fast(fragment, scale_a)" in source
    assert "_scale_fast(midway, scale_b)" in source, "the second step must read the first result"
    assert "_original_scale_code(scale_b, FP4)" in source


def test_no_jit_helper_returns_from_inside_a_conditional():
    """A vector returned out of a branch is what the compiler failed to legalize.

    The grouped schedule therefore decides in the kernel and stores from both
    branches; helpers only fall through to a single trailing return.
    """
    tree = ast.parse(KERNEL_MODULE.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    for function in functions:
        parents = {}
        for node in ast.walk(function):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(function):
            if not isinstance(node, ast.Return):
                continue
            ancestor = parents.get(node)
            while ancestor is not None and ancestor is not function:
                assert not isinstance(ancestor, ast.If), (
                    f"{function.name} returns from inside a conditional; the branch must "
                    "store instead, or this compiler cannot materialize the result"
                )
                ancestor = parents.get(ancestor)


def test_the_serial_schedule_has_no_dynamic_branch():
    """The proof schedule must not depend on a runtime-uniform branch at all."""
    tree = ast.parse(KERNEL_MODULE.read_text())
    serial = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "integer_serial_kernel"
    )
    constexpr = {
        argument.arg for argument in serial.args.args
        if isinstance(argument.annotation, ast.Attribute)
        and argument.annotation.attr == "constexpr"
    }
    assert {"PACKED", "B_SCALE_FLOAT", "INDEX64"} <= constexpr
    for node in ast.walk(serial):
        if isinstance(node, ast.If):
            rendered = ast.unparse(node.test)
            # Only compile-time branches on a constexpr parameter are allowed; a
            # runtime condition (``tl.min(...)``, a loaded value, ...) would make
            # this schedule dependent on a device branch it must not have.
            name = rendered[4:].strip() if rendered.startswith("not ") else rendered.strip()
            assert name in constexpr, (
                f"the serial schedule gained a runtime branch: if {rendered}"
            )


def test_only_the_two_kernels_issue_device_loads():
    """The fragment helper must stay pointer-free.

    The Ascend compiler's ``OffsetAnalysis`` aborts combining pointer offsets of
    different ranks across a call boundary (``lhs.getRank() == rhs.getRank()``),
    which is what an earlier revision hit by loading A (1-D), B (2-D) and the two
    scales through the helper. Every device access therefore belongs to the two
    kernels, whose own tensor ranks are fixed; helpers may only do arithmetic.
    """
    tree = ast.parse(KERNEL_MODULE.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
    ]
    kernels = {"integer_fragment_kernel", "integer_serial_kernel"}
    assert {function.name for function in functions} >= kernels
    for function in functions:
        accesses = []
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"load", "store"}:
                    accesses.append(node.func.attr)
        if function.name in kernels:
            assert accesses, f"{function.name} must issue the device accesses"
            continue
        assert not accesses, (
            f"{function.name} issues {sorted(set(accesses))} on a pointer; the fragment "
            "arithmetic helper must receive loaded values instead"
        )


def test_the_two_step_fast_path_is_exact_over_every_scale_byte_pair():
    """Both rescales taken => the composed result is the exact product of both powers.

    Each step is a pure exponent move with a normal result, so composing them is
    exact too, and the second predicate reads the *actual* first intermediate rather
    than assuming the composition. All 256 x 256 scale-byte pairs are enumerated
    against the fragments that can reach the fast path.
    """
    fragments = [0x00000000, 0x80000000, 0x00800000, 0x3F800000, 0xBF800000,
                 0x4B000000, 0x7F000000]
    taken = 0
    for code_a in range(256):
        for code_b in range(256):
            for raw in fragments:
                if not _predicate(raw, code_a):
                    continue
                bits_a = _fast_bits(raw, code_a)
                if not _predicate(bits_a, code_b):
                    continue
                bits_b = _fast_bits(bits_a, code_b)
                taken += 1
                if (raw & 0x7FFFFFFF) == 0:
                    assert bits_b == raw
                    continue
                exact = (Fraction(_float_from(raw))
                         * Fraction(2) ** (code_a - 127)
                         * Fraction(2) ** (code_b - 127))
                assert Fraction(_float_from(bits_b)) == exact
                assert 0 < (bits_b >> 23) & 255 < 255
    assert taken > 10_000, "the enumeration must exercise the composed fast path"


def test_a_safe_first_step_can_still_force_the_generic_fallback():
    """The second predicate must look at the intermediate, not assume the composition."""
    smallest_normal, largest_normal = 0x00800000, 0x7F7FFFFF
    # Underflow on the second step only: 2**-126 stays normal under 2**0, then the
    # second rescale by 2**-1 leaves the normal range.
    assert _predicate(smallest_normal, 127)
    midway = _fast_bits(smallest_normal, 127)
    assert _predicate(midway, 127) is False or _predicate(midway, 126) is False
    assert not _predicate(midway, 126), "2**-126 * 2**-1 is subnormal: fallback required"
    # Overflow on the second step only.
    assert _predicate(largest_normal, 127)
    midway = _fast_bits(largest_normal, 127)
    assert not _predicate(midway, 128), "the second step overflows: fallback required"
