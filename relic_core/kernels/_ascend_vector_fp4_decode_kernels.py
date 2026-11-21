"""Ascend kernels that build the exact original FP4 INT8-times-two pair from bits.

Two ways to avoid a per-element global lookup, both producing the frozen
widening's bytes:

``fp4_vector_arithmetic_kernel``
    computes each nibble's doubled magnitude with one integer expression and
    stores the 16-bit pair -- no table, no gather, one data load per tile;

``fp4_vector_lut_kernel``
    builds the 16-entry signed table once per program in on-chip memory and reads
    it with a local ``tl.gather``, so the table never returns to global memory.

Neither kernel reads a scale, a zero point or any new grid: the input is the
released packed E2M1 bytes and nothing else. ``_code_x2`` is the single
definition both kernels share, so the arithmetic and the table cannot drift.
"""
import triton
import triton.language as tl

#: Most rows in the gathered 2-D tile. The local table is broadcast down this axis
#: and the index tile reshapes to ``(ROWS, TILE // ROWS)``, so a small tile uses
#: fewer rows to keep the inner width at 16 -- the narrowest gathered row this
#: backend is known to compile.
MAX_LUT_ROWS = 64
#: Narrowest column count the gathered tile keeps.
MIN_LUT_WIDTH = 16


@triton.jit
def _code_x2(code):
    """Twice the numeric value of one original E2M1 code, as a signed int32.

    E2M1 is 1/2/1 and twice its magnitudes are ``0, 1, 2, 3, 4, 6, 8, 12``.
    Codes ``0..3`` are already that value. Codes ``4..7`` are ``4 + 2 * (code & 3)``
    -- ``4, 6, 8, 10`` -- plus two more when both low bits are set, which turns the
    last one into ``12``; that is exactly the E2M1 ladder. The sign bit is E2M1's
    own, so the ``-0`` codes become integer zero here, the same folding the frozen
    widening performs (an INT8 cannot carry the sign of zero).
    """
    magnitude = code & 7
    scaled = 4 + ((magnitude & 3) << 1) + (((magnitude & (magnitude >> 1)) & 1) << 1)
    value = tl.where(magnitude >= 4, scaled, magnitude)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _pair(byte):
    """The INT8-times-two pair of one packed byte, low nibble in the low byte."""
    low = _code_x2(byte & 15)
    high = _code_x2(byte >> 4)
    return (low & 255) | ((high & 255) << 8)


@triton.jit
def _table():
    """The 16 doubled magnitudes as signed bytes, one per E2M1 code."""
    return _code_x2(tl.arange(0, 16)).to(tl.int8)


@triton.jit
def fp4_vector_arithmetic_kernel(A, OUT, SIZE: tl.constexpr, TILE: tl.constexpr,
                                 INDEX64: tl.constexpr):
    """One 16-bit pair store per packed byte, computed without any lookup."""
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        packed = tl.load(A + offset, active, other=0).to(tl.int32)
        tl.store(OUT + offset, _pair(packed).to(tl.uint16), active)


@triton.jit
def fp4_vector_lut_kernel(A, OUT, SIZE: tl.constexpr, TILE: tl.constexpr,
                          ROWS: tl.constexpr, INDEX64: tl.constexpr):
    """The same pair, read from a 16-entry table that stays in on-chip memory.

    The gather uses the shape this backend already compiles for a 16-wide table:
    a 2-D source whose rows are identical copies of the table, and an index tile of
    the same rank. Every row therefore gathers from the same 16 entries, and the
    row-major reshape keeps each byte next to the two output values it produces.
    """
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    table = tl.broadcast_to(_table()[None, :], (ROWS, 16))
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        packed = tl.load(A + offset, active, other=0).to(tl.int32)
        # A masked-out tail lane reads entry zero and is stored nowhere.
        low = tl.gather(table, tl.reshape(packed & 15, (ROWS, TILE // ROWS)), 1)
        high = tl.gather(table, tl.reshape(packed >> 4, (ROWS, TILE // ROWS)), 1)
        pair = (low.to(tl.int32) & 255) | ((high.to(tl.int32) & 255) << 8)
        tl.store(OUT + offset, tl.reshape(pair, (TILE,)).to(tl.uint16), active)