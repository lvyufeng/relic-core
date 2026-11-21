"""Fused routing/operand-layout kernels for the opt-in original-FP4 MoE path.

All expert IDs are consumed on device. The weight kernel combines source
selection, low-nibble-first I8x2 decoding, original half-scale application and
transposed Cube-layout writes. It never changes checkpoint precision.
"""
import triton
import triton.language as tl

from ._ascend_fast_fused_fp8_activation_decode_kernels import (
    _e4m3_byte_from_bits_fast, _e4m3_e8m0_bf16_bits_fast,
    _nonfinite_bits, _pow2_bits, _scale_exponent_from_bits,
)


@triton.jit
def clamp_gate_up_kernel(X, UP, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                         LIMIT: tl.constexpr, TILE: tl.constexpr, SEPARATE: tl.constexpr):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = offset < ROWS * 2 * WIDTH
    gate = (offset % (2 * WIDTH)) < WIDTH
    if SEPARATE:
        source = (offset // (2 * WIDTH)) * WIDTH + offset % WIDTH
        left = tl.load(X + source, active & gate, other=0).to(tl.float32)
        right = tl.load(UP + source, active & ~gate, other=0).to(tl.float32)
        value = tl.where(gate, left, right)
    else:
        value = tl.load(X + offset, active, other=0).to(tl.float32)
    if LIMIT > 0:
        value = tl.where(value > LIMIT, LIMIT, value)
        value = tl.where((~gate) & (value < -LIMIT), -LIMIT, value)
    # F.silu(-inf) is NaN at the released elementwise boundary, not zero.
    value = tl.where(gate & (value == -float("inf")), float("nan"), value)
    tl.store(OUT + offset, value, active)


@triton.jit
def weighted_fp8_quant_decode_kernel(X, GATES, OUT, CODES, SCALES,
                                     NUM_BLOCKS: tl.constexpr, ROW_BLOCKS: tl.constexpr,
                                     BLOCKS: tl.constexpr, FLOOR: tl.constexpr,
                                     INV_FP8_MAX: tl.constexpr):
    for program in range(tl.program_id(0), tl.cdiv(NUM_BLOCKS, BLOCKS), tl.num_programs(0)):
        block = program * BLOCKS + tl.arange(0, BLOCKS)
        active = block < NUM_BLOCKS
        offset = block[:, None] * 32 + tl.arange(0, 32)[None, :]
        routing_weight = tl.load(GATES + block // ROW_BLOCKS, active, other=0).to(tl.float32)
        values = tl.load(X + offset, active[:, None], other=0).to(tl.float32)
        # Match the released FP32 routing multiply, then its BF16 boundary
        # BEFORE computing amax or quantization codes.
        product = values * routing_weight[:, None]
        raw = product.to(tl.int32, bitcast=True)
        rounding = 0x7FFF + ((raw >> 16) & 1)
        rounded = (raw + rounding) & -65536
        magnitude_raw = raw & 0x7FFFFFFF
        nonfinite_raw = magnitude_raw >= 0x7F800000
        exceptional = tl.where(magnitude_raw > 0x7F800000,
                               (raw & -2147483648) | 0x7FC00000, raw & -65536)
        # Explicit IEEE BF16 RNE bits make the original narrowing boundary
        # load bearing even if a backend optimizes a float cast round trip.
        values = tl.where(nonfinite_raw, exceptional, rounded).to(tl.float32, bitcast=True)
        magnitude = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        nonfinite = tl.max(_nonfinite_bits(magnitude), axis=1) != 0
        amax = tl.max(magnitude, axis=1).to(tl.float32, bitcast=True)
        scaled = tl.maximum(amax, FLOOR) * INV_FP8_MAX
        exponent = _scale_exponent_from_bits(scaled.to(tl.int32, bitcast=True))
        inverse = _pow2_bits(-exponent)[:, None].to(tl.float32, bitcast=True)
        normalized = tl.minimum(tl.maximum(values * inverse, -448.0), 448.0)
        code = _e4m3_byte_from_bits_fast(normalized.to(tl.int32, bitcast=True))
        code = tl.where(nonfinite[:, None], 0x7F, code)
        scale = tl.where(nonfinite, 255, exponent + 127)
        decoded = tl.cast(_e4m3_e8m0_bf16_bits_fast(code, scale[:, None]), tl.bfloat16, bitcast=True)
        tl.store(OUT + offset, decoded, active[:, None])
        tl.store(CODES + offset, code.to(tl.uint8), active[:, None])
        tl.store(SCALES + block, scale.to(tl.uint8), active)


@triton.jit
def _certified_e2m1_bits(code, scale):
    magnitude = code & 7
    exponent_delta = tl.where(magnitude >= 6, 2,
                              tl.where(magnitude >= 4, 1, tl.where(magnitude >= 2, 0, -1)))
    exponent = scale + exponent_delta
    mantissa = tl.where((magnitude == 3) | (magnitude == 5) | (magnitude == 7), 0x40, 0)
    bits = ((code & 8) << 12) | (exponent << 7) | mantissa
    bits = tl.where(magnitude == 0, 0, bits)
    # Load-time certification is 2..252. An unexpected source outside that
    # range still poisons the operand instead of silently rounding it.
    bits = tl.where((scale >= 2) & (scale <= 252), bits, 0x7FC0)
    return bits.to(tl.uint16)


@triton.jit
def _certified_e2m1_bf16(code, scale):
    return tl.cast(_certified_e2m1_bits(code, scale), tl.bfloat16, bitcast=True)


@triton.jit
def route_owned_kernel(IDS, WEIGHTS, SELECTED, GATES, GROUP_LIST, ERROR,
                       TOPK: tl.constexpr, LANES: tl.constexpr,
                       EXPERTS: tl.constexpr, RANK: tl.constexpr, WORLD: tl.constexpr):
    lane = tl.arange(0, LANES)
    live = lane < TOPK
    ids = tl.load(IDS + lane, live, other=-1)
    valid = live & (ids >= 0) & (ids < EXPERTS)
    duplicate = (ids[:, None] == ids[None, :]) & live[:, None] & live[None, :]
    duplicate = duplicate & (lane[:, None] != lane[None, :])
    bad = (tl.sum((live & ~valid).to(tl.int32), axis=0) > 0)
    bad = bad | (tl.sum(tl.sum(duplicate.to(tl.int32), axis=0), axis=0) > 0)
    # One program owns this latch, and the workspace is single-stream. Avoid
    # CANN's atomic initialization path, which performs a host copy in capture.
    previous_error = tl.load(ERROR)
    tl.store(ERROR, previous_error | bad.to(tl.int32))
    owned = valid & ((ids % WORLD) == RANK) & ~bad
    # This CANN build has no linked 1-D int32 tl.sort helper. Rank the at-most
    # eight selected IDs with pairwise comparisons instead; no CPU sort/read.
    less = (ids[None, :] < ids[:, None]) & owned[None, :]
    position = tl.sum(less.to(tl.int32), axis=1)
    count = tl.sum(owned.to(tl.int32), axis=0)
    match = (lane[:, None] == position[None, :]) & owned[None, :]
    ordered = tl.min(tl.where(match, ids[None, :], EXPERTS).to(tl.int32), axis=1)
    selected_slots = tl.min(tl.where(match, lane[None, :], LANES), axis=1)
    selected_weights = tl.load(WEIGHTS + selected_slots, lane < count, other=0).to(tl.float32)
    tl.store(SELECTED + lane, tl.where(lane < count, ordered, -1), live)
    tl.store(GATES + lane, tl.where(lane < count, selected_weights, 0.0), live)
    tl.store(GROUP_LIST + lane, tl.minimum(lane + 1, count).to(tl.int64), live)
    # The final zero-weight group consumes the padded activation rows, keeping
    # grouped-matmul's total M static even when this rank owns no selected ID.
    tl.store(GROUP_LIST + TOPK, TOPK)




@triton.jit
def prepare_selected_pairs_kernel(B, SB, C, SC, SELECTED, OUT,
                                  K: tl.constexpr, N: tl.constexpr,
                                  WORLD: tl.constexpr, TOPK: tl.constexpr,
                                  TILE: tl.constexpr, GATE_UP: tl.constexpr,
                                  INDEX64: tl.constexpr):
    """One original byte read and one uint32 BF16 pair write, with its original scale."""
    program = tl.program_id(0)
    group = tl.program_id(1)
    if INDEX64:
        program = program.to(tl.int64)
        group = group.to(tl.int64)
    expert = tl.load(SELECTED + group, group < TOPK, other=-1)
    lane = tl.arange(0, TILE)
    KP: tl.constexpr = K // 2
    if GATE_UP:
        SOURCE_N: tl.constexpr = N // 2
    else:
        SOURCE_N: tl.constexpr = N
    if (expert >= 0) | (group == TOPK):
        local = tl.maximum(expert, 0) // WORLD
        if INDEX64:
            local = local.to(tl.int64)
        for tile in range(program, N * tl.cdiv(KP, TILE), tl.num_programs(0)):
            column = tile // tl.cdiv(KP, TILE)
            pair = (tile % tl.cdiv(KP, TILE)) * TILE + lane
            active = pair < KP
            source_column = tl.where(column < SOURCE_N, column, column - SOURCE_N)
            byte_offset = local * SOURCE_N * KP + source_column * KP + pair
            scale_offset = local * SOURCE_N * (K // 32) + source_column * (K // 32) + pair // 16
            if group == TOPK:
                result = tl.full((TILE,), 0, tl.uint32)
            else:
                if GATE_UP:
                    first = column < SOURCE_N
                    byte_b = tl.load(B + byte_offset, active & first, other=0).to(tl.int32)
                    byte_c = tl.load(C + byte_offset, active & ~first, other=0).to(tl.int32)
                    scale_b = tl.load(SB + scale_offset, active & first, other=127).to(tl.int32)
                    scale_c = tl.load(SC + scale_offset, active & ~first, other=127).to(tl.int32)
                    byte = tl.where(first, byte_b, byte_c)
                    scale = tl.where(first, scale_b, scale_c)
                else:
                    byte = tl.load(B + byte_offset, active, other=0).to(tl.int32)
                    scale = tl.load(SB + scale_offset, active, other=127).to(tl.int32)
                # This Ascend backend sign-extends uint16->uint32. Mask after
                # widening, otherwise a negative low value fills the high half
                # with 0xffff and corrupts the second BF16 operand.
                low = (_certified_e2m1_bits(byte & 15, scale).to(tl.int32) & 65535).to(tl.uint32)
                high = (_certified_e2m1_bits(byte >> 4, scale).to(tl.int32) & 65535).to(tl.uint32)
                result = low | (high << 16)
            tl.store(OUT + group * N * KP + column * KP + pair, result, active)


@triton.jit
def _e2m1_x2_integer(code):
    magnitude = code & 7
    magnitude = tl.where(magnitude == 7, 12,
                         tl.where(magnitude >= 4, 2 * magnitude - 4, magnitude))
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def prepare_selected_i8_pairs_kernel(B, C, SELECTED, OUT,
                                     K: tl.constexpr, N: tl.constexpr,
                                     WORLD: tl.constexpr, TILE: tl.constexpr,
                                     GATE_UP: tl.constexpr, INDEX64: tl.constexpr):
    program, group = tl.program_id(0), tl.program_id(1)
    if INDEX64:
        program, group = program.to(tl.int64), group.to(tl.int64)
    expert = tl.load(SELECTED + group)
    local = tl.maximum(expert, 0) // WORLD
    if INDEX64:
        local = local.to(tl.int64)
    lane = tl.arange(0, TILE)
    SOURCE_N: tl.constexpr = N // 2 if GATE_UP else N
    for tile in range(program, N * tl.cdiv(K // 2, TILE), tl.num_programs(0)):
        column = tile // tl.cdiv(K // 2, TILE)
        pair = (tile % tl.cdiv(K // 2, TILE)) * TILE + lane
        active = pair < K // 2
        source_column = tl.where(column < SOURCE_N, column, column - SOURCE_N)
        source = local * SOURCE_N * (K // 2) + source_column * (K // 2) + pair
        if expert >= 0:
            if GATE_UP:
                first = column < SOURCE_N
                left = tl.load(B + source, active & first, other=0).to(tl.int32)
                right = tl.load(C + source, active & ~first, other=0).to(tl.int32)
                byte = tl.where(first, left, right)
            else:
                byte = tl.load(B + source, active, other=0).to(tl.int32)
            low = _e2m1_x2_integer(byte & 15) & 255
            high = _e2m1_x2_integer(byte >> 4) & 255
            result = (low | (high << 8)).to(tl.uint16)
        else:
            result = tl.full((TILE,), 0, tl.uint16)
        tl.store(OUT + group * N * (K // 2) + column * (K // 2) + pair, result, active)


@triton.jit
def prepare_selected_half_scales_kernel(SB, SC, SELECTED, OUT,
                                        K: tl.constexpr, N: tl.constexpr,
                                        WORLD: tl.constexpr, TILE: tl.constexpr,
                                        GATE_UP: tl.constexpr, INDEX64: tl.constexpr):
    program, group = tl.program_id(0), tl.program_id(1)
    if INDEX64:
        program, group = program.to(tl.int64), group.to(tl.int64)
    expert = tl.load(SELECTED + group)
    local = tl.maximum(expert, 0) // WORLD
    if INDEX64:
        local = local.to(tl.int64)
    SOURCE_N: tl.constexpr = N // 2 if GATE_UP else N
    SOURCE_SIZE: tl.constexpr = SOURCE_N * (K // 32)
    lane = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(N * (K // 32), TILE), tl.num_programs(0)):
        offset = tile * TILE + lane
        active = offset < N * (K // 32)
        if expert >= 0:
            source_offset = tl.where(offset < SOURCE_SIZE, offset, offset - SOURCE_SIZE)
            source = local * SOURCE_SIZE + source_offset
            if GATE_UP:
                first = offset < SOURCE_SIZE
                left = tl.load(SB + source, active & first, other=127).to(tl.int32)
                right = tl.load(SC + source, active & ~first, other=127).to(tl.int32)
                scale = tl.where(first, left, right)
            else:
                scale = tl.load(SB + source, active, other=127).to(tl.int32)
            bits = tl.where((scale >= 2) & (scale <= 252), (scale - 1) << 7, 0x7FC0)
        else:
            bits = tl.full((TILE,), 0x3F80, tl.int32)
        tl.store(OUT + group * N * (K // 32) + offset, bits.to(tl.uint16), active)




@triton.jit
def replicate_selected_activation_kernel(X, SELECTED, OUT,
                                         K: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(1)
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    selected = tl.load(SELECTED + group) >= 0
    value = tl.load(X + offset, (offset < K) & selected, other=0)
    tl.store(OUT + group * K + offset, value, offset < K)


@triton.jit
def ordered_combine_kernel(CONTRIBUTIONS, SELECTED, ERROR, OUT,
                           DIM: tl.constexpr, TOPK: tl.constexpr, TILE: tl.constexpr):
    column = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = column < DIM
    value = tl.full((TILE,), 0.0, tl.float32)
    # Metadata was sorted by global expert ID before either projection. Narrow
    # each expert to BF16, widen it, then add in exactly that ascending order.
    for group in range(TOPK):
        expert = tl.load(SELECTED + group)
        if expert >= 0:
            contribution = tl.load(CONTRIBUTIONS + group * DIM + column, active, other=0)
            value = value + contribution.to(tl.float32)
    invalid = tl.load(ERROR) != 0
    tl.store(OUT + column, tl.where(invalid, float("nan"), value), active)
