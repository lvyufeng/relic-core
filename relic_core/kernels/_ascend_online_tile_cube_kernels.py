"""Device-selected tile consumers for original packed E2M1/E8M0 weights.

Unlike whole-matrix staging, one program reads only its output tile and decodes
inside the same kernel that submits a native Cube dot. The resident tile-major
layout is a lossless byte permutation, not FRACTAL_NZ metadata or quantization.
"""
import triton
import triton.language as tl

from ._ascend_device_selected_moe_kernels import _certified_e2m1_bits
from ._ascend_native_cube_block_kernels import _bfloat16_tile


@triton.jit
def selected_tile_cube_kernel(X, BT, ST, SELECTED, OUT,
                              N: tl.constexpr, NTILES: tl.constexpr, KBLOCKS: tl.constexpr,
                              OWNED: tl.constexpr, WORLD: tl.constexpr,
                              X_SHARED: tl.constexpr, GROUP_K: tl.constexpr,
                              INDEX64: tl.constexpr):
    tile, group = tl.program_id(0), tl.program_id(1)
    expert = tl.load(SELECTED + group)
    local = tl.maximum(expert, 0) // WORLD
    if INDEX64:
        tile = tile.to(tl.int64)
        group = group.to(tl.int64)
        local = local.to(tl.int64)
    columns = tile * 16 + tl.arange(0, 16)
    rows = tl.arange(0, 16)
    k_lane = tl.arange(0, GROUP_K * 32)
    n_lane = tl.arange(0, 16)
    value = tl.zeros((16,), tl.float32)
    valid = (expert >= 0) & (local < OWNED)
    if valid:
        b_base = (local * NTILES + tile) * KBLOCKS * 256
        s_base = (local * NTILES + tile) * KBLOCKS * 16
        x_base = tl.where(X_SHARED, 0, group * KBLOCKS * 32)
        for block_base in range(0, KBLOCKS, GROUP_K):
            k = block_base * 32 + k_lane
            k_active = k < KBLOCKS * 32
            activation = tl.load(X + x_base + k, k_active, other=0)
            a = tl.where(rows[:, None] == 0, activation[None, :], 0).to(tl.bfloat16)
            block = block_base + k_lane // 32
            # N-major loads keep the adjacent nibble pairs together; transpose
            # only the decoded local tile, not a whole selected weight matrix.
            byte_offset = b_base + block[None, :] * 256 + n_lane[:, None] * 16 + (k_lane % 32 // 2)[None, :]
            scale_offset = s_base + block[None, :] * 16 + n_lane[:, None]
            byte = tl.load(BT + byte_offset, k_active[None, :], other=0).to(tl.int32)
            scale = tl.load(ST + scale_offset, k_active[None, :], other=127).to(tl.int32)
            code = tl.where((k_lane & 1)[None, :] == 0, byte & 15, byte >> 4)
            bits = _certified_e2m1_bits(code, scale)
            b = _bfloat16_tile(bits).trans(1, 0)
            partial = tl.dot(a, b, out_dtype=tl.float32)
            first_row = tl.sum(tl.where(rows[:, None] == 0, partial, 0.0), axis=0)
            value = value + first_row
    invalid = (expert < -1) | ((expert >= 0) & (local >= OWNED))
    result = tl.where(invalid, float("nan"), value).to(tl.bfloat16)
    tl.store(OUT + group * N + columns, result, columns < N)
