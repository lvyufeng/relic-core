"""Original online-unpack consumed by a device-selected native Cube tile.

This experiment retains packed E2M1 and raw E8M0 in the existing lossless
16-row tile layout. It materializes neither a selected INT8 matrix nor a BF16
weight matrix in global memory. A tile's exact I8x2/half-scale value is assembled
locally before its native Cube dot. The per-K-tile Cube reduction differs from
the exact window-13 default and is not installed by importing this module.
"""
from __future__ import annotations

import math

import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend
from .tile_major_packed_gemv import pack_tile_major, unpack_tile_major
from .unpack import fp4_int8_scale, unpack_fp4_to_int8

CONTRACT = "original_fp4_tile_local_unpack_native_cube_k_tiles_ascending_fp32_combine"
TILE_ROWS = 16
BLOCK_K = 32


def pack_expert_tile_bank(codes: torch.Tensor, scales: torch.Tensor):
    """Return a lossless tiled byte bank, preserving the original scale bytes.

    Intended for load-time staging, one projection/layer at a time. The caller
    must replace rather than retain a second resident row-major weight bank.
    """
    if codes.ndim != 3 or scales.ndim != 3 or codes.shape[0] != scales.shape[0]:
        raise ValueError("expert banks must be [owned,N,K/2] and [owned,N,K/32]")
    if codes.shape[0] <= 0:
        raise ValueError("tile-bank packing needs at least one owned expert")
    pairs = [pack_tile_major(code, scale) for code, scale in zip(codes, scales)]
    return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])


def _validate(x, b, s, selected, *, world, n_output, group_k, out):
    if isinstance(world, bool) or not isinstance(world, int) or world <= 0:
        raise ValueError("world must be a positive integer")
    if group_k not in (1, 2, 4) or isinstance(group_k, bool):
        raise ValueError("group_k must be 1, 2 or 4 original K32 blocks")
    if b.ndim != 5 or b.shape[-2:] != (16, 16) or not b.is_contiguous() or b.dtype != torch.uint8:
        raise ValueError("packed bank must be contiguous uint8 [owned,ntiles,kblocks,16,16]")
    owned, tiles, blocks = b.shape[:3]
    if min(owned, tiles, blocks) <= 0 or tuple(s.shape) != (owned, tiles, blocks, 16):
        raise ValueError("scale bank must match the original tiled weight geometry")
    if s.dtype != torch.uint8 or not s.is_contiguous():
        raise ValueError("the raw original E8M0 bank must be contiguous uint8")
    n = tiles * TILE_ROWS if n_output is None else n_output
    if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= tiles * TILE_ROWS:
        raise ValueError("n_output must be within the physical tile extent")
    if selected.ndim != 1 or selected.dtype != torch.int32 or not selected.is_contiguous():
        raise ValueError("selected IDs must be a contiguous int32 device vector")
    groups = selected.numel()
    if not 0 < groups <= 8 or tiles * groups > 65535:
        raise ValueError("at most eight selected slots and a bounded launch grid are supported")
    if x.ndim != 2 or x.shape[1] != blocks * BLOCK_K or x.shape[0] not in (1, groups):
        raise ValueError("decoded activation must be [1,K] or [selected_slots,K]")
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("activation must preserve the original contiguous BF16 decode boundary")
    if any(t.device != x.device for t in (b, s, selected)):
        raise ValueError("all tile-consumer operands must share a device")
    if any(t.requires_grad for t in (x, b, s, selected)):
        raise ValueError("the online tile consumer is inference only")
    if out is not None:
        if tuple(out.shape) != (groups, n) or out.dtype != torch.bfloat16 or out.device != x.device or not out.is_contiguous():
            raise ValueError("out must be contiguous BF16 [selected_slots,N] on the operand device")
        if any(out.untyped_storage()._cdata == t.untyped_storage()._cdata for t in (x, b, s, selected)):
            raise ValueError("output must not alias an input bank")
    return owned, tiles, blocks, groups, n


@torch.no_grad()
def selected_tile_cube_fp4_gemv(x, b_tile, s_tile, selected, *, world,
                                n_output=None, group_k=2, backend="auto",
                                certified_bf16_operands=False, out=None):
    """Consume selected original weight tiles without global expanded-weight scratch.

    ``x`` is the decoded original FP8 operand, not unquantized model input.
    Negative sentinel -1 is an empty slot; invalid source indices poison the
    corresponding output rather than accessing outside the bank. The owning
    device router must check duplicates and cyclic-rank membership beforehand.
    Certification of E8M0 2..252 is a load-time obligation, never a hot D2H guard.
    """
    owned, tiles, blocks, groups, n = _validate(x, b_tile, s_tile, selected,
                                               world=world, n_output=n_output,
                                               group_k=group_k, out=out)
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be auto, torch or triton")
    if out is None:
        out = torch.empty((groups, n), dtype=torch.bfloat16, device=x.device)
    if x.device.type == "cpu" and backend != "triton":
        for group, expert in enumerate(selected.tolist()):
            if expert == -1:
                out[group].zero_()
                continue
            local = expert // world
            if expert < 0 or not 0 <= local < owned:
                out[group].fill_(float("nan"))
                continue
            code, scale = unpack_tile_major(b_tile[local], s_tile[local], n_output=n)
            widened = unpack_fp4_to_int8(code, backend="torch").float()
            half = fp4_int8_scale(scale, backend="torch").repeat_interleave(BLOCK_K, dim=-1)
            decoded = (widened * half).to(torch.bfloat16)
            activation = x[0 if x.shape[0] == 1 else group]
            partials = torch.zeros(n, dtype=torch.float32)
            width = group_k * BLOCK_K
            for start in range(0, blocks * BLOCK_K, width):
                partials += (activation[start:start+width].float()[None] @ decoded[:, start:start+width].float().t())[0]
            out[group].copy_(partials.to(torch.bfloat16))
        return out
    if backend == "torch" or x.device.type != "npu" or not supports_triton_ascend():
        raise RuntimeError("the native tile consumer requires a Triton Ascend NPU")
    if certified_bf16_operands is not True:
        raise ValueError("certify exact BF16 weight operands at load time before native Cube")
    from ._ascend_online_tile_cube_kernels import selected_tile_cube_kernel

    with _npu_launch(x):
        selected_tile_cube_kernel[(tiles, groups)](
            x, b_tile, s_tile, selected, out, n, tiles, blocks, owned, world,
            x.shape[0] == 1, group_k,
            max(b_tile.numel(), s_tile.numel(), out.numel()) >= 2**31,
        )
    return out
