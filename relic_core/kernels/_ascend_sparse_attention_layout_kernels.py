"""Optional fused Ascend NPU gather/prepack for the sparse-attention operand layouts.

This module owns only the fused layout pass: it reads the original stored KV rows
through the original top-k indices and writes the two FP32 operand layouts the
unchanged ``torch.bmm`` arithmetic consumes. The pass is one launch for the
complete 64-key blocks plus, only when a tail exists, one exact-tail launch; the
tail launch writes the true width and never pads to a block.

The tile is deliberately conservative and EXPERIMENTAL: 16 gathered rows by 64
head-dimension lanes, with the true width a ``tl.constexpr`` (64 for the full
blocks, the exact tail width otherwise) so the compiler sees every loop bound.
The first prototype used a 32x128 tile and did not fit: the Ascend compiler
reported "ub overflow, requires 3146752 bits while 1572864 bits available"
(~393 KiB > 192 KiB), in part because the transposed store was strided. This
version materializes the transposed tile with ``tl.trans`` and stores it
contiguously. Whether the smaller tile fits is a property of the compiler, not of
this source: the root harness's Ascend compile is the only UB proof, and this
module makes no claim that it fits without that compile.

Indices follow the reference exactly: a negative entry is invalid and is gathered
from row 0 (then masked to -inf by the caller); positive entries must be in range
and are certified by the caller, because checking them here would need a device to
host copy and would break graph capture.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except Exception:  # pragma: no cover - the CPU-only host has no Triton
    triton = None
    tl = None
    _USE_TRITON = False

#: Source block of the reference arithmetic; fixed, not a scheduling knob.
BLOCK = 64
#: Program tile: 16 gathered rows by 64 head-dimension lanes (conservative, see module doc).
TILE_W = 16
TILE_D = 64
#: Head dimension of the released sparse-attention path.
HEAD_DIM = 512


if _USE_TRITON:

    @triton.jit
    def _gather_layouts_kernel(
        KV, IDX, STRAIGHT, TRANSPOSED,
        N, BASE,
        BLOCK: tl.constexpr, WIDTH: tl.constexpr,
        TILE_W: tl.constexpr, TILE_D: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Write one block's gathered rows into both FP32 layouts.

        ``WIDTH`` is a compile-time constant: 64 for a complete block, the exact
        tail width for the single tail launch. Grid: (blocks, WIDTH/TILE_W,
        HEAD_DIM/TILE_D). The transposed tile is materialized with ``tl.trans`` and
        stored contiguously so the compiler does not have to buffer a strided
        store; neither store leaves the tensor's true extent.
        """
        pid_b = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_d = tl.program_id(2)
        rows = pid_w * TILE_W + tl.arange(0, TILE_W)
        dims = pid_d * TILE_D + tl.arange(0, TILE_D)
        row_mask = rows < WIDTH
        dim_mask = dims < HEAD_DIM
        slot = BASE + pid_b * BLOCK + rows
        idx = tl.load(IDX + slot, mask=row_mask, other=0)
        # The reference clamps every negative index to row 0 and masks the slot
        # afterwards; the positive-index range is the caller's certificate.
        safe = tl.maximum(idx, 0)
        can_read = row_mask & (safe < N)
        values = tl.load(
            KV + safe[:, None] * HEAD_DIM + dims[None, :],
            mask=can_read[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        write_mask = row_mask[:, None] & dim_mask[None, :]
        tl.store(
            STRAIGHT + pid_b * BLOCK * HEAD_DIM + rows[:, None] * HEAD_DIM + dims[None, :],
            values,
            mask=write_mask,
        )
        # [TILE_W, TILE_D] -> [TILE_D, TILE_W]; the store below is contiguous along
        # the width axis of the transposed layout.
        transposed_values = tl.trans(values)
        tl.store(
            TRANSPOSED + pid_b * WIDTH * HEAD_DIM + dims[:, None] * WIDTH + rows[None, :],
            transposed_values,
            mask=dim_mask[:, None] & row_mask[None, :],
        )


def triton_available() -> bool:
    """Report whether this host exposes a Triton compiler for the fused pass."""
    return _USE_TRITON


def gather_layouts(
    kv,
    indices,
    straight_full,
    transposed_full,
    straight_tail,
    transposed_tail,
    *,
    n,
    nfull,
    tail,
):
    """Launch the fused gather/prepack and return nothing (layouts are written in place).

    The launch is on the caller's current stream, performs no device to host copy,
    and is safe to record in a graph: the only host values are shapes and the
    certified extent. ``n`` is the stored KV row count used for the memory-bound
    mask; ``nfull`` full blocks precede an optional ``tail`` of exact width. The
    tail width is a distinct compile-time specialization, so it compiles on its
    own instead of sharing a padded 64-wide shape.
    """
    if not _USE_TRITON:
        raise RuntimeError("the fused sparse-attention layout needs a Triton Ascend compiler")
    if nfull:
        _gather_layouts_kernel[(nfull, triton.cdiv(BLOCK, TILE_W), triton.cdiv(HEAD_DIM, TILE_D))](
            kv, indices, straight_full, transposed_full, n, 0,
            BLOCK=BLOCK, WIDTH=BLOCK, TILE_W=TILE_W, TILE_D=TILE_D, HEAD_DIM=HEAD_DIM,
        )
    if tail:
        _gather_layouts_kernel[(1, triton.cdiv(tail, TILE_W), triton.cdiv(HEAD_DIM, TILE_D))](
            kv, indices, straight_tail, transposed_tail, n, nfull * BLOCK,
            BLOCK=BLOCK, WIDTH=tail, TILE_W=TILE_W, TILE_D=TILE_D, HEAD_DIM=HEAD_DIM,
        )