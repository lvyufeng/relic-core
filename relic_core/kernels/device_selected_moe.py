"""Opt-in device-selected original-FP4 MoE decode using native Cube operands.

The checkpoint stays packed. Bounded reusable stage scratch holds selected
original I8x2/half-scale operands (or exact BF16 operands in the grouped
comparison), never another resident expert bank. Device kernels retain cyclic
ownership and ascending-global-expert ordering without reading IDs/counts back.

Native Cube K reduction differs from the released window-13 rule; this is an
explicit numerical experiment, not the exact default. Original block-32 FP8
activation encoding, FP4 bytes/scales, BF16 boundaries and the FP32 expert sum
order are retained. End-to-end measurements, not fusion alone, decide performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .ascend_unpack import _npu_launch, supports_triton_ascend
from .fast_fused_fp8_activation_decode import quantize_decode_fp8_fast

CONTRACT = "original_fp4_exact_operands_native_grouped_cube_ascending_fp32_combine"
WEIGHT_QUANT_CONTRACT = "original_fp4_i8x2_half_scales_native_cube_device_selection_ascending_fp32_combine"
BLOCK = 32


@dataclass
class DeviceMoEWorkspace:
    """Single-stream scratch shared across layers of identical geometry."""

    selected: torch.Tensor
    gate_weights: torch.Tensor
    group_list: torch.Tensor
    x_group: torch.Tensor
    weight_storage: torch.Tensor
    route_error: torch.Tensor
    dim: int
    inter_dim: int
    topk: int
    projection_mode: str = "grouped"

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.selected, self.gate_weights, self.group_list,
                    self.x_group, self.weight_storage, self.route_error))

    def projection(self, *, down: bool) -> torch.Tensor:
        k, n = (self.inter_dim, self.dim) if down else (self.dim, 2 * self.inter_dim)
        size = (self.topk + 1) * k * n
        return self.weight_storage[:size].view(self.topk + 1, n, k).transpose(1, 2)

    def weight_quant_projection(self, *, down: bool):
        k, n = (self.inter_dim, self.dim) if down else (self.dim, 2 * self.inter_dim)
        code_elements = self.topk * n * k
        scale_elements = self.topk * n * (k // BLOCK)
        values = self.weight_storage[:code_elements // 2].view(torch.int8)
        half = self.weight_storage[code_elements // 2:code_elements // 2 + scale_elements]
        return (values.view(self.topk, n, k).transpose(1, 2),
                half.view(self.topk, n, k // BLOCK).transpose(1, 2))


def make_device_moe_workspace(dim: int, inter_dim: int, topk: int, *,
                              device, max_bytes: int = 512 * 2**20,
                              projection_mode: str = "grouped") -> DeviceMoEWorkspace:
    for value, name in ((dim, "dim"), (inter_dim, "inter_dim"), (topk, "topk")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if dim % BLOCK or inter_dim % BLOCK or topk > 8:
        raise ValueError("decode widths must be multiples of 32 and topk at most 8")
    # Gate and up share one grouped matmul. Down reuses the first half of the
    # same storage on the same stream, after gate/up has consumed it.
    if projection_mode not in ("grouped", "weight_quant"):
        raise ValueError("only grouped or original weight_quant projections are supported")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer budget")
    if projection_mode == "weight_quant":
        matrix_elements = topk * dim * (2 * inter_dim)
        weight_elements = matrix_elements // 2 + matrix_elements // BLOCK
    else:
        weight_elements = (topk + 1) * dim * (2 * inter_dim)
    required = weight_elements * 2 + topk * dim * 2 + topk * 8 + (topk + 1) * 8 + 4
    if required > max_bytes:
        raise ValueError(f"selected-stage workspace needs {required} bytes, budget is {max_bytes}")
    return DeviceMoEWorkspace(
        selected=torch.empty(topk, dtype=torch.int32, device=device),
        gate_weights=torch.empty(topk, dtype=torch.float32, device=device),
        group_list=torch.empty(topk + 1, dtype=torch.int64, device=device),
        x_group=torch.empty((topk, dim), dtype=torch.bfloat16, device=device),
        weight_storage=torch.empty(weight_elements, dtype=torch.bfloat16, device=device),
        route_error=torch.zeros(1, dtype=torch.int32, device=device),
        dim=dim, inter_dim=inter_dim, topk=topk, projection_mode=projection_mode,
    )


def _validate(x, weights, indices, banks, *, rank, world, n_experts, workspace):
    if x.ndim != 2 or tuple(x.shape) != (1, workspace.dim) or x.dtype != torch.bfloat16:
        raise ValueError("device-selected MoE requires one BF16 activation row")
    if not x.is_contiguous() or x.requires_grad:
        raise ValueError("device-selected MoE requires contiguous inference input")
    if tuple(indices.shape) != (1, workspace.topk) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("routing indices must be contiguous [1, topk] int32/int64")
    if tuple(weights.shape) != tuple(indices.shape) or weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("routing weights must be [1, topk] BF16/FP32")
    if not indices.is_contiguous() or not weights.is_contiguous():
        raise ValueError("routing tensors must be contiguous")
    if isinstance(world, bool) or not isinstance(world, int) or world <= 0:
        raise ValueError("world must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
        raise ValueError("rank must be in [0, world)")
    if (isinstance(n_experts, bool) or not isinstance(n_experts, int)
            or not workspace.topk <= n_experts < 2**31):
        raise ValueError("expert count must be an int32-representable integer at least topk")
    storage_elements = ((workspace.topk + 1) * workspace.dim * 2 * workspace.inter_dim
                        if workspace.projection_mode == "grouped" else
                        workspace.topk * workspace.dim * 2 * workspace.inter_dim // 2
                        + workspace.topk * workspace.dim * 2 * workspace.inter_dim // BLOCK)
    scratch = ((workspace.selected, (workspace.topk,), torch.int32),
               (workspace.gate_weights, (workspace.topk,), torch.float32),
               (workspace.group_list, (workspace.topk + 1,), torch.int64),
               (workspace.x_group, (workspace.topk, workspace.dim), torch.bfloat16),
               (workspace.weight_storage, (storage_elements,), torch.bfloat16),
               (workspace.route_error, (1,), torch.int32))
    if any(tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous()
           for t, shape, dtype in scratch):
        raise ValueError("preallocated stage workspace metadata is inconsistent")
    owned = len(range(rank, n_experts, world))
    all_tensors = [x, weights, indices, workspace.selected, workspace.gate_weights,
                   workspace.group_list, workspace.x_group, workspace.weight_storage, workspace.route_error]
    for which, n, k in (("w1", workspace.inter_dim, workspace.dim),
                        ("w3", workspace.inter_dim, workspace.dim),
                        ("w2", workspace.dim, workspace.inter_dim)):
        codes, scales = banks[which]
        if tuple(codes.shape) != (owned, n, k // 2) or tuple(scales.shape) != (owned, n, k // BLOCK):
            raise ValueError(f"{which} arena shape does not match cyclic expert ownership")
        if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise TypeError("expert arenas must carry the original uint8 payloads")
        if not codes.is_contiguous() or not scales.is_contiguous():
            raise ValueError("expert arenas must be contiguous")
        all_tensors.extend((codes, scales))
    if any(t.device != x.device for t in all_tensors):
        raise ValueError("all operands and scratch must be on the same device")
    if x.device.type not in ("npu", "cpu"):
        raise ValueError("the implementation is NPU with a CPU numerical oracle")


def _torch_routes(indices, weights, *, rank, world, n_experts, workspace):
    ids = indices[0].tolist()
    if any(i < 0 or i >= n_experts for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("routing IDs must be valid and without replacement")
    slots = sorted((i, slot) for slot, i in enumerate(ids) if i % world == rank)
    workspace.selected.fill_(-1)
    workspace.gate_weights.zero_()
    for group, (expert, slot) in enumerate(slots):
        workspace.selected[group] = expert
        workspace.gate_weights[group] = weights[0, slot].float()
    count = len(slots)
    workspace.group_list.copy_(torch.tensor([min(i + 1, count) for i in range(workspace.topk)]
                                           + [workspace.topk], dtype=torch.int64))
    return slots


def _torch_weight(codes, scales):
    from .unpack import unpack_fp4_to_int8, fp4_int8_scale

    values = unpack_fp4_to_int8(codes, backend="torch").float()
    half = fp4_int8_scale(scales, backend="torch").repeat_interleave(BLOCK, dim=-1)
    return (values * half).to(torch.bfloat16)


def _torch_prepare(banks, slots, *, workspace, world, down, activation=None):
    target = workspace.projection(down=down)
    target[-1].zero_()  # Dummy group consumes the fixed-capacity padding rows.
    for group, (expert, _) in enumerate(slots):
        local = expert // world
        if down:
            codes, scales = banks["w2"]
            target[group].copy_(_torch_weight(codes[local], scales[local]).t())
        else:
            gate = _torch_weight(banks["w1"][0][local], banks["w1"][1][local])
            up = _torch_weight(banks["w3"][0][local], banks["w3"][1][local])
            target[group].copy_(torch.cat((gate, up), dim=0).t())
    if activation is not None:
        workspace.x_group.zero_()
        for group in range(len(slots)):
            workspace.x_group[group].copy_(activation[0])
    return target


def _grouped(x, weight, group_list, *, backend):
    if backend == "torch":
        result = torch.zeros((x.shape[0], weight.shape[2]), dtype=torch.bfloat16, device=x.device)
        previous = 0
        for group, end in enumerate(group_list.tolist()):
            if end > previous:
                result[previous:end] = x[previous:end] @ weight[group]
            previous = end
        return result
    import torch_npu

    return torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list,
                                      split_item=3, group_type=0, group_list_type=0)[0]


def _triton_routes(indices, weights, *, rank, world, n_experts, workspace):
    from ._ascend_device_selected_moe_kernels import route_owned_kernel

    with _npu_launch(indices):
        route_owned_kernel[(1,)](indices, weights, workspace.selected, workspace.gate_weights,
                                 workspace.group_list, workspace.route_error,
                                 workspace.topk, triton_power_two(workspace.topk),
                                 n_experts, rank, world)


def triton_power_two(value):
    return 1 << (value - 1).bit_length()


def _triton_prepare(banks, *, workspace, world, down, activation=None):
    from ._ascend_device_selected_moe_kernels import (
        prepare_selected_pairs_kernel, replicate_selected_activation_kernel,
    )

    target = workspace.projection(down=down)
    k, n = target.shape[1:]
    if down:
        b, sb = banks["w2"]
        c, sc = b, sb
    else:
        b, sb = banks["w1"]
        c, sc = banks["w3"]
    # Prepare N-major operands with contiguous source/output access, then pass
    # the [K,N] transpose view to Cube; no standalone physical transpose kernel.
    tile = min(2048, triton_power_two(k // 2))
    programs = min(n * math.ceil((k // 2) / tile), 128)
    pairs = workspace.weight_storage[:target.numel()].view(torch.uint32)
    with _npu_launch(b):
        if activation is not None:
            replicate_selected_activation_kernel[(math.ceil(k / 256), workspace.topk)](
                activation, workspace.selected, workspace.x_group, k, 256)
        prepare_selected_pairs_kernel[(programs, workspace.topk + 1)](
            b, sb, c, sc, workspace.selected, pairs,
            k, n, world, workspace.topk, tile, not down,
            max(b.numel(), c.numel(), target.numel()) >= 2**31,
        )
    return target




def _torch_prepare_weight_quant(banks, slots, *, workspace, world, down, activation=None):
    from .unpack import unpack_fp4_to_int8, fp4_int8_scale

    values, half = workspace.weight_quant_projection(down=down)
    values.zero_()
    half.fill_(1)
    for group, (expert, _) in enumerate(slots):
        local = expert // world
        names = ("w2",) if down else ("w1", "w3")
        codes = torch.cat([banks[name][0][local] for name in names], dim=0)
        scales = torch.cat([banks[name][1][local] for name in names], dim=0)
        values[group].copy_(unpack_fp4_to_int8(codes, backend="torch").t())
        half[group].copy_(fp4_int8_scale(scales, backend="torch").to(torch.bfloat16).t())
    if activation is not None:
        workspace.x_group.zero_()
        for group in range(len(slots)):
            workspace.x_group[group].copy_(activation[0])
    return values, half


def _triton_prepare_weight_quant(banks, *, workspace, world, down, activation=None):
    from ._ascend_device_selected_moe_kernels import (
        prepare_selected_i8_pairs_kernel, prepare_selected_half_scales_kernel,
        replicate_selected_activation_kernel,
    )

    values, half = workspace.weight_quant_projection(down=down)
    k, n = values.shape[1:]
    b, sb = banks["w2"] if down else banks["w1"]
    c, sc = (b, sb) if down else banks["w3"]
    tile = min(1024, triton_power_two(k // 2))
    storage = workspace.weight_storage
    code_elements = workspace.topk * n * k
    pairs = storage[:code_elements // 2].view(torch.uint16)
    scale_bits = storage[code_elements // 2:code_elements // 2 + half.numel()].view(torch.uint16)
    with _npu_launch(b):
        if activation is not None:
            replicate_selected_activation_kernel[(math.ceil(k / 256), workspace.topk)](
                activation, workspace.selected, workspace.x_group, k, 256)
        prepare_selected_i8_pairs_kernel[(min(n * math.ceil((k // 2) / tile), 128), workspace.topk)](
            b, c, workspace.selected, pairs, k, n, world, tile, not down,
            max(b.numel(), c.numel()) >= 2**31)
        prepare_selected_half_scales_kernel[(min(math.ceil(n * (k // 32) / 256), 128), workspace.topk)](
            sb, sc, workspace.selected, scale_bits, k, n, world, 256, not down,
            max(sb.numel(), sc.numel()) >= 2**31)
    return values, half


def _weight_quant_rows(x, operands, *, backend, shared_activation=False):
    values, half = operands
    groups, k, n = values.shape
    if shared_activation:
        flat_values = values.transpose(1, 2).reshape(groups * n, k).t()
        flat_half = half.transpose(1, 2).reshape(groups * n, k // BLOCK).t()
        if backend == "torch":
            decoded = (flat_values.float() * flat_half.float().repeat_interleave(BLOCK, dim=0)).to(torch.bfloat16)
            return (x @ decoded).reshape(groups, n)
        import torch_npu

        return torch_npu.npu_weight_quant_batchmatmul(
            x, flat_values, flat_half, antiquant_group_size=BLOCK, inner_precise=0).reshape(groups, n)
    if backend == "torch":
        decoded = (values.float() * half.float().repeat_interleave(BLOCK, dim=1)).to(torch.bfloat16)
        return torch.bmm(x[:, None, :], decoded).squeeze(1)
    import torch_npu

    # This stack's SDK weight-quant op accepts only rank-two matrices. A static
    # six-call graph consumes device-selected operands; Python never reads a
    # selected ID or local count. Empty slots carry a zero operand, not stale bytes.
    return torch.cat([
        torch_npu.npu_weight_quant_batchmatmul(x[group:group+1], values[group], half[group],
                                              antiquant_group_size=BLOCK, inner_precise=0)
        for group in range(x.shape[0])
    ], dim=0)


def fused_moe_epilogue(gate_up, gates, *, swiglu_limit=0.0, backend="auto", return_codes=False, up=None):
    """Fused clamps/widening and weighted original FP8 encode/decode, with original FP32 SiLU.

    No scale/encoding is changed. The routing product still narrows to BF16
    before the exact block-32 amax. Callers must validate parity at this
    boundary before enabling the native SwiGLU composition in an experiment.
    """
    if gate_up.ndim != 2 or gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous():
        raise ValueError("gate/up must be contiguous BF16 [rows, 2*inter]")
    rows, doubled = gate_up.shape
    if up is not None:
        if (tuple(up.shape) != tuple(gate_up.shape) or up.dtype != torch.bfloat16
                or up.device != gate_up.device or not up.is_contiguous()):
            raise ValueError("separate up must match the contiguous BF16 gate")
        doubled *= 2
    if doubled % 64 or tuple(gates.shape) != (rows,) or gates.dtype != torch.float32:
        raise ValueError("the paired width must be a multiple of 64 and gates FP32 [rows]")
    if gates.device != gate_up.device or not gates.is_contiguous():
        raise ValueError("gate weights must be contiguous on the gate/up device")
    if not isinstance(swiglu_limit, (int, float)) or not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("clamp limit must be finite and nonnegative")
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be auto, torch or triton")
    width = doubled // 2
    if rows == 0:
        empty = torch.empty((0, width), dtype=torch.bfloat16, device=gate_up.device)
        if return_codes:
            return (empty, torch.empty((0, width), dtype=torch.uint8, device=gate_up.device),
                    torch.empty((0, width // 32), dtype=torch.uint8, device=gate_up.device))
        return empty
    if gate_up.device.type == "cpu" and backend != "triton":
        if up is None:
            gate, up = gate_up.float().split(width, dim=-1)
        else:
            gate, up = gate_up.float(), up.float()
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        source = (gates[:, None] * (F.silu(gate) * up)).to(torch.bfloat16)
        return quantize_decode_fp8_fast(source, backend="torch", return_codes=return_codes)
    if gate_up.device.type != "npu" or backend == "torch":
        raise ValueError("the native fused epilogue requires NPU")
    import torch_npu
    from .fused_fp8_activation_decode import FLOOR, INV_FP8_MAX
    from ._ascend_device_selected_moe_kernels import (
        clamp_gate_up_kernel, weighted_fp8_quant_decode_kernel,
    )
    clamped = torch.empty((rows, doubled), dtype=torch.float32, device=gate_up.device)
    with _npu_launch(gate_up):
        clamp_gate_up_kernel[(math.ceil(rows * doubled / 256),)](
            gate_up, gate_up if up is None else up, clamped,
            rows, width, swiglu_limit, 256, up is not None)
    # Retain the original FP32 elementwise SiLU and multiply. Native SwiGLU
    # passed small fixtures but did not establish full-context byte parity; it
    # is not licensed as a substitute for this original activation boundary.
    gate = clamped[:, :width].contiguous()
    up_value = clamped[:, width:].contiguous()
    swiglu = F.silu(gate) * up_value
    result = torch.empty((rows, width), dtype=torch.bfloat16, device=gate_up.device)
    codes = torch.empty((rows, width), dtype=torch.uint8, device=gate_up.device)
    scales = torch.empty((rows, width // 32), dtype=torch.uint8, device=gate_up.device)
    with _npu_launch(swiglu):
        weighted_fp8_quant_decode_kernel[(math.ceil(rows * (width // 32) / 8),)](
            swiglu, gates, result, codes, scales, rows * (width // 32), width // 32,
            8, FLOOR, INV_FP8_MAX)
    return (result, codes, scales) if return_codes else result


@torch.no_grad()
def device_selected_fp4_moe(x, weights, indices, banks, *, rank: int, world: int,
                            n_experts: int, swiglu_limit: float = 0.0,
                            workspace: DeviceMoEWorkspace,
                            certified_bf16_operands: bool = False,
                            backend: str = "auto", projection_mode: str = "grouped",
                            fused_epilogue: bool = False) -> torch.Tensor:
    """One-row routed FP32 partial, with no NPU routing readback.

    ``certified_bf16_operands`` is a load-time certificate of the original
    E8M0 range (2..252). It is mandatory on NPU, never inferred by hot-loop
    readback. Invalid dynamic IDs latch ``route_error`` and poison the final
    output, so ordinary finite-logit validation rejects them.
    """
    _validate(x, weights, indices, banks, rank=rank, world=world,
              n_experts=n_experts, workspace=workspace)
    if not isinstance(swiglu_limit, (int, float)) or not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("swiglu_limit must be finite and nonnegative")
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be auto, torch or triton")
    if projection_mode not in ("grouped", "weight_quant"):
        raise ValueError("only grouped or original weight_quant projections are exposed; the direct-dot prototype failed NPU validation")
    if workspace.projection_mode != projection_mode:
        raise ValueError("projection mode must match the preallocated stage workspace")
    if not isinstance(fused_epilogue, bool):
        raise ValueError("fused_epilogue must be a bool")
    selected_backend = "torch" if x.device.type == "cpu" and backend != "triton" else "triton"
    if selected_backend == "triton":
        if x.device.type != "npu" or backend == "torch" or not supports_triton_ascend():
            raise ValueError("Triton Ascend on NPU is required; there is no silent fallback")
        if certified_bf16_operands is not True:
            raise ValueError("certify original E8M0 operands at load time before using native Cube")
    decoded_x = quantize_decode_fp8_fast(x, backend=selected_backend)
    if selected_backend == "torch":
        slots = _torch_routes(indices, weights, rank=rank, world=world,
                              n_experts=n_experts, workspace=workspace)
    else:
        _triton_routes(indices, weights, rank=rank, world=world,
                        n_experts=n_experts, workspace=workspace)
    if projection_mode == "weight_quant":
        if selected_backend == "torch":
            operands = _torch_prepare_weight_quant(banks, slots, workspace=workspace, world=world,
                                                   down=False)
        else:
            operands = _triton_prepare_weight_quant(banks, workspace=workspace, world=world,
                                                    down=False)
        # Gate/up of every selected expert reads the SAME activation row, so a
        # single wider SDK call covers all six original matrices. Down retains
        # separate activation rows and separate SDK calls, avoiding cross-products.
        gate_up = _weight_quant_rows(decoded_x, operands, backend=selected_backend, shared_activation=True)
    else:
        if selected_backend == "torch":
            w13 = _torch_prepare(banks, slots, workspace=workspace, world=world,
                                  down=False, activation=decoded_x)
        else:
            w13 = _triton_prepare(banks, workspace=workspace, world=world,
                                  down=False, activation=decoded_x)
        gate_up = _grouped(workspace.x_group, w13, workspace.group_list, backend=selected_backend)
    if fused_epilogue:
        decoded_intermediate = fused_moe_epilogue(gate_up, workspace.gate_weights,
                                                   swiglu_limit=swiglu_limit, backend=selected_backend)
    else:
        gate, up = gate_up.split(workspace.inter_dim, dim=-1)
        gate, up = gate.float(), up.float()
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        intermediate = (workspace.gate_weights[:, None] * (F.silu(gate) * up)).to(torch.bfloat16)
        decoded_intermediate = quantize_decode_fp8_fast(intermediate.contiguous(), backend=selected_backend)
    if projection_mode == "weight_quant":
        if selected_backend == "torch":
            operands = _torch_prepare_weight_quant(banks, slots, workspace=workspace, world=world, down=True)
        else:
            operands = _triton_prepare_weight_quant(banks, workspace=workspace, world=world, down=True)
        contributions = _weight_quant_rows(decoded_intermediate, operands, backend=selected_backend)
    else:
        if selected_backend == "torch":
            w2 = _torch_prepare(banks, slots, workspace=workspace, world=world, down=True)
        else:
            w2 = _triton_prepare(banks, workspace=workspace, world=world, down=True)
        contributions = _grouped(decoded_intermediate, w2, workspace.group_list, backend=selected_backend)
    if selected_backend == "torch":
        result = torch.zeros((1, workspace.dim), dtype=torch.float32)
        for group in range(len(slots)):
            result = result + contributions[group:group+1].float()
        return result
    from ._ascend_device_selected_moe_kernels import ordered_combine_kernel

    result = torch.empty((1, workspace.dim), dtype=torch.float32, device=x.device)
    with _npu_launch(contributions):
        ordered_combine_kernel[(math.ceil(workspace.dim / 256),)](
            contributions, workspace.selected, workspace.route_error, result,
            workspace.dim, workspace.topk, 256)
    return result
