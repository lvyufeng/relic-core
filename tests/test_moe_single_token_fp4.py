"""Numerical sanity check for moe_single_token_fp4_forward.

Validates the FP4 single-token MoE CUDA op by comparing it against a
PyTorch bf16 reference built from the same FP4 codes + e8m0 block scales.
The kernel quantizes activations to int8 per-token and dequantizes weights
on the fly via a LUT, so the expected error is dominated by activation
quantization noise (~1/127).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from relic_core.kernels.cuda_loader import load_cuda_kernel


_FP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _e4m3_byte_to_scale(byte_tensor: torch.Tensor) -> torch.Tensor:
    """Decode a scale byte the way `fp4_block_scale` does: E4M3, not E8M0.

    This test used to treat the byte as an e8m0 exponent (`2^(b-127)`). The kernel's
    `fp4_block_scale` parses sign / 4-bit exponent biased by 7 / 3-bit mantissa, and its own comment
    records why: a ModelOpt checkpoint stores these scales as F8_E4M3, and the old `2^(b-1)`-style
    rule turned every product into an overflow. The reference has to decode the same way or it is
    validating a format the kernel does not implement.
    """
    bits = byte_tensor.to(torch.int64)
    sign = (bits >> 7) & 0x1
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    mant = mantissa.to(torch.float32)
    subnormal = (mant / 8.0) * (2.0 ** -6)
    normal = (1.0 + mant / 8.0) * torch.pow(2.0, (exponent - 7).to(torch.float32))
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where(sign == 1, -value, value)


def _dequant_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """packed: [..., K/2] uint8; scales: [..., K/16] uint8 E4M3 bytes -> float32 [..., K].

    The doubled E2M1 codebook the kernel uses is applied here, so a scale byte of `amax/12`
    in this reference corresponds to a weight of `level * amax/6` -- see the note below.
    """
    raw = packed.contiguous().to(torch.uint8)
    *batch, k_half = raw.shape
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    # Even K = low nibble, odd K = high nibble.
    codes = torch.empty(*batch, k_half * 2, dtype=torch.uint8, device=raw.device)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    # `fp4_lut_x2` is the LUT the kernel's DP4A path uses and it stores the E2M1 levels **doubled**
    # ({0,1,2,3,4,6,8,12}), so a kernel-correct reference has to double them too. The half that
    # compensates belongs to whoever produces the scale bytes: a caller that wants a weight of
    # `level * amax/6` passes an E4M3 block scale of `amax/12`.
    levels = _FP4_LEVELS.to(raw.device) * 2.0
    weights = levels[codes.long()]
    # Scales: one byte per 16 K elems, which is the ModelOpt grouping the kernel implements
    # (`blocks_k = dim / 16`). This used to say 32 and the kernel refuses that shape outright.
    block_scales = _e4m3_byte_to_scale(scales).repeat_interleave(16, dim=-1)
    return weights * block_scales


def _reference_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    route_weights: torch.Tensor,
    w1q: torch.Tensor, w1s: torch.Tensor,
    w2q: torch.Tensor, w2s: torch.Tensor,
    w3q: torch.Tensor, w3s: torch.Tensor,
    experts_start_idx: int,
    n_local_experts: int,
    swiglu_limit: float,
) -> torch.Tensor:
    dim = x.shape[1]
    y = torch.zeros((1, dim), device=x.device, dtype=torch.float32)
    x_f = x.to(torch.float32)
    # x is single-token: simulate per-row int8 activation quant to match kernel.
    row_max = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    x_scale = row_max / 127.0
    x_q = torch.clamp(torch.round(x_f / x_scale), -127, 127)
    x_dq = x_q * x_scale  # float32 with same int8 quantization as kernel
    for r, idx in enumerate(indices.tolist()):
        local = idx - experts_start_idx
        if local < 0 or local >= n_local_experts:
            continue
        rw = float(route_weights[r])
        w1 = _dequant_fp4(w1q[local], w1s[local])  # [N, D]
        w3 = _dequant_fp4(w3q[local], w3s[local])
        gate = x_dq @ w1.T
        up = x_dq @ w3.T
        if swiglu_limit > 0.0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        silu = gate / (1.0 + torch.exp(-gate))
        hidden = silu * up * rw  # [1, N]
        # Per-route int8 hidden quant to match kernel.
        h_max = hidden.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        h_scale = h_max / 127.0
        h_q = torch.clamp(torch.round(hidden / h_scale), -127, 127)
        h_dq = h_q * h_scale
        w2 = _dequant_fp4(w2q[local], w2s[local])  # [D, N]
        y_route = h_dq @ w2.T  # [1, D]
        y += y_route
    return y


def _make_random_fp4(n_experts: int, out_dim: int, in_dim: int, seed: int, device: torch.device):
    g = torch.Generator(device=device).manual_seed(seed)
    # Random FP4 codes 0..15 (each byte = two codes).
    packed = torch.randint(0, 256, (n_experts, out_dim, in_dim // 2), generator=g, device=device, dtype=torch.uint8)
    # Random E4M3 scale bytes around 1.0: 0x30..0x40 decode to 0.5..2.0 under sign/exp/exp bias 7
    # and mantissa. Staying away from 0x7F matters because that byte is NaN in E4M3.
    scales = torch.randint(0x30, 0x41, (n_experts, out_dim, in_dim // 16), generator=g, device=device, dtype=torch.uint8)
    return packed, scales


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if ext is None:
        raise RuntimeError("cuda_kernel extension not available")
    if not hasattr(ext, "moe_single_token_fp4_forward"):
        raise RuntimeError("moe_single_token_fp4_forward symbol missing")

    n_experts = 8
    dim = 1024
    inter_dim = 256
    experts_start_idx = 16
    topk = 4
    swiglu_limit = 7.0

    w1q, w1s = _make_random_fp4(n_experts, inter_dim, dim, 11, device)
    w3q, w3s = _make_random_fp4(n_experts, inter_dim, dim, 12, device)
    w2q, w2s = _make_random_fp4(n_experts, dim, inter_dim, 13, device)

    g = torch.Generator(device=device).manual_seed(7)
    x = (torch.randn(1, dim, generator=g, device=device) * 0.5).to(torch.bfloat16)
    perm = torch.randperm(n_experts, generator=g, device=device)[:topk]
    indices = (perm + experts_start_idx).to(torch.int64)
    weights = torch.rand(topk, generator=g, device=device, dtype=torch.float32) * 0.4 + 0.05

    y_kernel = ext.moe_single_token_fp4_forward(
        x.contiguous(),
        indices.contiguous(),
        weights.contiguous(),
        w1q.contiguous(), w1s.contiguous(),
        w2q.contiguous(), w2s.contiguous(),
        w3q.contiguous(), w3s.contiguous(),
        int(experts_start_idx),
        float(swiglu_limit),
    )
    torch.cuda.synchronize()

    y_ref = _reference_forward(
        x, indices, weights,
        w1q, w1s, w2q, w2s, w3q, w3s,
        experts_start_idx, n_experts, swiglu_limit,
    )
    torch.cuda.synchronize()

    diff = (y_kernel - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-2)
    rel = diff / denom
    print("y_kernel norm =", float(y_kernel.norm().item()))
    print("y_ref    norm =", float(y_ref.norm().item()))
    print("max abs diff =", float(diff.max().item()))
    print("mean abs diff =", float(diff.mean().item()))
    print("p99 rel diff =", float(rel.flatten().quantile(0.99).item()))
    print("max rel diff =", float(rel.max().item()))

    # Reference uses identical int8 quantization for x and hidden, so kernel
    # should agree to within float rounding (~1e-3 relative is comfortable).
    assert float(rel.max().item()) <= 5e-3, "FP4 kernel disagrees with reference beyond 0.5%"
    print("PASS")


if __name__ == "__main__":
    main()
