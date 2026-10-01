import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from relic_core.kernels.ops import q8_0_weight_gemm, q8_0_weight_gemm_cuda_ext


def _pack_q8_0(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.float().contiguous()
    rows, cols = weight.shape
    blocks_per_row = (cols + 31) // 32
    padded = torch.zeros(rows, blocks_per_row * 32, dtype=torch.float32)
    padded[:, :cols] = weight
    view = padded.view(rows, blocks_per_row, 32)
    scales = view.abs().amax(dim=2).clamp_min(1e-6) / 127.0
    q = torch.clamp(torch.round(view / scales.unsqueeze(-1)), -127, 127).to(torch.int8)
    blocks = torch.empty(rows, blocks_per_row, 34, dtype=torch.uint8)
    blocks[:, :, :2] = scales.to(torch.float16).view(torch.uint8).view(rows, blocks_per_row, 2)
    blocks[:, :, 2:] = q.view(torch.uint8)
    return blocks


def test_q8_0_cuda_ext_matches_reference():
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return
    torch.manual_seed(0)
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(96, 64, dtype=torch.float32)
    blocks = _pack_q8_0(weight).to("cuda")
    ref = q8_0_weight_gemm(x, blocks, row_elems=64, out_dtype=torch.bfloat16)
    got = q8_0_weight_gemm_cuda_ext(x, blocks, row_elems=64, out_dtype=torch.bfloat16)
    diff = (got.float() - ref.float()).abs()
    print("max_abs", float(diff.max().item()))
    print("mean_abs", float(diff.mean().item()))
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.02)


if __name__ == "__main__":
    test_q8_0_cuda_ext_matches_reference()
    print("PASS")
