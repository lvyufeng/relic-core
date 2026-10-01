"""Numerical correctness check for hc_post_forward CUDA kernel.

Verifies that the CUDA kernel produces the same result as the Python broadcast
reference for both decode (S=1) and prefill (S>1) shapes. This guards the
relaxed CUDA guard in `Block.hc_post()` that lets long prefill skip the
`[B, S, 4, 4, D]` Python broadcast.

Run inside the deepseek conda env, e.g.:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
    /home/lvyufeng/miniconda3/envs/deepseek/bin/python tests/test_hc_cuda.py
"""

import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Run as a script (`python tests/test_hc_cuda.py`) without needing PYTHONPATH: the
# package lives at the repo root, one level up from tests/.
sys.path.insert(0, os.path.dirname(THIS_DIR))
from relic_core.kernels.cuda_loader import load_cuda_kernel  # noqa: E402


def hc_post_reference(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    # Mirror the eager broadcast in `Block.hc_post()` Python fallback.
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


def run_case(ext, device, B: int, S: int, D: int, dtype: torch.dtype, seed: int = 0) -> None:
    torch.manual_seed(seed)
    HC = 4
    x = torch.randn(B, S, D, dtype=dtype, device=device)
    residual = torch.randn(B, S, HC, D, dtype=dtype, device=device)
    post = torch.randn(B, S, HC, dtype=torch.float32, device=device)
    comb = torch.randn(B, S, HC, HC, dtype=torch.float32, device=device)

    y_ref = hc_post_reference(x, residual, post, comb)
    y_cuda = ext.hc_post_forward(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )

    diff = (y_ref.float() - y_cuda.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # The kernel accumulates in fp32 and writes in dtype, while the reference goes through
    # a sum of broadcasted bf16/fp16 tensors — small numerical differences are expected.
    # bf16 has ~7-8 bits of mantissa: per-fma rounding ~7.8e-3, ~5 fmas with sign cancellation
    # can reach ~5e-2 max abs diff at long sequences. fp16 has finer mantissa so ~5e-3.
    # fp32 reference and kernel both accumulate in fp32, so difference is near zero.
    if dtype == torch.float32:
        tol = 1e-5
    elif dtype == torch.float16:
        tol = 1e-2
    else:  # bfloat16
        tol = 5e-2
    print(
        f"[hc_post B={B} S={S} D={D} dtype={dtype}] "
        f"max abs diff = {max_diff:.5e}, mean = {mean_diff:.5e}, tol = {tol:.1e}"
    )
    assert max_diff < tol, (
        f"hc_post CUDA mismatch beyond tolerance: B={B} S={S} D={D} dtype={dtype} max={max_diff}"
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if not hasattr(ext, "hc_post_forward"):
        print("ERROR: extension was not rebuilt with hc_post_forward kernel.")
        sys.exit(2)

    # DeepSeek-V4-Flash uses dim D = 7168 / TP for the residual stream after HC merge.
    # Use a representative dim that divides cleanly and exercises the inner loop.
    dim = 1024

    # Decode shape: S=1.
    run_case(ext, device, B=1, S=1, D=dim, dtype=torch.bfloat16, seed=0)
    run_case(ext, device, B=1, S=1, D=dim, dtype=torch.float16, seed=1)

    # Mid-length prefill: ~257 tokens (covers the chunk boundary regime).
    run_case(ext, device, B=1, S=257, D=dim, dtype=torch.bfloat16, seed=2)
    run_case(ext, device, B=1, S=257, D=dim, dtype=torch.float16, seed=3)

    # Long prefill: 2048 tokens, the regime that previously OOMed the Python broadcast.
    run_case(ext, device, B=1, S=2048, D=dim, dtype=torch.bfloat16, seed=4)
    run_case(ext, device, B=1, S=2048, D=dim, dtype=torch.float16, seed=5)

    print("OK")


if __name__ == "__main__":
    main()
