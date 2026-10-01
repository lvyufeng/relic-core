import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from relic_core.kernels.ops import q8_0_weight_dequantize, q8_0_weight_gemm, q8_0_weight_gemm_cuda_ext
from tests.test_q8_0_cuda import _pack_q8_0


def _time_ms(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    shapes = [
        (1, 4096, 4096),
        (1, 4096, 1024),
        (2, 4096, 4096),
        (4, 4096, 4096),
    ]
    for m, k, n in shapes:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.float32)
        blocks = _pack_q8_0(weight).to("cuda")
        reference = lambda: torch.nn.functional.linear(x.float(), q8_0_weight_dequantize(blocks, row_elems=k), None).to(torch.bfloat16)
        fused = lambda: q8_0_weight_gemm_cuda_ext(x, blocks, row_elems=k, out_dtype=torch.bfloat16)
        fallback_ms = _time_ms(reference, iters=20)
        fused_ms = _time_ms(fused, iters=20)
        got = fused()
        ref_out = reference()
        max_abs = float((got.float() - ref_out.float()).abs().max().item())
        print(f"q8_0_bench m={m} k={k} n={n} fallback_ms={fallback_ms:.3f} fused_ms={fused_ms:.3f} speedup={fallback_ms / max(fused_ms, 1e-9):.2f} max_abs={max_abs:.4f}", flush=True)


if __name__ == "__main__":
    main()
