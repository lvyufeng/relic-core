// Independent HC4 up-mix candidate: share each activation load across four stream dot chains.
// Every chain retains the oracle's lane-strided FMA sequence, shuffle tree and fold order.
// This is an opt-in experiment; only the independent device oracle can qualify its bits.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ float hc_exact_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void hc4_up_mix_exact_kernel(
    const __nv_bfloat16* __restrict__ down,
    const __nv_bfloat16* __restrict__ mix_up,
    const __nv_bfloat16* __restrict__ normed,
    __nv_bfloat16* __restrict__ out,
    int inter, int hidden, int hc) {
    const int warps = blockDim.x >> 5;
    const int j = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.y;
    if (j >= hidden) return;
    const __nv_bfloat16* d = down + static_cast<long long>(row) * inter;
    const __nv_bfloat16* n = normed + static_cast<long long>(row) * hidden * hc;
    const __nv_bfloat16* w0 = mix_up + static_cast<long long>(j) * inter;
    const __nv_bfloat16* w1 = w0 + static_cast<long long>(hidden) * inter;
    const __nv_bfloat16* w2 = w1 + static_cast<long long>(hidden) * inter;
    const __nv_bfloat16* w3 = w2 + static_cast<long long>(hidden) * inter;
    float dot0 = 0.f, dot1 = 0.f, dot2 = 0.f, dot3 = 0.f;
    for (int k = lane; k < inter; k += 32) {
        const float value = __bfloat162float(d[k]);
        dot0 += value * __bfloat162float(w0[k]);
        dot1 += value * __bfloat162float(w1[k]);
        dot2 += value * __bfloat162float(w2[k]);
        dot3 += value * __bfloat162float(w3[k]);
    }
    dot0 = hc_exact_warp_sum(dot0);
    dot1 = hc_exact_warp_sum(dot1);
    dot2 = hc_exact_warp_sum(dot2);
    dot3 = hc_exact_warp_sum(dot3);
    const float gate0 = 1.f / (1.f + expf(-__shfl_sync(0xffffffffu, dot0, 0)));
    const float gate1 = 1.f / (1.f + expf(-__shfl_sync(0xffffffffu, dot1, 0)));
    const float gate2 = 1.f / (1.f + expf(-__shfl_sync(0xffffffffu, dot2, 0)));
    const float gate3 = 1.f / (1.f + expf(-__shfl_sync(0xffffffffu, dot3, 0)));
    // Keep the original stream-ordered accumulation, including the initial +0.0f.
    float folded = 0.f;
    folded += gate0 * __bfloat162float(n[j]);
    folded += gate1 * __bfloat162float(n[j + hidden]);
    folded += gate2 * __bfloat162float(n[j + 2 * hidden]);
    folded += gate3 * __bfloat162float(n[j + 3 * hidden]);
    if (lane == 0) {
        out[static_cast<long long>(row) * hidden + j] = __float2bfloat16(folded / hc);
    }
}
}  // namespace

torch::Tensor qwen4_exp_hc_up_mix_exact(
    const torch::Tensor& down,
    const torch::Tensor& mix_up,
    const torch::Tensor& normed,
    int64_t hc, int64_t hidden, int64_t threads) {
    TORCH_CHECK(down.is_cuda() && mix_up.is_cuda() && normed.is_cuda(), "HC exact operands must be CUDA");
    TORCH_CHECK(down.device() == mix_up.device() && down.device() == normed.device(), "HC exact operands must share a device");
    TORCH_CHECK(down.scalar_type() == torch::kBFloat16 && mix_up.scalar_type() == torch::kBFloat16 &&
                normed.scalar_type() == torch::kBFloat16, "HC exact operands must be bfloat16");
    TORCH_CHECK(down.is_contiguous() && mix_up.is_contiguous() && normed.is_contiguous(),
                "HC exact operands must be contiguous");
    TORCH_CHECK(down.dim() == 2 && mix_up.dim() == 2 && normed.dim() == 2,
                "HC exact operands must be matrices");
    TORCH_CHECK(hc == 4 && hidden == 2560 && down.size(1) == 320, "HC exact is scoped to HC4/H2560/I320");
    TORCH_CHECK(down.size(0) >= 1, "HC exact needs at least one row");
    TORCH_CHECK(mix_up.size(0) == hc * hidden && mix_up.size(1) == down.size(1), "HC exact mix_up shape mismatch");
    TORCH_CHECK(normed.size(0) == down.size(0) && normed.size(1) == hc * hidden, "HC exact normed shape mismatch");
    TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 || threads == 512,
                "HC exact threads must be 64, 128, 256 or 512");
    const c10::cuda::CUDAGuard guard(down.device());
    auto out = torch::empty({down.size(0), hidden}, down.options());
    const int warps = static_cast<int>(threads / 32);
    hc4_up_mix_exact_kernel<<<dim3((hidden + warps - 1) / warps, down.size(0)), threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(down.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mix_up.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(normed.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), 320, 2560, 4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
