// The hyper-connection mixer's up projection, fused with its gates and its fold.
//
// A mixer reads: from the `hc_count` streams, predict a weight a stream with `mix_up`, gate it,
// and average the streams weighted by it.  Written as tensor code that is a GEMV producing
// `hc_count * hidden` values, three elementwise passes over them, and a reduction -- and the
// width of that intermediate is the whole cost: at this model's geometry it is 10240 floats a
// mixer, twice, and there are 96 mixers a token.
//
// Fusing it means the intermediate never exists.  One block owns one output element, walks the
// `mix_up` rows of the streams it needs (row `j + s * hidden` is contiguous in `k`, so the reads
// coalesce), applies the sigmoid to each stream's dot product, and folds the result against
// `normed` on the spot.
//
// The arithmetic is the tensor code's, in the same order: a per-stream dot in bf16 inputs
// accumulated in fp32, a sigmoid, and an fp32 average that is cast once at the end.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;

__global__ void hc_up_mix_kernel(
    const __nv_bfloat16* __restrict__ down,     // [rows, inter]
    const __nv_bfloat16* __restrict__ mix_up,   // [hc*hidden, inter]
    const __nv_bfloat16* __restrict__ normed,   // [rows, hc*hidden]
    __nv_bfloat16* __restrict__ out,            // [rows, hidden]
    int inter,
    int hidden,
    int hc) {
    const int j = blockIdx.x;                   // output element within `hidden`
    const int row = blockIdx.y;
    if (j >= hidden) {
        return;
    }
    const __nv_bfloat16* d = down + static_cast<long long>(row) * inter;
    const __nv_bfloat16* n = normed + static_cast<long long>(row) * hidden * hc;
    float folded = 0.f;
    for (int s = 0; s < hc; ++s) {
        const __nv_bfloat16* w = mix_up + (static_cast<long long>(j) + static_cast<long long>(s) * hidden) * inter;
        float dot = 0.f;
        for (int k = threadIdx.x; k < inter; k += blockDim.x) {
            dot += __bfloat162float(d[k]) * __bfloat162float(w[k]);
        }
        // A block-wide sum: the per-thread partials live in shared memory, and the warp that reads
        // them back is one warp of `kThreads`.
        __shared__ float partial[kThreads];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float gate = 1.f / (1.f + expf(-partial[0]));
        folded += gate * __bfloat162float(n[j + s * hidden]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[static_cast<long long>(row) * hidden + j] = __float2bfloat16(folded / hc);
    }
}

}  // namespace

void qwen4_exp_hc_up_mix_cuda(
    const torch::Tensor& down,
    const torch::Tensor& mix_up,
    const torch::Tensor& normed,
    torch::Tensor& out,
    int64_t hc) {
    const int rows = static_cast<int>(down.size(0));
    const int inter = static_cast<int>(down.size(1));
    const int hidden = static_cast<int>(out.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hc_up_mix_kernel<<<dim3(hidden, rows), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(down.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mix_up.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(normed.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        inter, hidden, static_cast<int>(hc));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
