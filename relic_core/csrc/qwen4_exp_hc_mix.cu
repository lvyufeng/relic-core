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
//: Warps a block in the mixer kernels, so one launch covers many output elements.
constexpr int kWarpsPerBlock = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// A warp an output element, not a block: the dot product is `inter` long -- 320 at this model's
// geometry -- so a 128-thread block reduction spent seven barriers per stream, four streams an
// element, on two or three multiply-adds a thread.  The barriers were the kernel, and this one
// measured first by device time in a captured decode, ahead of the all-reduce.  A warp reduces
// with shuffles and needs no barrier at all, and a block carries `kWarpsPerBlock` of them so the
// launch still fills the card.
__global__ void hc_up_mix_kernel(
    const __nv_bfloat16* __restrict__ down,     // [rows, inter]
    const __nv_bfloat16* __restrict__ mix_up,   // [hc*hidden, inter]
    const __nv_bfloat16* __restrict__ normed,   // [rows, hc*hidden]
    __nv_bfloat16* __restrict__ out,            // [rows, hidden]
    int inter,
    int hidden,
    int hc) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * kWarpsPerBlock + warp;
    const int row = blockIdx.y;
    if (j >= hidden) {
        return;
    }
    const __nv_bfloat16* d = down + static_cast<long long>(row) * inter;
    const __nv_bfloat16* n = normed + static_cast<long long>(row) * hidden * hc;
    float folded = 0.f;
    for (int s = 0; s < hc; ++s) {
        const __nv_bfloat16* w =
            mix_up + (static_cast<long long>(j) + static_cast<long long>(s) * hidden) * inter;
        float dot = 0.f;
        for (int k = lane; k < inter; k += 32) {
            dot += __bfloat162float(d[k]) * __bfloat162float(w[k]);
        }
        dot = warp_sum(dot);
        // `warp_sum` leaves the total in lane 0, so it is broadcast before the gate: every lane
        // then folds the same value and lane 0's accumulator is the one that is written.
        const float gate = 1.f / (1.f + expf(-__shfl_sync(0xffffffff, dot, 0)));
        folded += gate * __bfloat162float(n[j + s * hidden]);
    }
    if (lane == 0) {
        out[static_cast<long long>(row) * hidden + j] = __float2bfloat16(folded / hc);
    }
}



// The mixer's two input projections, from one pass over the normalised streams.
//
// `mix_down` maps `hc * hidden` to `inter` and `block_inject` maps the same input to `hc`; read as
// tensor code that is two GEMVs over the same 10240-wide vector plus a silu and a sigmoid, so the
// input is read twice and four launches are spent on it, per mixer, 96 mixers a token.  Here one
// block owns one output row of the concatenated weight and the input is read once per block.
//
// The scalings are the model's: `silu(x / hc)` for the down row and `2 * sigmoid(x / hc)` for the
// inject row.  Which of the two a block computes is decided by where it sits in the grid, so the
// two weights are passed separately and a block reads only the one it needs.
__global__ void hc_down_inject_kernel(
    const __nv_bfloat16* __restrict__ normed,  // [rows, hc*hidden]
    const __nv_bfloat16* __restrict__ mix_down,   // [inter, hc*hidden]
    const __nv_bfloat16* __restrict__ block_inject,  // [hc, hc*hidden]
    __nv_bfloat16* __restrict__ down_out,      // [rows, inter], silu applied
    __nv_bfloat16* __restrict__ inject_out,    // [rows, hc], 2*sigmoid applied
    int width,
    int inter,
    int hc,
    float scale) {
    const int row = blockIdx.y;
    const int which = blockIdx.x;
    const bool is_down = which < inter;
    const int out_index = is_down ? which : which - inter;
    const int out_width = is_down ? inter : hc;
    if (out_index >= out_width) {
        return;
    }
    const __nv_bfloat16* n = normed + static_cast<long long>(row) * width;
    const __nv_bfloat16* w =
        (is_down ? mix_down : block_inject) + static_cast<long long>(out_index) * width;
    float dot = 0.f;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
        dot += __bfloat162float(n[k]) * __bfloat162float(w[k]);
    }
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) {
        return;
    }
    const float v = partial[0] * scale;
    if (is_down) {
        // silu(v) = v * sigmoid(v)
        const float g = 1.f / (1.f + expf(-v));
        down_out[static_cast<long long>(row) * inter + out_index] = __float2bfloat16(v * g);
    } else {
        inject_out[static_cast<long long>(row) * hc + out_index] =
            __float2bfloat16(2.f / (1.f + expf(-v)));
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
    const int up_blocks = (hidden + kWarpsPerBlock - 1) / kWarpsPerBlock;
    hc_up_mix_kernel<<<dim3(up_blocks, rows), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(down.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mix_up.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(normed.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        inter, hidden, static_cast<int>(hc));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen4_exp_hc_down_inject_cuda(
    const torch::Tensor& normed,
    const torch::Tensor& mix_down,
    const torch::Tensor& block_inject,
    torch::Tensor& down_out,
    torch::Tensor& inject_out,
    int64_t hc) {
    const int rows = static_cast<int>(normed.size(0));
    const int width = static_cast<int>(normed.size(1));
    const int inter = static_cast<int>(down_out.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hc_down_inject_kernel<<<dim3(inter + static_cast<int>(hc), rows), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(normed.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mix_down.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(block_inject.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(down_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(inject_out.data_ptr()),
        width, inter, static_cast<int>(hc), 1.f / static_cast<float>(hc));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
