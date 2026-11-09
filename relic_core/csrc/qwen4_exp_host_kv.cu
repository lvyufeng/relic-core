// Gather rows of the QSA key/value arena out of host memory and into the device.
//
// The arena for a 262144-token context is 3.2 GiB a rank at this model's geometry -- more than a
// fully resident expert bank leaves free -- so it lives in pinned host memory.  A sparse-attention
// layer reads `indexer_budget` positions a query whatever the context is, so only those rows have
// to be on the card, and this kernel is what puts them there.
//
// It reads the source through its host address.  Under UVA -- every CUDA this runtime targets --
// a `cudaHostAlloc` allocation is mapped into the device address space, so the pointer a pinned
// torch tensor exposes is one a kernel can dereference, and the traffic goes over PCIe without a
// host copy and, more to the point, without the host round trip that a `.to("cpu")` of the index
// list would force: a synchronous device-to-host read on every layer is what kept the offload from
// coexisting with a capture.
//
// One block a row keeps each row's 512 bytes contiguous, which is what PCIe wants; a thread a
// element would turn every 2-byte read into its own transaction.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

//: Head width this model uses, and the number of threads a row is gathered with.
constexpr int kRowElems = 256;

__global__ void gather_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ idx,
    __nv_bfloat16* __restrict__ dst,
    int rows) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int32_t from = idx[row];
    __nv_bfloat16* out = dst + static_cast<long long>(row) * kRowElems;
    if (from < 0) {
        // A padding slot: the attention masks it, but the buffer still has to be defined.
        for (int i = threadIdx.x; i < kRowElems; i += blockDim.x) {
            out[i] = __float2bfloat16(0.f);
        }
        return;
    }
    const __nv_bfloat16* in = src + static_cast<long long>(from) * kRowElems;
    for (int i = threadIdx.x; i < kRowElems; i += blockDim.x) {
        out[i] = in[i];
    }
}

// The mirror of the gather: write this step's rows into the host arena.
//
// It has to be a kernel for the same reason the gather does, from the other side.  Writing with a
// Python slice bakes the offset into the recording, so a replay would write every step's rows to
// the position the capture happened at; the offset has to be a device value the recording reads.
__global__ void scatter_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst_host,
    const int32_t* __restrict__ at,
    int rows) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int32_t to = at[row];
    const __nv_bfloat16* in = src + static_cast<long long>(row) * kRowElems;
    __nv_bfloat16* out = dst_host + static_cast<long long>(to) * kRowElems;
    for (int i = threadIdx.x; i < kRowElems; i += blockDim.x) {
        out[i] = in[i];
    }
}

// Rows arrive in runs -- the block indexer picks whole blocks of `compress_ratio`, measured at
// 8.3 rows a run for this model -- so copying a row at a time pays the DMA engine's per-transfer
// cost 248 times more often than it has to.  A block copies one run instead: `kMaxRunRows` rows of
// 256 halfs is 4 KiB, which one block moves in a handful of coalesced transactions.
__global__ void gather_runs_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ run_start,
    const int32_t* __restrict__ run_rows,
    const int32_t* __restrict__ run_dst,
    const int32_t* __restrict__ run_count,
    __nv_bfloat16* __restrict__ dst,
    int capacity) {
    // A fixed grid with the live count on the device: taking the count back to the host to size
    // the launch would be a synchronisation, and a capture cannot contain one.
    const int run = blockIdx.x;
    if (run >= capacity || run >= run_count[0]) {
        return;
    }
    const int32_t from = run_start[run];
    const int rows = run_rows[run];
    __nv_bfloat16* out = dst + static_cast<long long>(run_dst[run]) * kRowElems;
    if (from < 0) {
        for (int i = threadIdx.x; i < rows * kRowElems; i += blockDim.x) {
            out[i] = __float2bfloat16(0.f);
        }
        return;
    }
    const __nv_bfloat16* in = src + static_cast<long long>(from) * kRowElems;
    const int total = rows * kRowElems;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        out[i] = in[i];
    }
}

// One block a row and a block a row's worth of threads: the width is fixed by the architecture
// (256), so the block shape is too and there is nothing to tune per call.
constexpr int kBlockThreads = kRowElems;

}  // namespace

void qwen4_exp_gather_rows_bf16_cuda(
    const void* src_host,
    const torch::Tensor& index,
    torch::Tensor& out) {
    const int rows = static_cast<int>(index.numel());
    if (rows == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_rows_bf16_kernel<<<rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_host),
        index.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen4_exp_scatter_rows_bf16_cuda(
    const torch::Tensor& src,
    void* dst_host,
    const torch::Tensor& at) {
    const int rows = static_cast<int>(at.numel());
    if (rows == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scatter_rows_bf16_kernel<<<rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dst_host),
        at.data_ptr<int32_t>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen4_exp_gather_runs_bf16_cuda(
    const void* src_host,
    const torch::Tensor& run_start,
    const torch::Tensor& run_rows,
    const torch::Tensor& run_dst,
    const torch::Tensor& run_count,
    torch::Tensor& out) {
    const int capacity = static_cast<int>(run_start.numel());
    if (capacity == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_runs_bf16_kernel<<<capacity, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_host),
        run_start.data_ptr<int32_t>(),
        run_rows.data_ptr<int32_t>(),
        run_dst.data_ptr<int32_t>(),
        run_count.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
