// Cross-TU declaration for the Q4_K/Q5_K MMA prefill path implemented in
// llama_mmq/gguf_mma_wrapper.cu and dispatched from cuda_kernel_impl.cu.
#pragma once

#include <torch/extension.h>

// Computes out[rows, N] = act[rows, K] @ W[N, K]^T for Q4_K (type_id==3) or
// Q5_K (type_id==4) using INT8 MMA tensor cores (vendored llama.cpp MMQ).
torch::Tensor gguf_q4k_q5k_mma_prefill_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems,
        int64_t type_id);

// Decode path (rows == 1): Q4_K/Q5_K DP4A GEMV on CUDA cores.
torch::Tensor gguf_q4k_q5k_dp4a_decode_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems,
        int64_t type_id);
