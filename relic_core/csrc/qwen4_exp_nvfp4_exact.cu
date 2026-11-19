// Exact-operand NVFP4 decode: the same arithmetic as `nvfp4_moe_rows_forward`, with the two
// per-element format decoders replaced by register-only, bit-identical operand construction.
//
// Why this file exists. `nvfp4_e2m1` in `cuda_kernel_impl.cu` reads a `__device__ __constant__`
// table and `nvfp4_e4m3` calls `ldexpf`. Both are memory or libdevice work per decoded element, and
// the kernel decodes them throughout each selected, rank-owned expert projection. A round does not
// read the whole expert bank. The nibble is data-dependent, so the constant-memory load diverges
// inside every warp, and a divergent CMEM access serialises on distinct addresses. Replacing the
// two decoders with register arithmetic removes the loads without touching a single float
// operation: the values they produce are bit-identical for every one of the 256 E4M3 bytes and all
// 16 E2M1 codes, which is what makes a bit-equality test against the old entry the right acceptance
// criterion.
//
// The old entry point stays exactly as it was. It is the oracle: `nvfp4_moe_rows_forward` remains
// textually unchanged in `cuda_kernel_impl.cu`, so the equality test in
// `tests/test_nvfp4_exact_fast.py` compares two independently compiled implementations of the same
// specification rather than a kernel against itself.
//
// Two deliberate non-changes, both required for exactness:
//
//   * E4M3 byte 0x7F (exponent 15, mantissa 7) decodes to 480, not NaN. The OCP format reserves
//     that encoding as NaN, but the old kernel treats every one of the 256 bytes as a finite scale.
//     This is a compatibility witness, not a claim that NaN encodings occur in the checkpoint.
//     The exact helper must retain those finite bit patterns rather than reinterpret the format.
//   * E2M1 code 0x8 is -0.0, not +0.0. Only the sign bit distinguishes them and the packed payload
//     carries it, so the sign is applied by toggling the float's sign bit rather than by negating.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------------------------
// Decoders. Each one is the exact operand the old helper produced, built without a load.
// ---------------------------------------------------------------------------------------------

// E4M3 byte to float, matching `nvfp4_e4m3` bit for bit on all 256 inputs.
//
// The old helper computes `ldexpf(1 + m/8, e - 7)` for e != 0 -- exact for every such input,
// because `1 + m/8` needs three mantissa bits and the scale is a power of two -- and
// `ldexpf(m/8, -6)` for e == 0, which is exactly `m * 2^-9`. An exactly scaled value has a bit
// pattern that can be written down: the biased exponent is `e + 120` and the three mantissa bits
// sit at the top of the fp32 mantissa field, so no rounding is involved on either path.
__device__ __forceinline__ float nvfp4_exact_e4m3(uint8_t byte) {
    const unsigned bits = static_cast<unsigned>(byte);
    const unsigned exponent = (bits >> 3) & 0xFU;
    const unsigned mantissa = bits & 0x7U;
    const unsigned normal = ((exponent + 120U) << 23) | (mantissa << 20);
    // e == 0: the subnormal range, mant * 2^-9. Exact as a single multiply: the multiplicand is an
    // integer below 8 and the multiplier is a power of two; every nonzero result is a normal float.
    const unsigned subnormal = __float_as_uint(static_cast<float>(mantissa) * 0x1p-9f);
    const unsigned magnitude = exponent == 0U ? subnormal : normal;
    return __uint_as_float(magnitude | ((bits & 0x80U) << 24));
}

// E2M1 nibble to float, matching `nvfp4_e2m1` bit for bit on all 16 codes.
//
// The magnitude set doubled is {0, 1, 2, 3, 4, 6, 8, 12}; packed one per nibble it is the 32-bit
// literal 0xC8643210, which reads back code 7 -> 0 as 12, 8, 6, 4, 3, 2, 1, 0. Shifting by
// `4 * code` keeps the whole table in an immediate, so a warp whose lanes carry different codes
// costs one shift and one mask instead of a serialised constant-memory gather.
__device__ __forceinline__ float nvfp4_exact_e2m1(uint8_t nibble) {
    const unsigned magnitude_x2 = (0xC8643210U >> (4U * (nibble & 0x7U))) & 0xFU;
    const float magnitude = static_cast<float>(magnitude_x2) * 0.5f;
    // Sign by bit toggle, so code 0x8 stays -0.0 instead of becoming +0.0.
    return __uint_as_float(__float_as_uint(magnitude) | (static_cast<unsigned>(nibble & 0x8U) << 28));
}

// ---------------------------------------------------------------------------------------------
// The two stages. Every float operation, loop bound, grid dimension, shared-memory layout and
// route-order decision below is a copy of `nvfp4_gate_up_kernel` / `nvfp4_down_kernel`; only the
// decoder calls differ. Identical source expressions are not a guarantee of identical compilation
// across translation units: the independent eager/captured GPU oracle remains mandatory.
// ---------------------------------------------------------------------------------------------

// Gate/up for every route, SwiGLU applied, into `act[topk][inter]` -- one row a route, because a
// route's down projection reads only its own SwiGLU output. One warp owns one column: a column is
// `hidden/2` contiguous bytes and the warp walks it 32 lanes at a time.
__global__ void nvfp4_exact_gate_up_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ w1q, const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q, const uint8_t* __restrict__ w3s,
    float* __restrict__ act,
    int topk, int n_local_experts, int experts_start_idx,
    int hidden, int inter, float swiglu_limit,
    const float* __restrict__ w_scale1, int rows) {
    extern __shared__ float sh[];
    float* s_x = sh;
    const int hblocks = hidden / 16;
    const int cols_per_block = (inter + gridDim.x - 1) / gridDim.x;
    const int col_lo = blockIdx.x * cols_per_block;
    const int col_hi = min(col_lo + cols_per_block, inter);
    if (col_lo >= inter) return;
    // One row per block in z, so a K-row call is the same kernel with a bigger grid rather than a
    // loop inside it. Each block loads its own row into shared memory, so no row sees another's x.
    const int row = blockIdx.z;
    if (row >= rows) return;
    const float* x_row = x + static_cast<int64_t>(row) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        s_x[i] = x_row[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int bytes_per_col = hidden / 2;
    const int64_t* idx_row = indices + static_cast<int64_t>(row) * topk;
    const float* wt_row = weights + static_cast<int64_t>(row) * topk;
    float* act_row = act + static_cast<int64_t>(row) * topk * inter;
    for (int r = 0; r < topk; ++r) {
        const int local = static_cast<int>(idx_row[r]) - experts_start_idx;
        if (local < 0 || local >= n_local_experts) continue;
        const float w = wt_row[r];
        const float s2_up = w_scale1[local];
        const uint8_t* g_row = w1q + (static_cast<int64_t>(local) * inter) * bytes_per_col;
        const uint8_t* u_row = w3q + (static_cast<int64_t>(local) * inter) * bytes_per_col;
        const uint8_t* g_scl = w1s + (static_cast<int64_t>(local) * inter) * hblocks;
        const uint8_t* u_scl = w3s + (static_cast<int64_t>(local) * inter) * hblocks;
        for (int col = col_lo + warp; col < col_hi; col += warps) {
            const uint8_t* gp = g_row + static_cast<int64_t>(col) * bytes_per_col;
            const uint8_t* up = u_row + static_cast<int64_t>(col) * bytes_per_col;
            const uint8_t* gs = g_scl + static_cast<int64_t>(col) * hblocks;
            const uint8_t* us = u_scl + static_cast<int64_t>(col) * hblocks;
            float acc_g = 0.0f, acc_u = 0.0f;
            // 8 bytes a lane: a warp covers 256 bytes of the row per pass.
            for (int base = 0; base < bytes_per_col; base += 32 * 8) {
                const int off = base + lane * 8;
                if (off < bytes_per_col) {
                    const uint64_t wg = *reinterpret_cast<const uint64_t*>(gp + off);
                    const uint64_t wu = *reinterpret_cast<const uint64_t*>(up + off);
                    const int gb = off / 8;
                    const float2* xp = reinterpret_cast<const float2*>(s_x + gb * 16);
                    float dg0 = 0.0f, dg1 = 0.0f, du0 = 0.0f, du1 = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float2 xv = xp[j];
                        const int sg = static_cast<int>((wg >> (j * 8)) & 0xFF);
                        const int su = static_cast<int>((wu >> (j * 8)) & 0xFF);
                        dg0 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sg & 0xF)), xv.x, dg0);
                        dg1 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sg >> 4)), xv.y, dg1);
                        du0 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(su & 0xF)), xv.x, du0);
                        du1 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(su >> 4)), xv.y, du1);
                    }
                    acc_g += (dg0 + dg1) * nvfp4_exact_e4m3(gs[gb]);
                    acc_u += (du0 + du1) * nvfp4_exact_e4m3(us[gb]);
                }
            }
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, d);
                acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, d);
            }
            if (lane != 0) continue;
            float g = acc_g * s2_up;
            float u = acc_u * s2_up;
            if (swiglu_limit > 0.0f) {
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
                g = fminf(g, swiglu_limit);
            }
            // No route weight here: the down stage applies `weights[r]` to the accumulated row, so
            // multiplying here would square the routing weight.
            act_row[static_cast<int64_t>(r) * inter + col] = (g / (1.0f + __expf(-g))) * u;
        }
    }
}

__global__ void nvfp4_exact_down_kernel(
    const float* __restrict__ act,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ w2q, const uint8_t* __restrict__ w2s,
    float* __restrict__ out,
    int topk, int n_local_experts, int experts_start_idx,
    int hidden, int inter,
    const float* __restrict__ w_scale2, int rows) {
    const int iblocks = inter / 16;
    const int cols_per_block = (hidden + gridDim.x - 1) / gridDim.x;
    const int col_lo = blockIdx.x * cols_per_block;
    const int col_hi = min(col_lo + cols_per_block, hidden);
    if (col_lo >= hidden) return;
    const int row = blockIdx.z;
    if (row >= rows) return;
    float* out_row = out + static_cast<int64_t>(row) * hidden;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int bytes_per_col = inter / 2;
    const int64_t* idx_row = indices + static_cast<int64_t>(row) * topk;
    const float* wt_row = weights + static_cast<int64_t>(row) * topk;
    const float* act_row = act + static_cast<int64_t>(row) * topk * inter;
    for (int r = 0; r < topk; ++r) {
        const int local = static_cast<int>(idx_row[r]) - experts_start_idx;
        if (local < 0 || local >= n_local_experts) continue;
        const float w = wt_row[r];
        const float s2_down = w_scale2[local];
        const uint8_t* d_row = w2q + (static_cast<int64_t>(local) * hidden) * bytes_per_col;
        const uint8_t* d_scl = w2s + (static_cast<int64_t>(local) * hidden) * iblocks;
        const float* row_act = act_row + static_cast<int64_t>(r) * inter;
        // One warp per output column, same shape as the gate/up stage.
        for (int col = col_lo + warp; col < col_hi; col += warps) {
            const uint8_t* dp = d_row + static_cast<int64_t>(col) * bytes_per_col;
            const uint8_t* ds = d_scl + static_cast<int64_t>(col) * iblocks;
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int base = 0; base < bytes_per_col; base += 32 * 8) {
                const int off = base + lane * 8;
                if (off >= bytes_per_col) continue;
                const uint64_t wd = *reinterpret_cast<const uint64_t*>(dp + off);
                const int gb = off / 8;
                const float2* ap = reinterpret_cast<const float2*>(row_act + gb * 16);
                float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 av = ap[j];
                    const int sd = static_cast<int>((wd >> (j * 8)) & 0xFF);
                    d0 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sd & 0xF)), av.x, d0);
                    d1 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sd >> 4)), av.y, d1);
                }
                const float sc = nvfp4_exact_e4m3(ds[gb]);
                acc0 = fmaf(d0, sc, acc0);
                acc1 = fmaf(d1, sc, acc1);
            }
            float acc = acc0 + acc1;
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);
            }
            if (lane != 0) continue;
            atomicAdd(out_row + col, acc * w * s2_down);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder witness kernels. These exist so a test can read the two decoders' outputs as float bits
// instead of inferring them from a MoE result; they are diagnostic entry points, not part of a
// forward path, and they do no arithmetic beyond the decode.
// ---------------------------------------------------------------------------------------------

__global__ void nvfp4_exact_decode_e2m1_kernel(
    const uint8_t* __restrict__ codes,
    float* __restrict__ values,
    int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = nvfp4_exact_e2m1(static_cast<uint8_t>(codes[index] & 0xFU));
    }
}

__global__ void nvfp4_exact_decode_e4m3_kernel(
    const uint8_t* __restrict__ bytes,
    float* __restrict__ values,
    int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = nvfp4_exact_e4m3(bytes[index]);
    }
}

}  // namespace

namespace {

// Output columns one CTA is responsible for. This is the launch-geometry constant the grid is
// derived from, so it is deliberately independent of how many threads a CTA is launched with.
constexpr int kColumnsPerBlock = 4;
// Threads the oracle entry has always launched with: eight warps for four columns, so four of
// them were idle in the column loop. Kept as the oracle's exact launch configuration.
constexpr int kOracleThreads = 256;
// Threads for the compact entry: exactly the four warps a block can keep busy.
constexpr int kCompactThreads = 128;

// The gate/up stage, launched from one place. Both the shared body below and the ordered-store body
// call this, which is what makes "the ordered store reuses the qualified gate stage" a fact about
// the source rather than a promise about two copies of it: there is only one launch of
// `nvfp4_exact_gate_up_kernel` in the file, so the stage cannot drift between the entries.
//
// `threads_per_block` cannot change the arithmetic. The grid is derived from `kColumnsPerBlock`,
// every `act` column is owned by exactly one warp on either launch, and a block's column range is
// at most `kColumnsPerBlock` wide, so the four-warp and eight-warp launches write the same values.
// The guard refuses a launch too small to cover a block's columns -- that would be a silent wrong
// answer, not a rounding difference.
void launch_gate_up(
    const float* x,
    const int64_t* indices,
    const float* weights,
    const uint8_t* w1q, const uint8_t* w1s,
    const uint8_t* w3q, const uint8_t* w3s,
    float* act,
    int topk, int n_local_experts, int experts_start_idx,
    int hidden, int inter, float swiglu_limit,
    const float* w_scale1, int rows,
    int threads_per_block) {
    const int warps_per_block = kColumnsPerBlock;
    TORCH_CHECK(threads_per_block % 32 == 0 && threads_per_block >= warps_per_block * 32,
                "the exact NVFP4 rows kernels need at least ", warps_per_block,
                " whole warps a block, got ", threads_per_block, " threads");
    const int gate_blocks = static_cast<int>((inter + warps_per_block - 1) / warps_per_block);
    const size_t gate_shmem = static_cast<size_t>(hidden) * sizeof(float);
    nvfp4_exact_gate_up_kernel<<<dim3(static_cast<unsigned>(gate_blocks), 1,
                                      static_cast<unsigned>(rows)),
                                 threads_per_block, gate_shmem,
                                 at::cuda::getCurrentCUDAStream()>>>(
        x, indices, weights,
        w1q, w1s, w3q, w3s,
        act,
        topk, n_local_experts, experts_start_idx,
        hidden, inter, swiglu_limit,
        w_scale1, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K-row ModelOpt NVFP4 MoE with register-decoded operands. Signature, guards and launch geometry
// are the old `nvfp4_moe_rows_forward`'s; the guards are copied verbatim rather than tightened so
// the two entries accept exactly the same inputs and can be compared on them. `x_scale`,
// `down_scale` and `activation_fp4` are part of that shared signature and are unused here for the
// same reason they are unused in the old rows path: they belong to the W4A4 diagnostic stage, and
// inventing activation quantisation in this kernel would change the answer instead of the code.
//
// `threads_per_block` is the *only* thing the two public entries below differ in, so it is a
// parameter of this one body rather than a second copy of it. It cannot change the arithmetic:
// the grid and `cols_per_block` are computed from `kColumnsPerBlock`, every output column is owned
// by exactly one warp on both launches, and each column's accumulation loop is unchanged. What it
// changes is how many warps a CTA allocates, and therefore how many idle warp slots the scheduler
// carries. That is a performance question for the GPU oracle, not a specification question.
torch::Tensor run_rows_exact(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q, const torch::Tensor& w1s,
    const torch::Tensor& w3q, const torch::Tensor& w3s,
    const torch::Tensor& w2q, const torch::Tensor& w2s,
    int64_t experts_start_idx,
    double swiglu_limit,
    double x_scale,
    double down_scale,
    const torch::Tensor& w_scale1,
    const torch::Tensor& w_scale2,
    bool activation_fp4,
    int threads_per_block) {
    (void)x_scale;
    (void)down_scale;
    (void)activation_fp4;
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1, "x must be [rows, hidden]");
    TORCH_CHECK(w_scale1.numel() == w1q.size(0) && w_scale2.numel() == w2q.size(0),
                "weight_scale_2 must hold one value per expert");
    TORCH_CHECK(indices.dim() == 2 && weights.dim() == 2, "indices/weights must be [rows, k]");
    TORCH_CHECK(indices.sizes() == weights.sizes(), "indices/weights shape mismatch");
    TORCH_CHECK(indices.size(0) == x.size(0), "one route row per activation row");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3,
                "weights must be [E, out, in/2]");
    const int64_t hidden = w1q.size(2) * 2;
    const int64_t inter = w1q.size(1);
    TORCH_CHECK(hidden % 16 == 0 && inter % 16 == 0, "hidden and inter must be multiples of 16");
    TORCH_CHECK(w1s.size(2) == hidden / 16 && w3s.size(2) == hidden / 16,
                "w1/w3 scales must be [E, inter, hidden/16] for group_size 16");
    TORCH_CHECK(w2s.size(1) == hidden && w2s.size(2) == inter / 16,
                "w2 scales must be [E, hidden, inter/16] for group_size 16");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be a contiguous CUDA tensor");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous(), "weights must be a contiguous CUDA tensor");
    TORCH_CHECK(w1q.is_cuda() && w1q.is_contiguous(), "w1q must be a contiguous CUDA tensor");
    TORCH_CHECK(w1s.is_cuda() && w1s.is_contiguous(), "w1s must be a contiguous CUDA tensor");
    TORCH_CHECK(w3q.is_cuda() && w3q.is_contiguous(), "w3q must be a contiguous CUDA tensor");
    TORCH_CHECK(w3s.is_cuda() && w3s.is_contiguous(), "w3s must be a contiguous CUDA tensor");
    TORCH_CHECK(w2q.is_cuda() && w2q.is_contiguous(), "w2q must be a contiguous CUDA tensor");
    TORCH_CHECK(w2s.is_cuda() && w2s.is_contiguous(), "w2s must be a contiguous CUDA tensor");

    const int64_t rows = x.size(0);
    const int topk = static_cast<int>(indices.size(1));
    auto out = torch::zeros({rows, hidden}, x.options());
    // Zeroed, not empty: under a captured graph the allocation address is fixed and reused, so a
    // route this rank does not own would otherwise leave the previous step's SwiGLU row there for
    // the down stage to read.
    auto act = torch::zeros({rows * topk, inter}, x.options());
    const int warps_per_block = kColumnsPerBlock;
    launch_gate_up(x.data_ptr<float>(),
                   indices.data_ptr<int64_t>(),
                   weights.data_ptr<float>(),
                   w1q.data_ptr<uint8_t>(), w1s.data_ptr<uint8_t>(),
                   w3q.data_ptr<uint8_t>(), w3s.data_ptr<uint8_t>(),
                   act.data_ptr<float>(),
                   topk,
                   static_cast<int>(w1q.size(0)),
                   static_cast<int>(experts_start_idx),
                   static_cast<int>(hidden),
                   static_cast<int>(inter),
                   static_cast<float>(swiglu_limit),
                   w_scale1.data_ptr<float>(),
                   static_cast<int>(rows),
                   threads_per_block);
    const int down_blocks = static_cast<int>((hidden + warps_per_block - 1) / warps_per_block);
    nvfp4_exact_down_kernel<<<dim3(static_cast<unsigned>(down_blocks), 1,
                                   static_cast<unsigned>(rows)),
                              threads_per_block, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        act.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        w2q.data_ptr<uint8_t>(), w2s.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        topk,
        static_cast<int>(w1q.size(0)),
        static_cast<int>(experts_start_idx),
        static_cast<int>(hidden),
        static_cast<int>(inter),
        w_scale2.data_ptr<float>(),
        static_cast<int>(rows));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

// The oracle entry: the exact-operand rows path with the launch configuration it has always used.
// Eight warps a block for a four-column block means half of them are idle in the column loop, which
// is deliberately preserved here so the equality oracle is the implementation that shipped.
torch::Tensor nvfp4_moe_rows_forward_exact_fast(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q, const torch::Tensor& w1s,
    const torch::Tensor& w3q, const torch::Tensor& w3s,
    const torch::Tensor& w2q, const torch::Tensor& w2s,
    int64_t experts_start_idx,
    double swiglu_limit,
    double x_scale,
    double down_scale,
    const torch::Tensor& w_scale1,
    const torch::Tensor& w_scale2,
    bool activation_fp4) {
    return run_rows_exact(x, indices, weights, w1q, w1s, w3q, w3s, w2q, w2s,
                          experts_start_idx, swiglu_limit, x_scale, down_scale,
                          w_scale1, w_scale2, activation_fp4, kOracleThreads);
}

// The compact entry: the same kernels, the same grid and the same column partition, launched with
// exactly the four warps a block can keep busy. This is the entry the model's forward path selects
// when it wants the launch without the idle warp slots; the oracle above stays the equality
// reference, so selecting this one cannot change the answer by construction -- and the GPU test
// asserts that bit for bit rather than trusting the argument.
torch::Tensor nvfp4_moe_rows_forward_exact_fast_compact(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q, const torch::Tensor& w1s,
    const torch::Tensor& w3q, const torch::Tensor& w3s,
    const torch::Tensor& w2q, const torch::Tensor& w2s,
    int64_t experts_start_idx,
    double swiglu_limit,
    double x_scale,
    double down_scale,
    const torch::Tensor& w_scale1,
    const torch::Tensor& w_scale2,
    bool activation_fp4) {
    return run_rows_exact(x, indices, weights, w1q, w1s, w3q, w3s, w2q, w2s,
                          experts_start_idx, swiglu_limit, x_scale, down_scale,
                          w_scale1, w_scale2, activation_fp4, kCompactThreads);
}

// Witness for the E2M1 decoder: one float per incoming nibble, taken from the low four bits.
torch::Tensor nvfp4_exact_decode_e2m1(const torch::Tensor& codes) {
    TORCH_CHECK(codes.is_cuda(), "codes must be a CUDA tensor");
    TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
    TORCH_CHECK(codes.scalar_type() == torch::kUInt8, "codes must be uint8");
    auto values = torch::empty(codes.sizes(), codes.options().dtype(torch::kFloat32));
    const int64_t count = codes.numel();
    if (count != 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        nvfp4_exact_decode_e2m1_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            codes.data_ptr<uint8_t>(), values.data_ptr<float>(), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return values;
}

// Witness for the E4M3 decoder: one float per incoming scale byte.
torch::Tensor nvfp4_exact_decode_e4m3(const torch::Tensor& bytes) {
    TORCH_CHECK(bytes.is_cuda(), "bytes must be a CUDA tensor");
    TORCH_CHECK(bytes.is_contiguous(), "bytes must be contiguous");
    TORCH_CHECK(bytes.scalar_type() == torch::kUInt8, "bytes must be uint8");
    auto values = torch::empty(bytes.sizes(), bytes.options().dtype(torch::kFloat32));
    const int64_t count = bytes.numel();
    if (count != 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        nvfp4_exact_decode_e4m3_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            bytes.data_ptr<uint8_t>(), values.data_ptr<float>(), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return values;
}

// =============================================================================================
// Ordered single-writer epilogue.
//
// What changes, and why the answer must not. The entries above keep the original shape: a route
// outer loop and a column inner loop, with each route's scalar contribution added straight into
// `out` by a global float atomic -- one read-modify-write of the destination per route. This entry
// turns the loops inside out, a column outer loop with the routes inside, so one output element's
// contributions accumulate in a register and reach `out` in a single store.
//
// Three facts make that a relocation rather than a rounding change. Each one is asserted by a test
// here or in `tests/test_nvfp4_exact_store.py`, not assumed:
//
//   * Ownership. `grid.y` is 1, `blockIdx.z` is exactly one row, the `grid.x` blocks partition the
//     output columns contiguously and disjointly, and a block's column range is at most
//     `kColumnsPerBlock` wide while the launch supplies exactly that many warps. Every
//     `(row, column)` therefore belongs to one warp in one block, and only lane 0 of that warp
//     touches it. The old entry's atomics on one address already came from a single thread in
//     ascending route order: there was never a race to lose, only an atomic to pay for.
//   * Arithmetic. The per-route dot product, its FMA spelling, the five-step warp reduction and the
//     two multiplies forming the routed contribution are copied from the old kernel character for
//     character, so lane 0's scalar is bit-identical. The accumulation and the product are stated
//     in PTX -- `add.rn.ftz.f32`, and `mul.rn.ftz.f32` left-associatively -- which is what
//     `--use_fast_math` compiles those same C expressions to. The inline asm also refuses any
//     multiply-add contraction of the product into the accumulation.
//   * Completeness. Every output element is stored, so `out` need not start zeroed; and `act` need
//     not be zeroed either, because the gate stage writes every column of every owned route and the
//     down stage skips unowned routes before it forms a `row_act` pointer. Both are argued at the
//     kernels below.
//
// The one thing source alone cannot settle is the atomic's qualifier: whether the old
// `atomicAdd(float*)` under `-ftz=true` emits a flushing `red.add.ftz.f32` or a non-flushing
// `add.f32`. This entry takes the flushing form, because the products it accumulates are themselves
// formed by `mul.ftz.f32` and so can never be subnormal. The only construct that tells the two
// apart is a pair of normal products whose SUM is subnormal, and
// `tests/test_nvfp4_exact_store.py` carries exactly that case so the device answers it rather than
// the reasoning. Until that oracle passes, this entry is an independent experiment, not the
// default, and this file does not claim it is exact.
// =============================================================================================

namespace {

// IEEE-754 round-to-nearest-even float32 multiply with the denormal flush the project's
// `-ftz=true` codegen applies, stated in PTX so no compiler flag or contraction can retarget it.
__device__ __forceinline__ float nvfp4_store_mul_rn_ftz(float lhs, float rhs) {
    float product;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(lhs), "f"(rhs));
    return product;
}

// The add the old entry's `atomicAdd(float*)` performs on its memory destination: one
// round-to-nearest float32 addition that flushes denormals.
__device__ __forceinline__ float nvfp4_store_add_rn_ftz(float lhs, float rhs) {
    float sum;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(lhs), "f"(rhs));
    return sum;
}

// Down projection of every route's SwiGLU row, accumulated in a register and stored once.
//
// The dot-product body and the warp reduction below are `nvfp4_exact_down_kernel`'s, unchanged.
// Only the loop nesting and the epilogue differ: the old kernel entered the column loop once per
// route and issued a global atomic per route, while this one owns a column across the whole route
// sweep and writes it once.
__global__ void nvfp4_exact_down_store_kernel(
    const float* __restrict__ act,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ w2q, const uint8_t* __restrict__ w2s,
    float* __restrict__ out,
    int topk, int n_local_experts, int experts_start_idx,
    int hidden, int inter,
    const float* __restrict__ w_scale2, int rows) {
    const int iblocks = inter / 16;
    const int cols_per_block = (hidden + gridDim.x - 1) / gridDim.x;
    const int col_lo = blockIdx.x * cols_per_block;
    const int col_hi = min(col_lo + cols_per_block, hidden);
    if (col_lo >= hidden) return;
    const int row = blockIdx.z;
    if (row >= rows) return;
    float* out_row = out + static_cast<int64_t>(row) * hidden;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int bytes_per_col = inter / 2;
    const int64_t* idx_row = indices + static_cast<int64_t>(row) * topk;
    const float* wt_row = weights + static_cast<int64_t>(row) * topk;
    const float* act_row = act + static_cast<int64_t>(row) * topk * inter;
    for (int col = col_lo + warp; col < col_hi; col += warps) {
        // Starts at +0.0f, the value the old entry's zero-initialised destination held before its
        // first atomic, so a row with no owned route stores +0.0f -- the same bits it used to keep.
        float routed = 0.0f;
        for (int r = 0; r < topk; ++r) {
            const int local = static_cast<int>(idx_row[r]) - experts_start_idx;
            // Skipped before `row_act` is even formed. This is what makes an uninitialised `act`
            // honest: a route this rank does not own never has its SwiGLU row read, so nothing
            // depended on the buffer having been zeroed.
            if (local < 0 || local >= n_local_experts) continue;
            const float w = wt_row[r];
            const float s2_down = w_scale2[local];
            const uint8_t* dp = w2q + (static_cast<int64_t>(local) * hidden) * bytes_per_col
                              + static_cast<int64_t>(col) * bytes_per_col;
            const uint8_t* ds = w2s + (static_cast<int64_t>(local) * hidden) * iblocks
                              + static_cast<int64_t>(col) * iblocks;
            const float* row_act = act_row + static_cast<int64_t>(r) * inter;
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int base = 0; base < bytes_per_col; base += 32 * 8) {
                const int off = base + lane * 8;
                if (off >= bytes_per_col) continue;
                const uint64_t wd = *reinterpret_cast<const uint64_t*>(dp + off);
                const int gb = off / 8;
                const float2* ap = reinterpret_cast<const float2*>(row_act + gb * 16);
                float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 av = ap[j];
                    const int sd = static_cast<int>((wd >> (j * 8)) & 0xFF);
                    d0 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sd & 0xF)), av.x, d0);
                    d1 = fmaf(nvfp4_exact_e2m1(static_cast<uint8_t>(sd >> 4)), av.y, d1);
                }
                const float sc = nvfp4_exact_e4m3(ds[gb]);
                acc0 = fmaf(d0, sc, acc0);
                acc1 = fmaf(d1, sc, acc1);
            }
            float acc = acc0 + acc1;
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);
            }
            if (lane == 0) {
                // The old entry's `atomicAdd(out_row + col, acc * w * s2_down)`. The product is
                // `(acc * w) * s2_down`: two multiplies in that order, not a reassociation, and the
                // addition is one round-to-nearest float32 add into the destination.
                const float contribution = nvfp4_store_mul_rn_ftz(
                    nvfp4_store_mul_rn_ftz(acc, w), s2_down);
                routed = nvfp4_store_add_rn_ftz(routed, contribution);
            }
        }
        if (lane == 0) {
            // One store, once, for this column. Every column of every row is written, so `out` can
            // be allocated uninitialised.
            out_row[col] = routed;
        }
    }
}

// K-row ModelOpt NVFP4 MoE with the ordered single-writer epilogue.
//
// The guards are the oracle entry's, copied verbatim rather than tightened, so the two accept
// exactly the same inputs and can be compared on all of them. `x_scale`, `down_scale` and
// `activation_fp4` are unused for the same reason they are unused there: they belong to the W4A4
// diagnostic stage, and inventing activation quantisation would change the answer instead of the
// code.
torch::Tensor run_rows_exact_store(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q, const torch::Tensor& w1s,
    const torch::Tensor& w3q, const torch::Tensor& w3s,
    const torch::Tensor& w2q, const torch::Tensor& w2s,
    int64_t experts_start_idx,
    double swiglu_limit,
    double x_scale,
    double down_scale,
    const torch::Tensor& w_scale1,
    const torch::Tensor& w_scale2,
    bool activation_fp4) {
    (void)x_scale;
    (void)down_scale;
    (void)activation_fp4;
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1, "x must be [rows, hidden]");
    TORCH_CHECK(w_scale1.numel() == w1q.size(0) && w_scale2.numel() == w2q.size(0),
                "weight_scale_2 must hold one value per expert");
    TORCH_CHECK(indices.dim() == 2 && weights.dim() == 2, "indices/weights must be [rows, k]");
    TORCH_CHECK(indices.sizes() == weights.sizes(), "indices/weights shape mismatch");
    TORCH_CHECK(indices.size(0) == x.size(0), "one route row per activation row");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3,
                "weights must be [E, out, in/2]");
    const int64_t hidden = w1q.size(2) * 2;
    const int64_t inter = w1q.size(1);
    TORCH_CHECK(hidden % 16 == 0 && inter % 16 == 0, "hidden and inter must be multiples of 16");
    TORCH_CHECK(w1s.size(2) == hidden / 16 && w3s.size(2) == hidden / 16,
                "w1/w3 scales must be [E, inter, hidden/16] for group_size 16");
    TORCH_CHECK(w2s.size(1) == hidden && w2s.size(2) == inter / 16,
                "w2 scales must be [E, hidden, inter/16] for group_size 16");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be a contiguous CUDA tensor");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous(), "weights must be a contiguous CUDA tensor");
    TORCH_CHECK(w1q.is_cuda() && w1q.is_contiguous(), "w1q must be a contiguous CUDA tensor");
    TORCH_CHECK(w1s.is_cuda() && w1s.is_contiguous(), "w1s must be a contiguous CUDA tensor");
    TORCH_CHECK(w3q.is_cuda() && w3q.is_contiguous(), "w3q must be a contiguous CUDA tensor");
    TORCH_CHECK(w3s.is_cuda() && w3s.is_contiguous(), "w3s must be a contiguous CUDA tensor");
    TORCH_CHECK(w2q.is_cuda() && w2q.is_contiguous(), "w2q must be a contiguous CUDA tensor");
    TORCH_CHECK(w2s.is_cuda() && w2s.is_contiguous(), "w2s must be a contiguous CUDA tensor");

    const int64_t rows = x.size(0);
    const int topk = static_cast<int>(indices.size(1));
    // Uninitialised, not zeroed. `out`: the single-writer down stage stores every element, so no
    // element keeps a stale value. `act`: the gate stage writes every column of every owned route,
    // and the down stage's owned-route guard skips a foreign route before it forms the row pointer,
    // so a row the gate stage left alone is never read. The shared body above keeps both fills --
    // that is its shipped behaviour, not a requirement this entry inherits -- and
    // `tests/test_nvfp4_exact_store.py` asserts the column coverage that replaces them here.
    auto out = torch::empty({rows, hidden}, x.options());
    auto act = torch::empty({rows * topk, inter}, x.options());
    // The four-warp launch the compact entry qualified: the same one `kCompactThreads` names, reused
    // here rather than declared a second time so the launch geometry has one definition.
    const int threads_per_block = kCompactThreads;
    const int warps_per_block = kColumnsPerBlock;
    launch_gate_up(x.data_ptr<float>(),
                   indices.data_ptr<int64_t>(),
                   weights.data_ptr<float>(),
                   w1q.data_ptr<uint8_t>(), w1s.data_ptr<uint8_t>(),
                   w3q.data_ptr<uint8_t>(), w3s.data_ptr<uint8_t>(),
                   act.data_ptr<float>(),
                   topk,
                   static_cast<int>(w1q.size(0)),
                   static_cast<int>(experts_start_idx),
                   static_cast<int>(hidden),
                   static_cast<int>(inter),
                   static_cast<float>(swiglu_limit),
                   w_scale1.data_ptr<float>(),
                   static_cast<int>(rows),
                   threads_per_block);
    const int down_blocks = static_cast<int>((hidden + warps_per_block - 1) / warps_per_block);
    nvfp4_exact_down_store_kernel<<<dim3(static_cast<unsigned>(down_blocks), 1,
                                         static_cast<unsigned>(rows)),
                                    threads_per_block, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
        act.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        w2q.data_ptr<uint8_t>(), w2s.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        topk,
        static_cast<int>(w1q.size(0)),
        static_cast<int>(experts_start_idx),
        static_cast<int>(hidden),
        static_cast<int>(inter),
        w_scale2.data_ptr<float>(),
        static_cast<int>(rows));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

// Ordered single-writer store of the exact-operand rows path. An independent experiment until the
// device oracle in `tests/test_nvfp4_exact_store.py` passes: it changes where the routed
// contributions are summed (a register instead of the destination) and removes both zero fills.
// The entries above remain the specification and the default.
torch::Tensor nvfp4_moe_rows_forward_exact_fast_store(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q, const torch::Tensor& w1s,
    const torch::Tensor& w3q, const torch::Tensor& w3s,
    const torch::Tensor& w2q, const torch::Tensor& w2s,
    int64_t experts_start_idx,
    double swiglu_limit,
    double x_scale,
    double down_scale,
    const torch::Tensor& w_scale1,
    const torch::Tensor& w_scale2,
    bool activation_fp4) {
    return run_rows_exact_store(x, indices, weights, w1q, w1s, w3q, w3s, w2q, w2s,
                                experts_start_idx, swiglu_limit, x_scale, down_scale,
                                w_scale1, w_scale2, activation_fp4);
}