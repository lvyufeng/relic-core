#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kDim = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxHeadsPerKv = 12;
constexpr int kMmaRows = 16;
constexpr int kMmaKvTile = 32;
constexpr int kMmaSlices = 4;
constexpr int kMmaSteps = kDim / 8;
constexpr int kMmaWarps = kMmaSlices;
constexpr int kMmaThreads = kMmaWarps * kWarpSize;
constexpr int kMmaKsStride = kDim + 8;
constexpr int kMmaVtStride = kMmaKvTile + 2;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define QWEN4_EXP_QSA_MMA_AVAILABLE 1
#endif

union HalfBits {
    __half value;
    uint16_t bits;
};

__device__ __forceinline__ uint16_t float_to_half_bits(float value) {
    HalfBits converted;
    converted.value = __float2half_rn(value);
    return converted.bits;
}

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
    HalfBits converted;
    converted.bits = bits;
    return __half2float(converted.value);
}

__device__ __forceinline__ uint32_t pack_half2(float first, float second) {
    return static_cast<uint32_t>(float_to_half_bits(first)) |
        (static_cast<uint32_t>(float_to_half_bits(second)) << 16);
}

__device__ __forceinline__ void mma_m16n8k8_f16_f32(
    float (&d)[4], const uint32_t (&a)[2], uint32_t b) {
#ifdef QWEN4_EXP_QSA_MMA_AVAILABLE
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    (void)d;
    (void)a;
    (void)b;
#endif
}

template <int HeadsPerKv>
__global__ __launch_bounds__(kMmaThreads) void qsa_mma_bf16_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    c10::BFloat16* __restrict__ output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    __shared__ __align__(16) uint16_t ks[kMmaKvTile][kMmaKsStride];
    __shared__ __align__(16) uint16_t vt[kDim][kMmaVtStride];
    __shared__ float scores[kMmaRows][kMmaKvTile];
    __shared__ __align__(16) uint16_t probabilities[kMmaRows][kMmaVtStride];
    __shared__ float row_max[kMmaRows];
    __shared__ float row_sum[kMmaRows];
    __shared__ float row_alpha[kMmaRows];
    __shared__ int32_t token_ids[kMmaKvTile];

    const int query_row = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch_index = query_row / query_len;
    const int query_index = query_row - batch_index * query_len;
    if (batch_index >= batch || kv_head >= kv_heads) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int slice = warp;
    const int lane = tid & (kWarpSize - 1);
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int row0 = gid;
    const int row1 = gid + 8;
    const int first_head = kv_head * HeadsPerKv;
    const int active_heads = min(HeadsPerKv, query_heads - first_head);
    const int64_t query_row_base =
        (static_cast<int64_t>(batch_index) * query_len + query_index) *
        query_heads * kDim;
    const int64_t kv_batch_stride =
        static_cast<int64_t>(kv_heads) * kv_len * kDim;
    const int64_t kv_head_base =
        static_cast<int64_t>(batch_index) * kv_batch_stride +
        static_cast<int64_t>(kv_head) * kv_len * kDim;
    const int32_t* selected_row =
        selected + (static_cast<int64_t>(batch_index) * query_len + query_index) *
                       selected_len;

    uint32_t query_fragment[kMmaSteps][2];
#pragma unroll
    for (int step = 0; step < kMmaSteps; ++step) {
        const int dim = step * 8 + tig * 2;
        float first0 = 0.0f;
        float second0 = 0.0f;
        float first1 = 0.0f;
        float second1 = 0.0f;
        if (row0 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row0) * kDim + dim;
            first0 = static_cast<float>(query[at]);
            second0 = static_cast<float>(query[at + 1]);
        }
        if (row1 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row1) * kDim + dim;
            first1 = static_cast<float>(query[at]);
            second1 = static_cast<float>(query[at + 1]);
        }
        query_fragment[step][0] = pack_half2(first0, second0);
        query_fragment[step][1] = pack_half2(first1, second1);
    }

    float accumulator[8][4] = {};
    if (tid < kMmaRows) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int selected_base = 0; selected_base < selected_len;
         selected_base += kMmaKvTile) {
        const int count = min(kMmaKvTile, selected_len - selected_base);
        if (tid < kMmaKvTile) {
            token_ids[tid] = tid < count
                ? selected_row[selected_base + tid]
                : -1;
        }
        __syncthreads();

        for (int idx = tid; idx < kMmaKvTile * kMmaSteps;
             idx += kMmaThreads) {
            const int slot = idx / kMmaSteps;
            const int chunk = idx - slot * kMmaSteps;
            const int token = token_ids[slot];
            uint16_t key_half[8] = {};
            uint16_t value_half[8] = {};
            if (token >= 0 && token < kv_len) {
                const int64_t at = kv_head_base +
                    static_cast<int64_t>(token) * kDim + chunk * 8;
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    key_half[item] =
                        float_to_half_bits(static_cast<float>(key[at + item]));
                    value_half[item] =
                        float_to_half_bits(static_cast<float>(value[at + item]));
                }
            }
            *reinterpret_cast<uint4*>(&ks[slot][chunk * 8]) =
                *reinterpret_cast<const uint4*>(key_half);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                vt[chunk * 8 + item][slot] = value_half[item];
            }
        }
        __syncthreads();

        float score_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int step = 0; step < kMmaSteps; ++step) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(
                &ks[slice * 8 + gid][step * 8 + tig * 2]);
            mma_m16n8k8_f16_f32(score_fragment, query_fragment[step], b);
        }
        const int score_col = slice * 8 + tig * 2;
        scores[row0][score_col] = score_fragment[0];
        scores[row0][score_col + 1] = score_fragment[1];
        scores[row1][score_col] = score_fragment[2];
        scores[row1][score_col + 1] = score_fragment[3];
        __syncthreads();

        if (tid < kMmaRows) {
            const int head_slot = tid;
            const float old_max = row_max[head_slot];
            const float old_sum = row_sum[head_slot];
            float tile_max = -INFINITY;
            float scaled_scores[kMmaKvTile];
#pragma unroll
            for (int col = 0; col < kMmaKvTile; ++col) {
                const float score = head_slot < active_heads &&
                        token_ids[col] >= 0 && token_ids[col] < kv_len
                    ? scores[head_slot][col] * softmax_scale
                    : -INFINITY;
                scaled_scores[col] = score;
                tile_max = fmaxf(tile_max, score);
            }
            const float next_max = fmaxf(old_max, tile_max);
            const float alpha = old_max == -INFINITY
                ? 0.0f
                : expf(old_max - next_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (int col = 0; col < kMmaKvTile; ++col) {
                uint16_t bits = 0;
                if (scaled_scores[col] != -INFINITY) {
                    bits = float_to_half_bits(
                        expf(scaled_scores[col] - next_max));
                    tile_sum += half_bits_to_float(bits);
                }
                probabilities[head_slot][col] = bits;
            }
            row_alpha[head_slot] = alpha;
            row_max[head_slot] = next_max;
            row_sum[head_slot] = old_sum * alpha + tile_sum;
        }
        __syncthreads();

        const float alpha0 = row_alpha[row0];
        const float alpha1 = row_alpha[row1];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            accumulator[item][0] *= alpha0;
            accumulator[item][1] *= alpha0;
            accumulator[item][2] *= alpha1;
            accumulator[item][3] *= alpha1;
        }
#pragma unroll
        for (int kt = 0; kt < kMmaKvTile / 8; ++kt) {
            uint32_t a[2];
            a[0] = *reinterpret_cast<const uint32_t*>(
                &probabilities[row0][kt * 8 + tig * 2]);
            a[1] = *reinterpret_cast<const uint32_t*>(
                &probabilities[row1][kt * 8 + tig * 2]);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const uint32_t b = *reinterpret_cast<const uint32_t*>(
                    &vt[slice * 64 + item * 8 + gid][kt * 8 + tig * 2]);
                mma_m16n8k8_f16_f32(accumulator[item], a, b);
            }
        }
        __syncthreads();
    }

    const float inverse0 = row0 < active_heads && row_sum[row0] > 0.0f
        ? 1.0f / row_sum[row0]
        : 0.0f;
    const float inverse1 = row1 < active_heads && row_sum[row1] > 0.0f
        ? 1.0f / row_sum[row1]
        : 0.0f;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = slice * 64 + item * 8 + tig * 2;
        if (row0 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row0) * kDim + dim;
            output[at] = c10::BFloat16(accumulator[item][0] * inverse0);
            output[at + 1] = c10::BFloat16(accumulator[item][1] * inverse0);
        }
        if (row1 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row1) * kDim + dim;
            output[at] = c10::BFloat16(accumulator[item][2] * inverse1);
            output[at + 1] = c10::BFloat16(accumulator[item][3] * inverse1);
        }
    }
}

template <int HeadsPerKv>
__global__ void qsa_scalar_bf16_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    c10::BFloat16* __restrict__ output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    const int query_row = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch_index = query_row / query_len;
    const int query_index = query_row - batch_index * query_len;
    if (batch_index >= batch || kv_head >= kv_heads) return;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int head_slot = static_cast<int>(threadIdx.x) / kWarpSize;
    const int query_head = kv_head * HeadsPerKv + head_slot;
    if (head_slot >= HeadsPerKv || query_head >= query_heads) return;

    constexpr unsigned kFullMask = 0xffffffffU;
    const int64_t query_base =
        ((static_cast<int64_t>(batch_index) * query_len + query_index) *
             query_heads +
         query_head) *
        kDim;

    float query_fragment[8];
    float output_fragment[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = lane + item * kWarpSize;
        query_fragment[item] = static_cast<float>(query[query_base + dim]);
        output_fragment[item] = 0.0f;
    }

    const int32_t* selected_row =
        selected + (static_cast<int64_t>(batch_index) * query_len + query_index) *
                       selected_len;
    const int64_t kv_batch_stride =
        static_cast<int64_t>(kv_heads) * kv_len * kDim;
    const int64_t kv_head_base =
        static_cast<int64_t>(batch_index) * kv_batch_stride +
        static_cast<int64_t>(kv_head) * kv_len * kDim;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (int selected_index = 0; selected_index < selected_len;
         ++selected_index) {
        int token = lane == 0 ? selected_row[selected_index] : 0;
        token = __shfl_sync(kFullMask, token, 0);
        if (token < 0 || token >= kv_len) continue;

        const int64_t token_base =
            kv_head_base + static_cast<int64_t>(token) * kDim;
        float sum = 0.0f;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int dim = lane + item * kWarpSize;
            sum = fmaf(
                query_fragment[item],
                static_cast<float>(key[token_base + dim]),
                sum);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(kFullMask, sum, offset);
        }

        float alpha = 0.0f;
        float weight = 0.0f;
        if (lane == 0) {
            const float score = sum * softmax_scale;
            const float next_max = fmaxf(running_max, score);
            alpha = running_max == -INFINITY
                ? 0.0f
                : expf(running_max - next_max);
            weight = expf(score - next_max);
            running_max = next_max;
            running_sum = running_sum * alpha + weight;
        }
        alpha = __shfl_sync(kFullMask, alpha, 0);
        weight = __shfl_sync(kFullMask, weight, 0);

#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int dim = lane + item * kWarpSize;
            const float value_item = static_cast<float>(value[token_base + dim]);
            output_fragment[item] =
                output_fragment[item] * alpha + value_item * weight;
        }
    }

    float inverse = lane == 0 && running_sum > 0.0f
        ? 1.0f / running_sum
        : 0.0f;
    inverse = __shfl_sync(kFullMask, inverse, 0);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = lane + item * kWarpSize;
        output[query_base + dim] =
            c10::BFloat16(output_fragment[item] * inverse);
    }
}

template <int HeadsPerKv>
void launch_qsa(
    const c10::BFloat16* query,
    const c10::BFloat16* key,
    const c10::BFloat16* value,
    const int32_t* selected,
    c10::BFloat16* output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale,
    bool use_mma,
    cudaStream_t stream) {
    static_assert(HeadsPerKv >= 1 && HeadsPerKv <= kMaxHeadsPerKv);
    const dim3 grid(batch * query_len, kv_heads);
    if (use_mma) {
        qsa_mma_bf16_kernel<HeadsPerKv>
            <<<grid, kMmaThreads, 0, stream>>>(
                query,
                key,
                value,
                selected,
                output,
                batch,
                query_len,
                query_heads,
                kv_heads,
                kv_len,
                selected_len,
                softmax_scale);
        return;
    }
    qsa_scalar_bf16_kernel<HeadsPerKv>
        <<<grid, HeadsPerKv * kWarpSize, 0, stream>>>(
            query,
            key,
            value,
            selected,
            output,
            batch,
            query_len,
            query_heads,
            kv_heads,
            kv_len,
            selected_len,
            softmax_scale);
}

// ---------------------------------------------------------------------------------------------
// Split-K decode path.
//
// The scalar and MMA kernels above launch `grid = (batch * query_len, kv_heads)`, which for a
// batch-1 decode step with one KV head a rank is a *single* CTA, and it walks the whole selection
// in tiles of `kMmaKvTile` with a barrier between each. One SM of the card's 128 does the work, so
// the launch measures ~294 us for about 6.3 MFLOP -- roughly 150 times its bandwidth floor.
//
// This path splits the selection across `kSplitK` CTAs, each keeping its own running max, running
// sum and partial output (flash-decoding's first pass), and combines them in a second short kernel.
// Sixteen splits put 32 CTAs on the work and shorten the serial chain from 64 tiles to 4.
//
// It is decode-only by construction: one query row. A prefill call has one selection per query row
// and hundreds of rows, and dispatching that here would compute a fraction of the work and answer
// zeros -- which is exactly what an early prototype of this measured before the shapes were
// printed. `qwen4_exp_qsa_splitk_bf16_forward` therefore checks `query.size(1) == 1` and the
// existing op stays in place for everything else.
// ---------------------------------------------------------------------------------------------

constexpr int kSplitK = 16;
constexpr int kSplitThreads = 256;
constexpr int kSplitWarps = kSplitThreads / kWarpSize;
constexpr int kSplitChunk = 256;   // a 2048 position budget over 16 splits is 128, so this is slack

__device__ __forceinline__ float qsa_split_load(const c10::BFloat16* p) {
    return static_cast<float>(*p);
}

// One CTA per (split, kv_head): its slice of the selection reduced to a running max, a running sum
// and a partial output.
template <int HeadsPerKv>
__global__ __launch_bounds__(kSplitThreads) void qsa_splitk_partial_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    const int split = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;

    const int chunk = (selected_len + kSplitK - 1) / kSplitK;
    const int begin = split * chunk;
    const int count = min(chunk, selected_len - begin);

    __shared__ float q_sh[HeadsPerKv][kDim];
    __shared__ float scores[HeadsPerKv][kSplitChunk];
    __shared__ int32_t ids[kSplitChunk];

    const int first_head = kv_head * HeadsPerKv;
    const c10::BFloat16* q_base = query + static_cast<int64_t>(first_head) * kDim;
    for (int i = tid; i < HeadsPerKv * kDim; i += kSplitThreads) {
        q_sh[i / kDim][i % kDim] = qsa_split_load(q_base + i);
    }
    for (int i = tid; i < count; i += kSplitThreads) {
        ids[i] = selected[begin + i];
    }
    __syncthreads();

    const int64_t kv_head_base = static_cast<int64_t>(kv_head) * kv_len * kDim;

    // One warp a position, lane-parallel over the head dimension. Every warp walks the same number
    // of iterations so the barrier after this loop cannot be reached by some warps and not others.
    for (int base = 0; base < count; base += kSplitWarps) {
        const int slot = base + warp;
        const int token = slot < count ? ids[slot] : -1;
        float acc[HeadsPerKv];
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            acc[h] = 0.0f;
        }
        if (token >= 0 && token < kv_len) {
            const c10::BFloat16* kp =
                key + kv_head_base + static_cast<int64_t>(token) * kDim;
            for (int d = lane; d < kDim; d += kWarpSize) {
                const float kv = qsa_split_load(kp + d);
#pragma unroll
                for (int h = 0; h < HeadsPerKv; ++h) {
                    acc[h] += q_sh[h][d] * kv;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            float v = acc[h];
            for (int off = kWarpSize / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                const bool valid = token >= 0 && token < kv_len;
                scores[h][slot] = (valid && slot < count) ? v * softmax_scale : -INFINITY;
            }
        }
    }
    __syncthreads();

    // Three phases, not one. A thread that stored `exp(score - max)` while another was still
    // scanning for the maximum would hand it a value of at most 1, and the first version of this
    // kernel measured cosine 0.66-0.80 against a torch reference for exactly that reason.
    float m[HeadsPerKv];
    float l[HeadsPerKv];
    float out[HeadsPerKv];
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        m[h] = -INFINITY;
        for (int slot = 0; slot < count; ++slot) {
            m[h] = fmaxf(m[h], scores[h][slot]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        for (int slot = tid; slot < count; slot += kSplitThreads) {
            scores[h][slot] = scores[h][slot] == -INFINITY
                ? 0.0f
                : __expf(scores[h][slot] - m[h]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        l[h] = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            l[h] += scores[h][slot];
        }
        out[h] = 0.0f;
    }
    for (int slot = 0; slot < count; ++slot) {
        const int token = ids[slot];
        if (token < 0 || token >= kv_len) {
            continue;
        }
        const c10::BFloat16* vp =
            value + kv_head_base + static_cast<int64_t>(token) * kDim;
        const float vd = qsa_split_load(vp + tid);
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            out[h] += scores[h][slot] * vd;
        }
    }

    const int64_t split_index = static_cast<int64_t>(kv_head) * kSplitK + split;
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        part_acc[split_index * HeadsPerKv * kDim
                 + static_cast<int64_t>(h) * kDim + tid] = out[h];
    }
    if (tid < HeadsPerKv) {
        part_m[split_index * HeadsPerKv + tid] = m[tid];
        part_l[split_index * HeadsPerKv + tid] = l[tid];
    }
}

// Second pass: combine the splits with the max and sum each of them carries.
template <int HeadsPerKv>
__global__ void qsa_splitk_combine_kernel(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    c10::BFloat16* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    float m = -INFINITY;
    for (int split = 0; split < kSplitK; ++split) {
        m = fmaxf(m, part_m[(kv_head * kSplitK + split) * HeadsPerKv + head]);
    }
    float acc = 0.0f;
    float denom = 0.0f;
    for (int split = 0; split < kSplitK; ++split) {
        const int64_t idx = static_cast<int64_t>(kv_head) * kSplitK + split;
        const float p = part_l[idx * HeadsPerKv + head];
        const float weight =
            p == 0.0f ? 0.0f : __expf(part_m[idx * HeadsPerKv + head] - m);
        denom += p * weight;
        acc += part_acc[idx * HeadsPerKv * kDim + head * kDim + tid] * weight;
    }
    acc = denom == 0.0f ? 0.0f : acc / denom;
    output[static_cast<int64_t>(kv_head * HeadsPerKv + head) * kDim + tid] =
        c10::BFloat16(acc);
}

template <int HeadsPerKv>
void launch_qsa_splitk(
    const c10::BFloat16* query,
    const c10::BFloat16* key,
    const c10::BFloat16* value,
    const int32_t* selected,
    c10::BFloat16* output,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale,
    cudaStream_t stream) {
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // The partials are scratch for one launch; allocating them here keeps the call site and the
    // recorded graph unchanged, because the shapes are a constant of the geometry.
    auto part_acc = torch::empty({kv_heads, kSplitK, HeadsPerKv, kDim}, fopts);
    auto part_m = torch::empty({kv_heads, kSplitK, HeadsPerKv}, fopts);
    auto part_l = torch::empty({kv_heads, kSplitK, HeadsPerKv}, fopts);

    const dim3 grid(kSplitK, kv_heads);
    qsa_splitk_partial_kernel<HeadsPerKv><<<grid, kSplitThreads, 0, stream>>>(
        query, key, value, selected,
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        kv_len, selected_len, softmax_scale);
    const dim3 combine_grid(kv_heads, HeadsPerKv);
    qsa_splitk_combine_kernel<HeadsPerKv><<<combine_grid, kDim, 0, stream>>>(
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        output);
}

// Small-query split-K path. Each z plane is an independent copy of the single-query work above.
// Only integer addressing differs: the dot, softmax and split-combine loops deliberately retain
// the existing single-query operation order. Keep the old entry separate so its arithmetic remains
// an independent oracle for eager and captured bit-equality checks of this new path.

template <int HeadsPerKv>
__global__ __launch_bounds__(kSplitThreads) void qsa_splitk_rows_partial_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    const int query_row = static_cast<int>(blockIdx.z);
    const int split = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;

    const int chunk = (selected_len + kSplitK - 1) / kSplitK;
    const int begin = split * chunk;
    const int count = min(chunk, selected_len - begin);

    __shared__ float q_sh[HeadsPerKv][kDim];
    __shared__ float scores[HeadsPerKv][kSplitChunk];
    __shared__ int32_t ids[kSplitChunk];

    const int first_head = kv_head * HeadsPerKv;
    const int64_t query_head = static_cast<int64_t>(query_row) * kv_heads * HeadsPerKv + first_head;
    const c10::BFloat16* q_base = query + query_head * kDim;
    for (int i = tid; i < HeadsPerKv * kDim; i += kSplitThreads) {
        q_sh[i / kDim][i % kDim] = qsa_split_load(q_base + i);
    }
    for (int i = tid; i < count; i += kSplitThreads) {
        ids[i] = selected[static_cast<int64_t>(query_row) * selected_len + begin + i];
    }
    __syncthreads();

    const int64_t kv_head_base = static_cast<int64_t>(kv_head) * kv_len * kDim;

    // One warp a position, lane-parallel over the head dimension. Every warp walks the same number
    // of iterations so the barrier after this loop cannot be reached by some warps and not others.
    for (int base = 0; base < count; base += kSplitWarps) {
        const int slot = base + warp;
        const int token = slot < count ? ids[slot] : -1;
        float acc[HeadsPerKv];
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            acc[h] = 0.0f;
        }
        if (token >= 0 && token < kv_len) {
            const c10::BFloat16* kp =
                key + kv_head_base + static_cast<int64_t>(token) * kDim;
            for (int d = lane; d < kDim; d += kWarpSize) {
                const float kv = qsa_split_load(kp + d);
#pragma unroll
                for (int h = 0; h < HeadsPerKv; ++h) {
                    acc[h] += q_sh[h][d] * kv;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            float v = acc[h];
            for (int off = kWarpSize / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                const bool valid = token >= 0 && token < kv_len;
                scores[h][slot] = (valid && slot < count) ? v * softmax_scale : -INFINITY;
            }
        }
    }
    __syncthreads();

    // Three phases, not one. A thread that stored `exp(score - max)` while another was still
    // scanning for the maximum would hand it a value of at most 1, and the first version of this
    // kernel measured cosine 0.66-0.80 against a torch reference for exactly that reason.
    float m[HeadsPerKv];
    float l[HeadsPerKv];
    float out[HeadsPerKv];
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        m[h] = -INFINITY;
        for (int slot = 0; slot < count; ++slot) {
            m[h] = fmaxf(m[h], scores[h][slot]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        for (int slot = tid; slot < count; slot += kSplitThreads) {
            scores[h][slot] = scores[h][slot] == -INFINITY
                ? 0.0f
                : __expf(scores[h][slot] - m[h]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        l[h] = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            l[h] += scores[h][slot];
        }
        out[h] = 0.0f;
    }
    for (int slot = 0; slot < count; ++slot) {
        const int token = ids[slot];
        if (token < 0 || token >= kv_len) {
            continue;
        }
        const c10::BFloat16* vp =
            value + kv_head_base + static_cast<int64_t>(token) * kDim;
        const float vd = qsa_split_load(vp + tid);
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            out[h] += scores[h][slot] * vd;
        }
    }

    const int64_t split_index = (static_cast<int64_t>(query_row) * kv_heads + kv_head) * kSplitK + split;
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        part_acc[split_index * HeadsPerKv * kDim
                 + static_cast<int64_t>(h) * kDim + tid] = out[h];
    }
    if (tid < HeadsPerKv) {
        part_m[split_index * HeadsPerKv + tid] = m[tid];
        part_l[split_index * HeadsPerKv + tid] = l[tid];
    }
}


// Combine each query's sixteen splits independently.
template <int HeadsPerKv>
__global__ void qsa_splitk_rows_combine_kernel(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    c10::BFloat16* __restrict__ output,
    int kv_heads) {
    const int query_row = static_cast<int>(blockIdx.z);
    const int kv_head = static_cast<int>(blockIdx.x);
    const int64_t row_kv_head = static_cast<int64_t>(query_row) * kv_heads + kv_head;
    const int head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    float m = -INFINITY;
    for (int split = 0; split < kSplitK; ++split) {
        m = fmaxf(m, part_m[(row_kv_head * kSplitK + split) * HeadsPerKv + head]);
    }
    float acc = 0.0f;
    float denom = 0.0f;
    for (int split = 0; split < kSplitK; ++split) {
        const int64_t idx = row_kv_head * kSplitK + split;
        const float p = part_l[idx * HeadsPerKv + head];
        const float weight =
            p == 0.0f ? 0.0f : __expf(part_m[idx * HeadsPerKv + head] - m);
        denom += p * weight;
        acc += part_acc[idx * HeadsPerKv * kDim + head * kDim + tid] * weight;
    }
    acc = denom == 0.0f ? 0.0f : acc / denom;
    output[(row_kv_head * HeadsPerKv + head) * kDim + tid] =
        c10::BFloat16(acc);
}


template <int HeadsPerKv>
void launch_qsa_splitk_rows(
    const c10::BFloat16* query,
    const c10::BFloat16* key,
    const c10::BFloat16* value,
    const int32_t* selected,
    c10::BFloat16* output,
    int query_rows,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale,
    cudaStream_t stream) {
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // The partials are scratch for one launch; allocating them here keeps the call site and the
    // recorded graph unchanged, because the shapes are a constant of the geometry.
    auto part_acc = torch::empty({query_rows, kv_heads, kSplitK, HeadsPerKv, kDim}, fopts);
    auto part_m = torch::empty({query_rows, kv_heads, kSplitK, HeadsPerKv}, fopts);
    auto part_l = torch::empty({query_rows, kv_heads, kSplitK, HeadsPerKv}, fopts);

    const dim3 grid(kSplitK, kv_heads, query_rows);
    qsa_splitk_rows_partial_kernel<HeadsPerKv><<<grid, kSplitThreads, 0, stream>>>(
        query, key, value, selected,
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        kv_heads, kv_len, selected_len, softmax_scale);
    const dim3 combine_grid(kv_heads, HeadsPerKv, query_rows);
    qsa_splitk_rows_combine_kernel<HeadsPerKv><<<combine_grid, kDim, 0, stream>>>(
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        output, kv_heads);
}

}  // namespace

torch::Tensor qwen4_exp_qsa_bf16_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& selected,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(query.device());
    const int batch = static_cast<int>(query.size(0));
    const int query_len = static_cast<int>(query.size(1));
    const int query_heads = static_cast<int>(query.size(2));
    const int kv_heads = static_cast<int>(key.size(1));
    const int kv_len = static_cast<int>(key.size(2));
    const int selected_len = static_cast<int>(selected.size(2));
    const int heads_per_kv = query_heads / kv_heads;
    const char* mma_env = std::getenv("POCKETLLM_QWEN4_QSA_MMA");
    const bool use_mma = selected_len >= kMmaKvTile && mma_env != nullptr &&
        std::strcmp(mma_env, "0") != 0;
    auto output = torch::empty(query.sizes(), query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define QWEN4_EXP_LAUNCH_QSA(HEADS)                                       \
    launch_qsa<HEADS>(                                                     \
        query.data_ptr<c10::BFloat16>(),                                   \
        key.data_ptr<c10::BFloat16>(),                                     \
        value.data_ptr<c10::BFloat16>(),                                   \
        selected.data_ptr<int32_t>(),                                      \
        output.data_ptr<c10::BFloat16>(),                                  \
        batch,                                                             \
        query_len,                                                         \
        query_heads,                                                       \
        kv_heads,                                                          \
        kv_len,                                                            \
        selected_len,                                                      \
        static_cast<float>(softmax_scale),                                 \
        use_mma,                                                           \
        stream)
    switch (heads_per_kv) {
        case 1: QWEN4_EXP_LAUNCH_QSA(1); break;
        case 2: QWEN4_EXP_LAUNCH_QSA(2); break;
        case 3: QWEN4_EXP_LAUNCH_QSA(3); break;
        case 4: QWEN4_EXP_LAUNCH_QSA(4); break;
        case 5: QWEN4_EXP_LAUNCH_QSA(5); break;
        case 6: QWEN4_EXP_LAUNCH_QSA(6); break;
        case 7: QWEN4_EXP_LAUNCH_QSA(7); break;
        case 8: QWEN4_EXP_LAUNCH_QSA(8); break;
        case 9: QWEN4_EXP_LAUNCH_QSA(9); break;
        case 10: QWEN4_EXP_LAUNCH_QSA(10); break;
        case 11: QWEN4_EXP_LAUNCH_QSA(11); break;
        case 12: QWEN4_EXP_LAUNCH_QSA(12); break;
    }
#undef QWEN4_EXP_LAUNCH_QSA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor qwen4_exp_qsa_splitk_bf16_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& selected,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(query.device());
    const int kv_heads = static_cast<int>(key.size(1));
    const int kv_len = static_cast<int>(key.size(2));
    const int selected_len = static_cast<int>(selected.size(2));
    const int heads_per_kv = static_cast<int>(query.size(2)) / kv_heads;
    auto output = torch::empty(query.sizes(), query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define QWEN4_EXP_LAUNCH_QSA_SPLITK(HEADS)                                        \
    launch_qsa_splitk<HEADS>(                                                     \
        query.data_ptr<c10::BFloat16>(),                                          \
        key.data_ptr<c10::BFloat16>(),                                            \
        value.data_ptr<c10::BFloat16>(),                                          \
        selected.data_ptr<int32_t>(),                                             \
        output.data_ptr<c10::BFloat16>(),                                         \
        kv_heads,                                                                 \
        kv_len,                                                                   \
        selected_len,                                                             \
        static_cast<float>(softmax_scale),                                        \
        stream)
    switch (heads_per_kv) {
        case 1: QWEN4_EXP_LAUNCH_QSA_SPLITK(1); break;
        case 2: QWEN4_EXP_LAUNCH_QSA_SPLITK(2); break;
        case 3: QWEN4_EXP_LAUNCH_QSA_SPLITK(3); break;
        case 4: QWEN4_EXP_LAUNCH_QSA_SPLITK(4); break;
        case 5: QWEN4_EXP_LAUNCH_QSA_SPLITK(5); break;
        case 6: QWEN4_EXP_LAUNCH_QSA_SPLITK(6); break;
        case 7: QWEN4_EXP_LAUNCH_QSA_SPLITK(7); break;
        case 8: QWEN4_EXP_LAUNCH_QSA_SPLITK(8); break;
        case 9: QWEN4_EXP_LAUNCH_QSA_SPLITK(9); break;
        case 10: QWEN4_EXP_LAUNCH_QSA_SPLITK(10); break;
        case 11: QWEN4_EXP_LAUNCH_QSA_SPLITK(11); break;
        case 12: QWEN4_EXP_LAUNCH_QSA_SPLITK(12); break;
    }
#undef QWEN4_EXP_LAUNCH_QSA_SPLITK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor qwen4_exp_qsa_splitk_rows_bf16_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& selected,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(query.device());
    const int query_rows = static_cast<int>(query.size(1));
    const int kv_heads = static_cast<int>(key.size(1));
    const int kv_len = static_cast<int>(key.size(2));
    const int selected_len = static_cast<int>(selected.size(2));
    const int heads_per_kv = static_cast<int>(query.size(2)) / kv_heads;
    auto output = torch::empty(query.sizes(), query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(HEADS)                                        \
    launch_qsa_splitk_rows<HEADS>(                                                     \
        query.data_ptr<c10::BFloat16>(),                                          \
        key.data_ptr<c10::BFloat16>(),                                            \
        value.data_ptr<c10::BFloat16>(),                                          \
        selected.data_ptr<int32_t>(),                                             \
        output.data_ptr<c10::BFloat16>(),                                         \
        query_rows,                                                               \
        kv_heads,                                                                 \
        kv_len,                                                                   \
        selected_len,                                                             \
        static_cast<float>(softmax_scale),                                        \
        stream)
    switch (heads_per_kv) {
        case 1: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(1); break;
        case 2: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(2); break;
        case 3: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(3); break;
        case 4: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(4); break;
        case 5: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(5); break;
        case 6: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(6); break;
        case 7: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(7); break;
        case 8: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(8); break;
        case 9: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(9); break;
        case 10: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(10); break;
        case 11: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(11); break;
        case 12: QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS(12); break;
    }
#undef QWEN4_EXP_LAUNCH_QSA_SPLITK_ROWS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
