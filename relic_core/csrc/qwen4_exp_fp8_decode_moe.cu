// Decode experts: the SwiGLU activation of an FP8 expert at M == 1, from 16-byte loads.
//
// The slot kernels beside this file (`qwen4_exp_fp8_expert.cu`) read their tables one e4m3 code,
// or one pair, a lane per step. Measured on an RTX 3090 that read runs at 48-67 GB/s where a plain
// bf16 GEMV of the same shape reaches 298-735, and at ~4 ms a phase it is what kept the fused route
// at 4.0 ms against the 1.0 ms of the loop it was meant to replace. The cause is the instruction
// count, not the bytes: a pair a lane is a decode, a scale multiply and two FMAs per two bytes.
//
// This kernel reads a 16-byte chunk a lane -- a warp's 32 lanes hold 512 consecutive bytes of one
// row -- and decodes a pair with one instruction where the target has one:
// `__nv_cvt_fp8x2_to_halfraw2` on Ada, a field shift and rebias elsewhere (Ampere and Turing have
// F2FP too, but not for this pair of types, and the shift is four instructions). Both widenings are
// exact (e4m3's 4 exponent bits and 3 mantissa bits are a subset of fp16's), so the fp16 pair the
// accumulate sees is `code * scale` rounded once.
//
//   x       [hidden] bf16              the decode token's activation
//   gate_up [E, 2*inter, hidden] e4m3  the checkpoint's gate rows then its up rows
//   act     [slots, inter] bf16        the SwiGLU activation, one row a slot

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kBlockRows = 128;
constexpr int kBlockCols = 128;
constexpr int kLanes = 32;
constexpr int kThreads = 256;

// A packed pair of e4m3 codes (`q` = byte 0 | byte 1 << 8) to fp16, by hardware where it exists.
//
// Ampere has no such instruction and takes a bit shift: e4m3 -> fp16 moves the exponent field
// from bit 6 to bit 10, rebias 7 -> 15 (an add of 0x3800) and carries the 3-bit mantissa up to
// bits 12-9 (a shift of 3). It is the exact fp16 value of the code. A subnormal code has a zero
// exponent field before and after, so it survives the same way. NaN payloads are not preserved,
// which no checkpoint weight is.
__device__ __forceinline__ __half2 e4m3x2_to_half2(uint32_t q) {
#if 0
    return __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q), __NV_E4M3);
#else
    const unsigned short lo = static_cast<unsigned short>(((q & 0x7800u) >> 10) + 0x3800u + (q & 0x8000u) + ((q & 0x0780u) >> 3));
    const unsigned short hi = static_cast<unsigned short>((((q >> 16) & 0x7800u) >> 10) + 0x3800u + ((q >> 16) & 0x8000u) + (((q >> 16) & 0x0780u) >> 3));
    return __halves2half2(__ushort_as_half(lo), __ushort_as_half(hi));
#endif
}

__device__ __forceinline__ float bf16(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

// A slot's SwiGLU activation for M == 1, one block a slot, from 16-byte loads.
//
//   act[j] = silu(gate_up[j] . x) * (gate_up[inter + j] . x)
//
// A warp owns output row `j`, the 32 lanes hold codes [lane * 16, lane * 16 + 16) of it in one
// load, decode them by hardware and reduce them with the staged activation. The gate and up rows
// are two rows of the table whose 512-code window lies in the same four 128-column scale blocks,
// so the window's column scale index is lane independent; the two rows' scales differ because
// they sit in different 128-row tiles of the scale grid.
//
// A block an expert is what the per-code kernel beside it does, and the reason it is not split is
// that the gate and up halves are one reduction each: a block can fill the whole `act` row only by
// walking the whole of `hidden`.
__global__ void fp8_expert_act_fast_kernel(
    const __nv_bfloat16* __restrict__ x,         // [hidden]
    const uint8_t* __restrict__ gate_up,         // [E, 2*inter, hidden], e4m3
    const __nv_bfloat16* __restrict__ scale,     // [E, ceil(2*inter/128), ceil(hidden/128)]
    __nv_bfloat16* __restrict__ act,             // [slots, inter]
    const int* __restrict__ expert,              // [slots], -1 for none
    int hidden,
    int inter,
    int scale_cols) {
    const int slot = blockIdx.x;
    const int e = __ldg(expert + slot);
    if (e < 0) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // The activation is staged once: every warp reads the same `x`, so a shared copy is the load.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs = reinterpret_cast<__half2*>(smem_raw);  // [hidden / 2] fp16 pairs
    for (int j = threadIdx.x; j < hidden / 2; j += blockDim.x) {
        const __nv_bfloat162 b = reinterpret_cast<const __nv_bfloat162*>(x)[j];
        xs[j] = __floats2half2_rn(bf16(b.x), bf16(b.y));
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        const uint8_t* g0 = reinterpret_cast<const uint8_t*>(gate_up);
        volatile unsigned xors = 0u;
        for (int c = 0; c < 2560; ++c) xors ^= (unsigned)g0[c];
        const uint32_t raw = (uint32_t)g0[0] | ((uint32_t)g0[1] << 8);
        const __half2 dd = e4m3x2_to_half2(raw);
        printf("DBG row0 xor %08x b0 %d b1 %d dec %f %f || lo %x hi %x v %x\n", xors, (int)g0[0], (int)g0[1], __low2float(dd), __high2float(dd),
               (unsigned)((((raw) & 0x7800u) >> 10) + 0x3800u + ((raw) & 0x8000u) + ((raw & 0x0780u) >> 3)),
               (unsigned)((((raw >> 16) & 0x7800u) >> 10) + 0x3800u + ((raw >> 16) & 0x8000u) + (((raw >> 16) & 0x0780u) >> 3)),
               (unsigned)(*reinterpret_cast<const uint32_t*>(&dd)));
    }
    const auto* gu = gate_up + static_cast<int64_t>(e) * 2 * inter * hidden;
    const int64_t scale_stride =
        static_cast<int64_t>((2 * inter + kBlockRows - 1) / kBlockRows) * scale_cols;
    const auto* sc = scale + static_cast<int64_t>(e) * scale_stride;
    for (int j = warp; j < inter; j += blockDim.x / 32) {
        const int64_t grow_sc = static_cast<int64_t>(j / kBlockRows) * scale_cols;
        const int64_t urow_sc = static_cast<int64_t>((inter + j) / kBlockRows) * scale_cols;
        const uint8_t* grow = gu + static_cast<int64_t>(j) * hidden;
        const uint8_t* urow = gu + static_cast<int64_t>(inter + j) * hidden;
        float ga = 0.0f;
        float ua = 0.0f;
        for (int c0 = 0; c0 < hidden; c0 += kLanes * 16) {
            const int c = c0 + lane * 16;
            if (c < hidden) {
                const int col = c / kBlockCols;
                const float gs = bf16(__ldg(&sc[grow_sc + col]));
                const float us = bf16(__ldg(&sc[urow_sc + col]));
                const uint4 gq = *reinterpret_cast<const uint4*>(grow + c);
                const uint4 uq = *reinterpret_cast<const uint4*>(urow + c);
                const __half2 g0 = e4m3x2_to_half2(gq.x & 0xFFFFu);
                const __half2 g1 = e4m3x2_to_half2(gq.x >> 16);
                const __half2 g2 = e4m3x2_to_half2(gq.y & 0xFFFFu);
                const __half2 g3 = e4m3x2_to_half2(gq.y >> 16);
                const __half2 g4 = e4m3x2_to_half2(gq.z & 0xFFFFu);
                const __half2 g5 = e4m3x2_to_half2(gq.z >> 16);
                const __half2 g6 = e4m3x2_to_half2(gq.w & 0xFFFFu);
                const __half2 g7 = e4m3x2_to_half2(gq.w >> 16);
                const __half2 u0 = e4m3x2_to_half2(uq.x & 0xFFFFu);
                const __half2 u1 = e4m3x2_to_half2(uq.x >> 16);
                const __half2 u2 = e4m3x2_to_half2(uq.y & 0xFFFFu);
                const __half2 u3 = e4m3x2_to_half2(uq.y >> 16);
                const __half2 u4 = e4m3x2_to_half2(uq.z & 0xFFFFu);
                const __half2 u5 = e4m3x2_to_half2(uq.z >> 16);
                const __half2 u6 = e4m3x2_to_half2(uq.w & 0xFFFFu);
                const __half2 u7 = e4m3x2_to_half2(uq.w >> 16);
                const __half2* xv = xs + c / 2;
                const __half2 p0 = __hmul2(g0, xv[0]);
                const __half2 p1 = __hmul2(g1, xv[1]);
                const __half2 p2 = __hmul2(g2, xv[2]);
                const __half2 p3 = __hmul2(g3, xv[3]);
                const __half2 p4 = __hmul2(g4, xv[4]);
                const __half2 p5 = __hmul2(g5, xv[5]);
                const __half2 p6 = __hmul2(g6, xv[6]);
                const __half2 p7 = __hmul2(g7, xv[7]);
                const __half2 q0 = __hmul2(u0, xv[0]);
                const __half2 q1 = __hmul2(u1, xv[1]);
                const __half2 q2 = __hmul2(u2, xv[2]);
                const __half2 q3 = __hmul2(u3, xv[3]);
                const __half2 q4 = __hmul2(u4, xv[4]);
                const __half2 q5 = __hmul2(u5, xv[5]);
                const __half2 q6 = __hmul2(u6, xv[6]);
                const __half2 q7 = __hmul2(u7, xv[7]);
                const __half2 gt = __hadd2(__hadd2(__hadd2(p0, p1), __hadd2(p2, p3)),
                                           __hadd2(__hadd2(p4, p5), __hadd2(p6, p7)));
                const __half2 ut = __hadd2(__hadd2(__hadd2(q0, q1), __hadd2(q2, q3)),
                                           __hadd2(__hadd2(q4, q5), __hadd2(q6, q7)));
                ga = fmaf(__half2float(gt.x) + __half2float(gt.y), gs, ga);
                ua = fmaf(__half2float(ut.x) + __half2float(ut.y), us, ua);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            ga += __shfl_down_sync(0xffffffffu, ga, offset);
            ua += __shfl_down_sync(0xffffffffu, ua, offset);
        }
        if (lane == 0) {
            const float silu = ga / (1.0f + expf(-ga));
            act[static_cast<int64_t>(slot) * inter + j] = __float2bfloat16(silu * ua);

        }
    }
}

}  // namespace

torch::Tensor qwen4_exp_fp8_act_fast_cuda(const torch::Tensor& x,
                                          const torch::Tensor& expert,
                                          const torch::Tensor& gate_up,
                                          const torch::Tensor& gate_up_scale) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16 && x.is_contiguous() &&
                    x.dim() == 1,
                "x must be a contiguous CUDA bf16 [hidden]");
    TORCH_CHECK(expert.is_cuda() && expert.scalar_type() == at::ScalarType::Int && expert.is_contiguous(),
                "expert must be a contiguous CUDA int32 [slots]");
    TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                    gate_up.is_contiguous() && gate_up.dim() == 3 && gate_up.size(1) % 2 == 0,
                "gate_up must be a contiguous CUDA float8_e4m3fn [E, 2*inter, hidden]");
    TORCH_CHECK(gate_up_scale.is_cuda() && gate_up_scale.scalar_type() == at::ScalarType::BFloat16 &&
                    gate_up_scale.is_contiguous() && gate_up_scale.dim() == 3,
                "gate_up_scale must be a contiguous CUDA bf16 [E, ceil(2*inter/128), ceil(hidden/128)]");
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t slots = expert.numel();
    const int hidden = static_cast<int>(gate_up.size(2));
    const int inter = static_cast<int>(gate_up.size(1) / 2);
    TORCH_CHECK(hidden == x.numel(), "x must be as long as an expert's hidden size");
    TORCH_CHECK(hidden % kBlockCols == 0, "hidden must be a multiple of 128 (one scale a 16-code load)");
    TORCH_CHECK(static_cast<int>(gate_up_scale.size(1)) == (2 * inter + kBlockRows - 1) / kBlockRows &&
                    static_cast<int>(gate_up_scale.size(2)) == hidden / kBlockCols,
                "gate_up_scale must be [E, ceil(2*inter/128), hidden/128]");
    auto act = torch::zeros({slots, static_cast<int64_t>(inter)}, x.options());
    if (slots == 0) {
        return act;
    }
    // The kernel walks `inter` output rows by `threads / 32` warps, so a block is one expert and
    // its rows are its warps': at most 32 warps hold the 640 rows of one.
    const int threads = static_cast<int>(std::min<int64_t>(1024, (inter + 31) / 32 * 32));
    fp8_expert_act_fast_kernel<<<static_cast<unsigned>(slots), threads,
                                 static_cast<size_t>(hidden) * sizeof(__half),
                                 at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(gate_up.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up_scale.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(act.data_ptr()),
        expert.const_data_ptr<int>(),
        hidden,
        inter,
        static_cast<int>(gate_up_scale.size(2)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return act;
}