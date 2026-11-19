// Exact epilogue for Qwen4-Exp's GatedDeltaNet `RMSNormGated` output path.
//
// The model's reference sequence (relicllm/models/qwen4_exp/layers.py, RMSNormGated.__call__)
// with `x` BF16 [rows, head_v_dim], BF16 norm `weight` [head_v_dim] and gate `z` is:
//
//   h = x.to(float32)
//   h = h * rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)      // fp32
//   h = weight * h.to(bfloat16)                              // round 1, then BF16*BF16
//   h = h * act(gate.to(float32))                            // BF16 promotes to fp32
//   return h.to(bfloat16)                                    // round 3
//
// There are three rounding boundaries that a fusion must keep, not two: the cast of the
// normalised value to BF16, the BF16 product with the norm weight, and the final store. An
// "obviously equivalent" rewrite that keeps everything in fp32 until the end changes the answer.
//
// This file deliberately does **not** fuse the RMS statistic. `mean(-1)` over the contiguous
// head dimension runs through ATen's Reduce.cuh, whose summation tree depends on the number of
// output rows even at a fixed reduction width; a warp-tree sum is not bit-identical to it for
// every row count. The caller therefore passes `inv_rms`, computed by the exact torch expression
//   torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
// and only the elementwise epilogue is fused here. That boundary is provable from the ISA, which
// is what this operator needs before it can replace working arithmetic.
//
// `--use_fast_math` is on for this project's nvcc flags, and it implies `-ftz=true` (flush
// denormals) and `-fmad=true` (contract multiply+add into FMA). Neither is what torch does. Both
// are defeated here structurally: every float32 multiply is an explicit `mul.rn.f32` PTX
// instruction, and every BF16 rounding is integer bit manipulation, so no compiler flag can
// retarget them. `-ftz` would still apply to a hypothetical implicit operation; there is none in
// the epilogue path.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
//: A head width this small is what the model uses; the loop below still handles wider ones, but a
//: wider shared-memory weight cache is not worth specialising for a shape nothing asks for.
constexpr int kMaxHidden = 8192;

//: Round a float32 bit pattern to BF16 with round-to-nearest-even, in integer arithmetic.
//:
//: Software rather than `__float2bfloat16_rn` for two reasons: the intrinsic's sm_75 form is a
//: different code path from its sm_80+ `cvt` form, and the hardware `cvt` can carry the compiler's
//: `.ftz` modifier. Integer rounding is identical on every arch and immune to every flag.
__device__ __forceinline__ uint16_t f32_bits_to_bf16_rn(uint32_t bits) {
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu && mantissa != 0u) {
        // Match ATen's architecture-specific CUDA BF16 conversion. sm_80+ cvt canonicalizes to
        // 0x7fff; the sm_75 c10 software conversion uses 0x7fc0. CPU vectorized conversions are
        // not the oracle here (their NaN payload can differ from both CUDA paths).
#if __CUDA_ARCH__ >= 800
        return static_cast<uint16_t>(0x7fffu);
#else
        return static_cast<uint16_t>(0x7fc0u);
#endif
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float value) {
    return f32_bits_to_bf16_rn(__float_as_uint(value));
}

//: BF16 -> float32 is exact: it is a zero-fill of the low sixteen mantissa bits.
__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

//: IEEE-754 round-to-nearest-even float32 multiply, stated in PTX.
//:
//: `-ftz=true` from `--use_fast_math` otherwise compiles `a * b` as `mul.ftz.f32`, which flushes
//: a subnormal operand or result to zero; `-fmad=true` is irrelevant here because there is no
//: adjacent add to contract, but writing the multiply explicitly removes the question.
__device__ __forceinline__ float fmul_rn(float lhs, float rhs) {
    float product;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(lhs), "f"(rhs));
    return product;
}

//: The reference activation computed in double and rounded once to float32.
//:
//: This is *not* proven bit-identical to torch's float32 `sigmoid`/`silu`, which call CUDA's
//: `expf` (≤2 ulp) and a float division; double `exp` is a different polynomial. It is offered
//: only through the separately named `..._act` entry, which an on-device oracle must accept
//: before any runtime uses it. See the header comment of that entry.
__device__ __forceinline__ float sigmoid_f32_wide(float gate) {
    return static_cast<float>(1.0 / (1.0 + ::exp(-static_cast<double>(gate))));
}

__device__ __forceinline__ float silu_f32_wide(float gate) {
    const double wide = static_cast<double>(gate);
    return static_cast<float>(wide / (1.0 + ::exp(-wide)));
}

//: `activation` selects what `gate` means: -1 already-activated float32, 0 sigmoid, 1 silu.
constexpr int kActivationPrecomputed = -1;
constexpr int kActivationSigmoid = 0;
constexpr int kActivationSilu = 1;

template <bool kFusedActivation>
__global__ void qwen4_exp_gated_rms_norm_exact_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ inv_rms,
    const uint16_t* __restrict__ weight,
    const void* __restrict__ gate,
    uint16_t* __restrict__ output,
    int64_t rows,
    int hidden,
    int64_t row_stride,
    int activation,
    bool gate_is_float) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float shared_weight[];
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        shared_weight[col] = bf16_to_f32(weight[col]);
    }
    __syncthreads();

    const uint16_t* x_row = x + row * row_stride;
    uint16_t* out_row = output + row * row_stride;
    const int64_t gate_base = row * row_stride;
    const float inverse = inv_rms[row];

    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        // Rounding boundary 1: the normalised value is cast to BF16 before the weight multiply,
        // because the reference stores it through a `.to(bfloat16)`.
        const float normalised = fmul_rn(bf16_to_f32(x_row[col]), inverse);
        const uint16_t normalised_bf16 = f32_to_bf16_rn(normalised);
        // Rounding boundary 2: BF16 weight times BF16 activation. ATen's elementwise binary ops
        // compute in float `opmath_type` and store BF16, so this is one float product rounded once.
        const float scaled = fmul_rn(bf16_to_f32(normalised_bf16), shared_weight[col]);
        const uint16_t scaled_bf16 = f32_to_bf16_rn(scaled);

        float activated;
        if (kFusedActivation) {
            // `gate.to(float32)`: exact when the gate is BF16, an identity when it is already float.
            float gate_value;
            if (gate_is_float) {
                gate_value = static_cast<const float*>(gate)[gate_base + col];
            } else {
                gate_value =
                    bf16_to_f32(static_cast<const uint16_t*>(gate)[gate_base + col]);
            }
            activated = activation == kActivationSigmoid ? sigmoid_f32_wide(gate_value)
                                                         : silu_f32_wide(gate_value);
        } else {
            activated = static_cast<const float*>(gate)[gate_base + col];
        }
        // Rounding boundary 3: BF16 value times float32 activation in float32, stored BF16.
        out_row[col] = f32_to_bf16_rn(fmul_rn(bf16_to_f32(scaled_bf16), activated));
    }
}

//: A tensor the kernel can walk as `rows * hidden`. A transposed or otherwise irregular view is
//: refused rather than copied: staging a hidden copy would quietly undo the fusion's whole point,
//: and the caller has a working torch path to fall back to.
void require_row_major(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() >= 2, name, " must have at least two dimensions ([rows, hidden])");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous ([rows, hidden] row-major)");
}

torch::Tensor run_epilogue(
    const torch::Tensor& x,
    const torch::Tensor& inv_rms,
    const torch::Tensor& weight,
    const torch::Tensor& gate,
    int64_t activation,
    bool fused_activation) {
    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions ([rows, hidden])");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.is_cuda() && weight.device() == x.device(),
                "weight must be a CUDA tensor on the same device as x");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous(), "weight must be one contiguous row");
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.device() == x.device(),
                "inv_rms must be a CUDA tensor on the same device as x");
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32, "inv_rms must be float32");
    TORCH_CHECK(inv_rms.is_contiguous(), "inv_rms must be contiguous");

    const int64_t hidden = x.size(-1);
    TORCH_CHECK(hidden > 0 && hidden <= kMaxHidden, "unsupported hidden width ", hidden);
    TORCH_CHECK(weight.size(0) == hidden, "weight width must equal x's last dimension");
    require_row_major(x, "x");
    const int64_t rows = x.numel() / hidden;
    TORCH_CHECK(inv_rms.numel() == rows, "inv_rms must hold one value per row");

    TORCH_CHECK(gate.sizes() == x.sizes(), "gate shape must match x shape");
    TORCH_CHECK(gate.device() == x.device(), "gate must be on the same CUDA device as x");
    require_row_major(gate, "gate");
    bool gate_is_float = false;
    if (fused_activation) {
        TORCH_CHECK(activation == kActivationSigmoid || activation == kActivationSilu,
                    "the fused-activation entry takes sigmoid (0) or silu (1)");
        gate_is_float = gate.scalar_type() == torch::kFloat32;
        TORCH_CHECK(gate_is_float || gate.scalar_type() == torch::kBFloat16,
                    "gate must be bfloat16 or float32");
    } else {
        TORCH_CHECK(activation == kActivationPrecomputed,
                    "the exact entry takes an already-activated float32 gate (activation = -1)");
        TORCH_CHECK(gate.scalar_type() == torch::kFloat32,
                    "the exact entry needs the activation precomputed in float32");
    }

    // A fresh contiguous destination: `x` and `gate` are validated contiguous above, so every
    // tensor the kernel walks has the same row stride.
    auto output = torch::empty(x.sizes(), x.options());
    if (rows == 0) {
        return output;
    }
    TORCH_CHECK(rows <= 0x7fffffffLL, "too many rows");

    const size_t shared_bytes = static_cast<size_t>(hidden) * sizeof(float);
    const dim3 grid(static_cast<unsigned>(rows));
    const dim3 block(kThreads);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x.data_ptr<c10::BFloat16>());
    const uint16_t* weight_ptr =
        reinterpret_cast<const uint16_t*>(weight.data_ptr<c10::BFloat16>());
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>());
    const void* gate_ptr = gate.data_ptr();

    const int64_t row_stride = hidden;
    if (fused_activation) {
        qwen4_exp_gated_rms_norm_exact_kernel<true><<<grid, block, shared_bytes, stream>>>(
            x_ptr, inv_rms.data_ptr<float>(), weight_ptr, gate_ptr, out_ptr, rows,
            static_cast<int>(hidden), row_stride,
            static_cast<int>(activation), gate_is_float);
    } else {
        qwen4_exp_gated_rms_norm_exact_kernel<false><<<grid, block, shared_bytes, stream>>>(
            x_ptr, inv_rms.data_ptr<float>(), weight_ptr, gate_ptr, out_ptr, rows,
            static_cast<int>(hidden), row_stride,
            static_cast<int>(activation), false);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

//: Exact epilogue with the activation supplied already evaluated in float32 by the caller.
//:
//: This is the entry a runtime should bind: the caller keeps `torch.sigmoid(gate.float())` (or
//: `F.silu`) outside the kernel, so the activation's bits stay exactly what torch produces, and
//: only the three proven casting/multiply boundaries are fused. `inv_rms` is
//: `torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)` computed by torch for the same
//: reason on the reduction side.
torch::Tensor qwen4_exp_gated_rms_norm_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& inv_rms,
    const torch::Tensor& weight,
    const torch::Tensor& gate_activated) {
    return run_epilogue(x, inv_rms, weight, gate_activated, kActivationPrecomputed, false);
}

//: The same epilogue with the activation computed inside the kernel.
//:
//: **Not proven exact and not for production use.** It evaluates the activation through double
//: `exp`, which is not torch's float32 `expf`; a one-ulp float difference can flip a BF16 result
//: near a rounding boundary. It exists so an on-device oracle can measure that difference over
//: real and adversarial data before anyone decides whether the fused form is acceptable.
torch::Tensor qwen4_exp_gated_rms_norm_exact_act_cuda(
    const torch::Tensor& x,
    const torch::Tensor& inv_rms,
    const torch::Tensor& weight,
    const torch::Tensor& gate,
    int64_t activation) {
    return run_epilogue(x, inv_rms, weight, gate, activation, true);
}