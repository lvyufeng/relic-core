# Original-format block-32 GEMM on Ascend

This operator path consumes original floating encodings rather than replacing them with integer
activation quantization. It is a software implementation of the port's explicit reduction rule,
not native FP4/FP8 arithmetic and not a claim that an unexecuted TileLang kernel matches it.

## Operators

```python
from relic_core.kernels.block_fp8_quant import quantize_block_fp8
from relic_core.kernels.block_gemm import fp4_int8_block_gemm, fp8_block_gemm
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale

# x is contiguous BF16 or FP32, with a last dimension divisible by 32.
a_codes, a_scale_bytes = quantize_block_fp8(x, backend="triton")

# A packed E2M1 weight and its ORIGINAL E8M0 row grid are already on the NPU.
w_x2 = unpack_fp4_to_int8(packed, backend="triton")
original_half_scale = fp4_int8_scale(scale_bytes, backend="triton")
y_fp32 = fp4_int8_block_gemm(
    a_codes, a_scale_bytes, w_x2, original_half_scale, backend="triton"
)

# A dense E4M3 weight keeps its original square or row-wise grid.
y_fp32 = fp8_block_gemm(
    a_codes, a_scale_bytes, weight_codes, original_scale,
    block_rows=32, backend="triton",
)
```

The GEMM inputs are matrices `[M,K]` and `[N,K]`, with positive dimensions and `K % 32 == 0`.
Activation codes/scales are raw E4M3/E8M0 bytes; scales have shape `[M,K/32]`. FP4 values are exact
INT8 **times two**, not integer-requantized weights, and their FP32 half-scale grid is `[N,K/32]`.
Dense FP8 scales are raw E8M0 bytes or exact decoded FP32 powers, with `[ceil(N/block_rows),K/32]`
and `block_rows=1` or `32`. A malformed FP32 power grid yields NaNs rather than silently becoming a
different quantizer. `out=` is an optional disjoint contiguous FP32 `[M,N]` tensor.

`backend="torch"` is a bounded CPU oracle. `backend="triton"` is NPU-only. `auto` picks the device's
legal path, and a compiler/kernel error is never caught and replaced by another arithmetic path.
There is no model import or CUDA-extension dependency in these operators. For a Python-only install,
use the [explicit installation mode](python_only_install.md).

## Activation and reduction contracts

1. Each 32-element activation block computes an FP32 amax with floor `1e-4`.
2. The original scale is the bit-field ceiling of **FP32 `amax * fl32(1/448)`**. It is not a divide by
   448 followed by an approximate logarithm. Normalize by that exact power of two, clamp to ±448,
   then encode E4M3 with round-to-nearest-even. Raw UINT8 outputs avoid an unsupported NPU float8
   cast. A block containing non-finite values explicitly produces NaN codes and scale byte 255;
   that is a documented extension of the finite contract, not an assertion about TileLang NaNs.
3. The GEMM computes code products in two successive 16-term halves. Nominal exponent anchors,
   truncation toward zero on the window-13 grid, the first-half sum joining the second-half window,
   and final renormalization follow the port's independently tested tensor-core behavior model.
4. Each finished fragment is scaled by the **original activation scale, then original weight scale**
   with separate FP32 rounding. K blocks accumulate in their original order. Scale exponents are
   not combined ahead of the fragment, since that can change overflow/underflow.

The FP4 GEMM recovers the original E2M1 value as `INT8 / 2` before the reduction and reconstructs
the **original** scale from the half-scale's FP32 bits. It does not also apply the half-scale: that
would be a factor-of-two error. Byte-zero and byte-one half-scales (`2^-128`, `2^-127`) have explicit
subnormal bit cases. INT8 loses FP4 negative-zero sign only; original packed bytes remain unchanged.

The decoded product never becomes an entire `[M,N,K]` allocation. CPU slabs are at most
`16*64*32` elements, and each NPU program keeps a `16*32` product slab. The NPU path uses signed
32-bit bit fields: unsigned logical shifts in the tested CANN/Triton compiler caused an unsupported
UINT32→UINT64 vector cast, while equivalent signed fields plus masks compiled and passed parity.
No shared compiler/header/driver modification was used to resolve that kernel issue.

## Acceptance and boundaries

The continuation was tested on Ascend 910B3, CANN 9.2.0, **torch 2.10.0 / torch_npu 2.10.0.post7**,
with the retained private CPython 3.11 / Triton Ascend 3.2 environment documented in the
[online-unpack guide](online_float_unpack.md). Its existing private host-driver compatibility patch
is still required on this host; stock-wheel acceptance is not implied.

Exhaustive helper tests cover all E4M3/E8M0 scale pairs, arbitrary FP32 scale rounding, signed zero,
subnormal ties and invalid scale grids. Real NPU tests compare finite FP32 bits with an independent
scalar oracle, including all E4M3 versus E2M1 codes, scale endpoints, sequential cancellation,
square/row scale layouts and output-memory guards. Runtime tests additionally cover the original
SwiGLU/routing/narrowing order and expert-parallel composition.

An original checkpoint-backed expert forward was compared with the CPU port, including all three
matrices and routing-weight application before the down projection. This is an **expert-level**
acceptance case, not a full-model token/logit comparison. Activation quantization, GEMM, final BF16
narrowing and routing order have separate checks; passing one is not used as evidence for the rest.
Graph capture is not claimed: the runtime's expert routing reads indices on the host once per call.
