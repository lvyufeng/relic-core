# Online unpacking of original floating-weight encodings

These operators keep the released weight bytes unchanged. They decode at runtime on the tensor's
current device; they do not create an offline W8A8/W4A8 checkpoint or quantize activation tensors.
The original DeepSeek-V4.1-Flash scale layout is part of the contract, not inferred from an unrelated
ModelOpt/NVFP4 converter.

## Public operators

```python
from relic_core.kernels.unpack import (
    fp4_int8_scale,
    unpack_fp4_to_int8,
    unpack_fp8_to_bf16,
)

# Original packed E2M1 bytes: low nibble precedes high nibble along K.
q = unpack_fp4_to_int8(packed_bytes, backend="auto")
half_scale = fp4_int8_scale(original_e8m0_scales, backend="auto")
# q contains EXACTLY twice the E2M1 values. Keep the original per-32-K scale grid.
# Mathematical reconstruction: W[n,k] = q[n,k] * half_scale[n,k//32].

# Dense E4M3FN weights with one E8M0 exponent per 32x32 tile.
w = unpack_fp8_to_bf16(fp8_codes, e8m0_scales, block_shape=(32, 32))
# Engram/row-scaled matrices use one exponent per row and 32-wide K block.
rows = unpack_fp8_to_bf16(fp8_row_codes, row_scales, block_shape=(1, 32))
```

Inputs may be original typed FP8/E8M0 storage or `uint8`/`int8` **bit containers**. A dtype view,
not a numeric cast, reads those containers. Dense contiguous inputs and same-device scales are
required. Batched matrices and ragged edge tiles are supported by the generic FP8 operator;
`block_shape` explicitly selects the grid. Optional `out=` storage must match the exact shape,
dtype and device and must not share storage with an input.

### Exact FP4 integer widening

| E2M1 nibble | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 value | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 0 | -1 | -2 | -3 | -4 | -6 | -8 | -12 |

No amax, clamping, new row scale or rounding step is involved. Since E2M1 includes `0.5`, doubling
makes every value exactly integral. Compensate **once** with the original E8M0 scale divided by two.
The half-scales are FP32; the lowest original exponent requires `2^-128`, which cannot be encoded
by subtracting one in an unsigned E8M0 byte. Both FP4 zero codes become integer zero, so **negative
zero's sign is not preserved in INT8**. The original packed bytes still retain it.

### E4M3FN and E8M0 decoding

E4M3FN has 254 finite byte encodings and two NaN encodings; its signed zeros are retained. E8M0
bytes `0..254` mean `2**(byte-127)` and `255` means NaN. The scaled FP8 output is the FP32-equivalent
product narrowed **once** to BF16 with round-to-nearest-even. Finite BF16 bit patterns, including
subnormals and signed zeros, are tested independently. NaN classification is preserved; NaN payload
bits are not a parity contract.

The Torch path stages only constant lookup tables from CPU to the input device. Its largest table
is 128 KiB for the 256×256 code/scale combinations. Payload indexing and output writes stay on the
input device; no payload is transferred to CPU for decoding. Computing BF16 bits ahead of device
arithmetic avoids relying on NPU FP32 flush-to-zero behavior at extreme scales.

The optional Triton Ascend backend fuses byte reads, code/exponent decoding and output stores into
one kernel. It constructs BF16 bits using integer arithmetic rather than invoking native FP8 casts.
`backend="triton"` requires an installed compatible Ascend backend; `backend="auto"` selects it
before execution when available, otherwise uses Torch. A compiler or kernel failure is **not**
swallowed and retried through another implementation. CPU/CUDA auto dispatch remains Torch.

## Existing 2080 Ti paths are not all interchangeable

The `sm_75` CUDA code contains the same doubled E2M1 lookup idea, but the existing
`fp4_weight_to_int8_forward` is a **row-requantizing** converter: dequantize, compute row amax,
round to INT8 and replace the original K-block grid with a row scale. It is intentionally not used
by these exact unpack operators.

Inspection of clean core snapshot `aa02338fd8e695ba2be8e886e1587eff9c801f77` also found a scale-contract
drift: the current CUDA `fp4_block_scale` helper decodes ModelOpt E4M3 scale bytes, whereas historical
V4.1 documentation describes E8M0 half-scales. The grouped wrapper's K/16 scale check also differs
from its K/32 kernel loop. These are source-level observations, not new CUDA execution results.
The Ascend operators use an independent original-format oracle; this change does not rewrite CUDA
kernels or bless their scale helper as a parity reference.

## Chitu research

The inspected official repository is [thu-pacman/chitu](https://github.com/thu-pacman/chitu), pinned
to commit `6461680a462a57b73b37ea19a3b74475516ef2f0`. Relevant source files were fetched over verified
TLS and matched to that commit's Git blob IDs before inspection; repository code was not executed.
The reviewed subset is 143 files, not all 994 blobs in the tree.

- [FP4 conversion helpers](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/quant/blockfp4/convert.py)
  establish the same low-nibble-first ordering and E2M1 levels. No exact FP4→INT8 path was found in
  the inspected relevant conversion/dispatch sources; that is not a claim about every repository file.
- [NPU FP4 matmul dispatch](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/quant/blockfp4/matmul.py)
  calls external `cinfer_ascendc.grouped_gemm` / `grouped_soft_gemv` with `computeType="fp4"`, empty
  offset and multiplicative antiquantization scales. The required binary is not shipped in this
  source tree, and the source does not prove its SoC or PyTorch-2.10 ABI compatibility. It is not
  substituted for a tested 910B operator.
- [NPU native layout](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/native_layout/npu.py)
  transposes/repackages weight codes and requires K multiples of 64 and N multiples of 128. The
  generic unpack operators here do not impose that GEMM-specific layout.
- [FP8 soft conversion](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/triton_ops/quant/blockfp8/convert.py)
  and [FP4 soft floating GEMM](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/triton_ops/quant/blockfp4.py)
  demonstrate raising low-bit floating encodings to wider floating computation. Their existence is
  not proof of native 910B FP8/FP4 arithmetic or availability of a compatible Triton Ascend wheel.
- [Standard Ascend W8A8](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/quantization/ascend_w8a8.py)
  is a separate integer-quantized configuration. It is not used by this original-format unpack path.

Chitu's [repository license](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/LICENSE)
is Apache-2.0; some derived CUDA files additionally carry MIT/vLLM/GGML attribution requirements.
These operators are independently implemented from encoding arithmetic, not copied CUDA binaries
or the external `cinfer_ascendc` package.

## Measured Ascend acceptance

On 910B3 with **torch 2.10.0 / torch_npu 2.10.0.post7 / CANN 9.2.0**, the Torch/NPU operator suite
passed **68 cases with zero skips**. The final fused suite passed **75 cases with zero skips**,
including a real nondefault-stream consumer test and six guarded tail sizes. The scalar bit oracle
and mock launch-argument suite passed separately; they are not substituted for these device results.

The fused lane used a private CPython 3.11.18 environment with **triton-ascend 3.2.0**. Its stock
host utility source references an optional `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` absent from CANN
9.2. The measured environment has a **private compatibility patch**, which removes only that
mapping and explicitly reports it unsupported; original bytes and rollback are retained. No shared
CANN header/library or driver was edited. An unpatched public wheel cannot be credited with the
measured acceptance on this host.

The final FP4 kernel uses contiguous output lanes rather than two masked stride-2 byte stores;
the latter triggered Triton Ascend's interleave-optimization compiler assertion. Exact code values,
low/high ordering, padded-index widths and true-output-size tail guards were rechecked after the
change. Backend selection does not catch that compiler error and silently substitute Torch.

### Large-matrix launch follow-up

Actual checkpoint loading later exposed Ascend's `grid < 65536` restriction. All
three fused unpack launchers now cap their grid at **65535 programs** and the
kernels stride the remaining tiles. Index-width selection still uses the padded
extent; widening is performed before offset multiplication, not after overflow.
There is no `TRITON_ALL_BLOCKS_PARALLEL` guard bypass or execution fallback.

The follow-up scoped suite passed **106 cases with zero skips** on the same
910B3/Torch-2.10/private-Triton stack. Its new hardware checks inspect **every
BF16 output bit** plus allocation guards for a `65536 × 1024` FP8 matrix, beyond
the actual grid cap; two forced-two-program checks cover complete FP4 output and
all E8M0 half-scale bits/tail guards. The scale-tail test intentionally calls the
internal output-buffer launcher: the public `fp4_int8_scale` API still allocates
its result and does not accept `out=`. The 106-case total includes portable math
and launch-wiring cases as well as real NPU comparisons, not 106 independent
hardware kernels.

The unpack-only timings below are the earlier measurement before this
large-grid follow-up; they were not re-measured or generalized to the changed
launch path. The paired runtime separately records actual full-model generation
with complete host Engram placement.

For the same original expert `(2304,2560)` and FP8 attention `(512,5120)` matrices, 20 synchronized
warm iterations in the same private environment measured:

| Operation | Torch composition | Fused | Speedup |
| --- | ---: | ---: | ---: |
| Original packed FP4 → exact INT8 ×2 | 35.76 ms | 15.02 ms | 2.38× |
| Original E4M3/E8M0 → BF16 | 27.73 ms | 7.67 ms | 3.62× |

These are unpack-only matrix timings including Python launch overhead, not GEMM, native low-bit
arithmetic, expert-forward or model throughput. The original checkpoint was not rewritten.

## Verification and limits

```bash
# Independent host scalar oracle for the actual fused integer expressions:
python -m pytest tests/test_ascend_unpack_math.py -q
# Actual tensor/device checks; skips are reported separately:
python -m pytest tests/test_online_unpack.py -q -rs
RELIC_ONLINE_UNPACK_TEST_DEVICE=npu:0 RELIC_ONLINE_UNPACK_TEST_BACKEND=torch \
  python -m pytest tests/test_online_unpack.py -q -rs
# Fused backend must be compiled and tested explicitly, not credited by Torch tests:
RELIC_ONLINE_UNPACK_TEST_DEVICE=npu:0 RELIC_ONLINE_UNPACK_TEST_BACKEND=triton \
  python -m pytest tests/test_online_unpack.py -q -rs
```

An unpack test does not validate the model's activation FP8 quantization, 32-element partial-dot
scale/reduction rules, routing, expert GEMM or generated-token parity. INT8 weight widening does not
authorize replacing original FP8 activations with W8A8. FP32/BF16 arithmetic may still flush tiny
values or overflow inside a downstream GEMM even when its input bits were decoded correctly. A
whole-model Ascend provider therefore requires separate execution and parity work.
