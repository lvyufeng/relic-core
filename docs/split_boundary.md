# What belongs in relic-core

relic-core is the operator layer: kernels, and the loader that finds them. It holds no model code
and no engine. This document records where each part of the former monorepo landed, because one
layer at the boundary is easy to misplace and the mistake is silent — it shows up as an import edge
at runtime, not as a review comment.

## The rule

**A file goes in relic-core if it is an op, or the loader for ops. A file that *wraps* ops for a
model goes with the model.**

The rule exists because the wrapper looks like an op and is not one. `quantized_ops.py` sits in a
directory named `components/gguf/`, exports CUDA-op bindings, and is imported by four model
families — every signal says "operator". It is not: it imports the GGUF loader at module level, and
the loader is model-side. Putting it here would give relic-core an import edge into `src.loader.*`,
which is exactly the boundary this split is for.

## The boundary

| Path (former monorepo) | What it is | Home |
|---|---|---|
| `src/kernels/cuda_loader.py` | dlopen/extension loader, no op semantics | **relic-core** — the one amendment to the ops-only rule, since every op user needs it |
| `src/csrc/*` | the C++/CUDA sources | **relic-core** — one physical copy |
| `src/csrc/cuda_kernel_impl.cu` (`iq4nl_block_dot_256`) | the IQ4_NL CUDA kernel | **relic-core** — one physical copy |
| `src/kernels/{ops,int8_*_triton,moe_dispatch_loader,tq4nc_quantizer}.py` | the Python dispatch layer over the ops | **relic-core** |
| `src/components/gguf/quantized_ops.py` | Python CUDA-op wrapper; imports the GGUF loader | **with the model** (RelicLLM) — **not** here |
| `src/loader/gguf/{quantized_loader,quantized_tensor,quant_types,iq4_nl,pq2_0,ptq1_0,...}.py` | the GGUF dequant/load layer | **with the model** (RelicLLM) — same reason |

## Why the last two rows cannot be split apart

`quantized_ops.py` imports `src.loader.gguf.quantized_tensor` at module level, and is itself
imported by `src/models/{xing4_0,glm_dsa,minimax_m2}`. The wrapper and the loader are one layer:
separating them creates an import cycle, and moving the wrapper across the boundary creates a
reverse edge from the ops library into model code.

## Checks

These hold today and are the ones to re-run after a move:

```bash
# relic-core must not import the model side, at any depth.
grep -rn '^\s*\(from\|import\)\s\+src\.' relic_core/          # expect: no output

# exactly one IQ4_NL kernel, defined once.
grep -rln 'iq4nl' --include=*.cu --include=*.cpp .             # expect: relic_core/csrc/cuda_kernel_impl.cu

# and it is a definition, not a call site.
grep -rn '__forceinline__ float iq4nl_block_dot_256' \
     --include=*.cu --include=*.cpp .                          # expect: one hit
```

## Related

- `../README.md` — what relic-core is, and how to build it.
- RelicLLM's README — the model-side half, and what deliberately does not live there.