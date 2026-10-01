# relic-core documentation

The operator layer under [RelicLLM](https://github.com/lvyufeng/RelicLLM) and
[PocketLLM](https://github.com/lvyufeng/PocketLLM): CUDA `sm_75` kernels and the CPU host ops an
offload path runs. No model code, no engine, no serving front end.

The repository's `README.md` is the front door — what this package is, how to build it, and which
three extensions it produces. This page indexes the documents behind it.

## Why the set is this small

`docs/` here holds the pages whose *subject* is the operator layer or the hardware it exists for.
Everything else the three repositories used to share stayed with the runtime it describes, or went
to [relic-engine](https://github.com/lvyufeng/relic-engine), the frozen archive of the retired C++
engine. Five pages came across; that is not a truncation, it is the boundary doing its job.

## Pages

| Document | What it covers |
|---|---|
| [Where the boundary runs](split_boundary.md) | The rule that decides what belongs in this package, the one case that looks like a violation and is not, and the greps that check it after a move. **Start here.** |
| [Ternary-Bonsai-2-27B dense GEMM](architecture/ternary_bonsai_2_dense_gemm.md) | The ternary-weight dense GEMM path on sm_75 — block format, Hadamard fold, and why the kernel is shaped the way it is. |
| [CUDA extension builds](guides/cuda_extension_builds.md) | Building and loading the extensions: the two flags that matter, the `CUDA_HOME` mismatch trap, and the microbenchmark harness. |
| [Ascend SoC generations](guides/ascend_soc_generations.md) | `910B` against `910B1`–`910B4`: how to read the SoC generation off the machine rather than the card's printed name, and why the two need separate kernels. |
| [DeepSeek-V4.1-Flash MoE weight staging](performance/deepseek_v4_1_flash_moe_weight_staging.md) | The staging path that feeds the routed experts — what is copied, where it waits, and what it costs. |
| [DeepSeek-V4.1-Flash MoE reduction](performance/deepseek_v4_1_flash_moe_reduce_csr.md) | The expert-output reduction: CSR layout, the fixed-order reduction that replaced `atomicAdd`, and its cost. |

## A note on links that leave this site

Documents here cite measurements and designs that live in the other repositories. Those links are
written as **absolute URLs**, not as `../` relative paths, and that is deliberate:

- this site is built with `mkdocs build --strict`, which makes the build a link checker, and it can
  only check links inside this repository. A relative path into `../RelicLLM/docs/…` would be
  unresolvable from a published page and would pass the build by never being inspected;
- a relative path would also break silently in every direction the file could move — on GitHub, in
  the built site, and in a clone that has only this repository.

So a relative link here means "another page in this repository", and an absolute URL means "a
record kept somewhere else". The second kind is not verified by the build and is worth a click when
it matters.