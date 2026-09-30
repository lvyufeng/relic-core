# relic-core

Shared **torch operator library for older accelerators**: CUDA `sm_75` (RTX 2080 Ti) and CPU host
ops used by the offload path. It holds **no model code and no engine** — it is the operator layer
that [RelicLLM](https://github.com/lvyufeng/RelicLLM) and
[PocketLLM](https://github.com/lvyufeng/PocketLLM) both depend on.

## Layout

| Path | What it is |
|---|---|
| `relic_core/csrc/` | C++/CUDA sources, built into the extension modules |
| `relic_core/kernels/` | the Python op layer (`ops.py` dispatch, `cuda_loader.py` extension lookup) |

## Build

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required: `setup.py` imports `torch.utils.cpp_extension`, and a Torch
resolved into an isolated build env does not match the CUDA toolkit the extensions link against.
`TORCH_CUDA_ARCH_LIST` defaults to `7.5`; do not drop the sm_75-specific kernel paths.

Three extensions are produced, under their original names so downstream loaders keep working:
`cuda_kernel`, `deepseek_cpu_moe_ext`, `moe_dispatch_cuda_ext`.

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo (`src/csrc/`, `src/kernels/`),
history preserved. The package was renamed `src.kernels` → `relic_core.kernels` so it cannot
collide with a runtime's own `src` package on `sys.path`.
