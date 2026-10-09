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

Where the boundary with the model side runs — in particular why the GGUF op wrapper is *not* here —
is in [`docs/split_boundary.md`](docs/split_boundary.md), with the greps that check it.

## Build

```bash
pip install -e . --no-build-isolation --no-deps
```

Both flags matter, for different reasons.

`--no-build-isolation` is required because `setup.py` imports `torch.utils.cpp_extension` at module
level: a Torch resolved fresh into an isolated build env is not the one the extensions must link
against.

`--no-deps` is required because Torch is resolved from the environment, not from PyPI — the
dependency is `torch>=2.0` with no upper bound, and the box this is developed on runs a torch newer
than any pin would pick. Without the flag pip may reinstall a *different* torch and rebuild every
kernel in the tree against the wrong ABI. Install it explicitly first if the environment has none.

`TORCH_CUDA_ARCH_LIST` defaults to `7.5;8.9`, which serves both the RTX 2080 Ti this library was
written for and the RTX 4090 that runs the same sources through sm_89 device code; do not drop the
sm_75-specific kernel paths, and do not drop `7.5` from the default.

If the environment's torch is built against a CUDA version with no matching toolkit on the box,
point the build at the nearest one that exists rather than the one torch names — `torch` does not
check the minor version, but `nvcc` must be on `PATH` ahead of any other toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
```

Three extensions are produced, under their original names so downstream loaders keep working:
`cuda_kernel`, `deepseek_cpu_moe_ext`, `moe_dispatch_cuda_ext`.

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo (`src/csrc/`, `src/kernels/`),
history preserved. The package was renamed `src.kernels` → `relic_core.kernels` so it cannot
collide with a runtime's own `src` package on `sys.path`.
