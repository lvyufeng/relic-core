# Installing the Python package without the CUDA extensions

`setup.py` builds three native extensions — `cuda_kernel`, `deepseek_cpu_moe_ext` and
`moe_dispatch_cuda_ext`. On a host that has torch but **no CUDA toolkit**, describing that build is
already a failure: constructing `CUDAExtension` requires `CUDA_HOME`, so installation stops with

```text
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

This page is the explicit way to install the Python package on such a host (an Ascend 910B box, for
instance) and what is different about that install. For the normal card build see
[CUDA extension builds](cuda_extension_builds.md).

## The switch

`RELIC_CORE_BUILD_EXTENSIONS` is read by `setup.py` at build time only; nothing reads it at import
or at run time. Only the exact values `0` and `1` are accepted, and an unset variable means `1` —
the historical build — so a typo is refused rather than guessed:

| Value | What the build describes | When it is right |
| --- | --- | --- |
| unset or `1` | the three extensions, exactly as before | any host that builds and runs the `sm_75`/Ampere/Ada kernels |
| `0` | no extension at all; Python modules only | a host with torch but no CUDA toolkit |

With `0`, `CUDAExtension`/`Extension` are never instantiated, no `build_ext` command with the
staging step is registered, and `find_packages()` is untouched — so the install is the ordinary
Python package, including the newer `relic_core.kernels.unpack` and optional Triton-Ascend launcher
modules. Package discovery does not change between modes; only the extension list does.

## Command

From the `relic-core` checkout, in the environment whose torch the modules should run against:

```bash
RELIC_CORE_BUILD_EXTENSIONS=0 python -m pip install -e . --no-build-isolation --no-deps
```

Both flags are required for the same reasons as the normal build: `--no-build-isolation` keeps the
active environment's torch, and `--no-deps` stops the resolver from replacing it. A Python-only
install does not remove the runtime `torch>=2.0` dependency; it only skips compiling the extensions.
Run it in the same environment the modules will run in, because `setup.py` still imports
`torch.utils.cpp_extension` at module level (to define the staging `build_ext` class the default
mode uses), so torch — and whatever runtime variables its accelerator backend needs to import — has
to be in place even when no extension is built.

Re-running a plain `pip install -e . --no-deps` later on the same host will fail again unless the
variable is set again — that is the explicit switch working, not a broken install.

## Verifying the result

The loader reports a missing build as `None` by design, so absence is not visible at import:

```bash
python - <<'PY'
import relic_core
from relic_core.kernels.cuda_loader import load_cuda_kernel
from relic_core.kernels import unpack

assert load_cuda_kernel() is None, "the Python-only install should have no native extension"
assert hasattr(unpack, "unpack_fp8_to_bf16")
print("python-only relic_core:", relic_core.__version__)
PY
```

`python -m pytest tests/ -q -rs` is the other check: tests that need the extension turn the missing
`.so` into a named skip rather than an `AttributeError`, so a run of skips is "not built here", not
"passed".

## What this install does not provide

- **No CUDA kernels.** `cuda_kernel` and `moe_dispatch_cuda_ext` are absent, so the quantized
  GEMM/MoE kernels a CUDA card host runs do not exist in this install.
- **No x86 AVX2 host expert path.** `deepseek_cpu_moe_ext` is also skipped; on a non-x86 host it
  would not have been usable anyway, but on an x86 host without CUDA this mode also drops it.
- **No fused expert provider.** The Python-only build compiles nothing, so it does not build the
  fused MoE/expert kernels, and the Triton unpack launcher still needs an ABI-compatible
  Triton-Ascend installation of its own before it can run. This page does not claim that installing
  the Python package makes DeepSeek-V4.1-Flash loadable or servable on Ascend: a runtime capability
  gate and a missing expert executor are separate blockers recorded with the deployment work.
- **No native FP4/FP8 arithmetic.** The unpack modules decode original floating-format bytes to
  wider types in software; they do not make a card perform FP4/FP8 matrix multiplication.

The default build is unchanged: `RELIC_CORE_BUILD_EXTENSIONS=1` (or unset) still builds the same
three extensions with the same names, sources, `7.5;8.6;8.9` arch list and `build/extensions`
staging described in [CUDA extension builds](cuda_extension_builds.md).