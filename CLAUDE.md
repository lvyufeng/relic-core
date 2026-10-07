# relic-core

Shared **torch operator library for older accelerators**: CUDA `sm_75` (RTX 2080 Ti) and the CPU
host ops the offload path runs. No model code, no engine, no serving front end. The two consumers
are [RelicLLM](https://github.com/lvyufeng/RelicLLM) (multi-GPU runtime) and
[PocketLLM](https://github.com/lvyufeng/PocketLLM) (single-card / edge).

## Language convention

**All Markdown and code comments are written in English.** Commit messages, docstrings and every
`.md` file. A Chinese version of a document is a separate file (`README.md` / `README_CN.md`), never
a mixed-language one.

## Layout

| Path | What it is |
|---|---|
| `relic_core/csrc/` | C++/CUDA sources, built into three extension modules |
| `relic_core/kernels/` | Python op layer — `ops.py` dispatch, `cuda_loader.py` extension lookup |
| `docs/split_boundary.md` | where the boundary with the model side runs, and the greps that check it |
| `docs/` | the published site's source (`mkdocs.yml`), six pages — indexed by `docs/README.md` |
| `tests/` | the kernel test suite and its baseline |

**The boundary rule is the load-bearing invariant.** A file belongs here if it *is* an op or the
loader for ops; a file that *wraps* ops for a model belongs with the model. `docs/split_boundary.md`
records the one case that looks like a violation and is not, plus the checks to re-run after a move.
The reverse edge is what breaks: this package must never import model-side code, at any depth —

```bash
grep -rn '^\s*\(from\|import\)\s\+src\.' relic_core/     # expect: no output
```

## Build

```bash
pip install -e . --no-build-isolation --no-deps
```

**Both flags are required**, for different reasons.

`--no-build-isolation` — `setup.py` imports `torch.utils.cpp_extension` at module level, so a torch
resolved fresh into an isolated build env is not the one the extensions link against.

`--no-deps` — torch is resolved from the environment, not from PyPI. The dependency is `torch>=2.0`
with **no upper bound, and adding one is a bug**: the development box runs a torch newer than any pin
would pick, and a resolver that installs an older one rebuilds every kernel in the tree against the
wrong ABI. `requirements.txt` and `pyproject.toml` must agree on this.

`TORCH_CUDA_ARCH_LIST` defaults to `7.5;8.9` — the Turing / RTX 2080 Ti card this library was
written for, and Ada / RTX 4090, which runs the same sources through its own sm_89 device code.
**Do not drop the sm_75-specific kernel paths or the `7.5` from the default** — this library exists
for that card, and neither arch's SASS runs on the other. The two are one fatbin, no PTX, so a
mismatch is not a slow path but a launch failure ("no kernel image is available for execution on the
device") that only appears when a kernel is actually launched.

If the environment's torch names a CUDA version with no matching toolkit on the box, point the build
at the nearest one that exists; `torch` does not check the minor version, but `nvcc` must be on
`PATH` ahead of any other toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
```

Three extensions are produced under their original names, so downstream loaders keep working:
`cuda_kernel`, `deepseek_cpu_moe_ext`, `moe_dispatch_cuda_ext`. `setup.py` stages them into
`build/extensions/`, which is where `cuda_loader.load_cuda_kernel()` looks.

**A missing `.so` is silent.** The loader catches every exception and returns `None`, and the tests
turn that into a named skip rather than an `AttributeError`. So a run of all-skips means "not built
for this interpreter", not "this host has no card" — two different problems that look identical from
a test summary.

## Testing

Run from the repository root, or the modules cannot import (no `conftest.py`, no pytest config, and
`tests/` has no `__init__.py` — the CWD is what puts `tests` on `sys.path`):

```bash
python -m pytest tests/ -q
python scripts/check_test_baseline.py     # diff the run against tests/baseline_failures.txt
```

`tests/baseline_failures.txt` is a **set of node ids, not a count**, and the distinction is the
point: three tests fixed and one broken is a net improvement in a count and a regression in the
tree. An entry means the test *runs here and fails*; a skip belongs nowhere in it. `tests/README.md`
has the reasoning, the skip policy and the recorder.

**No CI runs the suite.** The only workflow here is `.github/workflows/pages.yml`, which builds the
documentation site, so the baseline check is a manual step.

That workflow runs `mkdocs build --strict`, which is the repository's link checker — but only
*within* `docs/`. It also fails when `docs/llms.txt` is stale with the nav, because
`hooks/llms_txt_staleness.py` checks it on every build and `--strict` promotes the warning. So a nav
edit and the regenerated `docs/llms.txt` (`python scripts/gen_llms_txt.py`) belong in the same
commit. Links that leave the repository are absolute URLs the build cannot see; `docs/README.md`
says why they are written that way.

## Git workflow

**Never commit directly to `master`.** Every change goes on a branch and through a pull request.

Branch prefixes: `feature/`, `fix/`, `refactor/`, `docs/`, `perf/` — `<prefix>/<description>`.

Commits: a one-line summary under 72 characters, a blank line, then the explanation starting on
line 3. **Every commit message ends with:**

```
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

The address is `noreply@anthropic.com`. Do not "normalize" older commits that carry a different one
toward this — rewriting them is a force-push over other people's work.

PRs: a title under 72 characters, a body covering summary, implementation details and testing
status, **one concern per PR**, and the body ends with:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Merged branches are not reliably deleted on `origin`; delete yours locally and remotely.

## These facts are per-host

The build flags, the CUDA-toolkit workaround and the torch version above were checked on one x86_64
box with 4 x RTX 2080 Ti. They are not universal claims — in particular the CUDA path does not exist
at all on an Ascend host, and a statement that a specific toolkit or driver is present is a statement
about the machine it was checked on. Re-check in place before trusting one elsewhere.

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo (`src/csrc/`, `src/kernels/`), history
preserved. The package was renamed `src.kernels` → `relic_core.kernels` so it cannot collide with a
runtime's own `src` package on `sys.path`.