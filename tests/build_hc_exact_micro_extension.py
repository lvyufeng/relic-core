"""Build the HC up-mix candidate as a standalone extension for isolated A/B testing.

The qualified native module is the only allowed source of every *existing* operator, so this script
never rebuilds it. It compiles ``qwen4_exp_hc_mix_exact.cu`` alone into a private extension named
``cuda_kernel`` -- the leaf name CPython requires for the entry point -- and the probe attaches the
single new binding to the qualified module as one extra attribute. Every other attribute keeps its
identity, so a measured divergence can only come from the new entry.

The compile flags are the qualified build's: ``-O3 --use_fast_math``. That matters for the bit
comparison, because fast math decides how the sigmoid division and the fused multiply-adds are
emitted.

Run:   python tests/build_hc_exact_micro_extension.py
Print: one line, the absolute path of the built shared object.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "relic_core" / "csrc" / "qwen4_exp_hc_mix_exact.cu"
BUILD = Path(os.environ.get("Q4_HC_EXACT_BUILD_DIR", "/tmp/qwen4-hc-exact-micro"))

WRAPPER = r"""
#include <torch/extension.h>

torch::Tensor qwen4_exp_hc_up_mix_exact(
    const torch::Tensor& down,
    const torch::Tensor& mix_up,
    const torch::Tensor& normed,
    int64_t hc,
    int64_t hidden,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qwen4_exp_hc_up_mix_exact", &qwen4_exp_hc_up_mix_exact,
          "Hyper-connection mixer: four stream dots from one activation read (isolated build)",
          pybind11::arg("down"), pybind11::arg("mix_up"), pybind11::arg("normed"),
          pybind11::arg("hc"), pybind11::arg("hidden"), pybind11::arg("threads") = 128);
}
"""


def _ensure_ninja_on_path() -> None:
    """torch's loader shells out to `ninja`, so an importable package alone is not enough."""
    import shutil
    import sys

    if shutil.which("ninja"):
        return
    candidate = Path(sys.executable).with_name("ninja")
    if candidate.is_file():
        os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def main() -> int:
    _ensure_ninja_on_path()
    if not SOURCE.is_file():
        raise SystemExit(f"missing source: {SOURCE}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: the candidate is a device operator")
    from torch.utils.cpp_extension import load

    BUILD.mkdir(parents=True, exist_ok=True)
    wrapper = BUILD / "hc_exact_binding.cpp"
    wrapper.write_text(WRAPPER)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    module = load(
        name="cuda_kernel",
        sources=[str(wrapper), str(SOURCE)],
        build_directory=str(BUILD),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    if getattr(module, "qwen4_exp_hc_up_mix_exact", None) is None:
        raise SystemExit("the built extension does not export qwen4_exp_hc_up_mix_exact")
    print(str(Path(module.__file__).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())