"""Build relic-core's extension modules.

Three separate extensions, exactly as the PocketLLM monorepo built them, so a
downstream runtime that already loads one of these names keeps working:

    cuda_kernel            the main CUDA/quantized-geMM/MoE kernel library
    deepseek_cpu_moe_ext   the AVX2/FMA/OpenMP CPU host kernel (offload path)
    moe_dispatch_cuda_ext  the MoE dispatch kernel

Build with ``--no-build-isolation`` (see pyproject.toml).  ``TORCH_CUDA_ARCH_LIST``
selects the arch; it defaults to ``7.5;8.6;8.9`` so that one build serves every card
this library has been validated on or targets -- Turing / RTX 2080 Ti, whose
sm_75-specific paths are half the tree; Ampere / RTX 3090; and Ada / RTX 4090.  Set
it to a single arch to build for one of them; all three are in the one fatbin a
default build produces.

Volta / V100 is deliberately **not** in this list, and it is not an omission: CUDA
13 removed offline compilation for Volta, so a build box with the toolkit this
library is built against cannot emit sm_70 at all.  A V100 artifact is a second
build from a CUDA 12.x toolkit, which is what ``docs`` in this repository record.

The default carries both arches rather than only the newer one because the two
cards cannot share a *binary*: ``sm_75`` SASS does not run on an Ada card and Ada
SASS does not run on Turing, and these gencodes embed SASS and no PTX, so there is
no JIT fallback to catch a mismatch -- a build for the wrong arch imports fine (the
loader only ``exec_module``s) and dies at its first kernel launch with "no kernel
image is available for execution on the device".  One fatbin holding both is what
lets ``cuda_loader`` stay arch-blind and keep loading by bare name.
"""

import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import Extension

ROOT = Path(__file__).resolve().parent
EXTENSIONS_DIR = ROOT / "build" / "extensions"

# Pin the arch rather than inheriting whatever the box has. A build that silently
# targets sm_80+ produces a library that cannot load on the 2080 Ti it exists for;
# one that targets only 7.5 cannot load on a 3090 or a 4090. All three are in this
# list, and the sm_75 half is not optional -- the release that dropped it would
# strand the card half the kernel tree is written for.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.6;8.9")


class BuildExtensions(BuildExtension):
    """BuildExtension that also stages the .so in build/extensions.

    ``relic_core.kernels.cuda_loader`` looks for the extension by bare name in
    ``<repo>/build/extensions`` and then next to the repo root -- that is how the
    monorepo found it, and keeping the layout means no caller has to change.  So
    every built module is copied there as well as into the install tree.
    """

    def run(self) -> None:
        super().run()
        self._stage_in_extensions_dir()

    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        self._stage_in_extensions_dir()

    def _stage_in_extensions_dir(self) -> None:
        if not self.build_lib:
            return
        built = sorted(Path(self.build_lib).glob("*.so"))
        if not built:
            return
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for module in built:
            shutil.copy2(module, EXTENSIONS_DIR / module.name)


ext_modules = [
    CUDAExtension(
        name="cuda_kernel",
        sources=[
            "relic_core/csrc/cuda_kernel.cpp",
            "relic_core/csrc/cuda_kernel_impl.cu",
            "relic_core/csrc/minimax_rope_kernel.cu",
            "relic_core/csrc/mimo_decode_ops.cpp",
            "relic_core/csrc/mimo_rope_kernel.cu",
            "relic_core/csrc/mimo_decode_attention.cu",
            "relic_core/csrc/llama_mmq/gguf_mma_wrapper.cu",
            "relic_core/csrc/qwen4_exp_moe.cu",
            "relic_core/csrc/qwen4_exp_gated_delta.cu",
            "relic_core/csrc/qwen4_exp_qsa.cu",
            "relic_core/csrc/qwen4_exp_hyper_connection.cu",
            "relic_core/csrc/xing4_hyper_connection.cu",
        ],
        libraries=["cublas"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    ),
    Extension(
        name="deepseek_cpu_moe_ext",
        sources=["relic_core/csrc/deepseek_cpu_moe_ext.cpp"],
        extra_compile_args=["-O3", "-mavx2", "-mfma", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    CUDAExtension(
        name="moe_dispatch_cuda_ext",
        sources=[
            "relic_core/csrc/moe_dispatch_cuda_ext.cpp",
            "relic_core/csrc/moe_dispatch_cuda_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    ),
]

setup(
    packages=find_packages(include=["relic_core", "relic_core.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensions},
)