"""Build-metadata tests for the ``RELIC_CORE_BUILD_EXTENSIONS`` switch.

These execute ``setup.py`` against stub ``setuptools`` and ``torch.utils.cpp_extension``
modules, so the assertions run on a host with no torch, no CUDA toolkit and no compiler.
What they cover is the *build graph* setup.py hands to setuptools in each mode: which
extensions are described, with which names, sources and flags, and whether the staging
``build_ext`` command is registered at all.

They do not build a wheel, compile a kernel or assert that any extension works. A stub
``CUDAExtension`` that raises proves the disabled mode never constructs one; it cannot
prove the enabled mode compiles. That is the CUDA build host's job, not this test's.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup.py"
ENV = "RELIC_CORE_BUILD_EXTENSIONS"
ARCH_ENV = "TORCH_CUDA_ARCH_LIST"
DEFAULT_ARCH = "7.5;8.6;8.9"

#: The stubbed ``find_packages`` result, standing in for ``relic_core`` and every
#: subpackage the tree has today (including the Python-only unpack modules).
PACKAGES = ["relic_core", "relic_core.kernels"]

CUDA_KERNEL_SOURCES = [
    "relic_core/csrc/cuda_kernel.cpp",
    "relic_core/csrc/cuda_kernel_impl.cu",
    "relic_core/csrc/minimax_rope_kernel.cu",
    "relic_core/csrc/mimo_decode_ops.cpp",
    "relic_core/csrc/mimo_rope_kernel.cu",
    "relic_core/csrc/mimo_decode_attention.cu",
    "relic_core/csrc/llama_mmq/gguf_mma_wrapper.cu",
    "relic_core/csrc/qwen4_exp_moe.cu",
    "relic_core/csrc/qwen4_exp_host_kv.cu",
    "relic_core/csrc/qwen4_exp_hc_mix.cu",
    "relic_core/csrc/qwen4_exp_fp8_expert.cu",
    "relic_core/csrc/qwen4_exp_fp8_decode_moe.cu",
    "relic_core/csrc/qwen4_exp_gated_delta.cu",
    "relic_core/csrc/qwen4_exp_qsa.cu",
    "relic_core/csrc/qwen4_exp_hyper_connection.cu",
    "relic_core/csrc/xing4_hyper_connection.cu",
]
CUDA_EXTRA_COMPILE_ARGS = {
    "cxx": ["-O3"],
    "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
}
MOE_DISPATCH_SOURCES = [
    "relic_core/csrc/moe_dispatch_cuda_ext.cpp",
    "relic_core/csrc/moe_dispatch_cuda_kernel.cu",
]
CPU_MOE_SOURCES = ["relic_core/csrc/deepseek_cpu_moe_ext.cpp"]

_MODULE_NAMES = ("setuptools", "torch", "torch.utils", "torch.utils.cpp_extension")


class Recorder:
    """What the stub modules saw, so a test can assert on the build graph."""

    def __init__(self) -> None:
        self.cuda_extensions: list[dict] = []
        self.setuptools_extensions: list[dict] = []
        self.setup_calls: list[dict] = []
        self.find_packages_calls: list[dict] = []
        self.arch_list_at_setup: str | None = None
        self.build_extension_base: object = object


def _stub_modules(recorder: Recorder, *, cuda_raises: bool) -> dict[str, types.ModuleType]:
    class BuildExtension:
        pass

    class CUDAExtension:
        def __init__(self, name, sources=(), **kwargs):
            if cuda_raises:
                # The real constructor's failure on a host with no CUDA toolkit. A
                # disabled-mode test that trips this has a genuine bug, not a stub.
                raise OSError(
                    "CUDA_HOME environment variable is not set. Please set it to your CUDA install root."
                )
            recorder.cuda_extensions.append(
                {"name": name, "sources": list(sources), "kwargs": kwargs}
            )
            self.name = name

    class Extension:
        def __init__(self, name, sources=(), **kwargs):
            recorder.setuptools_extensions.append(
                {"name": name, "sources": list(sources), "kwargs": kwargs}
            )
            self.name = name

    def find_packages(include=None, **kwargs):
        recorder.find_packages_calls.append({"include": include, "kwargs": kwargs})
        return list(PACKAGES)

    def setup(**kwargs):
        recorder.setup_calls.append(kwargs)
        recorder.arch_list_at_setup = os.environ.get(ARCH_ENV)

    recorder.build_extension_base = BuildExtension
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = BuildExtension
    cpp_extension.CUDAExtension = CUDAExtension
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.cpp_extension = cpp_extension
    torch_module = types.ModuleType("torch")
    torch_module.utils = torch_utils
    setuptools_module = types.ModuleType("setuptools")
    setuptools_module.find_packages = find_packages
    setuptools_module.setup = setup
    setuptools_module.Extension = Extension
    return {
        "setuptools": setuptools_module,
        "torch": torch_module,
        "torch.utils": torch_utils,
        "torch.utils.cpp_extension": cpp_extension,
    }


def _run_setup(
    recorder: Recorder,
    value: str | None,
    *,
    cuda_raises: bool = False,
    preset_arch: str | None = None,
) -> None:
    """Execute setup.py with stubs installed, restoring modules and env afterwards.

    ``value=None`` leaves the switch unset, which must mean the historical build.
    """
    saved_modules = {name: sys.modules.get(name) for name in _MODULE_NAMES}
    saved_environ = dict(os.environ)
    try:
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(_stub_modules(recorder, cuda_raises=cuda_raises))
        os.environ.pop(ENV, None)
        os.environ.pop(ARCH_ENV, None)
        if value is not None:
            os.environ[ENV] = value
        if preset_arch is not None:
            os.environ[ARCH_ENV] = preset_arch
        namespace = {"__name__": "__main__", "__file__": str(SETUP)}
        exec(compile(SETUP.read_text(encoding="utf-8"), str(SETUP), "exec"), namespace)
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)
        for name in _MODULE_NAMES:
            if saved_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_modules[name]


def _extension_names(extensions: list[dict]) -> list[str]:
    return [row["name"] for row in extensions]


def test_python_only_mode_describes_no_extension_and_no_build_command(recorder):
    """``0`` must not even reach the constructor that needs CUDA_HOME.

    The stub raises if constructed, which is the real failure this mode exists for.
    """
    _run_setup(recorder, "0", cuda_raises=True)

    assert recorder.cuda_extensions == []
    assert recorder.setuptools_extensions == []
    assert len(recorder.setup_calls) == 1
    kwargs = recorder.setup_calls[0]
    assert kwargs["ext_modules"] == []
    assert kwargs["cmdclass"] == {}


def test_python_only_mode_keeps_packages_and_the_arch_default(recorder):
    """The switch changes the extensions only: package discovery and the arch pin stay."""
    _run_setup(recorder, "0")

    kwargs = recorder.setup_calls[0]
    assert kwargs["packages"] == PACKAGES
    assert recorder.find_packages_calls == [{"include": ["relic_core", "relic_core.*"], "kwargs": {}}]
    assert recorder.arch_list_at_setup == DEFAULT_ARCH


def test_default_mode_builds_the_three_original_extensions(recorder):
    """With the switch unset, the three extensions are exactly the historical ones."""
    _run_setup(recorder, None)

    assert _extension_names(recorder.cuda_extensions) == ["cuda_kernel", "moe_dispatch_cuda_ext"]
    assert _extension_names(recorder.setuptools_extensions) == ["deepseek_cpu_moe_ext"]

    cuda_kernel, moe_dispatch = recorder.cuda_extensions
    assert cuda_kernel["sources"] == CUDA_KERNEL_SOURCES
    assert cuda_kernel["kwargs"] == {
        "libraries": ["cublas"],
        "extra_compile_args": CUDA_EXTRA_COMPILE_ARGS,
    }
    assert moe_dispatch["sources"] == MOE_DISPATCH_SOURCES
    assert moe_dispatch["kwargs"] == {"extra_compile_args": CUDA_EXTRA_COMPILE_ARGS}

    cpu_moe = recorder.setuptools_extensions[0]
    assert cpu_moe["sources"] == CPU_MOE_SOURCES
    assert cpu_moe["kwargs"] == {
        "extra_compile_args": ["-O3", "-mavx2", "-mfma", "-fopenmp"],
        "extra_link_args": ["-fopenmp"],
    }

    kwargs = recorder.setup_calls[0]
    assert [extension.name for extension in kwargs["ext_modules"]] == [
        "cuda_kernel",
        "deepseek_cpu_moe_ext",
        "moe_dispatch_cuda_ext",
    ]
    assert set(kwargs) == {"packages", "ext_modules", "cmdclass"}
    assert kwargs["packages"] == PACKAGES
    assert recorder.find_packages_calls[0]["include"] == ["relic_core", "relic_core.*"]
    assert recorder.arch_list_at_setup == DEFAULT_ARCH

    (build_ext,) = kwargs["cmdclass"].values()
    assert kwargs["cmdclass"].keys() == {"build_ext"}
    assert build_ext.__name__ == "BuildExtensions"
    assert issubclass(build_ext, recorder.build_extension_base)


def test_explicit_one_is_the_same_as_unset(recorder):
    """``1`` is the default spelled out, not a third behaviour."""
    _run_setup(recorder, "1")

    assert _extension_names(recorder.cuda_extensions) == ["cuda_kernel", "moe_dispatch_cuda_ext"]
    assert _extension_names(recorder.setuptools_extensions) == ["deepseek_cpu_moe_ext"]
    assert len(recorder.setup_calls[0]["ext_modules"]) == 3
    assert set(recorder.setup_calls[0]["cmdclass"]) == {"build_ext"}


def test_preset_arch_list_is_not_overwritten(recorder):
    """The arch pin is a default, so a caller's ``TORCH_CUDA_ARCH_LIST`` still wins."""
    _run_setup(recorder, "1", preset_arch="8.0")

    assert recorder.arch_list_at_setup == "8.0"


@pytest.mark.parametrize("value", ["", "2", "yes", "true", " 0", "0 ", "01"])
def test_invalid_values_are_refused_before_setup_is_called(recorder, value):
    """A typo must fail loudly instead of being read as either mode."""
    with pytest.raises(RuntimeError, match=ENV):
        _run_setup(recorder, value)

    assert recorder.setup_calls == []
    assert recorder.cuda_extensions == []
    assert recorder.setuptools_extensions == []


def test_the_stub_would_catch_a_construction_in_default_mode(recorder):
    """Sanity check on the test harness: the raising stub is not a no-op."""
    with pytest.raises(OSError, match="CUDA_HOME"):
        _run_setup(recorder, "1", cuda_raises=True)


def test_environment_and_modules_are_restored(recorder):
    """The harness leaks neither the switch nor the stubs into the rest of the suite."""
    saved_environ = dict(os.environ)
    saved_modules = {name: sys.modules.get(name) for name in _MODULE_NAMES}

    _run_setup(recorder, "0")

    assert dict(os.environ) == saved_environ
    assert {name: sys.modules.get(name) for name in _MODULE_NAMES} == saved_modules


def test_environment_and_modules_are_restored_when_refused(recorder):
    """The restore also has to survive the error path."""
    saved_environ = dict(os.environ)
    saved_modules = {name: sys.modules.get(name) for name in _MODULE_NAMES}

    with pytest.raises(RuntimeError, match=ENV):
        _run_setup(recorder, "false")

    assert dict(os.environ) == saved_environ
    assert {name: sys.modules.get(name) for name in _MODULE_NAMES} == saved_modules


@pytest.fixture
def recorder():
    return Recorder()