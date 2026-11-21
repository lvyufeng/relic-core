"""Launch-wiring contract tests for the fused Ascend unpack backend.

These are CPU-only unit tests. They load ``relic_core/kernels/ascend_unpack.py``
by file, inject a fake ``torch``/``triton`` plus a fake kernels module, and
record each kernel's launch grid and positional arguments. Nothing here compiles
Triton, talks to an NPU, or checks numerical results; those are covered by the
separate frontend and math tests. The point of this module is the layer in
between: which grid, which pointer-index width and which argument order the
launcher hands the kernels, including the masked tail of the final block.

Root contract pinned here (2026-11-17):
  * grid = min(65535, ceil(numel / 1024)); kernels stride remaining tiles;
  * the pointer-index width is selected from the *padded* element count, not
    from the logical count, because the kernel derives addresses for masked
    tail lanes before the mask suppresses the access:
      - FP4 unpack: ``padded(raw.numel()) * 2 >= 2**31``;
      - FP8 unpack and E8M0 scale decode: ``padded(raw.numel()) >= 2**31``.
    The padded decision is deliberately conservative by one block.
  * an empty input enters the NPU/backend checks but launches no kernel.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types

import pytest


LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1] / "relic_core" / "kernels" / "ascend_unpack.py"
)
BLOCK = 1024
INDEX_LIMIT = 2**31
FIXTURE_PACKAGE = "relic_core_kernels_launch_fixture"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _padded(numel: int) -> int:
    """Element count including the masked lanes of the final block."""
    return _ceil_div(numel, BLOCK) * BLOCK


def _grid(numel: int) -> tuple[int]:
    return (min(65535, _ceil_div(numel, BLOCK)),)


def _fp4_index64(numel: int) -> bool:
    return _padded(numel) * 2 >= INDEX_LIMIT


def _unary_index64(numel: int) -> bool:
    return _padded(numel) >= INDEX_LIMIT


class FakeDevice:
    def __init__(self, kind: str) -> None:
        self.type = kind


class FakeTensor:
    """Minimal stand-in: the launcher only reads ``device``, ``shape``, ``numel()``."""

    def __init__(self, shape: tuple[int, ...] = (), device: str = "npu") -> None:
        self.shape = tuple(shape)
        self.device = FakeDevice(device)

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


class DeviceRecorder:
    def __init__(self) -> None:
        self.stack: list[object] = []
        self.events: list[tuple[str, object]] = []

    @contextmanager
    def device(self, device: object):
        self.stack.append(device)
        self.events.append(("enter", device))
        try:
            yield
        finally:
            self.events.append(("exit", device))
            self.stack.pop()


class FakeKernel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.grid: object = None
        self.calls: list[tuple[object, tuple[object, ...]]] = []
        self.error: BaseException | None = None

    def __getitem__(self, grid: object) -> "FakeKernel":
        self.grid = grid
        return self

    def __call__(self, *args: object) -> None:
        self.calls.append((self.grid, args))
        if self.error is not None:
            raise self.error


class Target:
    def __init__(self, backend: str = "ascend") -> None:
        self.backend = backend


class Harness:
    def __init__(self, module, kernels, recorder: DeviceRecorder, target: Target) -> None:
        self.module = module
        self.kernels = kernels
        self.recorder = recorder
        self.target = target


@pytest.fixture
def harness(monkeypatch) -> Harness:
    spec = importlib.util.spec_from_file_location(
        FIXTURE_PACKAGE + ".ascend_unpack", LAUNCHER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Match the spec parent so the lazy ``from ._ascend_unpack_kernels`` import
    # resolves inside the injected fixture package without a deprecation path.
    module.__package__ = FIXTURE_PACKAGE
    spec.loader.exec_module(module)

    kernels = types.SimpleNamespace(
        fp4_unpack_int8_kernel=FakeKernel("fp4_unpack_int8"),
        fp8_unpack_bf16_kernel=FakeKernel("fp8_unpack_bf16"),
        fp4_int8_scale_kernel=FakeKernel("fp4_int8_scale"),
    )
    package = types.ModuleType(FIXTURE_PACKAGE)
    package.__path__ = []
    kernels_module = types.ModuleType(FIXTURE_PACKAGE + "._ascend_unpack_kernels")
    for name, value in vars(kernels).items():
        setattr(kernels_module, name, value)
    monkeypatch.setitem(sys.modules, FIXTURE_PACKAGE, package)
    monkeypatch.setitem(sys.modules, FIXTURE_PACKAGE + "._ascend_unpack_kernels", kernels_module)
    module.__package__ = FIXTURE_PACKAGE

    monkeypatch.setattr(module, "supports_triton_ascend", lambda: True)

    recorder = DeviceRecorder()
    fake_torch = types.ModuleType("torch")
    fake_torch.npu = types.SimpleNamespace(device=recorder.device)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    target = Target()
    fake_triton = types.ModuleType("triton")
    fake_triton.runtime = types.SimpleNamespace(
        driver=types.SimpleNamespace(
            active=types.SimpleNamespace(get_current_target=lambda: target)
        )
    )
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    return Harness(module, kernels, recorder, target)


def _expected(actual: object, expected: object) -> bool:
    """Identity for tensor-like sentinels, equality for integer/boolean scalars."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    return actual is expected


def assert_call(kernel: FakeKernel, grid: tuple[int], expected: tuple[object, ...]) -> None:
    assert len(kernel.calls) == 1, f"{kernel.name}: expected exactly one launch, got {kernel.calls}"
    recorded_grid, recorded = kernel.calls[0]
    assert recorded_grid == grid, f"{kernel.name}: grid {recorded_grid} != {grid}"
    assert len(recorded) == len(expected), (
        f"{kernel.name}: {len(recorded)} positional args, expected {len(expected)}"
    )
    for index, (actual, wanted) in enumerate(zip(recorded, expected)):
        assert _expected(actual, wanted), (
            f"{kernel.name}: positional arg {index} is {actual!r}, expected {wanted!r}"
        )


def assert_no_launch(*kernels: FakeKernel) -> None:
    for kernel in kernels:
        assert kernel.calls == [], f"{kernel.name}: empty input must not launch a kernel"


# --------------------------------------------------------------------------- FP4


def test_fp4_unpack_launch_grid_and_argument_order(harness: Harness) -> None:
    raw, out = FakeTensor((8,)), object()
    result = harness.module.fp4_unpack_int8(raw, out)
    assert result is out
    assert_call(
        harness.kernels.fp4_unpack_int8_kernel,
        _grid(8),
        (raw, out, 8, BLOCK, False),
    )


@pytest.mark.parametrize(
    ("numel", "grid"),
    [(1, (1,)), (1023, (1,)), (1024, (1,)), (1025, (2,))],
)
def test_fp4_small_counts_keep_one_grid_entry_per_block(
    harness: Harness, numel: int, grid: tuple[int]
) -> None:
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert_call(
        harness.kernels.fp4_unpack_int8_kernel,
        grid,
        (raw, out, numel, BLOCK, False),
    )


def test_fp4_empty_input_enters_checks_but_launches_nothing(harness: Harness) -> None:
    raw, out = FakeTensor((0,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert_no_launch(harness.kernels.fp4_unpack_int8_kernel)


def test_fp4_padded_tail_selects_wide_index_before_logical_limit(harness: Harness) -> None:
    # padded = 2**30, doubled = 2**31: the masked tail block reaches the limit
    # even though the logical doubled count (2**31 - 2046) is below it.
    numel = 2**30 - 1023
    assert 2 * numel < INDEX_LIMIT
    assert _padded(numel) * 2 == INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert_call(
        harness.kernels.fp4_unpack_int8_kernel,
        _grid(numel),
        (raw, out, numel, BLOCK, True),
    )


def test_fp4_padded_block_below_limit_keeps_narrow_index(harness: Harness) -> None:
    numel = 2**30 - 1024
    assert _padded(numel) * 2 < INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert_call(
        harness.kernels.fp4_unpack_int8_kernel,
        _grid(numel),
        (raw, out, numel, BLOCK, False),
    )


def test_fp4_index_width_at_exact_padded_limit_is_wide(harness: Harness) -> None:
    numel = 2**30
    assert _padded(numel) * 2 == INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert_call(
        harness.kernels.fp4_unpack_int8_kernel,
        _grid(numel),
        (raw, out, numel, BLOCK, True),
    )


# --------------------------------------------------------------------------- FP8


def test_fp8_unscaled_rank1_launch_contract(harness: Harness) -> None:
    raw, out = FakeTensor((100,)), object()
    result = harness.module.fp8_unpack_bf16(raw, None, out, (32, 32))
    assert result is out
    assert_call(
        harness.kernels.fp8_unpack_bf16_kernel,
        _grid(100),
        (raw, raw, out, 100, 1, 100, 1, 4, 32, 32, False, BLOCK, False),
    )


def test_fp8_scaled_batched_ragged_launch_contract(harness: Harness) -> None:
    raw = FakeTensor((2, 33, 65))
    scale, out = FakeTensor((2, 2, 3)), object()
    result = harness.module.fp8_unpack_bf16(raw, scale, out, (32, 32))
    assert result is out
    assert_call(
        harness.kernels.fp8_unpack_bf16_kernel,
        _grid(2 * 33 * 65),
        (raw, scale, out, 2 * 33 * 65, 33, 65, 2, 3, 32, 32, True, BLOCK, False),
    )


def test_fp8_scaled_row_block_uses_block_shape_order(harness: Harness) -> None:
    raw = FakeTensor((17, 40))
    scale, out = FakeTensor((17, 2)), object()
    assert harness.module.fp8_unpack_bf16(raw, scale, out, (1, 32)) is out
    assert_call(
        harness.kernels.fp8_unpack_bf16_kernel,
        _grid(17 * 40),
        (raw, scale, out, 17 * 40, 17, 40, 17, 2, 1, 32, True, BLOCK, False),
    )


def test_fp8_empty_input_enters_checks_but_launches_nothing(harness: Harness) -> None:
    raw, out = FakeTensor((0,)), object()
    assert harness.module.fp8_unpack_bf16(raw, None, out, (32, 32)) is out
    assert_no_launch(harness.kernels.fp8_unpack_bf16_kernel)


def test_fp8_padded_tail_selects_wide_index_before_logical_limit(harness: Harness) -> None:
    numel = INDEX_LIMIT - 512
    assert numel < INDEX_LIMIT
    assert _padded(numel) == INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp8_unpack_bf16(raw, None, out, (32, 32)) is out
    assert_call(
        harness.kernels.fp8_unpack_bf16_kernel,
        _grid(numel),
        (raw, raw, out, numel, 1, numel, 1, _ceil_div(numel, 32), 32, 32, False, BLOCK, True),
    )


def test_fp8_padded_block_below_limit_keeps_narrow_index(harness: Harness) -> None:
    numel = INDEX_LIMIT - 1024
    assert _padded(numel) < INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp8_unpack_bf16(raw, None, out, (32, 32)) is out
    assert_call(
        harness.kernels.fp8_unpack_bf16_kernel,
        _grid(numel),
        (raw, raw, out, numel, 1, numel, 1, _ceil_div(numel, 32), 32, 32, False, BLOCK, False),
    )


# ---------------------------------------------------------------- E8M0 FP4 scale


def test_fp4_scale_scalar_launch_contract(harness: Harness) -> None:
    raw, out = FakeTensor(()), object()
    assert harness.module.fp4_int8_scale(raw, out) is out
    assert_call(
        harness.kernels.fp4_int8_scale_kernel,
        (1,),
        (raw, out, 1, BLOCK, False),
    )


def test_fp4_scale_empty_input_enters_checks_but_launches_nothing(harness: Harness) -> None:
    raw, out = FakeTensor((0,)), object()
    assert harness.module.fp4_int8_scale(raw, out) is out
    assert_no_launch(harness.kernels.fp4_int8_scale_kernel)


def test_fp4_scale_padded_tail_selects_wide_index(harness: Harness) -> None:
    numel = INDEX_LIMIT - 512
    assert numel < INDEX_LIMIT
    assert _padded(numel) == INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_int8_scale(raw, out) is out
    assert_call(
        harness.kernels.fp4_int8_scale_kernel,
        _grid(numel),
        (raw, out, numel, BLOCK, True),
    )


def test_fp4_scale_padded_block_below_limit_keeps_narrow_index(harness: Harness) -> None:
    numel = INDEX_LIMIT - 1024
    assert _padded(numel) < INDEX_LIMIT
    raw, out = FakeTensor((numel,)), object()
    assert harness.module.fp4_int8_scale(raw, out) is out
    assert_call(
        harness.kernels.fp4_int8_scale_kernel,
        _grid(numel),
        (raw, out, numel, BLOCK, False),
    )


# ------------------------------------------------------------- device / backend


def test_device_context_is_the_input_device_and_is_restored(harness: Harness) -> None:
    raw, out = FakeTensor((8,)), object()
    assert harness.module.fp4_unpack_int8(raw, out) is out
    assert harness.recorder.events == [("enter", raw.device), ("exit", raw.device)]
    assert harness.recorder.stack == []


def test_device_context_is_restored_when_the_kernel_raises(harness: Harness) -> None:
    harness.kernels.fp4_unpack_int8_kernel.error = RuntimeError("compile failure")
    raw, out = FakeTensor((8,)), object()
    with pytest.raises(RuntimeError, match="compile failure"):
        harness.module.fp4_unpack_int8(raw, out)
    assert harness.recorder.stack == []
    assert len(harness.kernels.fp4_unpack_int8_kernel.calls) == 1


def test_non_npu_input_is_rejected_before_any_launch(harness: Harness) -> None:
    raw, out = FakeTensor((8,), device="cpu"), object()
    with pytest.raises(ValueError, match="NPU tensors only"):
        harness.module.fp4_unpack_int8(raw, out)
    assert harness.recorder.events == []
    assert_no_launch(harness.kernels.fp4_unpack_int8_kernel)


def test_missing_triton_ascend_backend_is_rejected(harness: Harness, monkeypatch) -> None:
    monkeypatch.setattr(harness.module, "supports_triton_ascend", lambda: False)
    raw, out = FakeTensor((8,)), object()
    with pytest.raises(RuntimeError, match="Triton Ascend backend is required"):
        harness.module.fp4_unpack_int8(raw, out)
    assert harness.recorder.events == []
    assert_no_launch(harness.kernels.fp4_unpack_int8_kernel)


def test_non_ascend_active_target_is_rejected(harness: Harness) -> None:
    harness.target.backend = "cuda"
    raw, out = FakeTensor((8,)), object()
    with pytest.raises(RuntimeError, match="not Ascend"):
        harness.module.fp4_unpack_int8(raw, out)
    assert harness.recorder.stack == []
    assert_no_launch(harness.kernels.fp4_unpack_int8_kernel)


def test_kernel_failure_propagates_without_a_torch_fallback(harness: Harness) -> None:
    # The launcher must not catch a compiler/runtime failure and retry another
    # implementation; the exception is the contract.
    harness.kernels.fp4_int8_scale_kernel.error = RuntimeError("device kernel failed")
    raw, out = FakeTensor((4,)), object()
    with pytest.raises(RuntimeError, match="device kernel failed"):
        harness.module.fp4_int8_scale(raw, out)
    assert len(harness.kernels.fp4_int8_scale_kernel.calls) == 1