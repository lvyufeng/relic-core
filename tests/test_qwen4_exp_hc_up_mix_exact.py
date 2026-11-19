"""Bit and timing checks for the independent HC4 up-mix candidate.

`qwen4_exp_hc_up_mix_exact` is a *candidate*, not a replacement: the qualified original stays bound
as `qwen4_exp_hc_up_mix` and is the only accepted oracle. The candidate folds the five mixer
operations the original folds, and additionally shares one activation load across the four stream
dot products. The claim it has to earn is raw-bit equality, so the comparison here is over the
BF16 *words* (``view(torch.int16)``), never over a tolerance: `--use_fast_math` is on for both, the
lane-strided FMA sequence, the shuffle tree, the sigmoid expression and the fold order are the same,
so any difference is a defect rather than rounding.

The reference always comes from the normally loaded (qualified) extension. The candidate comes from
that same module when the build contains it, or -- with ``Q4_HC_EXACT_SO`` -- from a separately built
shared object loaded under a private alias, which is the same isolation the runtime overlay uses so
that a comparison never silently substitutes the whole operator library.

Run:   python -m pytest tests/test_qwen4_exp_hc_up_mix_exact.py -q
Bench: python tests/test_qwen4_exp_hc_up_mix_exact.py --bench
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE = "qwen4_exp_hc_up_mix"
CANDIDATE = "qwen4_exp_hc_up_mix_exact"
SOURCE = Path(
    os.environ.get("Q4_HC_EXACT_SOURCE", ROOT / "relic_core" / "csrc" / "qwen4_exp_hc_mix_exact.cu")
)

HC = 4
HIDDEN = 2560
INTER = 320
ROWS = (1, 4, 512)

#: Private namespace for the aliased candidate module. CPython requires the entry point to be
#: ``PyInit_<leaf>``, so the alias always ends in the extension's own leaf name.
_ALIAS_NS = "_qwen4_hc_exact_probe_"
_MODULES: dict = {}


def _native():
    """The normally loaded extension: reference oracle, and candidate in a build that has it."""
    module = _MODULES.get("native")
    if module is None:
        from relic_core.kernels.cuda_loader import load_cuda_kernel

        module = load_cuda_kernel()
        if module is None:
            pytest.skip("the native extension is not built for this interpreter")
        _MODULES["native"] = module
    return module


def _alias(path: Path):
    """Load ``path`` as an extension module under the private namespace, once per path."""
    key = str(path)
    module = _MODULES.get(key)
    if module is not None:
        return module
    package = sys.modules.get(_ALIAS_NS)
    if package is None:
        package = types.ModuleType(_ALIAS_NS)
        package.__path__ = []
        sys.modules[_ALIAS_NS] = package
    full = f"{_ALIAS_NS}.cuda_kernel"
    spec = importlib.util.spec_from_file_location(full, key)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{key} is not a loadable CPython extension")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    _MODULES[key] = module
    return module


def _candidate_module():
    """Where the candidate lives: a separate build when asked, else the loaded module."""
    path = os.environ.get("Q4_HC_EXACT_SO")
    if not path:
        return _native()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        pytest.skip(f"Q4_HC_EXACT_SO does not name a file: {resolved}")
    return _alias(resolved)


def _operators():
    reference = getattr(_native(), REFERENCE, None)
    if reference is None:
        pytest.skip(f"the loaded extension has no {REFERENCE}")
    candidate = getattr(_candidate_module(), CANDIDATE, None)
    if candidate is None:
        pytest.skip(f"no {CANDIDATE} in the selected module; set Q4_HC_EXACT_SO")
    return reference, candidate


def requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("the HC up-mix comparison needs a CUDA device")


def _inputs(rows: int, seed: int = 0, device: str = "cuda", dtype=torch.bfloat16):
    """Real decode geometry with small weights, so every dot stays finite."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    down = (torch.randn(rows, INTER, generator=generator) * 0.1).to(dtype).to(device)
    mix_up = (torch.randn(HC * HIDDEN, INTER, generator=generator) * 0.05).to(dtype).to(device)
    normed = (torch.randn(rows, HC * HIDDEN, generator=generator) * 0.5).to(dtype).to(device)
    return down, mix_up, normed


def _words(tensor: torch.Tensor) -> torch.Tensor:
    """The BF16 words as int16, so a NaN still compares by bit pattern."""
    return tensor.view(torch.int16)


def _call(operator, down, mix_up, normed, threads=None):
    if threads is None:
        return operator(down, mix_up, normed, HC, HIDDEN)
    return operator(down, mix_up, normed, HC, HIDDEN, threads)


def test_candidate_reproduces_qualified_bits():
    requires_cuda()
    reference, candidate = _operators()
    for rows in ROWS:
        down, mix_up, normed = _inputs(rows, seed=rows)
        expected = _call(reference, down, mix_up, normed)
        actual = _call(candidate, down, mix_up, normed)
        assert torch.equal(_words(actual), _words(expected)), f"rows={rows} differs word-wise"
        assert torch.equal(_words(actual).cpu(), _words(expected).cpu())


def test_thread_choices_are_bit_identical():
    requires_cuda()
    reference, candidate = _operators()
    down, mix_up, normed = _inputs(4, seed=11)
    expected = _words(_call(reference, down, mix_up, normed)).cpu()
    for threads in (64, 128, 256, 512):
        actual = _words(_call(candidate, down, mix_up, normed, threads)).cpu()
        assert torch.equal(actual, expected), f"threads={threads} differs from the oracle"


def test_mutating_one_weight_word_changes_both_the_same_way():
    """A stale or partially-read weight would show up as one side not moving."""
    requires_cuda()
    reference, candidate = _operators()
    down, mix_up, normed = _inputs(4, seed=23)
    before_reference = _words(_call(reference, down, mix_up, normed)).cpu()
    before_candidate = _words(_call(candidate, down, mix_up, normed)).cpu()
    assert torch.equal(before_candidate, before_reference)

    mutated = mix_up.clone()
    mutated[HIDDEN + 7, 13] = torch.tensor(1.5, dtype=torch.bfloat16, device=mutated.device)
    after_reference = _words(_call(reference, down, mutated, normed)).cpu()
    after_candidate = _words(_call(candidate, down, mutated, normed)).cpu()
    assert torch.equal(after_candidate, after_reference)
    assert not torch.equal(after_reference, before_reference), "the mutation must be observable"


def test_candidate_is_recordable_and_replays_mutated_inputs():
    """The runtime calls this inside CUDA graphs: capture once, then replay over new values."""
    requires_cuda()
    reference, candidate = _operators()
    down_a, mix_up, normed_a = _inputs(4, seed=31)
    down_b, _, normed_b = _inputs(4, seed=37)

    down = down_a.clone()
    normed = normed_a.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=down.device)
    stream.wait_stream(torch.cuda.current_stream(down.device))
    with torch.cuda.stream(stream):
        _call(candidate, down, mix_up, normed)
    torch.cuda.current_stream(down.device).wait_stream(stream)
    torch.cuda.synchronize(down.device)
    with torch.cuda.graph(graph):
        captured = _call(candidate, down, mix_up, normed)
    torch.cuda.synchronize(down.device)
    graph.replay()
    torch.cuda.synchronize(down.device)
    assert torch.equal(_words(captured).cpu(), _words(_call(reference, down_a, mix_up, normed_a)).cpu())

    down.copy_(down_b)
    normed.copy_(normed_b)
    graph.replay()
    torch.cuda.synchronize(down.device)
    assert torch.equal(_words(captured).cpu(), _words(_call(reference, down_b, mix_up, normed_b)).cpu())


@pytest.mark.parametrize(
    "case",
    ["rows_zero", "wrong_width", "wrong_hc", "wrong_hidden", "wrong_dtype", "non_contiguous", "bad_threads", "cpu"],
)
def test_refusals_happen_before_launch(case):
    requires_cuda()
    _, candidate = _operators()
    down, mix_up, normed = _inputs(4, seed=41)
    kwargs = dict(hc=HC, hidden=HIDDEN, threads=128)
    operands = dict(down=down, mix_up=mix_up, normed=normed)
    if case == "rows_zero":
        operands["down"] = down[:0].contiguous()
        operands["normed"] = normed[:0].contiguous()
    elif case == "wrong_width":
        operands["down"] = torch.zeros(4, 256, dtype=torch.bfloat16, device="cuda")
    elif case == "wrong_hc":
        kwargs["hc"] = 3
    elif case == "wrong_hidden":
        kwargs["hidden"] = 2048
    elif case == "wrong_dtype":
        operands["down"] = down.float()
    elif case == "non_contiguous":
        # A genuinely non-contiguous view over real storage: stride 2 along the reduction axis.
        wide = torch.zeros(4, INTER * 2, dtype=torch.bfloat16, device="cuda")
        operands["down"] = wide[:, ::2]
    elif case == "bad_threads":
        kwargs["threads"] = 96
    elif case == "cpu":
        operands = {name: value.cpu() for name, value in operands.items()}
    with pytest.raises((RuntimeError, ValueError)):
        candidate(
            operands["down"], operands["mix_up"], operands["normed"],
            kwargs["hc"], kwargs["hidden"], kwargs["threads"],
        )


def test_source_keeps_the_qualified_arithmetic_and_scope():
    """The properties that make bit equality the *expected* outcome, checked on the source."""
    text = SOURCE.read_text()
    assert "const float value = __bfloat162float(d[k]);" in text, "one activation load per k"
    for tile, chain in (("w0", "dot0"), ("w1", "dot1"), ("w2", "dot2"), ("w3", "dot3")):
        assert f"{chain} += value * __bfloat162float({tile}[k]);" in text
    assert "offset = 16" in text and "offset > 0" in text and "offset >>= 1" in text
    assert "__shfl_down_sync" in text
    assert "1.f / (1.f + expf(-" in text
    assert "folded += gate" in text
    assert "hc == 4 && hidden == 2560 && down.size(1) == 320" in text
    assert "__nv_bfloat16" in text
    for forbidden in ("__half", "wgmma", "mma.sync", "cublas", "torch::matmul"):
        assert forbidden not in text, f"candidate must not introduce {forbidden}"


def _module_identity(module) -> dict:
    """Path and content hash of a loaded extension, so a report names the binary it measured."""
    path = getattr(module, "__file__", None)
    if not path:
        return {"path": None, "sha256": None}
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def bench_host_inclusive(iterations: int = 200, samples: int = 7):
    """Per-call latency including the Python/pybind dispatch.

    Kept as the counter-example to the device measurement: at this geometry the host floor is about
    the size of the kernel itself, so a ratio taken here is a binding measurement, not a kernel one.
    """
    requires_cuda()
    reference, candidate = _operators()
    device = torch.device("cuda")
    report = {
        "kind": "qwen4_hc_up_mix_exact_bench",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "reference_module": _module_identity(_native()),
        "candidate_module": _module_identity(_candidate_module()),
        "iteration_shape": {"rows": 4, "hidden": HIDDEN, "inter": INTER, "hc": HC},
        "weight_bytes_per_launch": HC * HIDDEN * INTER * 2,
        "iterations": iterations,
        "samples": samples,
        "arms": [],
    }
    for rows in (4, 1, 512):
        down, mix_up, normed = _inputs(rows, seed=rows)
        arm = {"rows": rows, "timings_ms": {}}
        for name, operator, threads in (
            ("qualified", reference, None),
            ("candidate_128", candidate, 128),
            ("candidate_256", candidate, 256),
        ):
            for _ in range(20):
                _call(operator, down, mix_up, normed, threads)
            torch.cuda.synchronize(device)
            times = []
            for _ in range(samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    _call(operator, down, mix_up, normed, threads)
                end.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end) / iterations)
            arm["timings_ms"][name] = {"median": _median(times), "samples": times}
        qualified = arm["timings_ms"]["qualified"]["median"]
        for name, timing in arm["timings_ms"].items():
            timing["ratio_to_qualified"] = timing["median"] / qualified
            timing["effective_gib_per_s"] = (HC * HIDDEN * INTER * 2) / timing["median"] / 1e6
        report["arms"].append(arm)
    print(json.dumps(report, indent=2))
    return report


def _captured_batch(operator, down, mix_up, normed, threads, iterations):
    """One recording of ``iterations`` launches -- the form the runtime actually replays."""
    for _ in range(10):
        _call(operator, down, mix_up, normed, threads)
    torch.cuda.synchronize(down.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            _call(operator, down, mix_up, normed, threads)
    torch.cuda.synchronize(down.device)
    return graph


def bench_device(samples: int = 9):
    """Device time per launch with the host out of the loop, arms alternated for equal clocks."""
    requires_cuda()
    reference, candidate = _operators()
    device = torch.device("cuda")
    report = {
        "kind": "qwen4_hc_up_mix_exact_device_bench",
        "scope": "graph-captured device time per launch; host dispatch excluded; arms alternated",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "reference_module": _module_identity(_native()),
        "candidate_module": _module_identity(_candidate_module()),
        "weight_bytes_per_launch": HC * HIDDEN * INTER * 2,
        "samples": samples,
        "arms": [],
    }
    for rows, iterations in ((4, 200), (1, 200), (512, 20)):
        down, mix_up, normed = _inputs(rows, seed=rows)
        selected = {
            "qualified": (reference, None),
            "candidate_128": (candidate, 128),
            "candidate_256": (candidate, 256),
        }
        graphs = {
            name: _captured_batch(operator, down, mix_up, normed, threads, iterations)
            for name, (operator, threads) in selected.items()
        }
        timings = {name: [] for name in selected}
        for _ in range(samples):
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize(device)
            for name, graph in graphs.items():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize(device)
                timings[name].append(start.elapsed_time(end) / iterations)
        best = _median(timings["qualified"])
        arm = {"rows": rows, "iterations_per_replay": iterations, "timings_ms": {}}
        for name, values in timings.items():
            median = _median(values)
            arm["timings_ms"][name] = {
                "median": median,
                "samples": values,
                "ratio_to_qualified": median / best,
                "effective_gib_per_s": (HC * HIDDEN * INTER * 2) / median / 1e6,
            }
        report["arms"].append(arm)
        del graphs
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_device()
    elif "--bench-host" in sys.argv:
        bench_host_inclusive()
    else:
        raise SystemExit("run with --bench (device) or --bench-host, or use pytest")