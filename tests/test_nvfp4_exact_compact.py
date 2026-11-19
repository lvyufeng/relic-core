"""The compact NVFP4 launch must be bit-identical to the exact path it compacts.

`nvfp4_moe_rows_forward_exact_fast_compact` is not a new kernel: it calls the same body, the same
device functions and the same grid as `nvfp4_moe_rows_forward_exact_fast`, and only reduces the
thread count a block is launched with from eight warps to the four warps a four-column block can use.
That is why the acceptance criterion is exact equality rather than a tolerance, and why this file
compares against **two** independent references:

* `nvfp4_moe_rows_forward`, the original table-decoder implementation, which is the specification;
* `nvfp4_moe_rows_forward_exact_fast`, the shipped non-compact twin, which isolates the launch change
  from the decoder change.

The CPU half of this file proves the structural claim the exactness argument rests on -- that the
two launch configurations cover the same output columns, once each -- without a device. A block that
covered fewer columns would be a silent wrong answer, not a rounding difference, so if the source
contract breaks this must fail before anyone reads a GPU number.
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest
import torch

from relic_core.kernels.cuda_loader import load_cuda_kernel
from test_nvfp4_exact_fast import (  # noqa: E402 - same suite's reference builders
    REAL_SHAPE,
    bits_of,
    build_case,
    call,
)

COMPACT = "nvfp4_moe_rows_forward_exact_fast_compact"
NONCOMPACT = "nvfp4_moe_rows_forward_exact_fast"
ORACLE = "nvfp4_moe_rows_forward"

SOURCE = Path(__file__).resolve().parents[1] / "relic_core/csrc/qwen4_exp_nvfp4_exact.cu"
BINDING_SOURCE = Path(__file__).resolve().parents[1] / "relic_core/csrc/cuda_kernel.cpp"
SETUP_SOURCE = Path(__file__).resolve().parents[1] / "setup.py"


# ---------------------------------------------------------------------------------------------
# CPU: the launch-geometry claim.
# ---------------------------------------------------------------------------------------------


def columns_covered(total: int, columns_per_block: int, threads: int) -> list[int]:
    """Mirror the kernels' block/warp column arithmetic exactly, in Python.

    `grid` comes from the block-level column budget, a block's width is `ceil(total / grid)`, and the
    kernel then walks `col = col_lo + warp; col < col_hi; col += warps`. Reproducing that spelling is
    the point: an "equivalent" rewrite here would stop being evidence about the kernel.
    """
    assert threads % 32 == 0
    grid = -(-total // columns_per_block)
    cols_per_block = -(-total // grid)
    warps = threads // 32
    covered: list[int] = []
    for block in range(grid):
        lo = block * cols_per_block
        hi = min(lo + cols_per_block, total)
        if lo >= total:
            continue
        for warp in range(warps):
            col = lo + warp
            while col < hi:
                covered.append(col)
                col += warps
    return covered


@pytest.mark.parametrize("total", [
    16, 32, 48, 64, 80, 96, 272, 320, 512, 640, 1024, 1280, 2560, 4096, 10240, 31040,
])
def test_four_warp_and_eight_warp_launches_cover_the_same_columns_once(total):
    compact = columns_covered(total, 4, 128)
    oracle = columns_covered(total, 4, 256)
    assert sorted(compact) == list(range(total)), f"{total}: compact left a column uncomputed"
    assert sorted(oracle) == list(range(total)), f"{total}: oracle left a column uncomputed"
    # Equal as multisets *and* as the per-warp partition: the extra four warps of the oracle must
    # contribute nothing, and no column may be produced by two warps.
    assert compact == oracle, f"{total}: the two launches covered different column orders"


@pytest.mark.parametrize("total", [16, 32, 48, 64, 80, 96, 272, 512, 640, 2560, 4096, 31040])
def test_a_blocks_column_range_never_exceeds_the_four_warp_budget(total):
    grid = -(-total // 4)
    cols_per_block = -(-total // grid)
    assert 1 <= cols_per_block <= 4, (
        f"{total}: a block owns {cols_per_block} columns, which four warps cannot cover"
    )


def test_more_warps_than_columns_leave_the_extra_warps_idle():
    """The oracle's four surplus warps must be provably idle; that is the whole compact saving."""
    for total in (640, 2560):
        grid = -(-total // 4)
        cols_per_block = -(-total // grid)
        warps = 256 // 32
        active: set[int] = set()
        for block in range(grid):
            lo = block * cols_per_block
            hi = min(lo + cols_per_block, total)
            for warp in range(warps):
                col = lo + warp
                if col < hi:
                    active.add(warp)
        assert active == {0, 1, 2, 3}, f"{total}: unexpected active warps {sorted(active)}"


# ---------------------------------------------------------------------------------------------
# CPU: source contracts, so the launch configuration cannot drift away from the argument above.
# ---------------------------------------------------------------------------------------------


def _kernel_source() -> str:
    return SOURCE.read_text()


def test_source_pins_one_shared_body_and_both_launch_constants():
    source = _kernel_source()
    assert "constexpr int kColumnsPerBlock = 4;" in source, "the grid's column budget is a contract"
    assert "constexpr int kOracleThreads = 256;" in source, "the oracle's launch must not move"
    assert "constexpr int kCompactThreads = 128;" in source, "the compact launch is four warps"
    # One body, so there is no second copy of the arithmetic to keep in step.
    assert source.count("torch::Tensor run_rows_exact(") == 1
    assert source.count("nvfp4_exact_gate_up_kernel<<<") == 1
    assert source.count("nvfp4_exact_down_kernel<<<") == 1
    # Both launches must read the parameter, not a hard-coded thread count.
    assert "threads_per_block, gate_shmem" in source
    assert "threads_per_block, 0," in source


def test_source_guards_a_launch_too_small_to_cover_a_blocks_columns():
    source = _kernel_source()
    assert "threads_per_block % 32 == 0 && threads_per_block >= warps_per_block * 32" in source


def test_both_public_entries_differ_only_in_the_launch_constant():
    source = _kernel_source()
    pattern = re.compile(
        r"torch::Tensor (nvfp4_moe_rows_forward_exact_fast(?:_compact)?)\((.*?)\) \{",
        re.DOTALL,
    )
    found = {match.group(1): match.group(2) for match in pattern.finditer(source)}
    assert set(found) == {NONCOMPACT, COMPACT}, f"unexpected entry set {sorted(found)}"
    assert found[COMPACT] == found[NONCOMPACT], (
        "the compact entry must declare the same signature as its oracle"
    )
    for name, constant in ((NONCOMPACT, "kOracleThreads"), (COMPACT, "kCompactThreads")):
        body = source.split(f"torch::Tensor {name}(", 1)[1]
        body = body.split("\n}\n", 1)[0]
        assert f"activation_fp4, {constant});" in body, f"{name} does not use {constant}"
        assert "run_rows_exact(" in body


def test_bindings_are_additive_and_registered():
    binding = BINDING_SOURCE.read_text()
    assert f"torch::Tensor {COMPACT}(" in binding, "the compact entry needs a declaration"
    assert f'"{COMPACT}"' in binding, "the compact entry needs a pybind registration"
    assert f"&{COMPACT}," in binding
    for unchanged in (NONCOMPACT, ORACLE):
        assert f'"{unchanged}"' in binding, f"{unchanged} must stay bound as the oracle"
    assert '"relic_core/csrc/qwen4_exp_nvfp4_exact.cu"' in SETUP_SOURCE.read_text()
    assert '"7.5;8.6;8.9"' in SETUP_SOURCE.read_text(), "the default arch list is a contract"


# ---------------------------------------------------------------------------------------------
# GPU: bit identity against both references.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extension():
    return _extension()


def _extension():
    """The built extension, or a skip. A plain helper so a test can call it through a fixture."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    ext = load_cuda_kernel()
    if ext is None:
        pytest.skip("the CUDA extension is not built for this interpreter")
    for name in (COMPACT, NONCOMPACT, ORACLE):
        if not hasattr(ext, name):
            pytest.skip(f"the built extension has no {name}")
    return ext


def assert_bit_identical(actual, expected, label):
    """All float32 words, including NaN payloads and signed zero, are part of this oracle."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # torch.equal on float values rejects identical NaNs; compare their raw words instead.
    assert torch.equal(bits_of(actual), bits_of(expected)), f"{label}: bits differ"


def _assert_all_three_agree(ext, case, label):
    compact = call(ext, COMPACT, case)
    assert_bit_identical(compact, call(ext, NONCOMPACT, case), f"{label} vs exact twin")
    assert_bit_identical(compact, call(ext, ORACLE, case), f"{label} vs table-decoder oracle")
    assert torch.equal(bits_of(compact), bits_of(call(ext, NONCOMPACT, case)))
    return compact


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 8, 512])
def test_real_per_rank_shape_is_bit_identical(rows):
    """2560/640/64 local experts/top 10 is one rank of the 8-way TP model, including a prefill row."""
    ext = _extension()
    case = build_case(rows, seed=11000 + rows, routes="mixed", **REAL_SHAPE)
    _assert_all_three_agree(ext, case, f"rows={rows}")


@pytest.mark.parametrize("shape", [
    dict(hidden=64, inter=32, experts=8, topk=3),      # smallest shape the guards accept
    dict(hidden=128, inter=16, experts=4, topk=1),     # a single route
    dict(hidden=272, inter=48, experts=5, topk=7),     # neither dimension divides the 4-column budget
    dict(hidden=4096, inter=256, experts=16, topk=8),
    dict(hidden=4096, inter=640, experts=32, topk=10),  # many column blocks, odd width
])
@pytest.mark.parametrize("rows", [1, 3, 4])
def test_other_shapes_are_bit_identical(shape, rows):
    ext = _extension()
    case = build_case(rows, seed=12000 + rows + shape["hidden"], routes="mixed", **shape)
    _assert_all_three_agree(ext, case, f"{shape} rows={rows}")


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("routes", ["valid", "duplicate", "out_of_range", "all_unowned", "mixed"])
def test_route_pathologies_are_bit_identical(rows, routes):
    ext = _extension()
    case = build_case(rows, seed=13000 + rows, routes=routes, **REAL_SHAPE)
    _assert_all_three_agree(ext, case, f"rows={rows} routes={routes}")


@pytest.mark.parametrize("rows", [1, 4])
def test_global_expert_offset_is_bit_identical(rows):
    ext = _extension()
    case = build_case(rows, seed=14000 + rows, start_idx=1000, routes="mixed", **REAL_SHAPE)
    _assert_all_three_agree(ext, case, f"rows={rows} global offset")


@pytest.mark.parametrize("scale_bytes", [
    [0x00],                      # zero scale: a whole expert contributes nothing
    [0x7F, 0xFF],                # the byte the OCP format reserves as NaN but this path reads as 480
    [0x01, 0x80],                # smallest subnormal magnitude, negative zero
    [0x7E, 0xFE],                # largest finite magnitude, both signs
])
def test_adversarial_scale_bytes_are_bit_identical(scale_bytes):
    ext = _extension()
    case = build_case(4, seed=15000, routes="duplicate", scale_bytes=scale_bytes, **REAL_SHAPE)
    _assert_all_three_agree(ext, case, f"scale_bytes={scale_bytes}")


def test_every_e4m3_byte_agrees_through_the_compact_entry():
    """All 256 scale encodings, cycled over every scale tensor, exercised through this launch."""
    ext = _extension()
    case = build_case(2, seed=16000, routes="mixed", scale_bytes=list(range(256)), **REAL_SHAPE)
    _assert_all_three_agree(ext, case, "all 256 scale bytes")


@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0, 1.5])
def test_swiglu_clamping_is_bit_identical(swiglu_limit):
    ext = _extension()
    case = build_case(4, seed=17000, swiglu_limit=swiglu_limit, routes="mixed", **REAL_SHAPE)
    _assert_all_three_agree(ext, case, f"swiglu_limit={swiglu_limit}")


def test_extreme_activations_are_bit_identical():
    """Signed zeros, subnormals, infinities, NaN payloads and cancelling magnitudes in `x`."""
    ext = _extension()
    case = build_case(4, seed=18000, routes="valid", **REAL_SHAPE)
    values = torch.tensor([
        0.0, -0.0, 1.0, -1.0,
        float("inf"), float("-inf"), float("nan"),
        1e-45, -1e-45,        # fp32 subnormals
        3.4e38, -3.4e38,      # fp32 max: the products overflow
        1e-30, -1e-30,
        12345.678, -12345.678,
    ], dtype=torch.float32, device="cuda")
    case["x"] = values.repeat((case["x"].numel() + values.numel() - 1) // values.numel())[
        : case["x"].numel()
    ].reshape_as(case["x"])
    # A cancelling row: every lane is nonzero but the accumulation is near zero, which is the shape
    # that exposes a different reduction rather than a different format decoder.
    case["x"][0] = torch.linspace(-1.0, 1.0, case["x"].shape[1], device="cuda")
    _assert_all_three_agree(ext, case, "extreme and cancelling activations")


def test_zero_and_unowned_routes_leave_an_all_zero_row():
    """A row no local expert owns must still be the zero initialization, not stale memory."""
    ext = _extension()
    case = build_case(3, seed=19000, routes="all_unowned", **REAL_SHAPE)
    compact = _assert_all_three_agree(ext, case, "all unowned")
    assert torch.equal(compact, torch.zeros_like(compact)), "unowned routes must contribute nothing"


def test_captured_replay_agrees_after_input_mutations():
    """Repeated capture over a mutated input is where a fixed launch address can go wrong."""
    ext = _extension()
    case = build_case(4, seed=20000, routes="mixed", **REAL_SHAPE)
    buffers = {name: case[name].clone() for name in case if torch.is_tensor(case[name])}

    def run(name):
        return getattr(ext, name)(
            *[buffers[key] for key in (
                "x", "indices", "weights", "w1q", "w1s", "w3q", "w3s", "w2q", "w2s",
            )],
            case["experts_start_idx"], case["swiglu_limit"], 1.0, 1.0,
            buffers["w_scale1"], buffers["w_scale2"], True,
        )

    torch.cuda.synchronize()
    graphs = {}
    for name in (COMPACT, NONCOMPACT):
        run(name)  # warm allocation outside the recording
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            recorded = run(name)
        graphs[name] = (graph, recorded)
    generator = torch.Generator(device="cuda").manual_seed(20001)
    for mutation in range(3):
        buffers["x"].normal_(generator=generator)
        buffers["weights"].uniform_(generator=generator)
        buffers["indices"].random_(
            case["experts_start_idx"], case["experts_start_idx"] + REAL_SHAPE["experts"],
            generator=generator,
        )
        if mutation == 1:
            buffers["indices"][0, 0] = -5   # a foreign route mid-capture
        torch.cuda.synchronize()
        outs = {}
        for name, (graph, recorded) in graphs.items():
            graph.replay()
            outs[name] = recorded.clone()
        torch.cuda.synchronize()
        assert_bit_identical(outs[COMPACT], outs[NONCOMPACT], f"captured mutation {mutation}")
        assert_bit_identical(outs[COMPACT], run(ORACLE), f"captured mutation {mutation} vs oracle")


def test_the_launch_constants_a_block_needs_are_whole_warps():
    """Both constants must satisfy the kernel guard, and the compact one must be the four warps."""
    source = _kernel_source()
    constants = dict(re.findall(r"constexpr int (k\w+Threads) = (\d+);", source))
    assert constants == {"kOracleThreads": "256", "kCompactThreads": "128"}, constants
    oracle, compact = int(constants["kOracleThreads"]), int(constants["kCompactThreads"])
    columns_per_block = int(re.search(r"constexpr int kColumnsPerBlock = (\d+);", source).group(1))
    for threads in (oracle, compact):
        assert threads % 32 == 0, f"{threads} is not a whole number of warps"
        assert threads >= columns_per_block * 32, f"{threads} cannot fill a block's columns"
    assert compact == columns_per_block * 32, "the compact launch is exactly the columns' warps"
    # The guard in the host body is what refuses a future bad constant instead of miscomputing.
    assert "threads_per_block >= warps_per_block * 32" in source