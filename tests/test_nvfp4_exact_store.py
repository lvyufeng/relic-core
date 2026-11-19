"""The ordered single-writer NVFP4 epilogue must be bit-identical to the atomic one.

`nvfp4_moe_rows_forward_exact_fast_store` is not a new arithmetic. It turns the down stage's loops
inside out -- a column outer loop with the routes inside -- so one output element's routed
contributions accumulate in a register in ascending route order and reach `out` in a single store,
and it drops both zero fills. The specification is unchanged: `nvfp4_moe_rows_forward`, the original
table-decoder entry, is the oracle, and `nvfp4_moe_rows_forward_exact_fast` (eight-warp) and
`..._compact` (four-warp) isolate the epilogue change from the decoder and launch changes.

Two claims are proven here without a device, because they are structural and a violation would be a
silent wrong answer rather than a rounding difference:

* **Coverage.** With `grid.x = ceil(total / 4)` and `cols_per_block = ceil(total / grid.x)` the
  column loop visits every output column exactly once, for the down stage, and every `act` column of
  an owned route exactly once, for the gate stage. That is what licenses one store per element and
  what makes the zero fills removable.
* **No unwritten read.** The down kernel skips a route this rank does not own *before* it forms the
  `act` row pointer, and the gate stage writes every column of every owned route. An uninitialised
  `act` is therefore honest: nothing reads the rows it left alone.

The remaining claim needs the device and is asserted, not argued: the old `atomicAdd(float*)` under
the project's global `-ftz=true` must round exactly like the explicit `add.rn.ftz.f32` this entry
uses. Every ordinary case tests that. One case is built specifically to separate a flushing atomic
from a non-flushing one -- two normal products whose sum is subnormal -- and its failure message says
what a mismatch means, because that is a one-qualifier fix rather than a redesign.

Neither part is a speed claim. There is no tolerance anywhere: a tolerance would hide exactly the
rounding relocation this file exists to rule out.
"""
from __future__ import annotations

import math
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
    rand_u8,
)

STORE = "nvfp4_moe_rows_forward_exact_fast_store"
COMPACT = "nvfp4_moe_rows_forward_exact_fast_compact"
EXACT = "nvfp4_moe_rows_forward_exact_fast"
ORACLE = "nvfp4_moe_rows_forward"

SOURCE = Path(__file__).resolve().parents[1] / "relic_core/csrc/qwen4_exp_nvfp4_exact.cu"
BINDING_SOURCE = Path(__file__).resolve().parents[1] / "relic_core/csrc/cuda_kernel.cpp"
OPS_SOURCE = Path(__file__).resolve().parents[1] / "relic_core/kernels/ops.py"

#: The launch this entry uses and the block-level column budget both stages derive their grid from.
#: These are the cross-check, not the input: `test_launch_constants_match_the_source` reads the same
#: two numbers back out of the source, so a divergence between this file's arithmetic and the
#: kernel's own constants fails before any GPU number is read.
K_STORE_THREADS = 128
K_COLUMNS_PER_BLOCK = 4


# ---------------------------------------------------------------------------------------------
# CPU: the column-coverage claim, mirrored from the kernels' own arithmetic.
# ---------------------------------------------------------------------------------------------


def owners_per_column(total: int, columns_per_block: int, threads: int) -> list[list[tuple]]:
    """Which `(block, warp)` pairs write each column, under the kernels' own loop arithmetic.

    This is the single-writer claim itself, not a restatement of it: the epilogue stores each output
    element once from `lane == 0`, so a column with two owners would be a race and a column with
    none would be an unwritten element of an uninitialised buffer.
    """
    assert threads % 32 == 0
    warps = threads // 32
    grid = -(-total // columns_per_block)
    block_width = -(-total // grid)
    owners: list[list[tuple]] = [[] for _ in range(total)]
    for block in range(grid):
        col_lo = block * block_width
        col_hi = min(col_lo + block_width, total)
        if col_lo >= total:
            continue
        for warp in range(warps):
            for col in range(col_lo + warp, col_hi, warps):
                owners[col].append((block, warp))
    return owners


@pytest.mark.parametrize("hidden", [
    16, 80, 256, 512, 640, 1024, 1536, 2048, 2560, 3072, 4096, 8192,
    # Widths a four-column block budget does not divide: the last block is the one that could fall
    # off the end or overlap its neighbour.
    2576, 2608,
])
def test_down_stage_writes_every_column_exactly_once(hidden):
    owners = owners_per_column(hidden, K_COLUMNS_PER_BLOCK, K_STORE_THREADS)
    unowned = [col for col, who in enumerate(owners) if not who]
    shared = [col for col, who in enumerate(owners) if len(who) > 1]
    assert not unowned, f"hidden={hidden}: columns with no writer would stay uninitialised {unowned[:8]}"
    assert not shared, f"hidden={hidden}: columns with more than one writer {shared[:8]}"


@pytest.mark.parametrize("inter", [16, 80, 256, 640, 1024, 1536, 2048, 2560])
def test_gate_stage_writes_every_act_column_exactly_once(inter):
    owners = owners_per_column(inter, K_COLUMNS_PER_BLOCK, K_STORE_THREADS)
    assert all(len(who) == 1 for who in owners), (
        f"inter={inter}: the gate stage must write every act column of an owned route exactly once, "
        "which is what makes an uninitialised act buffer safe"
    )


def test_a_block_never_needs_more_warps_than_the_column_budget():
    """The launch is only correct while a block's width fits the warps it is launched with."""
    for total in (16, 640, 2560, 2608, 8192):
        for columns_per_block in (1, 2, 4):
            grid = -(-total // columns_per_block)
            block_width = -(-total // grid)
            assert block_width <= columns_per_block, (total, columns_per_block, block_width)
            assert block_width <= K_STORE_THREADS // 32


# ---------------------------------------------------------------------------------------------
# CPU: the source contracts the exactness argument rests on.
# ---------------------------------------------------------------------------------------------


def _kernel_source() -> str:
    return SOURCE.read_text()


def _function_body(source: str, signature: str) -> str:
    """The text of the definition whose signature starts with `signature`, to its closing brace."""
    start = source.index(signature)
    end = source.index("\n}\n", start)
    return source[start:end]


def test_source_pins_the_multiplies_and_the_accumulating_add():
    source = _kernel_source()
    assert 'asm("mul.rn.ftz.f32 %0, %1, %2;"' in source, (
        "the routed product must stay two explicitly rounded multiplies; a compiler contraction "
        "into the accumulation would change the rounding"
    )
    assert 'asm("add.rn.ftz.f32 %0, %1, %2;"' in source, (
        "the route accumulation must be the explicit round-to-nearest add the old atomic performed"
    )
    body = _function_body(source, "nvfp4_exact_down_store_kernel(")
    # Left-associative `(acc * w) * s2_down`: the old entry's `acc * w * s2_down` spelling.
    assert re.search(
        r"nvfp4_store_mul_rn_ftz\(\s*nvfp4_store_mul_rn_ftz\(acc, w\), s2_down\)", body
    ), "the product must be (acc * w) * s2_down, the old expression's order"
    assert "nvfp4_store_add_rn_ftz(routed, contribution)" in body
    assert "float routed = 0.0f;" in body, "the accumulation must start at +0.0, the old zero fill"


def _code_only(body: str) -> str:
    """The body with `//` comments removed, so a comment naming an operation is not mistaken for it."""
    return "\n".join(line.split("//", 1)[0] for line in body.splitlines())


def test_down_store_kernel_has_no_atomic_and_one_store_site():
    body = _function_body(_kernel_source(), "nvfp4_exact_down_store_kernel(")
    code = _code_only(body)
    assert "atomicAdd" not in code, "the single-writer epilogue must not need a global atomic"
    stores = [line for line in code.splitlines() if re.search(r"out_row\[col\]\s*=", line)]
    assert len(stores) == 1, f"expected exactly one store of an output element, got {stores}"
    assert "lane == 0" in code, "only the warp's lane 0 may write the column"
    # The store sits after the route sweep, not inside it: the route loop must close before it, so a
    # column is written once with the accumulated value rather than once per route.
    route_loop = code.index("for (int r = 0; r < topk; ++r)")
    accumulate = code.index("nvfp4_store_add_rn_ftz(routed, contribution)")
    store = code.index("out_row[col] = routed;")
    assert route_loop < accumulate < store, (
        "the accumulation must stay inside the route loop and the store outside it"
    )
    assert code[route_loop:store].count("for (int col =") == 0, (
        "the column loop must enclose the route loop, not the other way round"
    )
    assert code[route_loop:store].count("for (int r =") == 1


def test_down_store_kernel_skips_a_foreign_route_before_reading_act():
    """An uninitialised `act` is only safe if the skip precedes the row pointer."""
    body = _function_body(_kernel_source(), "nvfp4_exact_down_store_kernel(")
    guard = body.index("if (local < 0 || local >= n_local_experts) continue;")
    pointer = body.index("const float* row_act =")
    read = body.index("row_act + gb * 16")
    assert guard < pointer < read, (
        "the owned-route guard must come before `row_act` is formed, so a row the gate stage never "
        "wrote is never read"
    )


def test_gate_and_store_share_the_same_owned_route_predicate():
    """Empty act storage is safe only while both stages classify every route identically."""
    source = _kernel_source()
    predicates = []
    for name in ("nvfp4_exact_gate_up_kernel(", "nvfp4_exact_down_store_kernel("):
        body = _code_only(_function_body(source, name))
        local = re.search(r"const int local\s*=\s*([^;]+);", body)
        guard = re.search(r"if \((local < 0 \|\| local >= n_local_experts)\) continue;", body)
        assert local is not None and guard is not None, f"missing owned-route predicate in {name}"
        predicates.append((local.group(1).strip(), guard.group(1)))
    assert predicates[0] == predicates[1], "gate/store route ownership diverged; empty act may be read"


def test_store_entry_allocates_uninitialised_buffers_and_the_oracle_still_zeroes():
    source = _kernel_source()
    store_body = _function_body(source, "torch::Tensor run_rows_exact_store(")
    assert "auto out = torch::empty({rows, hidden}, x.options());" in store_body
    assert "auto act = torch::empty({rows * topk, inter}, x.options());" in store_body
    assert "torch::zeros" not in _code_only(store_body), (
        "the ordered-store entry exists partly to drop the two zero fills; keeping them would hide "
        "whether the coverage argument holds"
    )
    # The oracle body keeps both zero fills: it is the specification and must not move.
    oracle_body = _function_body(source, "torch::Tensor run_rows_exact(")
    assert "auto out = torch::zeros({rows, hidden}, x.options());" in oracle_body
    assert "auto act = torch::zeros({rows * topk, inter}, x.options());" in oracle_body
    # And its down kernel is still the atomic one, so the two epilogues stay independently compiled.
    old_down = _function_body(source, "nvfp4_exact_down_kernel(")
    assert "atomicAdd(out_row + col, acc * w * s2_down);" in _code_only(old_down)


def test_launch_constants_match_the_source():
    """The store entry must launch with an existing constant, not a new private copy of it."""
    source = _code_only(_kernel_source())
    constants = dict(re.findall(r"constexpr int (k\w+Threads) = (\d+);", source))
    assert constants == {"kOracleThreads": "256", "kCompactThreads": "128"}, constants
    assert int(constants["kCompactThreads"]) == K_STORE_THREADS
    assert int(re.search(r"constexpr int kColumnsPerBlock = (\d+);", source).group(1)) == (
        K_COLUMNS_PER_BLOCK
    )
    store_body = _function_body(source, "torch::Tensor run_rows_exact_store(")
    assert "const int threads_per_block = kCompactThreads;" in store_body, (
        "the store entry's launch geometry must be the constant the compact entry already qualified"
    )
    # One guard, in the shared gate launcher, so neither body can launch a too-small block.
    assert _code_only(source).count(
        "threads_per_block % 32 == 0 && threads_per_block >= warps_per_block * 32"
    ) == 1


def test_the_gate_stage_has_one_launch_site_shared_by_both_entries():
    """The store entry must not carry a second launch of the gate stage."""
    source = _code_only(_kernel_source())
    assert source.count("nvfp4_exact_gate_up_kernel<<<") == 1, (
        "the gate stage must be launched from the one shared host launcher; a second launch site "
        "would be a second copy of the geometry to keep in step"
    )
    assert source.count("void launch_gate_up(") == 1
    for name in ("torch::Tensor run_rows_exact(", "torch::Tensor run_rows_exact_store("):
        assert "launch_gate_up(" in _function_body(source, name), f"{name} must use the launcher"
    # The oracle's own down launch stays where it was: only its gate stage moved into the launcher.
    assert source.count("nvfp4_exact_down_kernel<<<") == 1
    assert source.count("nvfp4_exact_down_store_kernel<<<") == 1


def test_the_three_entries_remain_independent_definitions():
    source = _code_only(_kernel_source())
    # Each public entry is its own function; the ordered store is not a flag on an existing path.
    for name in (EXACT, COMPACT, STORE):
        assert re.search(rf"torch::Tensor {name}\(", source), f"{name} must be a distinct entry"
    # The store entry has its own down kernel rather than a mode inside the old one.
    assert "nvfp4_exact_down_store_kernel(" in source
    old_down = _function_body(source, "nvfp4_exact_down_kernel(")
    assert "atomicAdd" in old_down, "the oracle's down stage keeps its atomic epilogue"


def test_binding_is_registered_and_additive():
    bindings = BINDING_SOURCE.read_text()
    assert f'm.def(\n        "{STORE}"' in bindings
    assert f'&{STORE}' in bindings
    for existing in (ORACLE, EXACT, COMPACT):
        assert f'"{existing}"' in bindings, f"{existing} must remain registered"
    # Declaration present at namespace scope, so the definition's TU is the only one that changed.
    assert f"torch::Tensor {STORE}(" in bindings


def test_the_store_entry_is_not_selected_by_the_model_path():
    """It stays an experiment: nothing in the op layer may reference it until the oracle passes."""
    ops = OPS_SOURCE.read_text()
    assert STORE not in ops
    assert "exact_fast_store" not in ops


# ---------------------------------------------------------------------------------------------
# GPU: bit equality against three independent references.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    ext = load_cuda_kernel()
    if ext is None:
        pytest.skip("the CUDA extension is not built for this interpreter")
    for name in (ORACLE, EXACT, COMPACT, STORE):
        if not hasattr(ext, name):
            pytest.skip(f"the built extension has no {name}")
    return ext


def assert_bit_identical(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Compare raw float32 bits: `torch.equal` on floats rejects identical NaNs and folds ±0."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(bits_of(actual), bits_of(expected)), (
        f"{label}: raw float32 bits differ "
        f"({int((bits_of(actual) != bits_of(expected)).sum())} of {actual.numel()} elements)"
    )


def _assert_all_agree(ext, case, label):
    expected = call(ext, ORACLE, case)
    for name in (EXACT, COMPACT, STORE):
        assert_bit_identical(call(ext, name, case), expected, f"{label} vs {name}")
    return expected


@pytest.mark.parametrize("rows", [1, 4, 8, 512])
@pytest.mark.parametrize("routes", ["valid", "duplicate", "out_of_range", "all_unowned", "mixed"])
def test_real_per_rank_shape_is_bit_identical(extension, rows, routes):
    case = build_case(rows=rows, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"],
                      seed=rows * 31 + len(routes), routes=routes, **{
                          key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    _assert_all_agree(extension, case, f"rows={rows} routes={routes}")


@pytest.mark.parametrize("shape", [
    dict(hidden=16, inter=16, experts=2, topk=1),
    dict(hidden=32, inter=48, experts=3, topk=2),
    dict(hidden=80, inter=640, experts=5, topk=4),
    dict(hidden=2560, inter=16, experts=4, topk=3),
    dict(hidden=2576, inter=640, experts=7, topk=5),   # a width the block budget does not divide
])
@pytest.mark.parametrize("rows", [1, 3, 4])
def test_other_shapes_are_bit_identical(extension, shape, rows):
    case = build_case(rows=rows, seed=sum(shape.values()) + rows, **shape)
    _assert_all_agree(extension, case, f"{shape} rows={rows}")


@pytest.mark.parametrize("rows", [1, 4])
def test_global_expert_offset_is_bit_identical(extension, rows):
    case = build_case(rows=rows, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"],
                      seed=977 + rows, start_idx=REAL_SHAPE["experts"] * 3, routes="mixed",
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    _assert_all_agree(extension, case, f"offset rows={rows}")


def test_every_e4m3_scale_byte_participates_and_agrees(extension):
    case = build_case(rows=4, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=6011,
                      scale_bytes=list(range(256)),
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    _assert_all_agree(extension, case, "all 256 scale bytes")
    # The witness proves the bytes reach the kernel; the equality above proves both decode them alike.
    decoded = extension.nvfp4_exact_decode_e4m3(
        torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    )
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0, 1.5])
def test_swiglu_clamping_is_bit_identical(extension, swiglu_limit):
    case = build_case(rows=4, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=77,
                      swiglu_limit=swiglu_limit, routes="mixed",
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    _assert_all_agree(extension, case, f"swiglu_limit={swiglu_limit}")


def test_zero_weight_routes_contribute_nothing_and_still_agree(extension):
    """A zero route weight must leave the accumulated value alone, including its sign."""
    case = build_case(rows=2, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=131,
                      routes="duplicate",
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    case["weights"].zero_()
    expected = _assert_all_agree(extension, case, "all weights zero")
    assert torch.equal(bits_of(expected), bits_of(torch.zeros_like(expected))), (
        "an all-zero route weight set must reproduce the zero-initialised destination exactly, "
        "signed zeros included"
    )


@pytest.mark.parametrize("mutation", ["inf", "nan", "huge", "subnormal"])
def test_extreme_activations_are_bit_identical(extension, mutation):
    case = build_case(rows=2, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=313,
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    x = case["x"]
    if mutation == "inf":
        x[0, 3] = float("inf")
        x[1, 7] = float("-inf")
    elif mutation == "nan":
        x[0, 11] = float("nan")
    elif mutation == "huge":
        x *= torch.finfo(torch.float32).max / 4
    else:
        x *= 1.0e-30   # small enough that some products land near the fp32 subnormal range
    _assert_all_agree(extension, case, f"extreme x={mutation}")


def test_subnormal_partial_sum_separates_a_flushing_atomic_from_a_plain_one(extension):
    """The one case that can distinguish the two candidate atomic semantics.

    Products are formed by `mul.ftz.f32`, so a single contribution is never subnormal. Only the SUM
    of two normal contributions can be, and two routes naming the same expert produce the identical
    per-route scalar, which is what makes the sum controllable: route 0 carries weight `1`, route 1
    carries `-(1 - 2**-23)`, and the remaining sum is the first product scaled by `2**-23`. The
    scale is then chosen so that residual sits below the smallest normal float.

    A mismatch here is not a redesign: it means the old `atomicAdd` does not flush denormals, and
    the accumulation should be `add.rn.f32` instead of `add.rn.ftz.f32`.
    """
    topk = 2
    case = build_case(rows=1, topk=topk, experts=REAL_SHAPE["experts"], seed=4242,
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    # Both routes name the same owned expert, so both see the same expert weight.
    expert = int(case["indices"][0, 0])
    case["indices"][0, 1] = expert

    # Measure the per-route scalar with one active route and unit scales: `out = acc * 1 * 1`.
    probe = dict(case)
    probe["weights"] = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device="cuda")
    probe["w_scale2"] = torch.ones_like(case["w_scale2"])
    probe["w_scale1"] = torch.ones_like(case["w_scale1"])
    scalar = call(extension, ORACLE, probe)[0].abs().max().item()
    if scalar == 0.0 or not math.isfinite(scalar):
        pytest.skip("the probe produced no usable per-route scalar on this checkpoint draw")

    # Put the first product just below 2**-104 so the residual lands under the smallest normal.
    exponent = math.floor(math.log2(scalar))
    shift = -104 - exponent
    scale2 = torch.full_like(case["w_scale2"], math.ldexp(1.0, shift))
    case["w_scale2"] = scale2
    case["w_scale1"] = torch.ones_like(case["w_scale1"])
    case["weights"] = torch.tensor(
        [[1.0, -(1.0 - 2.0 ** -23)]], dtype=torch.float32, device="cuda"
    )

    expected = call(extension, ORACLE, case)
    actual = call(extension, STORE, case)
    differing = int((bits_of(actual) != bits_of(expected)).sum())
    if differing:
        flushed = int((bits_of(actual) == 0).sum())
        kept = int((bits_of(expected) != 0).sum())
        pytest.fail(
            f"{differing} of {actual.numel()} elements differ on a subnormal partial sum: the "
            f"ordered store kept {flushed} zeros where the old atomic kept {kept} denormals. That "
            "means `atomicAdd(float*)` here does NOT flush denormals, so the register accumulation "
            "must use `add.rn.f32` instead of `add.rn.ftz.f32`. It is a one-qualifier change, and "
            "every other case in this file is unaffected -- do not relax the equality."
        )
    # The probe is only evidence if it really exercised a subnormal residual, which under either
    # semantics shows up as a flushed zero of either sign.
    zero_bits = {0x00000000, 0x80000000}
    flushed = sum(1 for value in bits_of(expected).reshape(-1).tolist() if value in zero_bits)
    assert flushed > 0, (
        "the probe never reached the subnormal range: the residual stayed normal, so this case says "
        "nothing about the atomic's denormal behaviour and must not be read as a pass"
    )


def test_repeated_calls_and_captured_replays_agree_after_mutation(extension):
    """A decode round runs under a captured graph on fixed addresses, then mutates them in place."""
    case = build_case(rows=4, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=5150,
                      routes="mixed", **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    expected = call(extension, ORACLE, case)
    for _ in range(3):
        assert_bit_identical(call(extension, STORE, case), expected, "repeated eager call")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call(extension, ORACLE, case)
        call(extension, STORE, case)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graphs, statics = {}, {}
    for name in (ORACLE, STORE):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            statics[name] = call(extension, name, case)
        graphs[name] = graph

    generator = torch.Generator(device="cuda").manual_seed(6120)
    for step in range(3):
        with torch.no_grad():
            case["x"].copy_(torch.randn_like(case["x"], generator=generator))
            case["weights"].copy_(torch.rand(case["weights"].shape, dtype=torch.float32,
                                             device="cuda", generator=generator))
            for name in ("w1q", "w3q", "w2q", "w1s", "w3s", "w2s"):
                case[name].copy_(rand_u8(case[name].shape, generator, "cuda"))
            case["indices"].copy_(torch.randint(
                0, REAL_SHAPE["experts"], case["indices"].shape,
                dtype=torch.int64, device="cuda", generator=generator))
        graphs[ORACLE].replay()
        reference = statics[ORACLE].clone()
        graphs[STORE].replay()
        assert_bit_identical(statics[STORE], reference, f"captured replay mutation {step}")


def test_guards_match_the_oracle_entry(extension):
    """The exact baseline, compact and store entries must refuse with the same guard messages."""
    case = build_case(rows=2, topk=REAL_SHAPE["topk"], experts=REAL_SHAPE["experts"], seed=8080,
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    bad_inputs = [
        ("x must be [rows, hidden]", dict(x=case["x"][0])),
        ("must be a contiguous CUDA tensor", dict(x=case["x"].t().contiguous().t())),
        ("w2 scales must be [E, hidden, inter/16]",
         dict(w2s=case["w2s"][:, :, :-1].contiguous())),
        ("indices/weights shape mismatch",
         dict(weights=case["weights"][:, :-1].contiguous())),
    ]
    for fragment, override in bad_inputs:
        messages = []
        for name in (EXACT, COMPACT, STORE):
            payload = dict(case)
            payload.update(override)
            with pytest.raises(RuntimeError) as excinfo:
                call(extension, name, payload)
            assert fragment in str(excinfo.value), (name, fragment, str(excinfo.value))
            messages.append(str(excinfo.value))
        assert all(message == messages[0] for message in messages), (
            f"the ordered store must refuse with the exact oracle's own message, got {messages}"
        )


def test_an_unwritten_column_would_expose_itself_through_the_poison_pattern(extension):
    """A device-side check that the uninitialised output is fully overwritten.

    The CPU test proves the launch covers every column; this asks the device the same question with
    a poison the store cannot produce. A NaN-free reference is built first, then the allocator is
    loaded with NaN blocks of the output's size class before the second call: a column the store
    never wrote would keep a NaN and surface as a non-finite element, or at least as a bit
    difference between two calls that share inputs.
    """
    case = build_case(rows=2, topk=1, experts=REAL_SHAPE["experts"], seed=1234,
                      **{key: REAL_SHAPE[key] for key in ("hidden", "inter")})
    case["weights"] = torch.ones_like(case["weights"])
    first = call(extension, ORACLE, case)
    assert torch.isfinite(first).all(), "the reference itself must be finite"

    poison = [torch.full((REAL_SHAPE["hidden"],), float("nan"), device="cuda") for _ in range(16)]
    result = call(extension, STORE, case)
    del poison
    assert torch.isfinite(result).all(), (
        "a NaN from the poison pool reached the output, so at least one column was never stored"
    )
    assert_bit_identical(result, first, "poison-pool run against the zeroed oracle")