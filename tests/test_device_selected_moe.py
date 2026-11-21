"""Original operands and ordering of the explicit device-selected MoE experiment."""
import pytest
import torch

from relic_core.kernels.device_selected_moe import (
    _torch_prepare, _torch_routes, _torch_weight,
    device_selected_fp4_moe, make_device_moe_workspace, fused_moe_epilogue,
)
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast


def fixture():
    generator = torch.Generator().manual_seed(81)
    banks = {}
    for name in ("w1", "w3", "w2"):
        banks[name] = (
            torch.randint(0, 256, (8, 64, 32), dtype=torch.uint8, generator=generator),
            torch.full((8, 64, 2), 121, dtype=torch.uint8),
        )
    x = (torch.randn((1, 64), generator=generator) * 0.1).to(torch.bfloat16)
    return x, banks


@pytest.mark.parametrize("ids,count", [
    ([1, 3, 5, 7, 9, 11], 0),
    ([1, 2, 5, 7, 9, 11], 1),
    ([10, 3, 2, 7, 6, 11], 3),
    ([10, 0, 2, 14, 6, 4], 6),
])
def test_route_sort_and_dynamic_empty_groups(ids, count):
    x, banks = fixture()
    ws = make_device_moe_workspace(64, 64, 6, device="cpu")
    routing = torch.arange(1, 7, dtype=torch.float32)[None] / 21
    indices = torch.tensor([ids], dtype=torch.int64)
    slots = _torch_routes(indices, routing, rank=0, world=2, n_experts=16, workspace=ws)
    assert len(slots) == count
    expected = sorted(i for i in ids if i % 2 == 0)
    assert ws.selected[:count].tolist() == expected
    assert ws.selected[count:].tolist() == [-1] * (6 - count)
    assert ws.group_list.tolist() == [min(i + 1, count) for i in range(6)] + [6]
    for row, expert in enumerate(expected):
        assert torch.equal(ws.gate_weights[row], routing[0, ids.index(expert)])
    out = device_selected_fp4_moe(x, routing, indices, banks, rank=0, world=2, n_experts=16,
                                  swiglu_limit=1.25, workspace=ws, backend="torch")
    assert out.shape == (1, 64) and out.dtype == torch.float32
    assert torch.isfinite(out).all()
    if count == 0:
        assert torch.equal(out, torch.zeros_like(out))


def test_preparation_keeps_original_payload_and_folds_integer_negative_zero():
    x, banks = fixture()
    ws = make_device_moe_workspace(64, 64, 6, device="cpu")
    raw = torch.arange(256, dtype=torch.int64).to(torch.uint8).repeat(8).view(64, 32)
    banks["w1"][0][0].copy_(raw)
    banks["w3"][0][0].copy_(raw.flip(0))
    before = {name: pair[0].clone() for name, pair in banks.items()}
    decoded_x = quantize_decode_fp8_fast(x, backend="torch")
    prepared = _torch_prepare(banks, [(0, 4)], workspace=ws, world=2,
                              down=False, activation=decoded_x)
    expected = torch.cat((_torch_weight(*[t[0] for t in banks["w1"]]),
                          _torch_weight(*[t[0] for t in banks["w3"]])), dim=0).t()
    assert torch.equal(prepared[0].view(torch.int16), expected.contiguous().view(torch.int16))
    assert torch.equal(ws.x_group[0], decoded_x[0])
    assert torch.count_nonzero(prepared[-1]) == 0
    for name in banks:
        assert torch.equal(banks[name][0], before[name])


def test_partitioned_ascending_sum_matches_one_rank_when_addends_exact():
    x, banks = fixture()
    # All selected IDs are on one rank; selecting them in a different order must
    # change neither weights associated with slots nor the FP32 summation order.
    ids = [10, 0, 2, 14, 6, 4]
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.125, 0.125]], dtype=torch.float32)
    ws = make_device_moe_workspace(64, 64, 6, device="cpu")
    left = device_selected_fp4_moe(x, weights, torch.tensor([ids]), banks,
                                  rank=0, world=2, n_experts=16, workspace=ws, backend="torch")
    permutation = [1, 2, 5, 4, 0, 3]
    right = device_selected_fp4_moe(x, weights[:, permutation], torch.tensor([[ids[i] for i in permutation]]), banks,
                                   rank=0, world=2, n_experts=16, workspace=ws, backend="torch")
    assert torch.equal(left.view(torch.int32), right.view(torch.int32))


@pytest.mark.parametrize("ids", [[0, 0, 2, 4, 6, 8], [16, 1, 2, 3, 4, 5], [-1, 1, 2, 3, 4, 5]])
def test_bad_route_is_refused(ids):
    x, banks = fixture()
    ws = make_device_moe_workspace(64, 64, 6, device="cpu")
    with pytest.raises(ValueError, match="valid and without replacement"):
        device_selected_fp4_moe(x, torch.ones(1, 6), torch.tensor([ids]), banks,
                                 rank=0, world=2, n_experts=16, workspace=ws, backend="torch")


@pytest.mark.parametrize("ids", [[1, 3, 5, 7, 9, 11], [1, 2, 5, 7, 9, 11],
                                  [10, 3, 2, 7, 6, 11], [10, 0, 2, 14, 6, 4]])
def test_original_i8_stage_matches_natural_grouped_fixture_without_requantization(ids):
    x, banks = fixture()
    gates = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.125, 0.125]], dtype=torch.float32)
    indices = torch.tensor([ids], dtype=torch.int64)
    grouped = make_device_moe_workspace(64, 64, 6, device="cpu")
    i8_stage = make_device_moe_workspace(64, 64, 6, device="cpu", projection_mode="weight_quant")
    original = device_selected_fp4_moe(x, gates, indices, banks, rank=0, world=2, n_experts=16,
                                      swiglu_limit=1.25, workspace=grouped, backend="torch")
    actual = device_selected_fp4_moe(x, gates, indices, banks, rank=0, world=2, n_experts=16,
                                    swiglu_limit=1.25, workspace=i8_stage, backend="torch",
                                    projection_mode="weight_quant", fused_epilogue=True)
    assert torch.equal(original.view(torch.int32), actual.view(torch.int32))
    assert i8_stage.nbytes < grouped.nbytes


def test_stage_mode_mismatch_and_failed_direct_dot_are_refused():
    x, banks = fixture()
    stage = make_device_moe_workspace(64, 64, 6, device="cpu")
    args = (x, torch.ones((1, 6)), torch.tensor([[0, 2, 4, 6, 8, 10]]), banks)
    with pytest.raises(ValueError, match="match the preallocated"):
        device_selected_fp4_moe(*args, rank=0, world=2, n_experts=16, workspace=stage,
                                 backend="torch", projection_mode="weight_quant")
    with pytest.raises(ValueError, match="direct-dot prototype failed"):
        device_selected_fp4_moe(*args, rank=0, world=2, n_experts=16, workspace=stage,
                                 backend="torch", projection_mode="fused")


@pytest.mark.parametrize("limit", [0.0, 1.25, 10.0])
def test_fused_epilogue_cpu_oracle_keeps_fp8_codes_scales_and_BF16_boundary(limit):
    generator = torch.Generator().manual_seed(82)
    gate_up = (torch.randn((6, 128), generator=generator) * 7).to(torch.bfloat16)
    gates = torch.tensor([0.0, 0.01, 0.1, 0.25, 0.5, 1.0], dtype=torch.float32)
    left, right = gate_up.float().split(64, dim=-1)
    if limit > 0:
        left = left.clamp(max=limit)
        right = right.clamp(-limit, limit)
    import torch.nn.functional as F
    narrowed = (gates[:, None] * (F.silu(left) * right)).to(torch.bfloat16)
    expected = quantize_decode_fp8_fast(narrowed.contiguous(), backend="torch", return_codes=True)
    paired = fused_moe_epilogue(gate_up, gates, swiglu_limit=limit, backend="torch", return_codes=True)
    separate = fused_moe_epilogue(gate_up[:, :64].contiguous(), gates,
                                 up=gate_up[:, 64:].contiguous(), swiglu_limit=limit,
                                 backend="torch", return_codes=True)
    for actual in (paired, separate):
        assert torch.equal(actual[0].view(torch.int16), expected[0].view(torch.int16))
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(actual[2], expected[2])


def test_empty_fused_epilogue_preserves_output_shapes():
    value = fused_moe_epilogue(torch.empty((0, 128), dtype=torch.bfloat16),
                               torch.empty((0,), dtype=torch.float32), return_codes=True)
    assert [tuple(item.shape) for item in value] == [(0, 64), (0, 64), (0, 2)]


def test_large_source_offset_widens_local_index_before_multiplication():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / "relic_core/kernels/_ascend_device_selected_moe_kernels.py"
    module = ast.parse(source.read_text())
    expected = ast.dump(ast.parse("local = local.to(tl.int64)").body[0])
    names = {"prepare_selected_pairs_kernel", "prepare_selected_i8_pairs_kernel",
             "prepare_selected_half_scales_kernel"}
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    for function in functions:
        guards = [node for node in ast.walk(function) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name) and node.test.id == "INDEX64"]
        assert any(any(ast.dump(statement) == expected for statement in guard.body) for guard in guards)


def test_workspace_bound_is_checked_before_allocation(monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("allocated over budget"))
    with pytest.raises(ValueError, match="budget"):
        make_device_moe_workspace(5120, 2304, 6, device="cpu", max_bytes=1)
