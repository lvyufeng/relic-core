"""N64 workspace admission is explicit; legacy N64 rejection is preserved."""
import pytest
import torch

from relic_core.kernels import ascendc_tile_moe as tile


def make(**kwargs):
    options = dict(device="cpu", group_k=4, weight_rows=64, compact_n64=True)
    options.update(kwargs)
    return tile.make_tile_moe_workspace(5120, 1152, 6, **options)


def test_n64_requires_the_explicit_new_flag():
    with pytest.raises(ValueError, match="original"):
        make(compact_n64=False)
    with pytest.raises(ValueError, match="original"):
        tile.make_tile_moe_workspace(5120, 1152, 6, device="cpu", group_k=4, weight_rows=64)


def test_n64_keeps_the_same_scratch_types_with_a_counted_8192_element_ring():
    workspace = make()
    assert workspace.weight_rows == 64 and workspace.group_k == 4
    assert workspace.weight_storage.shape == (24, 2, 8192)
    assert workspace.weight_storage.dtype == torch.bfloat16
    assert workspace.weight_storage.numel() * workspace.weight_storage.element_size() == 786432
    assert workspace.nbytes == 964716
    assert workspace.gate_accumulation.shape == workspace.up_accumulation.shape == (6, 1152)
    assert workspace.down_accumulation.shape == (6, 5120)
    assert all(value.is_contiguous() and value.device.type == "cpu" for value in workspace.tensors)
    assert len({value.untyped_storage()._cdata for value in workspace.tensors}) == 8


@pytest.mark.parametrize("options", [{"weight_rows": 32}, {"group_k": 2}, {"group_k": 8},
    {"compact_n64": 1}, {"compact_n64": "true"}, {"block_dim": 25}])
def test_n64_flag_does_not_widen_unrelated_workspace_geometry(options):
    with pytest.raises(ValueError):
        make(**options)


@pytest.mark.parametrize("dim,mid", [(5120, 1088), (5184, 1152), (32, 128), (128, 64)])
def test_n64_both_projection_directions_require_complete_k128(dim, mid):
    with pytest.raises(ValueError, match="N64/K128"):
        tile.make_tile_moe_workspace(dim, mid, 6, device="cpu", group_k=4, weight_rows=64, compact_n64=True)


def test_budget_refuses_before_any_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("budget refusal must precede every allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match="budget"):
        make(max_bytes=964715)


def test_exact_budget_admits_the_counted_n64_workspace():
    assert make(max_bytes=964716).nbytes == 964716


def test_existing_n16_and_n32_defaults_are_unchanged():
    n16 = tile.make_tile_moe_workspace(5120, 1152, 6, device="cpu", group_k=4)
    n32 = tile.make_tile_moe_workspace(5120, 1152, 6, device="cpu", group_k=4, weight_rows=32)
    assert n16.weight_rows == 16 and n16.nbytes == 374892
    assert n32.weight_rows == 32 and n32.nbytes == 571500
