"""Exact small-query split-K results against the unchanged single-query CUDA entry.

A torch/reference tolerance check is not the oracle here: the new operation must reproduce each
independent single-query operation bit for bit, including invalid selections and graph replays.
Run with the same interpreter/extension build that the model runtime uses.
"""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.cuda_loader import load_cuda_kernel


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_qsa_splitk_rows_bf16_forward"):
        pytest.skip("needs the built small-query split-K entry")
    if not hasattr(ext, "qwen4_exp_qsa_splitk_bf16_forward"):
        pytest.skip("needs the unchanged single-query exactness oracle")
    return ext


def inputs(rows=4, selected_len=2051, kv_heads=1, heads_per_kv=3, seed=37, batch=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv_len = max(37, selected_len + 17)
    query = torch.randn(batch, rows, kv_heads * heads_per_kv, 256, device="cuda",
                        dtype=torch.bfloat16, generator=generator) * 0.125
    key = torch.randn(batch, kv_heads, kv_len, 256, device="cuda",
                      dtype=torch.bfloat16, generator=generator) * 0.125
    value = torch.randn(batch, kv_heads, kv_len, 256, device="cuda",
                        dtype=torch.bfloat16, generator=generator)
    selected = torch.randint(0, kv_len, (batch, rows, selected_len), device="cuda",
                             dtype=torch.int32, generator=generator)
    # Every query has a different selection. The old kernel ignores both negative and positive
    # out-of-arena ids; duplicate valid slots retain their original multiplicity and order.
    if selected_len >= 8:
        selected[..., -6:] = torch.tensor([0, 0, -1, -2, kv_len, kv_len + 5],
                                          device="cuda", dtype=torch.int32)
    return query.contiguous(), key.contiguous(), value.contiguous(), selected.contiguous()


def serial(ext, query, key, value, selected, scale=256 ** -0.5):
    return torch.cat([
        ext.qwen4_exp_qsa_splitk_bf16_forward(
            query[:, row:row + 1].contiguous(), key, value,
            selected[:, row:row + 1].contiguous(), scale,
        ) for row in range(query.shape[1])
    ], dim=1)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("selected_len", [1, 15, 16, 17, 31, 2048, 2051, 4096])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("seed", [37, 20260830])
def test_batched_queries_equal_serial_single_queries(extension, rows, selected_len, kv_heads, seed):
    query, key, value, selected = inputs(rows, selected_len, kv_heads, seed=seed)
    expected = serial(extension, query, key, value, selected)
    actual = extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 256 ** -0.5)
    assert actual.shape == query.shape and actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected), f"bit mismatch at Q={rows}, S={selected_len}, Hkv={kv_heads}"


@pytest.mark.parametrize("heads_per_kv", range(1, 13))
def test_all_head_specializations_equal_single_query(extension, heads_per_kv):
    query, key, value, selected = inputs(rows=3, selected_len=31, heads_per_kv=heads_per_kv)
    actual = extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)
    assert torch.equal(actual, serial(extension, query, key, value, selected))


@pytest.mark.parametrize("pattern", ["all_invalid", "zero_query", "single_slot_sentinels", "large_finite"])
def test_padding_ties_sentinel_rows_and_large_finite_values(extension, pattern):
    query, key, value, selected = inputs(rows=4, selected_len=2051)
    if pattern == "all_invalid":
        selected.fill_(-1)
        selected[0, 1, ::2] = key.shape[2] + 5
        selected[0, 2, ::2] = -2
    elif pattern == "zero_query":
        query.zero_()
    elif pattern == "single_slot_sentinels":
        selected.fill_(-1)
        selected[0, :, 0] = torch.arange(4, device="cuda", dtype=torch.int32)
        for row in range(4):
            value[0, 0, row].fill_(row + 1)
    else:
        query.mul_(64)
        key.mul_(64)
        value.mul_(32)
    expected = serial(extension, query, key, value, selected)
    actual = extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    if pattern == "all_invalid":
        assert torch.count_nonzero(actual) == 0
    if pattern == "single_slot_sentinels":
        for row in range(4):
            assert torch.equal(actual[0, row], torch.full_like(actual[0, row], row + 1))


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_capture_replays_equal_independent_single_query_graph(extension, rows, kv_heads):
    query, key, value, selected = inputs(rows=rows, kv_heads=kv_heads)
    # Warm both paths on a side stream before capture. Independent graph pools and output buffers
    # keep comparisons from accidentally reading the same allocation.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)
        serial(extension, query, key, value, selected)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    batch_graph, serial_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(batch_graph):
        actual = extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)
    with torch.cuda.graph(serial_graph):
        expected = serial(extension, query, key, value, selected)
    assert actual.data_ptr() != expected.data_ptr()
    for iteration in range(4):
        new_query, _key, _value, new_selection = inputs(rows=rows, kv_heads=kv_heads, seed=37 + iteration)
        query.copy_(new_query)
        selected.copy_(new_selection)
        if iteration == 1:
            selected[0, 0].fill_(-1)
        elif iteration == 2:
            query.zero_()
        batch_graph.replay()
        serial_graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected), f"captured replay {iteration} differs"
        assert torch.equal(actual, serial(extension, query, key, value, selected))


@pytest.mark.parametrize("rows", [0, 9, 512])
def test_query_span_outside_small_row_guard_is_refused(extension, rows):
    query, key, value, selected = inputs(rows=rows, selected_len=17)
    with pytest.raises(RuntimeError, match="1 to 8 query rows"):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)


def test_multiple_batches_are_refused(extension):
    query, key, value, selected = inputs(batch=2, selected_len=17)
    with pytest.raises(RuntimeError, match="batch size 1 only"):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)


def test_head_group_larger_than_supported_specializations_is_refused(extension):
    query, key, value, selected = inputs(heads_per_kv=13, selected_len=17)
    with pytest.raises(RuntimeError, match="at most 12"):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)


@pytest.mark.parametrize("selected_len", [0, 4097])
def test_selection_width_guards(extension, selected_len):
    query, key, value, selected = inputs(selected_len=selected_len)
    with pytest.raises(RuntimeError):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, 0.0625)


@pytest.mark.parametrize("case", ["query_dtype", "selection_dtype", "query_noncontiguous",
                                   "selection_noncontiguous", "head_dimension", "query_device",
                                   "value_shape", "selection_rows", "zero_heads", "zero_scale"])
def test_existing_geometry_dtype_device_and_layout_guards(extension, case):
    query, key, value, selected = inputs(rows=2, selected_len=17)
    scale = 0.0625
    if case == "query_dtype":
        query = query.float()
    elif case == "selection_dtype":
        selected = selected.long()
    elif case == "query_noncontiguous":
        query = torch.cat([query, query], dim=1)[:, ::2]
    elif case == "selection_noncontiguous":
        selected = torch.cat([selected, selected], dim=-1)[..., ::2]
    elif case == "head_dimension":
        query = query[..., :128].contiguous()
    elif case == "query_device":
        query = query.cpu()
    elif case == "value_shape":
        value = value[..., :-1].contiguous()
    elif case == "selection_rows":
        selected = selected[:, :1].contiguous()
    elif case == "zero_heads":
        query = query[:, :, :0].contiguous()
    else:
        scale = 0
    with pytest.raises(RuntimeError):
        extension.qwen4_exp_qsa_splitk_rows_bf16_forward(query, key, value, selected, scale)
