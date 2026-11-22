"""Metadata and composition tests for the explicit scale-LUT load certificate.

Native execution is replaced by spies; these tests do not certify device numerics.
Only the measured compact N32/N64 K128 group-4 consumers carry a certificate;
ordinary and original n16 consumers keep their unchanged call shape.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import expert_tp_moe as moe


class LUTConsumer:
    scale_min = 117
    scale_max = 127
    weight_format = "relic.fp4.cube-n32-k128.v1"
    cube_packed = True
    group_k = 4
    output_rows = 32
    block_dim = 24

    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.calls = []

    def __call__(self, x, *args, **kwargs):
        self.calls.append((x, args, kwargs))
        dtype = kwargs.get("output_dtype", torch.bfloat16)
        return torch.ones((2, 64), dtype=dtype)


class OrdinaryConsumer:
    def __call__(self, *args, **kwargs):
        raise AssertionError("metadata validation must never call a consumer")


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("consumer", [None, OrdinaryConsumer()])
def test_ordinary_consumer_has_no_new_certificate_or_keyword(backend, consumer):
    assert moe._require_scale_lut_certificate(consumer, backend, False) is False


@pytest.mark.parametrize("value", [None, 0, 1, "yes", object()])
def test_certificate_must_be_an_explicit_bool(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        moe._require_scale_lut_certificate(LUTConsumer(), "triton", value)


def test_a_bounded_consumer_does_not_prove_its_bank_values():
    with pytest.raises(ValueError, match="load-time"):
        moe._require_scale_lut_certificate(LUTConsumer(), "triton", False)


@pytest.mark.parametrize("consumer", [None, OrdinaryConsumer()])
def test_certificate_is_not_attached_to_unrelated_consumers(consumer):
    with pytest.raises(ValueError, match="explicitly bounded"):
        moe._require_scale_lut_certificate(consumer, "triton", True)


@pytest.mark.parametrize("fields", [
    {"scale_min": 2, "scale_max": 252},
    {"scale_min": True},
    {"scale_min": "117"},
    {"scale_min": None},
    {"scale_max": None},
    {"scale_max": 128},
    {"weight_format": "original-n16"},
    {"cube_packed": False},
    {"group_k": True},
    {"group_k": 8},
    {"output_rows": 64},
    {"output_rows": True},
])
def test_malformed_or_mismatched_lut_contract_is_refused(fields):
    with pytest.raises(ValueError, match="range contract"):
        moe._require_scale_lut_certificate(LUTConsumer(**fields), "triton", True)


def test_cpu_oracle_is_not_a_scale_lut_fallback():
    with pytest.raises(ValueError, match="range contract"):
        moe._require_scale_lut_certificate(LUTConsumer(), "torch", True)


def test_both_explicit_compact_geometries_can_forward_a_certificate():
    assert moe._require_scale_lut_certificate(LUTConsumer(), "triton", True)
    assert moe._require_scale_lut_certificate(
        LUTConsumer(weight_format="relic.fp4.cube-n64-k128.v1", output_rows=64), "triton", True)


def test_group_eight_is_not_a_certifiable_scale_lut_geometry():
    """The slim path certifies the original group-4 K32 grid only."""
    for fields in ({"group_k": 8},
                   {"weight_format": "relic.fp4.cube-n64-k128.v1", "output_rows": 64, "group_k": 8}):
        with pytest.raises(ValueError, match="range contract"):
            moe._require_scale_lut_certificate(LUTConsumer(**fields), "triton", True)


def test_noncallable_metadata_is_not_a_lut_consumer():
    metadata = SimpleNamespace(scale_min=117, scale_max=127)
    with pytest.raises(ValueError, match="range contract"):
        moe._require_scale_lut_certificate(metadata, "triton", True)


def test_missing_certificate_stops_before_quantization_or_routing(monkeypatch):
    calls = []
    monkeypatch.setattr(moe, "_validate", lambda *args: None)
    monkeypatch.setattr(moe, "quantize_decode_fp8_fast", lambda *args, **kwargs: calls.append("quant"))
    with pytest.raises(ValueError, match="load-time"):
        moe.tp_moe_down_partials(None, None, None, {}, ep_rank=0, ep_world=1, n_experts=8,
                                workspace=None, consumer=LUTConsumer(), backend="triton")
    assert calls == []


def test_ordinary_original_n16_g8_consumer_remains_without_a_lut_keyword(monkeypatch):
    consumer = LUTConsumer(scale_min=None, scale_max=None, group_k=8, output_rows=16,
                           weight_format="original-n16", cube_packed=False)
    workspace = SimpleNamespace(group_k=8, block_dim=24, weight_rows=16,
        weight_storage=torch.empty(1, dtype=torch.bfloat16), selected=torch.zeros(2, dtype=torch.int32),
        gate_weights=torch.ones(2), gate_accumulation=torch.empty((2, 64)),
        up_accumulation=torch.empty((2, 64)), down_accumulation=torch.empty((2, 64)))
    monkeypatch.setattr(moe, "_validate", lambda *args: None)
    monkeypatch.setattr(moe, "_triton_routes", lambda *args, **kwargs: None)
    monkeypatch.setattr(moe, "fused_moe_epilogue", lambda *args, **kwargs: torch.ones((2, 64), dtype=torch.bfloat16))
    activation = torch.ones((1, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(moe, "quantize_decode_fp8_fast", lambda *args, **kwargs: activation)
    banks = {name: (torch.empty(1, dtype=torch.uint8), torch.empty(1, dtype=torch.uint8))
             for name in ("w1", "w3", "w2")}
    moe.tp_moe_down_partials(activation, None, None, banks, ep_rank=0, ep_world=1, n_experts=8,
                             workspace=workspace, consumer=consumer, backend="triton")
    assert all("certified_scale_lut_operands" not in call[2] for call in consumer.calls)


@pytest.mark.parametrize("rows,format_name", [(32, "relic.fp4.cube-n32-k128.v1"),
                                              (64, "relic.fp4.cube-n64-k128.v1")])
def test_certificate_is_forwarded_to_each_single_projection_not_requantized(
        monkeypatch, rows, format_name):
    # This is a call-composition spy, not an NPU fallback or a device test. The
    # real binding independently refuses CPU input and validates all extents.
    calls = []
    activation = torch.ones((1, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(moe, "_validate", lambda *args: None)
    def quantize(*args, **kwargs):
        calls.append("quant")
        return activation
    def routes(*args, **kwargs):
        calls.append("routes")
    monkeypatch.setattr(moe, "quantize_decode_fp8_fast", quantize)
    monkeypatch.setattr(moe, "_triton_routes", routes)
    local = torch.ones((2, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(moe, "fused_moe_epilogue", lambda *args, **kwargs: local)
    workspace = SimpleNamespace(group_k=4, block_dim=24, weight_rows=rows,
        weight_storage=torch.empty(1, dtype=torch.bfloat16), selected=torch.zeros(2, dtype=torch.int32),
        gate_weights=torch.ones(2), gate_accumulation=torch.empty((2, 64)),
        up_accumulation=torch.empty((2, 64)), down_accumulation=torch.empty((2, 64)))
    consumer = LUTConsumer(weight_format=format_name, output_rows=rows)
    banks = {name: (torch.empty(1, dtype=torch.uint8), torch.empty(1, dtype=torch.uint8))
             for name in ("w1", "w3", "w2")}
    result = moe.tp_moe_down_partials(activation, None, None, banks,
        ep_rank=0, ep_world=1, n_experts=8, workspace=workspace, consumer=consumer,
        backend="triton", certified_scale_lut_operands=True)
    assert calls == ["quant", "routes"]
    assert len(consumer.calls) == 3
    assert consumer.calls[0][0] is activation and consumer.calls[1][0] is activation
    assert consumer.calls[2][0] is local
    assert all(call[2]["certified_scale_lut_operands"] is True for call in consumer.calls)
    assert all(call[2]["certified_bf16_operands"] is True for call in consumer.calls)
    assert consumer.calls[2][2]["output_dtype"] == torch.float32
    assert result.dtype == torch.float32