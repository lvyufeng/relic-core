"""Original FP4 widening identity and static decode source guards."""
import ast
from pathlib import Path

import pytest
import torch

from relic_core.kernels.static_fp4_decode import unpack_fp4_static,unpack_fp4_pairs,_PAIR_BITS
from relic_core.kernels.unpack import unpack_fp4_to_int8


@pytest.mark.parametrize("tile",[512,1024,2048,4096])
def test_every_original_packed_byte_widens_to_the_same_pair(tile):
    packed=torch.arange(256,dtype=torch.int64).to(torch.uint8).reshape(16,16)
    before=packed.clone()
    expected=unpack_fp4_to_int8(packed,backend="torch")
    out=torch.empty_like(expected)
    result=unpack_fp4_static(packed,backend="torch",tile=tile,out=out)
    assert result is out
    assert torch.equal(result,expected)
    assert torch.equal(packed,before)


def test_static_nibbles_do_not_use_a_variable_shift():
    path=Path(__file__).resolve().parents[1]/"relic_core/kernels/_ascend_static_fp4_decode_kernels.py"
    tree=ast.parse(path.read_text())
    shifts=[node for node in ast.walk(tree) if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.LShift,ast.RShift))]
    assert shifts and all(isinstance(node.right,ast.Constant) for node in shifts)


def test_invalid_output_and_tile_are_refused():
    packed=torch.ones((16,16),dtype=torch.uint8)
    with pytest.raises(ValueError):unpack_fp4_static(packed,tile=8192,backend="torch")
    with pytest.raises((ValueError,TypeError)):
        unpack_fp4_static(packed,backend="torch",out=torch.empty((16,32),dtype=torch.bfloat16))


def test_paired_lookup_has_the_two_original_signed_bytes_in_the_right_order():
    packed=torch.arange(256,dtype=torch.int64).to(torch.uint8).reshape(16,16)
    expected=unpack_fp4_to_int8(packed,backend="torch")
    assert torch.equal(_PAIR_BITS.view(torch.int8).reshape(16,32),expected)
    assert torch.equal(unpack_fp4_pairs(packed,backend="torch"),expected)


def test_private_alias_keeps_a_nonrecursive_cpu_oracle(monkeypatch):
    from relic_core.kernels import unpack
    packed=torch.ones((16,16),dtype=torch.uint8)
    expected=unpack_fp4_to_int8(packed,backend="torch")
    monkeypatch.setattr(unpack,"unpack_fp4_to_int8",unpack_fp4_pairs)
    assert torch.equal(unpack.unpack_fp4_to_int8(packed,backend="torch"),expected)
