"""Input-bit and explicit-contract proof for the small native Cube lookup API."""
import math

import pytest
import torch

from relic_core.kernels import lookup_cube_gemv as op
from relic_core.kernels import native_cube_block_gemv as reference


@pytest.mark.parametrize("fp4",[False,True])
def test_lookup_has_every_exact_input_bit_and_never_rounds_an_unsafe_pair(fp4):
    table=op._E2_BITS if fp4 else op._E4_BITS
    values=op._E2_VALUES if fp4 else reference._E4M3_VALUES
    safe=reference._E2M1_SAFE if fp4 else reference._E4_SAFE
    for code,value in enumerate(values):
        for scale in range(256):
            actual=table[code,scale].view(torch.bfloat16)
            if bool(safe[code,scale]):
                expected=torch.tensor(math.ldexp(value,scale-127),dtype=torch.bfloat16)
                assert actual.view(torch.int16)==expected.view(torch.int16)
            else:
                assert bool(torch.isnan(actual))


def test_changed_cube_contract_and_original_operands_are_explicit():
    a=torch.full((1,128),0x38,dtype=torch.uint8)
    sa=torch.full((1,4),127,dtype=torch.uint8)
    b=torch.full((32,64),0x11,dtype=torch.uint8)
    sb=torch.full((32,4),127,dtype=torch.uint8)
    before=[item.clone() for item in (a,sa,b,sb)]
    out=torch.empty((1,32),dtype=torch.float32)
    result=op.fp4_lookup_cube_gemv(a,sa,b,sb,backend="torch",out=out)
    assert result is out
    assert torch.equal(result,torch.full_like(result,64.0))
    assert all(torch.equal(item,original) for item,original in zip((a,sa,b,sb),before))
    assert op.CONTRACT=="native_cube_lookup_scaled_bf16_full_K"


def test_original_fp8_rounding_is_not_replaced_by_an_integer_activation():
    a=torch.full((1,128),0x7e,dtype=torch.uint8)
    sa=torch.full((1,4),127,dtype=torch.uint8)
    b=torch.full((16,64),0x77,dtype=torch.uint8)
    sb=torch.full((16,4),127,dtype=torch.uint8)
    result=op.fp4_lookup_cube_gemv(a,sa,b,sb,backend="torch",block_n=16)
    assert torch.equal(result,torch.full_like(result,448.0*6.0*128))


def test_unsafe_nonzero_values_are_refused_instead_of_narrowed():
    a=torch.full((1,128),0x38,dtype=torch.uint8)
    sa=torch.full((1,4),127,dtype=torch.uint8)
    b=torch.full((16,64),0x11,dtype=torch.uint8)
    sb=torch.zeros((16,4),dtype=torch.uint8)
    result=op.fp4_lookup_cube_gemv(a,sa,b,sb,backend="torch")
    assert bool(torch.isnan(result).all())


def test_one_invalid_weight_scale_is_only_a_nan_in_its_own_output_column():
    a=torch.full((1,128),0x38,dtype=torch.uint8)
    sa=torch.full((1,4),127,dtype=torch.uint8)
    b=torch.full((16,64),0x11,dtype=torch.uint8)
    sb=torch.full((16,4),127,dtype=torch.uint8)
    sb[3,0]=255
    result=op.fp4_lookup_cube_gemv(a,sa,b,sb,backend="torch")
    assert bool(torch.isnan(result[0,3]))
    assert int(torch.isnan(result).sum())==1
