"""Independent scalar-oracle checks for byte-coded block-32 GEMM.

Set RELIC_EXPERT_TEST_DEVICE=npu:0 and RELIC_EXPERT_TEST_BACKEND=triton
for real compiled acceptance. A CPU run is a separate oracle/validation run.
"""
from __future__ import annotations

import math
import os
import struct

import pytest

torch = pytest.importorskip("torch")
from relic_core.kernels.block_gemm import fp4_int8_block_gemm, fp8_block_gemm
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8


FP4 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def e4(code):
    if code & 127 == 127:
        return math.nan
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(float(8 + mantissa), exponent - 10)
    return math.copysign(value, -1.0 if code & 128 else 1.0)


def nominal(code):
    return max(((code >> 3) & 15) - 7, -6)


def cut(value, anchor):
    if anchor is None:
        return value
    step = math.ldexp(1., anchor - 13)
    return math.trunc(value / step) * step


def fragment(a_codes, b_codes, fp4):
    products, exponents = [], []
    for ac, bc in zip(a_codes, b_codes):
        a = e4(ac)
        if fp4:
            b = FP4[bc]
            bn = math.frexp(abs(b))[1] - 1 if b else None
        else:
            b = e4(bc)
            bn = nominal(bc)
        p = a * b
        if math.isnan(p):
            return math.nan
        products.append(p)
        exponents.append(nominal(ac) + bn if p else None)
    first = max((v for v in exponents[:16] if v is not None), default=None)
    total = sum(cut(p, first) for p in products[:16])
    running = math.frexp(abs(total))[1] - 1 if total else None
    live = [v for v in exponents[16:] if v is not None]
    if running is not None:
        live.append(running)
    second = max(live, default=None)
    total = cut(total, second) + sum(cut(p, second) for p in products[16:])
    return cut(total, math.frexp(abs(total))[1] - 1) if total else total


def oracle(a_codes, a_scale, b_codes, b_scale, fp4, block_rows=1):
    a, sa, b, sb = (t.tolist() for t in (a_codes, a_scale, b_codes, b_scale))
    m, n, k = len(a), len(b), len(a[0])
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            value = 0.
            for block in range(k // 32):
                f = fragment(a[i][block*32:(block+1)*32], b[j][block*32:(block+1)*32], fp4)
                ca, cb = sa[i][block], sb[j // block_rows][block]
                scale_a = math.nan if ca == 255 else math.ldexp(1., ca - 127)
                scale_b = math.nan if cb == 255 else math.ldexp(1., cb - 127)
                scaled = f32(f32(f * scale_a) * scale_b)
                value = f32(value + scaled)
            row.append(value)
        result.append(row)
    return torch.tensor(result, dtype=torch.float32)


def device():
    name = os.environ.get("RELIC_EXPERT_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu
    return torch.device(name)


def backend():
    return os.environ.get("RELIC_EXPERT_TEST_BACKEND", "torch" if device().type == "cpu" else "triton")


def operands(fp4, m=3, n=17, k=96, seed=10, block_rows=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, 256, (m,k), dtype=torch.uint8, generator=g)
    a = torch.where((a & 127) == 127, torch.full_like(a, 126), a)
    sa = torch.randint(123, 132, (m,k//32), dtype=torch.uint8, generator=g)
    b = torch.randint(0, 16 if fp4 else 256, (n,k), dtype=torch.uint8, generator=g)
    if not fp4:
        b = torch.where((b & 127) == 127, torch.full_like(b, 254), b)
    sb = torch.randint(113, 132, (math.ceil(n/block_rows),k//32), dtype=torch.uint8, generator=g)
    return a, sa, b, sb


def invoke(a, sa, b, sb, fp4, block_rows=1, out=None):
    d = device()
    if fp4:
        packed = b[:,0::2] | (b[:,1::2] << 4)
        # Decode from the original packed byte stream. The oracle never reads
        # the widened values or reconstructed scales used by the implementation.
        bi = unpack_fp4_to_int8(packed, backend="torch").to(d)
        half = fp4_int8_scale(sb, backend="torch").to(d)
        result = fp4_int8_block_gemm(a.to(d), sa.to(d), bi, half, backend=backend(), out=out)
    else:
        result = fp8_block_gemm(a.to(d), sa.to(d), b.to(d), sb.to(d), block_rows=block_rows, backend=backend(), out=out)
    if d.type == "npu":
        torch.npu.synchronize(d)
    return result


def assert_bits(actual, expected):
    actual = actual.cpu()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(actual.view(torch.int32)[finite], expected.view(torch.int32)[finite])


@pytest.mark.parametrize("fp4", [True, False])
@pytest.mark.parametrize("m,n,k,seed", [(1,1,32,1),(3,17,96,2),(2,32,64,4),(17,35,32,6)])
def test_random_matrices_match_independent_scalar_bits(fp4,m,n,k,seed):
    args = operands(fp4,m,n,k,seed)
    expected = oracle(*args,fp4)
    assert_bits(invoke(*args,fp4), expected)


def test_dense_square_scale_grid_does_not_read_per_row():
    args = operands(False,3,67,96,12,32)
    expected = oracle(*args,False,32)
    assert_bits(invoke(*args,False,32),expected)


@pytest.mark.parametrize("scale_a", [0, 106, 127, 150, 254, 255])
def test_original_scale_edges_keep_separate_fp32_rounding(scale_a):
    a = torch.zeros(1,64,dtype=torch.uint8)
    a[0,0] = a[0,32] = 0x38
    sa = torch.full((1,2),scale_a,dtype=torch.uint8)
    b = torch.zeros(256,64,dtype=torch.uint8)
    b[:,0] = b[:,32] = 2  # E2M1 value 1.0
    sb = torch.arange(256,dtype=torch.uint8)[:,None].repeat(1,2)
    expected = oracle(a,sa,b,sb,True)
    assert_bits(invoke(a,sa,b,sb,True),expected)


def test_all_e4m3_codes_against_all_e2m1_values():
    a = torch.arange(256,dtype=torch.uint8)[:,None].repeat(1,32)
    a[:,16:] = 0  # independently exercise the first-half carry
    sa = torch.full((256,1),127,dtype=torch.uint8)
    b = torch.arange(16,dtype=torch.uint8)[:,None].repeat(1,32)
    sb = torch.full((16,1),127,dtype=torch.uint8)
    expected = oracle(a,sa,b,sb,True)
    assert_bits(invoke(a,sa,b,sb,True),expected)


def test_k_order_cancellation_and_non_power_of_two_final_fragment():
    a,sa,b,sb = operands(True,1,7,128,190)
    a[:,32:64] = a[:,:32] ^ 128
    b[:,32:64] = b[:,:32]
    sa[:] = 127
    sb[:] = 127
    assert_bits(invoke(a,sa,b,sb,True),oracle(a,sa,b,sb,True))


@pytest.mark.parametrize("n", [1,3,15,17,31,33])
def test_output_identity_and_neighbor_memory_guards(n):
    a,sa,b,sb = operands(True,1,n,32,n)
    storage = torch.full((n+32,),77.125,dtype=torch.float32,device=device())
    out = storage[16:16+n].reshape(1,n)
    actual = invoke(a,sa,b,sb,True,out=out)
    assert actual is out
    observed = storage.cpu()
    assert torch.equal(observed[:16],torch.full((16,),77.125,dtype=torch.float32))
    assert torch.equal(observed[16+n:],torch.full((16,),77.125,dtype=torch.float32))
    assert_bits(actual,oracle(a,sa,b,sb,True))


@pytest.mark.parametrize("fp4", [True, False])
def test_fp32_scale_grid_cannot_silently_be_a_different_quantizer(fp4):
    d=device()
    a=torch.zeros(1,32,dtype=torch.uint8);a[0,0]=0x38
    sa=torch.full((1,1),127,dtype=torch.uint8)
    scales=torch.tensor([.5 if fp4 else 1.,0.,-1.,.75 if fp4 else 1.5,
                         math.inf,math.nan,math.ldexp(1.,-149),
                         math.ldexp(1.,-128 if fp4 else -127),
                         math.ldexp(1.,-127 if fp4 else -126)],dtype=torch.float32)[:,None]
    expected=torch.tensor([[1.,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,
                            math.ldexp(1.,-127),math.ldexp(1.,-126)]],dtype=torch.float32)
    if fp4:
        b=torch.full((9,32),2,dtype=torch.int8,device=d)
        result=fp4_int8_block_gemm(a.to(d),sa.to(d),b,scales.to(d),backend=backend())
    else:
        b=torch.full((9,32),0x38,dtype=torch.uint8,device=d)
        result=fp8_block_gemm(a.to(d),sa.to(d),b,scales.to(d),block_rows=1,backend=backend())
    if d.type=="npu":torch.npu.synchronize(d)
    assert_bits(result,expected)


def test_shape_dtype_grid_and_alias_refusals():
    a = torch.zeros(1,32,dtype=torch.uint8)
    sa = torch.full((1,1),127,dtype=torch.uint8)
    bi = torch.zeros(32,32,dtype=torch.int8)
    half = torch.ones(32,1,dtype=torch.float32)
    with pytest.raises(ValueError,match="divisible"):
        fp4_int8_block_gemm(a[:,:31].contiguous(),sa,bi[:,:31].contiguous(),half)
    with pytest.raises(ValueError,match="grid"):
        fp4_int8_block_gemm(a,sa,bi,torch.ones(1,1))
    with pytest.raises(TypeError,match="dtype"):
        fp4_int8_block_gemm(a,sa,bi.float(),half)
    with pytest.raises(ValueError,match="overlap"):
        fp4_int8_block_gemm(a,sa,bi,half,out=bi.view(torch.float32).flatten()[:32].reshape(1,32))
    with pytest.raises(ValueError,match="backend"):
        fp4_int8_block_gemm(a,sa,bi,half,backend="native")
