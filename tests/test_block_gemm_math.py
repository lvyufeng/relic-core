"""Execute actual small Triton GEMM helper ASTs without a device or Triton.

These bit-expression tests complement, and do not replace, compiled NPU tests.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_block_gemm_kernels.py"


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


class Scalar:
    def __init__(self,value):
        self.value = value.value if isinstance(value,Scalar) else value
    def to(self,dtype,bitcast=False):
        if bitcast:
            if dtype in {"uint32", "int32"}:
                raw = bits(self.value)
                return Scalar(raw - 2**32 if dtype == "int32" and raw >= 2**31 else raw)
            if dtype == "float32":
                return Scalar(struct.unpack("<f",struct.pack("<I",int(self.value)&0xffffffff))[0])
            raise AssertionError(dtype)
        if dtype == "float32":
            return Scalar(f32(self.value))
        if dtype == "uint32":
            return Scalar(int(self.value)&0xffffffff)
        if dtype == "int32":
            value=int(self.value)&0xffffffff
            return Scalar(value-2**32 if value>=2**31 else value)
        return Scalar(self.value)
    def __bool__(self):return bool(self.value)
    def __int__(self):return int(self.value)
    def __float__(self):return float(self.value)
    def __add__(self,x):return Scalar(self.value+Scalar(x).value)
    __radd__=__add__
    def __sub__(self,x):return Scalar(self.value-Scalar(x).value)
    def __rsub__(self,x):return Scalar(Scalar(x).value-self.value)
    def __mul__(self,x):return Scalar(f32(self.value*Scalar(x).value))
    __rmul__=__mul__
    def __neg__(self):return Scalar(-self.value)
    def __and__(self,x):return Scalar(int(self.value)&int(Scalar(x)))
    __rand__=__and__
    def __or__(self,x):return Scalar(int(self.value)|int(Scalar(x)))
    __ror__=__or__
    def __lshift__(self,x):
        count=int(Scalar(x));assert 0<=count<32,"undefined shift, including unselected branch"
        return Scalar(int(self.value)<<count)
    def __rlshift__(self,x):return Scalar(x)<<self
    def __rshift__(self,x):
        count=int(Scalar(x));assert 0<=count<32,"undefined shift, including unselected branch"
        return Scalar(int(self.value)>>count)
    def __lt__(self,x):return Scalar(self.value<Scalar(x).value)
    def __le__(self,x):return Scalar(self.value<=Scalar(x).value)
    def __gt__(self,x):return Scalar(self.value>Scalar(x).value)
    def __ge__(self,x):return Scalar(self.value>=Scalar(x).value)
    def __eq__(self,x):return Scalar(self.value==Scalar(x).value)
    def __ne__(self,x):return Scalar(self.value!=Scalar(x).value)


def helpers():
    tree=ast.parse(SOURCE.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name!="block_gemm_kernel"]
    for node in functions:node.decorator_list=[]
    tl=SimpleNamespace(where=lambda c,x,y:Scalar(x if bool(c) else y),
                       maximum=lambda x,y:Scalar(max(Scalar(x).value,Scalar(y).value)),
                       minimum=lambda x,y:Scalar(min(Scalar(x).value,Scalar(y).value)),
                       uint32="uint32",int32="int32",float32="float32",constexpr="constexpr")
    namespace={"tl":tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(SOURCE),"exec"),namespace)
    return namespace


def assert_float_bits(actual,expected):
    actual=float(actual)
    if math.isnan(expected):assert math.isnan(actual)
    else:assert bits(actual)==bits(expected),(hex(bits(actual)),hex(bits(expected)),actual,expected)


def test_scale_each_e4m3_value_by_every_e8m0_byte():
    h=helpers()
    for raw in range(256):
        exponent,mantissa=(raw>>3)&15,raw&7
        if raw&127==127:value=math.nan
        else:
            value=math.ldexp(float(mantissa),-9) if exponent==0 else math.ldexp(float(8+mantissa),exponent-10)
            value=math.copysign(value,-1.0 if raw&128 else 1.0)
        for code in range(256):
            expected=math.nan if code==255 else f32(value*math.ldexp(1.,code-127))
            actual=h["_scale_e8m0"](Scalar(value),Scalar(code))
            assert_float_bits(actual,expected)


def test_arbitrary_float32_scaling_rounds_subnormal_ties_exactly():
    h=helpers();rng=random.Random(619)
    patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800001,0x7f7fffff,0x7f800000,0xff800000,0x7fc00000]
    patterns += [rng.getrandbits(32) for _ in range(1000)]
    for pattern in patterns:
        value=struct.unpack("<f",struct.pack("<I",pattern))[0]
        for code in (0,1,2,100,126,127,128,150,253,254,255):
            expected=math.nan if code==255 else f32(value*math.ldexp(1.,code-127))
            assert_float_bits(h["_scale_e8m0"](Scalar(value),Scalar(code)),expected)


def test_half_scale_bits_recover_only_original_powers():
    h=helpers()
    for half in (False,True):
        for code in range(255):
            value=math.ldexp(1.,code-127-int(half))
            assert int(h["_original_scale_code"](Scalar(value),half))==code
        for value in (0.,-0.,-1.,1.5,math.inf,math.nan):
            assert int(h["_original_scale_code"](Scalar(value),half))==255


def test_separate_scaling_is_not_combined_before_rounding():
    h=helpers()
    value=math.ldexp(1.,-100)
    first=h["_scale_e8m0"](Scalar(value),Scalar(227))
    actual=h["_scale_e8m0"](first,Scalar(227))
    assert_float_bits(actual,math.ldexp(1.,100))
    # The product of scales would overflow before it touched the fragment.
    assert math.isinf(f32(math.ldexp(1.,100)*math.ldexp(1.,100)))


def test_no_unsigned_64bit_data_literal_promotion():
    tree=ast.parse(SOURCE.read_text())
    assert not any(isinstance(node,ast.Constant) and isinstance(node.value,int)
                   and node.value>=2**31 for node in ast.walk(tree))
