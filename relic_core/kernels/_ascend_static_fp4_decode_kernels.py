"""Static original E2M1 byte widening; output is the exact INT8-times-two pair."""
import triton
import triton.language as tl


@triton.jit
def fp4_static_decode_kernel(A,OUT,SIZE:tl.constexpr,TILE:tl.constexpr,INDEX64:tl.constexpr):
    program=tl.program_id(0)
    if INDEX64:program=program.to(tl.int64)
    output_lane=tl.arange(0,2*TILE)
    for tile in range(program,tl.cdiv(SIZE,TILE),tl.num_programs(0)):
        offset=tile*TILE*2+output_lane
        packed=tl.load(A+offset//2,offset<SIZE*2,other=0).to(tl.int32)
        code=tl.where((output_lane&1)==0,packed&15,packed>>4)
        magnitude=code&7
        absolute=tl.where(magnitude==7,12,tl.where(magnitude==6,8,
                  tl.where(magnitude==5,6,tl.where(magnitude==4,4,magnitude))))
        value=tl.where((code&8)!=0,-absolute,absolute).to(tl.int8)
        tl.store(OUT+offset,value,offset<SIZE*2)


@triton.jit
def fp4_pair_decode_kernel(A,LUT,OUT,SIZE:tl.constexpr,TILE:tl.constexpr,INDEX64:tl.constexpr):
    program=tl.program_id(0)
    if INDEX64:program=program.to(tl.int64)
    for tile in range(program,tl.cdiv(SIZE,TILE),tl.num_programs(0)):
        offset=tile*TILE+tl.arange(0,TILE)
        byte=tl.load(A+offset,offset<SIZE,other=0).to(tl.int32)
        pair=tl.load(LUT+byte)
        tl.store(OUT+offset,pair,offset<SIZE)
