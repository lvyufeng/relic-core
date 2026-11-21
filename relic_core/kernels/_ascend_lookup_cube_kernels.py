"""Native BF16 Cube with certified exact bit lookup, not reference-window MMA."""
import triton
import triton.language as tl


@triton.jit
def fp4_lookup_cube_kernel(A,SA,B,SB,AT,BT,OUT,N:tl.constexpr,K:tl.constexpr,
                           BN:tl.constexpr,BK:tl.constexpr,INDEX64:tl.constexpr):
    program=tl.program_id(0)
    if INDEX64:program=program.to(tl.int64)
    rows=tl.arange(0,16)
    lane=tl.arange(0,BK)
    for tile in range(program,tl.cdiv(N,BN),tl.num_programs(0)):
        columns=tile*BN+tl.arange(0,BN)
        active=columns<N
        accumulator=tl.full((16,BN),0,tl.float32)
        for start in range(0,K,BK):
            offset=start+lane
            ac=tl.load(A+offset).to(tl.int32)
            asc=tl.load(SA+offset//32).to(tl.int32)
            abits=tl.load(AT+ac*256+asc)
            av=tl.cast(abits,tl.bfloat16,bitcast=True)
            a=tl.where(rows[:,None]==0,av[None,:],0).to(tl.bfloat16)
            packed=tl.load(B+columns[None,:]*(K//2)+(offset//2)[:,None],mask=active[None,:],other=0).to(tl.int32)
            nibble=tl.where((lane&1)[:,None]==0,packed&15,packed>>4)
            scale=tl.load(SB+columns[None,:]*(K//32)+(offset//32)[:,None],mask=active[None,:],other=127).to(tl.int32)
            bits=tl.load(BT+nibble*256+scale)
            b=tl.cast(bits,tl.bfloat16,bitcast=True)
            accumulator=tl.dot(a,b,acc=accumulator,out_dtype=tl.float32)
        result=tl.sum(tl.where(rows[:,None]==0,accumulator,0.0),axis=0)
        tl.store(OUT+columns,result,mask=active)
