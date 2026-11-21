"""BF16 native Cube with an exact FP32 weight decomposition and FP32 output."""
import triton
import triton.language as tl


@triton.jit
def exact_weight_plane_kernel(A,B0,B1,B2,OUT,N:tl.constexpr,K:tl.constexpr,
                              STRIDE:tl.constexpr,BK:tl.constexpr):
    columns=tl.program_id(0)*32+tl.arange(0,32)
    rows=tl.arange(0,16)
    lanes=tl.arange(0,BK)
    acc0=tl.zeros((16,32),tl.float32)
    acc1=tl.zeros((16,32),tl.float32)
    acc2=tl.zeros((16,32),tl.float32)
    for start in range(0,K,BK):
        k=start+lanes
        values=tl.load(A+k)
        a=tl.where(rows[:,None]==0,values[None,:],0).to(tl.bfloat16)
        offsets=k[:,None]*STRIDE+columns[None,:]
        b0=tl.load(B0+offsets)
        b1=tl.load(B1+offsets)
        b2=tl.load(B2+offsets)
        acc0=tl.dot(a,b0,acc=acc0,out_dtype=tl.float32)
        acc1=tl.dot(a,b1,acc=acc1,out_dtype=tl.float32)
        acc2=tl.dot(a,b2,acc=acc2,out_dtype=tl.float32)
    total=(acc0+acc1)+acc2
    result=tl.sum(tl.where(rows[:,None]==0,total,0.0),axis=0)
    tl.store(OUT+columns,result,columns<N)
