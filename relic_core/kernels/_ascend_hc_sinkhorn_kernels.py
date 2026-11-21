"""Register-resident FP32 four-by-four Sinkhorn normalizations."""
import triton
import triton.language as tl


@triton.jit
def sinkhorn_normalize_kernel(INPUT,OUTPUT,ROWS:tl.constexpr,ITERS:tl.constexpr,EPS:tl.constexpr):
    lanes=tl.arange(0,16)
    for row in range(tl.program_id(0),ROWS,tl.num_programs(0)):
        matrix=tl.reshape(tl.load(INPUT+row*16+lanes),(4,4))
        matrix=matrix/(tl.sum(matrix,axis=0)[None,:]+EPS)
        for _ in tl.static_range(ITERS-1):
            matrix=matrix/(tl.sum(matrix,axis=1)[:,None]+EPS)
            matrix=matrix/(tl.sum(matrix,axis=0)[None,:]+EPS)
        tl.store(OUTPUT+row*16+lanes,tl.reshape(matrix,(16,)))
