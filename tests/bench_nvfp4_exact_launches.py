"""Compare exact NVFP4 launch geometries on a leased CUDA card without a model-speed claim.

Both entries run from the same binary and use the same tensors. Expert banks rotate to avoid
mistaking a one-bank L2-resident micro for the model's many-layer working set. Rates are CUDA event
intervals of captured replays; capture/allocation/cleanup are reported outside kernel timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',default='1,4,512');p.add_argument('--banks',type=int,default=8)
    p.add_argument('--variant',choices=['compact','store'],default='compact')
    p.add_argument('--samples',type=int,default=7);p.add_argument('--target-ms',type=float,default=100)
    p.add_argument('--route-modes',default='tp8,all_local')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.banks<1 or args.samples<3 or args.target_ms<=0:p.error('invalid timing geometry')
    import torch
    from relic_core.kernels.cuda_loader import load_cuda_kernel
    from test_nvfp4_exact_fast import build_case,call
    torch.cuda.set_device(0);ext=load_cuda_kernel()
    names={'old':'nvfp4_moe_rows_forward_exact_fast',
           'compact':f'nvfp4_moe_rows_forward_exact_fast_{args.variant}'}
    if ext is None or any(not hasattr(ext,n) for n in names.values()):raise RuntimeError('both exact native entries are required')
    identity={'native_sha256':hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
              'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
              'banks':args.banks,'samples':args.samples,'variant':args.variant,
              'native_entries':names,
              'ops_overlay':getattr(ext,'_qwen4_nvfp4_overlay_identity',None)}
    stream=args.output.open('x');results=[]
    def emit(row):stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    emit({'kind':'nvfp4_launch_micro_start','schema':'nvfp4-launch-micro-v1','identity':identity,
          'measurement_scope':'CUDA kernel/replay micro; not full-model throughput'})
    try:
        for rows in [int(v) for v in args.rows.split(',')]:
            for mode in args.route_modes.split(','):
                if mode not in ('tp8','all_local','no_local'):raise ValueError('unknown route mode')
                began=time.perf_counter();cases=[]
                for bank in range(args.banks):
                    c=build_case(rows,2560,640,64,10,2026111900+bank+rows,scale_bytes=[0x28,0x30,0x38,0x40])
                    if mode=='tp8':
                        generator=torch.Generator(device='cuda').manual_seed(20261119000+bank+rows)
                        global_ids=torch.randint(0,512,(rows,10),generator=generator,device='cuda')
                        c['indices'].copy_(torch.where(global_ids.remainder(8)==0,global_ids.div(8,rounding_mode='floor'),-1))
                    if mode=='no_local':c['indices'].fill_(-1)
                    c['w_scale1'].mul_(0.02);c['w_scale2'].mul_(0.02)
                    cases.append(c)
                pairs=[]
                for c in cases:
                    old=call(ext,names['old'],c);new=call(ext,names['compact'],c)
                    if not torch.equal(old.view(torch.int32),new.view(torch.int32)):
                        raise RuntimeError(f'old/compact bits differ for rows={rows}, mode={mode}')
                    pairs.append((old,new))
                graphs={};outputs={}
                for variant,name in names.items():
                    for c in cases:call(ext,name,c)
                    torch.cuda.synchronize()
                    recording=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(recording):
                        outputs[variant]=[call(ext,name,c) for c in cases]
                    graphs[variant]=recording
                torch.cuda.synchronize();setup=time.perf_counter()-began
                for _ in range(16):
                    graphs['old'].replay();graphs['compact'].replay()
                torch.cuda.synchronize()
                def interval(graph,count):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(count):graph.replay()
                    end.record();end.synchronize()
                    return start.elapsed_time(end)/count
                pilot=interval(graphs['old'],4)
                count=max(2,min(512,int(args.target_ms/max(pilot,1e-6))))
                samples={'old':[],'compact':[]}
                for sample in range(args.samples):
                    order=['old','compact'] if sample%2==0 else ['compact','old']
                    for variant in order:
                        samples[variant].append(interval(graphs[variant],count)/args.banks)
                for c in cases:c['x'].mul_(0.91).add_(0.001)
                graphs['old'].replay();graphs['compact'].replay();torch.cuda.synchronize()
                all_equal=all(torch.equal(a.view(torch.int32),b.view(torch.int32))
                              for a,b in zip(outputs['old'],outputs['compact']))
                medians={k:statistics.median(v) for k,v in samples.items()}
                mean_local=sum(int((c['indices']>=0).sum()) for c in cases)/(args.banks*rows)
                emit({'kind':'nvfp4_launch_micro_result','rows':rows,'hidden':2560,'inter':640,
                      'route_mode':mode,'mean_owned_routes_per_row':mean_local,'banks':args.banks,
                      'bank_payload_bytes':sum(c[k].numel()*c[k].element_size() for c in cases for k in
                          ('w1q','w1s','w3q','w3s','w2q','w2s','w_scale1','w_scale2')),
                      'valid':all_equal,'bit_equal_after_mutation':all_equal,
                      'replays_per_sample':count,'ms_per_operator_samples':samples,'median_ms':medians,
                      'compact_time_ratio':medians['compact']/medians['old'],
                      'setup_s':setup,'identity':identity,'measurement_scope':'captured rotating-bank kernel micro; no model TPS claim'})
                if not all_equal:return 2
                results.append((rows,mode,medians))
                del graphs,outputs,cases,pairs,old,new,c,recording
                torch.cuda.synchronize();torch.cuda.empty_cache()
        emit({'kind':'nvfp4_launch_micro_summary','valid':True,'cases':len(results)})
    finally:stream.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
