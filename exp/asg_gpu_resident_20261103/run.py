"""Matched end-to-end comparison with the archived CPU-outer implementation."""
import argparse
import gc
import hashlib
import time
import platform
import json
from pathlib import Path
import numpy as np
import torch
from resident import Candidates, Resident, HERE, pilot

METHODS=['green','gj','asg','0.5','0.69','compact8']
ORIGINAL=pilot.OUT
OUT=HERE/'results'

def optimize(b,n,seed,method,repeat,chunk=4,load_mode='serial'):
    path=OUT/('resident' if load_mode=='parallel' else 'resident_serial')/f'n{n}_s{seed}_{method}_r{repeat}.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('chunk',4)!=chunk or existing.get('load_mode','parallel')!=load_mode:
            raise ValueError(f'{path} uses different execution settings; use --output-dir')
        return
    x=pilot.initial(n,seed);rows=[];stable=0
    with pilot.Guard(),torch.inference_mode():
        start=pilot.sync();r=Resident(b,n,chunk,load_mode);r.capture();r.outer.set_initial(x);r.solutions.zero_()
        setup_s=pilot.sync()-start
        for k in range(pilot.CONFIG['max_steps']):
            start=time.perf_counter()
            events=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
            events[0].record()
            beta=pilot.CONFIG['beta_stages'][min(k//pilot.CONFIG['stage_steps'],3)]
            r.outer.evaluate(beta);events[1].record()
            r.material(r.outer.a);events[2].record()
            r.select(method);events[3].record()
            infos,checks=r.solve();events[4].record()
            moduli,energy=r.energy();events[5].record()
            r.outer.step(energy);events[6].record()
            stats=torch.cat((moduli,r.outer.ph.mean().reshape(1),r.outer.change,r.alpha.reshape(1))).cpu().numpy()
            J=float(stats[:3].mean());change=float(stats[4])
            rel=abs(J-rows[-1]['J'])/abs(J) if rows else 1.
            if beta==8 and k>=75+pilot.CONFIG['min_final_steps'] and change<pilot.CONFIG['change_tol'] and rel<pilot.CONFIG['objective_tol']:stable+=1
            else:stable=0
            final=stable>=pilot.CONFIG['consecutive'] or k==pilot.CONFIG['max_steps']-1
            if not final:r.outer.advance()
            r.stream.synchronize()
            elapsed=time.perf_counter()-start
            stages={name:events[i].elapsed_time(events[i+1])/1000 for i,name in enumerate(
                ('filter_s','material_s','selection_precond_s','solve_s','response_s','outer_update_s'))}
            row=dict(step=k,beta=beta,method=method,J=J,moduli=stats[:3].tolist(),volume=float(stats[3]),change=change,
                     alpha=float(stats[5]),relative_objective_change=rel,step_s=elapsed,loads=infos,host_checks=checks,
                     iterations=sum(v['iterations'] for v in infos),converged=all(v['converged'] for v in infos),
                     host_and_reporting_s=elapsed-sum(stages.values()),**stages)
            rows.append(row)
            if final:break
        result=dict(n=n,seed=seed,method=method,repeat=repeat,chunk=chunk,load_mode=load_mode,setup_s=setup_s,total_s=setup_s+sum(row['step_s'] for row in rows),
                    rows=rows,steps=len(rows),optimization_converged=stable>=pilot.CONFIG['consecutive'],final=rows[-1],
                    stage_timer='CUDA events, including host launch gaps within each interval; remainder = step wall time minus event intervals',
                    initial_density_generation_included=False,compilation_warmup_included=False,graph_capture_included=True)
        # Independent per-run comparison to the previously certified design path.
        oldpath=HERE.parent/'asg_candidates_20261103/results/closed'/path.name
        if oldpath.exists():
            old=json.loads(oldpath.read_text())
            result['validation']=dict(relative_final_J=abs(J-old['final']['J'])/abs(old['final']['J']),
                                     absolute_final_volume=abs(float(stats[3])-old['final']['volume']),
                                     max_relative_path_J=max(abs(a['J']-c['J'])/abs(c['J']) for a,c in zip(rows,old['rows'])))
            density_path=oldpath.with_name(f'n{n}_s{seed}_{method}_r0.npz')
            if density_path.exists():
                oldx=np.load(density_path)['x'];newx=r.outer.x.cpu().numpy()
                result['validation']['relative_density_l2']=float(np.linalg.norm(newx-oldx)/np.linalg.norm(oldx))
            assert len(rows)==old['steps'] and result['validation']['max_relative_path_J']<1e-6,result['validation']
            assert result['validation']['absolute_final_volume']<1e-10,result['validation']
            assert result['validation'].get('relative_density_l2',0.)<1e-5,result['validation']
        if repeat==0:
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path.with_suffix('.npz'),x=r.outer.x.cpu().numpy(),physical=r.outer.ph.cpu().numpy())
    pilot.save(path,result)
    print(f'resident {load_mode} n={n} seed={seed} {method} r={repeat}: {result["total_s"]:.3f}s setup={setup_s:.3f}s J={J:.10f}',flush=True)
    del r;gc.collect()

def warm(b,n,chunk,load_mode):
    start=pilot.sync();r=Resident(b,n,chunk,load_mode);r.capture()
    state=np.load(ORIGINAL/f'trajectory_n{n}_s20261104/state_179.npz')
    op=b.build(state['a'])
    for method in METHODS:
        b.evaluate(op,state['a'],method)
        r.material(torch.tensor(state['a'],device='cuda'));r.select(method);r.solutions.zero_();r.solve()
    elapsed=pilot.sync()-start;del r,op;gc.collect();return elapsed

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=32)
    parser.add_argument('--methods',nargs='+',default=METHODS)
    parser.add_argument('--seeds',nargs='+',type=int,default=pilot.CONFIG['seeds'])
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--implementations',nargs='+',choices=['baseline','resident'],default=['baseline','resident'])
    parser.add_argument('--chunk',type=int,default=4)
    parser.add_argument('--load-mode',choices=['serial','parallel'],default='serial',help='serial reuses one PCG/FFT/graph workspace; parallel batches three loads')
    parser.add_argument('--output-dir',type=Path,default=OUT,help='use a separate directory for different graph chunk settings')
    args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmup=warm(b,args.n,args.chunk,args.load_mode)
    pilot.save(OUT/f'session_{time.time_ns()}.json',dict(args={**vars(args),'output_dir':str(OUT)},warmup_s=warmup,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),python=platform.python_version(),cuda=torch.version.cuda,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.cu')}))
    tasks=[(seed,method,rep,impl) for rep in range(args.repetitions) for seed in args.seeds for method in args.methods for impl in args.implementations]
    np.random.default_rng(731).shuffle(tasks)
    with torch.cuda.stream(stream):
        for seed,method,rep,impl in tasks:
            if impl=='resident':optimize(b,args.n,seed,method,rep,args.chunk,args.load_mode)
            else:
                pilot.OUT=OUT/'baseline'
                pilot.evaluate=lambda backend,op,a,method,previous=None,rtol=1e-8:backend.evaluate(op,a,method,previous,rtol)
                pilot.optimize(b,args.n,seed,method,rep)

if __name__=='__main__':main()
