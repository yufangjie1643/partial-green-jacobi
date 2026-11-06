"""Pre-registered complete workflow runs, with independently certified loads."""
import argparse
import gc
import hashlib
import json
import platform
import time
import numpy as np
import torch
from scenarios import HERE, ROOT, CASES, METHODS, SEEDS, MAX_STEPS, Candidates, ScenarioResident, initial, pilot

OUT=HERE/'results'
STAGES=['filter_s','material_s','selection_precond_s','solve_s','response_s','outer_update_s']

def sources():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.cu'))+[HERE/'PROTOCOL.md']
    paths+=list((HERE.parent/'asg_gpu_resident_20261103').glob('*.py'))
    paths+=list((HERE.parent/'asg_gpu_resident_20261103').glob('*.cu'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def warm(b,n):
    started=pilot.sync()
    with torch.inference_mode():
        for case in CASES:
            r=ScenarioResident(b,n,case);r.capture()
            r.outer.set_initial(initial(n,SEEDS[0],CASES[case]['volume']))
            for beta in [1.,8.]:
                r.outer.evaluate(beta);r.material(r.outer.a)
                for method in METHODS:
                    r.select(method)
                r.select('0.5');r.solutions.zero_();r.solve()
                _,e=r.energy();r.outer.step(e)
            del r;gc.collect()
    return pilot.sync()-started

def optimize(b,n,case,seed,method,repeat,max_steps):
    path=OUT/'full'/f'{case}_n{n}_s{seed}_{method}_r{repeat}.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['max_steps']!=max_steps:raise ValueError('existing run has a different evaluation budget')
        return
    spec=CASES[case];x=initial(n,seed,spec['volume']);rows=[];stable=0
    with pilot.Guard() as guard,torch.inference_mode():
        start=pilot.sync();r=ScenarioResident(b,n,case);r.capture();r.outer.set_initial(x)
        setup=pilot.sync()-start
        for k in range(max_steps):
            if guard.events:raise RuntimeError('GPU interference detected')
            start=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
            events[0].record();beta=[1.,2.,4.,8.][min(k//25,3)]
            r.outer.evaluate(beta);events[1].record();r.material(r.outer.a);events[2].record()
            r.select(method);events[3].record();loads,checks=r.solve();events[4].record()
            q,e=r.energy();J_device=r.objective(q);events[5].record()
            r.outer.step(e);events[6].record()
            stats=torch.cat((q,J_device.reshape(1),r.outer.ph.mean().reshape(1),r.outer.change,r.alpha.reshape(1))).cpu().numpy()
            J,volume,change,alpha=map(float,stats[-4:])
            relative=abs(J-rows[-1]['J'])/max(abs(J),1e-30) if rows else 1.
            stable=stable+1 if beta==8 and k>=95 and change<.005 and relative<1e-4 else 0
            final=stable>=5 or k==max_steps-1
            if not final:r.outer.advance()
            r.stream.synchronize();elapsed=time.perf_counter()-start
            stages={name:events[i].elapsed_time(events[i+1])/1000 for i,name in enumerate(STAGES)}
            rows.append(dict(step=k,beta=beta,J=J,responses=stats[:-4].tolist(),volume=volume,
                change=change,relative_objective_change=relative,alpha=alpha,loads=loads,
                iterations=sum(v['iterations'] for v in loads),host_checks=checks,step_s=elapsed,
                host_and_reporting_s=elapsed-sum(stages.values()),**stages))
            if (k+1)%100==0:print(f'  {case} n{n} s{seed} {method} r{repeat}: {k+1} J={J:.7f} change={change:.4g}',flush=True)
            if final:break
        result=dict(case=case,n=n,seed=seed,method=method,repeat=repeat,max_steps=max_steps,
            loads=spec['loads'].tolist(),weights=spec['weights'],normalizers=r.normalizers.cpu().tolist(),
            target_volume=spec['volume'],load_mode=r.load_mode,load_count=r.load_count,
            setup_s=setup,total_s=setup+sum(v['step_s'] for v in rows),rows=rows,steps=len(rows),
            linear_solves=len(rows)*r.load_count,optimization_converged=stable>=5,final=rows[-1],
            timing_valid=True,initial_density_generation_included=False,compilation_warmup_included=False,
            graph_capture_included=True,snapshot_and_output_included=False)
        arrays=dict(x=r.outer.x.cpu().numpy(),physical=r.outer.ph.cpu().numpy(),
            a=r.outer.a.cpu().numpy(),energy=e.cpu().numpy(),initial=x)
        # Independent final analysis with the archived general C21 operator and
        # a stricter tolerance. Outside all design/timing totals.
        refined=[];certificates=[]
        for i,macro in enumerate(spec['loads']):
            rhs=r.op.compute_rhs(macro)
            u,cert=b.solve(r.op,rhs,r.solutions[i:i+1],rtol=1e-11)
            if not cert['converged']:raise RuntimeError('independent final analysis failed')
            refined.append(u);certificates.append(cert)
        ee=r.energy_ext.evaluate(torch.cat(refined,dim=0),b.packed0,r.affine,n)
        qq=(ee*r.outer.a.reshape(1,-1)).mean(1)
        result['reanalysis']=dict(J=float(r.objective(qq)),responses=qq.cpu().tolist(),loads=certificates)
        result['reanalysis']['relative_J_difference']=abs(result['reanalysis']['J']-J)/max(abs(J),1e-30)
        if result['reanalysis']['relative_J_difference']>1e-7:raise RuntimeError('final objective mismatch')
        result['gray_fraction']=float(np.mean((arrays['physical']>.05)&(arrays['physical']<.95)))
        del r;gc.collect()
    # Only commit a formal result after the no-interference guard has exited.
    if repeat==0:
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'),**arrays)
    pilot.save(path,result)
    print(f'DONE {path.stem}: {result["total_s"]:.3f}s; {len(rows)} evaluations; '
          f'J={J:.8f}; converged={result["optimization_converged"]}',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=32)
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=MAX_STEPS)
    args=parser.parse_args()
    verification=json.loads((OUT/'verification.json').read_text())
    if not verification['passed']:raise RuntimeError('verification must pass first')
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):warmup=warm(b,args.n)
    pilot.save(OUT/f'session_{time.time_ns()}.json',dict(args=vars(args),warmup_s=warmup,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        python=platform.python_version(),source_sha256=sources()))
    rng=np.random.default_rng(51105)
    with torch.cuda.stream(stream):
        for rep in range(args.repetitions):
            tasks=[(case,seed,method) for case in args.cases for seed in args.seeds for method in args.methods]
            rng.shuffle(tasks)
            for case,seed,method in tasks:optimize(b,args.n,case,seed,method,rep,args.max_steps)

if __name__=='__main__':main()
