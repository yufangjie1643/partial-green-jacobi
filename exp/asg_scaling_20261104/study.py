"""Larger-grid application experiments, leaving the completed sources untouched."""
from pathlib import Path
import sys,time,json,hashlib,argparse,gc,platform
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'asg_gpu_resident_20261103'))
from resident import Candidates,Resident,pilot

OUT=HERE/'results'
METHODS=['green','gj','asg','0.5','0.69','compact8']
SNAPSHOTS={0,24,49,62,74,179}

def warm(b,n,mode):
    start=pilot.sync();r=Resident(b,n,load_mode=mode);r.capture()
    r.outer.set_initial(pilot.initial(n,20261104));r.outer.evaluate(1.);r.material(r.outer.a)
    for method in METHODS:r.select(method)
    r.select('0.5');r.solutions.zero_();r.solve()
    if n>=128:
        _,energy=r.energy();cpu_energy=energy.cpu().numpy();x=r.outer.x.cpu().numpy()
        validation=[]
        for beta in (1.,8.):
            r.outer.evaluate(beta);r.outer.step(energy);pilot.sync()
            ph,dp=pilot.physical(x,beta);g=pilot.filt(dp*3*(1-1e-4)*ph**2*cpu_energy);dv=pilot.filt(dp)
            new=pilot.oc(x,g,dv,beta)
            errors={key:float(np.max(np.abs(getattr(r.outer,key).cpu().numpy()-value))) for key,value in
                    dict(ph=ph,dp=dp,a=pilot.material(ph),g=g,dv=dv,new=new).items()}
            assert max(errors[k] for k in ('ph','dp','a','g','dv'))<2e-12 and errors['new']<2e-8,errors
            validation.append(dict(beta=beta,errors=errors))
        pilot.save(OUT/f'outer_validation_n{n}_{mode}.json',dict(passed=True,cases=validation))
    elapsed=pilot.sync()-start;del r;gc.collect();return elapsed

def full(b,n,seed,mode,repeat):
    path=OUT/'full'/f'n{n}_s{seed}_{mode}_r{repeat}.json'
    if path.exists():return
    x=pilot.initial(n,seed);rows=[];snapshots={};stable=0
    with pilot.Guard() as guard,torch.inference_mode():
        start=pilot.sync();r=Resident(b,n,load_mode=mode);r.capture();r.outer.set_initial(x)
        setup_s=pilot.sync()-start
        for k in range(180):
            if guard.events:raise RuntimeError('GPU interference detected; this full-run timing is rejected')
            keep=repeat==0 and mode=='parallel' and k in SNAPSHOTS
            previous=r.solutions.cpu().numpy().copy() if keep else None
            start=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
            events[0].record();beta=[1.,2.,4.,8.][min(k//25,3)]
            r.outer.evaluate(beta);events[1].record();r.material(r.outer.a);events[2].record()
            r.select('0.5');events[3].record();loads,checks=r.solve();events[4].record()
            moduli,energy=r.energy();events[5].record();r.outer.step(energy);events[6].record()
            stats=torch.cat((moduli,r.outer.ph.mean().reshape(1),r.outer.change)).cpu().numpy()
            J=float(stats[:3].mean());change=float(stats[4]);rel=abs(J-rows[-1]['J'])/abs(J) if rows else 1.
            if beta==8 and k>=95 and change<.005 and rel<1e-4:stable+=1
            else:stable=0
            # Snapshot downloads are outside the numerical timing. Storage writes
            # happen after the entire run; snapshots never change solver state.
            snapshot_s=0.
            if keep:
                t=pilot.sync()
                snapshots[k]=dict(x=r.outer.x.cpu().numpy().copy(),physical=r.outer.ph.cpu().numpy().copy(),
                    a=r.outer.a.cpu().numpy().copy(),energy=energy.cpu().numpy().copy(),
                    previous=previous,solution=r.solutions.cpu().numpy().copy(),beta=beta,J=J)
                snapshot_s=pilot.sync()-t
            final=stable>=5 or k==179
            if not final:r.outer.advance()
            elapsed=pilot.sync()-start-snapshot_s
            stages={name:events[i].elapsed_time(events[i+1])/1000 for i,name in enumerate(
                ['filter_s','material_s','selection_precond_s','solve_s','response_s','outer_update_s'])}
            row=dict(step=k,beta=beta,J=J,moduli=stats[:3].tolist(),volume=float(stats[3]),change=change,
                relative_objective_change=rel,step_s=elapsed,loads=loads,iterations=sum(v['iterations'] for v in loads),
                host_checks=checks,snapshot_download_s=snapshot_s,host_and_reporting_s=elapsed-sum(stages.values()),**stages)
            rows.append(row)
            if k%25==0 or final:print(f'FULL n={n} seed={seed} {mode} r={repeat} k={k} J={J:.8f} elapsed={sum(v["step_s"] for v in rows):.1f}s',flush=True)
            if final:break
        final_x=r.outer.x.cpu().numpy();final_ph=r.outer.ph.cpu().numpy()
        result=dict(n=n,seed=seed,load_mode=mode,method='0.5',repeat=repeat,setup_s=setup_s,
            total_s=setup_s+sum(v['step_s'] for v in rows),rows=rows,steps=len(rows),optimization_converged=stable>=5,
            final=rows[-1],precision='FP64 physical/PCG/FFT/outer',snapshot_download_s=sum(v['snapshot_download_s'] for v in rows),
            timing='Numerical wall time, including graph capture and scalar status transfers; excluding initial density generation, snapshot downloads and file writes.')
        ref=OUT/'full'/f'n{n}_s{seed}_parallel_r0.json'
        if ref.exists():
            d=json.loads(ref.read_text());refx=np.load(ref.with_suffix('.npz'))['x']
            validation=dict(max_relative_path_J=max(abs(a['J']-c['J'])/abs(c['J']) for a,c in zip(rows,d['rows'])),
                final_volume_difference=abs(rows[-1]['volume']-d['final']['volume']),relative_density_l2=float(np.linalg.norm(final_x-refx)/np.linalg.norm(refx)))
            assert len(rows)==d['steps'] and validation['max_relative_path_J']<1e-6 and validation['relative_density_l2']<1e-5,validation
            assert validation['final_volume_difference']<1e-10,validation
            result['validation']=validation
        if n==64 and seed==20261104:
            old=np.load(pilot.OUT/'trajectory_n64_s20261104/state_179.npz')
            result['archived_reference_J_error']=abs(J-float(old['J']))/abs(float(old['J']))
            assert result['archived_reference_J_error']<1e-6,result['archived_reference_J_error']
    path.parent.mkdir(parents=True,exist_ok=True)
    if repeat==0:np.savez(path.with_suffix('.npz'),x=final_x,physical=final_ph)
    if snapshots:
        folder=OUT/f'trajectory_n{n}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        for k,state in snapshots.items():np.savez(folder/f'state_{k:03d}.npz',**state)
    pilot.save(path,result)
    print(f'COMPLETE n={n} seed={seed} {mode} r={repeat}: {result["total_s"]:.3f}s, J={J:.10f}',flush=True)
    del r;gc.collect()

def replay(b,n):
    folder=OUT/f'trajectory_n{n}_s20261104'
    if n==32:folder=pilot.OUT/'trajectory_n32_s20261104'
    rng=np.random.default_rng(711+n)
    with pilot.Guard() as guard,torch.inference_mode():
        r=Resident(b,n,load_mode='parallel');r.capture();legacy=b.build(np.ones((n,n,n)))
        for k in [0,49,62,179]:
            if guard.events:raise RuntimeError('GPU interference; current replay block rejected')
            path=OUT/'replay'/f'n{n}_k{k:03d}.json'
            if path.exists():continue
            d=np.load(folder/f'state_{k:03d}.npz')
            a=d['a'];x=d['x'];beta=float(d['beta']);ga=torch.tensor(a,device='cuda')
            previous=torch.tensor(d['previous'],device='cuda',dtype=torch.float64)
            # Independent high-accuracy original-operator solve on the same state.
            reference,_,refenergy=b.evaluate(legacy,a,'0.5',[u[None] for u in previous],rtol=1e-11)
            assert reference['converged']
            r.material(ga);r.outer.set_initial(x);r.outer.evaluate(beta)
            for method in METHODS:
                r.select(method);r.solutions.copy_(previous);r.solve()
            rows=[]
            for repeat in range(3):
                for method in rng.permutation(METHODS):
                    r.solutions.copy_(previous);start=pilot.sync();r.material(ga);loaded=pilot.sync()
                    r.select(str(method));selected=pilot.sync();loads,checks=r.solve();solved=pilot.sync()
                    moduli,energy=r.energy();J=float(moduli.mean());end=pilot.sync()
                    err=abs(J-reference['J'])/abs(reference['J'])
                    assert err<1e-7,(n,k,method,err)
                    rows.append(dict(n=n,step=k,beta=float(d['beta']),method=str(method),repeat=repeat,load_mode='parallel',
                        total_s=end-start,material_s=loaded-start,selection_precond_s=selected-loaded,solve_s=solved-selected,
                        response_s=end-solved,alpha=float(r.alpha),J=J,relative_J_error=err,iterations=sum(v['iterations'] for v in loads),loads=loads))
            # Matched outer-chain microbenchmark, independent of the selected solver.
            outer=[];cpu_energy=refenergy;r.outer.set_initial(x);r.outer.evaluate(beta)
            gpu_energy=torch.tensor(cpu_energy,device='cuda',dtype=torch.float64)
            r.outer.step(gpu_energy);pilot.sync()
            for repeat in range(3):
                start=time.perf_counter();ph,dp=pilot.physical(x,beta);pilot.material(ph)
                g=pilot.filt(dp*3*(1-1e-4)*ph**2*cpu_energy);dv=pilot.filt(dp);new=pilot.oc(x,g,dv,beta)
                cpu_s=time.perf_counter()-start
                start=pilot.sync();r.outer.evaluate(beta);r.outer.step(gpu_energy);gpu_s=pilot.sync()-start
                error=float(np.max(np.abs(new-r.outer.new.cpu().numpy())))
                assert error<2e-8,error
                outer.append(dict(repeat=repeat,cpu_s=cpu_s,gpu_s=gpu_s,max_density_difference=error))
            guard.check()
            if guard.events:raise RuntimeError('GPU interference; current replay block rejected')
            pilot.save(path,dict(n=n,step=k,source=str(folder/f'state_{k:03d}.npz'),source_sha256=hashlib.sha256((folder/f'state_{k:03d}.npz').read_bytes()).hexdigest(),reference=reference,rows=rows,outer=outer))
            print(f'REPLAY n={n} k={k}: 18 three-load solves + outer comparison passed',flush=True)

def memory(b,n,mode):
    # Run this subcommand in a fresh process, separated from time measurements.
    import threading
    with pilot.Guard(),torch.inference_mode():
        r=Resident(b,n,load_mode=mode);r.capture();r.outer.set_initial(pilot.initial(n,20261104))
        r.outer.evaluate(1.);r.material(r.outer.a);r.select('0.5');r.solve();pilot.sync()
        del r;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated();free,total=torch.cuda.mem_get_info();used0=total-free
        r=Resident(b,n,load_mode=mode);r.capture();r.outer.set_initial(pilot.initial(n,20261104))
        samples=[];stop=threading.Event()
        def sample():
            torch.cuda.set_device(0)
            while not stop.wait(.01):
                free,total=torch.cuda.mem_get_info();samples.append(total-free)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        try:
            for _ in range(3):
                r.outer.evaluate(1.);r.material(r.outer.a);r.select('0.5');r.solve()
                _,energy=r.energy();r.outer.step(energy);r.outer.advance();pilot.sync()
            result=dict(n=n,load_mode=mode,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_increment_bytes=torch.cuda.max_memory_allocated()-base,sampled_device_peak_bytes=max(samples),baseline_device_bytes=used0,
                pcg_work_shape=list(r.p.shape),solutions_shape=list(r.solutions.shape),sample_interval_s=.01,
                protocol='Fresh process; warm compilation; allocation peak includes capture; device sampling covers three beta=1 outer steps after capture; alpha=0.5.')
        finally:stop.set();thread.join()
    pilot.save(OUT/f'memory_n{n}_{mode}.json',result);print(result,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['full','replay','memory']);p.add_argument('--n',type=int,required=True)
    p.add_argument('--mode',choices=['serial','parallel'],default='parallel');p.add_argument('--seeds',nargs='+',type=int,default=[20261104,20261105]);p.add_argument('--repetitions',type=int,default=3)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmup=warm(b,args.n,args.mode) if args.task=='full' else None
    pilot.save(OUT/f'session_{time.time_ns()}.json',dict(args=vars(args),warmup_s=warmup,python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [HERE,HERE.parent/'asg_gpu_resident_20261103'] for p in folder.glob('*') if p.suffix in ('.py','.cu')}))
    with torch.cuda.stream(stream):
        if args.task=='full':
            for repeat in range(args.repetitions):
                for seed in args.seeds:full(b,args.n,seed,args.mode,repeat)
        elif args.task=='replay':replay(b,args.n)
        else:memory(b,args.n,args.mode)

if __name__=='__main__':main()
