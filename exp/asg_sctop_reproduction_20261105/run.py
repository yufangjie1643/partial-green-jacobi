"""Paper-setting optimizations with certified partial-GJ/Green/GJ/ASG solves."""
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import torch
from reproduction import HERE,ROOT,PaperBackend,PaperResident,initial,pilot

STAGES=['filter_s','material_s','preconditioner_s','solve_s','response_s','outer_s']


def configurations():
    cases=[]
    def add(label,objective,n=30,volume=.2,field=1,rmin=3.,beta=4.,eta=.5,group='core'):
        cases.append(dict(label=label,objective=objective,n=n,volume=volume,field=field,
                          rmin=rmin,beta=beta,eta=eta,emin=1e-9,group=group))
    for objective,prefix,volume in [('shear','S20',.2),('bulk','B30',.3),('auxetic','A20',.2)]:
        for field in range(1,5):add(f'{prefix}_n30_i{field}',objective,volume=volume,field=field)
    for field in [2,3,4]:add(f'A20_n40_i{field}','auxetic',n=40,field=field,group='auxetic_resolution')
    for n in [40,50,60]:add(f'S20_n{n}_r{n//10}','shear',n=n,rmin=n/10,group='fixed_physical_filter')
    for n in [30,40,50,60,70]:add(f'S10_n{n}','shear',n=n,volume=.1,group='low_volume_resolution')
    for beta in [1.,2.,4.,8.]:
        for eta in [.3,.4,.5,.6,.7]:
            if beta==4. and eta==.5:continue # already S20_n30_i1
            add(f'S20_b{beta:g}_t{eta:g}','shear',beta=beta,eta=eta,group='projection')
    return cases


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def warm(b,n):
    start=pilot.sync()
    r=PaperResident(b,n,'auxetic',.2);r.capture()
    r.outer.set_initial(initial(n,1));r.outer.evaluate();r.material(r.outer.a)
    for method in ['green','gj','0.5']:r.select(method)
    r.select('0.5');r.solutions.zero_();r.solve();_,e,_=r.energy(1);r.outer.step(e)
    elapsed=pilot.sync()-start;del r;gc.collect()
    print('WARM',n,round(elapsed,2),'s',flush=True);return elapsed


def optimize(b,spec,method,repeat,max_steps=200,load_mode='parallel',guard=None):
    stem=f'{spec["label"]}_{method}_r{repeat}_{load_mode}'
    path=HERE/'results/full'/f'{stem}.json'
    if path.exists():
        stored=json.loads(path.read_text())
        assert stored['spec']==spec and stored['max_steps']==max_steps
        print('EXISTS',stem,flush=True);return
    n=spec['n'];rows=[];snapshots={};r=None;stage='setup'
    result=dict(spec=spec,method=method,repeat=repeat,max_steps=max_steps,load_mode=load_mode,
                analysis='full-integration periodic HEX8, one analysis element per design voxel',
                load_count=6,inner_rtol=1e-8,linear_iteration_cap=4000,rows=rows,status='running')
    start=pilot.sync()
    try:
        r=PaperResident(b,n,spec['objective'],spec['volume'],spec['rmin'],spec['beta'],spec['eta'],spec['emin'],load_mode)
        r.capture();r.outer.set_initial(initial(n,spec['field']));r.solutions.zero_()
        result['setup_s']=pilot.sync()-start
        for k in range(1,max_steps+1):
            if guard is not None and guard.events:raise RuntimeError('GPU interference: '+repr(guard.events))
            stage=f'iteration_{k}'
            begin=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
            events[0].record();r.outer.evaluate();events[1].record()
            r.material(r.outer.a);events[2].record();r.select(method);events[3].record()
            loads,checks=r.solve();events[4].record();f,e,d=r.energy(k);events[5].record()
            r.outer.step(e);events[6].record()
            stats=torch.cat([f.reshape(1),r.outer.ph.mean().reshape(1),r.outer.trial_ph.mean().reshape(1),
                r.outer.change,r.alpha.reshape(1),r.outer.bounds[2:],d]).cpu().numpy()
            r.stream.synchronize();elapsed=time.perf_counter()-begin
            assert stats[5]==0, 'OC graph capacity exceeded'
            stages={name:events[j].elapsed_time(events[j+1])/1000 for j,name in enumerate(STAGES)}
            row=dict(iteration=k,f=float(stats[0]),volume=float(stats[1]),updated_volume=float(stats[2]),
                     change=float(stats[3]),alpha=float(stats[4]),oc_bisections=int(stats[6]),
                     diagonal=stats[7:].tolist(),loads=loads,host_checks=checks,
                     pcg_iterations=sum(x['iterations'] for x in loads),step_s=elapsed,
                     host_other_s=elapsed-sum(stages.values()),**stages)
            rows.append(row)
            if k in [1,25,75,150]:snapshots[f'rho_{k:03d}']=r.outer.ph.cpu().numpy().copy()
            if k%25==0:print(stem,k,'f',f'{float(f):.7g}','change',f'{stats[3]:.4g}',
                              'PCG',row['pcg_iterations'],flush=True)
            r.outer.advance()
            if stats[3]<=.01:break
        result.update(status='complete',steps=len(rows),optimization_converged=rows[-1]['change']<=.01,
                      stop_reason='density_change' if rows[-1]['change']<=.01 else 'iteration_limit',
                      linear_solves=6*len(rows),total_s=result['setup_s']+sum(x['step_s'] for x in rows),
                      timing_scope='setup/graph capture, filtering, material, selection, six solves and true residuals, objective/sensitivity and OC; excludes warmup, snapshots and final reanalysis')
        stage='final_reanalysis'
        r.outer.evaluate();r.material(r.outer.a);r.select(method)
        certificates,_=r.solve(1e-9)
        f,_,_=r.energy(len(rows));C=r.effective_tensor().cpu().numpy()
        S=np.linalg.inv(C);nu=-S[:3,:3]/np.diag(S)[:3,None]
        ratios=[float(nu[i,j]) for i in range(3) for j in range(3) if i!=j]
        cw,cv=np.linalg.eigh(b.C0);inv=(cv/np.sqrt(cw))@cv.T
        eigen=np.linalg.eigvalsh(inv@C@inv)
        assert eigen.min()>=spec['emin']*.99
        result['final']=dict(f=float(f),volume=float(r.outer.ph.mean()),C_effective=C.tolist(),
             paper_voigt_C=C[np.ix_([0,1,2,5,3,4],[0,1,2,5,3,4])].tolist(),
             G_mean=float(np.trace(C[3:,3:])/3),K=float(C[:3,:3].sum()/9),
             poisson_order=['xy','xz','yx','yz','zx','zy'],poisson_ratios=ratios,nu_min=min(ratios),nu_max=max(ratios),
             all_axis_poisson_negative=max(ratios)<0,normalized_eigenvalues=eigen.tolist(),loads=certificates,
             gray_fraction=float(((r.outer.ph>.05)&(r.outer.ph<.95)).double().mean()))
        arrays=dict(initial=initial(n,spec['field']),x=r.outer.x.cpu().numpy(),physical=r.outer.ph.cpu().numpy(),
                    a=r.outer.a.cpu().numpy(),solutions=r.solutions.cpu().numpy(),**snapshots)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    except Exception as exc:
        result.update(status='failed',failure_stage=stage,error=repr(exc),steps=len(rows),
                      total_s=result.get('setup_s',0)+sum(x['step_s'] for x in rows))
        if r is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path.with_suffix('.npz'),x=r.outer.x.cpu().numpy(),physical=r.outer.ph.cpu().numpy(),
                                a=r.outer.a.cpu().numpy(),**snapshots)
    write(path,result)
    print('DONE',stem,result['status'],result['steps'],f'{result["total_s"]:.3f}s',
          result.get('final',{}).get('nu_min'),result.get('error',''),flush=True)
    del r;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+');parser.add_argument('--groups',nargs='+')
    parser.add_argument('--methods',nargs='+',choices=['green','gj','0.5'],default=['0.5']);parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--max-steps',type=int,default=200);parser.add_argument('--load-mode',choices=['serial','parallel'],default='parallel')
    args=parser.parse_args()
    assert json.loads((HERE/'results/verification.json').read_text())['passed']
    configs=configurations()
    if args.cases:configs=[x for x in configs if x['label'] in args.cases];assert len(configs)==len(args.cases)
    if args.groups:configs=[x for x in configs if x['group'] in args.groups]
    assert configs
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [HERE/'run.py',HERE/'reproduction.py',HERE/'outer.cu',HERE/'scalar_fi.cu']}
    session=dict(args=vars(args),source_sha256=source,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,python=platform.python_version(),cuda=torch.version.cuda,
        reference_commit=subprocess.check_output(['git','-C',str(HERE/'vendor/SCTOP'),'rev-parse','HEAD'],text=True).strip(),warmups={})
    with pilot.Guard() as guard,torch.inference_mode():
        b=PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmed=set()
            for config in configs:
                if config['n'] not in warmed:
                    session['warmups'][config['n']]=warm(b,config['n']);warmed.add(config['n'])
                for rep in range(args.repetitions):
                    for method in args.methods:optimize(b,config,method,rep,args.max_steps,args.load_mode,guard)
    write(HERE/'results'/f'session_{time.time_ns()}.json',session)


if __name__=='__main__':main()
