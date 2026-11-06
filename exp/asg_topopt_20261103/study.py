"""Periodic SIMP shear-stiffness optimization and matched frozen-ASG timing."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from backend import Backend, ROOT

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
METHODS=['green','gj','asg']
CONFIG=dict(volume=.30,eta=1e-4,p=3.,move=.10,filter_sigma_per_32=.9,
            beta_stages=[1.,2.,4.,8.],stage_steps=25,max_steps=180,
            min_final_steps=20,change_tol=.005,objective_tol=1e-4,consecutive=5,
            rtol=1e-8,reference_rtol=1e-11,seeds=[20261104,20261105],
            grids=[32,64],repetitions=3)
SNAPSHOTS={0,12,24,25,37,49,50,62,74,75,90,110,140,179}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class Guard:
    def __init__(self): self.events=[];self.stop=threading.Event()
    def check(self):
        data=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        other=[line for line in data.splitlines() if line.strip() and int(line.split(',')[0])!=os.getpid()]
        if other:self.events.append(dict(time=time.time(),processes=other))
    def poll(self):
        while not self.stop.wait(.5):
            try:self.check()
            except Exception as exc:self.events.append(dict(error=str(exc)))
    def __enter__(self):
        self.check()
        if self.events:raise RuntimeError('GPU occupied: '+str(self.events))
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.check()
        if self.events:raise RuntimeError('GPU interference; run rejected: '+str(self.events))


def filt(x):return gaussian_filter(x,CONFIG['filter_sigma_per_32']*x.shape[0]/32,mode='wrap')


def physical(x,beta):
    t=filt(x);den=2*np.tanh(beta/2)
    ph=(np.tanh(beta/2)+np.tanh(beta*(t-.5)))/den
    dp=beta*(1-np.tanh(beta*(t-.5))**2)/den
    return ph,dp


def material(ph):return CONFIG['eta']+(1-CONFIG['eta'])*ph**CONFIG['p']


def initial(n,seed):
    rng=np.random.default_rng(seed)
    z,y,x=np.meshgrid(*([(np.arange(n)+.5)/n]*3),indexing='ij')
    field=np.zeros((n,n,n))
    for _ in range(12):
        k=rng.integers(-2,3,3)
        if not np.any(k):k[0]=1
        field+=rng.normal()*np.cos(2*np.pi*(k[0]*x+k[1]*y+k[2]*z)+rng.uniform(0,2*np.pi))
    field=.12*field/field.std()
    lo,hi=-1.,1.
    for _ in range(50):
        shift=(lo+hi)/2;candidate=np.clip(.3+field+shift,.01,.99)
        if physical(candidate,1.)[0].mean()>.3:hi=shift
        else:lo=shift
    return candidate


def oc(x,gradient,volume_gradient,beta):
    lower=np.maximum(0.,x-CONFIG['move']);upper=np.minimum(1.,x+CONFIG['move'])
    ratio=np.maximum(gradient,1e-30)/np.maximum(volume_gradient,1e-30)
    lo,hi=0.,float(ratio.max())*1e4
    for _ in range(45):
        lam=(lo+hi)/2
        candidate=np.clip(x*np.sqrt(ratio/max(lam,1e-30)),lower,upper)
        if physical(candidate,beta)[0].mean()>CONFIG['volume']:lo=lam
        else:hi=lam
    return candidate


def evaluate(b,op,a,method,previous=None,rtol=1e-8):
    start=sync();b.update(op,a);loaded=sync()
    alpha=b.select(op,method);selected=sync()
    us=[];infos=[]
    for j,load in enumerate(b.loads):
        rhs=op.compute_rhs(load)
        u,info=b.solve(op,rhs,None if previous is None else previous[j],rtol)
        us.append(u);infos.append(info)
    solved=sync()
    energy=b.energies(op,us)
    moduli=(energy*op._material_field.reshape(1,-1)).mean(1).cpu().numpy()
    sensitivity=energy.mean(0).reshape(a.shape).cpu().numpy()
    ended=sync()
    row=dict(method=method,alpha=alpha,J=float(moduli.mean()),moduli=moduli.tolist(),
             material_s=loaded-start,selection_precond_s=selected-loaded,solve_s=solved-selected,
             response_s=ended-solved,total_s=ended-start,
             iterations=sum(i['iterations'] for i in infos),loads=infos,
             converged=all(i['converged'] for i in infos))
    return row,us,sensitivity


def warm(b,n):
    print(f'warm n={n}',flush=True)
    a=material(physical(initial(n,20261103),4.)[0]);op=b.build(a)
    with torch.inference_mode():
        for method in METHODS:evaluate(b,op,a,method)
    print(f'warm done n={n}',flush=True)


def protocol(b):
    OUT.mkdir(parents=True,exist_ok=True)
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.cu')}
    payload=dict(config=CONFIG,methods=METHODS,
                 definition='Green-PCG alpha=0; GJ-PCG alpha=1; ASG frozen primary model with threshold .2. Not classical Richardson iteration.',
                 scope='Each step includes filter/projection, material update, selection/scaling, three sequential loads, true residual checks, local energy, sensitivities, OC; shared reference setup included once per run. Disk writes and compilation/warmup excluded and reported separately.',
                 stopping='max(1e-8*norm(rhs), 1e-13/sqrt(n)); initial and final true residual checks. Reference rtol=1e-11.',
                 backend=b.provenance(),code=code,
                 python=sys_version(),torch=torch.__version__,numpy=np.__version__,
                 gpu=torch.cuda.get_device_name(),platform=platform.platform())
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['code']!=code or old['config']!=CONFIG:
            raise RuntimeError('Protocol/source changed: use a separate results directory')
    else:save(path,payload)


def sys_version():
    import sys
    return sys.version


def trajectory(b,n,seed):
    directory=OUT/f'trajectory_n{n}_s{seed}';done=directory/'complete.json'
    if done.exists():return
    directory.mkdir(parents=True,exist_ok=True)
    x=initial(n,seed);op=b.build(np.ones_like(x));previous=None;rows=[];stable=0
    with Guard(),torch.inference_mode():
        for k in range(CONFIG['max_steps']):
            beta=CONFIG['beta_stages'][min(k//CONFIG['stage_steps'],3)]
            ph,dp=physical(x,beta);a=material(ph)
            row,us,energy=evaluate(b,op,a,'0.5',previous,CONFIG['reference_rtol'])
            if not row['converged']:raise RuntimeError(('reference failure',k,row))
            g=filt(dp*CONFIG['p']*(1-CONFIG['eta'])*ph**(CONFIG['p']-1)*energy)
            dv=filt(dp);new=oc(x,g,dv,beta);change=float(abs(new-x).max())
            rel=abs(row['J']-rows[-1]['J'])/abs(row['J']) if rows else 1.
            row.update(step=k,beta=beta,volume=float(ph.mean()),change=change,relative_objective_change=rel)
            rows.append(row)
            if beta==8 and k>=75+CONFIG['min_final_steps'] and change<CONFIG['change_tol'] and rel<CONFIG['objective_tol']:stable+=1
            else:stable=0
            final=stable>=CONFIG['consecutive'] or k==CONFIG['max_steps']-1
            if k in SNAPSHOTS or final:
                prev=np.zeros((3,3*n**3)) if previous is None else torch.cat(previous).cpu().numpy()
                np.savez_compressed(directory/f'state_{k:03d}.npz',x=x,a=a,physical=ph,beta=beta,
                                    previous=prev,solution=torch.cat(us).cpu().numpy(),energy=energy,J=row['J'])
            save(directory/'history.json',rows)
            if k%10==0 or final:print(f'trajectory n={n} seed={seed} k={k} beta={beta:g} J={row["J"]:.7f} d={change:.5f} it={row["iterations"]}',flush=True)
            if final:break
            x=new;previous=us
    save(done,dict(steps=len(rows),optimization_converged=stable>=CONFIG['consecutive'],final=rows[-1]))


def replay(b,n,seed):
    directory=OUT/f'trajectory_n{n}_s{seed}';assert (directory/'complete.json').exists()
    destination=OUT/f'replay_n{n}_s{seed}';destination.mkdir(parents=True,exist_ok=True)
    op=b.build(np.ones((n,n,n)))
    rng=np.random.default_rng(seed+99)
    for path in sorted(directory.glob('state_*.npz')):
        result_path=destination/(path.stem+'.json')
        if result_path.exists():continue
        with np.load(path) as d:state={key:d[key].copy() for key in d.files}
        warm_previous=[torch.tensor(v[None],device='cuda') for v in state['previous']]
        records=[]
        with Guard(),torch.inference_mode():
            for mode in ['cold','warm']:
                for repeat in range(CONFIG['repetitions']):
                    for method in rng.permutation(METHODS):
                        row,us,energy=evaluate(b,op,state['a'],str(method),warm_previous if mode=='warm' else None)
                        row.update(mode=mode,repeat=repeat,step=int(path.stem[-3:]),beta=float(state['beta']),
                                   relative_J_error=float(abs(row['J']-state['J'])/abs(state['J'])),
                                   relative_energy_error=float(np.linalg.norm(energy-state['energy'])/np.linalg.norm(state['energy'])))
                        records.append(row)
        save(result_path,records)
        means={m:round(np.mean([r['total_s'] for r in records if r['method']==m and r['mode']=='warm'])*1000,2) for m in METHODS}
        print(f'replay n={n} seed={seed} {path.stem} warm_ms={means}',flush=True)


def optimize(b,n,seed,method,repeat):
    path=OUT/'closed'/f'n{n}_s{seed}_{method}_r{repeat}.json'
    if path.exists():return
    x=initial(n,seed);rows=[];previous=None;stable=0
    with Guard(),torch.inference_mode():
        start=sync();op=b.build(np.ones_like(x));setup_s=sync()-start
        for k in range(CONFIG['max_steps']):
            start=sync()
            beta=CONFIG['beta_stages'][min(k//CONFIG['stage_steps'],3)]
            ph,dp=physical(x,beta);a=material(ph);filtered=sync()
            row,us,energy=evaluate(b,op,a,method,previous)
            if not row['converged']:raise RuntimeError(('optimization solve failure',method,k,row))
            g=filt(dp*CONFIG['p']*(1-CONFIG['eta'])*ph**(CONFIG['p']-1)*energy)
            dv=filt(dp);new=oc(x,g,dv,beta);change=float(abs(new-x).max())
            rel=abs(row['J']-rows[-1]['J'])/abs(row['J']) if rows else 1.
            elapsed=sync()-start
            row.update(step=k,beta=beta,volume=float(ph.mean()),change=change,relative_objective_change=rel,
                       filter_s=filtered-start,step_s=elapsed,outer_s=elapsed-row['total_s'])
            rows.append(row)
            if beta==8 and k>=75+CONFIG['min_final_steps'] and change<CONFIG['change_tol'] and rel<CONFIG['objective_tol']:stable+=1
            else:stable=0
            if k%30==0:print(f'closed n={n} s={seed} {method} rep={repeat} k={k} J={row["J"]:.6f}',flush=True)
            if stable>=CONFIG['consecutive']:break
            if k<CONFIG['max_steps']-1:x=new;previous=us
    result=dict(n=n,seed=seed,method=method,repeat=repeat,setup_s=setup_s,
                total_s=setup_s+sum(r['step_s'] for r in rows),rows=rows,
                optimization_converged=stable>=CONFIG['consecutive'],steps=len(rows),final=rows[-1])
    save(path,result)
    if repeat==0:np.savez_compressed(path.with_suffix('.npz'),x=x,physical=ph)
    print(f'closed complete n={n} s={seed} {method} rep={repeat} {result["total_s"]:.2f}s J={rows[-1]["J"]:.7f} converged={result["optimization_converged"]}',flush=True)


def validate(b):
    output={};n=4;rng=np.random.default_rng(731)
    a=np.exp(rng.uniform(-4,0,(n,n,n)));op=b.build(a)
    ndof=3*n**3;K=np.zeros((ndof,ndof));rhs=np.zeros((3,ndof));Ke=np.zeros((24,24));Bq=[]
    for grad in b.geometry.gradients((1/n,)*3):
        B=np.zeros((6,24))
        for k,(dx,dy,dz) in enumerate(grad):
            B[0,3*k]=dx;B[1,3*k+1]=dy;B[2,3*k+2]=dz
            B[3,3*k+1]=dz;B[3,3*k+2]=dy;B[4,3*k]=dz;B[4,3*k+2]=dx;B[5,3*k]=dy;B[5,3*k+1]=dx
        Bq.append(B);Ke+=B.T@b.C0@B/(8*n**3)
    indices=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):
                nodes=[(((z+dz)%n*n+(y+dy)%n)*n+(x+dx)%n) for dx,dy,dz in b.offsets]
                ids=np.array([[3*j,3*j+1,3*j+2] for j in nodes]).ravel();indices.append(ids)
                K[np.ix_(ids,ids)]+=a[z,y,x]*Ke
                for j in range(3):rhs[j,ids]-=a[z,y,x]*sum(B.T@b.C0@b.loads[j] for B in Bq)/(8*n**3)
    v=rng.normal(size=ndof);gpu=op.apply(torch.tensor(v[None],device='cuda')).cpu().numpy().ravel()
    output['operator_relative_error']=float(np.linalg.norm(gpu-K@v)/np.linalg.norm(K@v))
    diag=b.frozen.diagonal(op).cpu().numpy().ravel()
    output['diagonal_relative_error']=float(np.linalg.norm(diag-K.diagonal())/np.linalg.norm(K.diagonal()))
    T=np.tile(np.eye(3),(n**3,1));direct=np.linalg.solve(K+T@T.T/n**3,rhs.T).T
    solutions=[];checks=[]
    with torch.inference_mode():
        for j in range(3):
            r=op.compute_rhs(b.loads[j]);checks.append(float(np.linalg.norm(r.cpu().numpy().ravel()-rhs[j])/np.linalg.norm(rhs[j])))
            b.select(op,'gj');u,info=b.solve(op,r,rtol=1e-12);assert info['converged'];solutions.append(u)
    actual=torch.cat(solutions).cpu().numpy()
    output['rhs_relative_error']=max(checks)
    output['solution_relative_error']=float(np.linalg.norm(actual-direct)/np.linalg.norm(direct))
    energies=b.energies(op,solutions).cpu().numpy();independent=np.zeros_like(energies)
    for j in range(3):
        for e,ids in enumerate(indices):
            for B in Bq:
                strain=b.loads[j]+B@direct[j,ids]
                independent[j,e]+=strain@b.C0@strain/8
    output['energy_relative_error']=float(np.linalg.norm(energies-independent)/np.linalg.norm(independent))
    moduli=(energies*a.ravel()).mean(1)
    stresses=[np.asarray(op.compute_avg_stress(u,load)).ravel()[j+3] for j,(u,load) in enumerate(zip(solutions,b.loads))]
    output['stress_energy_relative_error']=float(np.linalg.norm(np.array(stresses)-moduli)/np.linalg.norm(moduli))
    # Independent adjoint chain-rule check, including periodic filter and projection.
    n=8;x=initial(n,1337);beta=3.;ph,dp=physical(x,beta);op=b.build(material(ph))
    row,us,energy=evaluate(b,op,material(ph),'0.5',rtol=1e-12)
    gradient=filt(dp*CONFIG['p']*(1-CONFIG['eta'])*ph**(CONFIG['p']-1)*energy)
    direction=rng.normal(size=x.shape);direction/=np.max(abs(direction))
    exact=float(np.mean(gradient*direction));fd=[]
    for step in [1e-3,1e-4,1e-5]:
        jp=evaluate(b,op,material(physical(x+step*direction,beta)[0]),'0.5',rtol=1e-12)[0]['J']
        jm=evaluate(b,op,material(physical(x-step*direction,beta)[0]),'0.5',rtol=1e-12)[0]['J']
        estimate=(jp-jm)/(2*step);fd.append(dict(step=step,analytic=exact,finite_difference=estimate,relative_error=abs(estimate-exact)/abs(exact)))
    output['gradient_checks']=fd
    # Exact homogeneous and compatible near-zero RHS checks use the same stop rule.
    op=b.build(np.ones((4,4,4))*.13);b.select(op,'gj');zero=[]
    for load in b.loads:
        u,info=b.solve(op,op.compute_rhs(load));zero.append(info)
    output['zero_rhs_checks']=zero
    assert max(output[k] for k in ['operator_relative_error','diagonal_relative_error','rhs_relative_error','energy_relative_error','stress_energy_relative_error'])<1e-9,output
    assert output['solution_relative_error']<1e-9 and fd[-1]['relative_error']<1e-5,output
    assert all(i['converged'] and i['iterations']==0 for i in zero),output
    save(OUT/'validation.json',output);print(json.dumps(output,indent=2),flush=True)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','trajectory','replay','closed','all'])
    p.add_argument('--grids',nargs='+',type=int,default=CONFIG['grids']);p.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds'])
    p.add_argument('--repeats',type=int,default=CONFIG['repetitions']);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();OUT=args.out;b=Backend()
    if args.mode=='validate':validate(b);return
    protocol(b)
    for n in args.grids:
        warm(b,n)
        for seed in args.seeds:
            if args.mode in ('all','trajectory'):trajectory(b,n,seed)
            if args.mode in ('all','replay'):replay(b,n,seed)
            if args.mode in ('all','closed'):
                for repeat in range(args.repeats):
                    order=np.random.default_rng(seed+repeat).permutation(METHODS)
                    for method in order:optimize(b,n,seed,str(method),repeat)


if __name__=='__main__':main()
