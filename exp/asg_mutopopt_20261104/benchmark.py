"""Native muTopOpt/muGrid CUDA baseline; see PROTOCOL.md for timing boundaries."""
from pathlib import Path
import argparse, gc, hashlib, json, os, subprocess, sys, time
import numpy as np
import cupy as cp
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'asg_gpu_resident_20261103'))
from resident import Candidates, Resident, pilot
from muTopOpt import Homogenization, SimpMaterial
import muGrid

OUT = HERE / 'results'
METHODS = ['mutopopt-green', 'mutopopt-gj', 'mutopopt-partial-0.5']

def sync():
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter()

def cup(v):
    return cp.from_dlpack(v)

def to_native(v, n):
    return v.reshape(n, n, n, 3).transpose(3, 2, 1, 0)

def from_native(v):
    return cp.ascontiguousarray(v.transpose(3, 2, 1, 0)).reshape(1, -1)

def project(v):
    q = v.reshape(-1, 3)
    return (q - q.mean(0, keepdim=True)).reshape(1, -1).contiguous()

def macros():
    out = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        e = np.zeros((3, 3)); e[i, j] = e[j, i] = .5
        out.append(e)
    return out

class Native:
    def __init__(self, n, method):
        start = sync()
        self.n, self.method = n, method
        self.h = Homogenization((n,)*3, SimpMaterial(18/7, 2/7, penalty=3., void_ratio=1e-4),
            element='q1', preconditioner='green' if method.endswith('-green') else 'green-jacobi',
            cg_tol=1e-8, cg_maxiter=4000, device='cuda', dtype=np.float64)
        self.x = self.h.vector_field('comparison_x')
        self.rhs = self.h.vector_field('comparison_rhs')
        self.kx = self.h.vector_field('comparison_kx')
        self.h.set_density(cp.ones((n,)*3))
        self.one_time_s = sync()-start

    def update(self, physical):
        self.h.set_density(physical.transpose(2, 1, 0))
        if self.method == 'mutopopt-partial-0.5':
            # Native scale is D^-1/2. Taking its square root gives D^-1/4
            # on each side; no change to upstream PCG or operator kernels.
            q = self.h._prec._jhalf.p
            cp.sqrt(q, out=q)

    def action(self, v):
        self.x.p[...] = to_native(cup(v), self.n)
        self.h.engine.communicate_ghosts(self.x)
        self.h.op.apply(self.x, self.h.lam, self.h.mu, self.kx)
        return torch.from_dlpack(from_native(self.kx.p))

    def native_rhs(self, e):
        self.h.macro_rhs_tensor(e, self.rhs, scale=-1.)
        return torch.from_dlpack(from_native(self.rhs.p))

    def run(self, b, op, physical, previous, reference=None, response_check=False):
        start=sync(); self.update(physical); material_s=sync()-start
        setup_s=solve_s=certificate_s=0.; sols=[]; infos=[]; stresses=[]
        for j,e in enumerate(macros()):
            t=sync()
            native_rhs=self.native_rhs(e)
            rhs=op.compute_rhs(b.loads[j])
            bn=float(torch.linalg.vector_norm(rhs)); target=max(1e-8*bn, 1e-13/np.sqrt(self.n))
            u=project(previous[j:j+1].clone())
            absolute=float(torch.linalg.vector_norm(rhs-op.apply(u)))
            self.x.p[...] = to_native(cup(u),self.n)
            setup_s+=sync()-t
            it=0; restarts=0; load_s=0.
            while absolute>target and it<4000:
                t=sync()
                self.h.solve_rhs(self.rhs,self.x,rtol=.95*target/bn,maxiter=4000-it,
                                 rhs_scale=self.h.mat_scale*.5,warm_start=True)
                load_s+=sync()-t; it+=self.h.last_cg_iters
                t=sync(); u=project(torch.from_dlpack(from_native(self.x.p)))
                absolute=float(torch.linalg.vector_norm(rhs-op.apply(u)))
                certificate_s+=sync()-t
                if absolute<=target or not np.isfinite(absolute) or restarts>=2: break
                self.x.p[...] = to_native(cup(u),self.n)
                restarts+=1
            solve_s+=load_s; sols.append(u)
            infos.append(dict(iterations=it,restarts=restarts,solve_s=load_s,
                absolute_residual=absolute,rhs_norm=bn,relative_residual=absolute/bn,
                threshold=target,converged=bool(np.isfinite(absolute) and absolute<=target)))
            if response_check:
                self.x.p[...] = to_native(cup(u),self.n)
                self.h.engine.communicate_ghosts(self.x)
                sig=self.h.homogenized_stress(self.x,e)
                stresses.append(float(np.sum(sig*e)))
        t=sync(); energy=b.energies(op,sols)
        moduli=(energy*op._material_field.reshape(1,-1)).mean(1)
        J=float(moduli.mean()); response_s=sync()-t
        return dict(method=self.method,load_mode='serial',material_precond_s=material_s,
            rhs_bridge_initial_check_s=setup_s,solve_s=solve_s,certificate_s=certificate_s,
            response_s=response_s,total_s=sync()-start,J=J,native_moduli=stresses,
            moduli=moduli.tolist(),relative_J_error=None if reference is None else abs(J-reference)/abs(reference),
            iterations=sum(q['iterations'] for q in infos),loads=infos,
            converged=all(q['converged'] for q in infos))

def provenance():
    repos={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE/'vendor'/name,text=True).strip()
           for name in ('muGrid','muTopOpt')}
    binaries=list((HERE/'.venv/lib/python3.12/site-packages').glob('**/*muGrid*.so'))
    return dict(repos=repos,python=sys.version,torch=torch.__version__,cupy=cp.__version__,
        gpu=torch.cuda.get_device_name(),binaries={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
        compatibility_patch='results/mugrid_cuda131.patch',precision='float64',cuda_architecture=120,
        runtime_version=cp.cuda.runtime.runtimeGetVersion(),ld_preload=os.environ.get('LD_PRELOAD',''))

def validate(b):
    n=8; ph=pilot.physical(pilot.initial(n,20261119),4.)[0]; a=pilot.material(ph)
    op=b.build(a); v=torch.tensor(np.random.default_rng(162).normal(size=(1,3*n**3)),device='cuda')
    rows=[]
    for method in METHODS:
        ext=Native(n,method); ext.update(cp.asarray(ph))
        kv=op.apply(v); action=float(torch.linalg.vector_norm(ext.action(v)-kv)/torch.linalg.vector_norm(kv))
        rhs_errors=[]; null=[]
        for j,e in enumerate(macros()):
            rhs=op.compute_rhs(b.loads[j]); rn=ext.native_rhs(e)
            rhs_errors.append(float(torch.linalg.vector_norm(rn-rhs)/torch.linalg.vector_norm(rhs)))
            t=torch.zeros_like(v).reshape(-1,3);t[:,j]=1
            null.append(float(torch.linalg.vector_norm(ext.action(t.reshape(1,-1)))/torch.linalg.vector_norm(kv)))
        row=ext.run(b,op,cp.asarray(ph),torch.zeros((3,3*n**3),device='cuda',dtype=torch.float64),response_check=True)
        response_error=float(np.linalg.norm(np.array(row['native_moduli'])-row['moduli'])/np.linalg.norm(row['moduli']))
        row.update(action_error=action,rhs_errors=rhs_errors,null_errors=null,response_error=response_error)
        rows.append(row);print('VALIDATE',method,action,rhs_errors,response_error,row['converged'],flush=True)
    passed=all(max([r['action_error']]+r['rhs_errors']+r['null_errors'])<1e-12
               and r['response_error']<1e-7 and r['converged'] for r in rows)
    pilot.save(OUT/'validation.json',dict(n=n,rows=rows,provenance=provenance(),passed=passed))
    assert passed

def replay(b,n,steps,repetitions):
    assert json.loads((OUT/'validation.json').read_text())['passed'], 'Validate independently before timing'
    op=b.build(np.ones((n,)*3)); natives={m:Native(n,m) for m in METHODS}
    residents={}
    for mode in ('serial','parallel'):
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            r=Resident(b,n,load_mode=mode); r.capture(); residents[mode]=(r,stream)
    pilot.save(OUT/f'construction_n{n}.json',dict(one_time_s={m:r.one_time_s for m,r in natives.items()},provenance=provenance()))
    rng=np.random.default_rng(1104+n)
    cases=[(m,'serial') for m in METHODS]+[(m,mode) for mode in residents for m in ('green','gj','asg','0.5')]
    folder=HERE.parent/('asg_topopt_20261103' if n==32 else 'asg_scaling_20261104')/'results'/f'trajectory_n{n}_s20261104'
    for k in steps:
        dest=OUT/f'replay_n{n}_k{k:03d}.json'
        if dest.exists(): continue
        path=folder/f'state_{k:03d}.npz'; d=np.load(path); reference=float(d['J'])
        ph=cp.asarray(d['physical']); a=torch.tensor(d['a'],device='cuda',dtype=torch.float64)
        prev=torch.tensor(d['previous'],device='cuda',dtype=torch.float64)
        b.update(op,d['a']);sync()
        for ext in natives.values(): ext.run(b,op,ph,prev,reference)
        for mode,(r,stream) in residents.items():
            with torch.cuda.stream(stream):
                r.material(a)
                for m in ('green','gj','asg','0.5'): r.select(m);r.solutions.copy_(prev);r.solve();r.energy()
        rows=[]
        with pilot.Guard() as guard:
            for rep in range(repetitions):
                for idx in rng.permutation(len(cases)):
                    m,mode=cases[idx]
                    if m in natives: row=natives[m].run(b,op,ph,prev,reference)
                    else:
                        r,stream=residents[mode]
                        with torch.cuda.stream(stream):
                            r.solutions.copy_(prev);t=sync();r.material(a);t1=sync();r.select(m);t2=sync()
                            loads,_=r.solve();t3=sync();moduli,_=r.energy();J=float(moduli.mean());t4=sync()
                        row=dict(method=m,load_mode=mode,material_precond_s=t2-t,
                            solve_s=t3-t2,response_s=t4-t3,total_s=t4-t,J=J,
                            relative_J_error=abs(J-reference)/abs(reference),loads=loads,
                            iterations=sum(q['iterations'] for q in loads),converged=all(q['converged'] for q in loads))
                    row.update(n=n,step=k,repeat=rep);rows.append(row)
                    print('REPLAY',n,k,rep,m,mode,round(row['total_s'],6),row['iterations'],row['converged'],flush=True)
                    if guard.events: raise RuntimeError('GPU interference')
        pilot.save(dest,dict(n=n,step=k,reference_J=reference,source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows))
        assert all(r['converged'] and r['relative_J_error']<1e-7 for r in rows), 'failed records retained'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['validate','replay'])
    p.add_argument('--n',type=int,default=32);p.add_argument('--steps',type=int,nargs='+',default=[0,49,62,179])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/native')
    args=p.parse_args();OUT=args.output_dir
    with pilot.Guard(),torch.inference_mode():
        b=Candidates()
        if args.task=='validate': validate(b)
        else: replay(b,args.n,args.steps,args.repetitions)
