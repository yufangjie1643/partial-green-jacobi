"""Three independent FP64 PCG solves scheduled by CUDA graphs.

The scalar FI specialization uses the archived element formula; certificates
use the original general operator. Each load has an independent recurrence and
convergence mask. Only scalar status is read by the host; density,
displacement and sensitivities stay on device.
"""
from pathlib import Path
from functools import lru_cache
import sys
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'asg_candidates_20261103'))
from candidates import Candidates, pilot
from outer import ResidentOuter

@lru_cache(None)
def scalar_extension():
    from torch.utils.cpp_extension import load
    return load(name='asg_resident_scalar_fi_20261103',sources=[str(HERE/'scalar_fi.cu')],
                extra_include_paths=[str(HERE.parents[1]/'solver/src/materials')],
                extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)


@torch.compile(fullgraph=True)
def project(v):
    q=v.reshape(v.shape[0],-1,3)
    return (q-q.mean(1,keepdim=True)).reshape_as(v)

@torch.compile(fullgraph=True)
def pre_input(r,scale):return project(r)*scale

@torch.compile(fullgraph=True)
def pre_output(z,scale):return project(z*scale)

@torch.compile(fullgraph=True)
def update_ur(u,r,p,q,rz,active,iterations,bad,targets2):
    denominator=(p*q).sum(1)
    valid=torch.isfinite(denominator)&(denominator>0)&torch.isfinite(rz)&(rz>0)
    failed=active&~valid
    alpha=torch.where(active&valid,rz/torch.where(valid,denominator,1.),0.)
    unew=u+alpha[:,None]*p
    rnew=r-alpha[:,None]*q
    res2=rnew.square().sum(1)
    finite=torch.isfinite(res2)
    it=iterations+active.to(torch.int64)
    keep=active&valid&finite&(res2>targets2)&(it<4000)
    return unew,rnew,keep,it,bad|failed|(active&~finite)

@torch.compile(fullgraph=True)
def update_p(r,z,p,rz,active):
    newrz=(r*z).sum(1)
    beta=torch.where(active,newrz/torch.where(rz!=0,rz,1.),0.)
    newp=torch.where(active[:,None],z+beta[:,None]*p,0.)
    return newp,newrz


class Resident:
    def __init__(self,backend,n,chunk=4,load_mode='serial'):
        expected=np.zeros((6,6));expected[:3,:3]=4/3
        expected[np.arange(3),np.arange(3)]+=2
        expected[np.arange(3,6),np.arange(3,6)]=1
        if not np.array_equal(backend.C0,expected):
            raise ValueError('scalar CUDA kernel requires the application C0 (lambda=4/3, mu=1)')
        required=dict(volume=.30,eta=1e-4,p=3.,move=.10,filter_sigma_per_32=.9)
        if any(pilot.CONFIG[k]!=v for k,v in required.items()):
            raise ValueError('CUDA outer kernels require the documented frozen SIMP/OC configuration')
        if chunk<1:raise ValueError('positive graph chunk required')
        if load_mode not in ('serial','parallel'):raise ValueError('load_mode must be serial or parallel')
        self.load_mode=load_mode;self.batch=1 if load_mode=='serial' else 3
        self.b=backend;self.n=n;self.chunk=chunk
        self.stream=torch.cuda.current_stream()
        # The native cuFFT plans bind to this stream at construction.
        self.op=backend.build(np.ones((n,n,n)))
        self.outer=ResidentOuter(n,self.stream)
        self.u=torch.zeros((self.batch,3*n**3),device='cuda',dtype=torch.float64)
        # Keep the three warm starts in both modes. Serial allocates only one
        # recurrence/FFT/graph workspace and reuses it for each load.
        self.solutions=self.u if self.batch==3 else torch.zeros((3,3*n**3),device='cuda',dtype=torch.float64)
        for name in ('r','p','rhs'):setattr(self,name,torch.zeros_like(self.u))
        for name in ('rz','target2','norm','absolute','iterations'):
            setattr(self,name,torch.zeros(self.batch,device='cuda',dtype=torch.int64 if name=='iterations' else torch.float64))
        self.active=torch.zeros(self.batch,device='cuda',dtype=torch.bool)
        self.bad=torch.zeros_like(self.active)
        self.scale=torch.ones((1,3*n**3),device='cuda',dtype=torch.float64)
        self.alpha=torch.zeros((),device='cuda',dtype=torch.float64)
        self.macros=torch.tensor(backend.loads,device='cuda',dtype=torch.float64)
        ed=backend.frozen.diagonal.__globals__['s'].element_diagonal(self.op._voxel)
        self.element_diag=torch.tensor(ed,device='cuda',dtype=torch.float64)
        self.hybrid=backend.hybrid.extension()
        self.scalar=scalar_extension()
        self.energy_ext=backend.energy_extension();self.affine=backend.affine(n)
        self.init_graph=None;self.iter_graph=None;self.cert_graph=None

    def K(self,u):return self.scalar.apply(self.op._material_field,u,self.active,self.n)

    def true_K(self,u):
        if self.batch==1:return self.op.apply(u)
        return torch.cat([self.op.apply(u[j:j+1]) for j in range(self.batch)],dim=0)

    def B_serial(self,r):
        v=pre_input(r,self.scale)
        z=self.op._unscaled_green(v) if self.batch==1 else torch.cat([self.op._unscaled_green(v[j:j+1]) for j in range(self.batch)],dim=0)
        return pre_output(z,self.scale)

    def B(self,r):
        if self.batch==1:return self.B_serial(r)
        v=pre_input(r,self.scale).reshape(3,self.n,self.n,self.n,3).permute(0,4,1,2,3).contiguous()
        spectrum=torch.fft.rfftn(v,dim=(-3,-2,-1))
        self.scalar.spectral(spectrum,self.op._contest_coefficient)
        # The frozen inverse symbol already contains 1/N; do not normalize again.
        z=torch.fft.irfftn(spectrum,s=(self.n,)*3,dim=(-3,-2,-1),norm='forward')
        z=z.permute(0,2,3,4,1).contiguous().reshape_as(r)
        return pre_output(z,self.scale)

    def initialize(self):
        self.active.copy_((self.r.square().sum(1)>self.target2)&(self.iterations<4000)&~self.bad)
        z=self.B(self.r)
        self.p.copy_(torch.where(self.active[:,None],z,0.))
        self.rz.copy_((self.r*z).sum(1))

    def iteration(self):
        q=self.K(self.p)
        u,r,active,it,bad=update_ur(self.u,self.r,self.p,q,self.rz,self.active,self.iterations,self.bad,self.target2)
        self.u.copy_(u);self.r.copy_(r);self.active.copy_(active)
        self.iterations.copy_(it);self.bad.copy_(bad)
        p,rz=update_p(self.r,self.B(self.r),self.p,self.rz,self.active)
        self.p.copy_(p);self.rz.copy_(rz)

    def certify(self):
        self.u.copy_(project(self.u))
        self.r.copy_(self.rhs-self.true_K(self.u))
        self.absolute.copy_(self.r.square().sum(1).sqrt())

    def capture(self):
        self.outer.capture()
        # Compile and allocate cuFFT workspaces before capture, on the plan's stream.
        self.r.normal_();self.target2.fill_(1e-20)
        self.initialize();self.iteration();self.certify();self.stream.synchronize()
        self.init_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.init_graph,stream=self.stream):self.initialize()
        self.iter_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.iter_graph,stream=self.stream):
            for _ in range(self.chunk):self.iteration()
        self.cert_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cert_graph,stream=self.stream):self.certify()
        self.stream.synchronize();self.u.zero_();self.solutions.zero_()

    def material(self,a):
        self.op._material_field.copy_(a)
        self.op._hybrid_coeff.copy_(a.reshape(-1,1)*self.b.packed0)

    def select(self,method):
        a=self.op._material_field
        if method=='green':self.alpha.zero_();self.scale.fill_(1.);return
        if method=='gj':self.alpha.fill_(1.)
        elif method=='asg':
            pred=self.b.net(self.b.extract(a)).reshape(())
            self.alpha.copy_(torch.where(pred<self.b.threshold,0.,pred))
        elif method=='compact8':
            f=self.b.law_features(a).to(torch.float64)
            area,jump,contrast,complexity=f.unbind()
            value=1.-.75*(area*jump).clamp_min(0).sqrt()/contrast.clamp_min(1e-30).sqrt()-(2.5/complexity.clamp_min(1e-30))**8
            value=torch.where((complexity>1e-12)&(contrast>1e-12),value.clamp(0.,1.),0.)
            self.alpha.copy_(torch.where(value<.375,0.,value))
        else:self.alpha.fill_(float(method))
        d=torch.zeros((*a.shape,3),device='cuda',dtype=torch.float64)
        for offset,element in zip(self.b.offsets,self.element_diag):
            d.add_(torch.roll(a,tuple(map(int,offset[::-1])),(0,1,2))[...,None]*element)
        q=d.reshape(1,-1).log();q=q-q.mean()
        self.scale.copy_((-.5*self.alpha*q).exp())

    def solve(self,rtol=1e-8):
        if self.batch==3:return self._solve_batch(range(3),rtol)
        infos=[];checks=0
        for load in range(3):
            self.u.copy_(self.solutions[load:load+1])
            info,count=self._solve_batch([load],rtol)
            self.solutions[load:load+1].copy_(self.u)
            infos.extend(info);checks+=count
        return infos,checks

    def _solve_batch(self,load_indices,rtol):
        self.iterations.zero_();self.bad.zero_()
        for j,load in enumerate(load_indices):
            self.rhs[j:j+1].copy_(self.hybrid.rhs(self.op._hybrid_coeff,self.op._hybrid_index,self.macros[load],*self.op.geometry(),self.op._hybrid_indirect))
        self.norm.copy_(self.rhs.square().sum(1).sqrt())
        self.target2.copy_(torch.maximum(self.norm*rtol,torch.full_like(self.norm,1e-13/math.sqrt(self.n))).square())
        restarts=0;host_checks=0
        self.cert_graph.replay()
        while True:
            self.init_graph.replay()
            # Latched device masks preserve each load's stopping iteration.
            while True:
                host_checks+=1
                if not bool(self.active.any().item()):break
                self.iter_graph.replay()
            self.cert_graph.replay()
            diagnostics=torch.stack((self.norm,self.absolute,self.target2.sqrt(),self.iterations.to(torch.float64),self.bad.to(torch.float64)),dim=1).cpu().numpy()
            ok=np.isfinite(diagnostics[:,1])&(diagnostics[:,1]<=diagnostics[:,2])&(diagnostics[:,4]==0)
            if ok.all() or restarts>=2 or np.any(diagnostics[:,4]) or np.any((~ok)&(diagnostics[:,3]>=4000)):break
            restarts+=1
        info=[dict(rhs_norm=float(v[0]),absolute_residual=float(v[1]),threshold=float(v[2]),iterations=int(v[3]),
                   relative_residual=float(v[1]/max(v[0],1e-300)),converged=bool(ok[j]),restarts=restarts) for j,v in enumerate(diagnostics)]
        if not ok.all():raise RuntimeError(('true residual certification failed',info))
        return info,host_checks

    def energy(self):
        e=self.energy_ext.evaluate(self.solutions,self.b.packed0,self.affine,self.n)
        return (e*self.op._material_field.reshape(1,-1)).mean(1),e.mean(0).reshape(self.n,self.n,self.n)
