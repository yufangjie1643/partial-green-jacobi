"""Li2026 design problems on the existing full-integration HEX8/FFT-PCG solver.

Design variables and the analysis grid coincide here. This reproduces the
paper's material-design problems with our method, not its HEX20 sub-cell
discretization. All physical lengths are normalized by the unit-cell edge.
"""
from pathlib import Path
from functools import lru_cache
import math
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'asg_gpu_resident_20261103'))
sys.path.insert(0,str(HERE.parent/'asg_scenarios_20261105'))
from resident import Resident, Candidates, pilot
from scenarios import energy_extension


def initial(n,field):
    # MATLAB indices are 1-based. Output axes are z,y,x in the solver.
    z,y,x=np.meshgrid(*([np.arange(1,n+1,dtype=float)]*3),indexing='ij')
    if field==1:centers=[(n+1)/2];radius=n/3
    elif field==2:centers=[1,(n+1)/2,n];radius=math.floor(n/5)
    elif field==3:
        d=math.ceil((n-1)/6);centers=[1,(n+1)/2-d,(n+1)/2+d,n];radius=math.floor(n/7)
    elif field==4:centers=np.linspace(1,n,5);radius=math.floor(n/12+.5)
    else:raise ValueError(field)
    dx=np.min((x[...,None]-centers)**2,axis=-1)
    dy=np.min((y[...,None]-centers)**2,axis=-1)
    dz=np.min((z[...,None]-centers)**2,axis=-1)
    out=np.where(dx+dy+dz<radius**2,.001,.4)
    assert all(np.array_equal(out,np.flip(out,a)) for a in range(3))
    return out


def stencil(rmin):
    radius=math.ceil(rmin)-1;offsets=[];weights=[]
    for dz in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                w=rmin-math.sqrt(dx*dx+dy*dy+dz*dz)
                if w>0:offsets.append([dx,dy,dz]);weights.append(w)
    return np.array(offsets,dtype=np.int32),np.array(weights)


@lru_cache(None)
def extension(kind):
    from torch.utils.cpp_extension import load
    return load(name='sctop_reproduction_'+kind+'_20261105',sources=[str(HERE/(kind+'.cu'))],
                extra_include_paths=[str(ROOT/'solver/src/materials')],
                extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False)


class PaperBackend(Candidates):
    def __init__(self):
        super().__init__()
        lam=15/26;mu=5/13
        self.C0=np.zeros((6,6));self.C0[:3,:3]=lam
        self.C0[np.arange(3),np.arange(3)]+=2*mu
        self.C0[np.arange(3,6),np.arange(3,6)]=mu
        state=self.frozen.build.__globals__['s'];state.C0=self.C0
        state.element_diagonal.__globals__['C0']=self.C0
        self.packed0=torch.tensor([self.C0[i,j] for i in range(6) for j in range(i,6)],device='cuda',dtype=torch.float64)
        self.loads=np.eye(6);self.affine.cache_clear()


class PaperOuter:
    def __init__(self,n,volume,rmin=3.,beta=4.,eta=.5,emin=1e-9):
        self.n=n;self.volume=volume;self.beta=beta;self.eta=eta;self.emin=emin
        self.stream=torch.cuda.current_stream();self.ext=extension('outer')
        offsets,weights=stencil(rmin)
        self.offsets=torch.tensor(offsets,device='cuda');self.weights=torch.tensor(weights,device='cuda')
        for name in ['x','ph','dp','a','energy','new','g','dv','ratio','t','hs','trial_ph','trial_dp','trial_a']:
            setattr(self,name,torch.empty((n,n,n),device='cuda',dtype=torch.float64))
        self.bounds=torch.zeros(4,device='cuda',dtype=torch.float64)
        self.change=torch.zeros(1,device='cuda',dtype=torch.float64)
        self.raw_filter(torch.ones_like(self.x),self.hs)
        self.physical_graph=self.update_graph=None

    def raw_filter(self,x,out,conditional=False):
        self.ext.filter(x.contiguous(),self.offsets,self.weights,out,self.bounds,self.n,conditional)

    def project(self,t,ph,dp,a,conditional=False):
        self.ext.project(t,self.hs,ph,dp,a,self.bounds,self.beta,self.eta,self.emin,conditional)

    def physical(self):
        self.raw_filter(self.x,self.t);self.project(self.t,self.ph,self.dp,self.a)

    def update(self):
        self.raw_filter(self.dp*(3*(1-self.emin)*self.ph.square())*self.energy/self.hs,self.g)
        self.g.div_(self.n**3)
        self.raw_filter(self.dp/self.hs,self.dv)
        self.ratio.copy_(-self.g/self.dv)
        self.bounds[0].zero_();self.bounds[1].fill_(1e9);self.bounds[2].fill_(1.);self.bounds[3].zero_()
        # Fixed graph capacity; device condition freezes all OC state exactly
        # when the source MATLAB bisection stopping condition is satisfied.
        for _ in range(180):
            self.ext.candidate(self.x,self.ratio,self.bounds,self.new)
            self.raw_filter(self.new,self.t,True)
            self.project(self.t,self.trial_ph,self.trial_dp,self.trial_a,True)
            self.ext.bisect(self.trial_ph.mean().reshape(1),self.bounds,self.volume)
        self.change.copy_((self.new-self.x).abs().max())

    def capture(self):
        self.x.fill_(.4);self.energy.fill_(-1.)
        self.physical();self.update();self.stream.synchronize()
        self.physical_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.physical_graph,stream=self.stream):self.physical()
        self.update_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.update_graph,stream=self.stream):self.update()
        self.stream.synchronize()

    def set_initial(self,x):self.x.copy_(torch.as_tensor(x,device='cuda',dtype=torch.float64))
    def evaluate(self):self.physical_graph.replay()
    def step(self,e):self.energy.copy_(e);self.update_graph.replay()
    def advance(self):self.x.copy_(self.new)


class PaperResident(Resident):
    def __init__(self,backend,n,objective,volume,rmin=3.,beta=4.,eta=.5,emin=1e-9,load_mode='parallel',chunk=4):
        if load_mode not in ['serial','parallel']:raise ValueError(load_mode)
        self.load_mode=load_mode;self.batch=1 if load_mode=='serial' else 3
        self.b=backend;self.n=n;self.chunk=chunk;self.objective_kind=objective
        self.stream=torch.cuda.current_stream()
        self.op=backend.build(np.ones((n,n,n)))
        self.outer=PaperOuter(n,volume,rmin,beta,eta,emin)
        self.u=torch.zeros((self.batch,3*n**3),device='cuda',dtype=torch.float64)
        self.solutions=torch.zeros((6,3*n**3),device='cuda',dtype=torch.float64)
        for name in ['r','p','rhs']:setattr(self,name,torch.zeros_like(self.u))
        for name in ['rz','target2','norm','absolute','iterations']:
            setattr(self,name,torch.zeros(self.batch,device='cuda',dtype=torch.int64 if name=='iterations' else torch.float64))
        self.active=torch.zeros(self.batch,device='cuda',dtype=torch.bool);self.bad=torch.zeros_like(self.active)
        self.scale=torch.ones((1,3*n**3),device='cuda',dtype=torch.float64)
        self.alpha=torch.zeros((),device='cuda',dtype=torch.float64)
        self.macros=torch.tensor(backend.loads,device='cuda',dtype=torch.float64)
        ed=backend.frozen.diagonal.__globals__['s'].element_diagonal(self.op._voxel)
        self.element_diag=torch.tensor(ed,device='cuda',dtype=torch.float64)
        self.hybrid=backend.hybrid.extension();self.scalar=extension('scalar_fi')
        self.energy_ext=energy_extension();self.affine=backend.affine(n)
        self.init_graph=self.iter_graph=self.cert_graph=None

    def solve(self,rtol=1e-8):
        infos=[];checks=0
        for start in range(0,6,self.batch):
            self.u.copy_(self.solutions[start:start+self.batch])
            info,count=self._solve_batch(range(start,start+self.batch),rtol)
            self.solutions[start:start+self.batch].copy_(self.u)
            infos.extend(info);checks+=count
        return infos,checks

    def energy(self,iteration):
        energies=self.energy_ext.evaluate(self.solutions,self.b.packed0,self.affine,self.n)
        diagonal=(energies*self.op._material_field.reshape(1,-1)).mean(1)
        if self.objective_kind=='shear':local=-energies[3:].sum(0)
        else:
            hydro=self.energy_ext.evaluate(self.solutions[:3].sum(0,keepdim=True),self.b.packed0,
                                           self.affine[:3].sum(0,keepdim=True),self.n)[0]
            if self.objective_kind=='bulk':local=-hydro
            elif self.objective_kind=='auxetic':local=hydro-(1+.8**iteration)*energies[:3].sum(0)
            else:raise ValueError(self.objective_kind)
        objective=(local*self.op._material_field.reshape(-1)).mean()
        return objective,local.reshape(self.n,self.n,self.n),diagonal

    def effective_tensor(self):
        e=self.energy_ext.evaluate(self.solutions,self.b.packed0,self.affine,self.n)
        d=(e*self.op._material_field.reshape(1,-1)).mean(1)
        C=torch.diag(d)
        for i in range(6):
            for j in range(i+1,6):
                pair=self.energy_ext.evaluate(self.solutions[i:i+1]+self.solutions[j:j+1],self.b.packed0,
                                              self.affine[i:i+1]+self.affine[j:j+1],self.n)
                q=(pair*self.op._material_field.reshape(1,-1)).mean()
                C[i,j]=C[j,i]=.5*(q-d[i]-d[j])
        return C
