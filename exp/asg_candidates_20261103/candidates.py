"""Concrete, fixed SPD prototypes for the registered candidate screen."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.ndimage import label

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'asg_topopt_20261103'
sys.path.insert(0,str(PILOT))
from backend import Backend, definitions, module, SOLVER, ARCHIVE
import study as pilot

SPLITS=[(0.,.5),(0.,1.),(.5,0.),(.5,1.),(1.,0.),(1.,.5),(.75,.25),(.25,.75)]
METHODS=['green','gj','asg','0.25','0.5','0.69','0.75','compact4','compact8',
         'mlp_set','mlp_large','contrast_only','one_point','quantized2','quantized5',
         'block_0.5','block_1']+[f'split_{a:g}_{b:g}' for a,b in SPLITS]+[
         'direction_0','direction_0.5','direction_1','coarse4','coarse8','balanced4','regions']


def tensor(a):return torch.as_tensor(a,device='cuda',dtype=torch.float64)


@lru_cache(None)
def element_geometry(n):
    """Exact Q1 solid element stiffness, independent of frozen modal kernel."""
    offsets=np.array([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)])
    q=np.array([[x,y,z] for z in (-1,1) for y in (-1,1) for x in (-1,1)])/(2*np.sqrt(3))+.5
    grad=[]
    for p in q:
        g=[]
        for v in offsets:
            factors=np.where(v,p,1-p)
            g.append([(2*v[d]-1)*np.prod(np.delete(factors,d))*n for d in range(3)])
        grad.append(g)
    return offsets,np.asarray(grad)


def stiffness(grad,C,volume):
    shape=grad.shape[:-2]
    B=np.zeros((*shape,6,24))
    for a in range(8):
        dx,dy,dz=np.moveaxis(grad[...,a,:],-1,0)
        B[...,0,3*a]=dx;B[...,1,3*a+1]=dy;B[...,2,3*a+2]=dz
        B[...,3,3*a+1]=dz;B[...,3,3*a+2]=dy
        B[...,4,3*a]=dz;B[...,4,3*a+2]=dx
        B[...,5,3*a]=dy;B[...,5,3*a+1]=dx
    return np.einsum('...qsi,st,...qtj->...ij',B,C,B,optimize=True)*volume/8


class Galerkin:
    """Periodic trilinear Galerkin coarse space, exact heterogeneous integration."""
    def __init__(self,b,n,h):
        assert n%h==0 and h>=2
        self.n,self.h,self.r=n,h,n//h
        r=self.r;ndof=3*h**3
        off,qgrad=element_geometry(n)
        # Coarse Q1 functions restricted to each fine cell; exact 2x2x2 quadrature.
        sub=np.array([[x,y,z] for z in range(r) for y in range(r) for x in range(r)])
        q=np.array([[x,y,z] for z in (-1,1) for y in (-1,1) for x in (-1,1)])/(2*np.sqrt(3))+.5
        p=(sub[:,None,:]+q[None,:,:])/r
        grad=np.empty((r**3,8,8,3))
        for a,v in enumerate(off):
            factors=np.where(v,p,1-p)
            for d in range(3):grad[:,:,a,d]=(2*v[d]-1)*np.prod(np.delete(factors,d,axis=-1),axis=-1)*h
        self.weights=tensor(stiffness(grad,b.C0,1/n**3).reshape(r**3,576))
        z,y,x=np.meshgrid(*([np.arange(h)]*3),indexing='ij')
        nodes=(((z.ravel()[:,None]+off[:,2])%h*h+(y.ravel()[:,None]+off[:,1])%h)*h+(x.ravel()[:,None]+off[:,0])%h)
        ids=(3*nodes[:,:,None]+np.arange(3)).reshape(h**3,24)
        self.assembly=torch.tensor((ids[:,:,None]*ndof+ids[:,None,:]).ravel(),device='cuda')
        self.matrix=torch.empty((ndof,ndof),device='cuda',dtype=torch.float64)
        T=np.zeros((n,h));pos=np.arange(n)/r;lo=np.floor(pos).astype(int);f=pos-lo
        T[np.arange(n),lo]=1-f;T[np.arange(n),(lo+1)%h]+=f
        self.T=tensor(T)

    def update(self,a):
        h,r=self.h,self.r
        local=a.reshape(h,r,h,r,h,r).permute(0,2,4,1,3,5).reshape(h**3,r**3)@self.weights
        self.matrix.zero_().view(-1).index_add_(0,self.assembly,local.reshape(-1))
        # Pin one node's translations. Project the returned fine displacement.
        self.L=torch.linalg.cholesky(self.matrix[3:,3:].contiguous())

    def prolong(self,c):
        h=self.h;T=self.T
        c=c.reshape(h,h,h,3)
        c=torch.einsum('zh,hyxc->zyxc',T,c)
        c=torch.einsum('yh,zhxc->zyxc',T,c)
        return torch.einsum('xh,zyhc->zyxc',T,c).reshape(1,-1)

    def restrict(self,r):
        n=self.n;T=self.T
        r=r.reshape(n,n,n,3)
        r=torch.einsum('xh,zyxc->zyhc',T,r)
        r=torch.einsum('yh,zyxc->zhxc',T,r)
        return torch.einsum('zh,zyxc->hyxc',T,r).reshape(-1,1)

    def apply(self,r):
        coarse=self.restrict(r)
        sol=torch.cholesky_solve(coarse[3:].contiguous(),self.L)
        c=torch.cat((torch.zeros((3,1),device=r.device,dtype=r.dtype),sol),dim=0)
        u=self.prolong(c)
        v=u.reshape(-1,3)
        return (v-v.mean(0)).reshape_as(u)


class Candidates(Backend):
    def __init__(self):
        super().__init__()
        self.law=module('candidate_law',SOLVER/'experiments/global_alpha_compact_law/law.py')
        self.law_features=torch.compile(self.law.extract_law,fullgraph=True)
        models=definitions('experiments/asg_independent/train.py',['Model','load_model','subset_features'],dict(math=math))
        self.freeze=json.loads((ARCHIVE/'MODELS_FROZEN.json').read_text())
        self.nets={k:models.load_model(ARCHIVE,v['name']) for k,v in self.freeze['selected'].items()}
        self.subset={k:torch.compile(lambda a,mode=k:models.subset_features(a,mode),fullgraph=True) for k in ('contrast','one_point')}
        self.mesh_cache={}

    def build(self,a):
        op=super().build(a)
        op._baseline_coefficient=op._contest_coefficient.clone()
        op._direction_active=False
        return op

    def mesh(self,n):
        """Geometry-only preparation shared over a design task, reported separately."""
        if n in self.mesh_cache:return self.mesh_cache[n]
        start=pilot.sync()
        off,grad=element_geometry(n);Ke=stiffness(grad,self.C0,1/n**3)
        blocks=np.array([Ke[3*a:3*a+3,3*a:3*a+3] for a in range(8)])
        fq=np.fft.fftfreq(n)*n;fx=np.fft.rfftfreq(n)*n
        k2=fq[:,None,None]**2+fq[None,:,None]**2+fx[None,None,:]**2
        lowpass=tensor(np.exp(-2*np.pi**2*k2/16**2))
        m=dict(off=off,Ke=Ke,blocks=tensor(blocks),lowpass=lowpass,coarse={},coarse_setup_s={})
        m['setup_s']=pilot.sync()-start
        self.mesh_cache[n]=m
        return m

    def coarse(self,n,h):
        m=self.mesh(n)
        if h not in m['coarse']:
            start=pilot.sync();m['coarse'][h]=Galerkin(self,n,h)
            m['coarse_setup_s'][h]=pilot.sync()-start
        return m['coarse'][h]

    def scalar(self,op,psi):
        scale=torch.exp(-.5*psi).reshape(1,-1)
        green=op._unscaled_green;P=op.project_zero_mean
        op.apply_precond=lambda r:P(scale*green(scale*P(r)))

    def ell(self,op):
        if op._log_diagonal is None:
            q=self.frozen.diagonal(op).log();op._log_diagonal=q-q.mean()
        return op._log_diagonal

    def block(self,op,alpha):
        n=op.mesh.Nx;m=self.mesh(n);a=op._material_field
        D=torch.zeros((n,n,n,3,3),device='cuda',dtype=torch.float64)
        for v,block in zip(m['off'],m['blocks']):
            D+=torch.roll(a,tuple(map(int,v[::-1])),(0,1,2))[...,None,None]*block
        w,v=torch.linalg.eigh(D)
        logs=w.log();powers=torch.exp(-.5*alpha*(logs-logs.mean()))
        scale=(v*powers[...,None,:])@v.transpose(-1,-2)
        scale=scale.reshape(-1,3,3)
        def multiply(r):return torch.bmm(scale,r.reshape(-1,3,1)).reshape_as(r)
        P=op.project_zero_mean;G=op._unscaled_green
        op.apply_precond=lambda r:P(multiply(G(multiply(P(r)))))

    def directional(self,op,alpha):
        n=op.mesh.Nx;a=op._material_field
        axis=torch.stack([a.reciprocal().mean(d).reciprocal().mean() for d in (2,1,0)])
        axis=axis/axis.log().mean().exp()
        hx,hy,hz=axis.cpu().numpy()
        factors=np.sqrt(np.array([hx,hy,hz,np.sqrt(hy*hz),np.sqrt(hx*hz),np.sqrt(hx*hy)]))
        C=factors[:,None]*self.C0*factors[None,:]
        off,grad=element_geometry(n);Ke=stiffness(grad,C,1/n**3).reshape(8,3,8,3)
        stencil=np.zeros((3,3,n,n,n))
        for i,u in enumerate(off):
            for j,v in enumerate(off):
                dx,dy,dz=(v-u)%n;stencil[:,:,dz,dy,dx]+=Ke[i,:,j,:]
        spectrum=torch.fft.rfftn(tensor(stencil),dim=(-3,-2,-1)).real.permute(2,3,4,0,1).contiguous()
        spectrum[0,0,0]=torch.eye(3,device='cuda',dtype=torch.float64)
        inv=torch.linalg.inv(spectrum);inv[0,0,0]=0
        coeff=torch.stack([inv[...,i,j].reshape(-1) for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))])/n**3
        op._contest_coefficient.copy_(coeff.reshape_as(op._contest_coefficient))
        op._direction_active=True
        super().select(op,str(alpha))
        return dict(axis_conductances=[float(hx),float(hy),float(hz)])

    def regions(self,op):
        a=op._material_field.cpu().numpy();n=a.shape[0];labels,count=label(a>=.05)
        parent=np.arange(count+1)
        def root(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for axis in range(3):
            left=np.take(labels,0,axis).ravel();right=np.take(labels,-1,axis).ravel()
            pairs=np.unique(np.stack([left,right],axis=1),axis=0)
            for i,j in pairs:
                if i and j:parent[root(int(i))]=root(int(j))
        remap=np.array([root(i) for i in range(count+1)]);labels=remap[labels]
        sizes=np.bincount(labels.ravel());sizes[0]=0
        keep=[int(i) for i in np.argsort(sizes)[::-1][:6] if sizes[i]>0]
        columns=[]
        for i in keep:
            mask=(labels==i).astype(float)
            nodal=sum(np.roll(mask,tuple(map(int,v[::-1])),axis=(0,1,2)) for v in self.mesh(n)['off'])/8
            for d in range(3):
                z=np.zeros((n**3,3));z[:,d]=nodal.ravel();z-=z.mean(0);columns.append(z.ravel())
        if not columns:return None,0
        Z=tensor(np.stack(columns,axis=1))
        # Remove algebraic dependence before forming the exact Galerkin matrix.
        gram=Z.T@Z;w,v=torch.linalg.eigh(gram);good=w>float(w.max())*1e-10
        Z=(Z@v[:,good])/w[good].sqrt()[None,:]
        if Z.shape[1]==0:return None,0
        KZ=torch.cat([op.apply(z[None].contiguous()) for z in Z.T],dim=0).T.contiguous()
        E=Z.T@KZ;E=(E+E.T)/2;L=torch.linalg.cholesky(E)
        base=op.apply_precond;P=op.project_zero_mean
        def solve(v):return torch.cholesky_solve(v,L)
        def apply(r):
            r=P(r);c=solve(Z.T@r.T)
            z=base((r.T-KZ@c).T.contiguous())
            return P((Z@c+z.T-Z@solve(KZ.T@z.T)).T.contiguous())
        return apply,Z.shape[1]

    def select(self,op,method):
        self.initializer=None;self.last_metadata={};n=op.mesh.Nx
        if op._direction_active:
            op._contest_coefficient.copy_(op._baseline_coefficient);op._direction_active=False
        if method in ('green','gj','asg') or method.replace('.','',1).isdigit():
            return super().select(op,method)
        if method in ('compact4','compact8'):
            p=int(method[-1]);alpha=self.law.evaluate(self.law_features(op._material_field).cpu(),self.law.CONFIGS[f'law{p}'])
            alpha=0. if alpha<({4:.25,8:.375}[p]) else alpha
            return super().select(op,str(alpha))
        if method in self.nets or method.startswith('quantized'):
            name='mlp_mse' if method.startswith('quantized') else method
            net=self.nets[name];mode=net.config['inputs']
            features=self.extract(op._material_field) if mode=='full' else self.subset[mode](op._material_field)
            raw=float(net(features).reshape(()));alpha=raw if raw>=self.freeze['selected'][name]['threshold'] else 0.
            if method=='quantized2':alpha=float(alpha>=.5)
            if method=='quantized5':alpha=float(np.floor(4*alpha+.5)/4)
            return super().select(op,str(alpha))
        if method.startswith('block_'):
            alpha=float(method.split('_')[1]);self.block(op,alpha);return alpha
        if method.startswith('split_'):
            _,aa,bb=method.split('_');aa,bb=float(aa),float(bb)
            ell=self.ell(op).reshape(n,n,n,3).permute(3,0,1,2)
            g=torch.fft.irfftn(torch.fft.rfftn(ell,dim=(-3,-2,-1))*self.mesh(n)['lowpass'],s=(n,n,n),dim=(-3,-2,-1))
            psi=(aa*g+bb*(ell-g)).permute(1,2,3,0).contiguous()
            self.scalar(op,psi);self.last_metadata=dict(a=aa,b=bb);return None
        if method.startswith('direction_'):
            alpha=float(method.split('_')[1]);self.last_metadata=self.directional(op,alpha);return alpha
        if method.startswith('coarse') or method=='balanced4':
            h=int(method[-1]);coarse=self.coarse(n,h);coarse.update(op._material_field)
            super().select(op,'0.5');base=op.apply_precond;P=op.project_zero_mean
            if method.startswith('coarse'):
                self.initializer=lambda rhs,x:P(x+coarse.apply(rhs-op.apply(x)))
            else:
                def apply(r):
                    r=P(r);q=coarse.apply(r);z=base(r-op.apply(q))
                    return P(q+z-coarse.apply(op.apply(z)))
                op.apply_precond=apply
            self.last_metadata=dict(coarse_n=h,coarse_dof=3*h**3-3)
            return .5
        if method=='regions':
            super().select(op,'0.5');apply,rank=self.regions(op)
            if apply is not None:op.apply_precond=apply
            self.last_metadata=dict(coarse_rank=rank);return .5
        raise ValueError(method)

    def evaluate(self,op,a,method,previous=None,rtol=1e-8):
        start=pilot.sync();self.update(op,a);loaded=pilot.sync()
        alpha=self.select(op,method);selected=pilot.sync()
        us=[];infos=[];init_s=0.
        for j,load in enumerate(self.loads):
            rhs=op.compute_rhs(load);x=None if previous is None else previous[j]
            if self.initializer is not None:
                t=pilot.sync();x=self.initializer(rhs,torch.zeros_like(rhs) if x is None else x);init_s+=pilot.sync()-t
            u,info=self.solve(op,rhs,x,rtol);us.append(u);infos.append(info)
        solved=pilot.sync();energy=self.energies(op,us)
        moduli=(energy*op._material_field.reshape(1,-1)).mean(1).cpu().numpy()
        sensitivity=energy.mean(0).reshape(a.shape).cpu().numpy();ended=pilot.sync()
        row=dict(method=method,alpha=alpha,J=float(moduli.mean()),moduli=moduli.tolist(),
                 material_s=loaded-start,selection_precond_s=selected-loaded,solve_s=solved-selected,
                 initial_correction_s=init_s,response_s=ended-solved,total_s=ended-start,
                 iterations=sum(i['iterations'] for i in infos),loads=infos,
                 converged=all(i['converged'] for i in infos),metadata=self.last_metadata)
        return row,us,sensitivity
