"""GPU implementations of the CPU-verified formula reductions."""
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PIPE=Path(os.environ.get("FEAD_PIPELINE",str(__import__('project_paths').source_path('assembly_pipeline'))))
sys.path.insert(0,str(PIPE))
import pipeline_study as prior
import pipeline_adapter as pipeline


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ["PATH"]=str(Path(sys.executable).parent)+":/usr/local/cuda/bin:"+os.environ.get("PATH","")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST","12.0");os.environ.setdefault("MAX_JOBS","2")
    return load(name="fead_formula_20261020_v1",sources=[str(HERE/"formula_kernels.cu")],
        extra_include_paths=[str(PIPE)],extra_cuda_cflags=["-O3","--ptxas-options=-v"],extra_cflags=["-O3"],verbose=False)


def geometry(op):
    m=op.mesh
    return (m.Nz,m.Ny,m.Nx,m.dx,m.dy,m.dz)


def tables(op):
    result=[]
    for n,count in ((op.mesh.Nx,op.mesh.Nx//2+1),(op.mesh.Ny,op.mesh.Ny),(op.mesh.Nz,op.mesh.Nz)):
        theta=np.pi*np.arange(count,dtype=np.float64)/n
        result.append(torch.as_tensor(np.stack((np.sin(theta),np.cos(theta)),axis=-1),device=op.device).contiguous())
    return result


def enable_fi_reference(op,precision="mixed",layout="upper6"):
    if precision not in ("mixed","fp64"):raise ValueError(precision)
    reference=op._compute_reference_medium("zeman",verbose=False)[0]
    lam,mu=float(reference[0,1]),float(reference[3,3])
    expected=np.zeros((6,6));expected[:3,:3]=lam
    expected[np.arange(3),np.arange(3)]+=2*mu;expected[np.arange(3,6),np.arange(3,6)]=mu
    if not np.allclose(expected,reference,rtol=1e-14,atol=0):raise ValueError("Isotropic reference required")
    op._formula_tables=tables(op)
    op._formula_vtables=[]
    for n,count,h in ((op.mesh.Nx,op.mesh.Nx//2+1,op.mesh.dx),(op.mesh.Ny,op.mesh.Ny,op.mesh.dy),(op.mesh.Nz,op.mesh.Nz,op.mesh.dz)):
        theta=np.pi*np.arange(count,dtype=np.float64)/n
        s,cc=np.sin(theta),np.cos(theta)
        v=2*s*cc/(h*(cc*cc+s*s/3))
        op._formula_vtables.append(torch.as_tensor(v,device=op.device,dtype=torch.float64 if precision=="fp64" else torch.float32).contiguous())
    op._fft_symbol_inv=extension().fi_coeff(*op._formula_tables,*geometry(op),lam,mu,precision=="fp64")
    op._formula_reference_C0=reference.tolist()
    op._fft_enabled=True;op._fft_spectrum_mode="real_half";op._fft_symbol_layout="FI_diagonal_rank1"
    op._fft_inverse_norm_folded=False;op._fft_batch_chunk_size=None;op._M_inv_cache=None
    op._formula_precision=precision
    def apply(self,residual):
        dtype=torch.float32 if self._formula_precision=="mixed" else torch.float64
        shape=(self.mesh.Nz,self.mesh.Ny,self.mesh.Nx)
        field=residual.to(dtype).view(1,-1,3).transpose(1,2).contiguous().view(1,3,*shape)
        spectrum=torch.fft.rfftn(field,dim=(-3,-2,-1)).contiguous()
        values=extension().fi_apply(self._fft_symbol_inv,*self._formula_vtables,spectrum,*geometry(self))
        z=torch.fft.irfftn(values,s=shape,dim=(-3,-2,-1),norm="backward")
        return z.view(1,3,-1).transpose(1,2).contiguous().view_as(residual).to(residual.dtype)
    op.apply_precond=MethodType(apply,op)
    return op


def compact_hg(op,active,background):
    """Retain only contrast C21; certify with the FP64 physical split operator."""
    from femlib.c3d8r import hourglass_scale
    if op.B!=1 or op.dtype!=torch.float64:raise ValueError("B=1 FP64 required")
    if hasattr(op,"_analytic_reference_C0"):
        reference=np.asarray(op._analytic_reference_C0)
    else:
        reference=op._compute_reference_medium_from_packed("zeman",verbose=False)[0]
    active=torch.as_tensor(np.asarray(active,dtype=np.int32),device=op.device)
    pairs=[(i,j) for i in range(6) for j in range(i,6)]
    cb=torch.as_tensor(np.array([background[i,j] for i,j in pairs]),device=op.device,dtype=torch.float64)
    op._contrast_C=(op.C[0].index_select(0,active.long())-cb).contiguous()
    op._contrast_scale=(op.hg_scale[0].index_select(0,active.long())-
        hourglass_scale(background,(op.mesh.dx,op.mesh.dy,op.mesh.dz),op.hourglass_coefficient)).contiguous()
    op._contrast_active=active.contiguous()
    op._contrast_background=np.asarray(background).copy()
    op._contrast_tables=tables(op)
    voxel=(op.mesh.dx,op.mesh.dy,op.mesh.dz)
    op._contrast_coeff=extension().hg_coeff(*op._contrast_tables,*geometry(op),
        float(reference[0,1]),float(reference[3,3]),hourglass_scale(reference,voxel,op.hourglass_coefficient),
        float(background[0,1]),float(background[3,3]),hourglass_scale(background,voxel,op.hourglass_coefficient))
    op._contrast_C0=reference.tolist()
    op.C=None;op._C_voigt_gpu=None;op.hg_scale=None;op._modal_scales=None
    op._fft_symbol_inv=None
    op.__dict__.pop("_analytic_tables",None)
    def to_field(self,u):
        return u.view(1,-1,3).transpose(1,2).contiguous().view(1,3,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx)
    def to_flat(self,u):return u.view(1,3,-1).transpose(1,2).contiguous().view(1,-1)
    op.to_field=MethodType(to_field,op);op.to_flat=MethodType(to_flat,op)
    def project(self,value,mode=0):
        return extension().project(self._contrast_coeff,*self._contrast_tables,value,*geometry(self),mode)
    def delta(self,u):
        return extension().contrast(self._contrast_C,self._contrast_scale,self._contrast_active,u.contiguous(),*geometry(self))
    def apply(self,u):
        physical=self.to_field(u)
        spec=torch.fft.rfftn(physical,dim=(-3,-2,-1)).contiguous()
        base=torch.fft.irfftn(self.spectral_project(spec,2),s=physical.shape[-3:],dim=(-3,-2,-1),norm="backward")
        return self.to_flat(base+self.apply_contrast(physical))
    def rhs(self,macro):
        e=torch.as_tensor(np.asarray(macro),device=self.device,dtype=torch.float64)
        return extension().rhs(self._contrast_C,self._contrast_active,e,*geometry(self))
    def stress(self,u,macro):
        e=torch.as_tensor(np.asarray(macro),device=self.device,dtype=torch.float64)
        partial=extension().stress(self._contrast_C,self._contrast_active,e,u.contiguous(),*geometry(self))
        avg=partial.sum(dim=1).cpu().numpy()/self.mesh.n_elem+self._contrast_background@np.asarray(macro)
        return avg[None,:]
    def spectral_apply(self,y):
        shape=(self.mesh.Nz,self.mesh.Ny,self.mesh.Nx)
        u=torch.fft.irfftn(self.spectral_project(y),s=shape,dim=(-3,-2,-1),norm="backward")
        f=self.apply_contrast(u)
        spec=torch.fft.rfftn(f,dim=(-3,-2,-1)).contiguous()
        out,partial=extension().combine(self._contrast_coeff,*self._contrast_tables,y,spec,*geometry(self))
        return out,partial[0].sum().reshape(1)
    op.spectral_project=MethodType(project,op);op.apply_contrast=MethodType(delta,op)
    op.apply=MethodType(apply,op);op.compute_rhs=MethodType(rhs,op);op.compute_avg_stress=MethodType(stress,op)
    op.spectral_apply=MethodType(spectral_apply,op)
    return op


def spectral_solver(op,rhs,*,tol=1e-8,max_iter=10000,x0=None,convergence_check_stride=4,**kwargs):
    """Half-spectrum CG; FP64 physical fields and complex128 Krylov states.

    All spectral inner products use real-half Parseval weights. The library's
    forward transform is unnormalized; physical norm squares divide by N.
    Host checks are every four actual iterations. Final projected physical
    FP64 certification remains in benchmark.solve, including restarts.
    """
    shape=(op.mesh.Nz,op.mesh.Ny,op.mesh.Nx);n=op.mesh.n_elem
    field=op.to_field(rhs)
    if x0 is None:
        y=torch.zeros((1,3,shape[0],shape[1],shape[2]//2+1),device=op.device,dtype=torch.complex128)
        physical=field
    else:
        y=op.spectral_project(torch.fft.rfftn(op.to_field(x0),dim=(-3,-2,-1)).contiguous(),3)
        physical=op.to_field(rhs-op.apply(x0))
    r=op.spectral_project(torch.fft.rfftn(physical,dim=(-3,-2,-1)).contiguous())
    p=r.clone();dummy=torch.empty(0,device=op.device,dtype=torch.float64)
    parts=extension().update(op._contrast_coeff,*op._contrast_tables,y,r,p,p,dummy,dummy,*geometry(op),False)
    scalars=parts.sum(dim=1);rho=scalars[:1]
    normb=torch.linalg.vector_norm(rhs).clamp_min(1e-300)
    performed=0
    for it in range(max_iter):
        ap,pap=op.spectral_apply(p)
        parts=extension().update(op._contrast_coeff,*op._contrast_tables,y,r,p,ap,rho,pap,*geometry(op),True)
        next_scalars=parts.sum(dim=1)
        performed=it+1
        if performed%convergence_check_stride==0 or performed==max_iter:
            relative=torch.sqrt(next_scalars[1]/n)/normb
            if not bool(torch.isfinite(relative)):raise RuntimeError("Nonfinite spectral-CG residual")
            if float(relative)<=tol:break
        extension().direction(p,r,next_scalars[:1],rho)
        rho=next_scalars[:1]
    u=torch.fft.irfftn(op.spectral_project(y),s=shape,dim=(-3,-2,-1),norm="backward")
    return op.to_flat(u),performed,(torch.sqrt(next_scalars[1:2]/n)/normb)


def _single_update(u,r,p,s,gamma,denominator):
    alpha=gamma/denominator
    u=u+alpha[:,None]*p;r=r-alpha[:,None]*s
    return u,r,(r*r).sum(dim=1),alpha


def _single_directions(r,z,w,p,s,new_gamma,old_gamma,delta,alpha):
    beta=new_gamma/old_gamma
    denominator=delta-beta*new_gamma/alpha
    return z+beta[:,None]*p,w+beta[:,None]*s,denominator


@lru_cache(None)
def single_steps():
    options={"triton.cudagraphs":False}
    return torch.compile(_single_update,fullgraph=True,options=options),torch.compile(_single_directions,fullgraph=True,options=options)


def single_solver(op,rhs,*,tol=1e-8,max_iter=10000,x0=None,convergence_check_stride=4,**kwargs):
    update,directions=single_steps()
    u=torch.zeros_like(rhs) if x0 is None else x0.clone()
    r=rhs.clone() if x0 is None else rhs-op.apply(u)
    z=op.apply_precond(r);w,delta=op.apply_with_energy(z)
    gamma=(r*z).sum(dim=1);denominator=delta;p=z.clone();s=w
    normb=torch.linalg.vector_norm(rhs,dim=1).clamp_min(1e-300)
    for it in range(max_iter):
        u,r,rr,alpha=update(u,r,p,s,gamma,denominator)
        if (it+1)%convergence_check_stride==0 or it+1==max_iter:
            relative=torch.sqrt(rr)/normb
            if not bool(torch.isfinite(relative).all()):raise RuntimeError("Nonfinite single-reduction residual")
            if float(relative.max())<=tol:break
        z=op.apply_precond(r);w,delta=op.apply_with_energy(z)
        new_gamma=(r*z).sum(dim=1)
        p,s,denominator=directions(r,z,w,p,s,new_gamma,gamma,delta,alpha)
        gamma=new_gamma
    return u,it+1,torch.sqrt(rr)/normb


def configure(b,kind,variant):
    pipeline.configure(b,prior.modal,kind,"combined" if kind=="hg" else "modal")
    previous_build=b.build
    if kind=="fi" and variant=="analytic":
        import fi_benchmark
        fi_benchmark.original_reference=enable_fi_reference
    if kind=="hg" and variant=="contrast64":b.solve_pcg_batched=spectral_solver
    if kind=="hg" and variant=="single":b.solve_pcg_batched=single_solver
    def build(data,material,route,precision,stages):
        op=previous_build(data,material,route,precision,stages)
        if kind=="hg" and variant=="contrast64":
            from femlib.materials import isotropic_to_voigt
            matrix=material["matrix"]
            background=isotropic_to_voigt(matrix["E"],matrix["Nu"])
            active=np.flatnonzero(np.asarray(data["yarn_id"]).reshape(-1)>=0)
            compact_hg(op,active,background)
        stages.end("formula_attachment",op)
        return op
    b.build=build
