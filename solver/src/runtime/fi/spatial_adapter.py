"""Bound FI assembly workspace without altering the frozen operator kernels.

Only used for the CUDA FI benchmark. Connectivity is implicit in those kernels.
The original packed21-to-14tap contraction is evaluated in spatial z slabs.
"""
from types import MethodType
import numpy as np
import torch
from femlib.batched import stencil_operator as so
from femlib.fem_batched import FEMHomogenizerBatched
VALIDATION_SLAB_LIMIT=None

def slab_assembly(self, C_fields, B_np, detJ):
    assert self.element_type=='c3d8' and self._material_layout=='packed_element_major'
    B,Nz,Ny,Nx=self.B,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx
    slab=max(1,min(Nz,524288//(B*Ny*Nx)))
    if VALIDATION_SLAB_LIMIT is not None:slab=min(slab,VALIDATION_SLAB_LIMIT)
    C=self.C.view(B,Nz,Ny,Nx,21)
    Bmat=torch.as_tensor(np.ascontiguousarray(B_np),device=self.device,dtype=self.dtype)
    Bap=Bmat.reshape(8,6,8,3)
    T=torch.einsum('gIap,gJbq->apbqIJ',Bap,Bap)*detJ
    offsets=np.asarray(so.LOCAL_OFFSET,dtype=np.int64)
    S=torch.zeros((B,14,9,Nz,Ny,Nx),device=self.device,dtype=self.dtype)
    for a in range(8):
        ox,oy,oz=map(int,offsets[a])
        d=offsets-offsets[a]
        taps=(d[:,2]+1)*9+(d[:,1]+1)*3+d[:,0]+1
        selected=np.flatnonzero(taps<=13)
        Ta=T[a].permute(1,0,2,3,4).contiguous()
        packed=torch.stack([Ta[...,i,j] if i==j else Ta[...,i,j]+Ta[...,j,i]
                            for i in range(6)for j in range(i,6)],-1)
        selected_t=torch.as_tensor(selected,device=self.device)
        packed=packed.index_select(0,selected_t)
        tap_t=torch.as_tensor(taps[selected],device=self.device)
        for z0 in range(0,Nz,slab):
            z1=min(z0+slab,Nz)
            z=(torch.arange(z0,z1,device=self.device)-oz)%Nz
            rolled=C.index_select(1,z)
            if ox or oy:
                rolled=torch.roll(rolled,shifts=(oy,ox),dims=(2,3))
            Ke=torch.einsum('npqk,bzyxk->bnpqzyx',packed,rolled)
            Ke=Ke.reshape(B,len(selected),9,z1-z0,Ny,Nx)
            S[:,:,:,z0:z1].index_add_(1,tap_t,Ke)
            del Ke,rolled,z
    self.S_flat=S
    diag=S[:,13][:,[0,4,8]]
    diag=torch.where(diag.abs()>1e-30,diag,torch.ones_like(diag))
    self._diag_cache=diag.permute(0,2,3,4,1).reshape(B,-1).contiguous()
    self._assembly_slab_depth=slab

def forbidden_torch_path(self,*args,**kwargs):
    raise RuntimeError('Spatial adapter is CUDA-only: explicit connectivity was omitted')

def make_solver(C,grid,voxel=(1.,1.,1.),spectrum='complex_full'):
    original_build=so.BatchedStencilOperator._build_stencils
    original_connectivity=so.build_dof_table
    # These arrays are unused by native CUDA FI apply/RHS/mean stress kernels.
    so.build_dof_table=lambda *_:(np.empty((0,8),dtype=np.int64),np.empty((0,24),dtype=np.int64))
    so.BatchedStencilOperator._build_stencils=slab_assembly
    try:
        solver=FEMHomogenizerBatched(C,voxel,grid,dtype=torch.float64,element_type='C3D8')
    finally:
        so.build_dof_table=original_connectivity
        so.BatchedStencilOperator._build_stencils=original_build
    op=solver.op
    op.compute_rhs_torch=MethodType(forbidden_torch_path,op)
    op.compute_avg_stress_torch=MethodType(forbidden_torch_path,op)
    op.set_apply_backend('cuda',cuda_variant='thread',cuda_block_size=128)
    if spectrum!='complex_full':
        original_fft=op._build_fft_symbol_inv
        def build_fft(S0_center,**kwargs):
            kwargs['spectrum_mode']=spectrum
            return original_fft(S0_center,**kwargs)
        op._build_fft_symbol_inv=build_fft
    return solver

def validate():
    global VALIDATION_SLAB_LIMIT
    VALIDATION_SLAB_LIMIT=2
    torch.manual_seed(100215)
    grid=(5,6,7);voxel=(.3,.5,.7);n=int(np.prod(grid));B=2
    A=torch.randn((B,n,6,6),dtype=torch.float64,device='cuda')
    full=A.transpose(-1,-2)@A+torch.diag(torch.tensor([1.,2.,4.,8.,16.,32.],device='cuda',dtype=torch.float64))
    C=torch.stack([full[:,:,i,j]for i in range(6)for j in range(i,6)],-1)
    original=FEMHomogenizerBatched(C,voxel,grid,dtype=torch.float64,element_type='C3D8')
    original.op.set_apply_backend('cuda',cuda_variant='thread',cuda_block_size=128)
    original.enable_fft_precond(ref_strategy='zeman')
    x=torch.randn((B,3*n),device='cuda',dtype=torch.float64)
    relative=lambda a,b:float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-300))
    results={}
    for spectrum in ['complex_full','real_half']:
        candidate=make_solver(C,grid,voxel,spectrum)
        candidate.enable_fft_precond(ref_strategy='zeman')
        a,b=candidate.op,original.op
        row=dict(slab_depth=a._assembly_slab_depth,stencil=relative(a.S_flat,b.S_flat),Ku=relative(a.apply(x),b.apply(x)),
                 preconditioner=relative(a.apply_precond(x),b.apply_precond(x)),rhs=[],stress=[])
        for j in range(6):
            e=np.eye(6)[j]
            row['rhs'].append(relative(a.compute_rhs(e),b.compute_rhs(e)))
            sa,sb=a.compute_avg_stress(x,e),b.compute_avg_stress(x,e)
            row['stress'].append(float(np.linalg.norm(sa-sb)/np.linalg.norm(sb)))
        assert max(row['stencil'],row['Ku'],row['preconditioner'],*row['rhs'],*row['stress'])<1e-11,row
        results[spectrum]=row
        del candidate,a,b
    VALIDATION_SLAB_LIMIT=None
    return dict(setting='two heterogeneous full-anisotropy random SPD fields; 5x6x7 with depth-2 slabs and partial final slab; unequal voxel edges; six macro strains',results=results)
