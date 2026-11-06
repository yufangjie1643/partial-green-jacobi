from functools import lru_cache
from pathlib import Path
from types import MethodType
import torch
from grouped_adapter import make as prior_make
from femlib.batched import stencil_operator as so
from femlib.fem_batched import FEMHomogenizerBatched
from fi_reference import enable_reference
from memory_adapter import no_stencil,compute_rhs
import numpy as np

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='fead_fi_atomic_20261016',sources=[str(Path(__file__).with_name('atomic_kernels.cu'))],
                extra_cuda_cflags=['-O3','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=False)

def apply(self,u):
    self._validate_apply_input(u)
    return extension().apply(self.C,self.B_mat,u,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx,self.detJ,self._atomic_block)

def make(C,grid,variant,voxel=(1.,1.,1.)):
    if not variant.startswith('atomic'):return prior_make(C,grid,variant,voxel)
    # Skip both the stencil and the old direct solver's stress scratch array.
    old_conn,old_build=so.build_dof_table,so.BatchedStencilOperator._build_stencils
    so.build_dof_table=lambda *_:(np.empty((0,8),np.int64),np.empty((0,24),np.int64))
    so.BatchedStencilOperator._build_stencils=no_stencil
    try:solver=FEMHomogenizerBatched(C,voxel,grid,dtype=torch.float64,element_type='C3D8')
    finally:so.build_dof_table,so.BatchedStencilOperator._build_stencils=old_conn,old_build
    op=solver.op;op.set_apply_backend('cuda',cuda_variant='thread',cuda_block_size=128)
    enable_reference(op,'mixed','upper6')
    op._diag_cache=None;op._M_inv_cache=None
    op._B_integrated=(op.B_mat.sum(0)*op.detJ).contiguous()
    op.compute_rhs=MethodType(compute_rhs,op)
    op._atomic_block=int(variant.removeprefix('atomic'))
    op.apply=MethodType(apply,op)
    return solver
