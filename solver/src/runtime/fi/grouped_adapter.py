from functools import lru_cache
from pathlib import Path
from types import MethodType
import torch
import memory_adapter
@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='fead_fi_grouped_20261016',sources=[str(Path(__file__).with_name('grouped_kernels.cu'))],extra_cuda_cflags=['-O3'],extra_cflags=['-O3'])
def apply(self,u):
    self._validate_apply_input(u)
    return extension().apply(self.C,self.B_mat,u,self._stress_workspace,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx,self.detJ,self._group)
def make(C,grid,variant,voxel=(1.,1.,1.)):
    if not variant.startswith('group'):return memory_adapter.make(C,grid,variant,voxel)
    q=int(variant[-1]);solver=memory_adapter.make(C,grid,'direct',voxel);op=solver.op
    del op._stress_workspace
    op._stress_workspace=torch.empty((q,6,op.B,op.mesh.n_elem),device=op.device,dtype=op.dtype)
    op._group=q;op.apply=MethodType(apply,op)
    return solver
