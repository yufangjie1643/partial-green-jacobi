"""Isolated FI memory experiments; frozen production modules are not edited."""
from pathlib import Path
from functools import lru_cache
from types import MethodType
import numpy as np
import torch
from femlib.batched import stencil_operator as so
from femlib.fem_batched import FEMHomogenizerBatched
from spatial_adapter import make_solver as make_stencil
from fi_reference import enable_reference

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='fead_fi_memory_20261016', sources=[str(Path(__file__).with_name('memory_kernels.cu'))],
                extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)

def compute_rhs(self,E):
    return extension().rhs(self.C,self._B_integrated,self._macro_strain_tensor(E),
                           self.mesh.Nz,self.mesh.Ny,self.mesh.Nx)

def direct_apply(self,u):
    self._validate_apply_input(u)
    # One shared scratch field, reused for all eight quadrature points and
    # all serial calls. It is never returned, so PCG vectors cannot alias it.
    return extension().direct(self.C,self.B_mat,u,self._stress_workspace,
        self.mesh.Nz,self.mesh.Ny,self.mesh.Nx,self.detJ)

def no_stencil(self,C,B,detJ):
    self.S_flat=torch.empty(0,device=self.device,dtype=self.dtype)
    self._diag_cache=None

def make(C,grid,variant,voxel=(1.,1.,1.)):
    direct=variant=='direct'
    if direct:
        old_conn,old_build=so.build_dof_table,so.BatchedStencilOperator._build_stencils
        so.build_dof_table=lambda *_:(np.empty((0,8),np.int64),np.empty((0,24),np.int64))
        so.BatchedStencilOperator._build_stencils=no_stencil
        try:
            solver=FEMHomogenizerBatched(C,voxel,grid,dtype=torch.float64,element_type='C3D8')
        finally:
            so.build_dof_table,so.BatchedStencilOperator._build_stencils=old_conn,old_build
        op=solver.op
        op.apply=MethodType(direct_apply,op)
    else:
        solver=make_stencil(C,grid,voxel,spectrum='real_half');op=solver.op
    op.set_apply_backend('cuda',cuda_variant='thread',cuda_block_size=128)
    enable_reference(op,'mixed','upper6')
    if variant in ('lifetime','direct'):
        # FFT is mandatory for these variants; the unused Jacobi diagonal is
        # released after reference construction. No solver math changes.
        op._diag_cache=None
        op._M_inv_cache=None
    if variant!='baseline':
        op._B_integrated=(op.B_mat.sum(0)*op.detJ).contiguous()
        op.compute_rhs=MethodType(compute_rhs,op)
    if direct:
        op._stress_workspace=torch.empty((6,op.B,op.mesh.n_elem),device=op.device,dtype=op.dtype)
    return solver

def lifetime_pcg():
    """Same PCG statements, with dead full-size tensor references released."""
    import inspect
    from femlib.batched import pcg
    source=inspect.getsource(pcg.solve_pcg_batched)
    replacements=[
        ('    n_iter_final = max_iter','    del z\n    n_iter_final = max_iter'),
        ('        if verbose and ((it + 1)', '        del Ap\n        if verbose and ((it + 1)'),
        ('            p, rz = compiled_post(r, p, z, rz, eps)',
         '            p, rz = compiled_post(r, p, z, rz, eps)\n            del z')]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    context=dict(vars(pcg));exec(compile(source,'<lifetime-only-pcg>','exec'),context)
    return context['solve_pcg_batched'],source
