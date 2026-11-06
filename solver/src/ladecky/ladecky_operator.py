"""GPU realization of Ladecky et al. (2023), Section 5.1, full HEX8.

This is an independent CUDA implementation, not the authors' muSpectre binary.
The eight Gauss-point gradient, constitutive action and local transpose are
fused; global assembly uses either atomics or a deterministic force gather.
There is no modal transform, background split or shared material pool here.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
import numpy as np
import torch
from hybrid_adapter import HybridFIOperator

OFFSETS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]])


def gradients(voxel):
    signs = 2*OFFSETS-1
    answer = []
    for point in product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3):
        grad = np.empty((8,3))
        for j in range(3):
            others = [k for k in range(3) if k != j]
            grad[:,j] = signs[:,j]/(4*voxel[j])
            for k in others:
                grad[:,j] *= 1+signs[:,k]*point[k]
        answer.append(grad)
    return np.ascontiguousarray(answer)


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='voxel_ladecky_gauss_20261025_v1',
                sources=[str(Path(__file__).with_name('ladecky_kernels.cu'))],
                extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)


class GaussFIOperator(HybridFIOperator):
    def __init__(self, C, grid, voxel, *, assembly='gather', block=128):
        super().__init__(C, torch.empty(0, device=C.device, dtype=torch.int32),
                         grid, voxel, dense=True)
        if assembly not in ('gather','atomic') or block not in (64,128,256):
            raise ValueError('Unsupported assembly or block')
        self._gauss_grad = torch.tensor(gradients(voxel),device=C.device)
        self._gauss_assembly, self._gauss_block = assembly, block
        self._gauss_force = torch.empty((24,self.mesh.n_elem) if assembly=='gather' else (0,),
                                       dtype=torch.float64, device=C.device)
        self._gauss_empty = torch.empty(0,dtype=torch.float64,device=C.device)
        self._operator_metadata = dict(method='Ladecky et al. 2023 Section 5.1 GPU realization',
            quadrature=8,assembly=assembly,block=block,material='elementwise symmetric tensor',
            temporary_force_bytes=self._gauss_force.numel()*8,
            fused='gradient, constitutive action, element transpose',
            native_author_software=False)

    def _call(self,u,macro,mode):
        m=self.mesh
        return extension().evaluate(self._hybrid_coeff,self._gauss_grad,u,macro,
            self._gauss_force,m.Nz,m.Ny,m.Nx,m.V_elem/8,mode,
            self._gauss_assembly=='atomic',self._gauss_block)

    def apply(self,u):
        self._validate_apply_input(u)
        return self._call(u,self._gauss_empty,0)

    def compute_rhs(self,macro):
        return self._call(self._gauss_empty,self._macro_strain_tensor(macro),1)

    def compute_avg_stress(self,u,macro):
        self._validate_apply_input(u)
        return self._call(u,self._macro_strain_tensor(macro),2).mean(0).cpu().numpy()[None]
