"""Operator-only adapter: retain the frozen solver, RHS, recovery and reference."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import sys

import torch


@lru_cache(None)
def extension():
    os.environ["PATH"] = str(Path(sys.executable).parent)+":/usr/local/cuda/bin:"+os.environ.get("PATH", "")
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(name="fead_exact_modal_20261020", sources=[str(Path(__file__).with_name("modal_kernels.cu"))],
                extra_cuda_cflags=["-O3", "--ptxas-options=-v"], extra_cflags=["-O3"], verbose=False)


def apply(self, u):
    self._validate_apply_input(u)
    return extension().apply(self.C, self._modal_scales, u, self._modal_forces,
        self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, *self._modal_voxel,
        self._modal_kind == "fi", self._modal_variant, *self._modal_tile, self._modal_block)


def attach(op, kind, voxel, variant="stream", tile=(8, 8, 8), block=128):
    if op.B != 1 or op.C.dtype != torch.float64:
        raise ValueError("This controlled experiment requires B=1 and FP64 C21")
    variants = {"twostage":0, "atomic":1, "stream":2}
    if kind not in ("hg", "fi") or variant not in variants:
        raise ValueError((kind, variant))
    extension()
    op._modal_kind = kind
    op._modal_voxel = tuple(map(float, voxel))
    op._modal_variant = variants[variant]
    op._modal_tile = tuple(map(int, tile))
    op._modal_block = int(block)
    op._modal_scales = op.hg_scale if kind == "hg" else torch.empty(0, device=op.device, dtype=op.dtype)
    count = 24*op.mesh.n_elem if variant == "twostage" else 0
    op._modal_forces = torch.empty(count, device=op.device, dtype=op.dtype)
    op.apply = MethodType(apply, op)
    return op
