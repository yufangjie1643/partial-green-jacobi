"""Exact-grid FFT loop optimization. No coarse grid or changed preconditioner."""
from functools import lru_cache
import os
from pathlib import Path
from types import MethodType
import torch

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(name="fead_fft_contest_20261020", sources=[str(HERE/"contest_kernels.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)


def attach(op, kind, variant, release=True):
    if kind not in ("fi", "hg") or variant not in ("fused", "inplace", "cached", "dense"):
        raise ValueError((kind, variant))
    if op.B != 1 or op.dtype != torch.float64:
        raise ValueError("Controlled prototype requires B=1 FP64 physical vectors")
    if op._fft_symbol_inv.dtype != torch.float32:
        raise ValueError("Same FP32/complex64 FFT protocol required")
    fi = kind == "fi"
    tables = op._formula_vtables if fi else op._analytic_tables
    grid = (op.mesh.Nz, op.mesh.Ny, op.mesh.Nx)
    voxel = (op.mesh.dx, op.mesh.dy, op.mesh.dz)
    original_coefficient = op._fft_symbol_inv
    mode = {"fused":0, "inplace":0, "cached":1, "dense":2}[variant]
    coefficient = (extension().prepare(original_coefficient, *tables, *grid, *voxel, fi, mode == 2)
                   if mode else original_coefficient)
    op._contest_coefficient = coefficient
    # Stored tables are immutable; no PCG state or returned output is reused.
    op._contest_tables = tables
    op._contest_metadata = dict(kind=kind, variant=variant, grid=grid, voxel=voxel,
        coefficients_bytes=coefficient.numel()*coefficient.element_size(),
        grid_changed=False, physical_precision="FP64", FFT_precision="FP32/complex64",
        normalization="in unpack" if mode == 0 else "folded once into coefficients")
    def apply(self, residual):
        field = extension().pack(residual, *grid)
        spectrum = torch.fft.rfftn(field, dim=(-3, -2, -1)).contiguous()
        del field
        if variant == "fused":
            # Keep the old spectral arithmetic/allocation to isolate layout fusion.
            if fi:
                import formula_adapter
                values = formula_adapter.extension().fi_apply(original_coefficient, *tables, spectrum, *grid, *voxel)
            else:
                import pipeline_adapter
                values = pipeline_adapter.extension().spectral(original_coefficient, *tables, spectrum,
                    self.mesh.Nx, self.mesh.Ny, self.mesh.Nz, *voxel)
            del spectrum
        else:
            extension().spectral(spectrum, coefficient, *tables, *grid, *voxel, fi, mode)
            values = spectrum
            del spectrum
        physical = torch.fft.irfftn(values, s=grid, dim=(-3, -2, -1), norm="forward").contiguous()
        del values
        return extension().unpack(physical, *grid, mode == 0)
    op.apply_precond = MethodType(apply, op)
    if release and mode:
        # The closure only needs the new cache; avoid retaining the old cache.
        original_coefficient = None
        op._fft_symbol_inv = None
    return op
