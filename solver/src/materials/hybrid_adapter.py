"""Opt-in exact FI hybrid storage, with unchanged FFT and PCG dependencies."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from adapter import load_material_module

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def material_modules():
    mf = load_material_module()
    path = HERE / "hybrid_material.py"
    if not path.exists():
        path = HERE.parents[1] / "repro/source/femlib/hybrid_material.py"
    spec = importlib.util.spec_from_file_location("femlib.hybrid_material", path)
    hm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hm
    spec.loader.exec_module(hm)
    return mf, hm


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(name="fead_hybrid_storage_20261021_v1", sources=[str(HERE/"hybrid_kernels.cu")],
                extra_cuda_cflags=["-O3", "--ptxas-options=-v"], extra_cflags=["-O3"], verbose=False)


class HybridFIOperator:
    """B=1 periodic FI prototype. All material consumers read the same pool.

    dense=True is an ablation: identical kernels except direct e*21 addressing.
    Neither branch creates full element matrices or connectivity arrays.
    """
    def __init__(self, coefficients, index, grid, voxel, *, dense=False):
        nz, ny, nx = map(int, grid)
        if min(grid) < 2 or min(voxel) <= 0:
            raise ValueError("Periodic grid dimensions >=2 and positive voxel sizes required")
        self.B = 1
        self.device, self.dtype = coefficients.device, coefficients.dtype
        if self.device.type != "cuda" or self.dtype != torch.float64:
            raise ValueError("FI hybrid experiment requires CUDA FP64")
        n = nx*ny*nz
        self.mesh = SimpleNamespace(Nz=nz, Ny=ny, Nx=nx, dx=float(voxel[0]), dy=float(voxel[1]),
                                    dz=float(voxel[2]), n_elem=n, n_nodes=n, n_dofs=3*n,
                                    V_elem=float(np.prod(voxel)))
        self._hybrid_coeff = coefficients.reshape(-1, 21)
        self._hybrid_index = index
        self._hybrid_indirect = not dense
        self.C = None  # Deliberately no full-grid C21 alias or fallback.
        self._C_voigt_gpu = None
        self.element_type = "c3d8"
        self._modal_kind = "fi"
        self._modal_voxel = tuple(map(float, voxel))
        self._fft_enabled = False
        self._diag_cache = self._M_inv_cache = None
        self._fft_symbol_inv = None
        self._fft_args = None

    def geometry(self):
        m = self.mesh
        return (m.Nz, m.Ny, m.Nx, m.dx, m.dy, m.dz)

    def _validate_apply_input(self, u):
        if u.shape != (1, self.mesh.n_dofs) or u.dtype != self.dtype or u.device != self.device or not u.is_contiguous():
            raise ValueError("Expected contiguous FP64 displacement (1,3N) on the material device")

    def project_zero_mean(self, value):
        u = value.view(1, self.mesh.n_nodes, 3)
        return (u-u.mean(dim=1, keepdim=True)).reshape(1, -1)

    def _macro_strain_tensor(self, macro):
        if np.shape(macro) != (6,):
            raise ValueError("Six engineering macro strain components required")
        return torch.as_tensor(macro, dtype=self.dtype, device=self.device).contiguous()

    def apply(self, u):
        self._validate_apply_input(u)
        return extension().apply(self._hybrid_coeff, self._hybrid_index, u,
                                 *self.geometry(), self._hybrid_indirect)

    def compute_rhs(self, macro):
        return extension().rhs(self._hybrid_coeff, self._hybrid_index, self._macro_strain_tensor(macro),
                               *self.geometry(), self._hybrid_indirect)

    def compute_avg_stress(self, u, macro):
        self._validate_apply_input(u)
        return extension().stress(self._hybrid_coeff, self._hybrid_index, u, self._macro_strain_tensor(macro),
                                  *self.geometry(), self._hybrid_indirect).cpu().numpy()

    def _compute_reference_medium(self, strategy, *, verbose=False):
        # Same grid samples and same zeman arithmetic as the frozen C21 path.
        # Only sampled C21 rows are gathered; never expand all N rows.
        if strategy != "zeman":
            raise ValueError("This controlled prototype uses the frozen zeman reference only")
        m = self.mesh
        z = torch.arange(0, m.Nz, max(1, m.Nz//16), device=self.device)
        y = torch.arange(0, m.Ny, max(1, m.Ny//16), device=self.device)
        x = torch.arange(0, m.Nx, max(1, m.Nx//16), device=self.device)
        positions = (z[:, None, None]*(m.Ny*m.Nx)+y[None, :, None]*m.Nx+x[None, None, :]).reshape(-1)
        rows = self._hybrid_index.reshape(-1)[positions].long() if self._hybrid_indirect else positions
        packed = self._hybrid_coeff.index_select(0, rows).cpu().numpy()
        sub = np.zeros((len(packed), 6, 6))
        for k, (i, j) in enumerate(material_modules()[0].PAIRS):
            sub[:, i, j] = sub[:, j, i] = packed[:, k]
        mus = .5*(sub[:, 3, 3]+sub[:, 4, 4]+sub[:, 5, 5])/2.
        Ks = (sub[:, 0, 0]+sub[:, 1, 1]+sub[:, 2, 2]
              +2*(sub[:, 0, 1]+sub[:, 0, 2]+sub[:, 1, 2]))/9.
        mu = .5*(float(mus.max())+float(mus.min()))
        K = .5*(float(Ks.max())+float(Ks.min()))
        lam = max(0., K-2*mu/3.)
        out = np.zeros((6, 6));out[:3, :3] = lam
        out[np.arange(3), np.arange(3)] += 2*mu
        out[np.arange(3, 6), np.arange(3, 6)] = mu
        return out[None]


def install(b, cs, *, dense=False):
    mf, hm = material_modules()
    from reference_closed_form import design_closed

    def build(data, materials, route, precision, stages):
        phase = np.asarray(data["phase_id"])
        if phase.ndim == 4 and phase.shape[0] != 1:
            raise ValueError("The FI integration prototype requires B=1")
        if tuple(phase.shape[-3:]) != tuple(data["grid_shape"]):
            raise ValueError("phase_id and grid_shape disagree")
        if dense:
            coefficients = mf.build_material_field(phase, materials, orientation=data.get("orientation"))
            index = torch.empty(0, device=coefficients.device, dtype=torch.int32)
            b._storage = dict(mode="dense", n_elements=int(phase.size), coefficient_bytes=coefficients.numel()*8,
                              index_bytes=0, dense_bytes=coefficients.numel()*8)
        else:
            coefficients, index, b._storage = hm.build_hybrid_material(phase, materials, orientation=data.get("orientation"))
            b._storage["mode"] = "hybrid"
        stages.end("material")
        voxel = data["domain_size"][0]/np.asarray(data["grid_shape"][::-1])
        op = HybridFIOperator(coefficients, index, data["grid_shape"], voxel, dense=dense)
        stages.end("operator", op)
        cs.s.f.formula.enable_fi_reference(op)
        stages.end("reference", op)
        labels = np.unique(phase)
        phases = [mf.material_stiffness(materials[int(k)]) for k in labels]
        reference, metadata = design_closed(phases, op._formula_reference_C0, "fi")
        metadata["active_phase_labels"] = labels.tolist()
        original = op._compute_reference_medium
        op._compute_reference_medium = lambda *args, **kwargs: reference[None]
        try:
            cs.s.f.formula.enable_fi_reference(op)
        finally:
            op._compute_reference_medium = original
        b._reference_design = metadata
        stages.end("spectral_reference_design", op)
        return op

    b.build = build
    return mf
