"""Opt-in pipeline changes around the frozen, CPU-verified modal operator."""
from contextlib import contextmanager
from functools import lru_cache
import inspect
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(name="fead_pipeline_20261020_v1", sources=[str(HERE / "pipeline_kernels.cu")],
                extra_cuda_cflags=["-O3", "--ptxas-options=-v"], extra_cflags=["-O3"], verbose=False)


@contextmanager
def lean_hg_initialization():
    """Skip unused Jacobi setup; retain a lazy diagonal for explicit callers."""
    from femlib.batched.c3d8r_hourglass_operator import BatchedC3D8RHourglassTwoStageOperator as HG
    old_diag, old_scale = HG._build_diagonal, HG._build_hourglass_scale
    def scale(op):
        hbar = (op.mesh.dx + op.mesh.dy + op.mesh.dz) / 3
        return extension().scales(op.C, op.hourglass_coefficient * op.mesh.V_elem / hbar**2)
    HG._build_diagonal = lambda self: None
    HG._build_hourglass_scale = scale
    try:
        yield
    finally:
        HG._build_diagonal, HG._build_hourglass_scale = old_diag, old_scale


def attach_lazy_diag(op):
    original_diag = type(op)._build_diagonal
    def diag(self):
        if self._diag_cache is None:
            self._diag_cache = original_diag(self)
        return self._diag_cache
    op.diag = MethodType(diag, op)


def apply_with_energy(self, u):
    self._validate_apply_input(u)
    out, partial = extension().apply_energy(self.C, self._modal_scales, u,
        self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, *self._modal_voxel,
        self._modal_kind == "fi", True, 128)
    return out, partial.sum().reshape(1)


def attach_energy(op):
    op.apply_with_energy = MethodType(apply_with_energy, op)


def _pre_energy(u, r, p, Ap, pAp_raw, rz, rhs_norm, eps):
    bad = ~torch.isfinite(pAp_raw) | (pAp_raw <= 0)
    alpha = (rz / pAp_raw.clamp(min=eps)).unsqueeze(-1)
    alpha = torch.where(bad.unsqueeze(-1), torch.zeros_like(alpha), alpha)
    u_new = u + alpha * p
    r_new = r - alpha * Ap
    res = torch.linalg.vector_norm(r_new, dim=-1) / rhs_norm
    return u_new, r_new, res


@lru_cache(None)
def energy_solver():
    """Preserve frozen recurrence, best-iterate latching and host-check protocol."""
    from femlib.batched import pcg
    source = inspect.getsource(pcg.solve_pcg_batched)
    for old, new in [
        ("        Ap = op.apply(p)", "        Ap, energy_pAp = op.apply_with_energy(p)"),
        ("                    u, r, p, Ap, rz, rhs_norm, eps,",
         "                    u, r, p, Ap, energy_pAp, rz, rhs_norm, eps,")]:
        if source.count(old) != 1:
            raise RuntimeError("Frozen PCG source changed: " + old)
        source = source.replace(old, new)
    pre = torch.compile(_pre_energy, fullgraph=True, options={"triton.cudagraphs": False})
    context = dict(vars(pcg))
    context["_compiled_pcg_latched_pre_step"] = lambda: pre
    exec(compile(source, "<force-energy-pcg>", "exec"), context)
    fn = context["solve_pcg_batched"]
    def checked(op, rhs, **kwargs):
        if kwargs.get("update_mode") != "compiled" or kwargs.get("convergence_check_stride", 1) <= 1:
            raise ValueError("Energy solver requires compiled updates and latched convergence checks")
        return fn(op, rhs, **kwargs)
    return checked


def enable_analytic_reference(op):
    """Same isotropic HG FE reference; FP32 execution, analytic rank-one inverse."""
    from femlib.c3d8r import hourglass_scale
    if op.B != 1 or op._modal_kind != "hg":
        raise ValueError("Analytic reference is currently HG, B=1 only")
    reference = op._compute_reference_medium_from_packed("zeman", verbose=False)[0]
    lam, mu = float(reference[0, 1]), float(reference[3, 3])
    check = np.zeros((6, 6)); check[:3, :3] = lam
    check[np.arange(3), np.arange(3)] += 2 * mu
    check[np.arange(3, 6), np.arange(3, 6)] = mu
    if not np.allclose(check, reference, rtol=1e-14, atol=0):
        raise ValueError("Reference medium must be isotropic")
    nx, ny, nz = op.mesh.Nx, op.mesh.Ny, op.mesh.Nz
    tables = []
    for n, count in ((nx, nx//2+1), (ny, ny), (nz, nz)):
        t = np.pi * np.arange(count, dtype=np.float64) / n
        table = np.stack((np.sin(t), np.cos(t)), axis=-1)
        tables.append(torch.as_tensor(table, device=op.device).contiguous())
    coef = extension().coefficients(*tables, nx, ny, nz, *op._modal_voxel,
        lam, mu, hourglass_scale(reference, op._modal_voxel, op.hourglass_coefficient))
    op._analytic_tables = tables
    op._analytic_reference_C0 = reference.tolist()
    op._fft_symbol_inv = coef
    op._fft_enabled = True
    op._fft_symbol_layout = "analytic_HG_rank1"
    op._fft_spectrum_mode = "real_half"
    op._fft_inverse_norm_folded = False
    op._fft_symbol_shared = False
    op._fft_batch_chunk_size = None
    op._fft_args = {"C0_voigt": reference[None], "ref_strategy": "zeman"}
    op._M_inv_cache = None
    def apply_reference(self, residual):
        shape = (self.mesh.Nz, self.mesh.Ny, self.mesh.Nx)
        field = residual.to(torch.float32).view(1, -1, 3).transpose(1, 2).contiguous().view(1, 3, *shape)
        spectrum = torch.fft.rfftn(field, dim=(-3, -2, -1))
        values = extension().spectral(self._fft_symbol_inv, *self._analytic_tables, spectrum,
            self.mesh.Nx, self.mesh.Ny, self.mesh.Nz, *self._modal_voxel)
        z = torch.fft.irfftn(values, s=shape, dim=(-3, -2, -1), norm="backward")
        return z.view(1, 3, -1).transpose(1, 2).contiguous().view_as(residual).to(residual.dtype)
    op.apply_precond = MethodType(apply_reference, op)
    return op


def configure(b, old_modal, kind, variant):
    """Install one explicit ablation; returns frozen callables for restoration."""
    lean = variant in ("lean", "energy", "analytic", "combined")
    energy = variant in ("energy", "combined")
    analytic = kind == "hg" and variant in ("analytic", "combined")
    original_build, original_solver = b.build, b.solve_pcg_batched
    if energy:
        solve = energy_solver()
        def without_duplicate_verification(op, rhs, **kwargs):
            # benchmark.solve always certifies the projected result in FP64.
            # Only the unused pre-projection return residual is omitted.
            if variant == "combined":
                kwargs["verify_final_residual"] = False
            return solve(op, rhs, **kwargs)
        b.solve_pcg_batched = without_duplicate_verification
    def build(data, material, route, precision, stages):
        if variant == "paper":
            op = original_build(data, material, "direct" if kind == "hg" else "tap14", precision, stages)
            stages.end("candidate_attachment", op)
            return op
        if kind == "hg" and lean:
            c = b.construct(data, 0, 1, material, True, True); stages.end("material")
            voxel = data["domain_size"][0] / np.asarray(data["grid_shape"][::-1])
            with lean_hg_initialization():
                op = b.make_solver(c, data["grid_shape"], voxel, fft=False).op
            del c
            attach_lazy_diag(op)
            old_modal.attach(op, kind, voxel, variant="atomic", block=128)
            stages.end("operator", op)
            if analytic:
                enable_analytic_reference(op)
            else:
                b.enable_reference(op, 2, precision)
            stages.end("reference", op)
            stages.end("representation", op)
        else:
            op = original_build(data, material, "direct", precision, stages)
            voxel = data["domain_size"][0] / np.asarray(data["grid_shape"][::-1])
            old_modal.attach(op, kind, voxel, variant="atomic", block=128)
        if energy:
            attach_energy(op)
        stages.end("candidate_attachment", op)
        return op
    b.build = build
    return original_build, original_solver
