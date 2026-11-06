"""Opt-in integration of modal K, native FFT and representation-aware PCG."""
from functools import lru_cache
import inspect
import os
from pathlib import Path
import sys
from types import MethodType

import torch

FFT = Path(os.environ.get("FEAD_FFT", str(__import__('project_paths').source_path('spectral'))))
sys.path.insert(0, str(FFT))
import native_adapter as native
import contest_adapter as contest


def attach_raw(op):
    """Expose owned FP32 SoA IFFT output, without materializing FP64 AoS z."""
    grid = op._contest_metadata["grid"]
    voxel = op._contest_metadata["voxel"]
    mode = 1 if op._contest_metadata["variant"] == "native_cached" else 2
    fi = op._modal_kind == "fi"

    def raw(self, packed):
        spectrum = self._contest_plan.forward(packed)
        contest.extension().spectral(
            spectrum, self._contest_coefficient, *self._contest_tables,
            *grid, *voxel, fi, mode)
        return self._contest_plan.inverse_destructive(spectrum)

    op.precond_packed = MethodType(raw, op)


def _pre_core(u, r, p, ap, pap, rho, norm, eps, stopped=None, boundary=True):
    bad = ~torch.isfinite(pap) | (pap <= 0)
    alpha = torch.where(bad, torch.zeros_like(rho), rho / pap.clamp(min=eps))[:, None]
    un = u + alpha * p
    if stopped is not None:
        un = torch.where((stopped == 0)[:, None], un, u)
    rn = r - alpha * ap
    relative = torch.linalg.vector_norm(rn, dim=-1) / norm
    # Cast the completed FP64 recurrence, never update a rounded FP32 residual.
    if boundary:
        packed = rn.view(1, -1, 3).transpose(1, 2).to(torch.float32).contiguous()
        return un, rn, relative, packed
    return un, rn, relative


def _post(r, p, physical, rho, eps):
    # A view is sufficient. torch.compile fuses conversion into the consumers.
    z = physical.reshape(1, 3, -1).transpose(1, 2).to(torch.float64)
    rv, pv = r.view(1, -1, 3), p.view(1, -1, 3)
    new = (rv * z).sum(dim=(1, 2))
    beta = (new / rho.clamp(min=eps))[:, None, None]
    direction = (z + beta * pv).contiguous().view_as(p)
    return direction, new


@lru_cache(None)
def compiled(kind, boundary, snapshot):
    options = {"triton.cudagraphs": False}
    if kind == "hg":
        if snapshot:
            def pre(u, r, p, ap, pap, rho, norm, eps, stopped):
                return _pre_core(u, r, p, ap, pap, rho, norm, eps, stopped, boundary)
        else:
            def pre(u, r, p, ap, pap, rho, norm, eps):
                return _pre_core(u, r, p, ap, pap, rho, norm, eps, None, boundary)
    else:
        if snapshot:
            def pre(u, r, p, ap, rho, norm, eps, stopped):
                return _pre_core(u, r, p, ap, (p * ap).sum(-1), rho, norm, eps, stopped, boundary)
        else:
            def pre(u, r, p, ap, rho, norm, eps):
                return _pre_core(u, r, p, ap, (p * ap).sum(-1), rho, norm, eps, None, boundary)
    return (torch.compile(pre, fullgraph=True, options=options),
            torch.compile(_post, fullgraph=True, options=options))


def make_solver(kind, boundary, snapshot, fallback):
    """Keep the frozen control flow; substitute only audited boundary operations."""
    from femlib.batched import pcg
    source = inspect.getsource(pcg.solve_pcg_batched)
    context = dict(vars(pcg))

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError("Frozen PCG source changed: " + old)
        source = source.replace(old, new)

    if kind == "hg":
        replace("        Ap = op.apply(p)", "        Ap, energy_pAp = op.apply_with_energy(p)")
        replace("                    u, r, p, Ap, rz, rhs_norm, eps,",
                "                    u, r, p, Ap, energy_pAp, rz, rhs_norm, eps,")
    pre, post = compiled(kind, boundary, snapshot)
    context["_compiled_pcg_latched_pre_step"] = lambda: pre
    if boundary:
        original_steps = pcg._compiled_pcg_steps
        context["_compiled_pcg_steps"] = lambda: (original_steps()[0], post)
        replace("                u, r, res = compiled_latched_pre(",
                "                u, r, res, packed = compiled_latched_pre(")
        replace("        z = apply_M_inv(r)\n        if use_compiled:",
                "        z = op.precond_packed(packed.view(1, 3, op.mesh.Nz, op.mesh.Ny, op.mesh.Nx))\n"
                "        del packed\n        if use_compiled:")
        replace("            p, rz = compiled_post(r, p, z, rz, eps)",
                "            p, rz = compiled_post(r, p, z, rz, eps)\n            del z")
        # The initial full z otherwise stays live for the entire Krylov solve.
        replace("    if use_triton:\n", "    del z\n    if use_triton:\n")
    if snapshot:
        replace("    best_u = u.clone()", "    best_u = u")
        args = "                    u, r, p, Ap, " + ("energy_pAp, " if kind == "hg" else "") + "rz, rhs_norm, eps,"
        replace(args, args + " stop_iter,")
        replace("                _copy_best_latched_kernel[(triton.cdiv(N, 1024), B)](\n"
                "                    u, best_u, res, best_res, stop_iter, N, BLOCK=1024,\n"
                "                )",
                "                best_u = u  # u freezes after its first tolerance crossing.")
        context["_boundary_fallback"] = fallback
        # An unconverged solve can require an old best iterate. Re-run the frozen
        # solver only on that exceptional path rather than storing every iterate.
        replace("    return best_u, n_iter_final, best_res",
                "    return _boundary_fallback(op, rhs, tol=tol, max_iter=max_iter, verbose=verbose,\n"
                "        log_every=log_every, update_mode=update_mode, x0=x0,\n"
                "        recycle_callback=recycle_callback, recycle_stride=recycle_stride,\n"
                "        convergence_check_stride=convergence_check_stride,\n"
                "        convergence_check_adaptive_factor=convergence_check_adaptive_factor,\n"
                "        verify_final_residual=verify_final_residual)")
        # Do not try certifying uninitialized best_u before the fallback.
        replace("    if verify_final_residual:\n        best_res = _true_relative_residual(op, rhs, best_u, rhs_norm)",
                "    # No threshold crossing: fallback recovers the original best-iterate semantics.")
    exec(compile(source, "<chained-pcg>", "exec"), context)
    fn = context["solve_pcg_batched"]

    def checked(op, rhs, **kwargs):
        if op.B != 1 or kwargs.get("update_mode") != "compiled" or kwargs.get("convergence_check_stride", 1) <= 1:
            raise ValueError("Chained PCG requires B=1, compiled updates and latched host checks")
        if snapshot and kwargs.get("recycle_callback") is not None:
            raise ValueError("Snapshot fallback cannot replay an external callback")
        if kind == "hg":
            kwargs["verify_final_residual"] = False  # Outer solve certifies FP64 projected u.
        return fn(op, rhs, **kwargs)
    return checked


def configure(b, kind, variant):
    """b is already configured with modal K and the prior PCG pipeline."""
    boundary = variant in ("boundary", "combined", "compact_combined")
    snapshot = variant in ("snapshot", "combined", "compact_combined")
    compact = variant in ("compact", "compact_combined")
    if compact and kind != "hg":
        raise ValueError("Compact constitutive kernel is currently HG only")
    old_build, old_solver = b.build, b.solve_pcg_batched
    if boundary or snapshot:
        b.solve_pcg_batched = make_solver(kind, boundary, snapshot, old_solver)
    if compact:
        path = Path(os.environ.get("FEAD_ABLATION", str(__import__('project_paths').source_path('compact_operator'))))
        import importlib.util
        spec = importlib.util.spec_from_file_location("chain_compact_adapter", path / "adapter.py")
        compact_adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compact_adapter)
        compact_adapter.extension()

    def build(data, material, route, precision, stages):
        op = old_build(data, material, route, precision, stages)
        native.attach(op, kind, "cached" if kind == "hg" else "dense")
        if boundary:
            attach_raw(op)
        if compact:
            compact_adapter.attach(op, data, material, "compact")
        stages.end("chain_attachment", op)
        return op
    b.build = build
    return old_build, old_solver
