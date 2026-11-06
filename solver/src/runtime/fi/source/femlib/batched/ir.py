# -*- coding: utf-8 -*-
"""Batched mixed-precision iterative refinement (IR):

    Outer (op_outer, fp64):  r = f - K u              ← high-precision residual
    Inner (op_inner, fp32):  PCG solves δu, residual only needs to drop ~1 order
    Update:                  u ← u + δu  → zero-mean projection
    Repeat until ||r|| / ||f|| < outer_tol (decided on the batch-wise maximum)

op_outer and op_inner must share the same mesh / the same C field and differ only in dtype.
Each batch tracks its own best_u (keyed on its own outer residual) to prevent IR oscillations from causing drift.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .pcg import solve_pcg_batched


def solve_ir_batched(op_outer, op_inner, rhs, *,
                     outer_tol=1e-8, inner_tol=0.1,
                     max_outer=30, max_inner=80,
                     inner_pcg_update_mode="eager",
                     inner_convergence_check_stride=1,
                     inner_verify_final_residual=False,
                     verbose=False):
    """
    Parameters
    ----------
    op_outer : BatchedStencilOperator   (fp64)
    op_inner : BatchedStencilOperator   (fp32)
    rhs      : (B, n_dofs) torch, dtype=op_outer.dtype

    Returns
    -------
    best_u      : (B, n_dofs) op_outer.dtype, best solution per batch
    info_batch  : dict with keys:
                  outer_iters : int           outer iteration count at global stop
                  inner_iters : int           cumulative inner PCG iterations (summed over batches)
                  rel_res     : (B,) np.ndarray  final outer relative residual per batch
    """
    assert op_outer.B == op_inner.B, \
        f"batch size mismatch: outer={op_outer.B}, inner={op_inner.B}"
    B = op_outer.B
    device = rhs.device
    out_dtype = rhs.dtype
    in_dtype = op_inner.dtype

    u = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1).clamp(min=1e-30)   # (B,)

    best_res = torch.full((B,), float('inf'), device=device, dtype=out_dtype)
    best_u = u.clone()

    total_inner = 0
    last_k = 0
    last_res = torch.full((B,), float('inf'), device=device, dtype=out_dtype)

    for k in range(max_outer):
        r_out = rhs - op_outer.apply(u)
        rel_res = torch.linalg.vector_norm(r_out, dim=-1) / rhs_norm     # (B,)
        last_res = rel_res
        last_k = k

        improved = rel_res < best_res
        if improved.any():
            mask = improved.unsqueeze(-1)
            best_u = torch.where(mask, u, best_u)
            best_res = torch.where(improved, rel_res, best_res)

        if verbose:
            print(f"    [IR-batched] outer {k:3d}  "
                  f"max_res={rel_res.max().item():.3e}  "
                  f"min_res={rel_res.min().item():.3e}")

        if rel_res.max().item() < outer_tol:
            if verbose:
                print(f"    [IR-batched] outer {k:3d}  all batches converged")
            break

        # Low-precision residual → inner PCG
        r_in = r_out.to(in_dtype)
        du_in, n_in, _ = solve_pcg_batched(
            op_inner, r_in,
            tol=inner_tol, max_iter=max_inner, verbose=False,
            update_mode=inner_pcg_update_mode,
            convergence_check_stride=inner_convergence_check_stride,
            verify_final_residual=inner_verify_final_residual,
        )
        total_inner += int(n_in)

        # Back to high precision, accumulate + zero-mean projection (suppresses drift along the singular kernel)
        du_out = du_in.to(out_dtype)
        u = op_outer.project_zero_mean(u + du_out)

    info = {
        'outer_iters': int(last_k + 1),
        'inner_iters': int(total_inner),
        'rel_res':     best_res.detach().cpu().numpy().astype(np.float64),
        'converged':   bool(best_res.max().item() < outer_tol),
        'stop_reason': 'converged' if best_res.max().item() < outer_tol else 'max_outer',
        'inner_pcg_update_mode': str(inner_pcg_update_mode),
    }
    return best_u, info


def solve_ir_adaptive_batched(op_outer, op_inner, rhs, *,
                              outer_tol=1e-8,
                              inner_tols=(0.1, 0.03, 0.01),
                              max_outer=30,
                              max_inner=80,
                              adaptive_max_inner=160,
                              stagnation_factor=0.3,
                              stall_factor=0.9,
                              stall_patience=3,
                              inner_pcg_update_mode="eager",
                              inner_convergence_check_stride=1,
                              inner_verify_final_residual=False,
                              verbose=False):
    """Mixed-precision IR with adaptive fp32 inner PCG tolerance.

    The outer residual and solution update remain in fp64.  The fp32 inner
    correction starts loose and is tightened only when the previous correction
    did not reduce the outer residual enough:

        rho_{k+1} > stagnation_factor * rho_k

    Once the tightest inner tolerance is reached, repeated near-stagnation can
    stop the IR loop early so that the caller can fall back to a robust fp64
    solve instead of spending all remaining outer iterations.
    """
    assert op_outer.B == op_inner.B, \
        f"batch size mismatch: outer={op_outer.B}, inner={op_inner.B}"
    inner_tols = tuple(float(tol) for tol in inner_tols)
    if not inner_tols:
        raise ValueError("inner_tols must contain at least one tolerance")
    if any(tol <= 0.0 for tol in inner_tols):
        raise ValueError(f"inner_tols must be positive, got {inner_tols}")
    if max_outer < 1:
        raise ValueError("max_outer must be >= 1")
    if max_inner < 1 or adaptive_max_inner < 1:
        raise ValueError("max_inner and adaptive_max_inner must be >= 1")

    B = op_outer.B
    out_dtype = rhs.dtype
    in_dtype = op_inner.dtype

    u = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1).clamp(min=1e-30)

    best_res = torch.full((B,), float('inf'), device=rhs.device, dtype=out_dtype)
    best_u = u.clone()

    total_inner = 0
    last_k = 0
    prev_outer_max = None
    tol_index = 0
    current_max_inner = int(max_inner)
    stall_count = 0
    stop_reason = 'max_outer'
    history = []

    for k in range(max_outer):
        r_out = rhs - op_outer.apply(u)
        rel_res = torch.linalg.vector_norm(r_out, dim=-1) / rhs_norm
        outer_max = float(rel_res.max().item())
        outer_min = float(rel_res.min().item())
        last_k = k

        improved = rel_res < best_res
        if improved.any():
            mask = improved.unsqueeze(-1)
            best_u = torch.where(mask, u, best_u)
            best_res = torch.where(improved, rel_res, best_res)

        if verbose:
            print(f"    [IR-adaptive] outer {k:3d}  "
                  f"max_res={outer_max:.3e}  min_res={outer_min:.3e}  "
                  f"inner_tol={inner_tols[tol_index]:.2e}")

        if outer_max < outer_tol:
            stop_reason = 'converged'
            if verbose:
                print(f"    [IR-adaptive] outer {k:3d}  all batches converged")
            break

        tightened = False
        if prev_outer_max is not None and outer_max > float(stagnation_factor) * prev_outer_max:
            if tol_index < len(inner_tols) - 1:
                tol_index += 1
                current_max_inner = max(current_max_inner, int(adaptive_max_inner))
                tightened = True
                stall_count = 0
            elif outer_max > float(stall_factor) * prev_outer_max:
                stall_count += 1
            else:
                stall_count = 0
        else:
            stall_count = 0

        if stall_count >= int(stall_patience):
            stop_reason = 'stalled'
            if verbose:
                print(f"    [IR-adaptive] outer {k:3d}  stalled; "
                      "caller should fall back to fp64")
            break

        inner_tol = inner_tols[tol_index]
        r_in = r_out.to(in_dtype)
        du_in, n_in, inner_rel_res = solve_pcg_batched(
            op_inner, r_in,
            tol=inner_tol, max_iter=current_max_inner, verbose=False,
            update_mode=inner_pcg_update_mode,
            convergence_check_stride=inner_convergence_check_stride,
            verify_final_residual=inner_verify_final_residual,
        )
        total_inner += int(n_in)
        history.append({
            'outer_iter': int(k),
            'rel_res_max': outer_max,
            'rel_res_min': outer_min,
            'inner_tol': float(inner_tol),
            'inner_max_iter': int(current_max_inner),
            'inner_iters': int(n_in),
            'inner_rel_res_max': float(inner_rel_res.max().item()),
            'tightened': bool(tightened),
            'stall_count': int(stall_count),
        })

        du_out = du_in.to(out_dtype)
        u = op_outer.project_zero_mean(u + du_out)
        prev_outer_max = outer_max

    converged = bool(best_res.max().item() < outer_tol)
    if converged:
        stop_reason = 'converged'
    info = {
        'outer_iters': int(last_k + 1),
        'inner_iters': int(total_inner),
        'rel_res': best_res.detach().cpu().numpy().astype(np.float64),
        'converged': converged,
        'stop_reason': stop_reason,
        'adaptive': True,
        'inner_tols': [float(v) for v in inner_tols],
        'final_inner_tol': float(inner_tols[tol_index]),
        'max_inner_used': int(current_max_inner),
        'inner_pcg_update_mode': str(inner_pcg_update_mode),
        'history': history,
    }
    return best_u, info


def compute_effective_stiffness_ir_batched(op_outer, op_inner, *,
                                           outer_tol=1e-8, inner_tol=0.1,
                                           max_outer=30, max_inner=80,
                                           inner_pcg_update_mode="eager",
                                           inner_convergence_check_stride=1,
                                           inner_verify_final_residual=False,
                                           verbose=False):
    """IR driver over 6 load cases × B samples.

    Returns
    -------
    C_eff_batch     : np.ndarray (B, 6, 6)
    info_per_sample : list[list[dict]]  outer B × inner 6, each entry
                      {'outer_iters', 'inner_iters', 'rel_res', 'elapsed_s'}
    """
    Bsz = op_outer.B
    C_eff_all = np.zeros((Bsz, 6, 6), dtype=np.float64)
    info_per_sample = [[None] * 6 for _ in range(Bsz)]

    for J in range(6):
        E = np.zeros(6, dtype=np.float64)
        E[J] = 1.0
        t0 = time.time()
        rhs = op_outer.compute_rhs(E)                              # (B, n_dofs)
        u_star, info = solve_ir_batched(
            op_outer, op_inner, rhs,
            outer_tol=outer_tol, inner_tol=inner_tol,
            max_outer=max_outer, max_inner=max_inner,
            inner_pcg_update_mode=inner_pcg_update_mode,
            inner_convergence_check_stride=inner_convergence_check_stride,
            inner_verify_final_residual=inner_verify_final_residual,
            verbose=verbose,
        )
        u_star = op_outer.project_zero_mean(u_star)
        avg_sig = op_outer.compute_avg_stress(u_star, E)           # (B, 6)
        elapsed = time.time() - t0

        C_eff_all[:, :, J] = avg_sig

        per_load_elapsed = elapsed / Bsz
        for b in range(Bsz):
            info_per_sample[b][J] = {
                'outer_iters': info['outer_iters'],
                'inner_iters': info['inner_iters'],
                'rel_res':     float(info['rel_res'][b]),
                'elapsed_s':   float(per_load_elapsed),
                'inner_pcg_update_mode': info['inner_pcg_update_mode'],
                'outer_apply_backend': getattr(
                    op_outer, 'apply_backend', 'unspecified',
                ),
                'inner_apply_backend': getattr(
                    op_inner, 'apply_backend', 'unspecified',
                ),
                'outer_cuda_configuration': getattr(
                    op_outer, 'cuda_apply_configuration', None,
                ),
                'inner_cuda_configuration': getattr(
                    op_inner, 'cuda_apply_configuration', None,
                ),
            }

        if verbose:
            print(f"  [IR-batched] load case {J+1}/6: outer={info['outer_iters']}, "
                  f"inner={info['inner_iters']}, "
                  f"max_res={info['rel_res'].max():.3e}, {elapsed:.2f}s")

    C_eff_all = 0.5 * (C_eff_all + np.transpose(C_eff_all, (0, 2, 1)))
    return C_eff_all, info_per_sample


def compute_effective_stiffness_ir_adaptive_batched(
    op_outer, op_inner, *,
    outer_tol=1e-8,
    inner_tols=(0.1, 0.03, 0.01),
    max_outer=30,
    max_inner=80,
    adaptive_max_inner=160,
    stagnation_factor=0.3,
    stall_factor=0.9,
    stall_patience=3,
    inner_pcg_update_mode="eager",
    inner_convergence_check_stride=1,
    inner_verify_final_residual=False,
    verbose=False,
):
    """Adaptive IR driver over 6 load cases × B samples."""
    Bsz = op_outer.B
    C_eff_all = np.zeros((Bsz, 6, 6), dtype=np.float64)
    info_per_sample = [[None] * 6 for _ in range(Bsz)]

    for J in range(6):
        E = np.zeros(6, dtype=np.float64)
        E[J] = 1.0
        t0 = time.time()
        rhs = op_outer.compute_rhs(E)
        u_star, info = solve_ir_adaptive_batched(
            op_outer, op_inner, rhs,
            outer_tol=outer_tol,
            inner_tols=inner_tols,
            max_outer=max_outer,
            max_inner=max_inner,
            adaptive_max_inner=adaptive_max_inner,
            stagnation_factor=stagnation_factor,
            stall_factor=stall_factor,
            stall_patience=stall_patience,
            inner_pcg_update_mode=inner_pcg_update_mode,
            inner_convergence_check_stride=inner_convergence_check_stride,
            inner_verify_final_residual=inner_verify_final_residual,
            verbose=verbose,
        )
        u_star = op_outer.project_zero_mean(u_star)
        avg_sig = op_outer.compute_avg_stress(u_star, E)
        elapsed = time.time() - t0

        C_eff_all[:, :, J] = avg_sig

        per_load_elapsed = elapsed / Bsz
        for b in range(Bsz):
            info_per_sample[b][J] = {
                'outer_iters': info['outer_iters'],
                'inner_iters': info['inner_iters'],
                'rel_res': float(info['rel_res'][b]),
                'elapsed_s': float(per_load_elapsed),
                'converged': bool(info['converged']),
                'stop_reason': info['stop_reason'],
                'adaptive': True,
                'inner_tols': info['inner_tols'],
                'final_inner_tol': info['final_inner_tol'],
                'max_inner_used': info['max_inner_used'],
                'inner_pcg_update_mode': info['inner_pcg_update_mode'],
                'outer_apply_backend': getattr(
                    op_outer, 'apply_backend', 'unspecified',
                ),
                'inner_apply_backend': getattr(
                    op_inner, 'apply_backend', 'unspecified',
                ),
                'outer_cuda_configuration': getattr(
                    op_outer, 'cuda_apply_configuration', None,
                ),
                'inner_cuda_configuration': getattr(
                    op_inner, 'cuda_apply_configuration', None,
                ),
                'history': info['history'],
            }

        if verbose:
            print(f"  [IR-adaptive] load case {J+1}/6: outer={info['outer_iters']}, "
                  f"inner={info['inner_iters']}, stop={info['stop_reason']}, "
                  f"max_res={info['rel_res'].max():.3e}, {elapsed:.2f}s")

    C_eff_all = 0.5 * (C_eff_all + np.transpose(C_eff_all, (0, 2, 1)))
    return C_eff_all, info_per_sample
