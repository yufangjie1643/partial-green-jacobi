# -*- coding: utf-8 -*-
"""Batched homogenisation driver: 6 load cases, each processing B samples in parallel."""

from __future__ import annotations

import os
import time
import numpy as np

from .pcg import solve_pcg_batched


def compute_effective_stiffness_batched(
    op,
    *,
    tol=1e-6,
    max_iter=2000,
    verbose=False,
    pcg_update_mode="eager",
    convergence_check_stride=1,
    convergence_check_adaptive_factor=None,
    verify_final_residual=False,
):
    """Run 6 load cases in batched mode over `op.B` samples.

    Returns
    -------
    C_eff_batch : np.ndarray (B, 6, 6)
    info_per_sample : list[list[dict]]  outer len=B, inner len=6
                      each entry records iterations, the reported residual,
                      whether it meets ``tol``, whether that residual was
                      explicitly recomputed, and elapsed time.
    """
    Bsz = op.B
    C_eff_all = np.zeros((Bsz, 6, 6), dtype=np.float64)
    info_per_sample = [[None] * 6 for _ in range(Bsz)]
    log_every = int(os.environ.get("MFREE_PCG_LOG_EVERY", "50"))

    for J in range(6):
        E = np.zeros(6, dtype=np.float64)
        E[J] = 1.0
        t0 = time.time()
        rhs = op.compute_rhs(E)                                  # (B, n_dofs)
        u_star, n_iter, rel_res = solve_pcg_batched(
            op, rhs, tol=tol, max_iter=max_iter, verbose=verbose,
            log_every=log_every,
            update_mode=pcg_update_mode,
            convergence_check_stride=convergence_check_stride,
            convergence_check_adaptive_factor=convergence_check_adaptive_factor,
            verify_final_residual=verify_final_residual,
        )
        u_star = op.project_zero_mean(u_star)
        avg_sig = op.compute_avg_stress(u_star, E)               # (B, 6)
        elapsed = time.time() - t0

        C_eff_all[:, :, J] = avg_sig

        rel_res_np = rel_res.detach().cpu().numpy().astype(np.float64)
        per_load_elapsed = elapsed / Bsz  # Averaged over samples
        for b in range(Bsz):
            info_per_sample[b][J] = {
                'iters':     int(n_iter),
                'rel_res':   float(rel_res_np[b]),
                'converged': bool(rel_res_np[b] < tol),
                'residual_verified': bool(verify_final_residual),
                'elapsed_s': float(per_load_elapsed),
            }

        if verbose:
            print(f"  [Batched] load case {J+1}/6: iters={n_iter}, "
                  f"max_res={rel_res_np.max():.3e}, {elapsed:.2f}s")

    # Symmetrise independently for each sample
    C_eff_all = 0.5 * (C_eff_all + np.transpose(C_eff_all, (0, 2, 1)))
    return C_eff_all, info_per_sample
