"""Baseline C3D8R-HG Zeman FE-Green preconditioner implementation.

This module is the only production-code research surface exposed by the
Praxist task. The implementation below is a behavior-preserving extraction
from :mod:`femlib.batched.c3d8r_hourglass_operator`.
"""

from __future__ import annotations

import numpy as np
import torch

from femlib.c3d8r import build_c3d8r_reference_stencil


def apply_preconditioner(operator, r_flat):
    """Apply the configured stationary preconditioner to a flat residual."""

    if operator._fft_enabled:
        return operator._apply_fft_precond(r_flat)
    if operator._M_inv_cache is None:
        nonzero = operator._diag_cache != 0
        safe_diagonal = torch.where(
            nonzero,
            operator._diag_cache,
            torch.ones_like(operator._diag_cache),
        )
        operator._M_inv_cache = torch.where(
            nonzero,
            safe_diagonal.reciprocal(),
            torch.zeros_like(operator._diag_cache),
        )
    return r_flat * operator._M_inv_cache


def enable_fft_preconditioner(
    operator,
    *,
    ref_strategy="zeman",
    C0_voigt=None,
    verbose=False,
    spectrum_mode="complex_full",
    validation_mode="legacy_eig",
    allow_symbol_sharing=True,
):
    """Build the baseline homogeneous-reference FE-Green inverse symbol."""

    from femlib.batched.stencil_operator import BatchedStencilOperator

    if spectrum_mode not in {"complex_full", "real_half"}:
        raise ValueError(f"unknown FFT spectrum mode {spectrum_mode!r}")
    effective_allow_symbol_sharing = bool(allow_symbol_sharing)

    if C0_voigt is not None:
        C0 = np.array(C0_voigt, dtype=np.float64, copy=True)
        if C0.ndim == 2:
            C0 = np.broadcast_to(C0, (operator.B, 6, 6)).copy()
        if C0.shape != (operator.B, 6, 6):
            raise ValueError(
                "C0_voigt must be (6,6) or (B,6,6), "
                f"got {C0.shape}"
            )
        C0_voigt = C0
    else:
        C0 = BatchedStencilOperator._compute_reference_medium(
            operator,
            ref_strategy,
            verbose=verbose,
        )
        C0_voigt = C0

    voxel_size = (operator.mesh.dx, operator.mesh.dy, operator.mesh.dz)
    S0_center = np.stack(
        [
            build_c3d8r_reference_stencil(
                C0[b],
                voxel_size,
                hourglass_coefficient=operator.hourglass_coefficient,
            )
            for b in range(operator.B)
        ],
        axis=0,
    )
    operator._fft_symbol_shared = False
    if (
        effective_allow_symbol_sharing
        and operator.B > 1
        and np.array_equal(
            S0_center,
            np.broadcast_to(S0_center[:1], S0_center.shape),
        )
    ):
        S0_center = S0_center[:1].copy()
        operator._fft_symbol_shared = True

    BatchedStencilOperator._build_fft_symbol_inv(
        operator,
        S0_center,
        verbose=verbose,
        validate_nonzero=True,
        spectrum_mode=spectrum_mode,
        validation_mode=validation_mode,
    )
    operator._fft_enabled = True
    operator._fft_args = {
        "ref_strategy": ref_strategy,
        "C0_voigt": C0_voigt,
        "spectrum_mode": spectrum_mode,
        "validation_mode": validation_mode,
        "allow_symbol_sharing": effective_allow_symbol_sharing,
    }
    operator._M_inv_cache = None
