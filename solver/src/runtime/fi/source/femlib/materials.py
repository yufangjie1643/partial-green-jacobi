# -*- coding: utf-8 -*-
"""
Material utilities: Voigt/Mandel stiffness conversions, isotropic/orthotropic
stiffness construction, Bond-rotation of stiffness tensors, engineering-constant
extraction.

Voigt index order: 0→11, 1→22, 2→33, 3→23, 4→13, 5→12
"""

import numpy as np


# ============================================================
# Notation conventions
# ============================================================
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
MANDEL_W = np.array([1., 1., 1., np.sqrt(2.), np.sqrt(2.), np.sqrt(2.)])
MANDEL_W_INV = 1.0 / MANDEL_W


# ============================================================
# Material utilities
# ============================================================
def engineering_to_voigt(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
    """Orthotropic engineering constants -> 6x6 Voigt stiffness."""
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    return np.linalg.inv(S)


def isotropic_to_voigt(E, nu):
    """Isotropic (E, nu) -> 6x6 Voigt stiffness."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] += 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def voigt_to_mandel(C_voigt):
    """6x6 Voigt stiffness -> Mandel stiffness."""
    D = np.diag(MANDEL_W)
    return D @ C_voigt @ D


def mandel_to_voigt(C_mandel):
    """6x6 Mandel stiffness -> Voigt stiffness."""
    D = np.diag(MANDEL_W_INV)
    return D @ C_mandel @ D


def _mandel_basis_tensors():
    basis = np.zeros((6, 3, 3), dtype=np.float64)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0
    basis[3, 1, 2] = basis[3, 2, 1] = 1.0 / np.sqrt(2.0)
    basis[4, 0, 2] = basis[4, 2, 0] = 1.0 / np.sqrt(2.0)
    basis[5, 0, 1] = basis[5, 1, 0] = 1.0 / np.sqrt(2.0)
    return basis


_MANDEL_BASIS_TENSORS = _mandel_basis_tensors()


def mandel_rotation_matrix_batch(R):
    """Build Mandel-space rotation matrices induced by physical rotations.

    ``R[..., :, k]`` is the kth local basis vector expressed in global
    coordinates. Symmetric tensor components transform as
    ``A_global = R @ A_local @ R.T``.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must end with shape (3, 3), got {R.shape}")
    flat = R.reshape((-1, 3, 3))
    Rt = np.transpose(flat, (0, 2, 1))
    Q = np.empty((flat.shape[0], 6, 6), dtype=np.float64)
    for j, basis_j in enumerate(_MANDEL_BASIS_TENSORS):
        rotated = flat @ basis_j @ Rt
        for i, basis_i in enumerate(_MANDEL_BASIS_TENSORS):
            Q[:, i, j] = np.einsum("ab,nab->n", basis_i, rotated)
    return Q.reshape(R.shape[:-2] + (6, 6))


def rotate_mandel_batch(C_local, R):
    """Rotate Mandel stiffness matrices with ``C_global = Q C_local Q.T``."""
    Q = mandel_rotation_matrix_batch(R).reshape((-1, 6, 6))
    C = np.asarray(C_local, dtype=np.float64)
    if C.shape == (6, 6):
        rotated = np.einsum("nij,jk,nlk->nil", Q, C, Q)
    elif C.ndim == 3 and C.shape[1:] == (6, 6):
        C_flat = C.reshape((-1, 6, 6))
        if C_flat.shape[0] != Q.shape[0]:
            raise ValueError("Batched C_local and R must have the same leading size")
        rotated = np.einsum("nij,njk,nlk->nil", Q, C_flat, Q)
    else:
        raise ValueError(f"C_local must be (6, 6) or (N, 6, 6), got {C.shape}")
    return rotated.reshape(R.shape[:-2] + (6, 6))


def rotate_voigt_batch(C_local, R):
    """
    Batched rotation of engineering-Voigt stiffness through Mandel space.

    This avoids the stress/strain shear-scaling ambiguity of engineering Voigt:
    C_M = D C_V D, C_M_global = Q_M C_M_local Q_M.T, C_V = D^-1 C_M D^-1.

    C_local: (6, 6) local stiffness
    R: (N, 3, 3) rotation matrices
    Returns: (N, 6, 6) global stiffness
    """
    C_mandel = voigt_to_mandel(np.asarray(C_local, dtype=np.float64))
    C_global_mandel = rotate_mandel_batch(C_mandel, R)
    return (
        C_global_mandel
        * MANDEL_W_INV[None, :, None]
        * MANDEL_W_INV[None, None, :]
    )


def build_rotation_matrices(orient1, orient2):
    """
    Build orthogonal rotation matrices from direction vectors.
    orient1: (..., 3) fibre direction (local x axis)
    orient2: (..., 3) cross-section reference direction (used to fix the local y/z axes)
    Returns: (..., 3, 3) rotation matrices R, where R[..., :, k] is the k-th local axis expressed in global coordinates.
    """
    e1 = orient1 / np.linalg.norm(orient1, axis=-1, keepdims=True)
    e3 = np.cross(e1, orient2)
    e3_norm = np.linalg.norm(e3, axis=-1, keepdims=True)
    e3_norm = np.maximum(e3_norm, 1e-14)
    e3 = e3 / e3_norm
    e2 = np.cross(e3, e1)
    R = np.zeros(orient1.shape[:-1] + (3, 3))
    R[..., :, 0] = e1
    R[..., :, 1] = e2
    R[..., :, 2] = e3
    return R


# ============================================================
# Effective engineering-constants extraction
# ============================================================
def extract_engineering_constants(C_voigt):
    """Extract engineering constants from a 6x6 Voigt stiffness matrix."""
    S = np.linalg.inv(C_voigt)
    E11 = 1.0 / S[0, 0]
    E22 = 1.0 / S[1, 1]
    E33 = 1.0 / S[2, 2]
    G23 = 1.0 / S[3, 3]
    G13 = 1.0 / S[4, 4]
    G12 = 1.0 / S[5, 5]
    nu12 = -S[1, 0] * E11
    nu13 = -S[2, 0] * E11
    nu21 = -S[0, 1] * E22
    nu23 = -S[2, 1] * E22
    nu31 = -S[0, 2] * E33
    nu32 = -S[1, 2] * E33
    return {
        'E11': E11, 'E22': E22, 'E33': E33,
        'G12': G12, 'G13': G13, 'G23': G23,
        'V12': nu12, 'V13': nu13, 'V21': nu21,
        'V23': nu23, 'V31': nu31, 'V32': nu32,
    }
