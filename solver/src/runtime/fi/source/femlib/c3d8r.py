"""CPU/reference definitions for the stabilized C3D8R-HG element.

This module provides the low-level element algebra and reference stencil used
by later implementations. It is not a high-level solver or wrapper.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import LOCAL_OFFSET, NODE_REF, make_B_matrices

Array = np.ndarray
VoxelSize = tuple[float, float, float]

_PATTERNS = np.stack(
    (
        NODE_REF[:, 0] * NODE_REF[:, 1],
        NODE_REF[:, 1] * NODE_REF[:, 2],
        NODE_REF[:, 2] * NODE_REF[:, 0],
        NODE_REF[:, 0] * NODE_REF[:, 1] * NODE_REF[:, 2],
    ),
    axis=1,
) / math.sqrt(8.0)


def validate_hourglass_coefficient(value: float) -> float:
    """Return a finite positive coefficient for C3D8R-HG stabilization."""
    coefficient = float(value)
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("hourglass_coefficient must be finite and positive")
    return coefficient


def hourglass_basis() -> Array:
    """Return the normalized 24-by-12 C3D8R-HG basis matrix."""
    H = np.zeros((24, 12), dtype=np.float64)
    column = 0
    for component in range(3):
        for mode in range(4):
            H[component::3, column] = _PATTERNS[:, mode]
            column += 1
    return H


def hourglass_scale(C_voigt: Array, voxel_size: VoxelSize, coefficient: float) -> float:
    """Return the rotation-invariant scalar penalty scale for C3D8R-HG modes."""
    C = np.asarray(C_voigt, dtype=np.float64)
    if C.shape != (6, 6):
        raise ValueError(f"C_voigt must have shape (6, 6), got {C.shape}")
    hx, hy, hz = (float(v) for v in voxel_size)
    hbar = (hx + hy + hz) / 3.0
    normal = float(C[0, 0] + C[1, 1] + C[2, 2])
    first_invariant = normal + 2.0 * float(C[0, 1] + C[0, 2] + C[1, 2])
    second_invariant = normal + 2.0 * float(C[3, 3] + C[4, 4] + C[5, 5])
    material_scale = (first_invariant + 7.0 * second_invariant) / 60.0
    return float(coefficient) * max(material_scale, 1e-12) * (hx * hy * hz) / (hbar * hbar)


def make_c3d8r_element_stiffness(
    C_voigt: Array,
    voxel_size: VoxelSize,
    *,
    hourglass_coefficient: float = 0.1,
) -> Array:
    """Build the raw or hourglass-stabilized low-level C3D8R stiffness."""
    C = np.asarray(C_voigt, dtype=np.float64)
    B, weighted_detJ = make_B_matrices(*voxel_size, element_type="C3D8R")
    K = np.einsum("gip,ij,gjq->pq", B, C, B) * weighted_detJ
    coefficient = float(hourglass_coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("hourglass_coefficient must be finite and non-negative in the low-level reference API")
    if coefficient > 0.0:
        H = hourglass_basis()
        K += hourglass_scale(C, voxel_size, coefficient) * (H @ H.T)
    return 0.5 * (K + K.T)


def build_c3d8r_reference_stencil(
    C_voigt: Array,
    voxel_size: VoxelSize,
    *,
    hourglass_coefficient: float = 0.1,
) -> Array:
    """Build the low-level 27-tap reference stencil for C3D8R-HG."""
    K = make_c3d8r_element_stiffness(
        C_voigt, voxel_size, hourglass_coefficient=hourglass_coefficient,
    )
    stencil = np.zeros((27, 3, 3), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            delta = LOCAL_OFFSET[b] - LOCAL_OFFSET[a]
            tap = int((delta[2] + 1) * 9 + (delta[1] + 1) * 3 + delta[0] + 1)
            stencil[tap] += K[3*a:3*a+3, 3*b:3*b+3]
    return stencil
