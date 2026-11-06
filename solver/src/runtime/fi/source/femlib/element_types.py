# -*- coding: utf-8 -*-
"""Element stiffness kernels for regular voxel hexahedra.

The default M-free operator uses the standard C3D8/Q1 full-integration
hexahedron. ``c3d8r`` selects one-point reduced integration without hourglass
control. ``c3d8i`` adds nine Wilson-style incompatible displacement modes and
statically condenses the internal amplitudes at element level. The result is
still a 24x24 corner-node stiffness matrix that can be accumulated into the
same periodic node stencil.
"""

from __future__ import annotations

import numpy as np

from .geometry import GAUSS, LOCAL_OFFSET, make_B_matrices


SUPPORTED_ELEMENT_TYPES = ("c3d8", "c3d8r", "c3d8i")
N_INCOMPATIBLE_MODES_C3D8I = 9


def normalize_element_type(element_type: str | None) -> str:
    key = "c3d8" if element_type is None else str(element_type).strip().lower()
    aliases = {
        "q1": "c3d8",
        "full": "c3d8",
        "full_integration": "c3d8",
        "c3d8_full": "c3d8",
        "reduced": "c3d8r",
        "reduced_integration": "c3d8r",
        "c3d8_reduced": "c3d8r",
        "incompatible": "c3d8i",
        "c3d8i_wilson": "c3d8i",
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_ELEMENT_TYPES:
        valid = ", ".join(SUPPORTED_ELEMENT_TYPES)
        raise ValueError(f"unknown element_type {element_type!r}; valid values are: {valid}")
    return key


def normalize_input_element_type(element_type: str | None) -> str:
    """Return canonical uppercase metadata for an Abaqus element type."""
    key = "c3d8" if element_type is None else str(element_type).strip().lower()
    if key not in SUPPORTED_ELEMENT_TYPES:
        valid = ", ".join(name.upper() for name in SUPPORTED_ELEMENT_TYPES)
        raise ValueError(
            f"unknown element_type {element_type!r}; valid input types are: {valid}"
        )
    return key.upper()


def parse_abaqus_element_type(keyword_line: str) -> str | None:
    """Parse canonical element metadata from one Abaqus keyword line."""
    tokens = str(keyword_line).strip().split(",")
    if not tokens or tokens[0].strip().lower() != "*element":
        return None

    declared_types = []
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator and key.strip().lower() == "type":
            declared_types.append(normalize_input_element_type(value))

    if not declared_types:
        return normalize_input_element_type(None)
    if len(set(declared_types)) != 1:
        names = ", ".join(declared_types)
        raise ValueError(f"multiple element types declared in one *Element header: {names}")
    return declared_types[0]


def merge_input_element_type(current: str | None, declared: str) -> str:
    """Merge one canonical header value, rejecting heterogeneous meshes."""
    if current is not None and current != declared:
        raise ValueError(
            f"multiple element types are not supported: {current}, {declared}"
        )
    return declared


def make_incompatible_B_matrices(hx: float, hy: float, hz: float) -> np.ndarray:
    """Build strain-displacement matrices for the condensed C3D8I modes.

    Mode layout is grouped by displacement component, then by the local
    quadratic mode ``xi^2-1``, ``eta^2-1``, ``zeta^2-1``:

    ``[ux_xi, ux_eta, ux_zeta, uy_xi, ..., uz_zeta]``.
    """
    hx = float(hx)
    hy = float(hy)
    hz = float(hz)
    if hx <= 0.0 or hy <= 0.0 or hz <= 0.0:
        raise ValueError("voxel dimensions must be positive")

    B = np.zeros((8, 6, N_INCOMPATIBLE_MODES_C3D8I), dtype=np.float64)
    for g, (xi, eta, zeta) in enumerate(GAUSS):
        grads = (
            np.array([4.0 * xi / hx, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 4.0 * eta / hy, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 4.0 * zeta / hz], dtype=np.float64),
        )
        for component in range(3):
            for local_mode, grad in enumerate(grads):
                col = 3 * component + local_mode
                dx, dy, dz = grad
                if component == 0:
                    B[g, 0, col] = dx
                    B[g, 4, col] = dz
                    B[g, 5, col] = dy
                elif component == 1:
                    B[g, 1, col] = dy
                    B[g, 3, col] = dz
                    B[g, 5, col] = dx
                else:
                    B[g, 2, col] = dz
                    B[g, 3, col] = dy
                    B[g, 4, col] = dx
    return B


def make_element_stiffness(
    C_voigt: np.ndarray,
    voxel_size: tuple[float, float, float],
    *,
    element_type: str = "c3d8",
) -> np.ndarray:
    """Return a 24x24 corner-node stiffness for one voxel element.

    C3D8R is the raw one-point reference formulation and intentionally has no
    hourglass stabilization. Production C3D8R-HG operators live in
    :mod:`femlib.c3d8r` and :mod:`femlib.batched`.
    """
    element_type = normalize_element_type(element_type)
    C = np.asarray(C_voigt, dtype=np.float64)
    if C.shape != (6, 6):
        raise ValueError(f"C_voigt must have shape (6, 6), got {C.shape}")
    hx, hy, hz = (float(value) for value in voxel_size)
    geometry_element_type = "c3d8r" if element_type == "c3d8r" else "c3d8"
    B_u, detJ = make_B_matrices(hx, hy, hz, element_type=geometry_element_type)

    K_uu = np.einsum("gIp,IJ,gJq->pq", B_u, C, B_u) * detJ
    if element_type in {"c3d8", "c3d8r"}:
        return 0.5 * (K_uu + K_uu.T)

    B_i = make_incompatible_B_matrices(hx, hy, hz)
    K_ui = np.einsum("gIp,IJ,gJm->pm", B_u, C, B_i) * detJ
    K_ii = np.einsum("gIm,IJ,gJn->mn", B_i, C, B_i) * detJ
    condensed = K_uu - K_ui @ np.linalg.solve(K_ii, K_ui.T)
    return 0.5 * (condensed + condensed.T)


def element_stiffness_to_tap_blocks(K_elem: np.ndarray) -> np.ndarray:
    """Convert a 24x24 element matrix to 27 periodic node-stencil tap blocks.

    The returned array has shape ``(27, 3, 3)`` with tap order
    ``tap = iz * 9 + iy * 3 + ix`` and offsets in ``{-1, 0, 1}^3``.
    """
    K = np.asarray(K_elem, dtype=np.float64)
    if K.shape != (24, 24):
        raise ValueError(f"K_elem must have shape (24, 24), got {K.shape}")

    blocks = np.zeros((27, 3, 3), dtype=np.float64)
    for a in range(8):
        oa_x, oa_y, oa_z = LOCAL_OFFSET[a]
        for b in range(8):
            ob_x, ob_y, ob_z = LOCAL_OFFSET[b]
            dx = int(ob_x - oa_x)
            dy = int(ob_y - oa_y)
            dz = int(ob_z - oa_z)
            tap = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)
            blocks[tap] += K[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
    return blocks
