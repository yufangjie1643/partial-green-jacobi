# -*- coding: utf-8 -*-
"""
C3D8/C3D8R hexahedral element geometry: node layout, quadrature, B matrix.

Shared by all FEM backends (numpy / torch / stencil).
"""

from __future__ import annotations
import numpy as np


# C3D8 nodes follow the Abaqus convention (bottom face CCW 1-4, top face CCW 5-8)
NODE_REF = np.array([
    [-1, -1, -1],  # 1
    [ 1, -1, -1],  # 2
    [ 1,  1, -1],  # 3
    [-1,  1, -1],  # 4
    [-1, -1,  1],  # 5
    [ 1, -1,  1],  # 6
    [ 1,  1,  1],  # 7
    [-1,  1,  1],  # 8
], dtype=np.float64)

# 2x2x2 Gauss points, all weights equal to 1 (C3D8 full integration).
_g = 1.0 / np.sqrt(3.0)
GAUSS_C3D8 = np.array([
    [-_g, -_g, -_g], [ _g, -_g, -_g],
    [ _g,  _g, -_g], [-_g,  _g, -_g],
    [-_g, -_g,  _g], [ _g, -_g,  _g],
    [ _g,  _g,  _g], [-_g,  _g,  _g],
], dtype=np.float64)
GAUSS_W_C3D8 = np.ones(8, dtype=np.float64)

# One-point reduced integration (C3D8R without Abaqus hourglass control).
GAUSS_C3D8R = np.zeros((1, 3), dtype=np.float64)
GAUSS_W_C3D8R = np.array([8.0], dtype=np.float64)

# Backward-compatible names for the default full-integration rule.
GAUSS = GAUSS_C3D8
GAUSS_W = GAUSS_W_C3D8

# Local node offsets consistent with NODE_REF (the 8 corners of an element in the solid mesh)
LOCAL_OFFSET = np.array([
    [0, 0, 0],  # n1
    [1, 0, 0],  # n2
    [1, 1, 0],  # n3
    [0, 1, 0],  # n4
    [0, 0, 1],  # n5
    [1, 0, 1],  # n6
    [1, 1, 1],  # n7
    [0, 1, 1],  # n8
], dtype=np.int64)


SUPPORTED_HEX_ELEMENT_TYPES = ("C3D8", "C3D8R")


def normalize_hex_element_type(element_type="C3D8"):
    """Normalize the supported 8-node hexahedral element type name."""
    name = "C3D8" if element_type is None else str(element_type).strip().upper()
    if name not in SUPPORTED_HEX_ELEMENT_TYPES:
        supported = ", ".join(SUPPORTED_HEX_ELEMENT_TYPES)
        raise ValueError(f"unsupported 8-node hex element_type={element_type!r}; supported: {supported}")
    return name


def quadrature_rule(element_type="C3D8"):
    """Return reference Gauss points and weights for a supported element type."""
    element_type = normalize_hex_element_type(element_type)
    if element_type == "C3D8R":
        return GAUSS_C3D8R, GAUSS_W_C3D8R
    return GAUSS_C3D8, GAUSS_W_C3D8


def shape_derivs_at_gp(gp_xyz):
    """
    Evaluate dN/dξ, dN/dη, dN/dζ at the 8 nodes in reference coordinates (ξ, η, ζ).
    Returns an (8_node, 3_ξηζ) array.
    """
    xi, eta, zeta = gp_xyz
    out = np.zeros((8, 3), dtype=np.float64)
    for i, (xi_i, eta_i, zeta_i) in enumerate(NODE_REF):
        out[i, 0] = 0.125 * xi_i   * (1.0 + eta_i  * eta) * (1.0 + zeta_i * zeta)
        out[i, 1] = 0.125 * eta_i  * (1.0 + xi_i   * xi)  * (1.0 + zeta_i * zeta)
        out[i, 2] = 0.125 * zeta_i * (1.0 + xi_i   * xi)  * (1.0 + eta_i  * eta)
    return out


def make_B_matrices(hx, hy, hz, element_type="C3D8"):
    """
    Build the B matrices for a voxel of size (hx, hy, hz).

    Returns
    -------
    B : (n_gp, 6, 24)
        B[g] @ u_elem(24,) = strain_voigt(6,) in Voigt engineering-shear form.
    quad_weight : float
        Weight-scaled Jacobian determinant per quadrature point. For C3D8 this
        is hx*hy*hz/8; for C3D8R this is hx*hy*hz because the one Gauss point
        has reference weight 8.
    """
    gps, weights = quadrature_rule(element_type)
    if not np.allclose(weights, weights[0]):
        raise ValueError("make_B_matrices currently expects equal quadrature weights")

    B = np.zeros((len(gps), 6, 24), dtype=np.float64)
    for g, gp in enumerate(gps):
        dN = shape_derivs_at_gp(gp)  # (8, 3) in (ξ, η, ζ)
        dNx = dN[:, 0] * (2.0 / hx)
        dNy = dN[:, 1] * (2.0 / hy)
        dNz = dN[:, 2] * (2.0 / hz)
        for i in range(8):
            c = 3 * i
            # Normal strains
            B[g, 0, c + 0] = dNx[i]
            B[g, 1, c + 1] = dNy[i]
            B[g, 2, c + 2] = dNz[i]
            # γ23 = ∂u2/∂z + ∂u3/∂y
            B[g, 3, c + 1] = dNz[i]
            B[g, 3, c + 2] = dNy[i]
            # γ13 = ∂u1/∂z + ∂u3/∂x
            B[g, 4, c + 0] = dNz[i]
            B[g, 4, c + 2] = dNx[i]
            # γ12 = ∂u1/∂y + ∂u2/∂x
            B[g, 5, c + 0] = dNy[i]
            B[g, 5, c + 1] = dNx[i]
    detJ = hx * hy * hz / 8.0
    return B, detJ * float(weights[0])
