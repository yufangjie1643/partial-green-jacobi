# -*- coding: utf-8 -*-
"""Element-node-DOF connectivity tables for a periodic voxel mesh (pure NumPy, backend-agnostic)."""

from __future__ import annotations

import numpy as np

from .geometry import LOCAL_OFFSET


def build_dof_table(Nx: int, Ny: int, Nz: int):
    """
    Return (elem_node_ids, elem_dof_ids):

    elem_node_ids : (n_elem, 8) int64
    elem_dof_ids  : (n_elem, 24) int64

    Nodes are linearised as (k, j, i) -> k*Ny*Nx + j*Nx + i; periodic BC via mod.
    """
    ks, js, is_ = np.meshgrid(
        np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing='ij'
    )
    ks = ks.reshape(-1)
    js = js.reshape(-1)
    is_ = is_.reshape(-1)
    n_elem = Nx * Ny * Nz

    elem_node_ids = np.zeros((n_elem, 8), dtype=np.int64)
    for ln, (dx_, dy_, dz_) in enumerate(LOCAL_OFFSET):
        iN = (is_ + dx_) % Nx
        jN = (js + dy_) % Ny
        kN = (ks + dz_) % Nz
        elem_node_ids[:, ln] = (kN * Ny + jN) * Nx + iN

    elem_dof_ids = np.zeros((n_elem, 24), dtype=np.int64)
    for ln in range(8):
        for d in range(3):
            elem_dof_ids[:, 3 * ln + d] = 3 * elem_node_ids[:, ln] + d

    return elem_node_ids, elem_dof_ids
