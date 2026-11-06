# -*- coding: utf-8 -*-
"""Size metadata for a regular periodic voxel mesh."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshInfo:
    """Size metadata for a regular periodic voxel mesh.

    Attributes
    ----------
    Nx, Ny, Nz : int    grid size (element count per axis)
    dx, dy, dz : float  voxel dimensions
    """
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float

    @property
    def n_elem(self) -> int:
        return self.Nx * self.Ny * self.Nz

    @property
    def n_nodes(self) -> int:
        # Periodic topology: node count equals element count
        return self.n_elem

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def V_elem(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def V_total(self) -> float:
        return self.n_elem * self.V_elem

    @classmethod
    def from_args(cls, voxel_size, grid_shape) -> "MeshInfo":
        Nz, Ny, Nx = map(int, grid_shape)
        dx, dy, dz = map(float, voxel_size)
        return cls(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz)
