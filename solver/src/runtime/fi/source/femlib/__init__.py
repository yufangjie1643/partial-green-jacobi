# -*- coding: utf-8 -*-
"""Minimal C3D8R-HG voxel homogenization library."""

from .geometry import NODE_REF, GAUSS, LOCAL_OFFSET, make_B_matrices
from .materials import (
    isotropic_to_voigt, engineering_to_voigt,
    voigt_to_mandel, mandel_to_voigt,
    mandel_rotation_matrix_batch, rotate_mandel_batch,
    rotate_voigt_batch, build_rotation_matrices,
    extract_engineering_constants,
)
from .c3d8r import (
    validate_hourglass_coefficient,
    hourglass_basis,
    hourglass_scale,
    make_c3d8r_element_stiffness,
    build_c3d8r_reference_stencil,
)
from .mesh import TexGenMesh, build_voigt_stiffness_field

# Main FEM solver (CUDA + Triton).
try:
    from .fem_batched import FEMHomogenizerBatched
except ImportError:
    pass
