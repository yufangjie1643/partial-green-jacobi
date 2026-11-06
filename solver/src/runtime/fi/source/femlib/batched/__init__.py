# -*- coding: utf-8 -*-
"""Batched FEM homogenization: stack N samples along a leading batch dim and
solve on GPU with shared mesh / shared B matrices / shared Laplacian structure
but per-sample stiffness fields.

Only supports CUDA + Triton (that's where batching pays off).  Importing this
package on CPU-only environments keeps lightweight helpers available.
"""

from .mesh_info import MeshInfo
from .c3d8r_hourglass_operator import (
    BatchedC3D8RHourglassTwoStageOperator,
    c3d8r_hourglass_apply_reference_numpy,
    estimate_c3d8r_two_stage_storage,
)

try:
    from .stencil_operator import BatchedStencilOperator
    from .pcg import solve_pcg_batched
    from .driver import compute_effective_stiffness_batched
    from .ir import (
        compute_effective_stiffness_ir_adaptive_batched,
        compute_effective_stiffness_ir_batched,
        solve_ir_adaptive_batched,
        solve_ir_batched,
    )
except ImportError:
    pass
