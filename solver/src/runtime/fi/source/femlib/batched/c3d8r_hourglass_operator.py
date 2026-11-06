# -*- coding: utf-8 -*-
"""Two-stage C3D8R + hourglass matrix-free apply prototype.

This is an experimental low-storage operator for the reduced-integration
hexahedron.  It does not materialize the full ``(27, 9, Nz, Ny, Nx)`` stencil.
Instead, each apply runs:

1. an element kernel that computes ``elem_force[B, Nelem, 8, 3]`` from
   ``u`` and the per-element stiffness ``C``;
2. a node gather kernel that sums the eight adjacent element-force entries.

The low-frequency gather/scatter path supplies diagonal assembly, macroscopic
loading, stress recovery, and Jacobi preconditioning without storing element
connectivity.  Repeated matvecs continue to use the two Triton kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from femlib.c3d8r import hourglass_basis
from femlib.geometry import LOCAL_OFFSET, make_B_matrices

from .mesh_info import MeshInfo

try:  # Keep CPU-only helper imports working when torch/triton are unavailable.
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    triton = None
    tl = None


_INT32_MAX = (1 << 31) - 1
_FULL_STENCIL_SCALARS_PER_ELEM = 27 * 9
_BASE_PERSISTENT_SCALARS_PER_ELEM = 36 + 1 + 3
_JACOBI_PERSISTENT_SCALARS_PER_ELEM = _BASE_PERSISTENT_SCALARS_PER_ELEM + 3
_TRANSIENT_SCALARS_PER_ELEM = 8 * 3
_DIAGONAL_ROUNDOFF_TERMS = 6 * 6 + 12 + 8 * 8


def estimate_c3d8r_two_stage_storage(n_elem: int, bytes_per_scalar: int = 4) -> dict[str, int]:
    """Estimate storage for the two-stage C3D8R apply.

    Base persistent storage includes full anisotropic ``C``, one hourglass
    scale, and the three-DOF diagonal per element.  Jacobi persistent storage
    additionally includes the lazy inverse diagonal.  Transient storage is the
    ``elem_force`` buffer used during a matvec.  Input/output vectors and
    geometry constants are excluded.
    """

    n = int(n_elem)
    b = int(bytes_per_scalar)
    base_persistent = _BASE_PERSISTENT_SCALARS_PER_ELEM * n
    jacobi_persistent = _JACOBI_PERSISTENT_SCALARS_PER_ELEM * n
    transient = _TRANSIENT_SCALARS_PER_ELEM * n
    full = _FULL_STENCIL_SCALARS_PER_ELEM * n
    return {
        "base_persistent_scalars_per_elem": _BASE_PERSISTENT_SCALARS_PER_ELEM,
        "jacobi_persistent_scalars_per_elem": _JACOBI_PERSISTENT_SCALARS_PER_ELEM,
        "persistent_scalars_per_elem": _BASE_PERSISTENT_SCALARS_PER_ELEM,
        "transient_scalars_per_elem": _TRANSIENT_SCALARS_PER_ELEM,
        "total_apply_scalars_per_elem": _BASE_PERSISTENT_SCALARS_PER_ELEM + _TRANSIENT_SCALARS_PER_ELEM,
        "jacobi_total_apply_scalars_per_elem": (
            _JACOBI_PERSISTENT_SCALARS_PER_ELEM + _TRANSIENT_SCALARS_PER_ELEM
        ),
        "full_stencil_scalars_per_elem": _FULL_STENCIL_SCALARS_PER_ELEM,
        "base_persistent_bytes": base_persistent * b,
        "jacobi_persistent_bytes": jacobi_persistent * b,
        "persistent_bytes": base_persistent * b,
        "transient_bytes": transient * b,
        "total_apply_bytes": (base_persistent + transient) * b,
        "jacobi_total_apply_bytes": (jacobi_persistent + transient) * b,
        "full_stencil_bytes": full * b,
    }


def c3d8r_hourglass_apply_reference_numpy(
    C_field: np.ndarray,
    u_flat: np.ndarray,
    *,
    voxel_size=(1.0, 1.0, 1.0),
    hourglass_coefficient: float = 0.1,
) -> np.ndarray:
    """NumPy reference for the two-stage semantics.

    Stage 1 computes an explicit ``elem_force[Nelem, 8, 3]`` buffer.  Stage 2
    gathers the eight adjacent element-force entries into each periodic node.
    The element stiffness formula is shared with the inspectable NumPy element
    prototype so this routine validates the two-stage data flow rather than a
    second copy of the C3D8R constitutive algebra.
    """

    from femlib.c3d8r import make_c3d8r_element_stiffness
    from femlib.prototypes.element_stencils_numpy import (
        _iter_cells,
        _local_node_ids,
        _node_ids_to_dofs,
    )

    C = np.asarray(C_field, dtype=np.float64)
    if C.ndim != 5 or C.shape[-2:] != (6, 6):
        raise ValueError(f"C_field must have shape (Nz, Ny, Nx, 6, 6), got {C.shape}")
    grid_shape = tuple(map(int, C.shape[:3]))
    n_elem = int(np.prod(grid_shape))
    n_dofs = 3 * n_elem
    u = np.asarray(u_flat, dtype=np.float64)
    if u.shape != (n_dofs,):
        raise ValueError(f"u_flat must have shape {(n_dofs,)}, got {u.shape}")

    elem_force = np.empty((n_elem, 8, 3), dtype=np.float64)
    elem_ids = np.empty((n_elem, 8), dtype=np.int64)
    for e, (z, y, x) in enumerate(_iter_cells(grid_shape)):
        ids = _local_node_ids("C3D8R", z, y, x, grid_shape)
        dofs = _node_ids_to_dofs(ids)
        Ke = make_c3d8r_element_stiffness(
            C[z, y, x],
            voxel_size,
            hourglass_coefficient=hourglass_coefficient,
        )
        elem_ids[e] = ids
        elem_force[e] = (Ke @ u[dofs]).reshape(8, 3)

    out = np.zeros(n_dofs, dtype=np.float64)
    for e in range(n_elem):
        for a in range(8):
            node = int(elem_ids[e, a])
            out[3 * node : 3 * node + 3] += elem_force[e, a]
    return out


def _needs_int64_idx(B: int, n_elem: int) -> bool:
    worst_u = int(B) * 3 * int(n_elem)
    worst_c = int(B) * int(n_elem) * 36
    worst_ef = int(B) * int(n_elem) * 24
    return max(worst_u, worst_c, worst_ef) > _INT32_MAX


if triton is not None:

    @triton.jit
    def _corner_data(a: tl.constexpr):
        ox = 0
        oy = 0
        oz = 0
        sx = -1.0
        sy = -1.0
        sz = -1.0
        if a == 1:
            ox = 1
            sx = 1.0
        elif a == 2:
            ox = 1
            oy = 1
            sx = 1.0
            sy = 1.0
        elif a == 3:
            oy = 1
            sy = 1.0
        elif a == 4:
            oz = 1
            sz = 1.0
        elif a == 5:
            ox = 1
            oz = 1
            sx = 1.0
            sz = 1.0
        elif a == 6:
            ox = 1
            oy = 1
            oz = 1
            sx = 1.0
            sy = 1.0
            sz = 1.0
        elif a == 7:
            oy = 1
            oz = 1
            sy = 1.0
            sz = 1.0
        return ox, oy, oz, sx, sy, sz


    @triton.jit
    def _c3d8r_element_force_kernel(
        u_ptr, C_ptr, hg_scale_ptr, geometry_ptr, elem_force_ptr,
        Nz, Ny, Nx,
        stride_ub, stride_uc, stride_uz, stride_uy, stride_ux,
        BLOCK: tl.constexpr,
        IDX_INT64: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bidx = tl.program_id(1)
        n_elem = Nz * Ny * Nx
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem

        i = offs % Nx
        tmp = offs // Nx
        j = tmp % Ny
        k = tmp // Ny

        if IDX_INT64:
            su_b = tl.cast(stride_ub, tl.int64)
            su_c = tl.cast(stride_uc, tl.int64)
            su_z = tl.cast(stride_uz, tl.int64)
            su_y = tl.cast(stride_uy, tl.int64)
            su_x = tl.cast(stride_ux, tl.int64)
            elem_i = tl.cast(offs, tl.int64)
            i_i = tl.cast(i, tl.int64)
            j_i = tl.cast(j, tl.int64)
            k_i = tl.cast(k, tl.int64)
        else:
            su_b = stride_ub
            su_c = stride_uc
            su_z = stride_uz
            su_y = stride_uy
            su_x = stride_ux
            elem_i = offs
            i_i = i
            j_i = j
            k_i = k

        acc_ty = u_ptr.dtype.element_ty
        inv4dx = tl.load(geometry_ptr + 0)
        inv4dy = tl.load(geometry_ptr + 1)
        inv4dz = tl.load(geometry_ptr + 2)
        volume = tl.load(geometry_ptr + 3)
        hnorm = tl.load(geometry_ptr + 4)
        e0 = tl.zeros([BLOCK], dtype=acc_ty)
        e1 = tl.zeros([BLOCK], dtype=acc_ty)
        e2 = tl.zeros([BLOCK], dtype=acc_ty)
        e3 = tl.zeros([BLOCK], dtype=acc_ty)
        e4 = tl.zeros([BLOCK], dtype=acc_ty)
        e5 = tl.zeros([BLOCK], dtype=acc_ty)
        qx0 = tl.zeros([BLOCK], dtype=acc_ty)
        qx1 = tl.zeros([BLOCK], dtype=acc_ty)
        qx2 = tl.zeros([BLOCK], dtype=acc_ty)
        qx3 = tl.zeros([BLOCK], dtype=acc_ty)
        qy0 = tl.zeros([BLOCK], dtype=acc_ty)
        qy1 = tl.zeros([BLOCK], dtype=acc_ty)
        qy2 = tl.zeros([BLOCK], dtype=acc_ty)
        qy3 = tl.zeros([BLOCK], dtype=acc_ty)
        qz0 = tl.zeros([BLOCK], dtype=acc_ty)
        qz1 = tl.zeros([BLOCK], dtype=acc_ty)
        qz2 = tl.zeros([BLOCK], dtype=acc_ty)
        qz3 = tl.zeros([BLOCK], dtype=acc_ty)

        u_batch_off = bidx * su_b
        for a in tl.static_range(8):
            ox, oy, oz, sx, sy, sz = _corner_data(a)
            ni = (i + ox) % Nx
            nj = (j + oy) % Ny
            nk = (k + oz) % Nz
            if IDX_INT64:
                ni_i = tl.cast(ni, tl.int64)
                nj_i = tl.cast(nj, tl.int64)
                nk_i = tl.cast(nk, tl.int64)
            else:
                ni_i = ni
                nj_i = nj
                nk_i = nk

            u_base = u_batch_off + nk_i * su_z + nj_i * su_y + ni_i * su_x
            ux = tl.load(u_ptr + u_base + 0 * su_c, mask=mask, other=0.0)
            uy = tl.load(u_ptr + u_base + 1 * su_c, mask=mask, other=0.0)
            uz = tl.load(u_ptr + u_base + 2 * su_c, mask=mask, other=0.0)

            dx = sx * inv4dx
            dy = sy * inv4dy
            dz = sz * inv4dz
            e0 += dx * ux
            e1 += dy * uy
            e2 += dz * uz
            e3 += dz * uy + dy * uz
            e4 += dz * ux + dx * uz
            e5 += dy * ux + dx * uy

            h0 = sx * sy * hnorm
            h1 = sy * sz * hnorm
            h2 = sz * sx * hnorm
            h3 = sx * sy * sz * hnorm
            qx0 += h0 * ux
            qx1 += h1 * ux
            qx2 += h2 * ux
            qx3 += h3 * ux
            qy0 += h0 * uy
            qy1 += h1 * uy
            qy2 += h2 * uy
            qy3 += h3 * uy
            qz0 += h0 * uz
            qz1 += h1 * uz
            qz2 += h2 * uz
            qz3 += h3 * uz

        c_base = (tl.cast(bidx, tl.int64) * tl.cast(n_elem, tl.int64) + tl.cast(offs, tl.int64)) * 36
        s0 = (tl.load(C_ptr + c_base + 0, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 1, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 2, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 3, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 4, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 5, mask=mask, other=0.0) * e5)
        s1 = (tl.load(C_ptr + c_base + 6, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 7, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 8, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 9, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 10, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 11, mask=mask, other=0.0) * e5)
        s2 = (tl.load(C_ptr + c_base + 12, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 13, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 14, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 15, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 16, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 17, mask=mask, other=0.0) * e5)
        s3 = (tl.load(C_ptr + c_base + 18, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 19, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 20, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 21, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 22, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 23, mask=mask, other=0.0) * e5)
        s4 = (tl.load(C_ptr + c_base + 24, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 25, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 26, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 27, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 28, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 29, mask=mask, other=0.0) * e5)
        s5 = (tl.load(C_ptr + c_base + 30, mask=mask, other=0.0) * e0
              + tl.load(C_ptr + c_base + 31, mask=mask, other=0.0) * e1
              + tl.load(C_ptr + c_base + 32, mask=mask, other=0.0) * e2
              + tl.load(C_ptr + c_base + 33, mask=mask, other=0.0) * e3
              + tl.load(C_ptr + c_base + 34, mask=mask, other=0.0) * e4
              + tl.load(C_ptr + c_base + 35, mask=mask, other=0.0) * e5)

        scale = tl.load(
            hg_scale_ptr + tl.cast(bidx, tl.int64) * tl.cast(n_elem, tl.int64) + tl.cast(offs, tl.int64),
            mask=mask,
            other=0.0,
        )
        ef_base = (tl.cast(bidx, tl.int64) * tl.cast(n_elem, tl.int64) + tl.cast(offs, tl.int64)) * 24

        for a in tl.static_range(8):
            _ox, _oy, _oz, sx, sy, sz = _corner_data(a)
            dx = sx * inv4dx
            dy = sy * inv4dy
            dz = sz * inv4dz
            h0 = sx * sy * hnorm
            h1 = sy * sz * hnorm
            h2 = sz * sx * hnorm
            h3 = sx * sy * sz * hnorm
            fx = volume * (dx * s0 + dz * s4 + dy * s5) + scale * (h0 * qx0 + h1 * qx1 + h2 * qx2 + h3 * qx3)
            fy = volume * (dy * s1 + dz * s3 + dx * s5) + scale * (h0 * qy0 + h1 * qy1 + h2 * qy2 + h3 * qy3)
            fz = volume * (dz * s2 + dy * s3 + dx * s4) + scale * (h0 * qz0 + h1 * qz1 + h2 * qz2 + h3 * qz3)
            out = ef_base + a * 3
            tl.store(elem_force_ptr + out + 0, fx, mask=mask)
            tl.store(elem_force_ptr + out + 1, fy, mask=mask)
            tl.store(elem_force_ptr + out + 2, fz, mask=mask)


    @triton.jit
    def _c3d8r_node_gather_kernel(
        elem_force_ptr, f_ptr,
        Nz, Ny, Nx,
        stride_fb, stride_fc, stride_fz, stride_fy, stride_fx,
        BLOCK: tl.constexpr,
        IDX_INT64: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bidx = tl.program_id(1)
        n_elem = Nz * Ny * Nx
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem

        i = offs % Nx
        tmp = offs // Nx
        j = tmp % Ny
        k = tmp // Ny

        if IDX_INT64:
            sf_b = tl.cast(stride_fb, tl.int64)
            sf_c = tl.cast(stride_fc, tl.int64)
            sf_z = tl.cast(stride_fz, tl.int64)
            sf_y = tl.cast(stride_fy, tl.int64)
            sf_x = tl.cast(stride_fx, tl.int64)
            i_i = tl.cast(i, tl.int64)
            j_i = tl.cast(j, tl.int64)
            k_i = tl.cast(k, tl.int64)
        else:
            sf_b = stride_fb
            sf_c = stride_fc
            sf_z = stride_fz
            sf_y = stride_fy
            sf_x = stride_fx
            i_i = i
            j_i = j
            k_i = k

        acc_ty = elem_force_ptr.dtype.element_ty
        f0 = tl.zeros([BLOCK], dtype=acc_ty)
        f1 = tl.zeros([BLOCK], dtype=acc_ty)
        f2 = tl.zeros([BLOCK], dtype=acc_ty)

        for a in tl.static_range(8):
            ox, oy, oz, _sx, _sy, _sz = _corner_data(a)
            ex = (i - ox + Nx) % Nx
            ey = (j - oy + Ny) % Ny
            ez = (k - oz + Nz) % Nz
            if IDX_INT64:
                ex_i = tl.cast(ex, tl.int64)
                ey_i = tl.cast(ey, tl.int64)
                ez_i = tl.cast(ez, tl.int64)
            else:
                ex_i = ex
                ey_i = ey
                ez_i = ez
            elem = ez_i * (Ny * Nx) + ey_i * Nx + ex_i
            base = (tl.cast(bidx, tl.int64) * tl.cast(n_elem, tl.int64) + tl.cast(elem, tl.int64)) * 24 + a * 3
            f0 += tl.load(elem_force_ptr + base + 0, mask=mask, other=0.0)
            f1 += tl.load(elem_force_ptr + base + 1, mask=mask, other=0.0)
            f2 += tl.load(elem_force_ptr + base + 2, mask=mask, other=0.0)

        out_base = bidx * sf_b + k_i * sf_z + j_i * sf_y + i_i * sf_x
        tl.store(f_ptr + out_base + 0 * sf_c, f0, mask=mask)
        tl.store(f_ptr + out_base + 1 * sf_c, f1, mask=mask)
        tl.store(f_ptr + out_base + 2 * sf_c, f2, mask=mask)


class BatchedC3D8RHourglassTwoStageOperator:
    """Two-stage C3D8R+hourglass elasticity operator.

    ``apply`` retains the production Triton implementation.  ``apply_cuda``
    provides a lazily compiled native CUDA A/B path.
    """

    def __init__(
        self,
        mesh: MeshInfo,
        C_voigt_fields: Any,
        *,
        device=None,
        dtype=None,
        hourglass_coefficient: float = 0.1,
        verbose: bool = False,
    ):
        if torch is None or triton is None:
            raise RuntimeError("BatchedC3D8RHourglassTwoStageOperator requires torch and triton")
        if device is None:
            device = torch.device("cuda")
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("BatchedC3D8RHourglassTwoStageOperator requires a CUDA device")
        if dtype is None:
            dtype = torch.float32

        self.mesh = mesh
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.hourglass_coefficient = float(hourglass_coefficient)
        self._apply_backend = "triton"
        self._cuda_apply_variant = "thread"
        self._cuda_apply_block_size = 128

        if isinstance(C_voigt_fields, torch.Tensor):
            C_gpu = C_voigt_fields.to(device=device, dtype=dtype)
        else:
            C_gpu = torch.from_numpy(np.ascontiguousarray(np.asarray(C_voigt_fields))).to(
                device=device,
                dtype=dtype,
            )
        expected = (6, 6, mesh.Nz, mesh.Ny, mesh.Nx)
        if C_gpu.ndim != 6 or tuple(C_gpu.shape[1:]) != expected:
            raise ValueError(f"C_voigt_fields must be (B,6,6,Nz,Ny,Nx), got {tuple(C_gpu.shape)}")
        self.B = int(C_gpu.shape[0])
        self.C = C_gpu.permute(0, 3, 4, 5, 1, 2).reshape(self.B, mesh.n_elem, 6, 6).contiguous()
        self._C_voigt_gpu = self.C.view(
            self.B, mesh.Nz, mesh.Ny, mesh.Nx, 6, 6,
        ).permute(0, 4, 5, 1, 2, 3)
        self.geometry_params = torch.tensor(
            [
                1.0 / (4.0 * mesh.dx),
                1.0 / (4.0 * mesh.dy),
                1.0 / (4.0 * mesh.dz),
                mesh.V_elem,
                1.0 / np.sqrt(8.0),
            ],
            device=device,
            dtype=dtype,
        ).contiguous()
        B_np, _ = make_B_matrices(mesh.dx, mesh.dy, mesh.dz, element_type="C3D8R")
        self.B_center = torch.tensor(
            B_np[0].reshape(6, 8, 3), device=device, dtype=dtype,
        ).contiguous()
        self.hg_scale = self._build_hourglass_scale()
        self._diag_cache = self._build_diagonal()
        self._M_inv_cache = None
        self._fft_enabled = False
        self._fft_symbol_inv = None
        self._fft_args = None
        self._fft_symbol_shared = False
        self._fft_spectrum_mode = None
        self._fft_inverse_norm_folded = False
        self._fft_batch_chunk_size = None

    def _build_hourglass_scale(self):
        if self.hourglass_coefficient <= 0.0:
            return torch.zeros((self.B, self.mesh.n_elem), device=self.device, dtype=self.dtype)
        normal = self.C[..., 0, 0] + self.C[..., 1, 1] + self.C[..., 2, 2]
        first_invariant = normal + 2.0 * (
            self.C[..., 0, 1] + self.C[..., 0, 2] + self.C[..., 1, 2]
        )
        second_invariant = normal + 2.0 * (
            self.C[..., 3, 3] + self.C[..., 4, 4] + self.C[..., 5, 5]
        )
        material_scale = (first_invariant + 7.0 * second_invariant) / 60.0
        hbar = (self.mesh.dx + self.mesh.dy + self.mesh.dz) / 3.0
        scale = self.hourglass_coefficient * material_scale.clamp_min(1e-12)
        scale = scale * (self.mesh.V_elem / (hbar * hbar))
        return scale.to(dtype=self.dtype).contiguous()

    def _build_diagonal(self):
        if min(self.mesh.Nx, self.mesh.Ny, self.mesh.Nz) >= 2:
            local = torch.einsum(
                "iap,beij,jap->beap", self.B_center, self.C, self.B_center,
            ) * self.mesh.V_elem
            local = local + 0.5 * self.hg_scale[..., None, None]
            return self._finalize_diagonal(local, local.abs())

        local = torch.zeros(
            (self.B, self.mesh.n_elem, 8, 3),
            device=self.device,
            dtype=self.dtype,
        )
        local_abs_sum = torch.zeros_like(local)
        H = torch.as_tensor(
            hourglass_basis().reshape(8, 3, 12),
            device=self.device,
            dtype=self.dtype,
        )
        periods = (self.mesh.Nx, self.mesh.Ny, self.mesh.Nz)
        for a, offset_a in enumerate(LOCAL_OFFSET):
            for b, offset_b in enumerate(LOCAL_OFFSET):
                if any(
                    (int(offset_a[axis]) - int(offset_b[axis])) % periods[axis]
                    for axis in range(3)
                ):
                    continue
                physical = torch.einsum(
                    "ip,beij,jp->bep",
                    self.B_center[:, a, :],
                    self.C,
                    self.B_center[:, b, :],
                ) * self.mesh.V_elem
                hourglass = self.hg_scale[..., None] * (H[a] * H[b]).sum(dim=-1)
                local[:, :, a, :].add_(physical + hourglass)
                local_abs_sum[:, :, a, :].add_(physical.abs() + hourglass.abs())
        return self._finalize_diagonal(local, local_abs_sum)

    def _finalize_diagonal(self, local, local_abs_sum):
        diagonal = self._scatter_element_vectors(local)
        reference_scale = self._scatter_element_vectors(local_abs_sum)
        threshold = (
            torch.finfo(self.dtype).eps
            * _DIAGONAL_ROUNDOFF_TERMS
            * reference_scale
        )
        return torch.where(diagonal.abs() <= threshold, torch.zeros_like(diagonal), diagonal)

    def _gather_element_displacements(self, u_flat):
        u = u_flat.view(self.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, 3)
        corners = []
        for ox, oy, oz in LOCAL_OFFSET:
            corners.append(
                torch.roll(
                    u,
                    shifts=(-int(oz), -int(oy), -int(ox)),
                    dims=(1, 2, 3),
                )
            )
        return torch.stack(corners, dim=-2).reshape(self.B, self.mesh.n_elem, 8, 3)

    def _scatter_element_vectors(self, values):
        field = values.view(self.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, 8, 3)
        out = torch.zeros(
            (self.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, 3),
            device=self.device,
            dtype=self.dtype,
        )
        for a, (ox, oy, oz) in enumerate(LOCAL_OFFSET):
            out.add_(
                torch.roll(
                    field[..., a, :],
                    shifts=(int(oz), int(oy), int(ox)),
                    dims=(1, 2, 3),
                )
            )
        return out.reshape(self.B, -1)

    def diag(self):
        return self._diag_cache

    def apply_precond(self, r_flat):
        from .preconditioners import apply_preconditioner

        return apply_preconditioner(self, r_flat)

    def enable_fft_precond(self, ref_strategy="zeman", C0_voigt=None, verbose=False,
                           spectrum_mode="complex_full", validation_mode="legacy_eig",
                           allow_symbol_sharing=True):
        from .preconditioners import enable_fft_preconditioner

        enable_fft_preconditioner(
            self,
            ref_strategy=ref_strategy,
            C0_voigt=C0_voigt,
            verbose=verbose,
            spectrum_mode=spectrum_mode,
            validation_mode=validation_mode,
            allow_symbol_sharing=allow_symbol_sharing,
        )

    def enable_fe_green_precond(self, ref_strategy="zeman", C0_voigt=None,
                                verbose=False, spectrum_mode="complex_full",
                                validation_mode="legacy_eig", allow_symbol_sharing=True):
        self.enable_fft_precond(
            ref_strategy=ref_strategy, C0_voigt=C0_voigt, verbose=verbose,
            spectrum_mode=spectrum_mode,
            validation_mode=validation_mode,
            allow_symbol_sharing=allow_symbol_sharing,
        )

    def disable_fft_precond(self):
        self._fft_enabled = False
        self._fft_symbol_inv = None
        self._fft_symbol_shared = False
        self._fft_spectrum_mode = None
        self._fft_inverse_norm_folded = False
        self._fft_args = None
        self._M_inv_cache = None

    def _apply_fft_base_precond(self, r_flat):
        from .stencil_operator import BatchedStencilOperator

        return BatchedStencilOperator._apply_fft_base_precond(self, r_flat)

    def _apply_fft_base_precond_batch(self, r_flat, symbol_inv):
        from .stencil_operator import BatchedStencilOperator

        return BatchedStencilOperator._apply_fft_base_precond_batch(
            self,
            r_flat,
            symbol_inv,
        )

    def set_fft_batch_chunk_size(self, batch_chunk_size=None):
        from .stencil_operator import BatchedStencilOperator

        BatchedStencilOperator.set_fft_batch_chunk_size(
            self,
            batch_chunk_size,
        )

    def _fft_spectral_multiply(self, symbol_inverse, rhs_spectrum):
        """Apply the cached FFT symbol with the selected execution backend."""

        if self._apply_backend == "cuda":
            from .c3d8r_hourglass_cuda import (
                c3d8r_fft_spectral_multiply_cuda,
            )

            return c3d8r_fft_spectral_multiply_cuda(
                symbol_inverse,
                rhs_spectrum,
            )
        rhs_batch = int(rhs_spectrum.shape[0])
        if symbol_inverse.shape[0] == 1 and rhs_batch != 1:
            return torch.einsum(
                "pqzyx,bqzyx->bpzyx",
                symbol_inverse[0],
                rhs_spectrum,
            )
        if symbol_inverse.shape[0] == rhs_batch:
            return torch.einsum(
                "bpqzyx,bqzyx->bpzyx",
                symbol_inverse,
                rhs_spectrum,
            )
        raise RuntimeError(
            f"FFT symbol batch dimension {symbol_inverse.shape[0]} "
            f"is incompatible with RHS batch {rhs_batch}"
        )

    def _apply_fft_precond(self, r_flat):
        from .stencil_operator import BatchedStencilOperator

        return BatchedStencilOperator._apply_fft_precond(self, r_flat)

    def _macro_strain_tensor(self, E_voigt):
        try:
            shape = tuple(E_voigt.shape)
        except AttributeError:
            shape = np.shape(E_voigt)
        if shape != (6,):
            raise ValueError("E_voigt must have shape (6,)")
        return torch.as_tensor(
            E_voigt, device=self.device, dtype=self.dtype,
        ).contiguous()

    def compute_rhs_torch(self, E_voigt):
        """Assemble the load vector with the original PyTorch implementation."""

        E = self._macro_strain_tensor(E_voigt)
        sigma = torch.einsum("beij,j->bei", self.C, E)
        local = torch.einsum("iap,bei->beap", self.B_center, sigma) * self.mesh.V_elem
        return -self._scatter_element_vectors(local)

    def compute_rhs_cuda(
        self,
        E_voigt,
        *,
        block_size: int | None = None,
        variant: str | None = None,
        verbose_build: bool = False,
    ):
        """Assemble the load vector with fused native CUDA kernels."""

        from .c3d8r_hourglass_cuda import c3d8r_rhs_cuda

        E = self._macro_strain_tensor(E_voigt)
        if block_size is None:
            block_size = 128
        if variant is None:
            variant = self._cuda_apply_variant
        return c3d8r_rhs_cuda(
            self.C,
            self.geometry_params,
            E,
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            variant=variant,
            verbose_build=verbose_build,
        )

    def compute_rhs(self, E_voigt):
        if self._apply_backend == "cuda":
            return self.compute_rhs_cuda(E_voigt)
        return self.compute_rhs_torch(E_voigt)

    def recover_center_fields(self, u_flat, E_voigt):
        if u_flat.shape != (self.B, self.mesh.n_dofs):
            raise ValueError("u_flat has the wrong shape")
        E = self._macro_strain_tensor(E_voigt)
        u_elem = self._gather_element_displacements(u_flat)
        strain = torch.einsum("iap,beap->bei", self.B_center, u_elem) + E
        stress = torch.einsum("beij,bej->bei", self.C, strain)
        return strain, stress

    def compute_avg_stress_torch(self, u_flat, E_voigt):
        """Recover average stress with the original PyTorch implementation."""

        _strain, stress = self.recover_center_fields(u_flat, E_voigt)
        return stress.mean(dim=1).detach().cpu().numpy().astype(np.float64)

    def compute_avg_stress_cuda(
        self,
        u_flat,
        E_voigt,
        *,
        block_size: int = 128,
        verbose_build: bool = False,
    ):
        """Recover average center stress with a native CUDA reduction."""

        self._validate_apply_input(u_flat)
        from .c3d8r_hourglass_cuda import c3d8r_avg_stress_cuda

        E = self._macro_strain_tensor(E_voigt)
        average = c3d8r_avg_stress_cuda(
            u_flat,
            self.C,
            self.geometry_params,
            E,
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            verbose_build=verbose_build,
        )
        return average.detach().cpu().numpy().astype(np.float64)

    def compute_avg_stress(self, u_flat, E_voigt):
        if self._apply_backend == "cuda":
            return self.compute_avg_stress_cuda(u_flat, E_voigt)
        return self.compute_avg_stress_torch(u_flat, E_voigt)

    def project_zero_mean(self, u_flat):
        u = u_flat.view(self.B, self.mesh.n_nodes, 3)
        return (u - u.mean(dim=1, keepdim=True)).reshape(self.B, -1)

    def storage_estimate(self, bytes_per_scalar: int | None = None) -> dict[str, int]:
        if bytes_per_scalar is None:
            bytes_per_scalar = torch.tensor([], dtype=self.dtype).element_size()
        return estimate_c3d8r_two_stage_storage(self.mesh.n_elem, bytes_per_scalar)

    def _validate_apply_input(self, u_flat):
        if u_flat.shape != (self.B, self.mesh.n_dofs):
            raise ValueError(f"u_flat must have shape {(self.B, self.mesh.n_dofs)}, got {tuple(u_flat.shape)}")
        if not u_flat.is_cuda:
            raise RuntimeError("u_flat must be a CUDA tensor")
        if u_flat.device != self.C.device:
            raise RuntimeError(f"u_flat must be on {self.C.device}, got {u_flat.device}")
        if u_flat.dtype != self.dtype:
            raise RuntimeError(f"u_flat must use {self.dtype}, got {u_flat.dtype}")
        if not u_flat.is_contiguous():
            raise RuntimeError("u_flat must be contiguous")

    def set_apply_backend(
        self,
        backend: str,
        *,
        cuda_variant: str | None = None,
        cuda_block_size: int | None = None,
    ):
        """Select the C3D8R execution backend.

        ``cuda_variant="atomic"`` is the high-throughput float32 path.  Its
        parallel accumulation is numerically accurate but not bitwise
        deterministic across launches.  The CUDA backend also selects native
        RHS, stress-recovery, and FFT spectral-multiply kernels when those
        operations are active.
        """

        if backend not in {"triton", "cuda"}:
            raise ValueError(f"unknown C3D8R apply backend {backend!r}")
        if cuda_variant is not None:
            if cuda_variant not in {"thread", "atomic"}:
                raise ValueError(f"unknown CUDA kernel variant {cuda_variant!r}")
            self._cuda_apply_variant = cuda_variant
        if cuda_block_size is not None:
            if cuda_block_size not in {64, 128, 256, 512}:
                raise ValueError(
                    "cuda_block_size must be one of 64, 128, 256, or 512"
                )
            self._cuda_apply_block_size = int(cuda_block_size)
        self._apply_backend = backend
        return self

    @property
    def apply_backend(self) -> str:
        return self._apply_backend

    @property
    def cuda_apply_configuration(self) -> tuple[str, int]:
        return self._cuda_apply_variant, self._cuda_apply_block_size

    def apply_cuda(
        self,
        u_flat,
        *,
        block_size: int = 128,
        variant: str = "thread",
        verbose_build: bool = False,
    ):
        """Apply the native CUDA A/B implementation.

        The extension is compiled lazily on the first call.  Unlike the Triton
        path, this implementation consumes and produces the solver's node-major
        flat layout directly, avoiding two layout-conversion kernels.
        """

        self._validate_apply_input(u_flat)
        from .c3d8r_hourglass_cuda import c3d8r_two_stage_apply_cuda

        return c3d8r_two_stage_apply_cuda(
            u_flat,
            self.C,
            self.hg_scale,
            self.geometry_params,
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            variant=variant,
            verbose_build=verbose_build,
        )

    def apply_triton(self, u_flat):
        """Apply the production Triton element-force and node-gather kernels."""

        self._validate_apply_input(u_flat)
        mesh = self.mesh
        u_field = u_flat.view(self.B, mesh.n_nodes, 3).transpose(1, 2).contiguous().view(
            self.B,
            3,
            mesh.Nz,
            mesh.Ny,
            mesh.Nx,
        )
        elem_force = torch.empty((self.B, mesh.n_elem, 8, 3), device=self.device, dtype=self.dtype)
        f_field = torch.empty_like(u_field)
        total = mesh.n_elem
        block = 128
        grid = (triton.cdiv(total, block), self.B)
        idx64 = _needs_int64_idx(self.B, mesh.n_elem)
        _c3d8r_element_force_kernel[grid](
            u_field,
            self.C,
            self.hg_scale,
            self.geometry_params,
            elem_force,
            mesh.Nz,
            mesh.Ny,
            mesh.Nx,
            u_field.stride(0),
            u_field.stride(1),
            u_field.stride(2),
            u_field.stride(3),
            u_field.stride(4),
            BLOCK=block,
            IDX_INT64=idx64,
        )
        _c3d8r_node_gather_kernel[grid](
            elem_force,
            f_field,
            mesh.Nz,
            mesh.Ny,
            mesh.Nx,
            f_field.stride(0),
            f_field.stride(1),
            f_field.stride(2),
            f_field.stride(3),
            f_field.stride(4),
            BLOCK=block,
            IDX_INT64=idx64,
        )
        return f_field.view(self.B, 3, -1).transpose(1, 2).contiguous().view(self.B, -1)

    def apply(self, u_flat):
        if self._apply_backend == "cuda":
            return self.apply_cuda(
                u_flat,
                variant=self._cuda_apply_variant,
                block_size=self._cuda_apply_block_size,
            )
        return self.apply_triton(u_flat)
