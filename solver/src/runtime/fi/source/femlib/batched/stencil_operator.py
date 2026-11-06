# -*- coding: utf-8 -*-
"""Batched 27-tap node stencil operator (GPU + Triton).

Holds the stencil tensor for B samples; K@u runs through a single Triton kernel over the whole batch.

Data conventions
----------------
  C_voigt_fields : (B, 6, 6, Nz, Ny, Nx)   legacy field, or the canonical
                   contiguous (B, Nz*Ny*Nx, 6, 6) element-major torch field
  u_flat         : (B, n_dofs)             torch, layout [n0x,n0y,n0z,n1x,...]
  stencil S_flat : (B, 14, 9, Nz, Ny, Nx)  torch, contiguous for ordinary C3D8;
                 C3D8I retains the full 27-slot compatibility layout

tap index: tap = iz*9 + iy*3 + ix, where (iz,iy,ix) ∈ {0,1,2}³, corresponding to offset Δ=(iz-1,iy-1,ix-1)
pq index : p*3 + q, p=row (output component), q=column (input component)
"""

from __future__ import annotations

import operator
import os

import numpy as np
import torch
import triton
import triton.language as tl

from ..geometry import LOCAL_OFFSET, make_B_matrices
from ..element_types import (
    make_element_stiffness,
    make_incompatible_B_matrices,
    normalize_element_type,
)
from ..materials import MANDEL_W
from ..connectivity import build_dof_table
from .fft_symbol_validation import validate_nonzero_fft_symbol_vectorized
from .mesh_info import MeshInfo


_VOIGT_UPPER_ROWS = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5)
_VOIGT_UPPER_COLS = (0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5)
_VOIGT_PACKED_COMPONENTS = len(_VOIGT_UPPER_ROWS)
_SYMMETRIC_STENCIL_TAPS = 14


def _unpack_packed_voigt(C_packed):
    """Expand a bounded packed material chunk to a symmetric 6x6 view.

    This helper is used only for assembly chunks or small reference-medium
    samples; the persistent FI material owner remains packed.
    """
    if C_packed.shape[-1] != _VOIGT_PACKED_COMPONENTS:
        raise ValueError(
            f"packed material must have {_VOIGT_PACKED_COMPONENTS} entries, "
            f"got {C_packed.shape[-1]}"
        )
    full = torch.empty(
        C_packed.shape[:-1] + (6, 6),
        dtype=C_packed.dtype,
        device=C_packed.device,
    )
    for packed, (row, col) in enumerate(zip(_VOIGT_UPPER_ROWS, _VOIGT_UPPER_COLS)):
        value = C_packed[..., packed]
        full[..., row, col] = value
        full[..., col, row] = value
    return full


def _packed_voigt_matvec(C_packed, vector):
    """Multiply a symmetric packed Voigt field by a vector without expansion."""
    if C_packed.shape[-1] != _VOIGT_PACKED_COMPONENTS:
        raise ValueError(
            f"packed material must have {_VOIGT_PACKED_COMPONENTS} entries, "
            f"got {C_packed.shape[-1]}"
        )
    if vector.shape[-1] != 6:
        raise ValueError(f"Voigt vector must have six entries, got {vector.shape[-1]}")
    c = C_packed
    e = vector
    return torch.stack((
        c[..., 0] * e[..., 0] + c[..., 1] * e[..., 1]
        + c[..., 2] * e[..., 2] + c[..., 3] * e[..., 3]
        + c[..., 4] * e[..., 4] + c[..., 5] * e[..., 5],
        c[..., 1] * e[..., 0] + c[..., 6] * e[..., 1]
        + c[..., 7] * e[..., 2] + c[..., 8] * e[..., 3]
        + c[..., 9] * e[..., 4] + c[..., 10] * e[..., 5],
        c[..., 2] * e[..., 0] + c[..., 7] * e[..., 1]
        + c[..., 11] * e[..., 2] + c[..., 12] * e[..., 3]
        + c[..., 13] * e[..., 4] + c[..., 14] * e[..., 5],
        c[..., 3] * e[..., 0] + c[..., 8] * e[..., 1]
        + c[..., 12] * e[..., 2] + c[..., 15] * e[..., 3]
        + c[..., 16] * e[..., 4] + c[..., 17] * e[..., 5],
        c[..., 4] * e[..., 0] + c[..., 9] * e[..., 1]
        + c[..., 13] * e[..., 2] + c[..., 16] * e[..., 3]
        + c[..., 18] * e[..., 4] + c[..., 19] * e[..., 5],
        c[..., 5] * e[..., 0] + c[..., 10] * e[..., 1]
        + c[..., 14] * e[..., 2] + c[..., 17] * e[..., 3]
        + c[..., 19] * e[..., 4] + c[..., 20] * e[..., 5],
    ), dim=-1)


def _validate_nonzero_fft_symbol(symbol, real_dtype):
    """Reject singular symbol blocks away from the rigid translation mode."""
    Bsz, _p, _q, Nz, Ny, Nx = symbol.shape
    sym_flat = symbol.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
    zero_idxs = torch.arange(Bsz, device=symbol.device) * (Nz * Ny * Nx)
    chunk_size = 512
    for start in range(0, sym_flat.shape[0], chunk_size):
        end = min(start + chunk_size, sym_flat.shape[0])
        block = sym_flat[start:end]
        hermitian = 0.5 * (block + block.mH)
        eigenvalues = torch.linalg.eigvalsh(hermitian)
        minimum = eigenvalues[:, 0]
        threshold = (
            100.0
            * torch.finfo(real_dtype).eps
            * eigenvalues.abs().amax(dim=-1)
        )
        nonzero = torch.ones(end - start, dtype=torch.bool, device=symbol.device)
        rigid = zero_idxs[(zero_idxs >= start) & (zero_idxs < end)] - start
        nonzero[rigid] = False
        singular = nonzero & (minimum <= threshold)
        if singular.any():
            local_index = int(torch.nonzero(singular, as_tuple=False)[0, 0].item())
            flat_index = start + local_index
            batch, frequency = divmod(flat_index, Nz * Ny * Nx)
            kz_index, frequency = divmod(frequency, Ny * Nx)
            ky_index, kx_index = divmod(frequency, Nx)
            min_eigenvalue = float(minimum[local_index].item())
            raise RuntimeError(
                "FFT reference has a nonzero wavevector singular mode: "
                f"batch={batch}, k=({kz_index},{ky_index},{kx_index}), "
                f"min_eigenvalue={min_eigenvalue:.6e}"
            )


# ============================================================
# Reference-medium stencil build (numpy, used only by the FFT preconditioner)
# ------------------------------------------------------------
# Given a uniform isotropic C0 (6,6,1,1,1), return the nodal 27-tap stencil
# S[iz, iy, ix, p, q, 0, 0, 0]. Shares the T-tensor formula with the main assembly path:
#     T[a, p, b, q, I, J] = Σ_g B[g, I, 3a+p] * B[g, J, 3b+q] * detJ
#     Ke[a, p, b, q] = Σ_{I,J} T[a,p,b,q,I,J] * C0[I,J]
# Then accumulate into the tap indexed by Δ = offset_b - offset_a, with a periodic
# roll by offset_a.
# ============================================================
def build_node_stencil_from_C(
    C_voigt_field,
    B_mat,
    detJ,
    *,
    element_type="c3d8",
    voxel_size=None,
):
    """numpy path: given a (6,6,Nz,Ny,Nx) Voigt C field,
    return the (3,3,3, 3,3, Nz,Ny,Nx) 27-tap stencil."""
    element_type = normalize_element_type(element_type)
    assert C_voigt_field.ndim == 5 and C_voigt_field.shape[:2] == (6, 6)
    Nz, Ny, Nx = C_voigt_field.shape[2:]

    if element_type == "c3d8i":
        if voxel_size is None:
            raise ValueError("voxel_size is required when element_type='c3d8i'")
        S = np.zeros((3, 3, 3, 3, 3, Nz, Ny, Nx), dtype=np.float64)
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    K_elem = make_element_stiffness(
                        C_voigt_field[:, :, z, y, x],
                        voxel_size,
                        element_type="c3d8i",
                    )
                    for a in range(8):
                        oa_x, oa_y, oa_z = LOCAL_OFFSET[a]
                        tx = (x + int(oa_x)) % Nx
                        ty = (y + int(oa_y)) % Ny
                        tz = (z + int(oa_z)) % Nz
                        for b_node in range(8):
                            ob_x, ob_y, ob_z = LOCAL_OFFSET[b_node]
                            dx = int(ob_x - oa_x)
                            dy = int(ob_y - oa_y)
                            dz = int(ob_z - oa_z)
                            ix, iy, iz = dx + 1, dy + 1, dz + 1
                            S[iz, iy, ix, :, :, tz, ty, tx] += K_elem[
                                3 * a : 3 * a + 3,
                                3 * b_node : 3 * b_node + 3,
                            ]
        return S

    n_gp = int(B_mat.shape[0])
    B_apbq = B_mat.reshape(n_gp, 6, 8, 3)                    # (g, I, a, p)
    T = np.einsum('gIap, gJbq -> apbqIJ', B_apbq, B_apbq) * detJ
    S = np.zeros((3, 3, 3, 3, 3, Nz, Ny, Nx), dtype=np.float64)

    for a in range(8):
        oa_x, oa_y, oa_z = LOCAL_OFFSET[a]
        for b_node in range(8):
            ob_x, ob_y, ob_z = LOCAL_OFFSET[b_node]
            dx = int(ob_x - oa_x)
            dy = int(ob_y - oa_y)
            dz = int(ob_z - oa_z)
            ix, iy, iz = dx + 1, dy + 1, dz + 1

            Tapbq = T[a, :, b_node, :, :, :]                 # (3,3,6,6)
            Ke_ab = np.einsum('pqIJ, IJzyx -> pqzyx', Tapbq, C_voigt_field)
            Ke_ab_rolled = np.roll(Ke_ab, shift=(int(oa_z), int(oa_y), int(oa_x)),
                                   axis=(2, 3, 4))
            S[iz, iy, ix] += Ke_ab_rolled

    return S


# ============================================================
# Triton kernel (with batch dimension)
# ============================================================
@triton.jit
def _stencil_apply_batched_kernel(
    u_ptr, f_ptr, S_ptr,
    Nz, Ny, Nx,
    stride_ub, stride_uc, stride_uz, stride_uy, stride_ux,
    stride_sb, stride_stap, stride_spq, stride_sz, stride_sy, stride_sx,
    BLOCK: tl.constexpr,
    IDX_INT64: tl.constexpr,
):
    """27-tap node-centered stencil apply: f = S u.

    IDX_INT64 (constexpr) promotes all flat-offset pointer arithmetic to
    int64. Required whenever any intermediate offset product exceeds
    2^31 (signed int32 overflow); for smaller grids the int32 path is
    faster on Blackwell. Selection happens at the Python call site.
    """
    pid = tl.program_id(0)
    bidx = tl.program_id(1)
    total = Nz * Ny * Nx

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    i = offs % Nx
    tmp = offs // Nx
    j = tmp % Ny
    k = tmp // Ny

    # --- promote strides / coords used in flat offsets to int64 if needed ---
    if IDX_INT64:
        su_b  = tl.cast(stride_ub,  tl.int64)
        su_c  = tl.cast(stride_uc,  tl.int64)
        su_z  = tl.cast(stride_uz,  tl.int64)
        su_y  = tl.cast(stride_uy,  tl.int64)
        su_x  = tl.cast(stride_ux,  tl.int64)
        ss_b  = tl.cast(stride_sb,  tl.int64)
        ss_t  = tl.cast(stride_stap, tl.int64)
        ss_pq = tl.cast(stride_spq, tl.int64)
        ss_z  = tl.cast(stride_sz,  tl.int64)
        ss_y  = tl.cast(stride_sy,  tl.int64)
        ss_x  = tl.cast(stride_sx,  tl.int64)
        k_i = tl.cast(k, tl.int64)
        j_i = tl.cast(j, tl.int64)
        i_i = tl.cast(i, tl.int64)
    else:
        su_b  = stride_ub
        su_c  = stride_uc
        su_z  = stride_uz
        su_y  = stride_uy
        su_x  = stride_ux
        ss_b  = stride_sb
        ss_t  = stride_stap
        ss_pq = stride_spq
        ss_z  = stride_sz
        ss_y  = stride_sy
        ss_x  = stride_sx
        k_i = k
        j_i = j
        i_i = i

    acc_ty = u_ptr.dtype.element_ty
    f0 = tl.zeros([BLOCK], dtype=acc_ty)
    f1 = tl.zeros([BLOCK], dtype=acc_ty)
    f2 = tl.zeros([BLOCK], dtype=acc_ty)

    u_batch_off = bidx * su_b
    s_batch_off = bidx * ss_b

    for tap in tl.static_range(27):
        ix = tap % 3
        iy = (tap // 3) % 3
        iz = tap // 9
        dx = ix - 1
        dy = iy - 1
        dz = iz - 1

        ni = (i + dx + Nx) % Nx
        nj = (j + dy + Ny) % Ny
        nk = (k + dz + Nz) % Nz
        if IDX_INT64:
            ni_i = tl.cast(ni, tl.int64)
            nj_i = tl.cast(nj, tl.int64)
            nk_i = tl.cast(nk, tl.int64)
        else:
            ni_i = ni
            nj_i = nj
            nk_i = nk

        u_base = u_batch_off + nk_i * su_z + nj_i * su_y + ni_i * su_x
        u0 = tl.load(u_ptr + 0 * su_c + u_base, mask=mask, other=0.0)
        u1 = tl.load(u_ptr + 1 * su_c + u_base, mask=mask, other=0.0)
        u2 = tl.load(u_ptr + 2 * su_c + u_base, mask=mask, other=0.0)

        s_base = (s_batch_off + tap * ss_t
                  + k_i * ss_z + j_i * ss_y + i_i * ss_x)
        s00 = tl.load(S_ptr + s_base + 0 * ss_pq, mask=mask, other=0.0)
        s01 = tl.load(S_ptr + s_base + 1 * ss_pq, mask=mask, other=0.0)
        s02 = tl.load(S_ptr + s_base + 2 * ss_pq, mask=mask, other=0.0)
        s10 = tl.load(S_ptr + s_base + 3 * ss_pq, mask=mask, other=0.0)
        s11 = tl.load(S_ptr + s_base + 4 * ss_pq, mask=mask, other=0.0)
        s12 = tl.load(S_ptr + s_base + 5 * ss_pq, mask=mask, other=0.0)
        s20 = tl.load(S_ptr + s_base + 6 * ss_pq, mask=mask, other=0.0)
        s21 = tl.load(S_ptr + s_base + 7 * ss_pq, mask=mask, other=0.0)
        s22 = tl.load(S_ptr + s_base + 8 * ss_pq, mask=mask, other=0.0)

        f0 += s00 * u0 + s01 * u1 + s02 * u2
        f1 += s10 * u0 + s11 * u1 + s12 * u2
        f2 += s20 * u0 + s21 * u1 + s22 * u2

    f_base = u_batch_off + k_i * su_z + j_i * su_y + i_i * su_x
    tl.store(f_ptr + 0 * su_c + f_base, f0, mask=mask)
    tl.store(f_ptr + 1 * su_c + f_base, f1, mask=mask)
    tl.store(f_ptr + 2 * su_c + f_base, f2, mask=mask)


@triton.jit
def _stencil_apply_symmetric_batched_kernel(
    u_ptr, f_ptr, S_ptr,
    Nz, Ny, Nx,
    stride_ub, stride_uc, stride_uz, stride_uy, stride_ux,
    stride_sb, stride_stap, stride_spq, stride_sz, stride_sy, stride_sx,
    BLOCK: tl.constexpr,
    IDX_INT64: tl.constexpr,
):
    """Apply a symmetric 14-slot stencil.

    Slots 0..12 hold one representative of each +/- offset pair and slot 13
    holds the centre block.  For an offset d, symmetry gives
    ``S_-d(n) = S_d(n-d).T``; the second term is therefore read from the
    representative at the negative neighbour and transposed in registers.
    """
    pid = tl.program_id(0)
    bidx = tl.program_id(1)
    total = Nz * Ny * Nx
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

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
        ss_b = tl.cast(stride_sb, tl.int64)
        ss_t = tl.cast(stride_stap, tl.int64)
        ss_pq = tl.cast(stride_spq, tl.int64)
        ss_z = tl.cast(stride_sz, tl.int64)
        ss_y = tl.cast(stride_sy, tl.int64)
        ss_x = tl.cast(stride_sx, tl.int64)
        k_i = tl.cast(k, tl.int64)
        j_i = tl.cast(j, tl.int64)
        i_i = tl.cast(i, tl.int64)
    else:
        su_b = stride_ub
        su_c = stride_uc
        su_z = stride_uz
        su_y = stride_uy
        su_x = stride_ux
        ss_b = stride_sb
        ss_t = stride_stap
        ss_pq = stride_spq
        ss_z = stride_sz
        ss_y = stride_sy
        ss_x = stride_sx
        k_i = k
        j_i = j
        i_i = i

    acc_ty = u_ptr.dtype.element_ty
    f0 = tl.zeros([BLOCK], dtype=acc_ty)
    f1 = tl.zeros([BLOCK], dtype=acc_ty)
    f2 = tl.zeros([BLOCK], dtype=acc_ty)
    u_batch_off = bidx * su_b
    s_batch_off = bidx * ss_b

    for slot in tl.static_range(_SYMMETRIC_STENCIL_TAPS):
        if slot == 13:
            dx = 0
            dy = 0
            dz = 0
        else:
            ix = slot % 3
            iy = (slot // 3) % 3
            iz = slot // 9
            dx = ix - 1
            dy = iy - 1
            dz = iz - 1

        ni = (i + dx + Nx) % Nx
        nj = (j + dy + Ny) % Ny
        nk = (k + dz + Nz) % Nz
        if IDX_INT64:
            ni_i = tl.cast(ni, tl.int64)
            nj_i = tl.cast(nj, tl.int64)
            nk_i = tl.cast(nk, tl.int64)
        else:
            ni_i = ni
            nj_i = nj
            nk_i = nk
        u_pos_base = u_batch_off + nk_i * su_z + nj_i * su_y + ni_i * su_x
        up0 = tl.load(u_ptr + 0 * su_c + u_pos_base, mask=mask, other=0.0)
        up1 = tl.load(u_ptr + 1 * su_c + u_pos_base, mask=mask, other=0.0)
        up2 = tl.load(u_ptr + 2 * su_c + u_pos_base, mask=mask, other=0.0)

        s_base = (s_batch_off + slot * ss_t
                  + k_i * ss_z + j_i * ss_y + i_i * ss_x)
        p00 = tl.load(S_ptr + s_base + 0 * ss_pq, mask=mask, other=0.0)
        p01 = tl.load(S_ptr + s_base + 1 * ss_pq, mask=mask, other=0.0)
        p02 = tl.load(S_ptr + s_base + 2 * ss_pq, mask=mask, other=0.0)
        p10 = tl.load(S_ptr + s_base + 3 * ss_pq, mask=mask, other=0.0)
        p11 = tl.load(S_ptr + s_base + 4 * ss_pq, mask=mask, other=0.0)
        p12 = tl.load(S_ptr + s_base + 5 * ss_pq, mask=mask, other=0.0)
        p20 = tl.load(S_ptr + s_base + 6 * ss_pq, mask=mask, other=0.0)
        p21 = tl.load(S_ptr + s_base + 7 * ss_pq, mask=mask, other=0.0)
        p22 = tl.load(S_ptr + s_base + 8 * ss_pq, mask=mask, other=0.0)

        f0 += p00 * up0 + p01 * up1 + p02 * up2
        f1 += p10 * up0 + p11 * up1 + p12 * up2
        f2 += p20 * up0 + p21 * up1 + p22 * up2

        if slot != 13:
            mi = (i - dx + Nx) % Nx
            mj = (j - dy + Ny) % Ny
            mk = (k - dz + Nz) % Nz
            if IDX_INT64:
                mi_i = tl.cast(mi, tl.int64)
                mj_i = tl.cast(mj, tl.int64)
                mk_i = tl.cast(mk, tl.int64)
            else:
                mi_i = mi
                mj_i = mj
                mk_i = mk
            u_neg_base = u_batch_off + mk_i * su_z + mj_i * su_y + mi_i * su_x
            un0 = tl.load(u_ptr + 0 * su_c + u_neg_base, mask=mask, other=0.0)
            un1 = tl.load(u_ptr + 1 * su_c + u_neg_base, mask=mask, other=0.0)
            un2 = tl.load(u_ptr + 2 * su_c + u_neg_base, mask=mask, other=0.0)
            sn_base = (s_batch_off + slot * ss_t
                       + mk_i * ss_z + mj_i * ss_y + mi_i * ss_x)
            n00 = tl.load(S_ptr + sn_base + 0 * ss_pq, mask=mask, other=0.0)
            n01 = tl.load(S_ptr + sn_base + 1 * ss_pq, mask=mask, other=0.0)
            n02 = tl.load(S_ptr + sn_base + 2 * ss_pq, mask=mask, other=0.0)
            n10 = tl.load(S_ptr + sn_base + 3 * ss_pq, mask=mask, other=0.0)
            n11 = tl.load(S_ptr + sn_base + 4 * ss_pq, mask=mask, other=0.0)
            n12 = tl.load(S_ptr + sn_base + 5 * ss_pq, mask=mask, other=0.0)
            n20 = tl.load(S_ptr + sn_base + 6 * ss_pq, mask=mask, other=0.0)
            n21 = tl.load(S_ptr + sn_base + 7 * ss_pq, mask=mask, other=0.0)
            n22 = tl.load(S_ptr + sn_base + 8 * ss_pq, mask=mask, other=0.0)
            f0 += n00 * un0 + n10 * un1 + n20 * un2
            f1 += n01 * un0 + n11 * un1 + n21 * un2
            f2 += n02 * un0 + n12 * un1 + n22 * un2

    f_base = u_batch_off + k_i * su_z + j_i * su_y + i_i * su_x
    tl.store(f_ptr + 0 * su_c + f_base, f0, mask=mask)
    tl.store(f_ptr + 1 * su_c + f_base, f1, mask=mask)
    tl.store(f_ptr + 2 * su_c + f_base, f2, mask=mask)


# Safety margin: promote to int64 once worst-case offset exceeds half of int32.
# We check every intermediate that the kernel computes: stride_sb * B,
# tap_max (26) * stride_stap, and all the (B, 27, 9) S-flat offsets.
_INT32_MAX = (1 << 31) - 1


def _needs_int64_idx(B, Nz, Ny, Nx):
    """Return True if any offset computed inside the stencil kernel would
    overflow signed int32. The worst case is (26 * 9 * Nz*Ny*Nx)
    for the tap*stride_stap product, or B * 27 * 9 * Nz*Ny*Nx for the
    full S_flat flat index."""
    n_elem = int(Nz) * int(Ny) * int(Nx)
    worst_tap = 26 * 9 * n_elem                      # tap * stride_stap
    worst_s = int(B) * 27 * 9 * n_elem               # batch_off + ... at max
    worst_u = int(B) * 3 * n_elem                    # u-side max flat offset
    return max(worst_tap, worst_s, worst_u) > _INT32_MAX


def stencil_apply_triton_batched(u_field, S_flat):
    """Apply full 27-slot or symmetric 14-slot batched stencils."""
    assert u_field.is_contiguous() and S_flat.is_contiguous()
    Bsz, Nc, Nz, Ny, Nx = u_field.shape
    assert Nc == 3
    assert S_flat.shape[1] in (27, _SYMMETRIC_STENCIL_TAPS)
    f_field = torch.empty_like(u_field)
    total = Nz * Ny * Nx
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK), Bsz)
    idx_int64 = _needs_int64_idx(Bsz, Nz, Ny, Nx)
    kernel = (
        _stencil_apply_symmetric_batched_kernel
        if S_flat.shape[1] == _SYMMETRIC_STENCIL_TAPS
        else _stencil_apply_batched_kernel
    )
    kernel[grid](
        u_field, f_field, S_flat,
        Nz, Ny, Nx,
        u_field.stride(0), u_field.stride(1),
        u_field.stride(2), u_field.stride(3), u_field.stride(4),
        S_flat.stride(0), S_flat.stride(1), S_flat.stride(2),
        S_flat.stride(3), S_flat.stride(4), S_flat.stride(5),
        BLOCK=BLOCK,
        IDX_INT64=idx_int64,
    )
    return f_field


# ============================================================
# Batched operator
# ============================================================
class BatchedStencilOperator:
    """Batched stencil operator with compressed ordinary-C3D8 storage.

    Interface
    ---------
      apply(u_flat: (B, n_dofs)) -> (B, n_dofs)
      diag() -> (B, n_dofs)
      compute_rhs(E_voigt: (6,)) -> (B, n_dofs)
      compute_avg_stress(u_flat: (B, n_dofs), E_voigt: (6,)) -> np.ndarray (B, 6)
    """

    def __init__(self, mesh: MeshInfo, C_voigt_fields,
                 device=None, dtype=torch.float32, verbose=False,
                 element_type="c3d8"):
        if device is None:
            device = torch.device('cuda')
        self.mesh = mesh
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        element_key = str(element_type).strip().lower()
        self.element_type = (
            "c3d8r" if element_key == "c3d8r" else normalize_element_type(element_type)
        )

        # Accept numpy or torch inputs; torch (already on GPU) is zero-copy.
        if isinstance(C_voigt_fields, torch.Tensor):
            C_gpu = C_voigt_fields.to(device=device, dtype=dtype)
        else:
            C_gpu = torch.from_numpy(
                np.ascontiguousarray(np.asarray(C_voigt_fields))
            ).to(device=device, dtype=dtype)

        canonical_shape = (int(C_gpu.shape[0]), mesh.n_elem, 6, 6) \
            if C_gpu.ndim == 4 else None
        packed_shape = (int(C_gpu.shape[0]), mesh.n_elem, _VOIGT_PACKED_COMPONENTS) \
            if C_gpu.ndim == 3 else None
        legacy_shape = (int(C_gpu.shape[0]), 6, 6, mesh.Nz, mesh.Ny, mesh.Nx) \
            if C_gpu.ndim == 6 else None
        if canonical_shape is not None and tuple(C_gpu.shape) == canonical_shape:
            if not C_gpu.is_contiguous():
                raise ValueError(
                    "canonical material field must be contiguous; refusing an "
                    "implicit full constitutive copy"
                )
            material_layout = "element_major"
        elif packed_shape is not None and tuple(C_gpu.shape) == packed_shape:
            if not C_gpu.is_contiguous():
                raise ValueError(
                    "packed material field must be contiguous; refusing an "
                    "implicit constitutive copy"
                )
            material_layout = "packed_element_major"
        elif legacy_shape is not None and tuple(C_gpu.shape) == legacy_shape:
            material_layout = "legacy_voigt"
        else:
            raise ValueError(
                "C_voigt_fields must be (B,6,6,Nz,Ny,Nx), contiguous "
                f"(B,{mesh.n_elem},6,6), or packed (B,{mesh.n_elem},"
                f"{_VOIGT_PACKED_COMPONENTS}), got {tuple(C_gpu.shape)}"
            )
        self.B = int(C_gpu.shape[0])

        # The canonical FI paths already own their exact element-major tensor.
        # Keep it by reference. The legacy adapter is deliberately guarded:
        # its contiguous conversion is needed only for callers that still
        # supply the historical spatial Voigt layout.
        self._material_layout = material_layout
        if material_layout in {"element_major", "packed_element_major"}:
            self.C = C_gpu
        else:
            self.C = C_gpu.permute(0, 3, 4, 5, 1, 2).reshape(
                self.B, mesh.n_elem, 6, 6,
            ).contiguous()

        # B matrices and detJ. C3D8I uses the full-integration C3D8 corner
        # modes plus incompatible internal modes; C3D8R uses one-point B.
        geometry_element_type = "c3d8r" if self.element_type == "c3d8r" else "c3d8"
        B_np, detJ = make_B_matrices(
            mesh.dx, mesh.dy, mesh.dz, element_type=geometry_element_type,
        )
        self._B_np = B_np
        self.detJ = float(detJ)
        self.B_mat = torch.tensor(B_np, device=device, dtype=dtype)
        self._B_inc_np = None
        self.B_inc_mat = None
        if self.element_type == "c3d8i":
            B_inc_np = make_incompatible_B_matrices(mesh.dx, mesh.dy, mesh.dz)
            self._B_inc_np = B_inc_np
            self.B_inc_mat = torch.tensor(B_inc_np, device=device, dtype=dtype)

        # Connectivity elem_dof_ids: (n_elem, 24) -> flat (n_elem*24,)
        elem_node_ids_np, elem_dof_ids_np = build_dof_table(mesh.Nx, mesh.Ny, mesh.Nz)
        self.elem_node_ids = torch.tensor(elem_node_ids_np, device=device,
                                          dtype=torch.long)
        self.elem_dof_ids = torch.tensor(elem_dof_ids_np, device=device,
                                          dtype=torch.long)
        self.elem_dof_flat = self.elem_dof_ids.reshape(-1)

        # FFT reference medium consumes the packed owner directly and expands
        # only a small subsample.  Full canonical fields retain their old
        # stride-only spatial adapter.
        if material_layout == "packed_element_major":
            self._C_voigt_gpu = self.C
        else:
            self._C_voigt_gpu = self.C.view(
                self.B, mesh.Nz, mesh.Ny, mesh.Nx, 6, 6,
            ).permute(0, 4, 5, 1, 2, 3)

        # Build per-sample stencils (entirely on GPU)
        if self.element_type in {"c3d8", "c3d8r"}:
            self._build_stencils(self._C_voigt_gpu, B_np, detJ)
        else:
            self._build_stencils_c3d8i()

        self._M_inv_cache = None
        self._fft_enabled = False
        self._fft_symbol_inv = None       # (B, 3, 3, Nz, Ny, Nx) complex
        self._fft_spectrum_mode = None
        self._fft_inverse_norm_folded = False
        self._fft_batch_chunk_size = None
        self._fft_args = None             # Saved enable_fft_precond arguments, reused by the IR inner operator
        self._apply_backend = "triton"
        self._cuda_apply_variant = "thread"
        self._cuda_apply_block_size = 256
        if self.element_type == "c3d8":
            self._cuda_geometry = torch.tensor(
                [
                    1.0 / (4.0 * mesh.dx),
                    1.0 / (4.0 * mesh.dy),
                    1.0 / (4.0 * mesh.dz),
                    mesh.V_elem,
                ],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._cuda_geometry = None

    # --------------------------------------------------------
    def _build_stencils(self, C_fields_np, B_np, detJ):
        """Assemble a symmetric 14-slot stencil on GPU plus centre diagonal.

        All tensor operations run on self.device with self.dtype, avoiding a numpy CPU loop.
        """
        Bsz, Nz, Ny, Nx = self.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx

        # The canonical owner is physically contiguous as (B,z,y,x,I,J).
        # Use that order in the einsum so PyTorch can consume the spatial
        # adapter without materialising a stride-normalised constitutive copy.
        canonical_layout = self._material_layout == "element_major"
        packed_layout = self._material_layout == "packed_element_major"
        if canonical_layout:
            C_gpu = self.C.view(Bsz, Nz, Ny, Nx, 6, 6)
        elif packed_layout:
            C_gpu = self.C
        elif isinstance(C_fields_np, np.ndarray):
            C_gpu = torch.from_numpy(
                np.ascontiguousarray(C_fields_np)
            ).to(device=self.device, dtype=self.dtype)
        else:
            C_gpu = C_fields_np.to(device=self.device, dtype=self.dtype)
        if canonical_layout:
            assert tuple(C_gpu.shape) == (Bsz, Nz, Ny, Nx, 6, 6)
        elif packed_layout:
            assert tuple(C_gpu.shape) == (Bsz, Nz * Ny * Nx, _VOIGT_PACKED_COMPONENTS)
        else:
            assert tuple(C_gpu.shape) == (Bsz, 6, 6, Nz, Ny, Nx)

        # T tensor (8, 3, 8, 3, 6, 6), depends only on geometry, computed on GPU
        B_mat = torch.from_numpy(np.ascontiguousarray(B_np)).to(
            device=self.device, dtype=self.dtype,
        )
        n_gp = int(B_np.shape[0])
        B_apbq = B_mat.reshape(n_gp, 6, 8, 3)                    # (g, I, a, p)
        T = torch.einsum('gIap,gJbq->apbqIJ', B_apbq, B_apbq) * detJ

        offsets = np.asarray(LOCAL_OFFSET, dtype=np.int64)       # (8, 3)  (x,y,z)
        # The ordinary C3D8 global operator is symmetric.  Keep one tap from
        # each +/- offset pair plus the centre, and reconstruct the opposite
        # term in the apply kernel.  This avoids retaining an 8 GiB B=16 full
        # S_flat on the scored 64^3 workload.
        S_flat = torch.zeros((Bsz, _SYMMETRIC_STENCIL_TAPS, 9, Nz, Ny, Nx),
                             dtype=self.dtype, device=self.device)

        # The compressed S_flat is the sole ordinary-C3D8 operator
        # representation, and contraction temporaries are batch-chunked.
        # On the target backend a whole-B einsum can reserve a large workspace
        # in addition to C_rolled and Ke.  Bounding B for each contraction
        # bounds that lifetime without changing the FI stencil or work.
        try:
            assembly_chunk = int(
                os.environ.get("MFREE_FI_ASSEMBLY_BATCH_CHUNK", "1")
            )
        except ValueError:
            assembly_chunk = 1
        assembly_chunk = max(1, min(Bsz, assembly_chunk))

        # Outer loop over the 8 element-corners `a`, and inner chunks over the
        # sample dimension. T[a] is shared and each chunk writes directly to
        # its corresponding S_flat view before its temporary is released.
        for a in range(8):
            oa_x, oa_y, oa_z = [int(v) for v in offsets[a]]
            # T[a] has shape (3, 8, 3, 6, 6) = (p, b_node, q, I, J);
            # permute so b_node is leading.
            T_a = T[a].permute(1, 0, 2, 3, 4).contiguous()       # (8, 3, 3, 6, 6)

            # Map b_node -> tap for this a
            d_xyz = offsets - offsets[a]
            taps = ((d_xyz[:, 2] + 1) * 9
                    + (d_xyz[:, 1] + 1) * 3
                    + (d_xyz[:, 0] + 1))                          # (8,)
            taps_t = torch.from_numpy(taps).to(self.device)

            for b_start in range(0, Bsz, assembly_chunk):
                b_stop = min(b_start + assembly_chunk, Bsz)
                C_chunk = C_gpu[b_start:b_stop]
                if packed_layout:
                    # The packed owner is symmetric; contract the upper
                    # triangle directly rather than materialising a 6x6
                    # batch tile.  Off-diagonal T terms are paired here.
                    C_chunk_full = None
                    C_spatial = C_chunk.view(
                        b_stop - b_start, Nz, Ny, Nx,
                        _VOIGT_PACKED_COMPONENTS,
                    )
                elif canonical_layout:
                    C_chunk_full = None
                    C_spatial = C_chunk
                else:
                    C_chunk_full = None
                    C_spatial = C_chunk

                if oa_x or oa_y or oa_z:
                    if canonical_layout or packed_layout:
                        C_rolled = torch.roll(
                            C_spatial,
                            shifts=(oa_z, oa_y, oa_x),
                            dims=(1, 2, 3),
                        )
                    else:
                        C_rolled = torch.roll(
                            C_spatial,
                            shifts=(oa_z, oa_y, oa_x),
                            dims=(3, 4, 5),
                        )
                else:
                    C_rolled = C_spatial

                # Ke[b, n, p, q, z, y, x] = sum_{I,J}
                # T_a[n,p,q,I,J] * C_rolled[b,I,J,z,y,x].
                if packed_layout:
                    t_packed = torch.stack(
                        tuple(
                            T_a[..., row, col]
                            if row == col else
                            T_a[..., row, col] + T_a[..., col, row]
                            for row, col in zip(
                                _VOIGT_UPPER_ROWS, _VOIGT_UPPER_COLS,
                            )
                        ),
                        dim=-1,
                    )
                    Ke = torch.einsum(
                        'npqk,bzyxk->bnpqzyx', t_packed, C_rolled,
                    )
                    del t_packed
                elif canonical_layout:
                    Ke = torch.einsum(
                        'npqIJ,bzyxIJ->bnpqzyx', T_a, C_rolled,
                    )
                else:
                    Ke = torch.einsum(
                        'npqIJ,bIJzyx->bnpqzyx', T_a, C_rolled,
                    )
                Ke = Ke.reshape(b_stop - b_start, 8, 9, Nz, Ny, Nx)
                # Store only taps 0..12 and the centre tap 13.  For a
                # symmetric assembled operator, the omitted coefficient is
                # S_{-d}(n) = S_d(n-d)^T and is reconstructed at apply time.
                positive_b = taps <= 13
                if np.any(positive_b):
                    positive_indices = np.flatnonzero(positive_b)
                    S_flat[b_start:b_stop].index_add_(
                        1,
                        taps_t[positive_indices],
                        Ke[:, positive_indices],
                    )
                # Do not retain views of the chunk or contraction output into
                # the next corner; this is the lifetime under test.
                del Ke, C_rolled, C_spatial, C_chunk
                if C_chunk_full is not None:
                    del C_chunk_full

        self.S_flat = S_flat.contiguous()

        # Centre tap = iz=iy=ix=1 → tap=13, pq diagonal {0,4,8}
        diag_field = self.S_flat[:, 13][:, [0, 4, 8], :, :, :]    # (B,3,Nz,Ny,Nx)
        diag_field = torch.where(
            diag_field.abs() > 1e-30, diag_field,
            torch.ones_like(diag_field),
        )
        # (B,3,Nz,Ny,Nx) -> (B,Nz,Ny,Nx,3) -> (B, n_dofs)
        self._diag_cache = diag_field.permute(0, 2, 3, 4, 1).reshape(Bsz, -1).contiguous()

    def _c3d8i_chunk_size(self):
        raw = os.environ.get("MFREE_C3D8I_CHUNK_ELEMS", "1024")
        try:
            value = int(raw)
        except ValueError:
            value = 1024
        return max(1, value)

    def _c3d8i_element_matrices(self, C_chunk):
        """Return K_ui, K_ii, and condensed K_uu for a batch/element chunk."""
        if self.B_inc_mat is None:
            raise RuntimeError("C3D8I incompatible-mode matrices are not initialised")
        B_u = self.B_mat
        B_i = self.B_inc_mat
        K_uu = torch.einsum('gip,bcij,gjq->bcpq', B_u, C_chunk, B_u) * self.detJ
        K_ui = torch.einsum('gip,bcij,gjm->bcpm', B_u, C_chunk, B_i) * self.detJ
        K_ii = torch.einsum('gim,bcij,gjn->bcmn', B_i, C_chunk, B_i) * self.detJ
        condensed = K_uu - torch.matmul(K_ui, torch.linalg.solve(K_ii, K_ui.transpose(-1, -2)))
        condensed = 0.5 * (condensed + condensed.transpose(-1, -2))
        return K_ui, K_ii, condensed

    def _build_stencils_c3d8i(self):
        """Assemble C3D8I condensed element matrices into the same 27-tap stencil."""
        Bsz, Nz, Ny, Nx = self.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx
        S_linear = torch.zeros((Bsz, 27, 9, self.mesh.n_nodes),
                               dtype=self.dtype, device=self.device)
        offsets = np.asarray(LOCAL_OFFSET, dtype=np.int64)
        chunk_size = self._c3d8i_chunk_size()

        for start in range(0, self.mesh.n_elem, chunk_size):
            stop = min(start + chunk_size, self.mesh.n_elem)
            C_chunk = self.C[:, start:stop]                    # (B, chunk, 6, 6)
            _, _, K_eff = self._c3d8i_element_matrices(C_chunk)  # (B, chunk, 24, 24)

            for a in range(8):
                target = self.elem_node_ids[start:stop, a]
                for b_node in range(8):
                    d_xyz = offsets[b_node] - offsets[a]
                    tap = int((d_xyz[2] + 1) * 9 + (d_xyz[1] + 1) * 3 + (d_xyz[0] + 1))
                    block = K_eff[
                        :, :,
                        3 * a : 3 * a + 3,
                        3 * b_node : 3 * b_node + 3,
                    ]
                    values = block.contiguous().view(Bsz, stop - start, 9).permute(0, 2, 1).contiguous()
                    S_linear[:, tap].index_add_(2, target, values)

        self.S_flat = S_linear.view(Bsz, 27, 9, Nz, Ny, Nx).contiguous()

        diag_field = self.S_flat[:, 13][:, [0, 4, 8], :, :, :]
        diag_field = torch.where(
            diag_field.abs() > 1e-30, diag_field,
            torch.ones_like(diag_field),
        )
        self._diag_cache = diag_field.permute(0, 2, 3, 4, 1).reshape(Bsz, -1).contiguous()

    # --------------------------------------------------------
    def diag(self):
        return self._diag_cache

    # --------------------------------------------------------
    def _validate_apply_input(self, u_flat):
        expected = (self.B, self.mesh.n_dofs)
        if tuple(u_flat.shape) != expected:
            raise ValueError(f"u_flat must have shape {expected}, got {tuple(u_flat.shape)}")
        if not u_flat.is_cuda:
            raise RuntimeError("u_flat must be a CUDA tensor")
        expected_device = self.S_flat.device
        if u_flat.device != expected_device:
            raise RuntimeError(
                f"u_flat must be on {expected_device}, got {u_flat.device}"
            )
        if u_flat.dtype != self.dtype:
            raise RuntimeError(f"u_flat must use {self.dtype}, got {u_flat.dtype}")
        if not u_flat.is_contiguous():
            raise RuntimeError("u_flat must be contiguous")

    def set_apply_backend(
        self,
        backend,
        *,
        cuda_variant=None,
        cuda_block_size=None,
    ):
        """Select the standard C3D8 stencil execution backend.

        CUDA support is intentionally limited to ordinary full-integration
        C3D8.  C3D8I remains on the established Triton/PyTorch path.  Atomic
        RHS assembly is available only when explicitly requested.
        """
        if backend not in {"triton", "cuda"}:
            raise ValueError(f"unknown stencil apply backend {backend!r}")
        if backend == "cuda" and self.element_type != "c3d8":
            raise NotImplementedError(
                "native CUDA is currently implemented only for ordinary C3D8"
            )
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
    def apply_backend(self):
        return self._apply_backend

    @property
    def cuda_apply_configuration(self):
        return self._cuda_apply_variant, self._cuda_apply_block_size

    def apply_triton(self, u_flat):
        """Apply the established Triton stencil kernel."""
        Bsz = self.B
        mesh = self.mesh
        # (B, n_nodes*3) -> (B, n_nodes, 3) -> (B, 3, n_nodes) -> (B, 3, Nz, Ny, Nx)
        u_field = u_flat.view(Bsz, mesh.n_nodes, 3).transpose(1, 2).contiguous().view(
            Bsz, 3, mesh.Nz, mesh.Ny, mesh.Nx,
        )
        f_field = stencil_apply_triton_batched(u_field, self.S_flat)
        # Back to (B, n_dofs)
        return f_field.view(Bsz, 3, -1).transpose(1, 2).contiguous().view(Bsz, -1)

    def apply_cuda(self, u_flat, *, block_size=None, verbose_build=False):
        """Apply the native CUDA stencil directly in node-major flat layout."""
        self._validate_apply_input(u_flat)
        from .c3d8_cuda import c3d8_stencil_apply_cuda

        if block_size is None:
            block_size = self._cuda_apply_block_size
        return c3d8_stencil_apply_cuda(
            u_flat,
            self.S_flat,
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            verbose_build=verbose_build,
        )

    def apply(self, u_flat):
        """u_flat: (B, n_dofs) -> (B, n_dofs)."""
        if self._apply_backend == "cuda":
            return self.apply_cuda(u_flat)
        return self.apply_triton(u_flat)

    # --------------------------------------------------------
    def _macro_strain_tensor(self, E_voigt):
        try:
            shape = tuple(E_voigt.shape)
        except AttributeError:
            shape = np.shape(E_voigt)
        if shape != (6,):
            raise ValueError("E_voigt must have shape (6,)")
        return torch.as_tensor(
            E_voigt,
            device=self.device,
            dtype=self.dtype,
        ).contiguous()

    def _material_full_for_math(self):
        """Return a full C view for the non-native compatibility path."""
        if self._material_layout == "packed_element_major":
            return _unpack_packed_voigt(self.C)
        return self.C

    def compute_rhs_torch(self, E_voigt):
        """E_voigt: (6,) macroscopic strain. Returns (B, n_dofs) RHS = -∫ Bᵀ C E dΩ."""
        E_t = self._macro_strain_tensor(E_voigt)
        if self.element_type == "c3d8i":
            return self._compute_rhs_c3d8i(E_t)
        # C: (B, n_elem, 6, 6); sigma_macro = C @ E -> (B, n_elem, 6)
        if self._material_layout == "packed_element_major":
            sigma_macro = _packed_voigt_matvec(self.C, E_t)
        else:
            sigma_macro = torch.einsum('beij,j->bei', self.C, E_t)
        # fe = Bᵀ σ_macro: einsum('gij, bei -> bej', B_mat, sigma_macro) * detJ
        fe = torch.einsum('gij,bei->bej', self.B_mat, sigma_macro) * self.detJ
        # scatter: (B, n_elem*24) → (B, n_dofs)
        Bsz = self.B
        rhs = torch.zeros(Bsz, self.mesh.n_dofs,
                           device=self.device, dtype=self.dtype)
        idx = self.elem_dof_flat.unsqueeze(0).expand(Bsz, -1)
        rhs.scatter_add_(1, idx, fe.reshape(Bsz, -1))
        return -rhs

    def compute_rhs_cuda(
        self,
        E_voigt,
        *,
        block_size=None,
        variant=None,
        verbose_build=False,
    ):
        """Assemble the ordinary C3D8 periodic RHS with native CUDA."""
        if self.element_type != "c3d8":
            raise NotImplementedError(
                "native CUDA RHS assembly is currently limited to ordinary C3D8"
            )
        if self.C is None:
            raise RuntimeError("C3D8 constitutive data were released")
        from .c3d8_cuda import c3d8_rhs_cuda

        if block_size is None:
            block_size = self._cuda_apply_block_size
        if variant is None:
            variant = self._cuda_apply_variant
        return c3d8_rhs_cuda(
            self.C,
            self._cuda_geometry,
            self._macro_strain_tensor(E_voigt),
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            variant=variant,
            verbose_build=verbose_build,
        )

    def compute_rhs(self, E_voigt):
        if self._apply_backend == "cuda":
            return self.compute_rhs_cuda(E_voigt)
        return self.compute_rhs_torch(E_voigt)

    def _compute_rhs_c3d8i(self, E_t):
        """Condensed C3D8I macro-strain load vector."""
        Bsz = self.B
        rhs = torch.zeros(Bsz, self.mesh.n_dofs,
                          device=self.device, dtype=self.dtype)
        chunk_size = self._c3d8i_chunk_size()
        for start in range(0, self.mesh.n_elem, chunk_size):
            stop = min(start + chunk_size, self.mesh.n_elem)
            C_chunk = self.C[:, start:stop]
            sigma_macro = torch.einsum('bcij,j->bci', C_chunk, E_t)
            f_u = torch.einsum('gip,bci->bcp', self.B_mat, sigma_macro) * self.detJ
            f_i = torch.einsum('gim,bci->bcm', self.B_inc_mat, sigma_macro) * self.detJ
            K_ui, K_ii, _ = self._c3d8i_element_matrices(C_chunk)
            alpha = torch.linalg.solve(K_ii, f_i.unsqueeze(-1)).squeeze(-1)
            f_cond = f_u - torch.einsum('bcpm,bcm->bcp', K_ui, alpha)

            idx = self.elem_dof_ids[start:stop].reshape(-1)
            rhs.scatter_add_(1, idx.unsqueeze(0).expand(Bsz, -1),
                             f_cond.reshape(Bsz, -1))
        return -rhs

    # --------------------------------------------------------
    def compute_avg_stress_torch(self, u_flat, E_voigt):
        """u_flat: (B, n_dofs). Returns np.ndarray (B, 6)."""
        E_t = self._macro_strain_tensor(E_voigt)
        if self.element_type == "c3d8i":
            return self._compute_avg_stress_c3d8i(u_flat, E_t)
        Bsz = self.B
        C_full = None if self._material_layout == "packed_element_major" else self.C
        sum_sig = torch.zeros(Bsz, 6, device=self.device, dtype=self.dtype)
        chunk_elems = 32768
        for start in range(0, self.mesh.n_elem, chunk_elems):
            end = min(start + chunk_elems, self.mesh.n_elem)
            elem_ids = self.elem_dof_ids[start:end].reshape(-1)
            u_elem = u_flat[:, elem_ids].view(Bsz, end - start, 24)
            strain_fluc = torch.einsum('gij,bej->begi', self.B_mat, u_elem)
            strain_total = strain_fluc + E_t
            if self._material_layout == "packed_element_major":
                stress = _packed_voigt_matvec(
                    self.C[:, start:end].unsqueeze(-2), strain_total,
                )
            else:
                stress = torch.einsum(
                    'beij,begj->begi', C_full[:, start:end], strain_total,
                )
            sum_sig.add_(torch.einsum('begi->bi', stress) * self.detJ)
        avg = sum_sig / self.mesh.V_total                                # (B, 6)
        return avg.detach().cpu().numpy().astype(np.float64)

    def compute_avg_stress_cuda(
        self,
        u_flat,
        E_voigt,
        *,
        block_size=None,
        verbose_build=False,
    ):
        """Recover exact volume-average C3D8 stress with native CUDA."""
        if self.element_type != "c3d8":
            raise NotImplementedError(
                "native CUDA stress recovery is currently limited to ordinary C3D8"
            )
        self._validate_apply_input(u_flat)
        if self.C is None:
            raise RuntimeError("C3D8 constitutive data were released")
        from .c3d8_cuda import c3d8_avg_stress_cuda

        if block_size is None:
            block_size = self._cuda_apply_block_size
        average = c3d8_avg_stress_cuda(
            u_flat,
            self.C,
            self._cuda_geometry,
            self._macro_strain_tensor(E_voigt),
            grid_shape=(self.mesh.Nz, self.mesh.Ny, self.mesh.Nx),
            block_size=block_size,
            verbose_build=verbose_build,
        )
        return average.detach().cpu().numpy().astype(np.float64)

    def compute_avg_stress(self, u_flat, E_voigt):
        if self._apply_backend == "cuda":
            return self.compute_avg_stress_cuda(u_flat, E_voigt)
        return self.compute_avg_stress_torch(u_flat, E_voigt)

    def _compute_avg_stress_c3d8i(self, u_flat, E_t):
        """Recover average stress with condensed C3D8I incompatible modes."""
        Bsz = self.B
        sum_sig = torch.zeros(Bsz, 6, device=self.device, dtype=self.dtype)
        chunk_size = self._c3d8i_chunk_size()
        for start in range(0, self.mesh.n_elem, chunk_size):
            stop = min(start + chunk_size, self.mesh.n_elem)
            C_chunk = self.C[:, start:stop]
            elem_ids = self.elem_dof_ids[start:stop].reshape(-1)
            u_elem = u_flat[:, elem_ids].view(Bsz, stop - start, 24)

            K_ui, K_ii, _ = self._c3d8i_element_matrices(C_chunk)
            sigma_macro = torch.einsum('bcij,j->bci', C_chunk, E_t)
            f_i = torch.einsum('gim,bci->bcm', self.B_inc_mat, sigma_macro) * self.detJ
            Kiu_u = torch.einsum('bcpm,bcp->bcm', K_ui, u_elem)
            alpha = -torch.linalg.solve(K_ii, (Kiu_u + f_i).unsqueeze(-1)).squeeze(-1)

            strain_u = torch.einsum('gip,bcp->bcgi', self.B_mat, u_elem)
            strain_i = torch.einsum('gim,bcm->bcgi', self.B_inc_mat, alpha)
            strain_total = strain_u + strain_i + E_t
            stress = torch.einsum('bcij,bcgj->bcgi', C_chunk, strain_total)
            sum_sig += torch.einsum('bcgi->bi', stress) * self.detJ

        avg = sum_sig / self.mesh.V_total
        return avg.detach().cpu().numpy().astype(np.float64)

    # --------------------------------------------------------
    def project_zero_mean(self, u_flat):
        """Subtract the per-component translational mean independently for each batch."""
        Bsz = self.B
        u = u_flat.view(Bsz, self.mesh.n_nodes, 3)
        u = u - u.mean(dim=1, keepdim=True)
        return u.reshape(Bsz, -1)

    # --------------------------------------------------------
    # PCG preconditioner: Jacobi (default) or FFT
    # --------------------------------------------------------
    def apply_precond(self, r_flat):
        """z = M^{-1} r. Falls back to Jacobi when the FFT preconditioner is disabled."""
        if self._fft_enabled:
            return self._apply_fft_precond(r_flat)
        if self._M_inv_cache is None:
            self._M_inv_cache = 1.0 / self._diag_cache
        return r_flat * self._M_inv_cache

    def enable_fft_precond(self, ref_strategy='zeman', C0_voigt=None,
                           verbose=False):
        """Precompute the FE Green's-function FFT preconditioner.

        The reference medium is homogeneous, so its periodic FE stiffness
        matrix is block-circulant on this regular grid. We build the reference
        *FE* 27-tap nodal stencil, transform that discrete stencil to its
        Fourier symbol ``K0_hat(k)``, and cache ``K0_hat(k)^{-1}`` on GPU.

        This is the finite-element Green's function preconditioner described
        in the Ladecky et al. 2023 line of work, not the continuum
        Moulinec-Suquet Green operator. The operator being inverted here is
        exactly the same stencil family used by ``apply()``.

        Parameters
        ----------
        ref_strategy : 'mean_eig' | 'zeman' | 'scalar' | 'mean_C'
                       Strategy for picking the reference medium (K0, μ0), following the
                       Moulinec-Suquet literature.
        C0_voigt     : Optional Voigt reference stiffness of shape (6,6) or (B,6,6);
                       overrides ref_strategy. Useful for sanity tests (passing the
                       material itself → exact inverse).
        """
        if self.element_type == "c3d8r":
            raise NotImplementedError(
                "FFT preconditioning is not supported for unstabilized C3D8R; "
                "the one-point reduced-integration operator has hourglass zero modes. "
                "Use element_type='c3d8' or add an hourglass-stabilized C3D8R operator."
            )
        if C0_voigt is not None:
            arr = np.array(C0_voigt, dtype=np.float64, copy=True)
            if arr.ndim == 2:
                arr = np.broadcast_to(arr, (self.B, 6, 6)).copy()
            assert arr.shape == (self.B, 6, 6), \
                f"C0_voigt must be (6,6) or (B,6,6), got {arr.shape}"
            C0_voigt = arr
        else:
            C0_voigt = self._compute_reference_medium(ref_strategy, verbose=verbose)
        # Build the 27 centre blocks S0[b, 27, 3, 3] of the reference-medium stencil per sample.
        # A uniform field's stencil is space-invariant, so use a 1×1×1 mini-grid for ~60x savings in memory and time.
        S0_center = np.zeros((self.B, 27, 3, 3), dtype=np.float64)
        for b in range(self.B):
            C0_mini = C0_voigt[b][:, :, None, None, None]     # (6,6,1,1,1)
            S0_full = build_node_stencil_from_C(
                C0_mini,
                self._B_np,
                self.detJ,
                element_type=self.element_type,
                voxel_size=(self.mesh.dx, self.mesh.dy, self.mesh.dz),
            )
            # (3,3,3, 3,3, 1,1,1) -> (27, 3, 3)
            S0_center[b] = S0_full[..., 0, 0, 0].reshape(27, 3, 3)

        # Build the symbol \hat M_0(k) = Σ_δ S0^δ · exp(+2πi k·δ/N)
        self._build_fft_symbol_inv(S0_center, verbose=verbose)
        self._fft_enabled = True
        self._fft_cheb_enabled = False
        self._fft_cheb_degree = 1
        self._fft_cheb_lmin = 0.2
        self._fft_cheb_lmax = 1.8
        # Record the arguments so the IR inner fp32 operator can reuse them
        self._fft_args = {'ref_strategy': ref_strategy, 'C0_voigt': C0_voigt}
        # Clear the Jacobi cache to avoid mixing preconditioners
        self._M_inv_cache = None

    def enable_fe_green_precond(self, ref_strategy='zeman', C0_voigt=None,
                                verbose=False):
        """FE-based Green's-function FFT preconditioner (Ladecky et al. 2023).

        This is an alias for ``enable_fft_precond()``. They produce identical
        numerical behavior because this codebase already implements the FE
        Green's-function preconditioner, not the continuous-spectrum
        Moulinec-Suquet Green operator that the term "FFT preconditioner" can
        imply in the homogenization literature.

        The distinction matters:

        * Moulinec-Suquet-style spectral solvers use an analytic Green
          function of the continuous PDE in Fourier space. At high material
          contrast, that continuous spectral representation can exhibit Gibbs
          oscillations and voxel-grid artifacts.

        * The FE-Green/Ladecky-style preconditioner uses the discretized
          Green's function of the homogeneous reference FE problem. On a
          periodic regular grid, the homogeneous FE stiffness is
          block-circulant, so its 3x3 nodal symbol is block-diagonal in
          Fourier space and can be inverted independently at each frequency.

        Our ``_build_fft_symbol_inv`` constructs ``K0_hat(k)`` from the
        27-tap nodal stencil assembled from the homogeneous reference medium
        and caches ``K0_hat(k)^{-1}``. In other words, the existing
        ``enable_fft_precond`` path is already the FE-Green formulation. This
        alias gives users coming from the FE-Green literature an unambiguous
        entry point and avoids re-implementing what is already present.

        Parameters are identical to ``enable_fft_precond``.

        Reference
        ---------
        Ladecky et al., "An optimal preconditioned FFT-accelerated finite
        element solver for homogenization", Applied Mathematics and
        Computation, 2023.
        """
        self.enable_fft_precond(ref_strategy=ref_strategy,
                                C0_voigt=C0_voigt, verbose=verbose)

    def disable_fft_precond(self):
        self._fft_enabled = False
        self._fft_cheb_enabled = False
        self._fft_symbol_inv = None
        self._fft_spectrum_mode = None
        self._fft_inverse_norm_folded = False

    def enable_fft_chebyshev_precond(self, degree=2, lmin=0.2, lmax=1.8,
                                     ref_strategy='zeman', C0_voigt=None,
                                     verbose=False):
        """Enable a fixed-degree Chebyshev-polynomial wrapper around the FFT preconditioner.

        The base FFT preconditioner applies ``M0^{-1}``, where ``M0`` is the
        homogeneous reference-medium stencil. This wrapper uses a fixed number
        of Chebyshev semi-iteration steps to approximate ``K^{-1} r`` for the
        actual heterogeneous operator ``K``:

            x_k ~= p_k(M0^{-1} K) M0^{-1} r

        ``lmin`` and ``lmax`` are estimates for the eigenvalue interval of the
        preconditioned operator ``M0^{-1} K``. ``degree=1`` reduces to one
        damped FFT-preconditioned Richardson step; each extra degree adds one
        more stencil apply plus one more FFT-preconditioner call inside every
        outer PCG preconditioner application. In practice this can reduce PCG
        iterations but is only faster if the iteration reduction outweighs that
        extra per-iteration work.
        """
        degree = int(degree)
        lmin = float(lmin)
        lmax = float(lmax)
        if degree < 1:
            raise ValueError("Chebyshev preconditioner degree must be >= 1")
        if not (0.0 < lmin < lmax):
            raise ValueError(f"expected 0 < lmin < lmax, got {lmin}, {lmax}")
        if not self._fft_enabled:
            self.enable_fft_precond(ref_strategy=ref_strategy,
                                    C0_voigt=C0_voigt, verbose=verbose)
        self._fft_cheb_enabled = True
        self._fft_cheb_degree = degree
        self._fft_cheb_lmin = lmin
        self._fft_cheb_lmax = lmax
        self._fft_args = {
            'ref_strategy': ref_strategy,
            'C0_voigt': C0_voigt,
            'cheb_degree': degree,
            'cheb_lmin': lmin,
            'cheb_lmax': lmax,
        }
        if verbose:
            print(f"  [fft-cheb] degree={degree}, lambda in [{lmin:.3g}, {lmax:.3g}]")

    # ---- Internal --------------------------------------------
    def _compute_reference_medium(self, strategy, *, verbose=False):
        """Per-sample estimation of an isotropic reference medium (K0, μ0) from the Voigt C field,
        returning (B,6,6) C0_voigt.

        After subsampling we copy to CPU for np.linalg.eigvalsh (small enough that GPU is not worth it).
        """
        Bsz = self.B
        Nz, Ny, Nx = self.mesh.Nz, self.mesh.Ny, self.mesh.Nx
        sz = max(1, Nz // 16); sy = max(1, Ny // 16); sx_ = max(1, Nx // 16)
        D = np.diag(MANDEL_W)                 # Voigt→Mandel scaling
        C0_all = np.zeros((Bsz, 6, 6), dtype=np.float64)

        # Subsampling and mean both run on GPU, then one copy back to CPU.
        # A packed owner is indexed in contiguous z-y-x element order; only
        # the small sampled subset is expanded for the eigensolver.
        C_gpu = self._C_voigt_gpu
        # The reference-medium helper is shared by the C3D8 and C3D8R
        # preconditioner paths.  C3D8R owns a spatial Voigt view but does not
        # define the FI-only material-layout tag; treat that established path
        # as legacy rather than letting an FI representation probe alter E5.
        material_layout = getattr(self, "_material_layout", "legacy_voigt")
        if material_layout == "packed_element_major":
            z = torch.arange(0, Nz, sz, device=self.device, dtype=torch.long)
            y = torch.arange(0, Ny, sy, device=self.device, dtype=torch.long)
            x = torch.arange(0, Nx, sx_, device=self.device, dtype=torch.long)
            sample_idx = (
                z[:, None, None] * (Ny * Nx)
                + y[None, :, None] * Nx
                + x[None, None, :]
            ).reshape(-1)
            sampled = _unpack_packed_voigt(C_gpu.index_select(1, sample_idx))
            C_sub_np = sampled.permute(0, 2, 3, 1).reshape(
                Bsz, 6, 6, z.numel(), y.numel(), x.numel(),
            ).detach().cpu().numpy().astype(np.float64)
            C_mean_np = _unpack_packed_voigt(
                C_gpu.mean(dim=1, keepdim=True)
            ).squeeze(1).detach().cpu().numpy().astype(np.float64)
        else:
            C_sub_np = C_gpu[:, :, :, ::sz, ::sy, ::sx_].detach().cpu().numpy().astype(np.float64)
            C_mean_np = C_gpu.mean(dim=(-3, -2, -1)).detach().cpu().numpy().astype(np.float64)   # (B,6,6)

        for b in range(Bsz):
            sub = C_sub_np[b].reshape(6, 6, -1).transpose(2, 0, 1)   # (Ns,6,6) Voigt
            sub_m = np.einsum('ij,njk,kl->nil', D, sub, D)           # Mandel, spectrum-comparable
            eigs = np.linalg.eigvalsh(sub_m)
            eig_min = float(eigs.min()); eig_max = float(eigs.max())

            C_mean = C_mean_np[b]                                    # (6,6), mean already computed on GPU
            K_avg = (C_mean[0, 0] + C_mean[1, 1] + C_mean[2, 2]
                     + 2.0 * (C_mean[0, 1] + C_mean[0, 2] + C_mean[1, 2])) / 9.0

            sub_v = sub
            mu_samples = 0.5 * (sub_v[:, 3, 3] + sub_v[:, 4, 4] + sub_v[:, 5, 5]) / 2.0
            K_samples = (sub_v[:, 0, 0] + sub_v[:, 1, 1] + sub_v[:, 2, 2]
                         + 2.0 * (sub_v[:, 0, 1] + sub_v[:, 0, 2] + sub_v[:, 1, 2])) / 9.0

            if strategy == 'mean_eig':
                mu0 = (eig_max + eig_min) / 4.0
                K0 = K_avg
                lam0 = K0 - 2.0 * mu0 / 3.0
            elif strategy == 'zeman':
                mu0 = 0.5 * (float(mu_samples.max()) + float(mu_samples.min()))
                K0 = 0.5 * (float(K_samples.max()) + float(K_samples.min()))
                lam0 = K0 - 2.0 * mu0 / 3.0
                if lam0 < 0.0:
                    K0 = 2.0 * mu0 / 3.0; lam0 = 0.0
            elif strategy == 'scalar':
                s = 0.5 * (eig_max + eig_min)
                mu0 = 0.5 * s; lam0 = 0.0; K0 = 2.0 * mu0 / 3.0
            elif strategy == 'mean_C':
                mu0 = 0.5 * (C_mean[3, 3] + C_mean[4, 4] + C_mean[5, 5]) / 3.0
                K0 = K_avg
                lam0 = K0 - 2.0 * mu0 / 3.0
            else:
                raise ValueError(f"unknown ref_strategy {strategy!r}")

            C0 = np.zeros((6, 6))
            for i in range(3):
                for j in range(3):
                    C0[i, j] = lam0
                C0[i, i] += 2.0 * mu0
            C0[3, 3] = C0[4, 4] = C0[5, 5] = mu0     # Voigt: diagonal = μ
            C0_all[b] = C0

            if verbose:
                print(f"  [fft-precond] sample {b}: K0={K0:.3e}, μ0={mu0:.3e}, "
                      f"λ0={lam0:.3e}  eig∈[{eig_min:.2e},{eig_max:.2e}]")
        return C0_all

    def _build_fft_symbol_inv(self, S0_center, *, verbose=False,
                              validate_nonzero=False, spectrum_mode="complex_full",
                              validation_mode="legacy_eig"):
        """S0_center: (B|1, 27, 3, 3). Build the cached nodal symbol inverse."""
        S0_center = np.asarray(S0_center)
        Bsym = int(S0_center.shape[0])
        if Bsym not in (1, self.B):
            raise ValueError(
                f"S0_center batch dimension must be 1 or {self.B}, got {Bsym}"
            )
        Nz, Ny, Nx = self.mesh.Nz, self.mesh.Ny, self.mesh.Nx
        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        real_dtype    = torch.float32   if self.dtype == torch.float32 else torch.float64
        if spectrum_mode == "complex_full":
            Nx_spec = Nx
        elif spectrum_mode == "real_half":
            Nx_spec = Nx // 2 + 1
        else:
            raise ValueError(f"unknown FFT spectrum mode {spectrum_mode!r}")

        # Integer k grid, matching torch.fft.fftn's indexing
        kz = torch.arange(Nz, device=self.device, dtype=real_dtype)
        ky = torch.arange(Ny, device=self.device, dtype=real_dtype)
        kx = torch.arange(Nx_spec, device=self.device, dtype=real_dtype)
        # phase[tap, Nz, Ny, Nx_spec] = exp(+2πi (kz*dz/Nz + ky*dy/Ny + kx*dx/Nx))
        two_pi = 2.0 * np.pi
        symbol = torch.zeros((Bsym, 3, 3, Nz, Ny, Nx_spec), dtype=complex_dtype,
                             device=self.device)
        S0_t = torch.tensor(S0_center, device=self.device, dtype=real_dtype)  # (B,27,3,3)

        for tap in range(27):
            ix = tap % 3;       iy = (tap // 3) % 3;     iz = tap // 9
            dx = ix - 1;        dy = iy - 1;             dz = iz - 1
            if dx == 0 and dy == 0 and dz == 0:
                # Constant term: add directly
                symbol += S0_t[:, tap, :, :, None, None, None].to(complex_dtype)
                continue
            phase_arg = (two_pi * (kz.view(-1, 1, 1) * dz / Nz
                                   + ky.view(1, -1, 1) * dy / Ny
                                   + kx.view(1, 1, -1) * dx / Nx))
            phase = torch.complex(torch.cos(phase_arg), torch.sin(phase_arg))   # (Nz,Ny,Nx_spec)
            symbol = symbol + (S0_t[:, tap, :, :, None, None, None].to(complex_dtype)
                               * phase[None, None, None, :, :, :])

        # Take the 3×3 inverse at each k. k=(0,0,0) is the rigid-body-mode null space; set to 0.
        # (B,3,3,Nz,Ny,Nx) -> (B,Nz,Ny,Nx,3,3)
        sym_nhwc = symbol.permute(0, 3, 4, 5, 1, 2).contiguous()
        sym_flat = sym_nhwc.reshape(-1, 3, 3)
        zero_idxs = torch.arange(Bsym, device=self.device) * (Nz * Ny * Nx_spec)
        if validation_mode == "legacy_eig":
            validate = _validate_nonzero_fft_symbol
        elif validation_mode == "vectorized_hermitian":
            validate = validate_nonzero_fft_symbol_vectorized
        else:
            raise ValueError(f"unknown FFT symbol validation mode {validation_mode!r}")
        if validate_nonzero:
            validate(symbol, real_dtype)
        # Numerical safety: place a small identity at k=0 so inv doesn't fail, then zero it out
        I3 = torch.eye(3, dtype=complex_dtype, device=self.device)
        # Build the global batch indices: b * Nz*Ny*Nx + 0
        sym_flat[zero_idxs] = I3                    # Placeholder, cleared below
        inv_flat = torch.linalg.inv(sym_flat)
        # Zero out k=(0,0,0)
        inv_flat[zero_idxs] = 0.0
        # ``norm="forward"`` removes the inverse FFT's per-call 1/N scaling.
        # Fold that constant into the cached inverse once so every subsequent
        # preconditioner application avoids a separate normalization pass.
        inv_flat.mul_(1.0 / float(Nz * Ny * Nx))
        inv_nhwc = inv_flat.reshape(Bsym, Nz, Ny, Nx_spec, 3, 3)
        self._fft_symbol_inv = inv_nhwc.permute(0, 4, 5, 1, 2, 3).contiguous()
        self._fft_spectrum_mode = spectrum_mode
        self._fft_inverse_norm_folded = True

        if verbose:
            mem_mb = self._fft_symbol_inv.element_size() * self._fft_symbol_inv.numel() / 1e6
            print(f"  [fft-precond] symbol-inverse cache: {mem_mb:.1f} MB, dtype={complex_dtype}")

    def _apply_fft_precond(self, r_flat):
        """z = IFFT(symbol_inv · FFT(r)), applied in nodal space."""
        if getattr(self, '_fft_cheb_enabled', False):
            return self._apply_fft_chebyshev_precond(r_flat)
        return self._apply_fft_base_precond(r_flat)

    def set_fft_batch_chunk_size(self, batch_chunk_size=None):
        """Optionally split FFT preconditioning along the sample batch.

        ``None`` keeps the single batched transform. A positive integer is an
        explicit execution hint; the best value depends on GPU, dtype, grid,
        and batch size, so this setting is intentionally not auto-tuned.
        """
        if batch_chunk_size is None:
            self._fft_batch_chunk_size = None
            return
        try:
            chunk_size = operator.index(batch_chunk_size)
        except TypeError as exc:
            raise ValueError(
                "FFT batch chunk size must be a positive integer or None"
            ) from exc
        if chunk_size < 1:
            raise ValueError(
                "FFT batch chunk size must be a positive integer or None"
            )
        self._fft_batch_chunk_size = chunk_size

    def _apply_fft_base_precond(self, r_flat):
        """z = IFFT(symbol_inv · FFT(r)), applied in nodal space."""
        Bsz = self.B
        chunk_size = getattr(self, "_fft_batch_chunk_size", None)
        if chunk_size is None or chunk_size >= Bsz:
            return self._apply_fft_base_precond_batch(
                r_flat,
                self._fft_symbol_inv,
            )

        result = torch.empty_like(r_flat)
        symbol_inv = self._fft_symbol_inv
        for start in range(0, Bsz, chunk_size):
            stop = min(start + chunk_size, Bsz)
            symbol_chunk = (
                symbol_inv
                if symbol_inv.shape[0] == 1
                else symbol_inv[start:stop]
            )
            result[start:stop] = self._apply_fft_base_precond_batch(
                r_flat[start:stop],
                symbol_chunk,
            )
        return result

    def _apply_fft_base_precond_batch(self, r_flat, symbol_inv):
        """Apply one full or explicitly chunked FFT batch."""
        Bsz = int(r_flat.shape[0])
        mesh = self.mesh
        # (B, n_dofs) -> (B, n_nodes, 3) -> (B, 3, Nz, Ny, Nx)
        r_field = r_flat.view(Bsz, mesh.n_nodes, 3).transpose(1, 2).contiguous().view(
            Bsz, 3, mesh.Nz, mesh.Ny, mesh.Nx,
        )
        spectrum_mode = getattr(self, "_fft_spectrum_mode", "complex_full")
        if spectrum_mode == "real_half":
            r_hat = torch.fft.rfftn(
                r_field.to(self.dtype),
                s=(mesh.Nz, mesh.Ny, mesh.Nx),
                dim=(-3, -2, -1),
            )
        elif spectrum_mode == "complex_full":
            r_hat = torch.fft.fftn(
                r_field.to(symbol_inv.dtype), dim=(-3, -2, -1),
            )
        else:
            raise RuntimeError(f"unknown cached FFT spectrum mode {spectrum_mode!r}")
        inverse_norm = (
            "forward"
            if getattr(self, "_fft_inverse_norm_folded", False)
            else "backward"
        )
        spectral_multiply = getattr(self, "_fft_spectral_multiply", None)
        if spectral_multiply is not None:
            z_hat = spectral_multiply(symbol_inv, r_hat)
        elif symbol_inv.shape[0] == 1 and Bsz != 1:
            z_hat = torch.einsum('pqzyx,bqzyx->bpzyx', symbol_inv[0], r_hat)
        elif symbol_inv.shape[0] == Bsz:
            z_hat = torch.einsum('bpqzyx,bqzyx->bpzyx', symbol_inv, r_hat)
        else:
            raise RuntimeError(
                f"FFT symbol batch dimension {symbol_inv.shape[0]} is incompatible with B={Bsz}"
            )
        if spectrum_mode == "real_half":
            z_field = torch.fft.irfftn(
                z_hat,
                s=(mesh.Nz, mesh.Ny, mesh.Nx),
                dim=(-3, -2, -1),
                norm=inverse_norm,
            ).to(r_flat.dtype)
        else:
            z_field = torch.fft.ifftn(
                z_hat, dim=(-3, -2, -1), norm=inverse_norm,
            ).real.to(r_flat.dtype)
        return z_field.view(Bsz, 3, -1).transpose(1, 2).contiguous().view(Bsz, -1)

    def _fft_spectral_multiply(self, symbol_inverse, rhs_spectrum):
        """Multiply the cached 3x3 FFT symbol by a batched spectrum."""
        if self._apply_backend == "cuda":
            from .c3d8_cuda import c3d8_fft_spectral_multiply_cuda

            return c3d8_fft_spectral_multiply_cuda(
                symbol_inverse,
                rhs_spectrum,
                block_size=self._cuda_apply_block_size,
            )
        batch = int(rhs_spectrum.shape[0])
        if symbol_inverse.shape[0] == 1 and batch != 1:
            return torch.einsum(
                'pqzyx,bqzyx->bpzyx',
                symbol_inverse[0],
                rhs_spectrum,
            )
        if symbol_inverse.shape[0] == batch:
            return torch.einsum(
                'bpqzyx,bqzyx->bpzyx',
                symbol_inverse,
                rhs_spectrum,
            )
        raise RuntimeError(
            "FFT symbol batch dimension "
            f"{symbol_inverse.shape[0]} is incompatible with B={batch}"
        )

    def _apply_fft_chebyshev_precond(self, r_flat):
        """Apply the Chebyshev-polynomial FFT preconditioner.

        This is a stationary inner solve with exactly ``degree`` steps. It is
        not adaptive: the caller must provide a usable spectral interval. The
        recurrence below is the standard Chebyshev semi-iteration form for
        solving ``K x = r`` with left preconditioner ``M0^{-1}``.
        """
        degree = int(self._fft_cheb_degree)
        lmin = float(self._fft_cheb_lmin)
        lmax = float(self._fft_cheb_lmax)

        # d/c are the center and half-width of the estimated spectrum of
        # M0^{-1}K. Bad bounds do not break the code, but can make the
        # polynomial ineffective or unstable.
        d = 0.5 * (lmax + lmin)
        c = 0.5 * (lmax - lmin)

        # First damped Richardson/Chebyshev step:
        #   x_1 = (1/d) M0^{-1} r
        x = torch.zeros_like(r_flat)
        residual = r_flat
        z = self._apply_fft_base_precond(residual)
        alpha = 1.0 / d
        p = z
        x = x + alpha * p

        for it in range(1, degree):
            # Remaining fixed inner iterations. Each loop costs:
            #   1 heterogeneous stencil apply Kx
            #   1 base FFT preconditioner M0^{-1}(r - Kx)
            # so degree>1 is deliberately opt-in.
            residual = r_flat - self.apply(x)
            z = self._apply_fft_base_precond(residual)
            beta = (0.5 * c * alpha) ** 2
            alpha = 1.0 / (d - beta / alpha)
            p = z + beta * p
            x = x + alpha * p
        return x
