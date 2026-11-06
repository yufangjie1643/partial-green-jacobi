"""Geometry-independent, bounded-workspace construction of elastic C21 fields.

Engineering Voigt order is (11,22,33,23,13,12). Phase labels have no geometric
meaning. All input arrays are host arrays; only the output and bounded chunks
are transferred to the requested torch device. See material_field_interface.md.
"""
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .materials import engineering_to_voigt, isotropic_to_voigt

PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))
ORTHOTROPIC_KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")


def material_stiffness(spec):
    """Validate one phase and return its symmetric positive-definite 6x6 tensor.

    frame='global' (default) needs no orientation. frame='local' requires
    orientation for non-isotropic phases. Isotropy is explicitly declared,
    never inferred with a tolerance that could discard physical anisotropy.
    """
    kind = spec.get("kind")
    if spec.get("frame", "global") not in ("global", "local"):
        raise ValueError("Material frame must be 'global' or 'local'")
    if kind == "isotropic":
        E, nu = float(spec["E"]), float(spec["nu"])
        if not (np.isfinite([E, nu]).all() and E > 0 and -1 < nu < .5):
            raise ValueError("Isotropic elasticity requires E>0 and -1<nu<0.5")
        C = isotropic_to_voigt(E, nu)
    elif kind == "orthotropic":
        v = np.array([spec[key] for key in ORTHOTROPIC_KEYS], dtype=float)
        if not np.isfinite(v).all() or (v[[0, 1, 2, 6, 7, 8]] <= 0).any():
            raise ValueError("Orthotropic moduli must be finite and positive")
        try:
            C = engineering_to_voigt(*v)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular orthotropic compliance") from exc
    elif kind == "anisotropic":
        C = np.asarray(spec["C"], dtype=np.float64)
    else:
        raise ValueError("kind must be isotropic, orthotropic or anisotropic")
    if C.shape != (6, 6) or not np.isfinite(C).all():
        raise ValueError("C must be a finite engineering-Voigt 6x6 matrix")
    if not np.allclose(C, C.T, rtol=0, atol=1e-12 * np.max(np.abs(C))):
        raise ValueError("Elastic stiffness must be symmetric")
    C = (C + C.T) * .5  # remove only accepted roundoff asymmetry
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Elastic stiffness must be positive definite") from exc
    return C


def phase_table(materials):
    """Return sorted labels, validated tensors and local-frame flags."""
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError("materials must be a nonempty mapping of integer labels")
    if any(not isinstance(k, Integral) or isinstance(k, bool) for k in materials):
        raise ValueError("Material labels must be integers")
    labels = np.array(sorted(materials), dtype=np.int64)
    specs = [materials[int(k)] for k in labels]
    tensors = np.stack([material_stiffness(s) for s in specs])
    local = np.array([s["kind"] != "isotropic" and s.get("frame", "global") == "local"
                      for s in specs])
    return labels, tensors, local


def _orientation_values(orientation, shape, flat_indices):
    """Read only selected host entries, including broadcasted constant frames."""
    coordinates = np.unravel_index(flat_indices, shape)

    def select(values, tail):
        try:
            view = np.broadcast_to(np.asarray(values, dtype=float), shape + tail)
        except ValueError as exc:
            raise ValueError("Orientation shape is not broadcastable to phase_id") from exc
        selected = np.array(view[coordinates], copy=True)
        if not np.isfinite(selected).all():
            raise ValueError("Active local-frame orientations must be finite")
        return selected

    if isinstance(orientation, Mapping):
        a, b = select(orientation["axis1"], (3,)), select(orientation["axis2"], (3,))
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        if (na <= 0).any() or (nb <= 0).any():
            raise ValueError("Orientation axes must be nonzero")
        a, b = a / na[:, None], b / nb[:, None]
        c = np.cross(a, b)
        nc = np.linalg.norm(c, axis=-1)
        if (nc <= 1e-12).any():
            raise ValueError("Orientation axes must not be collinear")
        c /= nc[:, None]
        return np.stack([a, np.cross(c, a), c], axis=-1)
    R = select(orientation, (3, 3))
    if not np.allclose(R.swapaxes(-1, -2) @ R, np.eye(3), rtol=0, atol=1e-10):
        raise ValueError("Orientation must be an orthonormal rotation matrix")
    if not np.allclose(np.linalg.det(R), 1, rtol=0, atol=1e-10):
        raise ValueError("Orientation must be right-handed (det=+1)")
    return R


def _rotate_packed(C, R):
    """Bond stress transform, emitting only upper-triangle coefficients."""
    import torch
    i, j = [1, 0, 0], [2, 2, 1]
    r1, r2 = R[:, i, :], R[:, j, :]
    T = torch.cat((torch.cat((R * R, 2 * R[:, :, i] * R[:, :, j]), -1),
                   torch.cat((r1 * r2, r1[:, :, i] * r2[:, :, j]
                              + r1[:, :, j] * r2[:, :, i]), -1)), -2)
    TC = T @ C
    return torch.stack([(TC[:, i, :] * T[:, j, :]).sum(-1) for i, j in PAIRS], -1)


def build_material_field(phase_id, materials, *, orientation=None, device="cuda",
                         dtype=None, chunk_size=65536):
    """Return (B,N,21) C21 for a 3D or batched 3D integer phase_id array.

    orientation is a rotation array (...,3,3), or {'axis1': (...,3),
    'axis2': (...,3)}. Constant/broadcasted values are supported. It is read
    only at non-isotropic, frame='local' voxels. R columns are local axes in
    global coordinates. Material coefficients default to the global frame.
    No nodal, stencil, quadrature or solver change is made here.
    """
    import torch
    phase = np.asarray(phase_id)
    if phase.ndim not in (3, 4) or 0 in phase.shape or phase.dtype.kind not in "iu":
        raise ValueError("phase_id must be a nonempty integer (Nz,Ny,Nx) or (B,Nz,Ny,Nx) array")
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    dtype = torch.float64 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Material field dtype must be float32 or float64")
    labels, Cs, local = phase_table(materials)
    used = np.unique(phase)
    if not np.isin(used, labels).all():
        raise ValueError(f"Undefined material labels: {used[~np.isin(used, labels)].tolist()}")
    if orientation is None and local[np.searchsorted(labels, used)].any():
        raise ValueError("Active local-frame anisotropic phases require orientation")
    flat = phase.reshape(-1)
    table = torch.tensor(Cs[:, tuple(i for i, j in PAIRS), tuple(j for i, j in PAIRS)],
                         device=device, dtype=dtype)
    full_table = torch.tensor(Cs, device=device, dtype=dtype) if local.any() else None
    field = torch.empty((flat.size, 21), device=device, dtype=dtype)
    with torch.no_grad():
        for start in range(0, flat.size, chunk_size):
            stop = min(start + chunk_size, flat.size)
            indices = np.searchsorted(labels, flat[start:stop])
            ids = torch.tensor(indices, dtype=torch.int64, device=device)
            torch.index_select(table, 0, ids, out=field[start:stop])
            del ids
            for p in np.flatnonzero(local):
                selected = np.flatnonzero(indices == p)
                if not selected.size:
                    continue
                R = torch.tensor(_orientation_values(orientation, phase.shape, start + selected),
                                 device=device, dtype=dtype)
                rotated = _rotate_packed(full_table[p], R)
                target = torch.tensor(selected, dtype=torch.int64, device=device)
                field[start:stop].index_copy_(0, target, rotated)
                del R, rotated, target
    B = phase.shape[0] if phase.ndim == 4 else 1
    return field.reshape(B, -1, 21)


def from_legacy_yarn(data, material):
    """Explicit adapter for old matrix/yarn inputs; the core API needs no yarn_id.

    Yarn numbers identify geometry only. All nonnegative IDs use the one legacy
    yarn material. Distinct yarn materials should be mapped to distinct phase IDs
    by the caller. Legacy orthotropic parameters remain orthotropic, even if close
    to isotropic; use kind='isotropic' in the core API for exact isotropy.
    """
    m, y = material["matrix"], material["yarn"]
    table = {0: dict(kind="isotropic", E=m["E"], nu=m["Nu"])}
    if "E11" in y:
        old = ("E11", "E22", "E33", "Nu12", "Nu13", "Nu23", "G12", "G13", "G23")
        table[1] = dict(kind="orthotropic", frame="local", **dict(zip(ORTHOTROPIC_KEYS, (y[k] for k in old))))
    else:
        table[1] = dict(kind="isotropic", E=y["E"], nu=y["Nu"])
    phase = (np.asarray(data["yarn_id"]) >= 0).astype(np.uint8)
    orient = None
    if table[1].get("frame") == "local" and np.any(phase == 1):
        orient = {"axis1": data["orient1"], "axis2": data["orient2"]}
    return phase, table, orient
