"""Exact shared-phase / voxel-C21 storage; never expands a full C21 field.

Global-frame phases (including isotropic phases) share one coefficient row.
Local-frame anisotropic voxels retain their pre-rotated FP64/FP32 C21 rows.
An int32 element-to-row map selects either kind, without a kernel branch.
"""
from numbers import Integral

import numpy as np

from .material_field import PAIRS, phase_table, _orientation_values, _rotate_packed


def build_hybrid_material(phase_id, materials, *, orientation=None, device="cuda",
                          dtype=None, chunk_size=65536):
    """Return (coefficient_pool, row_index, metadata).

    pool: contiguous (number_of_phases + number_of_local_voxels, 21).
    index: contiguous int32 (B,N). Inputs follow build_material_field.
    Only explicit local-frame anisotropy gets individual rows. No tolerance
    clustering, quantization, isotropy inference, or dropped tensor entries.
    """
    import torch
    phase = np.asarray(phase_id)
    if phase.ndim not in (3, 4) or 0 in phase.shape or phase.dtype.kind not in "iu":
        raise ValueError("phase_id must be a nonempty integer 3D or batched 3D array")
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    dtype = torch.float64 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Material dtype must be float32 or float64")
    labels, tensors, local = phase_table(materials)
    used = np.unique(phase)
    if not np.isin(used, labels).all():
        raise ValueError("Undefined material labels")
    if orientation is None and local[np.searchsorted(labels, used)].any():
        raise ValueError("Active local-frame anisotropic phases require orientation")
    phase_rows = np.searchsorted(labels, phase.reshape(-1))
    count = int(np.count_nonzero(local[phase_rows]))
    rows = len(labels) + count
    if rows > np.iinfo(np.int32).max:
        raise ValueError("Hybrid coefficient pool exceeds int32 indexing")
    index = phase_rows.astype(np.int32)
    pool = torch.empty((rows, 21), device=device, dtype=dtype)
    ii, jj = tuple(i for i, j in PAIRS), tuple(j for i, j in PAIRS)
    pool[:len(labels)].copy_(torch.tensor(tensors[:, ii, jj], device=device, dtype=dtype))
    offset = len(labels)
    with torch.no_grad():
        for p in np.flatnonzero(local):
            selected = np.flatnonzero(phase_rows == p)
            if not selected.size:
                continue
            index[selected] = np.arange(offset, offset + selected.size, dtype=np.int32)
            tensor = torch.tensor(tensors[p], device=device, dtype=dtype)
            for start in range(0, selected.size, chunk_size):
                stop = min(start + chunk_size, selected.size)
                R = torch.tensor(_orientation_values(orientation, phase.shape, selected[start:stop]),
                                 device=device, dtype=dtype)
                pool[offset + start:offset + stop].copy_(_rotate_packed(tensor, R))
                del R
            offset += selected.size
            del tensor
    B = phase.shape[0] if phase.ndim == 4 else 1
    index_gpu = torch.tensor(index.reshape(B, -1), device=device, dtype=torch.int32)
    metadata = dict(labels=labels.tolist(), n_elements=int(phase.size), n_local_voxels=count,
                    n_shared_voxels=int(phase.size-count), pool_rows=rows,
                    coefficient_bytes=pool.numel()*pool.element_size(),
                    index_bytes=index_gpu.numel()*index_gpu.element_size(),
                    dense_bytes=int(phase.size)*21*pool.element_size())
    return pool, index_gpu, metadata
