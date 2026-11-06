"""Thermoelastic and elastic-localization extensions of the frozen HG solver.

Units: MPa, input length units, kelvin. Engineering strain order is
(11,22,33,23,13,12); stress shear components are tensor shear components.
This module never edits the frozen femlib or changes its stopping tolerance.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

os.environ['PATH'] = str(Path(sys.executable).parent) + ':/usr/local/cuda/bin:' + os.environ.get('PATH', '')
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('FEAD_SOURCE', str(ROOT / 'repro/source')))
sys.path.insert(0, str(SOURCE))
import torch
from femlib.fem_batched import FEMHomogenizerBatched
from femlib.batched.pcg import solve_pcg_batched
from femlib.batched.c3d8r_hourglass_operator import _packed_matvec
from femlib.mesh import build_voigt_stiffness_fields_gpu

PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
DATA_HASH = 'b430b0fd5398ff5274cc5ca273eef04d2471793b8f915138f4e7169875a6e9b3'
THERMAL = {
    'source': 'https://doi.org/10.3390/aerospace9100537',
    'source_tables': [8, 9],
    'matrix': {'E': 3350., 'Nu': .35},
    'yarn': dict(E11=174670., E22=9510., E33=9510., G12=8520., G13=8280.,
                 G23=3390., Nu12=.28, Nu13=.28, Nu23=.41),
    'alpha_matrix': 58.e-6,
    'alpha_yarn': [-.22e-6, 24.91e-6, 24.91e-6],
    'temperature_increment_K': 1.,
}


def _evaluation_peer(pid):
    """Allow siblings only when this process also descends from the named launcher."""
    root = os.environ.get('VOXEL_EVALUATION_PARENT_PID', '')
    if not root.isdigit():
        return False
    root = int(root)

    def belongs(candidate):
        visited = set()
        while candidate > 1 and candidate not in visited:
            if candidate == root:
                return True
            visited.add(candidate)
            try:
                candidate = int(Path(f'/proc/{candidate}/stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
        return False

    return belongs(os.getpid()) and belongs(pid)


def gpu_guard():
    state = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                     '--format=csv,noheader'], text=True).strip()
    if len(state.splitlines()) != 1 or 'RTX 5090' not in state:
        raise RuntimeError('Protocol requires one RTX 5090')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True).strip()
    foreign = [r for r in apps.splitlines()
               if int(r.split(',')[0]) != os.getpid() and not _evaluation_peer(int(r.split(',')[0]))]
    if foreign:
        raise RuntimeError('GPU OCCUPIED: stopping without touching foreign processes: ' + '; '.join(foreign))
    return state


def synchronize():
    torch.cuda.synchronize()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def read_dataset(path, indices=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_HASH:
        raise ValueError('Unexpected woven dataset hash')
    with np.load(path, allow_pickle=True) as src:
        if indices is None:
            indices = np.arange(len(src['sample_ids']))
        data = {k: np.asarray(src[k][indices]).copy() for k in
                ['sample_ids', 'domain_size', 'yarn_id', 'orient1', 'orient2']}
        data['grid_shape'] = tuple(map(int, src['grid_shape']))
        data['indices'] = np.asarray(indices)
    data['sha256'] = digest
    return data


def make_solver(C, grid, voxel, hg=.1, fft=True):
    solver = FEMHomogenizerBatched(C, voxel, grid, dtype=torch.float64,
                                  element_type='C3D8R', hourglass_coefficient=hg)
    op = solver.op
    op.set_apply_backend('cuda', cuda_variant='thread', cuda_block_size=128)
    if fft:
        op.enable_fft_precond(ref_strategy='zeman', spectrum_mode='real_half',
                              validation_mode='vectorized_hermitian', allow_symbol_sharing=False)
    return solver


def build_batch(data, start, stop, material=THERMAL, hg=.1):
    meshes = [SimpleNamespace(grid_shape=data['grid_shape'], yarn_id=data['yarn_id'][i],
                              orient1=data['orient1'][i], orient2=data['orient2'][i])
              for i in range(start, stop)]
    C = build_voigt_stiffness_fields_gpu(meshes, material['matrix'], material['yarn'],
                                        use_orientations=True, dtype=torch.float64)
    nz, ny, nx = data['grid_shape']
    domain = data['domain_size'][start:stop]
    if not np.allclose(domain, domain[0]):
        raise ValueError('A batch requires shared cell dimensions')
    voxel = domain[0] / np.array([nx, ny, nz])
    return make_solver(C, data['grid_shape'], voxel, hg=hg)


def solve_checked(op, rhs, max_iter=10000):
    """Accept only the freshly recomputed true residual <= 1e-10."""
    norm = torch.linalg.vector_norm(rhs, dim=1)
    if bool((norm == 0).all()):
        return torch.zeros_like(rhs), dict(iterations=0, restarts=0, true_residuals=norm.cpu().tolist())
    if bool((norm == 0).any()):
        # A zero member remains exactly zero; PCG's denominator floors handle it.
        pass
    total, restarts, u = 0, 0, None
    while total < max_iter:
        u, count, _ = solve_pcg_batched(op, rhs, tol=1e-10, max_iter=max_iter-total,
                                       update_mode='compiled', convergence_check_stride=4,
                                       verify_final_residual=True, x0=u)
        total += int(count)
        u = op.project_zero_mean(u)
        residual = torch.linalg.vector_norm(rhs-op.apply(u), dim=1)/norm.clamp_min(1e-300)
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError('Nonfinite true residual')
        if float(residual.max()) <= 1e-10:
            return u, dict(iterations=total, restarts=restarts, true_residuals=residual.cpu().tolist())
        if not count:
            break
        restarts += 1
    raise RuntimeError(f'Unconverged at {total} steps: {residual.tolist()}')


def elastic_basis(op, retain='none'):
    """Recover raw C_H; optionally retain displacements or center stress bases."""
    C = np.empty((op.B, 6, 6))
    saved, reports = [], []
    for j in range(6):
        e = np.eye(6)[j]
        u, report = solve_checked(op, op.project_zero_mean(op.compute_rhs(e)))
        C[:, :, j] = op.compute_avg_stress(u, e)
        if retain == 'displacements':
            saved.append(u)
        elif retain == 'stress':
            saved.append(op.recover_center_fields(u, e)[1])
        reports.append(report)
    return C, (torch.stack(saved, dim=-1) if saved else None), reports


def rotation_fields(data, start, stop, device):
    """Same right-handed orthonormalization as the frozen stiffness builder."""
    mask = torch.as_tensor(np.asarray(data['yarn_id'][start:stop] >= 0).reshape(stop-start, -1),
                           device=device)
    R = torch.eye(3, dtype=torch.float64, device=device).expand(*mask.shape, 3, 3).clone()
    a = torch.as_tensor(np.asarray(data['orient1'][start:stop]).reshape(stop-start, -1, 3),
                        dtype=torch.float64, device=device)[mask]
    b = torch.as_tensor(np.asarray(data['orient2'][start:stop]).reshape(stop-start, -1, 3),
                        dtype=torch.float64, device=device)[mask]
    a = a/torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-14)
    c = torch.cross(a, b, dim=-1)
    c = c/torch.linalg.vector_norm(c, dim=-1, keepdim=True).clamp_min(1e-14)
    b = torch.cross(c, a, dim=-1)
    R[mask] = torch.stack([a, b, c], dim=-1)
    if not bool(torch.isfinite(R).all()) or float((R[mask].transpose(-1,-2)@R[mask]-torch.eye(3,device=device)).abs().max()) > 1e-10:
        raise ValueError('Invalid yarn coordinate frame')
    return R, mask


def expansion_field(R, mask, material=THERMAL):
    a = torch.as_tensor(material['alpha_yarn'], dtype=R.dtype, device=R.device)
    tensor = (R*a)@R.transpose(-1, -2)
    field = torch.stack([tensor[..., i, j]*(1 if i == j else 2) for i,j in VOIGT], dim=-1)
    field[~mask] = field.new_tensor([material['alpha_matrix']]*3 + [0.]*3)
    return field


def eigenstrain_rhs(op, eigenstrain):
    stress = _packed_matvec(op.C, eigenstrain)
    force = torch.einsum('iap,bei->beap', op.B_center, stress)*op.mesh.V_elem
    return op.project_zero_mean(op._scatter_element_vectors(force))


def thermal_response(op, C_H, alpha):
    # Unit temperature increment; alpha uses engineering shear components.
    rhs = eigenstrain_rhs(op, alpha)
    u, report = solve_checked(op, rhs)
    _, sigma_el = op.recover_center_fields(u, np.zeros(6))
    sigma = sigma_el - _packed_matvec(op.C, alpha)
    average = sigma.mean(1).cpu().numpy()
    alpha_H = np.linalg.solve(C_H, -average[..., None])[..., 0]
    return alpha_H, u, average, report


def rotate_stress_local(stress, R):
    tensor = stress.new_empty(stress.shape[:-1] + (3, 3))
    for k, (i,j) in enumerate(VOIGT):
        tensor[..., i,j] = tensor[..., j,i] = stress[..., k]
    local = R.transpose(-1,-2)@tensor@R
    return torch.stack([local[..., i,j] for i,j in VOIGT], dim=-1)


def provenance():
    gpu = gpu_guard()
    source = SOURCE/'femlib'
    return dict(gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda,
                dtype='float64', tolerance=1e-10, max_iter=10000,
                hg_coefficient=.1, operator='CUDA thread128',
                fft='real-half Hermitian upper6, per-RVE reference',
                source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.rglob('*')) if p.suffix in ['.py','.cu','.cpp','.h']})
