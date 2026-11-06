"""Controlled 3D HEX8 transfer of Ladecky 2026 analytic coefficient fields."""
import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import MethodType
from project_paths import ROOT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def setup():
    global np, torch, old, comp, HybridFIOperator, GaussFIOperator, gradients, OFFSETS, pcg, C0
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location('smooth_old', ROOT/'experiments/ladecky_comparison/study.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    _, comp, _ = old.setup()
    from hybrid_adapter import HybridFIOperator
    from ladecky_operator import GaussFIOperator, gradients, OFFSETS
    from femlib.batched.pcg import solve_pcg_batched as pcg
    C0 = np.zeros((6, 6))
    C0[:3, :3] = 4/3
    C0[np.arange(3), np.arange(3)] += 2
    C0[np.arange(3, 6), np.arange(3, 6)] = 1
    torch.set_num_threads(1)


def density(case, n, p, chi):
    assert n % p == 0 and p >= 4
    x = np.arange(p)/p
    if case == 'laminate':
        plane = np.broadcast_to(chi+(1-chi)*x/(1-1/p), (p, p))
    else:
        xx, yy = np.meshgrid(x, x, indexing='xy')
        base = .5+.25*(np.cos(2*np.pi*(xx-yy))+np.cos(2*np.pi*(yy+xx)))
        plane = np.where(base >= .5, 1., 1/chi) if case == 'sharp_cosine' else base+1/chi
    plane = np.repeat(np.repeat(plane, n//p, axis=0), n//p, axis=1)
    return np.broadcast_to(plane, (n, n, n)).copy()


def element_diagonal(voxel):
    d = np.zeros(24)
    for grad in gradients(voxel):
        B = np.zeros((6, 24))
        for a, (dx, dy, dz) in enumerate(grad):
            B[0, 3*a] = dx; B[1, 3*a+1] = dy; B[2, 3*a+2] = dz
            B[3, 3*a+1] = dz; B[3, 3*a+2] = dy
            B[4, 3*a] = dz; B[4, 3*a+2] = dx
            B[5, 3*a] = dy; B[5, 3*a+1] = dx
        d += np.einsum('ij,ij->j', B, C0@B)*np.prod(voxel)/8
    return d.reshape(8, 3)


def build(rho, variant, preconditioner):
    grid = rho.shape
    voxel = tuple(1/np.array(grid[::-1]))
    field = torch.tensor(rho, device='cuda', dtype=torch.float64)
    pairs = [(i, j) for i in range(6) for j in range(i, 6)]
    c = torch.tensor([C0[i, j] for i, j in pairs], device='cuda')
    packed = (field.reshape(-1, 1)*c).contiguous()
    if variant == 'gauss':
        op = GaussFIOperator(packed, grid, voxel, assembly='atomic', block=128)
    else:
        op = HybridFIOperator(packed, torch.empty(0, device='cuda', dtype=torch.int32), grid, voxel, dense=True)
    diag = None
    if preconditioner != 'green':
        ed = element_diagonal(voxel)
        diag = torch.zeros((*grid, 3), device='cuda', dtype=torch.float64)
        for a, (dx, dy, dz) in enumerate(OFFSETS):
            diag += torch.roll(field, (int(dz), int(dy), int(dx)), (0, 1, 2))[..., None]*torch.tensor(ed[a], device='cuda')
        diag = diag.reshape(1, -1)
        op._physical_diag = diag
    if preconditioner != 'jacobi':
        op._compute_reference_medium = MethodType(lambda self, *a, **kw: C0[None].copy(), op)
        comp.cs.s.f.formula.enable_fi_reference(op, precision='fp64')
        comp.cs.a.native.attach(op, 'fi', 'dense')
        green = op.apply_precond
    # A common quotient-space projection fixes the three translational modes.
    # P D^-1/2 G D^-1/2 P is equivalent on compatible residuals.
    project = op.project_zero_mean
    if preconditioner == 'jacobi':
        inverse = diag.reciprocal()
        op.apply_precond = lambda r: project(inverse*r)
    elif preconditioner == 'green':
        op.apply_precond = lambda r: project(green(r))
    else:
        scale = diag.rsqrt()
        op.apply_precond = lambda r: project(scale*green(scale*r))
    return op


def solve(op, rhs, tol, cap):
    norm = torch.linalg.vector_norm(rhs)
    if float(norm) == 0:
        return torch.zeros_like(rhs), dict(iterations=0, restarts=0, relative_residual=0., converged=True)
    u = None
    count = restarts = 0
    rr = float('inf')
    while count < cap:
        u, it, _ = pcg(op, rhs, tol=tol, max_iter=cap-count, x0=u,
                      update_mode='compiled', convergence_check_stride=4,
                      verify_final_residual=False)
        count += it
        u = op.project_zero_mean(u)
        rr = float(torch.linalg.vector_norm(rhs-op.apply(u))/norm)
        if np.isfinite(rr) and rr <= tol:
            break
        if not np.isfinite(rr) or not it or restarts >= 2:
            break
        restarts += 1
    return u, dict(iterations=count, restarts=restarts, relative_residual=rr if np.isfinite(rr) else None,
                   converged=bool(np.isfinite(rr) and rr <= tol))


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def validate(args):
    setup()
    rng = np.random.default_rng(20261028)
    rho = density('cosine', 4, 4, 100)
    grid = rho.shape; N = rho.size; voxel = (1/4,)*3
    material = rho.reshape(-1, 1, 1)*C0
    eye = np.eye(3*N)
    K = np.stack([old.independent_cpu(material, grid, voxel, v[None])[0].ravel() for v in eye], axis=1)
    refK = np.stack([old.independent_cpu(np.broadcast_to(C0, material.shape), grid, voxel, v[None])[0].ravel() for v in eye], axis=1)
    G = np.linalg.pinv(refK, rcond=1e-13)
    diag = np.diag(K)
    v = rng.normal(size=(1, 3*N)); v -= v.reshape(N, 3).mean(0).reshape(1, 3).repeat(N, axis=0).reshape(1, -1)
    macro = np.array([1, 1, 0, 0, 0, np.sqrt(2)])
    load, _ = old.independent_cpu(material, grid, voxel, np.zeros_like(v), macro)
    exact = np.linalg.pinv(K, rcond=1e-13)@(-load.ravel())
    checks = []
    for variant, pre in itertools.product(('gauss', 'modal'), ('jacobi', 'green', 'green_jacobi')):
        op = build(rho, variant, pre)
        value = torch.tensor(v, device='cuda')
        if pre == 'jacobi': expected = v/diag
        elif pre == 'green': expected = (G@v.ravel())[None]
        else: expected = ((G@(v.ravel()/np.sqrt(diag)))/np.sqrt(diag))[None]
        expected = (expected.reshape(N, 3)-expected.reshape(N, 3).mean(0)).reshape(1, -1)
        rhs = op.compute_rhs(macro)
        u, solved = solve(op, rhs, 1e-11, 1000)
        row = dict(variant=variant, preconditioner=pre, Ku_error=relative(op.apply(value).cpu().numpy(), (K@v.ravel())[None]),
                   rhs_error=relative(rhs.cpu().numpy(), -load),
                   preconditioner_error=relative(op.apply_precond(value).cpu().numpy(), expected),
                   solution_error=relative(u.cpu().numpy().ravel(), exact), **solved)
        if pre != 'green': row['diagonal_error'] = relative(op._physical_diag.cpu().numpy().ravel(), diag)
        assert max(row[k] for k in ('Ku_error', 'rhs_error', 'preconditioner_error')) < 5e-12, row
        assert row['solution_error'] < 1e-8 and row['converged'], row
        checks.append(row)
        del op, value, rhs, u
        gc.collect(); torch.cuda.empty_cache()
    write(args.out/'validation.json', dict(passed=True, dense_dofs=3*N, checks=checks))
    print('VALIDATION PASS', flush=True)


def query_gpu():
    p = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True)
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], capture_output=True, text=True)
    return dict(gpu=p.stdout.strip(), processes=q.stdout.strip())


def measure(rho, variant, pre, tol, cap, micro=False):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    op = build(rho, variant, pre)
    torch.cuda.synchronize(); built = time.perf_counter()
    macro = np.array([1, 1, 0, 0, 0, np.sqrt(2)])
    rhs = op.compute_rhs(macro)
    torch.cuda.synchronize(); loaded = time.perf_counter()
    u, solved = solve(op, rhs, tol, cap)
    torch.cuda.synchronize(); done = time.perf_counter()
    stress = op.compute_avg_stress(u, macro).reshape(-1)
    torch.cuda.synchronize(); end = time.perf_counter()
    result = dict(variant=variant, preconditioner=pre, setup_s=built-start, rhs_s=loaded-built,
                  solve_cert_s=done-loaded, stress_s=end-done, total_s=end-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), baseline_allocated_bytes=baseline,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), stress=stress.tolist(), **solved)
    displacement = u.cpu().numpy()
    result['displacement_sha256'] = hashlib.sha256(displacement.tobytes()).hexdigest()
    if micro:
        result['micro'] = dict(Ku=old.event_time(op.apply, rhs), M_inv=old.event_time(op.apply_precond, rhs))
    del op, rhs, u
    gc.collect(); torch.cuda.empty_cache()
    return result, displacement


def worker(args):
    setup()
    rho = density(args.case, args.grid, args.sampling, args.contrast)
    pairs = list(itertools.product(('gauss', 'modal'), ('jacobi', 'green', 'green_jacobi')))
    warm = []
    for variant, pre in pairs:
        row, _ = measure(rho, variant, pre, args.tol, args.cap)
        warm.append(dict(variant=variant, preconditioner=pre, total_s=row['total_s']))
    reference = None
    rows = []
    for repeat in range(args.repeats):
        order = pairs[2*repeat:] + pairs[:2*repeat]
        if repeat % 2: order = list(reversed(order))
        for variant, pre in order:
            before = query_gpu()
            row, displacement = measure(rho, variant, pre, args.tol, args.cap, micro=(repeat == 0))
            row.update(repeat=repeat, gpu_before=before, gpu_after=query_gpu())
            if row['converged'] and reference is None:
                reference = displacement.copy()
            if reference is not None:
                row['relative_displacement_to_first_converged'] = relative(displacement, reference)
            rows.append(row)
            write(args.out/'records'/f'{args.tag}_{variant}_{pre}_r{repeat}.json', dict(
                case=args.case, grid=args.grid, sampling=args.sampling, contrast_parameter=args.contrast,
                actual_contrast=float(rho.max()/rho.min()), tolerance=args.tol,
                rho_sha256=hashlib.sha256(rho.tobytes()).hexdigest(), **row))
            print(args.tag, repeat, variant, pre, row['iterations'], f"{row['total_s']:.5f}s", row['relative_residual'], row['converged'], flush=True)
    sources = old.fingerprint(comp)
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.out/'cases'/f'{args.tag}.json', dict(case=args.case, grid=args.grid, sampling=args.sampling,
        contrast_parameter=args.contrast, actual_contrast=float(rho.max()/rho.min()), tolerance=args.tol,
        warmup=warm, rows=rows, sources=sources, environment=dict(
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, host=platform.node(), gpu=torch.cuda.get_device_name())))


def suite(args):
    jobs = []
    for case in ('laminate', 'cosine'):
        for n in (32, 64, 128, 256):
            jobs.append((case, n, n, 10000, 1e-5))
        for n in (32, 64, 128):
            jobs.append((case, n, 16, 10000, 1e-5))
        jobs.append((case, 64, 64, 10, 1e-5))
        jobs.append((case, 64, 64, 10000, 1e-8))
    for n in (64, 128):
        jobs.append(('sharp_cosine', n, n, 10000, 1e-5))
    commands = []
    for case, n, p, chi, tol in jobs:
        tag = f'{case}_n{n}_p{p}_c{chi}_t{tol:g}'
        cmd = [str(ROOT/'run'), 'smooth-contrast-study', 'worker', '--out', str(args.out),
               '--case', case, '--grid', str(n), '--sampling', str(p), '--contrast', str(chi),
               '--tol', str(tol), '--tag', tag, '--repeats', '3', '--cap', '4000']
        commands.append(dict(tag=tag, argv=cmd))
    write(args.out/'suite_manifest.json', dict(jobs=commands, protocol_version=2, warmup='full matched solve per configuration'))
    (args.out/'logs').mkdir(parents=True, exist_ok=True)
    for job in commands:
        print('START', job['tag'], flush=True)
        with (args.out/'logs'/f"{job['tag']}.log").open('x') as stream:
            completed = subprocess.run(job['argv'], stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Failed {job['tag']}, see log")
        print('DONE', job['tag'], flush=True)
    write(args.out/'suite_complete.json', dict(completed_cases=len(commands), measured_solves=len(commands)*18))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'worker', 'suite'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=['laminate', 'cosine', 'sharp_cosine'], default='cosine')
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--sampling', type=int, default=64)
    parser.add_argument('--contrast', type=float, default=1e4)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--cap', type=int, default=4000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tag', default='pilot')
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(ROOT/'runs')
    dict(validate=validate, worker=worker, suite=suite)[args.mode](args)


if __name__ == '__main__':
    main()
