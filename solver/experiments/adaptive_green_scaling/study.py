"""Frozen pilot protocol: morphology-guided symmetric Green scaling."""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import MethodType
import numpy as np
import torch
from project_paths import ROOT

SPEC = importlib.util.spec_from_file_location('smooth_base', ROOT/'experiments/smooth_contrast/study.py')
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)
MACRO = np.array([1., 1., 0., 0., 0., np.sqrt(2)])
ALPHAS = (0., .25, .5, .75, 1.)
METHODS = tuple(f'alpha_{a:g}' for a in ALPHAS)+('adaptive',)
CAP = 4000


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def field(case, n):
    chi = 10000.
    if case in ('cosine', 'laminate', 'sharp_cosine'):
        return s.density(case, n, n, chi)
    if case.endswith('_p16'):
        return s.density(case[:-4], n, 16, chi)
    z, y, x = np.meshgrid(*(np.arange(n)/n for _ in range(3)), indexing='ij')
    if case == 'smooth3d':
        q = (np.sin(2*np.pi*x)+np.cos(2*np.pi*y)+np.sin(2*np.pi*z)+3)/6
    elif case == 'mixed3d':
        smooth = (np.sin(2*np.pi*x)+np.cos(2*np.pi*y)+np.sin(2*np.pi*z)+3)/6
        phase = (np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z) >= .15)
        q = .65*phase+.35*smooth
    elif case in ('transition_wide', 'transition_thin'):
        width = .08 if case == 'transition_wide' else .01
        level = np.cos(2*np.pi*x)+np.cos(2*np.pi*y)+np.cos(2*np.pi*z)-.5
        q = .5*(1+np.tanh(level/(2*np.pi*width)))
    else:
        raise ValueError(case)
    q = (q-q.min())/(q.max()-q.min())
    return np.exp(np.log(chi)*q)


def build(rho):
    grid = rho.shape
    voxel = tuple(1/np.array(grid[::-1]))
    material = torch.tensor(rho, device='cuda', dtype=torch.float64)
    pairs = [(i, j) for i in range(6) for j in range(i, 6)]
    c = torch.tensor([s.C0[i, j] for i, j in pairs], device='cuda')
    packed = (material.reshape(-1, 1)*c).contiguous()
    op = s.HybridFIOperator(packed, torch.empty(0, device='cuda', dtype=torch.int32), grid, voxel, dense=True)
    op._compute_reference_medium = MethodType(lambda self, *a, **kw: s.C0[None].copy(), op)
    s.comp.cs.s.f.formula.enable_fi_reference(op, precision='fp64')
    s.comp.cs.a.native.attach(op, 'fi', 'dense')
    op._unscaled_green = op.apply_precond
    op._material_field = material
    op._voxel = voxel
    op._diagonal = None
    op._log_diagonal = None
    return op


def diagonal(op):
    if op._diagonal is None:
        ed = s.element_diagonal(op._voxel)
        rho = op._material_field
        d = torch.zeros((*rho.shape, 3), device='cuda', dtype=torch.float64)
        for a, (dx, dy, dz) in enumerate(s.OFFSETS):
            d += torch.roll(rho, (int(dz), int(dy), int(dx)), (0, 1, 2))[..., None]*torch.tensor(ed[a], device='cuda')
        op._diagonal = d.reshape(1, -1)
    return op._diagonal


def set_alpha(op, alpha):
    project = op.project_zero_mean
    green = op._unscaled_green
    if alpha == 0:
        op.apply_precond = lambda r: project(green(project(r)))
    else:
        if op._log_diagonal is None:
            q = diagonal(op).log()
            op._log_diagonal = q-q.mean()
        scale = torch.exp(-.5*alpha*op._log_diagonal)
        op.apply_precond = lambda r: project(scale*green(scale*project(r)))


def morphology(op):
    """Fixed, GPU-resident descriptors; no labels, fitted values, or sweep outcomes."""
    q = op._material_field.log()
    rows = []
    for axis in range(3):
        delta = (q-torch.roll(q, 1, axis)).abs()
        values = torch.stack(((delta > 1e-10).double().mean(),
                              (delta >= np.log(2)).double().mean(), delta.square().mean(), delta.max()))
        rows.append(values)
    values = torch.stack(rows).cpu().numpy()
    active = values[:, 0] > 1e-10
    coverage = float(values[active, 0].mean()) if active.any() else 0.
    jump_fraction = float(values[active, 1].sum()/values[active, 0].sum()) if active.any() else 0.
    if not active.any() or jump_fraction >= .8:
        route, alpha = 'jump_dominated', 0.
    elif coverage >= .75 and jump_fraction <= .1:
        route, alpha = 'resolved_variation', 1.
    else:
        route, alpha = 'probe', None
    return dict(coverage=coverage, jump_fraction_of_changing_edges=jump_fraction,
                axis_descriptors=values.tolist(), route=route, prior_alpha=alpha)


def certified(op, rhs, tol, x0=None):
    norm = torch.linalg.vector_norm(rhs)
    u = x0
    total = restarts = 0
    while total < CAP:
        u, it, _ = s.pcg(op, rhs, tol=tol, max_iter=CAP-total, x0=u,
                        update_mode='compiled', convergence_check_stride=4, verify_final_residual=False)
        total += it
        u = op.project_zero_mean(u)
        relative = float(torch.linalg.vector_norm(rhs-op.apply(u))/norm)
        if np.isfinite(relative) and relative <= tol:
            break
        if not np.isfinite(relative) or not it or restarts >= 2:
            break
        restarts += 1
    return u, dict(iterations=total, restarts=restarts, relative_residual=relative,
                   converged=bool(np.isfinite(relative) and relative <= tol))


def choose(op, rhs, features):
    if features['prior_alpha'] is not None:
        alpha = features['prior_alpha']
        set_alpha(op, alpha)
        return alpha, None, []
    invsqrt = diagonal(op).rsqrt()
    denom = torch.linalg.vector_norm(invsqrt*rhs)
    norm = torch.linalg.vector_norm(rhs)
    candidates = []
    best_u = None
    best_score = float('inf')
    alpha = None
    for a in (0., .5, 1.):
        set_alpha(op, a)
        u, it, _ = s.pcg(op, rhs, tol=1e-5, max_iter=4, update_mode='compiled',
                        convergence_check_stride=4, verify_final_residual=False)
        u = op.project_zero_mean(u)
        r = rhs-op.apply(u)
        score = float(torch.linalg.vector_norm(invsqrt*r)/denom)
        candidates.append(dict(alpha=a, iterations=it, common_scaled_residual=score,
                               true_residual=float(torch.linalg.vector_norm(r)/norm)))
        if score < best_score:
            best_score, alpha, best_u = score, a, u.clone()
    set_alpha(op, alpha)
    return alpha, best_u, candidates


def measure(rho, method, tol=1e-5):
    gc.collect(); torch.cuda.empty_cache()
    sync(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    op = build(rho)
    built = sync()
    rhs = op.compute_rhs(MACRO)
    loaded = sync()
    features = morphology(op) if method == 'adaptive' else None
    featured = sync()
    if method == 'adaptive':
        alpha, initial, probes = choose(op, rhs, features)
    else:
        alpha = float(method[6:]); initial = None; probes = []
        set_alpha(op, alpha)
    chosen = sync()
    u, solved = certified(op, rhs, tol, initial)
    done = sync()
    stress = op.compute_avg_stress(u, MACRO).reshape(-1)
    end = sync()
    row = dict(method=method, alpha=alpha, features=features, probes=probes,
               probe_iterations=sum(p['iterations'] for p in probes),
               build_s=built-start, rhs_s=loaded-built, morphology_s=featured-loaded,
               select_s=chosen-featured, solve_cert_s=done-chosen, stress_s=end-done,
               total_s=end-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               stress=stress.tolist(), **solved)
    result = u.cpu().numpy().copy()
    del op, rhs, u, stress, initial
    gc.collect(); torch.cuda.empty_cache()
    return row, result


def validate(out):
    rng = np.random.default_rng(20261028)
    rho = s.density('cosine', 4, 4, 100)
    N = rho.size
    mat = rho.reshape(-1, 1, 1)*s.C0
    eye = np.eye(3*N)
    K = np.stack([s.old.independent_cpu(mat, rho.shape, (.25,)*3, v[None])[0].ravel() for v in eye], axis=1)
    K0 = np.stack([s.old.independent_cpu(np.broadcast_to(s.C0, mat.shape), rho.shape, (.25,)*3, v[None])[0].ravel() for v in eye], axis=1)
    G = np.linalg.pinv(K0, rcond=1e-13)
    T = np.tile(np.eye(3), (N, 1))
    P = np.eye(3*N)-T@T.T/N
    Q = np.linalg.eigh(P)[1][:, 3:]
    diag = np.diag(K)
    ld = np.log(diag); ld -= ld.mean()
    op = build(rho)
    v = P@rng.normal(size=3*N)
    checks = []
    b = op.compute_rhs(MACRO)
    exact = np.linalg.pinv(K, rcond=1e-13)@b.cpu().numpy().ravel()
    for a in ALPHAS:
        scale = np.exp(-.5*a*ld)
        B = P@(scale[:, None]*G*scale[None, :])@P
        set_alpha(op, a)
        observed = op.apply_precond(torch.tensor(v[None], device='cuda')).cpu().numpy().ravel()
        u, info = certified(op, b, 1e-11)
        row = dict(alpha=a, action_error=s.relative(observed, B@v),
                   symmetry_error=s.relative(B, B.T), min_eigenvalue=float(np.linalg.eigvalsh(Q.T@B@Q).min()),
                   solution_error=s.relative(u.cpu().numpy().ravel(), exact), **info)
        assert row['action_error'] < 1e-11 and row['symmetry_error'] < 1e-11
        assert row['min_eigenvalue'] > 0 and row['solution_error'] < 1e-8 and row['converged']
        checks.append(row)
    endpoint = []
    for a, pre in ((0., 'green'), (1., 'green_jacobi')):
        other = s.build(rho, 'modal', pre)
        set_alpha(op, a)
        t = torch.tensor(v[None], device='cuda')
        left = op.apply_precond(t).cpu().numpy()
        right = other.apply_precond(t).cpu().numpy()
        ratio = float((left*right).sum()/(right*right).sum())
        error = s.relative(left, ratio*right)
        assert error < 1e-11
        endpoint.append(dict(alpha=a, original=pre, scalar=ratio, relative_error=error))
    save(out/'validation.json', dict(passed=True, dofs=3*N, checks=checks, endpoints=endpoint))
    print('DENSE VALIDATION PASS', flush=True)


def snapshot(out):
    files = s.old.fingerprint(s.comp)
    for path in (Path(__file__), Path(s.__file__), Path(inspect.getfile(s.pcg)), ROOT/'run.py', ROOT/'project_paths.py'):
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in files.items():
        path = Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        destination = out/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return files


def worker(args):
    rho = field(args.case, args.grid)
    tag = f'{args.case}_n{args.grid}'
    gpu = s.query_gpu()
    assert not gpu['processes'], gpu
    refs = build(rho)
    set_alpha(refs, 1. if args.case in ('cosine', 'laminate', 'smooth3d') else 0.)
    rhs = refs.compute_rhs(MACRO)
    exact, ref_info = certified(refs, rhs, 1e-10)
    assert ref_info['converged'], ref_info
    energy = float((exact*refs.apply(exact)).sum())
    unorm = float(torch.linalg.vector_norm(exact))
    ref_stress = refs.compute_avg_stress(exact, MACRO).reshape(-1)
    exact_cpu = exact.cpu().numpy().copy()
    ref_stress_cpu = ref_stress.copy() if isinstance(ref_stress, np.ndarray) else ref_stress.cpu().numpy()
    del refs, rhs, exact, ref_stress
    gc.collect(); torch.cuda.empty_cache()
    warm = []
    for method in METHODS:
        r, _ = measure(rho, method)
        warm.append(r)
    rows = []
    for repeat in range(args.repeats):
        order = METHODS[repeat:]+METHODS[:repeat]
        if repeat % 2: order = tuple(reversed(order))
        for method in order:
            row, displacement = measure(rho, method)
            row.update(repeat=repeat, case=args.case, n=args.grid, tolerance=1e-5)
            # Accuracy verification is outside the performance timing.
            check = build(rho)
            delta = torch.tensor(displacement-exact_cpu, device='cuda')
            row['energy_error'] = np.sqrt(max(float((delta*check.apply(delta)).sum()), 0)/energy)
            row['displacement_error'] = float(np.linalg.norm(displacement-exact_cpu)/unorm)
            row['stress_error'] = s.relative(np.array(row['stress']), ref_stress_cpu)
            assert row['converged'] and np.isfinite(row['energy_error']), row
            del check, delta
            rows.append(row)
            save(args.out/'records'/f'{tag}_{method}_r{repeat}.json', row)
            print(tag, repeat, method, 'a=', row['alpha'], 'it=',row['iterations'],
                  'probe=',row['probe_iterations'], f"{row['total_s']:.6f}s", 'eK=', row['energy_error'], flush=True)
    sources = snapshot(args.out/'snapshots'/tag)
    save(args.out/'cases'/f'{tag}.json', dict(case=args.case, n=args.grid,
         rho_sha256=hashlib.sha256(rho.tobytes()).hexdigest(), actual_contrast=float(rho.max()/rho.min()),
         reference=ref_info, reference_energy=energy, warmup=warm, rows=rows, sources=sources,
         environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                          gpu_before=gpu, gpu_after=s.query_gpu())))


def suite(args):
    import subprocess
    families = ('cosine', 'laminate', 'sharp_cosine', 'cosine_p16', 'laminate_p16',
                'smooth3d', 'mixed3d', 'transition_wide', 'transition_thin')
    jobs = [(case, n) for n in (64, 128) for case in families]
    commands = []
    for case, n in jobs:
        command = [str(ROOT/'run'), 'adaptive-green-study', 'worker', '--out', str(args.out),
                   '--case', case, '--grid', str(n), '--repeats', str(args.repeats)]
        commands.append(dict(case=case, n=n, argv=command))
    save(args.out/'manifest.json', dict(jobs=commands, frozen_rule=dict(
        jump_threshold=np.log(2), constant_edge_threshold=1e-10,
        jump_fraction_for_green=.8, coverage_for_gj=.75, jump_fraction_for_gj=.1,
        probe_alphas=[0,.5,1], probe_steps=4, score='common D^-1/2 true residual',
        pilot_reuse='winner displacement reused with restarted fixed-preconditioner PCG'),
        fixed_alphas=ALPHAS, method_version=1, tolerance=1e-5, reference_tolerance=1e-10,
        scope='Positive scalar stiffness, full-integration periodic HEX8, fixed Modal FI'))
    for item in commands:
        log = args.out/'logs'/f"{item['case']}_n{item['n']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as f:
            result = subprocess.run(item['argv'], stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print('FAILED', log, result.returncode, flush=True)
            raise SystemExit(result.returncode)
        print('COMPLETE', item['case'], item['n'], flush=True)
    save(args.out/'complete.json', dict(cases=len(commands), methods=len(METHODS), repeats=args.repeats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validate','worker','suite'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', default='cosine')
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(ROOT/'runs')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'suite':
        suite(args)
    else:
        s.setup()
        if args.mode == 'validate': validate(args.out)
        else: worker(args)


if __name__ == '__main__':
    main()
