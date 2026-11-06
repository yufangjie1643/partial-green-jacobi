"""FI-only, fresh-process experiment suite for revised sections 4 and 5.

Run through run_jobs.py; every measurement follows a complete six-load warmup.
Frozen solver dependencies are imported without modifying their sources.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("VOXEL_RUN_DIR", str(__import__("project_paths").ROOT / "runs/solver_comparison")))
RUN_DIR.mkdir(parents=True, exist_ok=True)
CHAIN = __import__('project_paths').source_path('solver_chain')
OPERATOR = __import__('project_paths').source_path('hybrid_operator')
os.environ['FEAD_TOL'] = '1e-8'
arch = torch.cuda.get_device_capability()
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{arch[0]}.{arch[1]}')
sys.path.insert(0, str(CHAIN))
import chain_study as cs
import closed_study as closed

VARIANTS = ('paper', 'modal', 'analytic', 'fft', 'fusion', 'chain', 'dense', 'hybrid', 'latest')


def configure(variant):
    mf = None
    if variant in ('paper', 'modal'):
        b = cs.s.load('fi', configure=False)
        cs.s.f.formula.pipeline.configure(b, cs.s.f.prior.modal, 'fi', variant)
    else:
        b = cs.s.load('fi')
        if variant == 'chain':
            closed.install(b, 'fi')
        if variant in ('dense', 'hybrid', 'latest'):
            sys.path.insert(0, str(OPERATOR))
            import operator_adapter as oa
            mf = oa.prior.install(b, cs, dense=variant == 'dense')
        if variant != 'analytic':
            cs.a.configure(b, 'fi', 'baseline' if variant == 'fft' else 'combined')
            cs.a.native.extension()
        if variant == 'latest':
            oa.install(b, 'split_delta', 64)
    b.TOL = 1e-8
    return b, mf


def inputs(b, mf, case, grid):
    if case in ('sphere', 'grosmont'):
        from legacy_inputs import prepare_inputs
        phase, data, geometry, material, digest, audit = prepare_inputs(HERE/'inputs'/f'{case}_{grid}.npy')
        metadata = audit['source_metadata']
        def adapt():
            if mf is None:
                return data, material
            table = {label: dict(kind='isotropic', E=m['E'], nu=m['nu'])
                     for label, m in zip((0, 255), metadata['phase_materials'])}
            return dict(grid_shape=data['grid_shape'], domain_size=data['domain_size'], phase_id=phase[None]), table
        return adapt, digest, dict(geometry=geometry, material=material, input_metadata=metadata)
    data, geometry, material, digest = b.inputs(grid, 'strong')
    if case.startswith('fraction'):
        assert mf is not None
        fraction = int(case.removeprefix('fraction'))/100
        # Controlled local-material volume: planar layers with a fixed global
        # orientation. Local-frame declaration deliberately retains one C21 row
        # per local voxel, so active fraction and pool size can be studied.
        shape = (grid, grid, grid)
        phase = np.zeros(shape, dtype=np.uint8)
        layers = round(fraction*grid)
        phase[:layers] = 1
        _, table, _ = mf.from_legacy_yarn(data, material)
        rotation = np.array([[1., -1., 0.], [1., 1., 0.], [0., 0., np.sqrt(2.)]])/np.sqrt(2.)
        orientation = np.broadcast_to(rotation, shape+(3,3)).copy()
        synthetic = dict(grid_shape=shape, domain_size=data['domain_size'], phase_id=phase[None], orientation=orientation[None])
        digest = hashlib.sha256(phase.tobytes()+rotation.tobytes()+json.dumps(material, sort_keys=True).encode()).hexdigest()
        def adapt():
            return synthetic, table
        return adapt, digest, dict(geometry='planar local-anisotropic layers normal to z', material=material,
                                  active_fraction=float(phase.mean()), layers=layers, orientation=rotation.tolist())
    def adapt():
        if mf is None:
            return data, material
        phase, table, orientation = mf.from_legacy_yarn(data, material)
        return dict(grid_shape=data['grid_shape'], domain_size=data['domain_size'], phase_id=phase, orientation=orientation), table
    return adapt, digest, dict(geometry=geometry, material=material, active_fraction=float(np.mean(data['yarn_id'] >= 0)))


def microbenchmark(op):
    torch.manual_seed(193)
    u = torch.randn((1, op.mesh.n_dofs), device='cuda', dtype=torch.float64)
    def measure(fn):
        for _ in range(5):
            value = fn(u)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                value = fn(u)
            end.record(); end.synchronize()
            samples.append(start.elapsed_time(end)/20)
        return dict(samples_ms=samples, median_ms=statistics.median(samples))
    return dict(Ku=measure(op.apply), preconditioner=measure(op.apply_precond),
                scope='resident CUDA-event batches outside full-workflow time and memory window')


def sources():
    paths = set(HERE.glob('*.py')) | {HERE/'plan.json'}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).resolve().is_relative_to(__import__('project_paths').ROOT):
            p = Path(name)
            if p.is_file() and p.suffix == '.py':
                paths.add(p)
                paths.update(p.parent.glob('*.cu'))
                paths.update(p.parent.glob('*.cuh'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--case', choices=('woven', 'sphere', 'grosmont', 'fraction50', 'fraction90'), default='woven')
    parser.add_argument('--grid', type=int, default=192)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    name = f'{args.case}_{args.grid}_{args.variant}_r{args.repeat}{args.tag}'
    output = RUN_DIR/'results'/f'{name}.json'
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    b, mf = configure(args.variant)
    adapt, digest, meta = inputs(b, mf, args.case, args.grid)
    def task():
        start = time.perf_counter()
        data, material = adapt()
        adaptation = time.perf_counter()-start
        result, _ = b.workload(data, material, 'direct', 'mixed')
        return result, time.perf_counter()-start, adaptation
    b.gpu_guard()
    task()
    b.gpu_guard()
    print('MEASUREMENT_BEGIN', flush=True)
    result, elapsed, adaptation = task()
    print('MEASUREMENT_END', flush=True)
    residual = max(v for call in result['calls'] for v in call['true_residuals'])
    assert len(result['calls']) == 6 and np.isfinite(residual) and residual <= 1e-8
    assert np.isfinite(result['CH']).all()
    data, material = adapt()
    b.cleanup()
    op = b.build(data, material, 'direct', 'mixed', b.Stages())
    micro = microbenchmark(op)
    operator = getattr(op, '_operator_metadata', None)
    del op
    gc.collect(); b.cleanup()
    report = dict(complete=True, case=args.case, grid=args.grid, variant=args.variant, repeat=args.repeat,
                  input_hash=digest, input=meta, measurement=result,
                  workflow_seconds=elapsed, adaptation_seconds=adaptation,
                  max_true_residual=residual, microbenchmark=micro,
                  storage=getattr(b, '_storage', None), operator=operator,
                  reference_design=getattr(b, '_reference_design', None),
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                                   python=sys.version, threads=torch.get_num_threads()), sources=sources(),
                  protocol='FI eight-point integration; six unit engineering strains; FP64 physics/FP32 FFT; true residual <=1e-8; same-grid full warmup; fresh process; geometry/JIT/warmup/I/O excluded; setup, loading, solve, certification and stress recovery included.')
    b.gpu_guard()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('PASS', name, result['seconds'], elapsed, residual, flush=True)


if __name__ == '__main__':
    main()
