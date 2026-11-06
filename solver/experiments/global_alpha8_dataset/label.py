"""Exhaustive 101-point, time-to-true-residual alpha labels; resumable per solve."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
from statistics import mean, stdev
import subprocess
import time
import numpy as np
from tqdm import tqdm
from common import (SEED, digest, encode_torch, filters, lock, object_hash, output_path,
                    read, save_json, save_npz, select)
from dataset import load_field, validate_manifest

TOLERANCE = 1e-8
ALPHAS = tuple(j/100 for j in range(101))


def record_path(out, sample_id, repeat, j):
    return out/'labels/records'/sample_id/f'r{repeat:03d}_a{j:03d}.json'


def collect(out, sample_id, repeats, protocol_id):
    rows = []
    for repeat in range(repeats):
        for j, alpha in enumerate(ALPHAS):
            path = record_path(out,sample_id,repeat,j)
            if not path.exists():
                continue
            r = read(path)
            assert (r['sample_id'],r['repeat'],r['alpha_index'],r['alpha'],r['protocol_id']) == (
                sample_id,repeat,j,alpha,protocol_id)
            assert r['time_s']>0 and r['iterations']>=0
            assert abs(r['time_s']-sum(r[k] for k in ('build_s','rhs_s','input_s','scale_s','solve_cert_s')))<1e-9
            assert bool(r['converged']) == (r['relative_residual'] is not None and r['relative_residual']<=TOLERANCE)
            rows.append(r)
    return rows


def summarize(rows, repeats):
    expected = {(repeat,j) for repeat in range(repeats) for j in range(101)}
    keys = [(r['repeat'],r['alpha_index']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate solve records')
    if set(keys) != expected:
        return dict(status='incomplete', expected=101*repeats, present=len(rows), alpha_star=None)
    curve = []
    for j, alpha in enumerate(ALPHAS):
        group = [r for r in rows if r['alpha_index']==j]
        valid = all(r['converged'] and r['relative_residual'] is not None
                    and 0<=r['relative_residual']<=TOLERANCE for r in group)
        times = [r['time_s'] for r in group]
        curve.append(dict(alpha_index=j, alpha=alpha, eligible=valid,
                          mean_time_s=mean(times), std_time_s=stdev(times),
                          mean_iterations=mean(r['iterations'] for r in group),
                          residuals=[r['relative_residual'] for r in group],
                          failed_repeats=sum(not r['converged'] for r in group)))
    eligible = sorted((r for r in curve if r['eligible']), key=lambda r:(r['mean_time_s'],r['alpha_index']))
    if not eligible:
        return dict(status='no_converged_candidate', alpha_star=None, curve=curve, repeats=repeats)
    best = eligible[0]
    runner = eligible[1] if len(eligible)>1 else None
    near = [r['alpha'] for r in curve if r['eligible'] and r['mean_time_s']<=1.05*best['mean_time_s']]
    noisy = bool(runner and runner['mean_time_s']-best['mean_time_s']<=2*math.sqrt(
        (runner['std_time_s']**2+best['std_time_s']**2)/repeats))
    return dict(status='complete', alpha_star=best['alpha'], alpha_index=best['alpha_index'],
                best_mean_time_s=best['mean_time_s'], near_optimal_alphas_5pct=near,
                needs_more_repeats_heuristic=noisy, repeats=repeats, formal_solves=len(rows),
                objective='arithmetic mean time to certified true residual <= 1e-8', curve=curve)


def assert_gpu_exclusive(allowed_pids=()):
    # Refuse noisy concurrent compute workloads; never kill another process.
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    allowed = {os.getpid(), *allowed_pids}
    other = [int(v.strip()) for v in text.splitlines() if v.strip().isdigit() and int(v.strip()) not in allowed]
    if other:
        raise RuntimeError('Other GPU compute processes are active: '+str(other))


class Runner:
    def __init__(self, out, m, execution=None, allowed_pids=()):
        assert_gpu_exclusive(allowed_pids)
        import torch
        from project_paths import ROOT
        self.torch = torch
        self.root = ROOT
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        spec = importlib.util.spec_from_file_location('alpha_scan_base',ROOT/'experiments/adaptive_green_scaling/study.py')
        self.a = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.a)
        self.a.s.setup()
        np.testing.assert_allclose(self.a.MACRO,m['macro_strain'],rtol=0,atol=0)
        np.testing.assert_allclose(self.a.s.C0,m['C0'],rtol=0,atol=0)
        files = self.a.s.old.fingerprint(self.a.s.comp)
        for p in (Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('dataset.py'),
                  Path(__file__).with_name('parallel_scan.py'),
                  Path(self.a.__file__),Path(self.a.s.__file__),ROOT/'project_paths.py'):
            files[str(p.resolve())] = digest(p)
        hardware = dict(host=platform.node(), machine=platform.machine(), cpu=platform.processor(),
                        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                        python=platform.python_version(), gpu_driver_uuid=subprocess.check_output([
                            'nvidia-smi','--query-gpu=driver_version,uuid','--format=csv,noheader'],text=True).strip())
        self.protocol = dict(version='alpha-grid-time-v1', manifest_sha256=digest(out/'manifest.json'),
                             alpha_grid=list(ALPHAS), minimum_repeats=3, tolerance=TOLERANCE,
                             cap=self.a.CAP, check_stride=4, zero_initial_guess=True,
                             warmup='one unrecorded full solve for every alpha at each process/sample entry',
                             candidate_order='independently shuffled each repeat; deterministic by sample and repeat',
                             timing='fresh material/operator/Green, rhs, input8 preprocessing, scaling, PCG, true residual certification',
                             omitted='offline field generation, startup/JIT, GC, warmups, IO, stress, alpha search, NN forward',
                             omitted_network_reason='one fixed network forward is alpha-independent for a given field; measured later in deployment evaluation',
                             alpha_zero='existing exact Green endpoint skips diagonal/scaling; deployment must retain this branch when predicting zero',
                             reset='fresh operator and zero initial guess every solve; GC and empty CUDA cache outside timer',
                             hardware=hardware, source_hashes=files)
        self.protocol['execution'] = execution or dict(workers=1, timing_scope='isolated_single_task')
        if execution and execution['workers'] > 1:
            self.protocol['version'] = 'alpha-grid-time-concurrent-v2'
            self.protocol['warmup'] = 'each worker warms alpha 0 and 0.5 once; then all 101 candidates per field through the shared pool'
            self.protocol['objective'] = 'mean wall time to true residual <=1e-8 under shared-GPU contention; not isolated latency'
            if execution.get('backend') == 'threads':
                self.protocol['warmup'] = 'serial alpha 0 and 0.5 before threads; then all 101 candidates per field on worker CUDA streams from the PyTorch pool'
                self.protocol['reset'] = 'fresh operator and zero initial guess; stream synchronization; GC/cache cleanup only after a completed warmup/round'
        path = out/'labels/protocol.json'
        if path.exists():
            if read(path)!=self.protocol:
                raise RuntimeError('Protocol, source or environment changed; refusing to mix timings')
        else:
            save_json(path,self.protocol)
            for name in files:
                src = Path(name); dst = out/'labels/evidence/source'/src.relative_to(ROOT)
                dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        self.protocol_id = object_hash(self.protocol)
        assert_gpu_exclusive(allowed_pids)

    def measure(self, rho, alpha, concurrent=False):
        t = self.torch; a = self.a
        def sync():
            if concurrent:
                t.cuda.current_stream().synchronize()
            else:
                t.cuda.synchronize()
            return time.perf_counter()
        if not concurrent:
            gc.collect(); t.cuda.empty_cache()
        sync()
        start = time.perf_counter()
        op = a.build(rho); built = sync()
        rhs = op.compute_rhs(a.MACRO)
        rhs_norm = float(t.linalg.vector_norm(rhs))
        if not math.isfinite(rhs_norm) or rhs_norm==0:
            raise ValueError('Invalid/zero RHS: no meaningful alpha time label; do not replace the load silently')
        loaded = sync()
        with t.inference_mode():
            x = encode_torch(op._material_field); encoded = sync()
            a.set_alpha(op,alpha); scaled = sync()
            u, info = a.certified(op,rhs,TOLERANCE,x0=None); done = sync()
        relative = info['relative_residual']
        if not math.isfinite(relative):
            info['relative_residual'] = None
            info['converged'] = False
        result = dict(time_s=done-start, build_s=built-start, rhs_s=loaded-built, input_s=encoded-loaded,
                      scale_s=scaled-encoded, solve_cert_s=done-scaled, rhs_norm=rhs_norm, **info)
        del op,rhs,x,u
        if not concurrent:
            gc.collect(); t.cuda.empty_cache()
        return result


def verify_existing(out, m):
    for name, expected in m['source_hashes'].items():
        if digest(Path(__file__).with_name(name))!=expected:
            raise RuntimeError('Dataset source changed: '+name)
    path = out/'labels/protocol.json'
    if path.exists():
        p = read(path)
        assert p['manifest_sha256']==digest(out/'manifest.json')
        for name, expected in p['source_hashes'].items():
            if digest(name)!=expected:
                raise RuntimeError('Solver or labeling source changed: '+name)
        return p, object_hash(p)
    return None,None


def summary_file(out, sid, repeats):
    return out/'labels/summaries'/f'{sid}_r{repeats}.json'


def scan(out,m,rows,args):
    from parallel_scan import scan as run_scan
    return run_scan(out,m,rows,args)


def export(out,m,rows,args,pid):
    usable=[]; missing=[]
    for row in rows:
        result=summarize(collect(out,row['sample_id'],args.repeats,pid),args.repeats)
        if result['status']!='complete':
            missing.append(row['sample_id']); continue
        _,x,_=load_field(out,m,row)
        usable.append((row,x,result))
    if missing and not args.allow_partial:
        raise RuntimeError(f'{len(missing)} missing/invalid labels; export refused. --allow-partial is for explicit pilot use.')
    if not usable:
        raise RuntimeError('No complete labels to export')
    for split in ('train','validation','test'):
        data=[v for v in usable if v[0]['split']==split]
        if not data:continue
        path=out/'exports'/f'{split}_r{args.repeats}_{len(data)}.npz'
        arrays=dict(x=np.stack([v[1] for v in data]), alpha_star=np.array([v[2]['alpha_star'] for v in data],np.float32)[:,None],
                    sample_id=np.array([v[0]['sample_id'] for v in data]),geometry_id=np.array([v[0]['geometry_id'] for v in data]),
                    family=np.array([v[0]['family'] for v in data]),contrast=np.array([v[0]['contrast'] for v in data]),
                    time_curve_s=np.array([[r['mean_time_s'] for r in v[2]['curve']] for v in data]),
                    eligible=np.array([[r['eligible'] for r in v[2]['curve']] for v in data]),
                    noisy_label=np.array([v[2]['needs_more_repeats_heuristic'] for v in data]))
        if path.exists():
            with np.load(path,allow_pickle=False) as prev:
                assert set(prev.files)==set(arrays) and all(np.array_equal(prev[k],a) for k,a in arrays.items())
        else:save_npz(path,**arrays)
        save_json(path.with_suffix('.json'),dict(protocol_id=pid,manifest_sha256=digest(out/'manifest.json'),
                  execution=read(out/'labels/protocol.json').get('execution'),
                  samples=len(data),split=split,repeats=args.repeats,partial=bool(missing),missing=missing,archive_sha256=digest(path)),replace=True)
        print('EXPORTED',path,arrays['x'].shape,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('scan','status','export'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3,help='At least 3; higher values add full 101-point rounds on resume')
    parser.add_argument('--workers',type=int,default=64,help='Concurrent GPU evaluation tasks (default: 64); >1 measures contended wall time')
    parser.add_argument('--backend',choices=('threads','processes'),default='threads',help='threads share one CUDA context and use one stream per worker')
    parser.add_argument('--max-new-records',type=int,help='Stop safely after this many new solves; incomplete samples get no label')
    parser.add_argument('--allow-partial',action='store_true',help='Explicitly export only completed pilot samples')
    filters(parser);parser.set_defaults(split='train')
    args=parser.parse_args()
    if args.workers<1 or args.repeats<3 or (args.max_new_records is not None and args.max_new_records<=0):
        parser.error('workers/max-new-records must be positive and repeats >=3')
    out=output_path(args.out)
    with lock(out):
        m=read(out/'manifest.json');validate_manifest(m);rows=select(m,args)
        protocol,pid=verify_existing(out,m)
        if args.mode=='scan':scan(out,m,rows,args)
        elif args.mode=='export':
            if not protocol:raise RuntimeError('No labeling protocol exists yet')
            export(out,m,rows,args,pid)
        else:
            statuses={}
            for row in rows:
                r=summarize(collect(out,row['sample_id'],args.repeats,pid) if protocol else [],args.repeats)
                statuses[r['status']]=statuses.get(r['status'],0)+1
            print(json.dumps(dict(selected=len(rows),repeats=args.repeats,statuses=statuses)),flush=True)


if __name__=='__main__':main()
