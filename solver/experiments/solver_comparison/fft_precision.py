"""Matched final-solver FFT precision experiment, launched through ./run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from project_paths import ROOT, source_path


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def worker(args):
    import numpy as np
    import torch
    study = load_file("paired_solver_study", ROOT / "experiments/solver_comparison/study.py")
    if args.precision != "original":
        sys.path.insert(0, str(source_path("spectral")))
        import precision_adapter
        precision_adapter.install(study.cs, args.precision)
    b, mf = study.configure("latest")
    adapt, digest, meta = study.inputs(b, mf, "woven", args.grid)
    b.gpu_guard()

    def task():
        start = time.perf_counter()
        data, material = adapt()
        result, _ = b.workload(data, material, "direct", "mixed")
        return result, time.perf_counter() - start

    task()
    b.gpu_guard()
    print("MEASUREMENT_BEGIN", flush=True)
    result, elapsed = task()
    print("MEASUREMENT_END", flush=True)
    residual = max(v for row in result["calls"] for v in row["true_residuals"])
    assert len(result["calls"]) == 6 and np.isfinite(residual) and residual <= 1e-8
    data, material = adapt()
    b.cleanup()
    op = b.build(data, material, "direct", "mixed", b.Stages())
    cache = op._contest_coefficient
    precision_metadata = dict(op._contest_metadata)
    precision_metadata.update(cache_dtype=str(cache.dtype), cache_bytes=cache.numel()*cache.element_size(),
                              reference=op._formula_reference_C0)
    micro = study.microbenchmark(op)
    validation = None
    if args.grid <= 8:
        # Independent dense 3x3 inverses of the assembled tensor-product symbol.
        m = op.mesh
        lam, mu = np.asarray(op._formula_reference_C0)[0, 1], np.asarray(op._formula_reference_C0)[3, 3]
        angles = [2*np.pi*np.arange(n)/n for n in (m.Nx, m.Ny, m.Nz)]
        axes = []
        for theta, h in zip(angles, (m.dx, m.dy, m.dz)):
            mass = (2+np.cos(theta))/3
            axes.append((mass, (2-2*np.cos(theta))/(h*h*mass), np.sin(theta)/(h*mass)))
        symbol = np.empty((m.Nz, m.Ny, m.Nx, 3, 3))
        for z in range(m.Nz):
            for y in range(m.Ny):
                for x in range(m.Nx):
                    values = [axes[j][1][k] for j, k in enumerate((x, y, z))]
                    v = np.array([axes[j][2][k] for j, k in enumerate((x, y, z))])
                    weight = m.V_elem*np.prod([axes[j][0][k] for j, k in enumerate((x, y, z))])
                    matrix = (lam+mu)*np.outer(v, v)
                    np.fill_diagonal(matrix, mu*sum(values)+(lam+mu)*np.array(values))
                    symbol[z, y, x] = weight*matrix
        inverse = np.zeros_like(symbol)
        flat = inverse.reshape(-1, 3, 3)
        flat[1:] = np.linalg.inv(symbol.reshape(-1, 3, 3)[1:])
        rng = np.random.default_rng(20261024)
        rhs = rng.normal(size=(m.Nz, m.Ny, m.Nx, 3))
        rhs -= rhs.mean(axis=(0, 1, 2))
        spectrum = np.fft.fftn(rhs, axes=(0, 1, 2))
        expected = np.fft.ifftn(np.einsum("...ij,...j->...i", inverse, spectrum), axes=(0, 1, 2)).real
        actual = op.apply_precond(torch.tensor(rhs.reshape(1, -1), device="cuda", dtype=torch.float64)).cpu().numpy()
        error = float(np.linalg.norm(actual.reshape(expected.shape)-expected)/np.linalg.norm(expected))
        assert error < (2e-12 if args.precision == "fp64" else 3e-6), error
        validation = dict(preconditioner_relative_to_dense_symbol=error, includes_nyquist=True,
                          reference="FP64 numpy FFT and dense inverse of tensor-product FE symbol")
    b.gpu_guard()
    sources = study.sources()
    for path in [Path(__file__), ROOT/"run.py", ROOT/"project_paths.py", *source_path("spectral").glob("precision_*")]:
        if path.is_file():
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(complete=True, precision=args.precision, grid=args.grid, repeat=args.repeat,
                  input_hash=digest, input=meta, measurement=result, workflow_seconds=elapsed,
                  max_true_residual=residual, precision_metadata=precision_metadata,
                  reference_design=b._reference_design, storage=b._storage, operator=b._operator_metadata,
                  microbenchmark=micro, validation=validation, sources=sources,
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                   cuda=torch.version.cuda, python=sys.version, threads=torch.get_num_threads()),
                  protocol="fresh process, complete same-grid warmup; six loads; same final hybrid/background solver; FP64 physics/PCG; native FFT and coefficient dtype only; true residual <=1e-8")
    save(args.out / "results" / f"{args.grid}_{args.precision}_r{args.repeat}.json", report)
    print("PASS", args.grid, args.precision, elapsed, residual, flush=True)


def launch(args):
    # Reuse the established external 20-ms NVML sampler unchanged.
    runner = load_file("precision_nvml_runner", ROOT / "experiments/solver_comparison/run_jobs.py")
    nvml = runner.NVML()
    pairs = [(0, "original"), (0, "fp32"), (0, "fp64")] if args.grid <= 8 else [
        (i, p) for i in range(3) for p in (("fp32", "fp64") if i % 2 == 0 else ("fp64", "fp32"))]
    jobs = []
    for repeat, precision in pairs:
        name = f"{args.grid}_{precision}_r{repeat}"
        command = [str(ROOT/"run"), "fft-precision", "--phase", "worker", "--grid", str(args.grid),
                   "--precision", precision, "--repeat", str(repeat), "--out", str(args.out)]
        jobs.append(dict(name=name, arguments=[str(Path(__file__))], command=command))
    save(args.out/"plan.json", jobs)
    completed = []
    for job in jobs:
        completed.append(runner.execute(job, nvml, .02))
        print("DONE", job["name"], flush=True)
    save(args.out/"complete.json", dict(complete=True, jobs=completed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("launch", "worker"), required=True)
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--precision", choices=("original", "fp32", "fp64"), default="fp32")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT / "runs"):
        p.error("Outputs must be under runs/")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["VOXEL_RUN_DIR"] = str(args.out)
    worker(args) if args.phase == "worker" else launch(args)


if __name__ == "__main__":
    main()
