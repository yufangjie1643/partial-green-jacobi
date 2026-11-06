"""Six-load chained optimization ablation; independent CPU verification first."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time

import chain_adapter as a

spec = importlib.util.spec_from_file_location("chain_fft_study", a.FFT / "study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
VARIANTS = ("baseline", "boundary", "snapshot", "combined", "compact", "compact_combined")


def save(path, data):
    s.save(path, data)


def provenance(b):
    result = s.provenance(b)
    result["chain_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in HERE.glob("*.py")}
    return result


def validate(kind):
    b = s.load(kind, configure=False)
    torch, np = b.torch, b.np
    b.TOL = 1e-10
    a.native.extension()
    grid, voxel = (5, 6, 7), (.3, .5, .7)
    n = int(np.prod(grid))
    torch.manual_seed(1701)
    q = torch.randn((1, n, 6, 6), device="cuda", dtype=torch.float64)
    full = q @ q.transpose(-1, -2) + 2 * torch.eye(6, device="cuda", dtype=torch.float64)
    packed = torch.stack([full[..., i, j] for i, j in b.PAIRS], dim=-1).contiguous()
    if kind == "hg":
        with s.f.formula.pipeline.lean_hg_initialization():
            op = b.make_solver(packed, grid, voxel, fft=False).op
        s.f.prior.modal.attach(op, kind, voxel, variant="atomic", block=128)
        s.f.formula.pipeline.enable_analytic_reference(op)
        s.f.formula.pipeline.attach_energy(op)
        fallback = s.f.formula.pipeline.energy_solver()
        b.solve_pcg_batched = fallback
    else:
        import atomic_adapter
        op = atomic_adapter.make(packed, grid, "atomic128", voxel).op
        s.f.prior.modal.attach(op, kind, voxel, variant="atomic", block=128)
        s.f.formula.enable_fi_reference(op)
        fallback = b.solve_pcg_batched
    op._validation_kind = kind
    matrix, rhs_cpu, u_cpu = s.f.previous.cpu_reference(b, op, full, voxel)
    a.native.attach(op, kind, "cached" if kind == "hg" else "dense")
    a.attach_raw(op)
    r = torch.randn((1, 3*n), device="cuda", dtype=torch.float64)
    raw = op.precond_packed(s.c.extension().pack(r, *grid))
    z = raw.reshape(1, 3, -1).transpose(1, 2).contiguous().view_as(r).double()
    assert torch.equal(z, op.apply_precond(r)), "Raw FFT layout/conversion differs"
    rows, base = [], None
    for variant in ("baseline", "boundary", "snapshot", "combined"):
        b.solve_pcg_batched = fallback if variant == "baseline" else a.make_solver(
            kind, variant in ("boundary", "combined"), variant in ("snapshot", "combined"), fallback)
        columns, calls, solutions = [], [], []
        for j, macro in enumerate(np.eye(6)):
            rhs = op.project_zero_mean(op.compute_rhs(macro))
            u, report = b.solve(op, rhs)
            value = u[0].cpu().numpy()
            independent = float(np.linalg.norm(rhs_cpu[:, j] - matrix @ value) / np.linalg.norm(rhs_cpu[:, j]))
            assert independent <= 1.01e-10
            columns.append(op.compute_avg_stress(u, macro)[0])
            calls.append(dict(report=report, independent_true_residual=independent,
                              displacement_vs_direct=s.f.previous.relative(value.reshape(n, 3), u_cpu[:, :, j])))
            solutions.append(value)
        ch = np.column_stack(columns)
        if base is None:
            base = ch.copy()
        error = s.f.previous.relative(ch, base)
        assert error < 1e-8
        # Both zero RHS and nonzero x0 paths retain the outer true-residual check.
        zero, zero_report = b.solve(op, torch.zeros_like(rhs))
        assert bool((zero == 0).all()) and zero_report["iterations"] == 0
        ux, _, _ = b.solve_pcg_batched(op, rhs, tol=1e-10, max_iter=2000,
                                      update_mode="compiled", convergence_check_stride=4, x0=.7*u)
        ux = op.project_zero_mean(ux)
        x0_res = float(torch.linalg.vector_norm(rhs-op.apply(ux)) / torch.linalg.vector_norm(rhs))
        assert x0_res <= 1.01e-10
        # Force the exceptional max-iteration path. The fallback must return
        # an actual historical iterate, not an uninitialized/frozen buffer.
        limited, count, _ = b.solve_pcg_batched(op, rhs, tol=1e-30, max_iter=1,
                                               update_mode="compiled", convergence_check_stride=4)
        trusted, _, _ = fallback(op, rhs, tol=1e-30, max_iter=1,
                                 update_mode="compiled", convergence_check_stride=4)
        cap_error = s.relative(limited, trusted)
        assert cap_error < 1e-12 and count == 1
        row = dict(variant=variant, CH=ch.tolist(), CH_relative=error, calls=calls,
                   nonzero_initial_true_residual=x0_res, capped_solve_vs_fallback=cap_error,
                   zero_rhs_passed=True)
        rows.append(row)
        print("VALIDATE", kind, variant, error, x0_res, cap_error, flush=True)
    save(OUT/f"validate_{kind}.json", dict(complete=True, kind=kind, grid=grid, voxel=voxel,
                                        rows=rows, provenance=provenance(b)))


def measure(kind, variant, grid, case, repeat, stage):
    path = OUT/f"{stage}_{kind}_{variant}_{case}_{grid}_{repeat}.json"
    if path.exists():
        raise FileExistsError(path)
    b = s.load(kind)
    a.native.extension()
    a.configure(b, kind, variant)
    data, geometry, material, digest = b.inputs(grid, case)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    result, _ = b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    assert max(v for call in result["calls"] for v in call["true_residuals"]) <= b.TOL
    record = dict(complete=True, kind=kind, variant=variant, grid=grid, case=case, repeat=repeat,
                  stage=stage, tolerance=b.TOL, input_hash=digest, geometry=geometry, material=material,
                  measurement=result, provenance=provenance(b),
                  protocol="fresh process; full same-grid six-load warmup; full six-load timed workflow including setup and outer FP64 true residual; native FFT baseline already includes modal K and prior PCG optimizations")
    save(path, record)
    print(kind, variant, grid, case, result["seconds"], result["peak_allocated_bytes"], flush=True)


def orchestrate():
    def run(*args):
        s.guard()
        with (OUT/("_".join(map(str, args))+".log")).open("x") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), *map(str, args)],
                                     stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    time.sleep(2)
                    s.guard((child.pid,))
            except BaseException:
                child.terminate()
                child.wait()
                raise
            if child.returncode:
                raise RuntimeError(f"Child failed: {args}")
        print("DONE", *args, flush=True)
    for kind in ("hg", "fi"):
        run("--phase", "validate", "--kind", kind)
    for kind in ("hg", "fi"):
        for variant in VARIANTS if kind == "hg" else VARIANTS[:4]:
            run("--phase", "measure", "--kind", kind, "--variant", variant, "--stage", "tune")
    selected = {}
    for kind in ("hg", "fi"):
        rows = [json.loads(p.read_text()) for p in OUT.glob(f"tune_{kind}_*_strong_192_0.json")]
        selected[kind] = min(rows, key=lambda x:x["measurement"]["seconds"])["variant"]
    save(OUT/"selection.json", selected)
    for repeat in range(3):
        pairs = [(k, v) for k in ("hg", "fi") for v in dict.fromkeys(("baseline", selected[k]))]
        random.Random(902+repeat).shuffle(pairs)
        for kind, variant in pairs:
            run("--phase", "measure", "--kind", kind, "--variant", variant,
                "--stage", "formal", "--repeat", repeat)
    for kind in ("hg", "fi"):
        for variant in dict.fromkeys(("baseline", selected[kind])):
            run("--phase", "measure", "--kind", kind, "--variant", variant, "--grid", 128,
                "--case", "near", "--stage", "smoke")
    save(OUT/"complete.json", dict(complete=True, selected=selected))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("validate", "measure", "orchestrate"), required=True)
    p.add_argument("--kind", choices=("hg", "fi"), default="hg")
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--case", choices=("strong", "near"), default="strong")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--stage", default="tune")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "orchestrate":
        orchestrate()
    elif args.phase == "validate":
        validate(args.kind)
    else:
        measure(args.kind, args.variant, args.grid, args.case, args.repeat, args.stage)
