"""Validated pipeline ablations and fresh-process, interleaved 192^3 timings."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = Path(os.environ.get("FEAD_MODAL", str(__import__('project_paths').source_path('hex8_modal'))))
sys.path.insert(0, str(OLD))
import gpu_study as previous
import modal_adapter as modal
import pipeline_adapter as pipeline


def save(path, value):
    previous.save(path, value)


def provenance(b):
    result = previous.provenance(b)
    result["pipeline_source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in HERE.iterdir() if p.suffix in (".py", ".cu", ".cuh")}
    return result


def validate(kind):
    b = previous.load_runtime(kind)
    import numpy as np
    from femlib.geometry import make_B_matrices, LOCAL_OFFSET
    torch = b.torch
    b.gpu_guard(); modal.extension(); pipeline.extension()
    b.TOL = 1e-10
    grid, voxel = (5, 6, 7), (.3, .5, .7)
    n = int(np.prod(grid))
    torch.manual_seed(20261020)
    a = torch.randn((1, n, 6, 6), device="cuda", dtype=torch.float64)
    full = a.transpose(-1, -2) @ a + 2 * torch.eye(6, device="cuda", dtype=torch.float64)
    c = torch.stack([full[..., i, j] for i, j in b.PAIRS], dim=-1).contiguous()
    if kind == "hg":
        with pipeline.lean_hg_initialization():
            op = b.make_solver(c, grid, voxel, fft=False).op
        pipeline.attach_lazy_diag(op)
        assert op._diag_cache is None
        b.enable_reference(op, 2, "mixed")
    else:
        import atomic_adapter
        op = atomic_adapter.make(c, grid, "atomic128", voxel).op
    modal.attach(op, kind, voxel, variant="atomic", block=128)
    pipeline.attach_energy(op)
    op._validation_kind = kind
    matrix, rhs_cpu, u_cpu = previous.cpu_reference(b, op, full, voxel)
    v = torch.randn((1, 3*n), device="cuda", dtype=torch.float64)
    ku, energy = op.apply_with_energy(v)
    csr_ku = matrix @ v[0].cpu().numpy()
    energy_ref = float(v[0].cpu().numpy() @ csr_ku)
    row = dict(kind=kind, grid=grid, voxel=voxel,
        Ku_relative=previous.relative(ku[0].cpu().numpy(), csr_ku),
        energy_relative=abs(float(energy)-energy_ref)/abs(energy_ref))
    if kind == "hg":
        z_before = op.apply_precond(v)
        pipeline.enable_analytic_reference(op)
        z_after = op.apply_precond(v)
        row["mixed_reference_relative"] = previous.relative(z_before.cpu().numpy(), z_after.cpu().numpy())
        # Test lazy setup explicitly, but not in measured FFT-only runs.
        diag = op.diag().cpu().numpy()[0]
        row["lazy_diagonal_relative"] = previous.relative(diag, matrix.diagonal())
    fn = pipeline.energy_solver()
    b.solve_pcg_batched = fn
    bg, _ = make_B_matrices(*voxel, element_type="C3D8R" if kind=="hg" else "C3D8")
    ce = full[0].cpu().numpy()
    nodes = []
    for e in range(n):
        z, rem = divmod(e, grid[1]*grid[2]); y, x = divmod(rem, grid[2])
        nodes.append([(((z+oz)%grid[0]*grid[1]+(y+oy)%grid[1])*grid[2]+(x+ox)%grid[2])
                      for ox, oy, oz in LOCAL_OFFSET])
    nodes = np.asarray(nodes)
    loads, ch, ch_cpu = [], [], []
    def fields(u, macro):
        strain = np.einsum("gij,ej->egi", bg, u[nodes].reshape(n, 24)) + macro
        return np.einsum("eij,egj->egi", ce, strain)
    for j, macro in enumerate(np.eye(6)):
        rhs = op.project_zero_mean(op.compute_rhs(macro))
        u, report = b.solve(op, rhs)
        un = u[0].cpu().numpy().reshape(n, 3)
        stress, ref_stress = fields(un, macro), fields(u_cpu[:, :, j], macro)
        column = op.compute_avg_stress(u, macro)[0]
        ch.append(column); ch_cpu.append(ref_stress.mean(axis=(0, 1)))
        loads.append(dict(**report, independent_true_residual=float(np.linalg.norm(rhs_cpu[:,j]-matrix@un.ravel())/np.linalg.norm(rhs_cpu[:,j])),
            displacement_relative=previous.relative(un, u_cpu[:,:,j]),
            local_stress_relative=previous.relative(stress, ref_stress),
            recovery_relative=previous.relative(column, stress.mean(axis=(0,1)))))
    row.update(loads=loads, CH_relative=previous.relative(np.column_stack(ch), np.column_stack(ch_cpu)),
               provenance=provenance(b))
    row["passed"] = (row["Ku_relative"] < 1e-12 and row["energy_relative"] < 1e-12
        and row.get("mixed_reference_relative", 0) < 2e-5 and row.get("lazy_diagonal_relative", 0) < 1e-12
        and row["CH_relative"] < 1e-8
        and all(x["independent_true_residual"] <= 1.02e-10 and x["local_stress_relative"] < 1e-8 for x in loads))
    save(HERE/"results"/f"validate_{kind}.json", row)
    print("VALIDATION", kind, row["passed"], row["Ku_relative"], row["CH_relative"], flush=True)
    if not row["passed"]:
        raise RuntimeError("Pipeline validation failed")


def measure(kind, variant, grid, repeat, pilot=False):
    b = previous.load_runtime(kind)
    b.gpu_guard(); modal.extension(); pipeline.extension()
    pipeline.configure(b, modal, kind, variant)
    data, geom, material, digest = b.inputs(grid, "strong")
    path = HERE/("pilot" if pilot else "results")/f"measure_{kind}_{variant}_{grid}_{repeat}.json"
    if path.exists():
        raise FileExistsError(path)
    record = dict(kind=kind, variant=variant, grid=grid, repeat=repeat, pilot=pilot,
        input_hash=digest, geometry=geom, material=material, tolerance=b.TOL,
        provenance=provenance(b),
        protocol="fresh process; complete six-load same-size warmup; all GPU material/operator/reference setup, RHS, PCG, projected FP64 true residual and average stress included; CPU geometry/disk/JIT/warmup excluded; baseline phase synchronizations retained")
    try:
        record["phase"]="warmup"; save(path, record)
        b.workload(data, material, "direct", "mixed")
        b.gpu_guard(); record["phase"]="measurement"; save(path, record)
        result, _ = b.workload(data, material, "direct", "mixed")
        b.gpu_guard()
        if max(v for c in result["calls"] for v in c["true_residuals"]) > 1e-8:
            raise RuntimeError("Unconverged formal load")
        record.update(status="passed", measurement=result)
        print("MEASURE", kind, variant, repeat, result["seconds"], flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save(path, record)


def orchestrate(grid):
    state_path = HERE/"status.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state=dict(status="running", grid=grid, started_utc=datetime.now(timezone.utc).isoformat(), completed=[])
    save(state_path, state)
    def run(phase, kind, variant=None, repeat=0):
        if previous.foreign():
            raise RuntimeError("GPU occupied: " + str(previous.foreign()))
        label="_".join(str(x) for x in (phase,kind,variant,repeat) if x is not None)
        state["active"]=label;save(state_path,state)
        command=[sys.executable,"-u",str(HERE/"pipeline_study.py"),"--phase",phase,"--kind",kind,"--grid",str(grid)]
        if variant is not None:command += ["--variant",variant,"--repeat",str(repeat)]
        samples=[]
        with (HERE/"logs"/(label+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(2)
                others=previous.foreign(child.pid)
                if others:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
                    raise RuntimeError("Our job stopped because another GPU process appeared: "+str(others))
                raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
                samples.append(dict(time=time.time(),gpu=raw))
        save(HERE/"logs"/(label+"_telemetry.json"),samples)
        if child.returncode:
            raise RuntimeError(label+" failed; inspect log")
        state["completed"].append(label);save(state_path,state)
        print("DONE",label,flush=True)
    try:
        for kind in ("hg","fi"):run("validate",kind)
        candidates={"hg":["modal","lean","energy","analytic","combined"],"fi":["modal","energy","combined"]}
        for kind, variants in candidates.items():
            for variant in variants:run("pilot",kind,variant)
        selected={}
        for kind, variants in candidates.items():
            records=[json.loads((HERE/"pilot"/f"measure_{kind}_{variant}_{grid}_0.json").read_text()) for variant in variants]
            chosen=min(records,key=lambda r:r["measurement"]["seconds"])
            selected[kind]=chosen["variant"]
        save(HERE/"selection.json",dict(selected=selected,scope="one complete pilot per predefined candidate; formal repetitions follow"))
        state["selected"]=selected;save(state_path,state)
        for repeat in range(3):
            jobs=[(kind,variant) for kind in ("hg","fi") for variant in dict.fromkeys(("paper","modal",selected[kind]))]
            random.Random(20261020+repeat).shuffle(jobs)
            for kind, variant in jobs:run("measure",kind,variant,repeat)
        state.update(status="complete",ended_utc=datetime.now(timezone.utc).isoformat());state.pop("active",None)
        save(state_path,state)
    except BaseException as error:
        state.update(status="stopped",error=repr(error));save(state_path,state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("validate","pilot","measure","orchestrate"),required=True)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--variant",choices=("paper","modal","lean","energy","analytic","combined"),default="modal")
    parser.add_argument("--grid",type=int,default=192)
    parser.add_argument("--repeat",type=int,default=0)
    args=parser.parse_args()
    for name in ("results","pilot","logs"):(HERE/name).mkdir(exist_ok=True)
    if args.phase=="validate":validate(args.kind)
    elif args.phase=="orchestrate":orchestrate(args.grid)
    else:measure(args.kind,args.variant,args.grid,args.repeat,args.phase=="pilot")


if __name__=="__main__":main()
