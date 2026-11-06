"""Matched RTX 5090 study using the frozen manuscript runtime.

Phases: validate (tiny independent CPU CSR), tune (resident K*u), measure
(one fresh-process full workflow), orchestrate (serial complete study).
No frozen source files are edited.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("FEAD_RUNTIME", str(__import__('project_paths').source_path('runtime'))))
OUT = HERE/"results"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def load_runtime(kind):
    os.environ.setdefault("FEAD_TOL", "1e-8")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ["PATH"] = str(Path(sys.executable).parent)+":/usr/local/cuda/bin:"+os.environ.get("PATH", "")
    sys.path.insert(0, str(RUNTIME))
    if kind == "fi":
        import fi_benchmark
    import benchmark as b
    b.torch.set_num_threads(1)
    return b


def provenance(b):
    import modal_adapter
    paths = [*HERE.glob("*.py"), *HERE.glob("*.cu"), RUNTIME/"benchmark.py", RUNTIME/"fi_benchmark.py",
             RUNTIME/"repro/extension_20261006/material_control.py",
             RUNTIME/"repro/extension_20261006/run_resolution.py"]
    for root in [RUNTIME/"repro/source/femlib", RUNTIME/"fi/source/femlib"]:
        paths.extend(p for p in root.rglob("*") if p.suffix in (".py", ".cu", ".cpp", ".h"))
    device = b.torch.cuda.get_device_properties(0)
    return dict(runtime=str(RUNTIME), experiment=str(HERE), python=sys.executable,
        torch=b.torch.__version__, cuda=b.torch.version.cuda, gpu=b.gpu_guard(),
        gpu_properties=str(device), allocator=os.environ.get("PYTORCH_ALLOC_CONF", "default"),
        compiler_flags=["-O3", "no --use_fast_math"],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})


def relative(a, b):
    import numpy as np
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def cpu_reference(b, op, full, voxel):
    """Conventional frozen B matrices and explicit CPU CSR/direct factorization."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from femlib.geometry import make_B_matrices, LOCAL_OFFSET
    shape = (op.mesh.Nz, op.mesh.Ny, op.mesh.Nx)
    n = int(np.prod(shape))
    bg, weight = make_B_matrices(*voxel, element_type="C3D8" if op._validation_kind == "fi" else "C3D8R")
    hourglass = b.hourglass_basis()
    ce = full[0].cpu().numpy()
    scales = op.hg_scale[0].cpu().numpy() if op._validation_kind == "hg" else None
    rows, columns, values = [], [], []
    rhs = np.zeros((3*n, 6))
    for element in range(n):
        z, rem = divmod(element, shape[1]*shape[2])
        y, x = divmod(rem, shape[2])
        nodes = [(((z+oz)%shape[0]*shape[1]+(y+oy)%shape[1])*shape[2]+(x+ox)%shape[2])
                 for ox, oy, oz in LOCAL_OFFSET]
        dofs = np.array([3*node+d for node in nodes for d in range(3)])
        ke = sum(weight*point.T@ce[element]@point for point in bg)
        if scales is not None:
            ke += scales[element]*(hourglass@hourglass.T)
        rhs[dofs] -= sum(weight*point.T@ce[element] for point in bg)
        rows.extend(np.repeat(dofs, 24));columns.extend(np.tile(dofs, 24));values.extend(ke.ravel())
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(3*n, 3*n)).tocsr()
    factor = splu(matrix[3:, 3:].tocsc())
    u = np.zeros((3*n, 6));u[3:] = factor.solve(rhs[3:])
    u = u.reshape(n, 3, 6);u -= u.mean(axis=0, keepdims=True)
    return matrix, rhs, u


def validation(kind):
    b = load_runtime(kind)
    import numpy as np
    import modal_adapter as modal
    from femlib.geometry import LOCAL_OFFSET, make_B_matrices
    b.gpu_guard();modal.extension();b.TOL=1e-10
    grid, voxel = (5, 6, 7), (.3, .5, .7)
    n = int(np.prod(grid));torch=b.torch
    torch.manual_seed(20261020)
    a = torch.randn((1, n, 6, 6), device="cuda", dtype=torch.float64)
    full = a.transpose(-1, -2)@a+2*torch.eye(6, device="cuda", dtype=torch.float64)
    c = torch.stack([full[..., i, j] for i, j in b.PAIRS], dim=-1).contiguous()
    if kind == "hg":
        op = b.make_solver(c, grid, voxel, fft=False).op
        b.enable_reference(op, 2, "mixed")
    else:
        import memory_adapter
        op = memory_adapter.make(c, grid, "lifetime", voxel).op
    op._validation_kind=kind
    original = op.apply
    matrix, cpu_rhs, cpu_u = cpu_reference(b, op, full, voxel)
    bg, _ = make_B_matrices(*voxel, element_type="C3D8" if kind == "fi" else "C3D8R")
    ce = full[0].cpu().numpy()
    nodes = []
    for element in range(n):
        z, rem=divmod(element, grid[1]*grid[2]);y,x=divmod(rem,grid[2])
        nodes.append([(((z+oz)%grid[0]*grid[1]+(y+oy)%grid[1])*grid[2]+(x+ox)%grid[2]) for ox,oy,oz in LOCAL_OFFSET])
    nodes=np.array(nodes)
    def fields(u, macro):
        local=u[nodes].reshape(n,24)
        strain=np.einsum("gij,ej->egi",bg,local)+macro
        stress=np.einsum("eij,egj->egi",ce,strain)
        return stress.mean(axis=(0,1)),stress
    reference_ch=np.column_stack([fields(cpu_u[:,:,j],np.eye(6)[j])[0] for j in range(6)])
    v=torch.randn((1,3*n),device="cuda",dtype=torch.float64)
    ref=matrix@v[0].cpu().numpy()
    native_relative=relative(original(v)[0].cpu().numpy(),ref)
    configs=[dict(variant="twostage",tile=[8,8,8],block=128),
             dict(variant="atomic",tile=[8,8,8],block=128),
             dict(variant="stream",tile=[8,8,8],block=128),
             dict(variant="stream",tile=[16,8,16],block=128),
             dict(variant="stream",tile=[4,4,4],block=64)]
    results=[]
    for index, config in enumerate(configs):
        b.gpu_guard();modal.attach(op,kind,voxel,**config)
        value=op.apply(v)
        row=dict(config=config,Ku_relative=relative(value[0].cpu().numpy(),ref),
                 attributes=modal.extension().attributes(kind=="fi",op._modal_variant))
        if index<3:
            loads=[];columns=[]
            for j,macro in enumerate(np.eye(6)):
                rhs=op.project_zero_mean(op.compute_rhs(macro))
                u,report=b.solve(op,rhs)
                un=u[0].cpu().numpy().reshape(n,3)
                independent=float(np.linalg.norm(cpu_rhs[:,j]-matrix@un.ravel())/np.linalg.norm(cpu_rhs[:,j]))
                column,stress=fields(un,macro)
                _,reference_stress=fields(cpu_u[:,:,j],macro)
                columns.append(op.compute_avg_stress(u,macro)[0])
                loads.append(dict(load=j,**report,independent_true_residual=independent,
                    displacement_relative=relative(un,cpu_u[:,:,j]),local_stress_relative=relative(stress,reference_stress),
                    recovery_relative=relative(columns[-1],column)))
            row.update(CH_relative=relative(np.column_stack(columns),reference_ch),loads=loads)
        row["passed"]=(row["Ku_relative"]<1e-12 and row.get("CH_relative",0)<1e-8
            and all(load["independent_true_residual"]<1.01e-10 and load["local_stress_relative"]<1e-8
                    and load["recovery_relative"]<1e-12 for load in row.get("loads",[])))
        results.append(row)
        print("VALIDATE",kind,config,row["Ku_relative"],row.get("CH_relative"),row["passed"],flush=True)
    report=dict(kind=kind,grid=grid,voxel=voxel,tolerance=b.TOL,reference="independent CPU CSR/direct solve using frozen conventional B matrices",
        native_Ku_relative=native_relative,results=results,passed=native_relative<1e-12 and all(r["passed"] for r in results),provenance=provenance(b))
    save(OUT/f"validate_{kind}.json",report)
    if not report["passed"]:raise RuntimeError("GPU validation failed")


def tuning(kind, grid):
    b=load_runtime(kind)
    import modal_adapter as modal
    torch=b.torch
    b.gpu_guard();modal.extension()
    data,geom,material,digest=b.inputs(grid,"strong")
    stages=b.Stages()
    op=b.build(data,material,"direct" if kind=="hg" else "tap14","mixed",stages)
    voxel=data['domain_size'][0]/b.np.array(data['grid_shape'][::-1])
    original=op.apply
    torch.manual_seed(91620)
    v=torch.randn((1,3*grid**3),device="cuda",dtype=torch.float64)
    ref=original(v);b.synchronize()
    configs=[dict(variant="twostage",tile=[8,8,8],block=128),
             dict(variant="twostage",tile=[8,8,8],block=256),
             dict(variant="atomic",tile=[8,8,8],block=128),
             dict(variant="atomic",tile=[8,8,8],block=256)]
    configs.extend(dict(variant="stream",tile=list(tile),block=block)
                   for tile in [(8,4,8),(8,8,8),(8,8,16),(16,4,8),(16,8,8),(16,8,16)]
                   for block in [128,256])
    random.Random(20261020).shuffle(configs)
    def event_times(fn):
        for _ in range(3):out=fn()
        b.synchronize();samples=[]
        for _ in range(7):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):out=fn()
            end.record();end.synchronize();samples.append(begin.elapsed_time(end)/10)
        return samples
    baseline=event_times(lambda:original(v))
    rows=[]
    for config in configs:
        b.gpu_guard();modal.attach(op,kind,voxel,**config)
        value=op.apply(v);b.synchronize()
        error=float(torch.linalg.vector_norm(value-ref)/torch.linalg.vector_norm(ref))
        if error>=1e-12:raise RuntimeError(f"192-grid operator mismatch {error} {config}")
        samples=event_times(lambda:op.apply(v))
        row=dict(config=config,milliseconds=samples,median_ms=statistics.median(samples),
                 Ku_relative=error,attributes=modal.extension().attributes(kind=="fi",op._modal_variant))
        rows.append(row);print("TUNE",kind,grid,config,row["median_ms"],flush=True)
        save(OUT/f"tune_{kind}_{grid}.json",dict(complete=False,rows=rows))
    best=min(rows,key=lambda r:r["median_ms"])
    report=dict(kind=kind,grid=grid,baseline="HG direct CUDA thread128" if kind=="hg" else "FI 14-tap CUDA thread128",
        baseline_milliseconds=baseline,baseline_median_ms=statistics.median(baseline),rows=rows,best_config=best["config"],
        input_hash=digest,geometry=geom,complete=True,provenance=provenance(b),
        scope="resident physical K*u only; seven CUDA-event batches of ten applies; shared baseline C21/reference; JIT/setup excluded")
    save(OUT/f"tune_{kind}_{grid}.json",report)


def measure(kind,method,repeat,grid):
    b=load_runtime(kind)
    import modal_adapter as modal
    b.gpu_guard();modal.extension()
    out=OUT/f"measure_{kind}_{method}_{grid}_{repeat}.json"
    if out.exists():raise FileExistsError(out)
    config=None
    if method=="modal_best":
        config=json.loads((OUT/f"tune_{kind}_{grid}.json").read_text())["best_config"]
    elif method=="modal_twostage":config=dict(variant="twostage",tile=[8,8,8],block=128)
    elif method!="baseline":raise ValueError(method)
    original_build=b.build
    def build(data,material,route,precision,stages):
        base_route=("direct" if kind=="hg" or method!="baseline" else "tap14")
        op=original_build(data,material,base_route,precision,stages)
        if config is not None:
            voxel=data['domain_size'][0]/b.np.array(data['grid_shape'][::-1])
            modal.attach(op,kind,voxel,**config)
        stages.end("candidate_attachment",op)
        return op
    b.build=build
    data,geom,material,digest=b.inputs(grid,"strong")
    record=dict(kind=kind,method=method,grid=grid,repeat=repeat,configuration=config,
        input_hash=digest,geometry=geom,material=material,tolerance=b.TOL,
        provenance=provenance(b),
        protocol="fresh process per repetition; one same-size six-load warmup; mixed RFFT upper6 preconditioner; frozen compiled PCG; all material/operator/reference setup, RHS, solves, true residuals and stress recovery included; geometry/disk/JIT/warmup excluded")
    try:
        record["phase"]="warmup";save(out,record)
        b.workload(data,material,"direct","mixed")
        b.gpu_guard();record["phase"]="measurement";save(out,record)
        result,_=b.workload(data,material,"direct","mixed")
        b.gpu_guard();record.update(status="passed",measurement=result)
        if max(v for c in result["calls"] for v in c["true_residuals"])>1e-8:
            raise RuntimeError("True residual did not reach 1e-8")
        print("MEASURE",kind,method,repeat,result["seconds"],"s",flush=True)
    except BaseException as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:save(out,record)


def foreign(own=None):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    return [row for row in rows.splitlines() if row.strip() and int(row.split(',')[0]) not in (os.getpid(),own)]


def orchestrate(grid):
    OUT.mkdir(exist_ok=True);(HERE/"logs").mkdir(exist_ok=True)
    state_path=HERE/"status.json"
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),grid=grid,completed=[])
    jobs=[("validate",kind,None,None) for kind in ("hg","fi")]
    jobs += [("tune",kind,None,None) for kind in ("hg","fi")]
    for repeat in range(3):
        methods=[(kind,method) for kind in ("hg","fi") for method in ("baseline","modal_twostage","modal_best")]
        random.Random(20261020+repeat).shuffle(methods)
        jobs += [("measure",kind,method,repeat) for kind,method in methods]
    state["total_jobs"]=len(jobs)
    save(state_path,state)
    try:
        for phase,kind,method,repeat in jobs:
            label="_".join(str(v) for v in (phase,kind,method,repeat) if v is not None)
            if phase=="validate" and (OUT/f"validate_{kind}.json").exists():
                previous=json.loads((OUT/f"validate_{kind}.json").read_text())
                current={str(HERE/name):hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                         for name in ("modal_adapter.py","modal_kernels.cu")}
                if previous.get("passed") and all(previous["provenance"]["source_sha256"].get(k)==v for k,v in current.items()):
                    state["completed"].append(label);save(state_path,state)
                    print("REUSE PASSED",label,flush=True)
                    continue
            if foreign():raise RuntimeError("GPU occupied before job: "+str(foreign()))
            state["active"]=label;save(state_path,state)
            command=[sys.executable,"-u",str(Path(__file__).resolve()),"--phase",phase,"--kind",kind,"--grid",str(grid)]
            if method is not None:command += ["--method",method,"--repeat",str(repeat)]
            samples=[]
            with (HERE/"logs"/(label+".log")).open("w") as log:
                child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    time.sleep(1)
                    others=foreign(child.pid)
                    if others:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
                        raise RuntimeError("Stopped our job because GPU became occupied: "+str(others))
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    samples.append(dict(time=time.time(),gpu=raw))
                code=child.returncode
            save(HERE/"logs"/(label+"_telemetry.json"),samples)
            if code:raise RuntimeError(f"{label} failed with {code}; inspect its log")
            state["completed"].append(label);save(state_path,state)
            print("DONE",label,flush=True)
        state.update(status="complete",ended_utc=datetime.now(timezone.utc).isoformat());state.pop("active",None)
        save(state_path,state)
    except BaseException as error:
        state.update(status="stopped",error=repr(error));save(state_path,state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["validate","tune","measure","orchestrate"],required=True)
    parser.add_argument("--kind",choices=["hg","fi"],default="hg")
    parser.add_argument("--method",choices=["baseline","modal_twostage","modal_best"],default="baseline")
    parser.add_argument("--repeat",type=int,default=0)
    parser.add_argument("--grid",type=int,default=192)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.phase=="validate":validation(args.kind)
    elif args.phase=="tune":tuning(args.kind,args.grid)
    elif args.phase=="measure":measure(args.kind,args.method,args.repeat,args.grid)
    else:orchestrate(args.grid)


if __name__=="__main__":main()
