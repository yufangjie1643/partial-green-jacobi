"""Same-GPU accuracy and complete-workflow tests of formula reductions."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import formula_adapter as formula
prior=formula.prior
previous=prior.previous
HERE=Path(__file__).resolve().parent


def save(path,value):previous.save(path,value)


def provenance(b):
    result=prior.provenance(b)
    result["formula_source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in (".py",".cu",".cuh")}
    result["precision_protocol"]="FI analytic reference FP32/complex64; HG contrast solver FP64/complex128; physical operator, final true residual and stresses FP64"
    return result


def validate(kind,variant,even=False):
    b=previous.load_runtime(kind)
    import numpy as np
    from femlib.geometry import make_B_matrices,LOCAL_OFFSET
    torch=b.torch;b.gpu_guard();formula.extension();prior.modal.extension();formula.pipeline.extension();b.TOL=1e-10
    grid,voxel=((4,6,8) if even else (5,6,7)),(.3,.5,.7);n=int(np.prod(grid));torch.manual_seed(20261020)
    a=torch.randn((1,n,6,6),device="cuda",dtype=torch.float64)
    full=a.transpose(-1,-2)@a+2*torch.eye(6,device="cuda",dtype=torch.float64)
    contrast_active=None
    cb=None
    if kind=="hg" and variant=="contrast64":
        cb=np.zeros((6,6));cb[:3,:3]=2.3;cb[np.arange(3),np.arange(3)]+=3.4;cb[np.arange(3,6),np.arange(3,6)]=1.7
        contrast_active=np.sort(np.random.default_rng(20261020).choice(n,n//5,replace=False)).astype(np.int32)
        mixed=torch.as_tensor(cb,device="cuda").expand_as(full).clone()
        mixed[0,torch.as_tensor(contrast_active,dtype=torch.long,device="cuda")]=full[0,torch.as_tensor(contrast_active,dtype=torch.long,device="cuda")]
        full=mixed
    c=torch.stack([full[...,i,j] for i,j in b.PAIRS],dim=-1).contiguous()
    if kind=="hg":
        with formula.pipeline.lean_hg_initialization():op=b.make_solver(c,grid,voxel,fft=False).op
        prior.modal.attach(op,kind,voxel,variant="atomic",block=128)
        formula.pipeline.enable_analytic_reference(op);formula.pipeline.attach_energy(op)
    else:
        import atomic_adapter
        op=atomic_adapter.make(c,grid,"atomic128",voxel).op
        prior.modal.attach(op,kind,voxel,variant="atomic",block=128)
    op._validation_kind=kind
    matrix,rhs_cpu,u_cpu=previous.cpu_reference(b,op,full,voxel)
    v=torch.randn((1,3*n),device="cuda",dtype=torch.float64)
    row=dict(kind=kind,variant=variant,grid=grid,voxel=voxel)
    if kind=="fi":
        old=op.apply_precond(v)
        formula.enable_fi_reference(op)
        row["mixed_reference_relative"]=previous.relative(old.cpu().numpy(),op.apply_precond(v).cpu().numpy())
        from fi_reference import enable_reference
        op.__dict__.pop("apply_precond",None)
        enable_reference(op,"fp64","upper6");old=op.apply_precond(v)
        formula.enable_fi_reference(op,"fp64")
        row["fp64_reference_relative"]=previous.relative(old.cpu().numpy(),op.apply_precond(v).cpu().numpy())
        formula.enable_fi_reference(op,"mixed")
    elif variant=="contrast64":
        formula.compact_hg(op,contrast_active,cb)
        row["contrast_elements"]=len(contrast_active)
        b.solve_pcg_batched=formula.spectral_solver
        spectrum=torch.fft.rfftn(op.to_field(v),dim=(-3,-2,-1)).contiguous()
        actual,energy=op.spectral_apply(spectrum)
        physical=torch.fft.irfftn(op.spectral_project(spectrum),s=grid,dim=(-3,-2,-1),norm="backward")
        flat=op.to_flat(physical)[0].cpu().numpy()
        expected=matrix@flat
        rhs_spec=torch.fft.rfftn(op.to_field(torch.as_tensor(expected[None],device="cuda")),dim=(-3,-2,-1)).contiguous()
        expected_spec=op.spectral_project(rhs_spec)
        row["whitened_operator_relative"]=previous.relative(actual.cpu().numpy(),expected_spec.cpu().numpy())
        row["spectral_energy_relative"]=abs(float(energy)-n*float(flat@expected))/abs(n*float(flat@expected))
        residual=op.spectral_project(spectrum);dummy=torch.empty(0,device="cuda",dtype=torch.float64)
        parts=formula.extension().update(op._contrast_coeff,*op._contrast_tables,spectrum,residual,residual,residual,dummy,dummy,*formula.geometry(op),False)
        projected=op.project_zero_mean(v)
        norm_reference=float((projected*projected).sum())
        row["parseval_physical_norm_relative"]=abs(float(parts[1].sum())/n-norm_reference)/norm_reference
    elif variant=="single":b.solve_pcg_batched=formula.single_solver
    row["Ku_relative"]=previous.relative(op.apply(v)[0].cpu().numpy(),matrix@v[0].cpu().numpy())
    bg,_=make_B_matrices(*voxel,element_type="C3D8R" if kind=="hg" else "C3D8")
    ce=full[0].cpu().numpy();nodes=[]
    for e in range(n):
        z,rem=divmod(e,grid[1]*grid[2]);y,x=divmod(rem,grid[2])
        nodes.append([(((z+oz)%grid[0]*grid[1]+(y+oy)%grid[1])*grid[2]+(x+ox)%grid[2]) for ox,oy,oz in LOCAL_OFFSET])
    nodes=np.asarray(nodes)
    def fields(u,macro):
        strain=np.einsum("gij,ej->egi",bg,u[nodes].reshape(n,24))+macro
        return np.einsum("eij,egj->egi",ce,strain)
    loads=[];ch=[];ch_cpu=[]
    for j,macro in enumerate(np.eye(6)):
        rhs=op.project_zero_mean(op.compute_rhs(macro));u,report=b.solve(op,rhs)
        un=u[0].cpu().numpy().reshape(n,3);stress=fields(un,macro);ref=fields(u_cpu[:,:,j],macro)
        column=op.compute_avg_stress(u,macro)[0];ch.append(column);ch_cpu.append(ref.mean(axis=(0,1)))
        loads.append(dict(**report,rhs_relative=previous.relative(rhs[0].cpu().numpy(),rhs_cpu[:,j]),
            independent_true_residual=float(np.linalg.norm(rhs_cpu[:,j]-matrix@un.ravel())/np.linalg.norm(rhs_cpu[:,j])),
            displacement_relative=previous.relative(un,u_cpu[:,:,j]),local_stress_relative=previous.relative(stress,ref),
            recovery_relative=previous.relative(column,stress.mean(axis=(0,1)))))
    row.update(loads=loads,CH_relative=previous.relative(np.column_stack(ch),np.column_stack(ch_cpu)),provenance=provenance(b))
    row["passed"]=(row["Ku_relative"]<1e-12 and row.get("mixed_reference_relative",0)<2e-5
        and row.get("fp64_reference_relative",0)<1e-11 and row.get("whitened_operator_relative",0)<1e-12
        and row.get("spectral_energy_relative",0)<1e-12 and row.get("parseval_physical_norm_relative",0)<1e-12
        and row["CH_relative"]<1e-8 and all(x["independent_true_residual"]<=1.02e-10 and x["local_stress_relative"]<1e-8 and x["rhs_relative"]<1e-12 for x in loads))
    save(HERE/"results"/f"validate_{kind}_{variant}{'_even' if even else ''}.json",row)
    print("VALIDATE",kind,variant,row["passed"],row["Ku_relative"],row["CH_relative"],flush=True)
    if not row["passed"]:raise RuntimeError("GPU formula validation failed")


def measure(kind,variant,grid,repeat,pilot=False):
    b=previous.load_runtime(kind);b.gpu_guard();formula.extension();prior.modal.extension();formula.pipeline.extension()
    formula.configure(b,kind,variant)
    data,geom,material,digest=b.inputs(grid,"strong")
    out=HERE/("pilot" if pilot else "results")/f"measure_{kind}_{variant}_{grid}_{repeat}.json"
    if out.exists():raise FileExistsError(out)
    record=dict(kind=kind,variant=variant,grid=grid,repeat=repeat,pilot=pilot,input_hash=digest,
        geometry=geom,material=material,tolerance=b.TOL,provenance=provenance(b),
        protocol="fresh process; one complete six-load same-size warmup; complete GPU material/operator/reference/RHS/solve/true-residual/stress timing; CPU geometry/disk/JIT/warmup excluded; baseline stage synchronizations retained",
        solver_protocol="baseline frozen latched checks; new HG solvers use checks every four actual steps, report actual iterations and certify projected physical FP64 residual with restart if needed")
    try:
        record["phase"]="warmup";save(out,record);b.workload(data,material,"direct","mixed")
        b.gpu_guard();record["phase"]="measurement";save(out,record)
        result,_=b.workload(data,material,"direct","mixed");b.gpu_guard()
        if max(v for c in result["calls"] for v in c["true_residuals"])>1e-8:raise RuntimeError("Formal residual failed")
        record.update(status="passed",measurement=result)
        print("MEASURE",kind,variant,repeat,result["seconds"],flush=True)
    except BaseException as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc());raise
    finally:save(out,record)


def orchestrate(grid):
    path=HERE/"status.json"
    if path.exists():raise FileExistsError(path)
    state=dict(status="running",grid=grid,started_utc=datetime.now(timezone.utc).isoformat(),completed=[])
    save(path,state)
    def run(phase,kind,variant,repeat=0):
        if previous.foreign():raise RuntimeError("GPU occupied: "+str(previous.foreign()))
        label=f"{phase}_{kind}_{variant}_{repeat}";state["active"]=label;save(path,state)
        command=[sys.executable,"-u",str(HERE/"formula_study.py"),"--phase",phase,"--kind",kind,"--variant",variant,"--grid",str(grid),"--repeat",str(repeat)]
        samples=[]
        with (HERE/"logs"/(label+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(2)
                others=previous.foreign(child.pid)
                if others:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
                    raise RuntimeError("Stopped our job because GPU became occupied: "+str(others))
                raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
                samples.append(dict(time=time.time(),gpu=raw))
        save(HERE/"logs"/(label+"_telemetry.json"),samples)
        if child.returncode:raise RuntimeError(label+" failed; inspect log")
        state["completed"].append(label);save(path,state);print("DONE",label,flush=True)
    try:
        for kind,variant in (("fi","analytic"),("hg","contrast64"),("hg","single")):
            validation=HERE/"results"/f"validate_{kind}_{variant}.json"
            fresh=False
            if validation.exists():
                earlier=json.loads(validation.read_text())
                old=earlier.get("provenance",{}).get("formula_source_sha256",{})
                fresh=earlier.get("passed") and all(old.get(str(HERE/name))==hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("formula_adapter.py","formula_kernels.cu"))
            if not fresh:
                run("validate",kind,variant)
        configs=[("hg","baseline"),("hg","contrast64"),("hg","single"),("fi","baseline"),("fi","analytic")]
        for kind,variant in configs:run("pilot",kind,variant)
        for repeat in range(3):
            jobs=configs.copy();random.Random(20261020+repeat).shuffle(jobs)
            for kind,variant in jobs:run("measure",kind,variant,repeat)
        state.update(status="complete",ended_utc=datetime.now(timezone.utc).isoformat());state.pop("active",None);save(path,state)
    except BaseException as error:
        state.update(status="stopped",error=repr(error));save(path,state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("validate","pilot","measure","orchestrate"),required=True)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--variant",choices=("baseline","analytic","contrast64","single"),default="baseline")
    parser.add_argument("--grid",type=int,default=192);parser.add_argument("--repeat",type=int,default=0)
    parser.add_argument("--even",action="store_true",help="include an even x size in small validation")
    args=parser.parse_args()
    for name in ("results","pilot","logs"):(HERE/name).mkdir(exist_ok=True)
    if args.phase=="validate":validate(args.kind,args.variant,args.even)
    elif args.phase=="orchestrate":orchestrate(args.grid)
    else:measure(args.kind,args.variant,args.grid,args.repeat,args.phase=="pilot")


if __name__=="__main__":main()
