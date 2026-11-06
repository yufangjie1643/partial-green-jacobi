"""Does optimizing a material spectral bound reduce complete-workflow cost?"""
import argparse
import json
import subprocess
import sys
import time
from types import MethodType

import chain_study as cs
from reference_design import design_reference


def install(b, kind):
    old = b.build
    def build(data, material, route, precision, stages):
        op = old(data, material, route, precision, stages)
        from femlib.materials import engineering_to_voigt, isotropic_to_voigt
        m, y = material["matrix"], material["yarn"]
        phases = [isotropic_to_voigt(m["E"], m["Nu"]),
                  engineering_to_voigt(*(y[k] for k in
                    ("E11", "E22", "E33", "Nu12", "Nu13", "Nu23", "G12", "G13", "G23")))]
        base = op._analytic_reference_C0 if kind == "hg" else op._formula_reference_C0
        reference, metadata = design_reference(phases, base, kind)
        name = "_compute_reference_medium_from_packed" if kind == "hg" else "_compute_reference_medium"
        previous = getattr(op, name)
        setattr(op, name, MethodType(lambda self, *args, **kwargs: reference[None], op))
        try:
            if kind == "hg":
                cs.s.f.formula.pipeline.enable_analytic_reference(op)
            else:
                cs.s.f.formula.enable_fi_reference(op)
        finally:
            setattr(op, name, previous)
        b._reference_design = metadata
        stages.end("spectral_reference_design", op)
        return op
    b.build = build


def measure(kind, mode, grid, case, repeat, stage):
    path = cs.OUT/f"ref_{stage}_{kind}_{mode}_{case}_{grid}_{repeat}.json"
    if path.exists():
        raise FileExistsError(path)
    b = cs.s.load(kind)
    cs.a.native.extension()
    if mode == "optimized":
        install(b, kind)
    variant = "compact_combined" if kind == "hg" else "combined"
    cs.a.configure(b, kind, variant)
    data, geometry, material, digest = b.inputs(grid, case)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    result, _ = b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    assert max(v for call in result["calls"] for v in call["true_residuals"]) <= b.TOL
    cs.save(path, dict(complete=True, kind=kind, variant=variant, reference_mode=mode,
        grid=grid, case=case, repeat=repeat, stage=stage, input_hash=digest, material=material,
        geometry=geometry, measurement=result, reference_design=getattr(b, "_reference_design", None),
        provenance=cs.provenance(b), tolerance=b.TOL,
        protocol="fresh process; full warmup; full six-load solve including material spectral optimization and reference rebuilding; same physical K and true tolerance"))
    print(kind, mode, grid, case, result["seconds"],
          sum(c["iterations"] for c in result["calls"]), flush=True)


def orchestrate():
    def run(*args):
        cs.s.guard()
        with (cs.OUT/("ref_"+"_".join(map(str, args))+".log")).open("x") as log:
            p = subprocess.Popen([sys.executable, "-u", __file__, *map(str, args)],
                                 stdout=log, stderr=subprocess.STDOUT)
            try:
                while p.poll() is None:
                    time.sleep(2)
                    cs.s.guard((p.pid,))
            except BaseException:
                p.terminate()
                p.wait()
                raise
            if p.returncode:
                raise RuntimeError(f"Reference child failed {args}")
        print("DONE", *args, flush=True)
    for kind in ("hg", "fi"):
        for mode in ("baseline", "optimized"):
            run("--kind", kind, "--mode", mode, "--grid", 8, "--stage", "small")
    for kind in ("hg", "fi"):
        for grid, case in ((192, "strong"), (128, "near")):
            for mode in ("baseline", "optimized"):
                run("--kind", kind, "--mode", mode, "--grid", grid, "--case", case)
    cs.save(cs.OUT/"reference_complete.json", dict(complete=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("measure", "orchestrate"), default="measure")
    p.add_argument("--kind", choices=("hg", "fi"), default="hg")
    p.add_argument("--mode", choices=("baseline", "optimized"), default="optimized")
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--case", choices=("strong", "near"), default="strong")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--stage", default="tune")
    args = p.parse_args()
    cs.OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "orchestrate":
        orchestrate()
    else:
        measure(args.kind, args.mode, args.grid, args.case, args.repeat, args.stage)
