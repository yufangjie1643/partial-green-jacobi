"""Formal test of the ordered-soft-phase closed-form reference."""
import argparse
import json
import random
import subprocess
import sys
import time

import chain_study as cs
import reference_study as rs
from reference_closed_form import design_closed


def install(b, kind):
    rs.design_reference = design_closed
    rs.install(b, kind)


def measure(kind, mode, grid, case, repeat, stage):
    path = cs.OUT/f"closed_{stage}_{kind}_{mode}_{case}_{grid}_{repeat}.json"
    if path.exists():
        raise FileExistsError(path)
    b = cs.s.load(kind)
    cs.a.native.extension()
    if mode == "closed":
        install(b, kind)
    variant = "baseline" if mode == "prior" else ("compact_combined" if kind == "hg" else "combined")
    cs.a.configure(b, kind, variant)
    data, geometry, material, digest = b.inputs(grid, case)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    result, _ = b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    assert max(v for call in result["calls"] for v in call["true_residuals"]) <= b.TOL
    cs.save(path, dict(complete=True, kind=kind, variant=variant, reference_mode=mode,
        grid=grid, case=case, repeat=repeat, stage=stage, input_hash=digest, geometry=geometry,
        material=material, measurement=result, reference_design=getattr(b, "_reference_design", None),
        provenance=cs.provenance(b), tolerance=b.TOL,
        protocol="fresh process; full same-grid warmup; all setup, closed-form selection, six loads and true residual checks timed"))
    print(kind, mode, grid, case, repeat, result["seconds"],
          sum(c["iterations"] for c in result["calls"]), flush=True)


def orchestrate():
    def run(*args):
        cs.s.guard()
        with (cs.OUT/("closed_"+"_".join(map(str, args))+".log")).open("x") as log:
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
                raise RuntimeError(f"Closed-reference child failed {args}")
        print("DONE", *args, flush=True)
    for kind in ("hg", "fi"):
        run("--kind", kind, "--mode", "closed", "--grid", 8, "--stage", "small")
    for repeat in range(3):
        pairs = [(kind, mode) for kind in ("hg", "fi") for mode in ("prior", "baseline", "closed")]
        random.Random(922+repeat).shuffle(pairs)
        for kind, mode in pairs:
            run("--kind", kind, "--mode", mode, "--repeat", repeat, "--stage", "formal")
    for kind in ("hg", "fi"):
        for mode in ("baseline", "closed"):
            run("--kind", kind, "--mode", mode, "--grid", 128, "--case", "near", "--stage", "smoke")
    cs.save(cs.OUT/"closed_complete.json", dict(complete=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("measure", "orchestrate"), default="measure")
    p.add_argument("--kind", choices=("hg", "fi"), default="hg")
    p.add_argument("--mode", choices=("prior", "baseline", "closed"), default="closed")
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
