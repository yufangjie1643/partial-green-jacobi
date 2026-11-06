"""Latest general-C21 FI chain on the exact phase images used by chfem.

Examples:
  python our_study.py --phase measure --input cases/sphere192.npy \
      --metadata cases/sphere192.json --output results/our_sphere192_0.json
  python our_study.py --phase export --input cases/sphere8.npy \
      --metadata cases/sphere8.json --tol 1e-10 \
      --fields results/our_sphere8_fields.npz --output results/our_sphere8.json

Input/metadata IO and CPU input preparation occur before the timed workload.
The existing workload performs GPU setup, all six loads, true-residual checks,
and stress recovery. It is called once for same-size warmup and once to measure.
The export phase is a separate, untimed diagnostic with six displacement fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
CHAIN = Path(os.environ.get("FEAD_CHAIN", str(__import__("project_paths").source_path("solver_chain"))))
FORMAL_TOLERANCE = 1.0e-8
EXPORT_MAX_ELEMENTS = 32768
VOIGT_ORDER = ("11", "22", "33", "23", "13", "12")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, value):
    """Update this process's exclusively reserved result; no IO in timed work."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_inputs(input_path, metadata_path=None):
    """Translate binary isotropic phases into the frozen general-C21 API."""
    import numpy as np
    import cases

    input_path = Path(input_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else input_path.with_suffix(".json")
    phase, meta = cases.load_case(input_path, metadata_path)
    phase = np.asarray(phase)
    if phase.dtype != np.uint8 or phase.ndim != 3 or min(phase.shape) < 2:
        raise ValueError("Input must be uint8 (Nz,Ny,Nx), with each dimension >=2")
    phase = np.ascontiguousarray(phase)
    actual_array_hash = hashlib.sha256(phase.tobytes()).hexdigest()
    actual_npy_hash = _sha_file(input_path)
    for key, actual in (("array_sha256", actual_array_hash), ("npy_sha256", actual_npy_hash)):
        if key in meta and meta[key] != actual:
            raise ValueError(f"Metadata {key} does not match the original input")

    specifications = meta["phase_materials"]
    if len(specifications) != 2:
        raise ValueError("Exactly two isotropic phase materials are required")
    phases = []
    for index, entry in enumerate(specifications):
        E, nu = float(entry["E"]), float(entry["nu"])
        if not (math.isfinite(E) and E > 0 and math.isfinite(nu) and -1 < nu < 0.5):
            raise ValueError(f"Phase {index} is not stable isotropic elasticity")
        phases.append({"E": E, "nu": nu})
    voxel = np.asarray(meta["voxel_size_xyz"], dtype=np.float64)
    if voxel.shape != (3,) or not np.isfinite(voxel).all() or not (voxel > 0).all():
        raise ValueError("voxel_size_xyz must contain three finite positive lengths")

    nz, ny, nx = map(int, phase.shape)
    inclusion = phase != 0
    yarn_id = np.where(inclusion, 0, -1).astype(np.int8)[None]
    # Constant axes are represented by CPU broadcast views. The unchanged
    # construct() still builds and rotates general C21 tensors on the GPU.
    orientation_shape = (1, nz, ny, nx, 3)
    data = {
        "grid_shape": (nz, ny, nx),
        "domain_size": (voxel * np.array([nx, ny, nz], dtype=np.float64))[None],
        "yarn_id": yarn_id,
        "orient1": np.broadcast_to(np.array([1., 0., 0.]), orientation_shape),
        "orient2": np.broadcast_to(np.array([0., 1., 0.]), orientation_shape),
        "indices": np.array([0], dtype=np.int64),
    }
    E1, nu1 = phases[1]["E"], phases[1]["nu"]
    shear1 = E1 / (2 * (1 + nu1))
    material = {
        "matrix": {"E": phases[0]["E"], "Nu": phases[0]["nu"]},
        "yarn": {
            "E11": E1, "E22": E1, "E33": E1,
            "Nu12": nu1, "Nu13": nu1, "Nu23": nu1,
            "G12": shear1, "G13": shear1, "G23": shear1,
        },
    }
    phase_counts = [int((~inclusion).sum()), int(inclusion.sum())]
    definition = {
        "grid_shape_zyx": [nz, ny, nx],
        "voxel_size_xyz": voxel.tolist(),
        "phase_materials": phases,
        "phase_mapping": "raw value 0 -> matrix; every nonzero value -> inclusion",
        "array_sha256": actual_array_hash,
    }
    problem_hash = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    geometry = {
        "case": meta.get("case", Path(input_path).stem),
        "grid_shape_zyx": [nz, ny, nx], "elements": int(phase.size),
        "voxel_size_xyz": voxel.tolist(),
        "domain_size_xyz": data["domain_size"][0].tolist(),
        "phase_counts": phase_counts,
        "inclusion_fraction": phase_counts[1] / int(phase.size),
        "raw_phase_values": np.unique(phase).tolist(),
        "source_info": meta.get("source_info"),
    }
    audit = {
        "input_path": str(Path(input_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "array_sha256": actual_array_hash, "npy_sha256": actual_npy_hash,
        "problem_sha256": problem_hash,
        "problem_definition": definition, "source_metadata": meta,
    }
    audit["metadata_sha256"] = _sha_file(metadata_path)
    return phase, data, geometry, material, problem_hash, audit


def setup(tolerance):
    import torch

    # Set this before any frozen runtime import: do not inherit an unrelated
    # experiment's stopping criterion or its historical default architecture.
    os.environ["FEAD_TOL"] = format(tolerance, ".17g")
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    sys.path.insert(0, str(CHAIN))
    import chain_study as cs
    import closed_study as closed

    b = cs.s.load("fi")
    b.TOL = tolerance
    closed.install(b, "fi")
    cs.a.configure(b, "fi", "combined")
    b.gpu_guard()
    cs.a.native.extension()
    return b, cs


def provenance(b, cs):
    result = cs.provenance(b)
    result["comparison_source_sha256"] = {
        str(path): _sha_file(path)
        for path in sorted(HERE.iterdir()) if path.suffix in (".py", ".cu", ".cpp")
    }
    result["gpu_query"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader"], text=True,
    ).strip()
    result["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    result["fead_chain"] = str(CHAIN)
    return result


def timing_breakdown(measurement):
    groups = {"setup": 0.0, "rhs": 0.0, "solve": 0.0, "post": 0.0}
    for stage in measurement["stages"]:
        name = stage["name"]
        group = next((g for g in ("rhs", "solve", "post") if name.startswith(g)), "setup")
        groups[group] += float(stage["seconds"])
    groups = {name + "_seconds": value for name, value in groups.items()}
    groups["total_seconds"] = float(measurement["seconds"])
    groups["total_minus_stage_sum_seconds"] = (
        float(measurement["seconds"]) - sum(groups[g + "_seconds"] for g in ("setup", "rhs", "solve", "post"))
    )
    return groups


def check_calls(calls, tolerance):
    """Check six final B=1 certificates without altering solver diagnostics.

    Frozen benchmark.solve retries internally and returns true_residuals only
    after convergence. Its entries index the batch, not restart attempts.
    The pipeline wrapper changes solve_pcg_batched, not this return contract.
    """
    if len(calls) != 6:
        raise RuntimeError("Expected exactly six macroscopic load cases")
    residuals = []
    for load, call in enumerate(calls):
        final_batch = call["true_residuals"]
        if len(final_batch) != 1:
            raise RuntimeError(
                f"Load {load}: expected one final batch residual (B=1), "
                f"received {len(final_batch)}"
            )
        residual = float(final_batch[0])
        if not math.isfinite(residual) or not 0 <= residual <= tolerance:
            raise RuntimeError(
                f"Load {load}: true residual certification failed: {residual}"
            )
        residuals.append(residual)
    return max(residuals)


def measure(b, data, material, record, output):
    record["attempt_phase"] = "warmup"
    save(output, record)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard()
    record["attempt_phase"] = "measurement"
    save(output, record)
    print("MEASUREMENT_BEGIN", flush=True)
    wrapper_start = time.perf_counter()
    measured, _ = b.workload(data, material, "direct", "mixed")
    wrapper_elapsed = time.perf_counter() - wrapper_start
    print("MEASUREMENT_END", flush=True)
    b.gpu_guard()
    maximum = check_calls(measured["calls"], b.TOL)
    if not b.np.isfinite(b.np.asarray(measured["CH"])).all():
        raise RuntimeError("Nonfinite homogenized stiffness")
    record.update(
        measurement=measured, timing_breakdown=timing_breakdown(measured),
        workload_with_cleanup_seconds=wrapper_elapsed,
        max_true_residual=maximum,
        reference_design=getattr(b, "_reference_design", None),
    )


def export_fields(b, phase, data, material, record, fields_path):
    """Untimed six-load diagnostic; transfers never enter a formal measurement."""
    import numpy as np

    if phase.size > EXPORT_MAX_ELEMENTS:
        raise ValueError(f"Export is limited to small grids with N<={EXPORT_MAX_ELEMENTS}")
    fields_path = Path(fields_path)
    fields_path.parent.mkdir(parents=True, exist_ok=True)
    if fields_path.exists():
        raise FileExistsError(fields_path)
    b.gpu_guard()
    b.cleanup()
    op = b.build(data, material, "direct", "mixed", b.Stages())
    U, RHS, calls = [], [], []
    ch = np.empty((6, 6), dtype=np.float64)
    for j, macro in enumerate(np.eye(6)):
        rhs = op.project_zero_mean(op.compute_rhs(macro))
        u, call = b.solve(op, rhs)
        ch[:, j] = np.asarray(op.compute_avg_stress(u, macro)[0])
        U.append(u.detach().cpu().numpy().reshape(-1, 3).copy())
        RHS.append(rhs.detach().cpu().numpy().reshape(-1, 3).copy())
        calls.append(call)
        del rhs, u
    maximum = check_calls(calls, b.TOL)
    if not np.isfinite(ch).all():
        raise RuntimeError("Nonfinite homogenized stiffness")
    b.synchronize()
    b.gpu_guard()
    with fields_path.open("xb") as stream:
        np.savez(
            stream, U=np.stack(U), RHS=np.stack(RHS), CH=ch,
            macro_strains=np.eye(6), phase=phase,
            grid_shape_zyx=np.array(phase.shape, dtype=np.int64),
            voxel_size_xyz=np.array(record["geometry"]["voxel_size_xyz"]),
            voigt_order=np.array(VOIGT_ORDER),
        )
    record.update(
        fields_path=str(fields_path.resolve()), fields_sha256=_sha_file(fields_path),
        field_schema={
            "U": "(load,N,3), zero-mean periodic displacement; x fastest",
            "RHS": "(load,N,3), projected forcing; x fastest",
            "CH": "(6,6), columns correspond to unit engineering-strain loads",
            "phase": "(Nz,Ny,Nx), original uint8 encoding unchanged",
        },
        CH=ch.tolist(), calls=calls, max_true_residual=maximum,
        reference_design=getattr(b, "_reference_design", None),
    )
    del op
    b.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("measure", "export"), default="measure")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--tol", type=float, default=FORMAL_TOLERANCE)
    args = parser.parse_args()
    if not math.isfinite(args.tol) or not 0 < args.tol < 1:
        parser.error("--tol must be finite and between zero and one")
    if args.phase == "measure" and args.tol != FORMAL_TOLERANCE:
        parser.error("Formal comparison requires --tol 1e-8")
    if (args.phase == "export") != (args.fields is not None):
        parser.error("--fields is required only for --phase export")
    if args.fields is not None and args.fields.exists():
        raise FileExistsError(args.fields)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "complete": False, "status": "running", "phase": args.phase,
        "kind": "fi", "variant": "latest_general_C21_closed_reference_combined",
        "repeat": args.repeat, "tolerance": args.tol,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "CH_convention": "engineering Voigt (11,22,33,23,13,12); unsymmetrized six columns",
        "residual_definition": "||f-K Pi u||_2/||f||_2, with FP64 physical K and zero-mean displacement projection; zero RHS returns zero displacement",
        "protocol": (
            "fresh process; full same-grid six-load warmup; primary "
            "workload_with_cleanup_seconds surrounds the complete frozen workload, "
            "including initial/final cleanup, result assembly, internal CPU material "
            "processing, GPU C21 construction, operator/reference setup, six RHS/solves, "
            "true-residual certification, stress recovery and stage instrumentation; "
            "unchanged measurement.seconds is the auxiliary inner time excluding "
            "initial/final cleanup and result assembly; external CPU input preparation, "
            "IO, JIT, warmup, GPU guards and measurement marker prints are excluded "
            "from both times; peak allocated/reserved include explicit workspaces "
            "but are allocator measurements, not total driver memory"
            if args.phase == "measure" else
            "small-grid correctness diagnostic only; six-load displacement/RHS export; "
            "no performance result and no timing comparison with the formal workload"
        ),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    try:
        record["attempt_phase"] = "input"
        phase, data, geometry, material, digest, audit = prepare_inputs(
            args.input, args.metadata)
        record.update(
            geometry=geometry, material=material, input_hash=digest,
            array_sha256=audit["array_sha256"], npy_sha256=audit["npy_sha256"],
            input_audit=audit,
        )
        if args.phase == "export" and phase.size > EXPORT_MAX_ELEMENTS:
            raise ValueError(f"Export is limited to N<={EXPORT_MAX_ELEMENTS}")
        record["attempt_phase"] = "runtime_setup"
        b, cs = setup(args.tol)

        def supplied_inputs(grid=None, case=None):
            return data, geometry, material, digest

        b.inputs = supplied_inputs
        record["provenance"] = provenance(b, cs)
        save(output, record)
        data, geometry, material, digest = b.inputs()
        if args.phase == "measure":
            measure(b, data, material, record, output)
        else:
            record["attempt_phase"] = "export"
            export_fields(b, phase, data, material, record, args.fields)
        record.update(complete=True, status="passed")
        record.pop("attempt_phase", None)
    except BaseException as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(output, record)
    print("RESULT", args.phase, geometry["case"], record["status"],
          record.get("measurement", {}).get("seconds"),
          record["max_true_residual"], flush=True)


if __name__ == "__main__":
    main()
