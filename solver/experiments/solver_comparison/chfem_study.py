"""Run the isolated, instrumented chfem FI baseline on a shared uint8 input.

The original PCG kernels and native Jacobi stopping rule are retained. A new
post-solve observer recomputes the Euclidean residual with the native physical
operator. This script must be launched by the experiment's serial GPU runner.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = __import__("project_paths").ROOT / "baselines/chfem"
ORIGINAL_SOURCE = __import__("project_paths").ROOT / "baselines/chfem_reference"
UPSTREAM_COMMIT = "309c1f64a90760ada5c45fa61ffb013d1b1a0271"
TRUE_TOLERANCE = 1.0e-8


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source):
    paths = [Path(__file__).resolve(), HERE / "cases.py"]
    paths.extend(path for path in (source / "chfem").rglob("*")
                 if path.suffix in (".py", ".c", ".cu", ".h", ".so"))
    paths.extend(path for path in
                 (source / "setup.py",
                  source.parent / "chfem_instrumentation.patch",
                  source.parent / "chfem_instrumentation_manifest.json")
                 if path.is_file())
    pcg_relative = Path("chfem/src/femhmg/cudapcg/solvers/pcg.cu")
    result = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source": str(source),
        "original_build": str(ORIGINAL_SOURCE),
        "source_sha256": {str(path): sha(path) for path in paths},
        "pcg_kernel_sha256": sha(source / pcg_relative),
    }
    if (ORIGINAL_SOURCE / pcg_relative).is_file():
        result["original_pcg_kernel_sha256"] = sha(ORIGINAL_SOURCE / pcg_relative)
        result["pcg_kernel_byte_identical"] = (
            result["pcg_kernel_sha256"] == result["original_pcg_kernel_sha256"])
        if not result["pcg_kernel_byte_identical"]:
            raise RuntimeError("The instrumented copy changed the baseline PCG kernel")
    return result


def _parse_observations(log, native_tolerance):
    loads, totals = [], []
    for line in log.splitlines():
        if not line.startswith(("FEAD_AUDIT_LOAD ", "FEAD_AUDIT_TOTAL ")):
            continue
        row = {}
        for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", line):
            parsed = float(value)
            row[key] = int(parsed) if key in (
                "load", "iterations", "native_found", "scalar_bytes",
                "device_used_baseline_bytes", "device_used_sampled_peak_bytes"
            ) else parsed
        (loads if line.startswith("FEAD_AUDIT_LOAD ") else totals).append(row)
    if len(loads) != 6 or sorted(row["load"] for row in loads) != list(range(6)):
        raise RuntimeError("Expected exactly six native post-solve audit records")
    if len(totals) != 1 or totals[0].get("scalar_bytes") != 8:
        raise RuntimeError("Missing unique FP64 native timing record")
    loads.sort(key=lambda row: row["load"])
    for row in loads:
        row["native_converged"] = bool(
            math.isfinite(row["native_residual"])
            and row["native_residual"] <= native_tolerance
            and (row["native_found"] or row["rhs_norm"] == 0.0))
        row["true_residual_passed"] = bool(
            math.isfinite(row["true_residual"])
            and row["true_residual"] <= TRUE_TOLERANCE)
    return loads, totals[0]


def _capture_call(compute_property, phase, materials, voxel, native_tolerance,
                  max_iterations, export_prefix=None):
    """Capture native stdout in RAM, writing persistent logs after the timer."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    if hasattr(os, "memfd_create"):
        memory_fd = os.memfd_create("chfem-native-log")
    elif hasattr(libc, "memfd_create"):
        # Some standalone Python builds omit os.memfd_create on Linux.
        libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc.memfd_create.restype = ctypes.c_int
        memory_fd = libc.memfd_create(b"chfem-native-log", 1)
        if memory_fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    else:
        raise RuntimeError("The controlled runner requires Linux memfd_create")
    previous = os.environ.get("CHFEM_AUDIT_EXPORT_PREFIX")
    if export_prefix is None:
        os.environ.pop("CHFEM_AUDIT_EXPORT_PREFIX", None)
    else:
        os.environ["CHFEM_AUDIT_EXPORT_PREFIX"] = str(export_prefix)
    sys.stdout.flush()
    libc.fflush(None)
    saved_stdout = os.dup(1)
    caught = None
    try:
        os.dup2(memory_fd, 1)
        start = time.perf_counter()
        ch = compute_property(
            "elasticity", phase, mat_props=materials, voxel_size=voxel,
            solver="cg", solver_tolerance=native_tolerance,
            solver_maxiter=max_iterations, precondition=True,
            type_of_rhs=0, direction="all")
        api_seconds = time.perf_counter() - start
        sys.stdout.flush()
        libc.fflush(None)
    except BaseException as error:
        caught = error
    finally:
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        if previous is None:
            os.environ.pop("CHFEM_AUDIT_EXPORT_PREFIX", None)
        else:
            os.environ["CHFEM_AUDIT_EXPORT_PREFIX"] = previous
    os.lseek(memory_fd, 0, os.SEEK_SET)
    with os.fdopen(memory_fd, "rb") as stream:
        log = stream.read().decode("utf-8", errors="replace")
    if caught is not None:
        caught.native_log = log
        raise caught
    try:
        loads, native = _parse_observations(log, native_tolerance)
    except BaseException as error:
        error.native_log = log
        raise
    import numpy as np
    ch = np.asarray(ch, dtype=np.float64)
    if ch.shape != (6, 6) or not np.isfinite(ch).all():
        raise RuntimeError("Invalid effective elasticity matrix from chfem")
    sign = np.array([1., 1., 1., 1., -1., -1.])
    result = {
        "api_seconds": api_seconds,
        "core_seconds": native["core_seconds"],
        "native_timing": native,
        "CH_raw": ch.tolist(),
        "CH_project_convention": (sign[:, None] * ch * sign[None, :]).tolist(),
        "voigt_alignment_sign": sign.tolist(),
        "loads": loads,
        "iterations": [row["iterations"] for row in loads],
        "true_residuals": [row["true_residual"] for row in loads],
        "native_residuals": [row["native_residual"] for row in loads],
        "native_converged": all(row["native_converged"] for row in loads),
        "passed": all(row["native_converged"] and row["true_residual_passed"]
                      for row in loads),
    }
    return result, log


def _export_npz(path, prefix, phase, metadata, measurement):
    import numpy as np
    n = phase.size
    raw_files = []
    values = {}
    for field in ("u", "rhs"):
        rows = []
        for load in range(6):
            raw = Path(f"{prefix}_{field}_{load}.bin")
            data = np.fromfile(raw, dtype=np.float64)
            if data.size != 3 * n:
                raise RuntimeError(f"Wrong native field length: {raw}")
            rows.append(data.reshape(n, 3))
            raw_files.append({"path": str(raw), "sha256": sha(raw),
                              "bytes": raw.stat().st_size})
        values[field + "_raw"] = np.stack(rows)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, **values, phase_zyx=phase,
            grid_shape_zyx=np.asarray(phase.shape),
            voxel_size_xyz=np.asarray(metadata["voxel_size_xyz"]),
            phase_materials=np.asarray(
                [[item["E"], item["nu"]] for item in metadata["phase_materials"]]),
            CH_raw=np.asarray(measurement["CH_raw"]),
            CH_project_convention=np.asarray(measurement["CH_project_convention"]),
            voigt_alignment_sign=np.asarray(measurement["voigt_alignment_sign"]),
            true_residuals=np.asarray(measurement["true_residuals"]),
            native_residuals=np.asarray(measurement["native_residuals"]))
    return {
        "npz": str(path), "npz_sha256": sha(path), "raw_files": raw_files,
        "ordering": ("u_raw and rhs_raw have shape (load,native_periodic_node,3). "
                     "Native displacement coordinates have not been converted."),
    }


def _safe_json(value):
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def run(args):
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    from cases import load_case
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = args.fields.resolve() if args.fields else None
    warm_log = output.with_suffix(".warmup.native.log")
    native_log = output.with_suffix(".native.log")
    for path in (output, warm_log, native_log, fields):
        if path is not None and path.exists():
            raise FileExistsError(path)
    if args.phase == "export" and fields is None:
        raise ValueError("--phase export requires --fields")
    prefix = fields.with_suffix(".native") if fields is not None else None
    if fields is not None:
        fields.parent.mkdir(parents=True, exist_ok=True)
        for field in ("u", "rhs"):
            for load in range(6):
                path = Path(f"{prefix}_{field}_{load}.bin")
                if path.exists():
                    raise FileExistsError(path)
    report = {
        "status": "running", "passed": False, "method": "chfem",
        "phase": args.phase, "repeat": args.repeat,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "native_tolerance": args.native_tol,
        "acceptance_true_tolerance": TRUE_TOLERANCE,
        "native_residual_definition": "sqrt((r^T J^-1 r)/(f^T J^-1 f)); recursive r",
        "certification_definition": "||f-K_native u||_2/||f||_2; recomputed FP64, no residual projection",
        "preconditioner": "Unchanged native default Jacobi",
        "initial_guess": "Zero for every load; no coarse initial guess",
        "element": "Native fully integrated periodic HEX8",
        "max_iterations": args.max_iterations,
        "threads": {"OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                    "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"]},
        "timing_scope": {
            "core_seconds": ("Native wrapper model setup, GPU preparation, all six "
                             "loads, true-residual certification, recovery and "
                             "cleanup; subtract neutral-file read/parse and "
                             "optional raw-field export."),
            "api_seconds": ("Complete compute_property call from resident uint8 "
                            "input, including temporary neutral-file generation "
                            "and IO. Persistent log/NPZ writes are outside."),
            "warmup": "One complete six-load solve at the same grid and tolerance",
            "native_logging": "Original native progress preserved; stdout captured in RAM",
        },
        "memory_scope": {
            "primary": "External common NVML sampler during MEASUREMENT_BEGIN/END",
            "native_device_samples": ("cudaMemGetInfo snapshots at start and after "
                                      "each load; whole-device committed use, not "
                                      "an allocator peak or exact temporal peak."),
            "comparison_rule": ("Do not compare native estimates or NVML bytes "
                                "against Torch allocated bytes as if identical."),
        },
    }
    with output.open("x", encoding="utf-8") as stream:
        try:
            phase, metadata = load_case(args.input, args.metadata)
            report["input_metadata"] = metadata
            source = args.source.resolve()
            report["source_identity"] = source_identity(source)
            sys.path.insert(0, str(source))
            chfem = importlib.import_module("chfem")
            wrapper = importlib.import_module("chfem.wrapper")
            if not Path(wrapper.__file__).resolve().is_relative_to(source):
                raise RuntimeError("Imported chfem binary is outside the isolated source")
            # Resolve, but do not call, the instrumentation symbol.
            getattr(ctypes.CDLL(wrapper.__file__), "feadAuditBegin")
            report["binary"] = str(Path(wrapper.__file__).resolve())
            report["python"] = sys.executable
            import numpy as np
            report["numpy"] = np.__version__
            voxel = tuple(float(value) for value in metadata["voxel_size_xyz"])
            if len(set(voxel)) != 1:
                raise ValueError("Native chfem comparison requires cubic voxels")
            materials = {label: (float(item["E"]), float(item["nu"]))
                         for label, item in zip((0, 255), metadata["phase_materials"])}
            report["active_stage"] = "warmup"
            print("WARMUP_BEGIN", flush=True)
            try:
                warmup, log = _capture_call(
                    chfem.compute_property, phase, materials, voxel[0],
                    args.native_tol, args.max_iterations)
            finally:
                print("WARMUP_END", flush=True)
            report["warmup"] = warmup
            with warm_log.open("x", encoding="utf-8") as log_stream:
                log_stream.write(log)
            report["active_stage"] = "measurement"
            print("MEASUREMENT_BEGIN", flush=True)
            try:
                measured, log = _capture_call(
                    chfem.compute_property, phase, materials, voxel[0],
                    args.native_tol, args.max_iterations, prefix)
            finally:
                print("MEASUREMENT_END", flush=True)
            report["measurement"] = measured
            with native_log.open("x", encoding="utf-8") as log_stream:
                log_stream.write(log)
            report["logs"] = {"warmup": str(warm_log), "measurement": str(native_log)}
            if fields is not None:
                report["fields"] = _export_npz(fields, prefix, phase, metadata, measured)
            report["passed"] = bool(measured["passed"])
            report["status"] = "passed" if report["passed"] else "failed_acceptance"
            report.pop("active_stage", None)
        except BaseException as error:
            if hasattr(error, "native_log"):
                failed_log = (warm_log if report.get("active_stage") == "warmup"
                              else native_log)
                with failed_log.open("x", encoding="utf-8") as log_stream:
                    log_stream.write(error.native_log)
            report.update(status="error", error=repr(error),
                          traceback=traceback.format_exc())
            raise
        finally:
            report["ended_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(_safe_json(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("RESULT", str(output), report["status"], flush=True)
    return 0 if report["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("measure", "export"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--native-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--source", type=Path,
                        default=Path(os.environ.get("CHFEM_SOURCE", DEFAULT_SOURCE)))
    args = parser.parse_args()
    if not (math.isfinite(args.native_tol) and 0.0 < args.native_tol < 1.0):
        parser.error("--native-tol must be finite and in (0,1)")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
