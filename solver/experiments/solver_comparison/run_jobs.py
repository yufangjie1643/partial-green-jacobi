"""Serial fresh-process runner with the same NVML footprint sampler for both solvers."""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("VOXEL_RUN_DIR", str(__import__("project_paths").ROOT / "runs/solver_comparison")))
RUN_DIR.mkdir(parents=True, exist_ok=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ProcessInfo(ct.Structure):
    _fields_ = [("pid", ct.c_uint), ("usedGpuMemory", ct.c_ulonglong),
                ("gpuInstanceId", ct.c_uint), ("computeInstanceId", ct.c_uint)]


class MemoryInfo(ct.Structure):
    _fields_ = [("total", ct.c_ulonglong), ("free", ct.c_ulonglong),
                ("used", ct.c_ulonglong)]


class NVML:
    def __init__(self):
        self.lib = ct.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handle = ct.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(self.handle)))

    @staticmethod
    def check(result):
        if result != 0:
            raise RuntimeError(f"NVML error {result}")

    def sample(self):
        count = ct.c_uint(256)
        info = (ProcessInfo * count.value)()
        self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
            self.handle, ct.byref(count), info))
        memory = MemoryInfo()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ct.byref(memory)))
        processes = []
        for item in info[:count.value]:
            value = int(item.usedGpuMemory)
            if value == 2**64 - 1:
                raise RuntimeError("NVML does not expose per-process GPU memory")
            processes.append(dict(pid=int(item.pid), bytes=value))
        return dict(processes=processes, device_used_bytes=int(memory.used))


def belongs(pid, group):
    try:
        return os.getpgid(pid) == group
    except ProcessLookupError:
        return False


def execute(job, nvml, interval):
    initial = nvml.sample()
    if initial["processes"]:
        raise RuntimeError(f"GPU occupied before job: {initial['processes']}")
    name = job["name"]
    log = RUN_DIR / "logs" / f"{name}.log"
    telemetry_path = RUN_DIR / "logs" / f"{name}_telemetry.json"
    if log.exists() or telemetry_path.exists():
        raise FileExistsError(name)
    log.parent.mkdir(parents=True, exist_ok=True)
    arguments = list(job["arguments"])
    arguments[0] = str(HERE / arguments[0])
    for flag in ("--input", "--metadata", "--output", "--fields"):
        if flag in arguments:
            i = arguments.index(flag) + 1
            value = Path(arguments[i])
            if not value.is_absolute():
                arguments[i] = str((RUN_DIR if flag in ("--output", "--fields") else HERE) / value)
    command = job.get("command") or [sys.executable, "-u", *arguments]
    environment = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    started = time.perf_counter()
    proc = subprocess.Popen(command, cwd=RUN_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    chunks = queue.Queue()

    def reader():
        with log.open("xb") as stream:
            while chunk := proc.stdout.read1(65536):
                stream.write(chunk)
                stream.flush()
                chunks.put(chunk)
        chunks.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    samples, markers = [], []
    owned_pids = {proc.pid}
    measured = False
    marker_buffer = ""
    failure = None
    try:
        while True:
            while True:
                try:
                    chunk = chunks.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:
                    continue
                marker_buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in marker_buffer:
                    line, marker_buffer = marker_buffer.split("\n", 1)
                    if "MEASUREMENT_BEGIN" in line:
                        measured = True
                        markers.append(dict(kind="begin", seconds=time.perf_counter() - started))
                    if "MEASUREMENT_END" in line:
                        measured = False
                        markers.append(dict(kind="end", seconds=time.perf_counter() - started))
            frame = nvml.sample()
            outsiders = [
                p for p in frame["processes"]
                if p["pid"] not in owned_pids and not belongs(p["pid"], proc.pid)
            ]
            if outsiders:
                raise RuntimeError(f"Foreign GPU workload detected: {outsiders}")
            owned_pids.update(p["pid"] for p in frame["processes"])
            frame.update(seconds=time.perf_counter() - started, measurement=measured)
            frame["owned_process_bytes"] = sum(p["bytes"] for p in frame["processes"])
            samples.append(frame)
            if proc.poll() is not None and not thread.is_alive() and chunks.empty():
                break
            time.sleep(interval)
    except BaseException as error:
        failure = repr(error)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        raise
    finally:
        thread.join(timeout=5)
        selected = [s for s in samples if s["measurement"]]
        telemetry = dict(
            job=name, command=command, pid=proc.pid, returncode=proc.poll(),
            sample_interval_target_seconds=interval, markers=markers,
            peak_process_bytes_all=max((s["owned_process_bytes"] for s in samples), default=0),
            peak_process_bytes_measured=max((s["owned_process_bytes"] for s in selected), default=0),
            measurement_sample_count=len(selected), initial=initial, samples=samples, failure=failure,
            scope="Same NVML process GPU footprint for both solvers; includes CUDA context, libraries and allocator reservation. Sampled peak is a lower bound, not live tensor allocations.",
        )
        save(telemetry_path, telemetry)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} exited {proc.returncode}; see {log}")
    return dict(name=name, log=str(log), telemetry=str(telemetry_path),
                process_wall_seconds=time.perf_counter() - started, complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    nvml = NVML()
    if args.probe:
        print(json.dumps(nvml.sample()))
        return
    if not args.plan:
        parser.error("--plan is required unless --probe is used")
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    status_path = RUN_DIR / (args.plan.stem + "_status.json")
    if status_path.exists():
        raise FileExistsError(status_path)
    status = dict(complete=False, jobs=plan, completed=[], failed=None)
    save(status_path, status)
    for job in plan:
        try:
            status["completed"].append(execute(job, nvml, args.interval))
            save(status_path, status)
            print("DONE", job["name"], flush=True)
        except BaseException as error:
            status["failed"] = dict(name=job["name"], error=repr(error))
            save(status_path, status)
            raise
    status["complete"] = True
    save(status_path, status)


if __name__ == "__main__":
    main()
