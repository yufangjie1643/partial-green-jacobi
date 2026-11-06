"""All project dependencies resolve inside this bundle, including subprocesses."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def source_path(relative=""):
    return ROOT / "src" / relative


def configure():
    components = dict(FEAD_ABLATION="compact_operator", FEAD_STRUCTURAL="constitutive_basis", FEAD_RUNTIME="runtime",
                      FEAD_MODAL="hex8_modal",
                      FEAD_PIPELINE="assembly_pipeline",
                      FEAD_FORMULA="reference_medium",
                      FEAD_FFT="spectral",
                      FEAD_CHAIN="solver_chain",
                      FEAD_HYBRID="materials",
                      FEAD_OPERATOR="hybrid_operator")
    for key, directory in components.items():
        value = source_path(directory)
        if not value.is_dir():
            raise FileNotFoundError(value)
        os.environ[key] = str(value)
    os.environ["FEAD_PACKAGE_ROOT"] = str(ROOT)
    os.environ["VOXEL_PROJECT_ROOT"] = str(ROOT)
    os.environ["FEAD_TOL"] = "1e-8"
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / ".cache/torch_extensions")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MAX_JOBS", "2")
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                 if p and not p.startswith(str(Path.home()/"code"))]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *inherited])
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/cuda/bin", os.environ.get("PATH", "")])
    for key in ("FEAD_CHAIN", "FEAD_OPERATOR"):
        sys.path.insert(0, os.environ[key])
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

