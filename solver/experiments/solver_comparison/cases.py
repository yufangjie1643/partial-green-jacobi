"""Shared, immutable isotropic input fields for the FI/chfem comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MATERIALS = [{"E": 3500.0, "nu": 0.35}, {"E": 70000.0, "nu": 0.20}]
DEFAULT_ARCHIVE = __import__("project_paths").ROOT / "data/grosmont"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(phase):
    return hashlib.sha256(np.ascontiguousarray(phase).tobytes()).hexdigest()


def load_case(input_path, metadata_path=None):
    input_path = Path(input_path)
    metadata_path = Path(metadata_path) if metadata_path else input_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    phase = np.load(input_path, allow_pickle=False)
    assert phase.dtype == np.uint8 and phase.ndim == 3
    assert len(set(phase.shape)) == 1
    assert set(np.unique(phase)).issubset({0, 255})
    assert list(phase.shape) == metadata["grid_shape_zyx"]
    assert array_sha(phase) == metadata["array_sha256"]
    assert sha(input_path) == metadata["npy_sha256"]
    assert metadata["phase_materials"] == MATERIALS
    return np.ascontiguousarray(phase), metadata


def prepare(case, grid, directory, archive=DEFAULT_ARCHIVE):
    directory, archive = Path(directory), Path(archive)
    path = directory / f"{case}_{grid}.npy"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        load_case(path, meta_path)
        return path
    directory.mkdir(parents=True, exist_ok=True)
    if case == "sphere":
        x = (np.arange(grid, dtype=np.float64) + 0.5) / grid - 0.5
        r = (3.0 * 0.30 / (4.0 * np.pi)) ** (1.0 / 3.0)
        phase = np.where(
            x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2 <= r * r,
            255, 0,
        ).astype(np.uint8)
        source = dict(
            definition="Centered sphere; cell-center sampling in normalized unit cube",
            target_volume_fraction=0.30, normalized_radius=r,
            coordinate_rule="(i+0.5)/N - 0.5", surface_inclusion_rule="distance_squared <= radius_squared",
            material_source="doc/exp/experiment_report_isotropic_sphere_unified.md",
        )
        np.save(path, phase)
    elif case == "grosmont":
        old_path = archive / path.name
        old_meta = json.loads(old_path.with_suffix(".json").read_text())
        phase = np.load(old_path, allow_pickle=False)
        assert phase.dtype == np.uint8 and phase.shape == (grid,) * 3
        assert array_sha(phase) == old_meta["array_sha256"]
        assert sha(old_path) == old_meta["npy_sha256"]
        # Preserve the archived file bytes, not a second resampling of the crop.
        path.write_bytes(old_path.read_bytes())
        source = dict(
            archive_path=str(old_path), archive_metadata=old_meta,
            operation="Byte-identical reuse of existing 400-cube crop to N-cube NN input",
        )
    else:
        raise ValueError(case)
    metadata = dict(
        case=case, grid=grid, grid_shape_zyx=list(phase.shape),
        phase_encoding={"0": "soft phase", "255": "stiff phase"},
        phase_materials=MATERIALS, material_units="MPa",
        voxel_size_xyz=[1.0, 1.0, 1.0],
        solid_fraction=float((phase != 0).mean()),
        array_sha256=array_sha(phase), npy_sha256=sha(path), source_info=source,
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_case(path, meta_path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[8, 32, 128, 192])
    parser.add_argument("--cases", nargs="+", choices=["sphere", "grosmont"],
                        default=["sphere", "grosmont"])
    parser.add_argument("--directory", type=Path, default=HERE / "inputs")
    parser.add_argument("--grosmont-archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    for case in args.cases:
        for grid in args.grids:
            path = prepare(case, grid, args.directory, args.grosmont_archive)
            _, metadata = load_case(path)
            print(json.dumps(dict(path=str(path), **metadata), ensure_ascii=False))


if __name__ == "__main__":
    main()
