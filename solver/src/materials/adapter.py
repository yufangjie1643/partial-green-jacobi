"""Connect the generic phase API to the frozen FI/FFT/PCG chain, without edits."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType

import numpy as np


def load_material_module():
    path = Path(__file__).with_name("material_field.py")
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "repro/source/femlib/material_field.py"
    import femlib
    spec = importlib.util.spec_from_file_location("femlib.material_field", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install(b, cs, *, chunk_size=65536):
    """Install before chain_adapter.configure; this chain driver supports B=1.

    data: grid_shape, domain_size, phase_id, optional orientation.
    material: the core API's integer-keyed phase table. No yarn fields.
    """
    mf = load_material_module()
    from reference_closed_form import design_closed

    def construct(data, lo, hi, materials, selected, packed):
        if not selected or not packed:
            raise ValueError("Generic FI adapter requires packed C21 construction")
        phase = np.asarray(data["phase_id"])
        if phase.ndim == 3:
            phase = phase[None]
        if phase.shape[0] != 1 or (lo, hi) != (0, 1):
            raise ValueError("The frozen FI experiment adapter supports B=1; core construction supports batches")
        if tuple(phase.shape[-3:]) != tuple(data["grid_shape"]):
            raise ValueError("phase_id and grid_shape disagree")
        return mf.build_material_field(phase, materials, orientation=data.get("orientation"),
                                       chunk_size=chunk_size)

    b.construct = construct
    previous_build = b.build

    def build(data, materials, route, precision, stages):
        op = previous_build(data, materials, route, precision, stages)
        active_labels = np.unique(data["phase_id"])
        phases = [mf.material_stiffness(materials[int(k)]) for k in active_labels]
        reference, metadata = design_closed(phases, op._formula_reference_C0, "fi")
        metadata["active_phase_labels"] = active_labels.tolist()
        previous = op._compute_reference_medium
        op._compute_reference_medium = MethodType(lambda self, *args, **kwargs: reference[None], op)
        try:
            cs.s.f.formula.enable_fi_reference(op)
        finally:
            op._compute_reference_medium = previous
        b._reference_design = metadata
        stages.end("spectral_reference_design", op)
        return op

    b.build = build
    return mf
