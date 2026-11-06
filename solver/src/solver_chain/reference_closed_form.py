"""Closed-form reference selection when an actual isotropic soft phase exists.

If Cm is one of the material phases and Cm <= Ce for every phase, then C0=s*Cm
attains the smallest possible worst material energy bound among *all* SPD
references, not only isotropic ones:

    min_C0 b(C0)/a(C0) = max_e lambda_max(Ce, Cm).

Indeed, a*C0 <= Cm and Ce <= b*C0 imply Ce <= (b/a)*Cm.  For C0=s*Cm the
lower bound is attained, since a=1/s and b=max_e lambda_max(Ce,Cm)/s.
The HG scale is a positive linear functional and obeys the same inequalities.
This result optimizes the material bound, not actual PCG iteration count.

All verification here is tolerance-based float64 arithmetic; it is not an
interval-arithmetic certificate.  No GPU imports or calls are performed.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.linalg import eigvalsh

try:
    from .reference_design import (
        _bound, _stiffness, design_reference, hg_scale, isotropic_stiffness,
    )
except ImportError:
    from reference_design import (
        _bound, _stiffness, design_reference, hg_scale, isotropic_stiffness,
    )


def _isotropy(matrix: np.ndarray, tolerance: float) -> dict:
    bulk = float(np.sum(matrix[:3, :3]) / 9.0)
    mu = float(np.trace(matrix[3:, 3:]) / 3.0)
    candidate = isotropic_stiffness(bulk, mu)
    error = float(np.max(np.abs(matrix - candidate)) / np.max(np.abs(matrix)))
    return {"is_isotropic": bool(error <= tolerance), "relative_error": error,
            "bulk": bulk, "mu": mu, "ratio": bulk / mu}


def design_closed(
    phase_stiffnesses: Iterable,
    baseline,
    kind: str = "fi",
    *,
    fallback: str = "baseline",
    isotropy_rtol: float = 1e-10,
    order_rtol: float = 1e-10,
    tie_rtol: float = 1e-10,
    numeric_options: dict | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Return (reference, metadata), preserving baseline on equal material bounds.

    An eligible soft phase must be both an actual input material and isotropic.
    Its ordering is checked through lambda_min(Ce,Cm) >= 1-order_rtol.
    Scaling fixes the reference shear modulus to baseline[3,3].

    If no eligible phase exists, the default fallback="baseline" preserves the
    baseline and reports status="unsupported".  fallback="unsupported" instead
    returns (None, metadata).  fallback="numeric" calls
    reference_design.design_reference with optional numeric_options.
    No numeric scan is executed on the supported path.
    """
    kind = str(kind).lower()
    if kind not in ("fi", "hg"):
        raise ValueError("kind must be 'fi' or 'hg'.")
    if fallback not in ("baseline", "unsupported", "numeric"):
        raise ValueError("fallback must be 'baseline', 'unsupported' or 'numeric'.")
    for name, value in (("isotropy_rtol", isotropy_rtol), ("order_rtol", order_rtol),
                        ("tie_rtol", tie_rtol)):
        if not (math.isfinite(value) and 0.0 <= value < 1.0):
            raise ValueError(f"{name} must lie in [0,1).")
    phases = [_stiffness(c, f"phase[{i}]") for i, c in enumerate(phase_stiffnesses)]
    if not phases:
        raise ValueError("At least one material phase is required.")
    baseline = _stiffness(baseline, "baseline")
    if not _isotropy(baseline, isotropy_rtol)["is_isotropic"]:
        raise ValueError("baseline must be isotropic in engineering Voigt notation.")
    baseline_bound = _bound(phases, baseline, kind)
    mu = float(baseline[3, 3])
    ordered_checks = []
    candidates = []
    for i, soft in enumerate(phases):
        check = {"phase_index": i, **_isotropy(soft, isotropy_rtol)}
        check["ordered"] = False
        check["phase_generalized_eigenvalues"] = []
        if check["is_isotropic"]:
            spectra = [eigvalsh(c, soft, check_finite=False) for c in phases]
            check["phase_generalized_eigenvalues"] = [values.tolist() for values in spectra]
            check["minimum_order_eigenvalue"] = float(min(values[0] for values in spectra))
            check["maximum_order_deficit"] = max(0.0, 1.0 - check["minimum_order_eigenvalue"])
            check["ordered"] = bool(check["minimum_order_eigenvalue"] >= 1.0 - order_rtol)
            if check["ordered"]:
                scale = mu / float(soft[3, 3])
                reference = scale * soft
                bound = _bound(phases, reference, kind)
                optimum = float(max(values[-1] for values in spectra))
                formula = {
                    "soft_phase_index": i,
                    "reference_scale": scale,
                    "a": 1.0 / scale,
                    "b": optimum / scale,
                    "condition_bound": optimum,
                    "expression": "max_e lambda_max(Ce,Cm)",
                    "candidate_relative_formula_error":
                        abs(bound["condition_bound"] - optimum) / optimum,
                }
                candidates.append((reference, bound, formula))
        ordered_checks.append(check)
    metadata = {
        "kind": kind,
        "phase_count": len(phases),
        "fixed_mu": mu,
        "method": "ordered_isotropic_soft_phase_closed_form",
        "baseline": baseline_bound,
        "selected": None,
        "formula": None,
        "ordered_validation": ordered_checks,
        "validation_tolerances": {
            "isotropy_rtol": float(isotropy_rtol),
            "order_rtol": float(order_rtol),
            "tie_rtol": float(tie_rtol),
        },
        "numeric_search_used": False,
        "interpretation":
            "Global optimum of the worst material energy bound under the exact stated "
            "ordering; not a guarantee of minimum PCG iterations or runtime.",
        "floating_point_verification":
            "Float64 tolerance tests, not interval-arithmetic certificates.",
        "hg_argument":
            "scale(C)=trace(W*C), W=(t*t.T+7*diag(1,1,1,2,2,2))/60 positive definite; "
            "material Loewner bounds therefore also bound stabilization scale ratios.",
    }
    if candidates:
        reference, bound, formula = min(candidates, key=lambda item: item[1]["condition_bound"])
        metadata["formula"] = formula
        metadata["closed_candidate"] = bound
        metadata["status"] = "supported"
        if baseline_bound["condition_bound"] <= bound["condition_bound"] * (1.0 + tie_rtol):
            reference, bound = baseline.copy(), baseline_bound.copy()
            metadata["selected_from"] = "baseline"
        else:
            metadata["selected_from"] = "closed_form"
        metadata["selected"] = bound
        metadata["bound_ratio_selected_over_baseline"] = (
            bound["condition_bound"] / baseline_bound["condition_bound"]
        )
        return reference, metadata
    metadata["status"] = "unsupported"
    metadata["reason"] = "No actual isotropic phase is Loewner-below every input phase."
    if fallback == "baseline":
        metadata["selected_from"] = "baseline"
        metadata["selected"] = baseline_bound.copy()
        metadata["bound_ratio_selected_over_baseline"] = 1.0
        return baseline.copy(), metadata
    if fallback == "numeric":
        reference, numeric = design_reference(
            phases, baseline, kind, **({} if numeric_options is None else numeric_options)
        )
        metadata["status"] = "numeric_fallback"
        metadata["selected_from"] = "numeric_fallback"
        metadata["numeric_search_used"] = True
        metadata["numeric_metadata"] = numeric
        metadata["selected"] = numeric["selected"]
        metadata["bound_ratio_selected_over_baseline"] = numeric["bound_ratio_selected_over_baseline"]
        return reference, metadata
    return None, metadata


def self_check() -> dict:
    """Reproducible CPU checks, including arbitrary SPD references and rotations."""
    import json

    rng = np.random.default_rng(20261021)
    soft = isotropic_stiffness(10.0, 2.0)
    baseline = isotropic_stiffness(3.0, 3.0)
    proportional = []
    for kind in ("fi", "hg"):
        selected, meta = design_closed([soft, 7.0 * soft], baseline, kind)
        assert selected is not None and meta["selected_from"] == "closed_form"
        assert abs(meta["selected"]["condition_bound"] - 7.0) < 1e-12
        assert abs(meta["selected"]["mu"] - baseline[3, 3]) < 1e-14
        retained, tied = design_closed([soft, 7.0 * soft], 2.0 * soft, kind)
        assert np.array_equal(retained, 2.0 * soft)
        assert tied["selected_from"] == "baseline" and not tied["numeric_search_used"]
        json.dumps(meta, allow_nan=False)
        proportional.append({"kind": kind, "baseline": meta["baseline"]["condition_bound"],
                             "selected": meta["selected"]["condition_bound"]})
    x = rng.normal(size=(6, 6))
    stiff = soft + 20.0 * (x.T @ x)
    phases = [soft, stiff]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pairs = ((1, 2), (0, 2), (0, 1))
    transform = np.empty((6, 6))
    for j in range(6):
        strain = np.zeros((3, 3))
        if j < 3:
            strain[j, j] = 1.0
        else:
            a, b = pairs[j - 3]
            strain[a, b] = strain[b, a] = 0.5
        local = q.T @ strain @ q
        transform[:, j] = [*np.diag(local), *(2.0 * local[a, b] for a, b in pairs)]
    rotated = [soft, transform.T @ stiff @ transform]
    comparisons = []
    for kind in ("fi", "hg"):
        selected, meta = design_closed(phases, baseline, kind)
        selected_rot, meta_rot = design_closed(rotated, baseline, kind)
        assert selected is not None and selected_rot is not None
        optimum = meta["formula"]["condition_bound"]
        rotation_error = abs(meta_rot["formula"]["condition_bound"] - optimum) / optimum
        assert rotation_error < 1e-12
        assert abs(meta["closed_candidate"]["condition_bound"] - optimum) < 1e-10 * optimum
        ratios = []
        for _ in range(32):
            x = rng.normal(size=(6, 6))
            arbitrary_reference = x.T @ x + np.eye(6)
            material_spectra = [eigvalsh(c, arbitrary_reference) for c in phases]
            a = min(values[0] for values in material_spectra)
            b = max(values[-1] for values in material_spectra)
            if kind == "hg":
                hg_ratios = [hg_scale(c) / hg_scale(arbitrary_reference) for c in phases]
                a, b = min(a, min(hg_ratios)), max(b, max(hg_ratios))
            ratio = float((b / a) / optimum)
            assert ratio >= 1.0 - 1e-12
            ratios.append(ratio)
        comparisons.append({
            "kind": kind, "closed_bound": optimum,
            "rotation_relative_error": rotation_error,
            "arbitrary_spd_references_tested": len(ratios),
            "minimum_arbitrary_bound_over_closed_bound": min(ratios),
        })
    unordered = [isotropic_stiffness(10.0, 1.0), isotropic_stiffness(1.0, 10.0)]
    retained, meta = design_closed(unordered, baseline)
    assert np.array_equal(retained, baseline) and meta["status"] == "unsupported"
    unsupported, meta = design_closed(unordered, baseline, fallback="unsupported")
    assert unsupported is None and meta["status"] == "unsupported"
    fallback, meta = design_closed(
        unordered, baseline, fallback="numeric", numeric_options={"grid_points": 33}
    )
    assert fallback is not None and meta["status"] == "numeric_fallback"
    assert meta["numeric_search_used"]
    anisotropic = np.diag([2.0, 3.0, 4.0, 1.0, 1.5, 2.0])
    unsupported, meta = design_closed(
        [anisotropic, 2.0 * anisotropic], baseline, fallback="unsupported"
    )
    assert unsupported is None and meta["status"] == "unsupported"
    return {"status": "passed", "seed": 20261021, "proportional_phases": proportional,
            "ordered_anisotropic_phase": comparisons,
            "baseline_tie_retained": True, "unsupported_order_checked": True,
            "unsupported_anisotropy_checked": True, "numeric_fallback_checked": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2, allow_nan=False))
