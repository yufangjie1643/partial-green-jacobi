"""CPU-only selection of an isotropic reference from material energy bounds.

``design_reference(phases, baseline_C0, kind)`` returns (C0, metadata).
Inputs use engineering Voigt strain: three normal entries followed by the three
engineering shear entries, in any consistent order.  No GPU or solver imports
are made.  NumPy and SciPy are the only non-standard-library dependencies.

For FI, a*C0 <= Ce <= b*C0 gives a*M <= K <= b*M under identical FE assembly.
For HG the same assertion also requires bounding each stabilization scale
scale(Ce)/scale(C0); this module includes those scalar ratios.  The resulting
b/a is an energy bound, not a prediction or guarantee of iteration count.
The bound concerns exact arithmetic, not the FP32 realization of an FFT inverse.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import minimize_scalar


def _stiffness(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real.")
    matrix = np.array(raw, dtype=np.float64, copy=True)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 6x6 matrix.")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0 or np.max(np.abs(matrix - matrix.T)) > 1e-11 * scale:
        raise ValueError(f"{name} must be symmetric and nonzero.")
    matrix = 0.5 * (matrix + matrix.T)
    if eigvalsh(matrix, check_finite=False)[0] <= 0.0:
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def isotropic_stiffness(bulk: float, mu: float) -> np.ndarray:
    """Return engineering-Voigt isotropic stiffness with positive K and mu."""
    bulk, mu = float(bulk), float(mu)
    if not (math.isfinite(bulk) and math.isfinite(mu) and bulk > 0 and mu > 0):
        raise ValueError("bulk and mu must be finite and positive.")
    lam = bulk - 2.0 * mu / 3.0
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2.0 * mu
    matrix[np.arange(3, 6), np.arange(3, 6)] = mu
    return matrix


def hg_scale(stiffness) -> float:
    """Rotation-invariant HG scale (I1 + 7 I2)/60; no material validation."""
    matrix = np.asarray(stiffness, dtype=np.float64)
    i1 = float(np.sum(matrix[:3, :3]))
    i2 = float(np.trace(matrix[:3, :3]) + 2.0 * np.trace(matrix[3:, 3:]))
    return (i1 + 7.0 * i2) / 60.0


def _bound(phases: list[np.ndarray], reference: np.ndarray, kind: str) -> dict:
    spectra = [eigvalsh(c, reference, check_finite=False) for c in phases]
    material_a = float(min(s[0] for s in spectra))
    material_b = float(max(s[-1] for s in spectra))
    a, b = material_a, material_b
    ratios = []
    if kind == "hg":
        reference_scale = hg_scale(reference)
        ratios = [float(hg_scale(c) / reference_scale) for c in phases]
        a, b = min(a, min(ratios)), max(b, max(ratios))
    if not (a > 0.0 and math.isfinite(b / a)):
        raise ValueError("The requested reference produced an invalid energy bound.")
    bulk = float(np.sum(reference[:3, :3]) / 9.0)
    mu = float(reference[3, 3])
    return {
        "a": a,
        "b": b,
        "condition_bound": b / a,
        "material_a": material_a,
        "material_b": material_b,
        "bulk": bulk,
        "mu": mu,
        "ratio": bulk / mu,
        "log_ratio": math.log(bulk / mu),
        "phase_generalized_eigenvalues": [s.tolist() for s in spectra],
        "hg_scale_ratios": ratios,
    }


def design_reference(
    phase_stiffnesses: Iterable,
    baseline_C0,
    kind: str = "fi",
    *,
    log_ratio_bounds: tuple[float, float] | None = None,
    grid_points: int = 161,
    xatol: float = 1e-10,
    maxiter: int = 200,
    tie_rtol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Minimize the sampled/refined b/a bound over log(Kbulk/mu).

    mu is fixed to baseline_C0[3,3].  Default bounds cover ratios 1e-3..1e3
    and at least two decades on either side of the baseline ratio.  A grid
    search is followed by bounded refinement of every sampled local basin.
    Endpoints and the unchanged baseline are always candidates, even when the
    latter lies outside explicitly supplied bounds.  Near ties prefer the
    baseline, then the ratio closest to it, to avoid arbitrary plateau choices.
    The finite search is not claimed to certify the global minimum.
    """
    kind = str(kind).lower()
    if kind not in ("fi", "hg"):
        raise ValueError("kind must be 'fi' or 'hg'.")
    phases = [_stiffness(c, f"phase[{i}]") for i, c in enumerate(phase_stiffnesses)]
    if not phases:
        raise ValueError("At least one material phase is required.")
    baseline = _stiffness(baseline_C0, "baseline_C0")
    mu = float(baseline[3, 3])
    bulk = float(np.sum(baseline[:3, :3]) / 9.0)
    expected = isotropic_stiffness(bulk, mu)
    if np.max(np.abs(expected - baseline)) > 1e-10 * np.max(np.abs(baseline)):
        raise ValueError("baseline_C0 must be isotropic in engineering Voigt notation.")
    if int(grid_points) != grid_points or grid_points < 3:
        raise ValueError("grid_points must be an integer >= 3.")
    if not (math.isfinite(xatol) and xatol > 0.0 and maxiter >= 1):
        raise ValueError("xatol and maxiter must be positive.")
    if not (math.isfinite(tie_rtol) and 0.0 <= tie_rtol < 1.0):
        raise ValueError("tie_rtol must lie in [0,1).")
    baseline_bound = _bound(phases, baseline, kind)
    baseline_log = math.log(bulk / mu)
    if log_ratio_bounds is None:
        lo = min(math.log(1e-3), baseline_log - math.log(100.0))
        hi = max(math.log(1e3), baseline_log + math.log(100.0))
    else:
        lo, hi = map(float, log_ratio_bounds)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("log_ratio_bounds must contain two finite increasing values.")

    evaluated: dict[float, dict] = {}

    def evaluate(log_ratio: float) -> dict:
        key = float(log_ratio)
        if key not in evaluated:
            c0 = isotropic_stiffness(mu * math.exp(key), mu)
            evaluated[key] = _bound(phases, c0, kind)
        return evaluated[key]

    def objective(log_ratio: float) -> float:
        return math.log(evaluate(log_ratio)["condition_bound"])

    grid = np.linspace(lo, hi, int(grid_points))
    values = np.array([objective(x) for x in grid])
    local = [
        i for i in range(1, len(grid) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]
    # Merge adjacent plateau minima into one bracket, preserving separate basins.
    groups: list[list[int]] = []
    for i in local:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    refinements = []
    for group in groups:
        left, right = float(grid[group[0] - 1]), float(grid[group[-1] + 1])
        result = minimize_scalar(
            objective, bounds=(left, right), method="bounded",
            options={"xatol": float(xatol), "maxiter": int(maxiter)},
        )
        refinements.append({
            "log_bracket": [left, right],
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "log_ratio": float(result.x),
            "condition_bound": evaluate(float(result.x))["condition_bound"],
        })
    candidates = [(x, point) for x, point in evaluated.items()]
    best_bound = min(point["condition_bound"] for _, point in candidates)
    eligible = [
        (x, point) for x, point in candidates
        if point["condition_bound"] <= best_bound * (1.0 + tie_rtol)
    ]
    selected_log, selected_bound = min(eligible, key=lambda pair: abs(pair[0] - baseline_log))
    selected = isotropic_stiffness(mu * math.exp(selected_log), mu)
    selected_from = "grid_or_bounded_refinement"
    if baseline_bound["condition_bound"] <= selected_bound["condition_bound"] * (1.0 + tie_rtol):
        selected = baseline.copy()
        selected_bound = baseline_bound.copy()
        selected_from = "baseline"

    metadata = {
        "kind": kind,
        "phase_count": len(phases),
        "parameter": "log(Kbulk/mu)",
        "fixed_mu": mu,
        "baseline": baseline_bound,
        "selected": selected_bound,
        "selected_from": selected_from,
        "bound_ratio_selected_over_baseline":
            selected_bound["condition_bound"] / baseline_bound["condition_bound"],
        "search": {
            "method": "log-grid_then_bounded_local_refinement",
            "log_ratio_bounds": [lo, hi],
            "ratio_bounds": [math.exp(lo), math.exp(hi)],
            "grid_points": int(grid_points),
            "grid": [
                {"log_ratio": float(x), "ratio": math.exp(float(x)),
                 "a": evaluate(x)["a"], "b": evaluate(x)["b"],
                 "condition_bound": evaluate(x)["condition_bound"]}
                for x in grid
            ],
            "refinements": refinements,
            "evaluated_points": len(evaluated),
            "xatol": float(xatol),
            "tie_rtol": float(tie_rtol),
            "baseline_always_considered": True,
            "baseline_inside_interval": bool(lo <= baseline_log <= hi),
        },
        "assumptions": [
            "All material tensors are symmetric positive definite.",
            "Identical FE geometry, integration and assembly for K and M.",
            "For HG, identical stabilization geometry with the supplied scalar scale(C).",
            "Condition bound applies on the common positive-energy subspace.",
            "Exact-arithmetic bound; FP32 FFT rounding is not covered.",
        ],
        "interpretation":
            "Minimizes a finite-search energy-bound surrogate, not measured PCG iterations.",
    }
    return selected, metadata


def self_check() -> dict:
    """Small deterministic CPU checks; no project solver imports or file writes."""
    rng = np.random.default_rng(20261021)
    baseline = isotropic_stiffness(3.0, 3.0)
    phase = isotropic_stiffness(10.0, 2.0)
    proportional = []
    for kind in ("fi", "hg"):
        _, meta = design_reference([phase, 7.0 * phase], baseline, kind)
        assert abs(meta["selected"]["condition_bound"] - 7.0) < 1e-7
        proportional.append({"kind": kind, "baseline": meta["baseline"]["condition_bound"],
                             "selected": meta["selected"]["condition_bound"]})

    phases = []
    for factor in (1.0, 12.0):
        x = rng.normal(size=(6, 6))
        phases.append(factor * (x.T @ x + np.eye(6)))
    # A physical rotation maps global engineering strains to local ones.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    shear_pairs = ((1, 2), (0, 2), (0, 1))
    transform = np.empty((6, 6))
    for j in range(6):
        e = np.zeros((3, 3))
        if j < 3:
            e[j, j] = 1.0
        else:
            a, b = shear_pairs[j - 3]
            e[a, b] = e[b, a] = 0.5
        local = q.T @ e @ q
        transform[:, j] = [*np.diag(local), *(2.0 * local[a, b] for a, b in shear_pairs)]
    rotated = [transform.T @ c @ transform for c in phases]
    scale_error = max(abs(hg_scale(c) - hg_scale(cr)) / hg_scale(c)
                      for c, cr in zip(phases, rotated))
    assert scale_error < 1e-12
    checks = []
    for kind in ("fi", "hg"):
        c0, meta = design_reference(phases, baseline, kind)
        before, after = _bound(phases, c0, kind), _bound(rotated, c0, kind)
        rotation_error = max(abs(before[k] - after[k]) / before[k] for k in ("a", "b"))
        assert rotation_error < 1e-12
        # Sum of congruences is the exact algebra used in FE assembly.
        k_global = np.zeros((11, 11))
        m_global = np.zeros((11, 11))
        for i in range(8):
            strain = rng.normal(size=(6, 11))
            c = rotated[i % 2]
            k_global += strain.T @ c @ strain
            m_global += strain.T @ c0 @ strain
            if kind == "hg":
                hourglass = rng.normal(size=(4, 11))
                gram = hourglass.T @ hourglass
                k_global += hg_scale(c) * gram
                m_global += hg_scale(c0) * gram
        spectrum = eigvalsh(k_global, m_global)
        assert spectrum[0] >= before["a"] * (1.0 - 1e-12)
        assert spectrum[-1] <= before["b"] * (1.0 + 1e-12)
        assert meta["selected"]["condition_bound"] <= meta["baseline"]["condition_bound"]
        checks.append({"kind": kind, "rotation_relative_error": rotation_error,
                       "bound": [before["a"], before["b"]],
                       "assembled_extremes": [float(spectrum[0]), float(spectrum[-1])]})
    return {"status": "passed", "seed": 20261021,
            "proportional_isotropic": proportional,
            "hg_scale_rotation_relative_error": scale_error,
            "random_congruence_assembly": checks}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2, allow_nan=False))
