"""Tolerance analysis for 3D rigid body assemblies.

Computes how feature tolerances and mate tolerances propagate to the
assembly measurement using:

1. Numerical Jacobian — finite-difference sensitivity of the measurement
   to each tolerance parameter.
2. Worst-Case — Jacobian-based sum of absolute sensitivities.
3. RSS — Jacobian-based root-sum-of-squares.
4. Monte Carlo — full assembly evaluation with sampled parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tolerance_stack.assembly import Assembly, _perpendicular_axes
from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AssemblyAnalysisResult:
    """Results from an assembly tolerance analysis.

    Attributes:
        method: Analysis method name.
        nominal_value: Nominal measurement value.
        value_max: Upper bound of the measurement.
        value_min: Lower bound of the measurement.
        plus_tolerance: Upper tolerance.
        minus_tolerance: Lower tolerance.
        sigma_level: Sigma level (RSS only).
        sensitivity: List of (param_name, dMeasurement/dParam).
        mc_samples: Monte Carlo measurement samples.
        mc_mean: MC mean.
        mc_std: MC standard deviation.
    """
    method: str
    nominal_value: float
    value_max: float
    value_min: float
    plus_tolerance: float
    minus_tolerance: float
    sigma_level: Optional[float] = None
    sensitivity: list[tuple[str, float]] = field(default_factory=list)
    mc_samples: Optional[np.ndarray] = None
    mc_mean: Optional[float] = None
    mc_std: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"=== {self.method} Assembly Analysis ===",
            f"  Nominal value:    {self.nominal_value:+.6f}",
            f"  Value range:      [{self.value_min:+.6f}, {self.value_max:+.6f}]",
            f"  Plus tolerance:   +{self.plus_tolerance:.6f}",
            f"  Minus tolerance:  -{self.minus_tolerance:.6f}",
        ]
        if self.sigma_level is not None:
            lines.append(f"  Sigma level:      {self.sigma_level:.1f}")
        if self.mc_mean is not None:
            lines.append(f"  MC mean:          {self.mc_mean:+.6f}")
            lines.append(f"  MC std dev:       {self.mc_std:.6f}")
        if self.sensitivity:
            lines.append("  Sensitivity (measurement shift per unit parameter change):")
            # Sort by |sensitivity|
            sorted_sens = sorted(self.sensitivity, key=lambda x: abs(x[1]), reverse=True)
            for name, s in sorted_sens:
                if abs(s) > 1e-10:
                    lines.append(f"    {name:40s}  {s:+.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _build_perturbation(
    assembly: Assembly,
    param: dict,
    delta: float,
) -> dict:
    """Build the perturbation arguments for compute_measurement.

    Returns dict with keys: feature_offsets, feature_angle_offsets,
    mate_distance_offsets — ready to pass to compute_measurement.
    """
    fo = {}
    fao = {}
    mdo = {}

    source = param["source"]

    if source == "feature_position":
        body = param["body"]
        feat_name = param["feature"]
        comp = param["component"]

        if comp == "directed":
            bp = assembly.bodies[body]
            feat = bp.body.get_feature(feat_name)
            d = np.array(feat.position_tol_direction, dtype=float)
            d = d / np.linalg.norm(d)
            fo[(body, feat_name)] = d * delta
        else:
            offset = np.zeros(3)
            offset[comp] = delta
            fo[(body, feat_name)] = offset

    elif source == "feature_orientation":
        body = param["body"]
        feat_name = param["feature"]
        comp_idx = param["component"]  # 0 or 1

        bp = assembly.bodies[body]
        feat = bp.body.get_feature(feat_name)
        perp = _perpendicular_axes(np.array(feat.direction))
        axis = perp[comp_idx]

        # Convert axis to Euler-ish small-angle rotation
        # Small angle: rotation about axis by delta degrees
        # Approximate as euler angles: drx = axis[0]*delta, etc.
        angles = axis * delta
        fao[(body, feat_name)] = angles

    elif source == "mate_distance":
        mdo[param["mate"]] = delta

    return {"feature_offsets": fo, "feature_angle_offsets": fao,
            "mate_distance_offsets": mdo}


def _compute_jacobian(
    assembly: Assembly,
    delta: float = 1e-6,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Numerical Jacobian of the measurement w.r.t. all tolerance parameters.

    Returns:
        (J, param_names, params)
        J: (1 x n_params) or just (n_params,) sensitivity vector.
    """
    params = assembly.tolerance_parameters()
    n = len(params)
    J = np.zeros(n)
    nominal = assembly.compute_measurement()

    for i, p in enumerate(params):
        pert_plus = _build_perturbation(assembly, p, +delta)
        val_plus = assembly.compute_measurement(**pert_plus)

        pert_minus = _build_perturbation(assembly, p, -delta)
        val_minus = assembly.compute_measurement(**pert_minus)

        J[i] = (val_plus - val_minus) / (2.0 * delta)

    param_names = [p["name"] for p in params]
    return J, param_names, params


# ---------------------------------------------------------------------------
# Worst-Case
# ---------------------------------------------------------------------------

def assembly_worst_case(assembly: Assembly) -> AssemblyAnalysisResult:
    """Worst-case assembly tolerance analysis."""
    J, param_names, params = _compute_jacobian(assembly)
    nominal = assembly.compute_measurement()

    total_plus = 0.0
    total_minus = 0.0
    sensitivity = []

    for i, p in enumerate(params):
        s = J[i]
        half = p["half_tol"]
        sensitivity.append((p["name"], s))

        abs_s = abs(s)
        total_plus += abs_s * half
        total_minus += abs_s * half

    return AssemblyAnalysisResult(
        method="Worst-Case",
        nominal_value=nominal,
        value_max=nominal + total_plus,
        value_min=nominal - total_minus,
        plus_tolerance=total_plus,
        minus_tolerance=total_minus,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def assembly_rss(assembly: Assembly, sigma: float = 3.0) -> AssemblyAnalysisResult:
    """RSS assembly tolerance analysis."""
    J, param_names, params = _compute_jacobian(assembly)
    nominal = assembly.compute_measurement()

    sum_var = 0.0
    sensitivity = []

    for i, p in enumerate(params):
        s = J[i]
        half = p["half_tol"]
        p_sigma = p["sigma"]
        sensitivity.append((p["name"], s))

        std_i = half / p_sigma
        sum_var += (s * std_i) ** 2

    rss_std = np.sqrt(sum_var)
    rss_tol = sigma * rss_std

    return AssemblyAnalysisResult(
        method="RSS",
        nominal_value=nominal,
        value_max=nominal + rss_tol,
        value_min=nominal - rss_tol,
        plus_tolerance=rss_tol,
        minus_tolerance=rss_tol,
        sigma_level=sigma,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def assembly_monte_carlo(
    assembly: Assembly,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> AssemblyAnalysisResult:
    """Monte Carlo assembly tolerance analysis."""
    rng = np.random.default_rng(seed)
    params = assembly.tolerance_parameters()
    nominal = assembly.compute_measurement()

    # Pre-generate all parameter samples
    param_samples = {}
    for p in params:
        half = p["half_tol"]
        center = p["nominal"]
        sigma = p["sigma"]
        dist = p["distribution"]

        if dist == Distribution.NORMAL:
            std = half / sigma
            samples = rng.normal(loc=center, scale=std, size=n_samples)
        elif dist == Distribution.UNIFORM:
            samples = rng.uniform(low=center - half, high=center + half, size=n_samples)
        elif dist == Distribution.TRIANGULAR:
            samples = rng.triangular(left=center - half, mode=center,
                                      right=center + half, size=n_samples)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        param_samples[p["name"]] = samples

    # Evaluate measurement for each sample
    measurements = np.zeros(n_samples)

    for s in range(n_samples):
        fo = {}
        fao = {}
        mdo = {}

        for p in params:
            val = param_samples[p["name"]][s]
            source = p["source"]

            if source == "feature_position":
                body = p["body"]
                feat_name = p["feature"]
                comp = p["component"]

                key = (body, feat_name)
                if comp == "directed":
                    bp = assembly.bodies[body]
                    feat = bp.body.get_feature(feat_name)
                    d = np.array(feat.position_tol_direction, dtype=float)
                    d = d / np.linalg.norm(d)
                    fo[key] = fo.get(key, np.zeros(3)) + d * val
                else:
                    offset = fo.get(key, np.zeros(3))
                    offset[comp] = val
                    fo[key] = offset

            elif source == "feature_orientation":
                body = p["body"]
                feat_name = p["feature"]
                comp_idx = p["component"]

                bp = assembly.bodies[body]
                feat = bp.body.get_feature(feat_name)
                perp = _perpendicular_axes(np.array(feat.direction))
                axis = perp[comp_idx]

                key = (body, feat_name)
                angles = fao.get(key, np.zeros(3))
                angles = angles + axis * val
                fao[key] = angles

            elif source == "mate_distance":
                mdo[p["mate"]] = val

        measurements[s] = assembly.compute_measurement(
            feature_offsets=fo,
            feature_angle_offsets=fao,
            mate_distance_offsets=mdo,
        )

    mc_mean = float(np.mean(measurements))
    mc_std = float(np.std(measurements, ddof=1))

    # Also compute Jacobian for sensitivity
    J, param_names, _ = _compute_jacobian(assembly)
    sensitivity = [(pn, J[i]) for i, pn in enumerate(param_names)]

    return AssemblyAnalysisResult(
        method="Monte Carlo",
        nominal_value=nominal,
        value_max=float(np.max(measurements)),
        value_min=float(np.min(measurements)),
        plus_tolerance=float(np.max(measurements)) - mc_mean,
        minus_tolerance=mc_mean - float(np.min(measurements)),
        sensitivity=sensitivity,
        mc_samples=measurements,
        mc_mean=mc_mean,
        mc_std=mc_std,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def analyze_assembly(
    assembly: Assembly,
    methods: Optional[list[str]] = None,
    sigma: float = 3.0,
    mc_samples: int = 100_000,
    mc_seed: Optional[int] = None,
) -> dict[str, AssemblyAnalysisResult]:
    """Run analysis methods on an assembly.

    Args:
        assembly: The assembly to analyze.
        methods: ["wc", "rss", "mc"] or subset. Defaults to all.
        sigma: Sigma level for RSS.
        mc_samples: Number of Monte Carlo samples.
        mc_seed: Random seed.

    Returns:
        Dict mapping method name to result.
    """
    if methods is None:
        methods = ["wc", "rss", "mc"]

    results: dict[str, AssemblyAnalysisResult] = {}

    for m in methods:
        key = m.lower().strip()
        if key in ("wc", "worst-case", "worst_case"):
            results["wc"] = assembly_worst_case(assembly)
        elif key == "rss":
            results["rss"] = assembly_rss(assembly, sigma=sigma)
        elif key in ("mc", "monte-carlo", "monte_carlo"):
            results["mc"] = assembly_monte_carlo(
                assembly, n_samples=mc_samples, seed=mc_seed
            )
        else:
            raise ValueError(f"Unknown method: {m!r}")

    return results
