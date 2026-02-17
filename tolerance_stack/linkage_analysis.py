"""Tolerance analysis for 3D linkages.

Computes how tolerances on joint angles and link lengths propagate to the
end-effector position using:

1. Numerical Jacobian — finite-difference sensitivity of end-effector XYZ
   to each tolerance parameter.
2. Worst-Case — Jacobian-based, summing absolute sensitivities.
3. RSS — Jacobian-based, root-sum-of-squares of variances.
4. Monte Carlo — full forward-kinematics simulation with sampled parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tolerance_stack.linkage import Linkage, JointType, Link, Joint
from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LinkageAnalysisResult:
    """Results from a linkage tolerance analysis.

    All positional results are 3-component vectors [X, Y, Z].

    Attributes:
        method: Name of the analysis method.
        nominal_position: Nominal end-effector [X, Y, Z].
        position_max: Per-axis upper bound [X, Y, Z].
        position_min: Per-axis lower bound [X, Y, Z].
        plus_tolerance: Per-axis upper tolerance [X, Y, Z].
        minus_tolerance: Per-axis lower tolerance [X, Y, Z].
        radial_tolerance: Scalar RSS tolerance radius (3D sphere).
        jacobian: (3 x n_params) Jacobian matrix if computed.
        param_names: Names of the parameters corresponding to Jacobian columns.
        sensitivity: List of (name, [dX, dY, dZ]) per-parameter sensitivity.
        mc_samples: (n_samples x 3) Monte Carlo position samples.
        mc_mean: Monte Carlo mean position.
        mc_std: Monte Carlo per-axis standard deviation.
        mc_cov: Monte Carlo 3x3 covariance matrix.
    """
    method: str
    nominal_position: np.ndarray
    position_max: np.ndarray
    position_min: np.ndarray
    plus_tolerance: np.ndarray
    minus_tolerance: np.ndarray
    radial_tolerance: float = 0.0
    jacobian: Optional[np.ndarray] = None
    param_names: list[str] = field(default_factory=list)
    sensitivity: list[tuple[str, np.ndarray]] = field(default_factory=list)
    mc_samples: Optional[np.ndarray] = None
    mc_mean: Optional[np.ndarray] = None
    mc_std: Optional[np.ndarray] = None
    mc_cov: Optional[np.ndarray] = None

    def summary(self) -> str:
        nom = self.nominal_position
        lines = [
            f"=== {self.method} Linkage Analysis ===",
            f"  Nominal position:  X={nom[0]:+.6f}  Y={nom[1]:+.6f}  Z={nom[2]:+.6f}",
            f"  Position max:      X={self.position_max[0]:+.6f}  Y={self.position_max[1]:+.6f}  Z={self.position_max[2]:+.6f}",
            f"  Position min:      X={self.position_min[0]:+.6f}  Y={self.position_min[1]:+.6f}  Z={self.position_min[2]:+.6f}",
            f"  Plus tolerance:    X=+{self.plus_tolerance[0]:.6f}  Y=+{self.plus_tolerance[1]:.6f}  Z=+{self.plus_tolerance[2]:.6f}",
            f"  Minus tolerance:   X=-{self.minus_tolerance[0]:.6f}  Y=-{self.minus_tolerance[1]:.6f}  Z=-{self.minus_tolerance[2]:.6f}",
            f"  Radial tolerance:  {self.radial_tolerance:.6f}",
        ]
        if self.mc_mean is not None:
            lines.append(f"  MC mean:           X={self.mc_mean[0]:+.6f}  Y={self.mc_mean[1]:+.6f}  Z={self.mc_mean[2]:+.6f}")
            lines.append(f"  MC std dev:        X={self.mc_std[0]:.6f}  Y={self.mc_std[1]:.6f}  Z={self.mc_std[2]:.6f}")
        if self.sensitivity:
            lines.append("  Sensitivity (end-effector shift per unit parameter change):")
            for name, s in self.sensitivity:
                lines.append(f"    {name:30s}  dX={s[0]:+.6f}  dY={s[1]:+.6f}  dZ={s[2]:+.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Numerical Jacobian
# ---------------------------------------------------------------------------

def _compute_jacobian(
    linkage: Linkage,
    delta: float = 1e-6,
) -> tuple[np.ndarray, list[str], list[tuple[str, str, float, float, float]]]:
    """Compute the numerical Jacobian of the end-effector position.

    Perturbs each tolerance-bearing parameter (joint angle or link length)
    by ±delta and computes the central-difference derivative.

    Returns:
        (jacobian, param_names, param_info)
        jacobian: (3 x n_params) matrix.
        param_names: list of parameter names.
        param_info: list of (name, kind, nominal, plus_tol, minus_tol).
    """
    params = linkage.parameter_list()
    n_params = len(params)
    J = np.zeros((3, n_params))
    nominal_pos = linkage.end_effector_position()

    for i, (name, kind, nom_val, pt, mt) in enumerate(params):
        # Positive perturbation
        if kind == "joint":
            jv_plus = {name: nom_val + delta}
            pos_plus = linkage.end_effector_position(joint_values=jv_plus)
            jv_minus = {name: nom_val - delta}
            pos_minus = linkage.end_effector_position(joint_values=jv_minus)
        else:  # link
            ll_plus = {name: nom_val + delta}
            pos_plus = linkage.end_effector_position(link_lengths=ll_plus)
            ll_minus = {name: nom_val - delta}
            pos_minus = linkage.end_effector_position(link_lengths=ll_minus)

        J[:, i] = (pos_plus - pos_minus) / (2.0 * delta)

    param_names = [p[0] for p in params]
    return J, param_names, params


# ---------------------------------------------------------------------------
# Worst-Case analysis
# ---------------------------------------------------------------------------

def linkage_worst_case(linkage: Linkage) -> LinkageAnalysisResult:
    """Worst-case linkage tolerance analysis using the Jacobian.

    For each parameter, the maximum effect on the end-effector is
    |J_col| * half_tolerance, summed across all parameters per axis.
    """
    J, param_names, params = _compute_jacobian(linkage)
    nominal_pos = linkage.end_effector_position()

    total_plus = np.zeros(3)
    total_minus = np.zeros(3)
    sensitivity = []

    for i, (name, kind, nom_val, pt, mt) in enumerate(params):
        half = (pt + mt) / 2.0
        shift = (pt - mt) / 2.0
        col = J[:, i]

        sensitivity.append((name, col.copy()))

        # Per-axis worst-case accumulation
        abs_col = np.abs(col)
        total_plus += abs_col * half + col * shift
        total_minus += abs_col * half - col * shift

    radial = np.linalg.norm(total_plus)

    return LinkageAnalysisResult(
        method="Worst-Case",
        nominal_position=nominal_pos,
        position_max=nominal_pos + total_plus,
        position_min=nominal_pos - total_minus,
        plus_tolerance=total_plus,
        minus_tolerance=total_minus,
        radial_tolerance=radial,
        jacobian=J,
        param_names=param_names,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# RSS analysis
# ---------------------------------------------------------------------------

def linkage_rss(linkage: Linkage, sigma: float = 3.0) -> LinkageAnalysisResult:
    """RSS linkage tolerance analysis using the Jacobian.

    Each parameter's tolerance band is treated as ±(sigma) standard
    deviations. The per-axis RSS tolerance is sigma * sqrt(sum of variances).
    """
    J, param_names, params = _compute_jacobian(linkage)
    nominal_pos = linkage.end_effector_position()

    mean_shift = np.zeros(3)
    sum_var = np.zeros(3)
    sensitivity = []

    for i, (name, kind, nom_val, pt, mt) in enumerate(params):
        half = (pt + mt) / 2.0
        shift = (pt - mt) / 2.0
        col = J[:, i]

        sensitivity.append((name, col.copy()))

        # Find the contributor's sigma value
        src = _find_source(linkage, name, kind)
        src_sigma = src.sigma if src else 3.0

        std_i = half / src_sigma
        mean_shift += col * shift
        sum_var += (col * std_i) ** 2

    rss_std = np.sqrt(sum_var)
    rss_tol = sigma * rss_std
    adjusted_nominal = nominal_pos + mean_shift
    radial = np.linalg.norm(rss_tol)

    return LinkageAnalysisResult(
        method="RSS",
        nominal_position=adjusted_nominal,
        position_max=adjusted_nominal + rss_tol,
        position_min=adjusted_nominal - rss_tol,
        plus_tolerance=rss_tol,
        minus_tolerance=rss_tol,
        radial_tolerance=radial,
        jacobian=J,
        param_names=param_names,
        sensitivity=sensitivity,
    )


def _find_source(linkage: Linkage, name: str, kind: str):
    """Find the Joint or Link object by name."""
    if kind == "joint":
        for j in linkage.joints:
            if j.name == name:
                return j
    else:
        for lk in linkage.links:
            if lk.name == name:
                return lk
    return None


# ---------------------------------------------------------------------------
# Monte Carlo analysis
# ---------------------------------------------------------------------------

def linkage_monte_carlo(
    linkage: Linkage,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> LinkageAnalysisResult:
    """Monte Carlo linkage tolerance analysis.

    Samples each tolerance-bearing parameter from its distribution,
    computes full forward kinematics for each sample, and collects
    the end-effector position distribution.
    """
    rng = np.random.default_rng(seed)
    linkage.validate()

    nominal_pos = linkage.end_effector_position()

    # Build parameter sample arrays
    joint_samples: dict[str, np.ndarray] = {}
    link_samples: dict[str, np.ndarray] = {}

    for j in linkage.joints:
        if not j.has_tolerance:
            continue
        nom = j.nominal if isinstance(j.nominal, (int, float)) else 0.0
        half = j.half_tolerance
        shift = j.midpoint_shift
        center = nom + shift
        samples = _sample(rng, center, half, j.sigma, j.distribution, n_samples)
        joint_samples[j.name] = samples

    for lk in linkage.links:
        if not lk.has_tolerance:
            continue
        center = lk.length + lk.midpoint_shift
        half = lk.half_tolerance
        samples = _sample(rng, center, half, lk.sigma, lk.distribution, n_samples)
        link_samples[lk.name] = samples

    # Run forward kinematics for all samples
    positions = np.zeros((n_samples, 3))

    for s in range(n_samples):
        jv = {name: float(arr[s]) for name, arr in joint_samples.items()}
        ll = {name: float(arr[s]) for name, arr in link_samples.items()}
        positions[s, :] = linkage.end_effector_position(
            joint_values=jv,
            link_lengths=ll,
        )

    mc_mean = np.mean(positions, axis=0)
    mc_std = np.std(positions, axis=0, ddof=1)
    mc_cov = np.cov(positions, rowvar=False)
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)

    # Also compute Jacobian for sensitivity info
    J, param_names, params = _compute_jacobian(linkage)
    sensitivity = [(pn, J[:, i].copy()) for i, pn in enumerate(param_names)]

    radial_std = np.sqrt(np.sum(mc_std ** 2))

    return LinkageAnalysisResult(
        method="Monte Carlo",
        nominal_position=nominal_pos,
        position_max=pos_max,
        position_min=pos_min,
        plus_tolerance=pos_max - mc_mean,
        minus_tolerance=mc_mean - pos_min,
        radial_tolerance=radial_std * 3.0,
        jacobian=J,
        param_names=param_names,
        sensitivity=sensitivity,
        mc_samples=positions,
        mc_mean=mc_mean,
        mc_std=mc_std,
        mc_cov=mc_cov,
    )


def _sample(
    rng: np.random.Generator,
    center: float,
    half: float,
    sigma: float,
    distribution: Distribution,
    n: int,
) -> np.ndarray:
    """Generate n samples from the given distribution."""
    if distribution == Distribution.NORMAL:
        std = half / sigma
        return rng.normal(loc=center, scale=std, size=n)
    elif distribution == Distribution.UNIFORM:
        return rng.uniform(low=center - half, high=center + half, size=n)
    elif distribution == Distribution.TRIANGULAR:
        return rng.triangular(left=center - half, mode=center, right=center + half, size=n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def analyze_linkage(
    linkage: Linkage,
    methods: Optional[list[str]] = None,
    sigma: float = 3.0,
    mc_samples: int = 100_000,
    mc_seed: Optional[int] = None,
) -> dict[str, LinkageAnalysisResult]:
    """Run one or more analysis methods on a linkage.

    Args:
        linkage: The linkage to analyze.
        methods: List of method names ("wc", "rss", "mc"). Defaults to all.
        sigma: Sigma level for RSS analysis.
        mc_samples: Number of Monte Carlo samples.
        mc_seed: Random seed for reproducibility.

    Returns:
        Dict mapping method name to LinkageAnalysisResult.
    """
    if methods is None:
        methods = ["wc", "rss", "mc"]

    results: dict[str, LinkageAnalysisResult] = {}

    for m in methods:
        key = m.lower().strip()
        if key in ("wc", "worst-case", "worst_case"):
            results["wc"] = linkage_worst_case(linkage)
        elif key == "rss":
            results["rss"] = linkage_rss(linkage, sigma=sigma)
        elif key in ("mc", "monte-carlo", "monte_carlo"):
            results["mc"] = linkage_monte_carlo(
                linkage, n_samples=mc_samples, seed=mc_seed
            )
        else:
            raise ValueError(f"Unknown analysis method: {m!r}")

    return results
