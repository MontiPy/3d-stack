"""Tolerance stack analysis engine supporting WC, RSS, and Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tolerance_stack.models import Contributor, ContributorType, Distribution, ToleranceStack


@dataclass
class AnalysisResult:
    """Results from a tolerance stack analysis.

    Attributes:
        method: Name of the analysis method used.
        nominal_gap: Nominal gap value.
        gap_max: Maximum gap (worst-case upper bound or statistical bound).
        gap_min: Minimum gap (worst-case lower bound or statistical bound).
        plus_tolerance: Upper tolerance on the gap.
        minus_tolerance: Lower tolerance on the gap (positive value).
        sigma_level: Sigma level for statistical results (None for WC).
        percent_yield: Estimated yield percentage (None for WC).
        sensitivity: Per-contributor sensitivity (projection factor * sign).
        mc_samples: Monte Carlo sample array if applicable.
        mc_mean: Monte Carlo mean if applicable.
        mc_std: Monte Carlo standard deviation if applicable.
    """
    method: str
    nominal_gap: float
    gap_max: float
    gap_min: float
    plus_tolerance: float
    minus_tolerance: float
    sigma_level: Optional[float] = None
    percent_yield: Optional[float] = None
    sensitivity: list[tuple[str, float]] = field(default_factory=list)
    mc_samples: Optional[np.ndarray] = None
    mc_mean: Optional[float] = None
    mc_std: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"=== {self.method} Analysis ===",
            f"  Nominal gap:      {self.nominal_gap:+.6f}",
            f"  Gap range:        [{self.gap_min:+.6f}, {self.gap_max:+.6f}]",
            f"  Upper tolerance:  +{self.plus_tolerance:.6f}",
            f"  Lower tolerance:  -{self.minus_tolerance:.6f}",
        ]
        if self.sigma_level is not None:
            lines.append(f"  Sigma level:      {self.sigma_level:.1f}")
        if self.percent_yield is not None:
            lines.append(f"  Est. yield:       {self.percent_yield:.4f}%")
        if self.mc_mean is not None:
            lines.append(f"  MC mean:          {self.mc_mean:+.6f}")
            lines.append(f"  MC std dev:       {self.mc_std:.6f}")
        if self.sensitivity:
            lines.append("  Sensitivity (contribution to gap per unit):")
            for name, sens in self.sensitivity:
                lines.append(f"    {name:30s}  {sens:+.4f}")
        return "\n".join(lines)


def _projection_factor(contributor: Contributor, closure_dir: np.ndarray) -> float:
    """Compute the scalar projection of a contributor's direction onto the closure direction.

    For angular contributors, this is a linearized sensitivity at the nominal.
    """
    c_dir = np.array(contributor.direction, dtype=float)
    dot = float(np.dot(c_dir, closure_dir))

    if contributor.contributor_type == ContributorType.ANGULAR:
        # For angular contributions the sensitivity is the sine-projection
        # scaled by the nominal lever arm. The nominal already encodes the
        # lever arm length, and the tolerance is the angular variation in
        # degrees. Convert to radians for the sensitivity.
        return dot  # caller handles deg->rad on the tolerance value

    return dot


def _effective_tolerance(contributor: Contributor) -> tuple[float, float]:
    """Return (half_tol, midpoint_shift) in linear units.

    For angular contributors the tolerance is converted from degrees to
    radians (small-angle linearisation).
    """
    half = contributor.half_tolerance
    shift = contributor.midpoint_shift

    if contributor.contributor_type == ContributorType.ANGULAR:
        half = half * (np.pi / 180.0)
        shift = shift * (np.pi / 180.0)

    return half, shift


# ---------------------------------------------------------------------------
# Worst-Case analysis
# ---------------------------------------------------------------------------

def worst_case(stack: ToleranceStack) -> AnalysisResult:
    """Perform worst-case (min/max) tolerance stack analysis.

    Every contributor is assumed to be at its extreme limit simultaneously.
    """
    closure_dir = np.array(stack.closure_direction, dtype=float)
    nominal_gap = 0.0
    total_plus = 0.0
    total_minus = 0.0
    sensitivity = []

    for c in stack.contributors:
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf
        sensitivity.append((c.name, sens))

        half, shift = _effective_tolerance(c)

        # Nominal contribution
        nominal_gap += sens * c.nominal

        # Tolerance contribution — absolute value of sensitivity * half tolerance
        abs_sens = abs(sens)
        total_plus += abs_sens * half + sens * shift
        total_minus += abs_sens * half - sens * shift

    if stack.gap_nominal is not None:
        nominal_gap = stack.gap_nominal

    return AnalysisResult(
        method="Worst-Case",
        nominal_gap=nominal_gap,
        gap_max=nominal_gap + total_plus,
        gap_min=nominal_gap - total_minus,
        plus_tolerance=total_plus,
        minus_tolerance=total_minus,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# RSS (Root Sum of Squares) analysis
# ---------------------------------------------------------------------------

def rss(stack: ToleranceStack, sigma: float = 3.0) -> AnalysisResult:
    """Perform RSS statistical tolerance stack analysis.

    Assumes each contributor's tolerance band represents ±(sigma) standard
    deviations of a normal distribution. The RSS gap tolerance is then
    computed at the same sigma level.
    """
    closure_dir = np.array(stack.closure_direction, dtype=float)
    nominal_gap = 0.0
    mean_shift = 0.0
    sum_var = 0.0
    sensitivity = []

    for c in stack.contributors:
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf
        sensitivity.append((c.name, sens))

        half, shift = _effective_tolerance(c)

        nominal_gap += sens * c.nominal
        mean_shift += sens * shift

        # Variance contribution: (sens * half / c.sigma)^2
        std_i = half / c.sigma
        sum_var += (sens * std_i) ** 2

    if stack.gap_nominal is not None:
        nominal_gap = stack.gap_nominal

    rss_std = np.sqrt(sum_var)
    rss_tol = sigma * rss_std

    adjusted_nominal = nominal_gap + mean_shift

    # Yield estimate assuming normal distribution
    from scipy.stats import norm as _norm  # deferred import
    percent_yield = (_norm.cdf(sigma) - _norm.cdf(-sigma)) * 100.0

    return AnalysisResult(
        method="RSS",
        nominal_gap=adjusted_nominal,
        gap_max=adjusted_nominal + rss_tol,
        gap_min=adjusted_nominal - rss_tol,
        plus_tolerance=rss_tol,
        minus_tolerance=rss_tol,
        sigma_level=sigma,
        percent_yield=percent_yield,
        sensitivity=sensitivity,
    )


def rss_no_scipy(stack: ToleranceStack, sigma: float = 3.0) -> AnalysisResult:
    """RSS analysis without scipy dependency — yield estimate omitted."""
    closure_dir = np.array(stack.closure_direction, dtype=float)
    nominal_gap = 0.0
    mean_shift = 0.0
    sum_var = 0.0
    sensitivity = []

    for c in stack.contributors:
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf
        sensitivity.append((c.name, sens))

        half, shift = _effective_tolerance(c)

        nominal_gap += sens * c.nominal
        mean_shift += sens * shift
        std_i = half / c.sigma
        sum_var += (sens * std_i) ** 2

    if stack.gap_nominal is not None:
        nominal_gap = stack.gap_nominal

    rss_std = np.sqrt(sum_var)
    rss_tol = sigma * rss_std
    adjusted_nominal = nominal_gap + mean_shift

    return AnalysisResult(
        method="RSS",
        nominal_gap=adjusted_nominal,
        gap_max=adjusted_nominal + rss_tol,
        gap_min=adjusted_nominal - rss_tol,
        plus_tolerance=rss_tol,
        minus_tolerance=rss_tol,
        sigma_level=sigma,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# Monte Carlo analysis
# ---------------------------------------------------------------------------

def monte_carlo(
    stack: ToleranceStack,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> AnalysisResult:
    """Perform Monte Carlo tolerance stack analysis.

    Each contributor is sampled according to its specified distribution,
    and the resulting gap distribution is computed.
    """
    rng = np.random.default_rng(seed)
    closure_dir = np.array(stack.closure_direction, dtype=float)

    gap_samples = np.zeros(n_samples)
    sensitivity = []

    for c in stack.contributors:
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf
        sensitivity.append((c.name, sens))

        half, shift = _effective_tolerance(c)
        center = c.nominal + shift

        # Generate samples for this contributor
        if c.distribution == Distribution.NORMAL:
            std = half / c.sigma
            samples = rng.normal(loc=center, scale=std, size=n_samples)
        elif c.distribution == Distribution.UNIFORM:
            lo = center - half
            hi = center + half
            samples = rng.uniform(low=lo, high=hi, size=n_samples)
        elif c.distribution == Distribution.TRIANGULAR:
            lo = center - half
            hi = center + half
            samples = rng.triangular(left=lo, mode=center, right=hi, size=n_samples)
        else:
            raise ValueError(f"Unknown distribution: {c.distribution}")

        gap_samples += sens * samples

    mc_mean = float(np.mean(gap_samples))
    mc_std = float(np.std(gap_samples, ddof=1))
    gap_min = float(np.min(gap_samples))
    gap_max = float(np.max(gap_samples))

    # Compute nominal gap for reference
    nominal_gap = 0.0
    for c in stack.contributors:
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf
        nominal_gap += sens * c.nominal
    if stack.gap_nominal is not None:
        nominal_gap = stack.gap_nominal

    return AnalysisResult(
        method="Monte Carlo",
        nominal_gap=nominal_gap,
        gap_max=gap_max,
        gap_min=gap_min,
        plus_tolerance=gap_max - mc_mean,
        minus_tolerance=mc_mean - gap_min,
        sensitivity=sensitivity,
        mc_samples=gap_samples,
        mc_mean=mc_mean,
        mc_std=mc_std,
    )


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def analyze_stack(
    stack: ToleranceStack,
    methods: Optional[list[str]] = None,
    sigma: float = 3.0,
    mc_samples: int = 100_000,
    mc_seed: Optional[int] = None,
) -> dict[str, AnalysisResult]:
    """Run one or more analysis methods on a tolerance stack.

    Args:
        stack: The tolerance stack to analyze.
        methods: List of method names ("wc", "rss", "mc"). Defaults to all.
        sigma: Sigma level for RSS analysis.
        mc_samples: Number of Monte Carlo samples.
        mc_seed: Random seed for reproducibility.

    Returns:
        Dict mapping method name to AnalysisResult.
    """
    if methods is None:
        methods = ["wc", "rss", "mc"]

    results: dict[str, AnalysisResult] = {}

    for m in methods:
        key = m.lower().strip()
        if key in ("wc", "worst-case", "worst_case"):
            results["wc"] = worst_case(stack)
        elif key == "rss":
            try:
                results["rss"] = rss(stack, sigma=sigma)
            except ImportError:
                results["rss"] = rss_no_scipy(stack, sigma=sigma)
        elif key in ("mc", "monte-carlo", "monte_carlo"):
            results["mc"] = monte_carlo(stack, n_samples=mc_samples, seed=mc_seed)
        else:
            raise ValueError(f"Unknown analysis method: {m!r}")

    return results
