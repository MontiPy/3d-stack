"""Tolerance optimization, DOE, and critical tolerance identification.

Provides tools matching 3DCS AAO and VisVSA DOE capabilities:
- Tolerance optimization (minimize cost subject to quality constraints)
- Design of Experiments (parameter sweeps and interaction analysis)
- Critical Tolerance Identifier (rank tolerances across all measurements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Critical Tolerance Identifier
# ---------------------------------------------------------------------------

@dataclass
class CriticalToleranceResult:
    """Result from Critical Tolerance Identifier analysis.

    Attributes:
        rankings: List of (param_name, score) sorted by criticality.
        contributions: Dict of param_name -> dict of measurement contributions.
    """
    rankings: list[tuple[str, float]] = field(default_factory=list)
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Critical Tolerance Identifier ==="]
        for i, (name, score) in enumerate(self.rankings[:20], 1):
            lines.append(f"  {i:3d}. {name:45s}  score={score:.4f}")
        return "\n".join(lines)


def critical_tolerance_identifier(
    sensitivity_maps: dict[str, list[tuple[str, float]]],
    tolerance_maps: dict[str, list[float]],
) -> CriticalToleranceResult:
    """Identify the most critical tolerances across multiple measurements.

    Computes a composite criticality score for each tolerance parameter
    based on its contribution across all measurements.

    Args:
        sensitivity_maps: {measurement_name: [(param_name, sensitivity), ...]}
        tolerance_maps: {measurement_name: [half_tol, ...]} (same order)

    Returns:
        CriticalToleranceResult with ranked tolerances.
    """
    param_scores: dict[str, float] = {}
    param_contributions: dict[str, dict[str, float]] = {}

    for meas_name, sens_list in sensitivity_maps.items():
        tols = tolerance_maps.get(meas_name, [])
        if len(tols) != len(sens_list):
            continue

        total_var = sum((s * t) ** 2 for (_, s), t in zip(sens_list, tols))
        if total_var < 1e-30:
            continue

        for (pname, sens), tol in zip(sens_list, tols):
            contribution = (sens * tol) ** 2 / total_var * 100.0
            if pname not in param_scores:
                param_scores[pname] = 0.0
                param_contributions[pname] = {}
            param_scores[pname] += contribution
            param_contributions[pname][meas_name] = contribution

    rankings = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
    return CriticalToleranceResult(rankings=rankings, contributions=param_contributions)


# ---------------------------------------------------------------------------
# Design of Experiments
# ---------------------------------------------------------------------------

@dataclass
class DOEFactor:
    """A factor in a DOE study.

    Attributes:
        name: Factor name.
        levels: List of values to test.
        nominal: Nominal value.
    """
    name: str
    levels: list[float]
    nominal: float = 0.0


@dataclass
class DOEResult:
    """Result from a DOE study.

    Attributes:
        factor_names: Names of factors.
        runs: List of {factor_name: value} dicts for each run.
        responses: Array of response values (one per run).
        main_effects: Dict of factor_name -> effect magnitude.
        interactions: Dict of (factor_a, factor_b) -> interaction magnitude.
    """
    factor_names: list[str] = field(default_factory=list)
    runs: list[dict[str, float]] = field(default_factory=list)
    responses: np.ndarray = field(default_factory=lambda: np.array([]))
    main_effects: dict[str, float] = field(default_factory=dict)
    interactions: dict[tuple[str, str], float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Design of Experiments ===",
            f"  Factors: {len(self.factor_names)}",
            f"  Runs:    {len(self.runs)}",
            "",
            "  Main Effects (sorted by magnitude):",
        ]
        sorted_me = sorted(self.main_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, effect in sorted_me:
            lines.append(f"    {name:40s}  {effect:+.6f}")

        if self.interactions:
            lines.append("")
            lines.append("  Top Interactions:")
            sorted_int = sorted(self.interactions.items(), key=lambda x: abs(x[1]), reverse=True)
            for (a, b), effect in sorted_int[:10]:
                lines.append(f"    {a} x {b:30s}  {effect:+.6f}")

        return "\n".join(lines)


def hlm_sensitivity(
    evaluate_fn: Callable[[dict[str, float]], float],
    factors: list[DOEFactor],
) -> DOEResult:
    """High-Low-Median sensitivity analysis (3DCS HLM method).

    Tests each factor at its high, low, and nominal (median) values while
    holding all others at nominal. Computes main effect as range of response.

    Args:
        evaluate_fn: Function mapping {factor_name: value} -> scalar response.
        factors: List of DOEFactor objects.

    Returns:
        DOEResult with main effects.
    """
    nominal_inputs = {f.name: f.nominal for f in factors}
    nominal_response = evaluate_fn(nominal_inputs)

    runs = [nominal_inputs.copy()]
    responses = [nominal_response]
    main_effects = {}

    for factor in factors:
        factor_responses = []
        for level in factor.levels:
            inputs = nominal_inputs.copy()
            inputs[factor.name] = level
            resp = evaluate_fn(inputs)
            runs.append(inputs)
            responses.append(resp)
            factor_responses.append(resp)

        if factor_responses:
            effect = max(factor_responses) - min(factor_responses)
            main_effects[factor.name] = effect

    return DOEResult(
        factor_names=[f.name for f in factors],
        runs=runs,
        responses=np.array(responses),
        main_effects=main_effects,
    )


def full_factorial_doe(
    evaluate_fn: Callable[[dict[str, float]], float],
    factors: list[DOEFactor],
) -> DOEResult:
    """Full factorial DOE with interaction analysis.

    Tests all combinations of factor levels. Computes main effects and
    two-factor interactions.

    Args:
        evaluate_fn: Function mapping {factor_name: value} -> scalar response.
        factors: List of DOEFactor objects (each with 2-3 levels).

    Returns:
        DOEResult with main effects and interactions.
    """
    n_factors = len(factors)
    level_counts = [len(f.levels) for f in factors]
    n_runs = 1
    for c in level_counts:
        n_runs *= c

    # Generate all combinations
    runs = []
    responses = np.zeros(n_runs)

    # Build index arrays for each combination
    indices = np.zeros((n_runs, n_factors), dtype=int)
    repeat = n_runs
    for fi in range(n_factors):
        repeat //= level_counts[fi]
        tile = n_runs // (level_counts[fi] * repeat)
        pattern = np.repeat(np.arange(level_counts[fi]), repeat)
        indices[:, fi] = np.tile(pattern, tile)

    for r in range(n_runs):
        inputs = {}
        for fi, factor in enumerate(factors):
            inputs[factor.name] = factor.levels[indices[r, fi]]
        runs.append(inputs)
        responses[r] = evaluate_fn(inputs)

    # Compute main effects: average response at high vs low
    main_effects = {}
    for fi, factor in enumerate(factors):
        if len(factor.levels) >= 2:
            low_val = factor.levels[0]
            high_val = factor.levels[-1]
            low_mask = indices[:, fi] == 0
            high_mask = indices[:, fi] == (level_counts[fi] - 1)
            main_effects[factor.name] = float(
                np.mean(responses[high_mask]) - np.mean(responses[low_mask])
            )

    # Two-factor interactions
    interactions = {}
    for fi in range(n_factors):
        for fj in range(fi + 1, n_factors):
            if level_counts[fi] >= 2 and level_counts[fj] >= 2:
                # Interaction = change in effect of fi across levels of fj
                fj_low = indices[:, fj] == 0
                fj_high = indices[:, fj] == (level_counts[fj] - 1)

                fi_low_at_fj_low = np.mean(responses[(indices[:, fi] == 0) & fj_low])
                fi_high_at_fj_low = np.mean(responses[(indices[:, fi] == (level_counts[fi] - 1)) & fj_low])
                fi_low_at_fj_high = np.mean(responses[(indices[:, fi] == 0) & fj_high])
                fi_high_at_fj_high = np.mean(responses[(indices[:, fi] == (level_counts[fi] - 1)) & fj_high])

                effect_at_low = fi_high_at_fj_low - fi_low_at_fj_low
                effect_at_high = fi_high_at_fj_high - fi_low_at_fj_high
                interaction = (effect_at_high - effect_at_low) / 2.0
                interactions[(factors[fi].name, factors[fj].name)] = float(interaction)

    return DOEResult(
        factor_names=[f.name for f in factors],
        runs=runs,
        responses=responses,
        main_effects=main_effects,
        interactions=interactions,
    )


# ---------------------------------------------------------------------------
# Tolerance Optimizer
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from tolerance optimization.

    Attributes:
        original_tolerances: Dict of param_name -> original half_tol.
        optimized_tolerances: Dict of param_name -> optimized half_tol.
        original_variation: Original assembly variation (sigma or WC).
        optimized_variation: Optimized assembly variation.
        original_cost: Original manufacturing cost estimate.
        optimized_cost: Optimized manufacturing cost estimate.
        iterations: Number of optimizer iterations.
        converged: Whether the optimizer converged.
    """
    original_tolerances: dict[str, float] = field(default_factory=dict)
    optimized_tolerances: dict[str, float] = field(default_factory=dict)
    original_variation: float = 0.0
    optimized_variation: float = 0.0
    original_cost: float = 0.0
    optimized_cost: float = 0.0
    iterations: int = 0
    converged: bool = False

    def summary(self) -> str:
        lines = [
            "=== Tolerance Optimization ===",
            f"  Converged:          {self.converged}",
            f"  Iterations:         {self.iterations}",
            f"  Original variation: {self.original_variation:.6f}",
            f"  Optimized variation:{self.optimized_variation:.6f}",
            f"  Original cost:      {self.original_cost:.2f}",
            f"  Optimized cost:     {self.optimized_cost:.2f}",
            f"  Cost reduction:     {(1 - self.optimized_cost/max(self.original_cost, 1e-10))*100:.1f}%",
            "",
            "  Tolerance Changes:",
        ]
        for name in sorted(self.original_tolerances.keys()):
            orig = self.original_tolerances[name]
            opt = self.optimized_tolerances.get(name, orig)
            change = (opt - orig) / max(orig, 1e-10) * 100
            marker = " **" if abs(change) > 5 else ""
            lines.append(f"    {name:40s}  {orig:.6f} -> {opt:.6f}  ({change:+.1f}%){marker}")
        return "\n".join(lines)


def optimize_tolerances(
    sensitivity: list[tuple[str, float]],
    tolerances: dict[str, float],
    target_variation: float,
    cost_fn: Optional[Callable[[str, float], float]] = None,
    min_tol_fraction: float = 0.2,
    max_tol_fraction: float = 3.0,
    max_iterations: int = 100,
) -> OptimizationResult:
    """Optimize tolerance allocations to minimize cost while meeting variation target.

    Uses a Lagrange-multiplier-inspired iterative approach:
    - Tolerances with high sensitivity get tightened
    - Tolerances with low sensitivity get loosened
    - Cost function penalizes tight tolerances

    Args:
        sensitivity: List of (param_name, sensitivity_value).
        tolerances: Dict of param_name -> current half_tolerance.
        target_variation: Desired RSS variation (at 3-sigma).
        cost_fn: Optional function(param_name, half_tol) -> cost.
            Defaults to 1/tol^2 cost model (tighter = exponentially more expensive).
        min_tol_fraction: Minimum tolerance as fraction of original (default 0.2).
        max_tol_fraction: Maximum tolerance as fraction of original (default 3.0).
        max_iterations: Maximum optimization iterations.

    Returns:
        OptimizationResult.
    """
    if cost_fn is None:
        def cost_fn(name: str, half_tol: float) -> float:
            return 1.0 / max(half_tol, 1e-10) ** 2

    param_names = [name for name, _ in sensitivity]
    sens_values = {name: abs(s) for name, s in sensitivity}

    # Current state
    current_tols = {name: tolerances.get(name, 0.01) for name in param_names}
    original_tols = dict(current_tols)

    # Compute original variation and cost
    def compute_rss_var(tols):
        total = 0.0
        for name in param_names:
            s = sens_values.get(name, 0.0)
            t = tols.get(name, 0.01)
            total += (s * t) ** 2
        return np.sqrt(total)

    def compute_cost(tols):
        return sum(cost_fn(name, tols[name]) for name in param_names)

    original_variation = compute_rss_var(current_tols)
    original_cost = compute_cost(current_tols)

    converged = False
    for iteration in range(max_iterations):
        current_var = compute_rss_var(current_tols)

        if abs(current_var - target_variation) / max(target_variation, 1e-10) < 0.01:
            converged = True
            break

        # Adjust tolerances: redistribute based on sensitivity-weighted cost
        for name in param_names:
            s = sens_values.get(name, 0.0)
            if s < 1e-12:
                continue

            orig = original_tols[name]
            min_t = orig * min_tol_fraction
            max_t = orig * max_tol_fraction

            if current_var > target_variation:
                # Need to tighten: reduce high-sensitivity tolerances more
                factor = 1.0 - 0.05 * s / max(max(sens_values.values()), 1e-10)
                current_tols[name] = max(min_t, current_tols[name] * factor)
            else:
                # Can loosen: increase low-sensitivity tolerances
                factor = 1.0 + 0.05 * (1.0 - s / max(max(sens_values.values()), 1e-10))
                current_tols[name] = min(max_t, current_tols[name] * factor)

    return OptimizationResult(
        original_tolerances=original_tols,
        optimized_tolerances=current_tols,
        original_variation=original_variation,
        optimized_variation=compute_rss_var(current_tols),
        original_cost=original_cost,
        optimized_cost=compute_cost(current_tols),
        iterations=iteration + 1 if not converged else iteration,
        converged=converged,
    )
