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


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling DOE
# ---------------------------------------------------------------------------

def latin_hypercube_doe(
    evaluate_fn: Callable[[dict[str, float]], float],
    factors: list[DOEFactor],
    n_samples: int = 100,
    seed: int = 42,
) -> DOEResult:
    """Latin Hypercube Sampling DOE.

    Provides space-filling sampling with better coverage than random sampling
    and far fewer runs than full factorial for large factor counts.

    Each factor's range is divided into n_samples equal intervals, and exactly
    one sample is placed in each interval (stratified sampling).

    Args:
        evaluate_fn: Function mapping {factor_name: value} -> scalar response.
        factors: List of DOEFactor objects (each with at least 2 levels
                 defining [low, high] range).
        n_samples: Number of LHS samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DOEResult with runs, responses, and estimated main effects.
    """
    rng = np.random.default_rng(seed)
    n_factors = len(factors)

    # Generate LHS design matrix
    lhs_matrix = np.zeros((n_samples, n_factors))
    for fi in range(n_factors):
        perm = rng.permutation(n_samples)
        for si in range(n_samples):
            lhs_matrix[si, fi] = (perm[si] + rng.random()) / n_samples

    # Scale to factor ranges
    runs = []
    responses = np.zeros(n_samples)

    for si in range(n_samples):
        inputs = {}
        for fi, factor in enumerate(factors):
            low = min(factor.levels)
            high = max(factor.levels)
            val = low + lhs_matrix[si, fi] * (high - low)
            inputs[factor.name] = val
        runs.append(inputs)
        responses[si] = evaluate_fn(inputs)

    # Estimate main effects via correlation
    main_effects = {}
    for fi, factor in enumerate(factors):
        factor_values = lhs_matrix[:, fi]
        correlation = np.corrcoef(factor_values, responses)[0, 1]
        low = min(factor.levels)
        high = max(factor.levels)
        main_effects[factor.name] = float(correlation * (high - low))

    return DOEResult(
        factor_names=[f.name for f in factors],
        runs=runs,
        responses=responses,
        main_effects=main_effects,
    )


# ---------------------------------------------------------------------------
# Response Surface Methodology (RSM)
# ---------------------------------------------------------------------------

@dataclass
class RSMResult:
    """Result from Response Surface Methodology analysis.

    Attributes:
        coefficients: Dict of term_name -> coefficient.
        r_squared: R-squared goodness of fit.
        adj_r_squared: Adjusted R-squared.
        predicted_optimum: Dict of factor_name -> optimal value.
        optimum_response: Predicted response at optimum.
        main_effects: Linear coefficients.
        quadratic_effects: Quadratic coefficients.
        interaction_effects: Cross-term coefficients.
        residuals: Array of residuals.
    """
    coefficients: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    predicted_optimum: dict[str, float] = field(default_factory=dict)
    optimum_response: float = 0.0
    main_effects: dict[str, float] = field(default_factory=dict)
    quadratic_effects: dict[str, float] = field(default_factory=dict)
    interaction_effects: dict[tuple[str, str], float] = field(default_factory=dict)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=== Response Surface Methodology ===",
            f"  R-squared:      {self.r_squared:.6f}",
            f"  Adj R-squared:  {self.adj_r_squared:.6f}",
            "",
            "  Linear (main) effects:",
        ]
        for name, coeff in sorted(self.main_effects.items(),
                                   key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {name:40s}  {coeff:+.6f}")

        if self.quadratic_effects:
            lines.append("")
            lines.append("  Quadratic effects:")
            for name, coeff in sorted(self.quadratic_effects.items(),
                                       key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"    {name}^2  {' ' * max(0, 37 - len(name))}  {coeff:+.6f}")

        if self.interaction_effects:
            lines.append("")
            lines.append("  Interactions:")
            sorted_int = sorted(self.interaction_effects.items(),
                                key=lambda x: abs(x[1]), reverse=True)
            for (a, b), coeff in sorted_int[:10]:
                lines.append(f"    {a} x {b}  {' ' * max(0, 30 - len(a) - len(b))}  {coeff:+.6f}")

        if self.predicted_optimum:
            lines.append("")
            lines.append(f"  Predicted optimum (response={self.optimum_response:.6f}):")
            for name, val in self.predicted_optimum.items():
                lines.append(f"    {name:40s}  {val:.6f}")

        return "\n".join(lines)


def response_surface_doe(
    evaluate_fn: Callable[[dict[str, float]], float],
    factors: list[DOEFactor],
    n_samples: int = 0,
    seed: int = 42,
    include_quadratic: bool = True,
) -> RSMResult:
    """Response Surface Methodology with quadratic model fitting.

    Generates a design matrix (Central Composite or LHS), evaluates the
    response function, and fits a second-order polynomial model:
        y = b0 + sum(bi*xi) + sum(bii*xi^2) + sum(bij*xi*xj)

    Args:
        evaluate_fn: Function mapping {factor_name: value} -> scalar response.
        factors: List of DOEFactor objects.
        n_samples: Number of samples (0 = auto-select based on factor count).
        seed: Random seed.
        include_quadratic: Include x^2 terms in model.

    Returns:
        RSMResult with coefficients, R-squared, and predicted optimum.
    """
    n_factors = len(factors)

    if n_samples <= 0:
        # Auto: use ~10 samples per coefficient for good fit
        n_terms = 1 + n_factors  # intercept + linear
        if include_quadratic:
            n_terms += n_factors + n_factors * (n_factors - 1) // 2
        n_samples = max(n_terms * 3, 2 ** n_factors + 2 * n_factors + 1)
        n_samples = min(n_samples, 1000)  # Cap for performance

    # Generate LHS design
    rng = np.random.default_rng(seed)
    lhs = np.zeros((n_samples, n_factors))
    for fi in range(n_factors):
        perm = rng.permutation(n_samples)
        for si in range(n_samples):
            lhs[si, fi] = (perm[si] + rng.random()) / n_samples

    # Scale to [-1, 1] coded space
    coded = lhs * 2 - 1

    # Add center point
    coded = np.vstack([coded, np.zeros(n_factors)])
    n_total = len(coded)

    # Evaluate
    runs = []
    responses = np.zeros(n_total)
    for si in range(n_total):
        inputs = {}
        for fi, factor in enumerate(factors):
            low = min(factor.levels)
            high = max(factor.levels)
            mid = (low + high) / 2
            span = (high - low) / 2
            inputs[factor.name] = mid + coded[si, fi] * span
        runs.append(inputs)
        responses[si] = evaluate_fn(inputs)

    # Build design matrix for regression
    # Columns: [1, x1, x2, ..., x1^2, x2^2, ..., x1*x2, ...]
    columns = [np.ones(n_total)]  # intercept
    col_names = ["intercept"]

    # Linear terms
    for fi, factor in enumerate(factors):
        columns.append(coded[:, fi])
        col_names.append(factor.name)

    # Quadratic terms
    if include_quadratic:
        for fi, factor in enumerate(factors):
            columns.append(coded[:, fi] ** 2)
            col_names.append(f"{factor.name}^2")

        # Interaction terms
        for fi in range(n_factors):
            for fj in range(fi + 1, n_factors):
                columns.append(coded[:, fi] * coded[:, fj])
                col_names.append(f"{factors[fi].name}*{factors[fj].name}")

    X = np.column_stack(columns)

    # Least-squares fit
    try:
        coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, responses, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(len(col_names))
        residuals_arr = np.array([])

    # Predictions and R-squared
    y_pred = X @ coeffs
    ss_res = np.sum((responses - y_pred) ** 2)
    ss_tot = np.sum((responses - np.mean(responses)) ** 2)
    r_sq = 1 - ss_res / max(ss_tot, 1e-30)

    n_coeffs = len(coeffs)
    if n_total > n_coeffs:
        adj_r_sq = 1 - (1 - r_sq) * (n_total - 1) / (n_total - n_coeffs - 1)
    else:
        adj_r_sq = r_sq

    residuals = responses - y_pred

    # Extract effects
    coeff_dict = dict(zip(col_names, coeffs))
    main_eff = {}
    quad_eff = {}
    inter_eff = {}

    for fi, factor in enumerate(factors):
        main_eff[factor.name] = float(coeffs[1 + fi])

    if include_quadratic:
        offset = 1 + n_factors
        for fi, factor in enumerate(factors):
            quad_eff[factor.name] = float(coeffs[offset + fi])

        offset += n_factors
        idx = 0
        for fi in range(n_factors):
            for fj in range(fi + 1, n_factors):
                inter_eff[(factors[fi].name, factors[fj].name)] = float(coeffs[offset + idx])
                idx += 1

    # Find optimum (by evaluating on a grid in coded space)
    best_response = None
    best_inputs = None
    grid_n = 11
    grid_1d = np.linspace(-1, 1, grid_n)

    if n_factors <= 4:
        # Grid search feasible for small factor count
        from itertools import product as cart_product
        for combo in cart_product(*([grid_1d] * n_factors)):
            coded_pt = np.array(combo)
            row = [1.0] + list(coded_pt)
            if include_quadratic:
                row += list(coded_pt ** 2)
                for fi in range(n_factors):
                    for fj in range(fi + 1, n_factors):
                        row.append(coded_pt[fi] * coded_pt[fj])
            pred = np.dot(row, coeffs)
            if best_response is None or pred < best_response:
                best_response = pred
                best_inputs = coded_pt.copy()
    else:
        # For many factors, just use center + search along main effects
        best_coded = np.zeros(n_factors)
        best_response = float(coeffs[0])
        for fi in range(n_factors):
            for val in grid_1d:
                test = best_coded.copy()
                test[fi] = val
                row = [1.0] + list(test)
                if include_quadratic:
                    row += list(test ** 2)
                    for ffi in range(n_factors):
                        for ffj in range(ffi + 1, n_factors):
                            row.append(test[ffi] * test[ffj])
                pred = np.dot(row, coeffs)
                if pred < best_response:
                    best_response = pred
                    best_coded = test.copy()
        best_inputs = best_coded

    # Convert optimum from coded to natural units
    predicted_opt = {}
    if best_inputs is not None:
        for fi, factor in enumerate(factors):
            low = min(factor.levels)
            high = max(factor.levels)
            mid = (low + high) / 2
            span = (high - low) / 2
            predicted_opt[factor.name] = float(mid + best_inputs[fi] * span)

    return RSMResult(
        coefficients=coeff_dict,
        r_squared=float(r_sq),
        adj_r_squared=float(adj_r_sq),
        predicted_optimum=predicted_opt,
        optimum_response=float(best_response) if best_response is not None else 0.0,
        main_effects=main_eff,
        quadratic_effects=quad_eff,
        interaction_effects=inter_eff,
        residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Sobol' Global Sensitivity Analysis
# ---------------------------------------------------------------------------

@dataclass
class SobolResult:
    """Result from Sobol' sensitivity analysis.

    Attributes:
        factor_names: Names of factors.
        first_order: First-order sensitivity indices (S_i).
        total_order: Total-effect sensitivity indices (S_Ti).
        n_samples: Number of base samples used.
        total_variance: Total output variance.
    """
    factor_names: list[str] = field(default_factory=list)
    first_order: dict[str, float] = field(default_factory=dict)
    total_order: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    total_variance: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== Sobol' Global Sensitivity Analysis ===",
            f"  Total variance:  {self.total_variance:.6f}",
            f"  Base samples:    {self.n_samples}",
            "",
            f"  {'Factor':<40s}  {'S_i':>8s}  {'S_Ti':>8s}",
            f"  {'-'*40}  {'-'*8}  {'-'*8}",
        ]
        # Sort by total-order index
        sorted_names = sorted(self.factor_names,
                              key=lambda n: self.total_order.get(n, 0),
                              reverse=True)
        for name in sorted_names:
            si = self.first_order.get(name, 0)
            sti = self.total_order.get(name, 0)
            lines.append(f"  {name:<40s}  {si:8.4f}  {sti:8.4f}")

        # Check sum
        si_sum = sum(self.first_order.values())
        lines.append(f"\n  Sum of S_i: {si_sum:.4f} (=1.0 if no interactions)")

        return "\n".join(lines)


def sobol_sensitivity(
    evaluate_fn: Callable[[dict[str, float]], float],
    factors: list[DOEFactor],
    n_samples: int = 1024,
    seed: int = 42,
) -> SobolResult:
    """Sobol' global sensitivity analysis using Saltelli's sampling scheme.

    Computes first-order (S_i) and total-effect (S_Ti) Sobol' indices.
    - S_i measures the fraction of output variance due to factor i alone.
    - S_Ti measures the fraction due to factor i including all interactions.

    Uses the Saltelli (2010) estimator with N*(2k+2) model evaluations
    where N = n_samples and k = number of factors.

    Args:
        evaluate_fn: Function mapping {factor_name: value} -> scalar response.
        factors: List of DOEFactor objects (levels define [low, high] range).
        n_samples: Number of base samples (recommend 512-4096).
        seed: Random seed.

    Returns:
        SobolResult with first-order and total-order indices.
    """
    k = len(factors)
    rng = np.random.default_rng(seed)

    # Generate two independent quasi-random sample matrices A and B
    # Each is N x k, values in [0, 1]
    A = rng.random((n_samples, k))
    B = rng.random((n_samples, k))

    # Scale to factor ranges
    lows = np.array([min(f.levels) for f in factors])
    highs = np.array([max(f.levels) for f in factors])
    spans = highs - lows
    spans = np.where(spans < 1e-15, 1.0, spans)

    def scale(matrix):
        return lows + matrix * spans

    def evaluate_matrix(matrix):
        scaled = scale(matrix)
        results = np.zeros(len(matrix))
        for i in range(len(matrix)):
            inputs = {factors[fi].name: scaled[i, fi] for fi in range(k)}
            results[i] = evaluate_fn(inputs)
        return results

    # Evaluate A and B
    f_A = evaluate_matrix(A)
    f_B = evaluate_matrix(B)

    # Total variance from combined A and B
    f_all = np.concatenate([f_A, f_B])
    total_variance = float(np.var(f_all))

    if total_variance < 1e-30:
        return SobolResult(
            factor_names=[f.name for f in factors],
            first_order={f.name: 0.0 for f in factors},
            total_order={f.name: 0.0 for f in factors},
            n_samples=n_samples,
            total_variance=total_variance,
        )

    # For each factor i, build matrix AB_i (take column i from B, rest from A)
    # and matrix BA_i (take column i from A, rest from B)
    first_order = {}
    total_order = {}

    for fi in range(k):
        # AB_i: A with column fi replaced by B's column fi
        AB_i = A.copy()
        AB_i[:, fi] = B[:, fi]
        f_AB_i = evaluate_matrix(AB_i)

        # First-order index: Jansen (1999) estimator
        # S_i = 1 - Var(f_B - f_AB_i) / (2 * Var_total)
        var_diff = np.mean((f_B - f_AB_i) ** 2) / 2.0
        si = 1.0 - var_diff / total_variance
        first_order[factors[fi].name] = float(np.clip(si, 0.0, 1.0))

        # Total-effect index: Jansen estimator
        # S_Ti = Var(f_A - f_AB_i) / (2 * Var_total)
        var_diff_t = np.mean((f_A - f_AB_i) ** 2) / 2.0
        sti = var_diff_t / total_variance
        total_order[factors[fi].name] = float(np.clip(sti, 0.0, 1.0))

    return SobolResult(
        factor_names=[f.name for f in factors],
        first_order=first_order,
        total_order=total_order,
        n_samples=n_samples,
        total_variance=total_variance,
    )


# ---------------------------------------------------------------------------
# Genetic Algorithm Tolerance Optimizer
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """Configuration for the genetic algorithm optimizer.

    Attributes:
        population_size: Number of individuals per generation.
        n_generations: Maximum number of generations.
        crossover_rate: Probability of crossover between parents.
        mutation_rate: Probability of mutating each gene.
        mutation_scale: Scale of mutation as fraction of tolerance range.
        elitism_count: Number of best individuals carried forward unchanged.
        tournament_size: Tournament selection group size.
    """
    population_size: int = 80
    n_generations: int = 200
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    mutation_scale: float = 0.2
    elitism_count: int = 4
    tournament_size: int = 3


@dataclass
class GAResult:
    """Result from genetic algorithm tolerance optimization.

    Attributes:
        original_tolerances: Dict of param_name -> original half_tol.
        optimized_tolerances: Dict of param_name -> optimized half_tol.
        original_variation: Original assembly variation.
        optimized_variation: Optimized assembly variation.
        original_cost: Original manufacturing cost estimate.
        optimized_cost: Optimized manufacturing cost estimate.
        generations: Number of generations run.
        converged: Whether the optimizer converged.
        fitness_history: Best fitness per generation.
    """
    original_tolerances: dict[str, float] = field(default_factory=dict)
    optimized_tolerances: dict[str, float] = field(default_factory=dict)
    original_variation: float = 0.0
    optimized_variation: float = 0.0
    original_cost: float = 0.0
    optimized_cost: float = 0.0
    generations: int = 0
    converged: bool = False
    fitness_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Genetic Algorithm Tolerance Optimization ===",
            f"  Converged:          {self.converged}",
            f"  Generations:        {self.generations}",
            f"  Original variation: {self.original_variation:.6f}",
            f"  Optimized variation:{self.optimized_variation:.6f}",
            f"  Original cost:      {self.original_cost:.2f}",
            f"  Optimized cost:     {self.optimized_cost:.2f}",
            f"  Cost reduction:     {(1 - self.optimized_cost / max(self.original_cost, 1e-10)) * 100:.1f}%",
            "",
            "  Tolerance Changes:",
        ]
        for name in sorted(self.original_tolerances.keys()):
            orig = self.original_tolerances[name]
            opt = self.optimized_tolerances.get(name, orig)
            change = (opt - orig) / max(orig, 1e-10) * 100
            marker = " **" if abs(change) > 5 else ""
            lines.append(
                f"    {name:40s}  {orig:.6f} -> {opt:.6f}  ({change:+.1f}%){marker}"
            )
        return "\n".join(lines)


def ga_optimize_tolerances(
    sensitivity: list[tuple[str, float]],
    tolerances: dict[str, float],
    target_variation: float,
    cost_fn: Optional[Callable[[str, float], float]] = None,
    min_tol_fraction: float = 0.2,
    max_tol_fraction: float = 3.0,
    config: Optional[GAConfig] = None,
    seed: int = 42,
) -> GAResult:
    """Optimize tolerance allocations using a genetic algorithm.

    Uses tournament selection, single-point crossover, and Gaussian mutation
    to evolve a population of tolerance allocations that minimize cost while
    meeting the target assembly variation constraint.

    The fitness function penalizes designs that violate the variation
    constraint and rewards lower manufacturing cost.

    Args:
        sensitivity: List of (param_name, sensitivity_value).
        tolerances: Dict of param_name -> current half_tolerance.
        target_variation: Desired RSS variation (at 3-sigma).
        cost_fn: Optional function(param_name, half_tol) -> cost.
            Defaults to 1/tol^2 cost model.
        min_tol_fraction: Minimum tolerance as fraction of original.
        max_tol_fraction: Maximum tolerance as fraction of original.
        config: GA configuration. Uses defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        GAResult with optimized tolerances and convergence history.
    """
    if config is None:
        config = GAConfig()

    if cost_fn is None:
        def cost_fn(name: str, half_tol: float) -> float:
            return 1.0 / max(half_tol, 1e-10) ** 2

    rng = np.random.default_rng(seed)

    param_names = [name for name, _ in sensitivity]
    sens_values = {name: abs(s) for name, s in sensitivity}
    n_params = len(param_names)

    if n_params == 0:
        return GAResult(converged=True)

    # Bounds for each tolerance parameter
    original_tols = {name: tolerances.get(name, 0.01) for name in param_names}
    bounds_low = np.array([original_tols[n] * min_tol_fraction for n in param_names])
    bounds_high = np.array([original_tols[n] * max_tol_fraction for n in param_names])

    def compute_rss_var(tol_array):
        total = 0.0
        for i, name in enumerate(param_names):
            s = sens_values.get(name, 0.0)
            total += (s * tol_array[i]) ** 2
        return np.sqrt(total)

    def compute_cost(tol_array):
        return sum(cost_fn(param_names[i], tol_array[i]) for i in range(n_params))

    def fitness(tol_array):
        var = compute_rss_var(tol_array)
        cost = compute_cost(tol_array)
        # Penalty for exceeding target variation
        penalty = 0.0
        if var > target_variation:
            penalty = 1e6 * ((var - target_variation) / target_variation) ** 2
        # Minimize cost + penalty (lower is better)
        return -(cost + penalty)

    # Original metrics
    orig_array = np.array([original_tols[n] for n in param_names])
    original_variation = compute_rss_var(orig_array)
    original_cost = compute_cost(orig_array)

    # Initialize population
    pop_size = config.population_size
    population = np.zeros((pop_size, n_params))
    for i in range(pop_size):
        population[i] = rng.uniform(bounds_low, bounds_high)
    # Seed the population with the original tolerances
    population[0] = orig_array.copy()

    # Evaluate initial fitness
    fit = np.array([fitness(population[i]) for i in range(pop_size)])

    fitness_history = []
    best_ever = population[np.argmax(fit)].copy()
    best_ever_fit = np.max(fit)

    converged = False

    for gen in range(config.n_generations):
        fitness_history.append(float(-best_ever_fit))  # Store cost (positive)

        # Check convergence: variation met and cost stable
        best_var = compute_rss_var(best_ever)
        if best_var <= target_variation * 1.01 and gen > 10:
            # Check if fitness has plateaued
            if len(fitness_history) >= 20:
                recent = fitness_history[-20:]
                if max(recent) - min(recent) < abs(recent[-1]) * 0.001:
                    converged = True
                    break

        # Selection + Crossover + Mutation
        new_pop = np.zeros_like(population)

        # Elitism: carry forward best individuals
        elite_idx = np.argsort(fit)[-config.elitism_count:]
        for ei, idx in enumerate(elite_idx):
            new_pop[ei] = population[idx].copy()

        # Fill rest with offspring
        for i in range(config.elitism_count, pop_size):
            # Tournament selection for parent 1
            t1 = rng.choice(pop_size, size=config.tournament_size, replace=False)
            p1_idx = t1[np.argmax(fit[t1])]

            # Tournament selection for parent 2
            t2 = rng.choice(pop_size, size=config.tournament_size, replace=False)
            p2_idx = t2[np.argmax(fit[t2])]

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover
            if rng.random() < config.crossover_rate:
                cx_point = rng.integers(1, n_params) if n_params > 1 else 1
                child = np.concatenate([parent1[:cx_point], parent2[cx_point:]])
            else:
                child = parent1.copy()

            # Mutation
            for j in range(n_params):
                if rng.random() < config.mutation_rate:
                    scale = (bounds_high[j] - bounds_low[j]) * config.mutation_scale
                    child[j] += rng.normal(0, scale)

            # Clip to bounds
            child = np.clip(child, bounds_low, bounds_high)
            new_pop[i] = child

        population = new_pop
        fit = np.array([fitness(population[i]) for i in range(pop_size)])

        # Track best
        gen_best_idx = np.argmax(fit)
        if fit[gen_best_idx] > best_ever_fit:
            best_ever = population[gen_best_idx].copy()
            best_ever_fit = fit[gen_best_idx]

    # Build result
    optimized_tols = {param_names[i]: float(best_ever[i]) for i in range(n_params)}

    return GAResult(
        original_tolerances=original_tols,
        optimized_tolerances=optimized_tols,
        original_variation=original_variation,
        optimized_variation=compute_rss_var(best_ever),
        original_cost=original_cost,
        optimized_cost=compute_cost(best_ever),
        generations=gen + 1 if not converged else gen,
        converged=converged,
        fitness_history=fitness_history,
    )


# ---------------------------------------------------------------------------
# Contingency Analysis (3DCS-style)
# ---------------------------------------------------------------------------

@dataclass
class ContingencyItem:
    """A single contingency analysis entry.

    Attributes:
        param_name: Name of the tolerance parameter.
        baseline_variation: Assembly variation with all tolerances at nominal.
        failure_variation: Assembly variation when this tolerance is at worst case.
        impact: Increase in variation from baseline.
        impact_percent: Impact as percentage of baseline.
        rank: Ranking by impact (1 = highest).
        failure_mode: Description of failure mode.
    """
    param_name: str
    baseline_variation: float = 0.0
    failure_variation: float = 0.0
    impact: float = 0.0
    impact_percent: float = 0.0
    rank: int = 0
    failure_mode: str = ""


@dataclass
class ContingencyResult:
    """Result from contingency analysis.

    Attributes:
        items: List of ContingencyItem sorted by impact.
        baseline_variation: Assembly variation at nominal.
        worst_case_variation: Assembly variation with all tolerances at worst case.
        method: Analysis method used.
    """
    items: list[ContingencyItem] = field(default_factory=list)
    baseline_variation: float = 0.0
    worst_case_variation: float = 0.0
    method: str = "contingency"

    def summary(self) -> str:
        lines = [
            "=== Contingency Analysis ===",
            f"  Baseline variation: {self.baseline_variation:.6f}",
            f"  Worst-case variation: {self.worst_case_variation:.6f}",
            "",
            f"  {'Rank':>4s}  {'Parameter':<45s}  {'Impact':>10s}  {'Impact%':>8s}  {'Failure Var':>12s}",
            f"  {'----':>4s}  {'-'*45:<45s}  {'-'*10:>10s}  {'-'*8:>8s}  {'-'*12:>12s}",
        ]
        for item in self.items[:20]:
            lines.append(
                f"  {item.rank:4d}  {item.param_name:<45s}  "
                f"{item.impact:10.6f}  {item.impact_percent:7.2f}%  "
                f"{item.failure_variation:12.6f}"
            )
        return "\n".join(lines)


def contingency_analysis(
    sensitivity: list[tuple[str, float]],
    tolerances: dict[str, float],
    sigma: float = 3.0,
    mode: str = "one_at_a_time",
) -> ContingencyResult:
    """Contingency (failure-mode) analysis for tolerance parameters.

    Evaluates the impact on assembly variation when each tolerance parameter
    independently goes to its worst case, while all other parameters remain
    at their nominal statistical contribution.

    This identifies which tolerances, if their manufacturing process drifts
    or fails, will cause the most severe impact on assembly quality.

    Modes:
        "one_at_a_time": Each tolerance is set to its worst case (full range)
            one at a time. This is the 3DCS-style contingency approach.
        "dropped": Each tolerance is dropped from the stack (set to zero)
            to see how much variation is reduced. Identifies which tolerances
            contribute most to the baseline.

    Args:
        sensitivity: List of (param_name, sensitivity_value).
        tolerances: Dict of param_name -> half_tolerance.
        sigma: Sigma level for RSS calculation.
        mode: Analysis mode ("one_at_a_time" or "dropped").

    Returns:
        ContingencyResult with ranked failure modes.
    """
    param_names = [name for name, _ in sensitivity]
    sens_values = {name: abs(s) for name, s in sensitivity}

    def rss_variation(tol_dict, override=None):
        """Compute RSS variation, with optional overrides."""
        total = 0.0
        for name in param_names:
            s = sens_values.get(name, 0.0)
            t = tol_dict.get(name, 0.0)
            if override and name in override:
                t = override[name]
            std_i = t / sigma
            total += (s * std_i) ** 2
        return sigma * np.sqrt(total)

    baseline = rss_variation(tolerances)

    # Worst case: all at max simultaneously
    worst_case = sum(abs(sens_values.get(n, 0.0)) * tolerances.get(n, 0.0)
                     for n in param_names)

    items = []
    for name in param_names:
        s = sens_values.get(name, 0.0)
        t = tolerances.get(name, 0.0)
        if s < 1e-15 or t < 1e-15:
            continue

        if mode == "one_at_a_time":
            # Set this parameter to worst case (full tolerance range instead of
            # sigma-based), while others remain at statistical contribution
            override = {name: t * sigma}  # worst case = full range
            failure_var = rss_variation(tolerances, override)
            failure_mode = f"{name} at worst case ({t * sigma:.6f})"
        elif mode == "dropped":
            # Remove this parameter to see how much variation drops
            override = {name: 0.0}
            failure_var = rss_variation(tolerances, override)
            failure_mode = f"{name} removed from stack"
        else:
            raise ValueError(f"Unknown contingency mode: {mode!r}")

        impact = abs(failure_var - baseline)
        impact_pct = (impact / max(baseline, 1e-15)) * 100.0

        items.append(ContingencyItem(
            param_name=name,
            baseline_variation=baseline,
            failure_variation=failure_var,
            impact=impact,
            impact_percent=impact_pct,
            failure_mode=failure_mode,
        ))

    # Sort by impact descending and assign ranks
    items.sort(key=lambda x: x.impact, reverse=True)
    for i, item in enumerate(items):
        item.rank = i + 1

    return ContingencyResult(
        items=items,
        baseline_variation=baseline,
        worst_case_variation=worst_case,
        method=mode,
    )
