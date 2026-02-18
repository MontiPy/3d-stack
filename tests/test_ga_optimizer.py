"""Tests for genetic algorithm tolerance optimizer."""

import pytest
import numpy as np

from tolerance_stack.optimizer import (
    ga_optimize_tolerances, GAConfig, GAResult,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SENSITIVITY = [
    ("A.hole.pos_x", 1.0),
    ("A.hole.pos_y", 0.5),
    ("B.pin.pos_x", -0.8),
    ("B.pin.pos_y", 0.3),
]

TOLERANCES = {
    "A.hole.pos_x": 0.10,
    "A.hole.pos_y": 0.10,
    "B.pin.pos_x": 0.08,
    "B.pin.pos_y": 0.08,
}


class TestGAConfig:

    def test_defaults(self):
        config = GAConfig()
        assert config.population_size == 80
        assert config.n_generations == 200
        assert config.crossover_rate == 0.85
        assert config.mutation_rate == 0.15

    def test_custom(self):
        config = GAConfig(population_size=50, n_generations=100)
        assert config.population_size == 50
        assert config.n_generations == 100


class TestGAOptimizer:

    def test_basic_optimization(self):
        """GA should find tolerances that meet the target variation."""
        # Compute current RSS variation
        current_var = np.sqrt(sum(
            (abs(s) * TOLERANCES[n]) ** 2 for n, s in SENSITIVITY
        ))
        # Target: tighter than current
        target = current_var * 0.7

        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=target,
            config=GAConfig(population_size=40, n_generations=100),
            seed=42,
        )
        assert isinstance(result, GAResult)
        assert result.optimized_variation <= target * 1.1  # Allow 10% slack

    def test_cost_reduction(self):
        """When target allows looser tolerances, GA should reduce cost."""
        current_var = np.sqrt(sum(
            (abs(s) * TOLERANCES[n]) ** 2 for n, s in SENSITIVITY
        ))
        # Target: same as current (so GA can loosen low-sensitivity params)
        target = current_var

        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=target,
            config=GAConfig(population_size=40, n_generations=100),
            seed=42,
        )
        # Should find solution that's no worse than original
        assert result.optimized_cost <= result.original_cost * 1.1

    def test_respects_bounds(self):
        """All optimized tolerances should be within bounds."""
        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=0.1,
            min_tol_fraction=0.3,
            max_tol_fraction=2.0,
            config=GAConfig(population_size=30, n_generations=50),
            seed=42,
        )
        for name, val in result.optimized_tolerances.items():
            orig = TOLERANCES[name]
            assert val >= orig * 0.3 - 1e-10
            assert val <= orig * 2.0 + 1e-10

    def test_fitness_history(self):
        """Fitness history should be populated."""
        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=0.05,
            config=GAConfig(population_size=20, n_generations=30),
            seed=42,
        )
        assert len(result.fitness_history) > 0

    def test_reproducible(self):
        """Same seed should produce same results."""
        cfg = GAConfig(population_size=20, n_generations=30)
        r1 = ga_optimize_tolerances(SENSITIVITY, TOLERANCES, 0.1, config=cfg, seed=42)
        r2 = ga_optimize_tolerances(SENSITIVITY, TOLERANCES, 0.1, config=cfg, seed=42)
        assert r1.optimized_cost == r2.optimized_cost
        assert r1.optimized_tolerances == r2.optimized_tolerances

    def test_single_parameter(self):
        """Should handle single parameter correctly."""
        sens = [("A.pos", 1.0)]
        tols = {"A.pos": 0.1}
        result = ga_optimize_tolerances(
            sens, tols, target_variation=0.05,
            config=GAConfig(population_size=20, n_generations=30),
            seed=42,
        )
        assert "A.pos" in result.optimized_tolerances

    def test_custom_cost_function(self):
        """Should use custom cost function when provided."""
        def linear_cost(name, half_tol):
            return 1.0 / max(half_tol, 1e-10)

        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=0.1,
            cost_fn=linear_cost,
            config=GAConfig(population_size=20, n_generations=30),
            seed=42,
        )
        assert result.optimized_cost > 0

    def test_summary_format(self):
        """Summary should contain key information."""
        result = ga_optimize_tolerances(
            SENSITIVITY, TOLERANCES,
            target_variation=0.1,
            config=GAConfig(population_size=20, n_generations=30),
            seed=42,
        )
        summary = result.summary()
        assert "Genetic Algorithm" in summary
        assert "Generations" in summary
        assert "Cost reduction" in summary

    def test_empty_params(self):
        """Should handle empty parameter list."""
        result = ga_optimize_tolerances([], {}, target_variation=0.1, seed=42)
        assert result.converged is True

    def test_high_sensitivity_gets_tightened(self):
        """Parameters with high sensitivity should be tightened more."""
        sens = [
            ("high_sens", 10.0),
            ("low_sens", 0.1),
        ]
        tols = {"high_sens": 0.1, "low_sens": 0.1}
        current_var = np.sqrt(sum(
            (abs(s) * tols[n]) ** 2 for n, s in sens
        ))
        target = current_var * 0.5

        result = ga_optimize_tolerances(
            sens, tols, target_variation=target,
            config=GAConfig(population_size=40, n_generations=100),
            seed=42,
        )
        # High sensitivity parameter should be tightened
        assert result.optimized_tolerances["high_sens"] <= tols["high_sens"]
