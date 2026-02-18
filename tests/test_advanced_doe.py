"""Tests for Latin Hypercube, Response Surface, and Sobol' sensitivity."""

import pytest
import numpy as np

from tolerance_stack.optimizer import (
    DOEFactor, latin_hypercube_doe, response_surface_doe,
    sobol_sensitivity, RSMResult, SobolResult,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def linear_fn(inputs):
    """y = 2*x1 + 3*x2 + 1"""
    return 2 * inputs["x1"] + 3 * inputs["x2"] + 1


def quadratic_fn(inputs):
    """y = x1^2 + 2*x1*x2 + x2"""
    return inputs["x1"] ** 2 + 2 * inputs["x1"] * inputs["x2"] + inputs["x2"]


FACTORS_2 = [
    DOEFactor(name="x1", levels=[-1, 1], nominal=0),
    DOEFactor(name="x2", levels=[-1, 1], nominal=0),
]

FACTORS_3 = [
    DOEFactor(name="x1", levels=[-2, 2], nominal=0),
    DOEFactor(name="x2", levels=[-1, 1], nominal=0),
    DOEFactor(name="x3", levels=[-3, 3], nominal=0),
]


# ---------------------------------------------------------------------------
# Latin Hypercube DOE
# ---------------------------------------------------------------------------

class TestLatinHypercubeDOE:

    def test_correct_sample_count(self):
        result = latin_hypercube_doe(linear_fn, FACTORS_2, n_samples=50, seed=42)
        assert len(result.runs) == 50
        assert len(result.responses) == 50

    def test_samples_in_range(self):
        result = latin_hypercube_doe(linear_fn, FACTORS_2, n_samples=100, seed=42)
        for run in result.runs:
            assert -1 <= run["x1"] <= 1
            assert -1 <= run["x2"] <= 1

    def test_main_effects_direction(self):
        result = latin_hypercube_doe(linear_fn, FACTORS_2, n_samples=500, seed=42)
        # x2 has larger coefficient (3) than x1 (2)
        assert abs(result.main_effects["x2"]) > abs(result.main_effects["x1"]) * 0.5

    def test_three_factors(self):
        def fn3(inputs):
            return 5 * inputs["x1"] + inputs["x2"] + 0.1 * inputs["x3"]
        result = latin_hypercube_doe(fn3, FACTORS_3, n_samples=200, seed=42)
        assert len(result.factor_names) == 3
        assert "x1" in result.main_effects

    def test_reproducible(self):
        r1 = latin_hypercube_doe(linear_fn, FACTORS_2, n_samples=50, seed=99)
        r2 = latin_hypercube_doe(linear_fn, FACTORS_2, n_samples=50, seed=99)
        np.testing.assert_array_equal(r1.responses, r2.responses)


# ---------------------------------------------------------------------------
# Response Surface Methodology
# ---------------------------------------------------------------------------

class TestResponseSurfaceDOE:

    def test_linear_model_high_r_squared(self):
        result = response_surface_doe(linear_fn, FACTORS_2, n_samples=50, seed=42)
        assert isinstance(result, RSMResult)
        # Linear function should fit perfectly or near-perfectly
        assert result.r_squared > 0.95

    def test_quadratic_model(self):
        result = response_surface_doe(quadratic_fn, FACTORS_2, n_samples=100, seed=42)
        assert result.r_squared > 0.8
        # Should detect quadratic effects
        assert "x1" in result.quadratic_effects
        assert "x2" in result.quadratic_effects

    def test_interaction_detection(self):
        result = response_surface_doe(quadratic_fn, FACTORS_2, n_samples=100, seed=42)
        # Should detect x1*x2 interaction
        assert ("x1", "x2") in result.interaction_effects
        assert abs(result.interaction_effects[("x1", "x2")]) > 0.1

    def test_predicted_optimum_exists(self):
        result = response_surface_doe(quadratic_fn, FACTORS_2, n_samples=100, seed=42)
        assert "x1" in result.predicted_optimum
        assert "x2" in result.predicted_optimum

    def test_summary_format(self):
        result = response_surface_doe(linear_fn, FACTORS_2, n_samples=50, seed=42)
        summary = result.summary()
        assert "Response Surface" in summary
        assert "R-squared" in summary

    def test_auto_sample_count(self):
        result = response_surface_doe(linear_fn, FACTORS_2, n_samples=0, seed=42)
        assert len(result.residuals) > 0

    def test_without_quadratic(self):
        result = response_surface_doe(linear_fn, FACTORS_2, n_samples=50, seed=42,
                                       include_quadratic=False)
        assert result.r_squared > 0.9
        assert len(result.quadratic_effects) == 0
        assert len(result.interaction_effects) == 0


# ---------------------------------------------------------------------------
# Sobol' Sensitivity
# ---------------------------------------------------------------------------

class TestSobolSensitivity:

    def test_linear_indices(self):
        result = sobol_sensitivity(linear_fn, FACTORS_2, n_samples=2048, seed=42)
        assert isinstance(result, SobolResult)
        # x2 has larger coefficient -> higher S_i
        assert result.first_order["x2"] > result.first_order["x1"]
        # Total order should be >= first order
        assert result.total_order["x2"] >= result.first_order["x2"] - 0.1
        assert result.total_order["x1"] >= result.first_order["x1"] - 0.1

    def test_sum_first_order_near_one(self):
        # For linear function, sum of S_i should be ~1 (no interactions)
        result = sobol_sensitivity(linear_fn, FACTORS_2, n_samples=4096, seed=42)
        si_sum = sum(result.first_order.values())
        assert abs(si_sum - 1.0) < 0.15  # Allow some MC noise

    def test_interaction_detection(self):
        def interaction_fn(inputs):
            return inputs["x1"] * inputs["x2"]
        result = sobol_sensitivity(interaction_fn, FACTORS_2, n_samples=2048, seed=42)
        # Pure interaction: S_i should be small, S_Ti should be large
        assert result.total_order["x1"] > result.first_order["x1"]
        assert result.total_order["x2"] > result.first_order["x2"]

    def test_three_factors(self):
        def fn3(inputs):
            return 10 * inputs["x1"] + inputs["x2"] + 0.01 * inputs["x3"]
        result = sobol_sensitivity(fn3, FACTORS_3, n_samples=1024, seed=42)
        assert result.first_order["x1"] > result.first_order["x3"]
        assert result.total_variance > 0

    def test_summary_format(self):
        result = sobol_sensitivity(linear_fn, FACTORS_2, n_samples=512, seed=42)
        summary = result.summary()
        assert "Sobol" in summary
        assert "S_i" in summary
        assert "S_Ti" in summary

    def test_constant_function(self):
        def const_fn(inputs):
            return 42.0
        result = sobol_sensitivity(const_fn, FACTORS_2, n_samples=512, seed=42)
        assert result.total_variance < 1e-10
