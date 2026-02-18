"""Tests for tolerance optimizer, DOE, and critical tolerance identifier."""

import math

import numpy as np
import pytest

from tolerance_stack.optimizer import (
    critical_tolerance_identifier,
    optimize_tolerances,
    hlm_sensitivity,
    full_factorial_doe,
    DOEFactor,
)


class TestCriticalToleranceIdentifier:
    def test_basic_ranking(self):
        sens_maps = {
            "meas1": [("A", 1.0), ("B", 3.0)],
            "meas2": [("A", 0.5), ("B", 2.0)],
        }
        tol_maps = {
            "meas1": [0.1, 0.1],
            "meas2": [0.1, 0.1],
        }
        result = critical_tolerance_identifier(sens_maps, tol_maps)
        assert len(result.rankings) == 2
        # B has higher sensitivity in both measurements
        assert result.rankings[0][0] == "B"

    def test_contributions_dict(self):
        sens_maps = {"m1": [("A", 1.0), ("B", 1.0)]}
        tol_maps = {"m1": [0.1, 0.1]}
        result = critical_tolerance_identifier(sens_maps, tol_maps)
        assert "A" in result.contributions
        assert "m1" in result.contributions["A"]

    def test_summary(self):
        sens_maps = {"m1": [("A", 1.0)]}
        tol_maps = {"m1": [0.1]}
        result = critical_tolerance_identifier(sens_maps, tol_maps)
        s = result.summary()
        assert "Critical Tolerance" in s


class TestHLMSensitivity:
    def test_linear_response(self):
        """Linear function: effect should match level range."""
        def evaluate(inputs):
            return inputs["x"] * 2 + inputs["y"]

        factors = [
            DOEFactor("x", levels=[-1.0, 0.0, 1.0], nominal=0.0),
            DOEFactor("y", levels=[-0.5, 0.0, 0.5], nominal=0.0),
        ]
        result = hlm_sensitivity(evaluate, factors)
        # x ranges from -1 to +1, response changes by 4
        assert result.main_effects["x"] == pytest.approx(4.0)
        # y ranges from -0.5 to +0.5, response changes by 1
        assert result.main_effects["y"] == pytest.approx(1.0)

    def test_summary(self):
        def evaluate(inputs):
            return inputs["x"]

        factors = [DOEFactor("x", levels=[0.0, 1.0], nominal=0.5)]
        result = hlm_sensitivity(evaluate, factors)
        s = result.summary()
        assert "Main Effects" in s


class TestFullFactorialDOE:
    def test_two_factor_two_level(self):
        def evaluate(inputs):
            return inputs["a"] + inputs["b"]

        factors = [
            DOEFactor("a", levels=[-1.0, 1.0], nominal=0.0),
            DOEFactor("b", levels=[-1.0, 1.0], nominal=0.0),
        ]
        result = full_factorial_doe(evaluate, factors)
        assert len(result.runs) == 4  # 2 x 2
        assert result.main_effects["a"] == pytest.approx(2.0)
        assert result.main_effects["b"] == pytest.approx(2.0)

    def test_no_interaction_for_linear(self):
        """Pure additive function should have zero interaction."""
        def evaluate(inputs):
            return inputs["a"] + inputs["b"]

        factors = [
            DOEFactor("a", levels=[-1.0, 1.0], nominal=0.0),
            DOEFactor("b", levels=[-1.0, 1.0], nominal=0.0),
        ]
        result = full_factorial_doe(evaluate, factors)
        interaction = result.interactions.get(("a", "b"), 0)
        assert abs(interaction) < 0.01

    def test_interaction_detected(self):
        """Multiplicative function should produce interaction."""
        def evaluate(inputs):
            return inputs["a"] * inputs["b"]

        factors = [
            DOEFactor("a", levels=[-1.0, 1.0], nominal=0.0),
            DOEFactor("b", levels=[-1.0, 1.0], nominal=0.0),
        ]
        result = full_factorial_doe(evaluate, factors)
        interaction = result.interactions.get(("a", "b"), 0)
        assert abs(interaction) > 0.5

    def test_three_factors(self):
        def evaluate(inputs):
            return inputs["a"] + inputs["b"] * 2 + inputs["c"] * 3

        factors = [
            DOEFactor("a", levels=[0, 1], nominal=0.5),
            DOEFactor("b", levels=[0, 1], nominal=0.5),
            DOEFactor("c", levels=[0, 1], nominal=0.5),
        ]
        result = full_factorial_doe(evaluate, factors)
        assert len(result.runs) == 8  # 2^3
        assert result.main_effects["c"] > result.main_effects["b"] > result.main_effects["a"]


class TestOptimizeTolerances:
    def test_meets_target(self):
        sens = [("A", 1.0), ("B", 1.0)]
        tols = {"A": 0.1, "B": 0.1}
        # Current RSS variation = sqrt(0.1^2 + 0.1^2) = 0.1414
        # Target slightly below current
        result = optimize_tolerances(sens, tols, target_variation=0.10)
        assert result.optimized_variation < result.original_variation + 0.01

    def test_preserves_low_sensitivity(self):
        sens = [("A", 1.0), ("B", 0.01)]
        tols = {"A": 0.1, "B": 0.1}
        result = optimize_tolerances(sens, tols, target_variation=0.05)
        # B has low sensitivity, should not be tightened much
        assert result.optimized_tolerances["B"] >= result.optimized_tolerances["A"]

    def test_summary(self):
        sens = [("A", 1.0)]
        tols = {"A": 0.1}
        result = optimize_tolerances(sens, tols, target_variation=0.08)
        s = result.summary()
        assert "Tolerance Optimization" in s
