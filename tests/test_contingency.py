"""Tests for contingency analysis."""

import pytest
import numpy as np

from tolerance_stack.optimizer import (
    contingency_analysis, ContingencyResult, ContingencyItem,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SENSITIVITY = [
    ("A.hole.pos_x", 1.0),
    ("A.hole.pos_y", 0.5),
    ("B.pin.pos_x", -0.8),
    ("B.pin.pos_y", 0.3),
    ("C.washer.pos", 0.1),
]

TOLERANCES = {
    "A.hole.pos_x": 0.10,
    "A.hole.pos_y": 0.10,
    "B.pin.pos_x": 0.08,
    "B.pin.pos_y": 0.08,
    "C.washer.pos": 0.05,
}


class TestContingencyOneAtATime:

    def test_basic_result(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        assert isinstance(result, ContingencyResult)
        assert len(result.items) == len(SENSITIVITY)
        assert result.baseline_variation > 0

    def test_items_are_ranked(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        ranks = [item.rank for item in result.items]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_highest_impact_has_high_sensitivity(self):
        """Parameter with highest |sensitivity * tolerance| should rank high."""
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        # A.hole.pos_x has sensitivity=1.0, tol=0.10 -> contribution = 0.10
        top_name = result.items[0].param_name
        # Should be A.hole.pos_x or B.pin.pos_x (high sensitivity)
        assert top_name in ("A.hole.pos_x", "B.pin.pos_x")

    def test_impact_is_positive(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        for item in result.items:
            assert item.impact >= 0

    def test_failure_variation_exceeds_baseline(self):
        """When a parameter goes to worst case, variation should increase."""
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        for item in result.items:
            assert item.failure_variation >= result.baseline_variation - 1e-10

    def test_worst_case_variation(self):
        """Worst case variation should equal sum of |sens * tol|."""
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        expected = sum(
            abs(s) * TOLERANCES[n] for n, s in SENSITIVITY
        )
        assert abs(result.worst_case_variation - expected) < 1e-10

    def test_summary_format(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        summary = result.summary()
        assert "Contingency" in summary
        assert "Baseline" in summary
        assert "Rank" in summary
        assert "Impact" in summary

    def test_impact_percent_correct(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES)
        for item in result.items:
            expected_pct = (item.impact / max(result.baseline_variation, 1e-15)) * 100
            assert abs(item.impact_percent - expected_pct) < 0.01


class TestContingencyDropped:

    def test_dropped_mode(self):
        result = contingency_analysis(SENSITIVITY, TOLERANCES, mode="dropped")
        assert isinstance(result, ContingencyResult)
        assert result.method == "dropped"
        assert len(result.items) == len(SENSITIVITY)

    def test_dropped_variation_less_than_baseline(self):
        """Dropping a parameter should reduce variation."""
        result = contingency_analysis(SENSITIVITY, TOLERANCES, mode="dropped")
        for item in result.items:
            assert item.failure_variation <= result.baseline_variation + 1e-10

    def test_highest_impact_removal(self):
        """Removing the highest contributor should give biggest impact."""
        result = contingency_analysis(SENSITIVITY, TOLERANCES, mode="dropped")
        # The first ranked item should have the largest variance contribution
        top = result.items[0]
        assert top.impact >= result.items[-1].impact


class TestContingencyEdgeCases:

    def test_zero_sensitivity(self):
        """Parameters with zero sensitivity should still be handled."""
        sens = [("A", 0.0), ("B", 1.0)]
        tols = {"A": 0.1, "B": 0.1}
        result = contingency_analysis(sens, tols)
        # A has zero sensitivity so shouldn't be in results
        assert len(result.items) == 1
        assert result.items[0].param_name == "B"

    def test_single_parameter(self):
        sens = [("A", 1.0)]
        tols = {"A": 0.1}
        result = contingency_analysis(sens, tols)
        assert len(result.items) == 1
        assert result.items[0].rank == 1

    def test_equal_parameters(self):
        """Parameters with same contribution should all be ranked."""
        sens = [("A", 1.0), ("B", 1.0), ("C", 1.0)]
        tols = {"A": 0.1, "B": 0.1, "C": 0.1}
        result = contingency_analysis(sens, tols)
        assert len(result.items) == 3
        # All should have same impact
        impacts = [item.impact for item in result.items]
        assert abs(max(impacts) - min(impacts)) < 1e-10

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown contingency mode"):
            contingency_analysis(SENSITIVITY, TOLERANCES, mode="invalid")

    def test_different_sigma(self):
        """Different sigma levels should produce different failure impacts."""
        r3 = contingency_analysis(SENSITIVITY, TOLERANCES, sigma=3.0)
        r6 = contingency_analysis(SENSITIVITY, TOLERANCES, sigma=6.0)
        # The failure variations differ because worst case override scales with sigma
        assert r3.items[0].failure_variation != r6.items[0].failure_variation
