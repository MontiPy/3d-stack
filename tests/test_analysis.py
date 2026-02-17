"""Tests for the tolerance stack analysis engine."""

import math

import numpy as np
import pytest

from tolerance_stack.models import Contributor, ContributorType, Distribution, ToleranceStack
from tolerance_stack.analysis import worst_case, rss_no_scipy, monte_carlo, analyze_stack


def _simple_stack() -> ToleranceStack:
    """Two-part stack: housing minus shaft = gap."""
    stack = ToleranceStack(name="Simple", closure_direction=(1, 0, 0))
    stack.add(Contributor(
        name="Housing",
        nominal=10.0,
        plus_tol=0.1,
        minus_tol=0.1,
        direction=(1, 0, 0),
        sign=+1,
    ))
    stack.add(Contributor(
        name="Shaft",
        nominal=9.0,
        plus_tol=0.05,
        minus_tol=0.05,
        direction=(1, 0, 0),
        sign=-1,
    ))
    return stack


class TestWorstCase:
    def test_simple_nominal(self):
        stack = _simple_stack()
        r = worst_case(stack)
        assert r.nominal_gap == pytest.approx(1.0)

    def test_simple_range(self):
        stack = _simple_stack()
        r = worst_case(stack)
        # WC: gap_max = 1.0 + 0.1 + 0.05 = 1.15
        # WC: gap_min = 1.0 - 0.1 - 0.05 = 0.85
        assert r.gap_max == pytest.approx(1.15)
        assert r.gap_min == pytest.approx(0.85)

    def test_asymmetric_tolerance(self):
        stack = ToleranceStack(name="Asym", closure_direction=(1, 0, 0))
        stack.add(Contributor(
            name="Part",
            nominal=10.0,
            plus_tol=0.2,
            minus_tol=0.1,
            direction=(1, 0, 0),
            sign=+1,
        ))
        r = worst_case(stack)
        assert r.nominal_gap == pytest.approx(10.0)
        assert r.gap_max == pytest.approx(10.2)
        assert r.gap_min == pytest.approx(9.9)

    def test_3d_projection(self):
        """Contributor at 45 degrees should project cos(45) onto X."""
        angle = math.radians(45)
        stack = ToleranceStack(name="3D", closure_direction=(1, 0, 0))
        stack.add(Contributor(
            name="Angled part",
            nominal=10.0,
            plus_tol=0.1,
            minus_tol=0.1,
            direction=(math.cos(angle), math.sin(angle), 0),
            sign=+1,
        ))
        r = worst_case(stack)
        proj = math.cos(angle)
        assert r.nominal_gap == pytest.approx(10.0 * proj)
        assert r.plus_tolerance == pytest.approx(0.1 * proj)

    def test_zero_projection(self):
        """Contributor perpendicular to closure has zero effect."""
        stack = ToleranceStack(name="Perp", closure_direction=(1, 0, 0))
        stack.add(Contributor(
            name="Y-part",
            nominal=10.0,
            plus_tol=0.1,
            minus_tol=0.1,
            direction=(0, 1, 0),
            sign=+1,
        ))
        r = worst_case(stack)
        assert r.nominal_gap == pytest.approx(0.0)
        assert r.plus_tolerance == pytest.approx(0.0)
        assert r.minus_tolerance == pytest.approx(0.0)


class TestRSS:
    def test_simple(self):
        stack = _simple_stack()
        r = rss_no_scipy(stack, sigma=3.0)
        assert r.nominal_gap == pytest.approx(1.0)
        # RSS tol = 3 * sqrt((0.1/3)^2 + (0.05/3)^2)
        #         = sqrt(0.1^2 + 0.05^2)
        #         = sqrt(0.01 + 0.0025) = sqrt(0.0125)
        expected_tol = math.sqrt(0.1**2 + 0.05**2)
        assert r.plus_tolerance == pytest.approx(expected_tol)

    def test_sigma_scaling(self):
        stack = _simple_stack()
        r3 = rss_no_scipy(stack, sigma=3.0)
        r6 = rss_no_scipy(stack, sigma=6.0)
        # At 6-sigma the tolerance should be exactly 2x the 3-sigma tolerance
        assert r6.plus_tolerance == pytest.approx(r3.plus_tolerance * 2.0)


class TestMonteCarlo:
    def test_mean_near_nominal(self):
        stack = _simple_stack()
        r = monte_carlo(stack, n_samples=200_000, seed=42)
        assert r.mc_mean == pytest.approx(1.0, abs=0.01)

    def test_std_consistent_with_rss(self):
        stack = _simple_stack()
        r_mc = monte_carlo(stack, n_samples=500_000, seed=42)
        r_rss = rss_no_scipy(stack, sigma=3.0)
        # MC std should be close to RSS std = rss_tol / 3
        rss_std = r_rss.plus_tolerance / 3.0
        assert r_mc.mc_std == pytest.approx(rss_std, rel=0.05)

    def test_uniform_distribution(self):
        stack = ToleranceStack(name="Uniform", closure_direction=(1, 0, 0))
        stack.add(Contributor(
            name="Part",
            nominal=10.0,
            plus_tol=0.1,
            minus_tol=0.1,
            direction=(1, 0, 0),
            sign=+1,
            distribution=Distribution.UNIFORM,
        ))
        r = monte_carlo(stack, n_samples=100_000, seed=42)
        assert r.mc_mean == pytest.approx(10.0, abs=0.01)
        # Uniform std = (b-a) / sqrt(12) = 0.2 / sqrt(12)
        expected_std = 0.2 / math.sqrt(12)
        assert r.mc_std == pytest.approx(expected_std, rel=0.05)

    def test_seed_reproducibility(self):
        stack = _simple_stack()
        r1 = monte_carlo(stack, n_samples=10_000, seed=123)
        r2 = monte_carlo(stack, n_samples=10_000, seed=123)
        np.testing.assert_array_equal(r1.mc_samples, r2.mc_samples)


class TestAnalyzeStack:
    def test_all_methods(self):
        stack = _simple_stack()
        results = analyze_stack(stack, mc_seed=42)
        assert "wc" in results
        assert "rss" in results
        assert "mc" in results

    def test_single_method(self):
        stack = _simple_stack()
        results = analyze_stack(stack, methods=["wc"])
        assert "wc" in results
        assert "rss" not in results


class TestModels:
    def test_direction_normalization(self):
        c = Contributor(name="A", nominal=1, plus_tol=0.1, minus_tol=0.1,
                        direction=(3, 4, 0))
        assert math.isclose(sum(d**2 for d in c.direction), 1.0)
        assert c.direction == pytest.approx((0.6, 0.8, 0.0))

    def test_invalid_sign(self):
        with pytest.raises(ValueError, match="sign"):
            Contributor(name="A", nominal=1, plus_tol=0.1, minus_tol=0.1, sign=0)

    def test_zero_direction(self):
        with pytest.raises(ValueError, match="non-zero"):
            Contributor(name="A", nominal=1, plus_tol=0.1, minus_tol=0.1,
                        direction=(0, 0, 0))

    def test_save_load_roundtrip(self, tmp_path):
        stack = ToleranceStack(name="Test", closure_direction=(0, 1, 0))
        stack.add(Contributor(
            name="C1", nominal=5, plus_tol=0.1, minus_tol=0.2,
            direction=(0, 1, 0), sign=-1,
            distribution=Distribution.UNIFORM,
            contributor_type=ContributorType.GEOMETRIC,
        ))
        path = str(tmp_path / "test_stack.json")
        stack.save(path)
        loaded = ToleranceStack.load(path)
        assert loaded.name == "Test"
        assert len(loaded.contributors) == 1
        c = loaded.contributors[0]
        assert c.name == "C1"
        assert c.nominal == 5
        assert c.distribution == Distribution.UNIFORM
        assert c.contributor_type == ContributorType.GEOMETRIC
        assert c.sign == -1
