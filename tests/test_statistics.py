"""Tests for process capability metrics and statistical utilities."""

import math

import numpy as np
import pytest

from tolerance_stack.statistics import (
    compute_process_capability,
    percent_contribution,
    geo_factor,
    sample_distribution,
)
from tolerance_stack.models import Distribution


class TestProcessCapability:
    def test_centered_normal(self):
        """Perfect process centered at target."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=1.0, size=10000)
        pc = compute_process_capability(samples, usl=53.0, lsl=47.0)
        assert pc.cp == pytest.approx(1.0, rel=0.1)
        assert pc.cpk == pytest.approx(1.0, rel=0.1)
        assert pc.yield_percent > 99.0

    def test_shifted_process(self):
        """Off-center process: Cpk < Cp."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=51.0, scale=1.0, size=10000)
        pc = compute_process_capability(samples, usl=53.0, lsl=47.0)
        assert pc.cp > pc.cpk
        assert pc.cpk == pytest.approx(0.67, rel=0.15)

    def test_tight_process(self):
        """Very capable process: Cp > 2."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=0.3, size=10000)
        pc = compute_process_capability(samples, usl=53.0, lsl=47.0)
        assert pc.cp > 2.0
        assert pc.cpk > 2.0

    def test_ppm_out_of_spec(self):
        """Samples outside spec limits produce non-zero PPM."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=2.0, size=100000)
        pc = compute_process_capability(samples, usl=53.0, lsl=47.0)
        assert pc.ppm_total > 0
        assert pc.percent_out_of_spec > 0

    def test_target_override(self):
        """Custom target affects Cpm."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=1.0, size=10000)
        pc_default = compute_process_capability(samples, usl=53.0, lsl=47.0)
        pc_shifted = compute_process_capability(samples, usl=53.0, lsl=47.0, target=52.0)
        # Cpm should be lower when target is far from mean
        assert pc_shifted.cpm < pc_default.cpm

    def test_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_process_capability(np.array([1.0]), usl=2.0, lsl=0.0)

    def test_summary(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=1.0, size=1000)
        pc = compute_process_capability(samples, usl=53.0, lsl=47.0)
        s = pc.summary()
        assert "Cp:" in s
        assert "Cpk:" in s
        assert "Yield:" in s


class TestPercentContribution:
    def test_two_equal(self):
        sens = [("A", 1.0), ("B", 1.0)]
        tols = [0.1, 0.1]
        result = percent_contribution(sens, tols)
        assert len(result) == 2
        assert result[0][1] == pytest.approx(50.0)

    def test_dominant_contributor(self):
        sens = [("A", 1.0), ("B", 1.0)]
        tols = [0.5, 0.01]
        result = percent_contribution(sens, tols)
        # A should dominate
        a_pct = next(p for n, p in result if n == "A")
        assert a_pct > 99.0

    def test_sensitivity_weighted(self):
        sens = [("A", 2.0), ("B", 1.0)]
        tols = [0.1, 0.1]
        result = percent_contribution(sens, tols)
        a_pct = next(p for n, p in result if n == "A")
        b_pct = next(p for n, p in result if n == "B")
        assert a_pct == pytest.approx(80.0)
        assert b_pct == pytest.approx(20.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            percent_contribution([("A", 1.0)], [0.1, 0.2])


class TestGeoFactor:
    def test_amplification(self):
        sens = [("A", 2.0), ("B", 0.5)]
        result = geo_factor(sens)
        # Sorted by factor descending
        assert result[0] == ("A", 2.0)
        assert result[1] == ("B", 0.5)

    def test_negative_sens(self):
        sens = [("A", -3.0)]
        result = geo_factor(sens)
        assert result[0] == ("A", 3.0)


class TestSampleDistribution:
    def _stats(self, dist, n=100000, seed=42):
        rng = np.random.default_rng(seed)
        return sample_distribution(rng, dist, center=10.0, half_tol=1.0, sigma=3.0, n_samples=n)

    def test_normal(self):
        samples = self._stats(Distribution.NORMAL)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.05)
        assert np.std(samples) == pytest.approx(1.0 / 3.0, rel=0.05)

    def test_uniform(self):
        samples = self._stats(Distribution.UNIFORM)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.05)
        assert np.min(samples) >= 9.0 - 0.01
        assert np.max(samples) <= 11.0 + 0.01

    def test_triangular(self):
        samples = self._stats(Distribution.TRIANGULAR)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.05)

    def test_weibull_right(self):
        samples = self._stats(Distribution.WEIBULL_RIGHT)
        assert len(samples) == 100000

    def test_weibull_left(self):
        samples = self._stats(Distribution.WEIBULL_LEFT)
        assert len(samples) == 100000

    def test_lognormal(self):
        samples = self._stats(Distribution.LOGNORMAL)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.5)

    def test_rayleigh(self):
        samples = self._stats(Distribution.RAYLEIGH)
        assert len(samples) == 100000

    def test_bimodal(self):
        samples = self._stats(Distribution.BIMODAL)
        assert len(samples) == 100000
        # Should have two peaks (check that std is larger than normal)

    def test_empirical(self):
        rng = np.random.default_rng(42)
        data = np.array([9.5, 10.0, 10.5, 10.2, 9.8])
        samples = sample_distribution(
            rng, Distribution.EMPIRICAL, center=10.0, half_tol=1.0,
            sigma=3.0, n_samples=1000, empirical_data=data,
        )
        assert len(samples) == 1000
        # All samples should be from the data
        for s in samples:
            assert s in data

    def test_empirical_no_data(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="empirical_data"):
            sample_distribution(rng, Distribution.EMPIRICAL, 10.0, 1.0, 3.0, 100)
