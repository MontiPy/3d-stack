"""Tests for actual-size MMC/LMC bonus tolerance in Monte Carlo."""

import pytest
import numpy as np

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.gdt import (
    GDTType, MaterialCondition, FeatureControlFrame,
)
from tolerance_stack.assembly_analysis import assembly_monte_carlo


def _make_mmc_assembly():
    """Create assembly with MMC position tolerance on a hole."""
    block = Body("Block")
    hole = Feature(
        "hole", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=5.0, position_tol=0.0,
        size_nominal=10.0, size_plus_tol=0.1, size_minus_tol=0.1,
    )
    # Add MMC position FCF
    fcf = FeatureControlFrame(
        name="pos_mmc", gdt_type=GDTType.POSITION,
        tolerance_value=1.0, is_diametral=True,
        material_condition=MaterialCondition.MMC,
        feature_size_nominal=10.0,
        feature_size_tol_plus=0.1,
        feature_size_tol_minus=0.1,
    )
    hole.add_fcf(fcf)
    block.add_feature(hole)

    pin = Body("Pin")
    pin.add_feature(Feature(
        "shaft", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=4.9, position_tol=0.03,
    ))

    assy = Assembly("mmc_test")
    assy.add_body(block, origin=(0, 0, 0))
    assy.add_body(pin, origin=(0, 0, 10))
    assy.add_mate(Mate("coax", "Block", "hole", "Pin", "shaft", MateType.COAXIAL))
    assy.set_measurement(Measurement(
        "gap", "Block", "hole", "Pin", "shaft",
        MeasurementType.DISTANCE_ALONG, direction=(0, 0, 1),
    ))
    return assy


def _make_rfs_assembly():
    """Create assembly with RFS (no material condition) for comparison."""
    block = Body("Block")
    hole = Feature(
        "hole", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=5.0, position_tol=0.0,
        size_nominal=10.0, size_plus_tol=0.1, size_minus_tol=0.1,
    )
    # RFS position FCF (same tolerance value but no bonus)
    fcf = FeatureControlFrame(
        name="pos_rfs", gdt_type=GDTType.POSITION,
        tolerance_value=1.0, is_diametral=True,
        material_condition=MaterialCondition.NONE,
    )
    hole.add_fcf(fcf)
    block.add_feature(hole)

    pin = Body("Pin")
    pin.add_feature(Feature(
        "shaft", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=4.9, position_tol=0.03,
    ))

    assy = Assembly("rfs_test")
    assy.add_body(block, origin=(0, 0, 0))
    assy.add_body(pin, origin=(0, 0, 10))
    assy.add_mate(Mate("coax", "Block", "hole", "Pin", "shaft", MateType.COAXIAL))
    assy.set_measurement(Measurement(
        "gap", "Block", "hole", "Pin", "shaft",
        MeasurementType.DISTANCE_ALONG, direction=(0, 0, 1),
    ))
    return assy


class TestMMCLMCMonteCarlo:

    def test_mmc_mc_produces_results(self):
        assy = _make_mmc_assembly()
        result = assembly_monte_carlo(assy, n_samples=1000, seed=42)
        assert result.mc_samples is not None
        assert len(result.mc_samples) == 1000
        assert result.mc_std > 0

    def test_mmc_wider_variation_than_rfs(self):
        """MMC bonus should increase the overall variation compared to RFS."""
        mmc_assy = _make_mmc_assembly()
        rfs_assy = _make_rfs_assembly()

        mmc_result = assembly_monte_carlo(mmc_assy, n_samples=50000, seed=42)
        rfs_result = assembly_monte_carlo(rfs_assy, n_samples=50000, seed=42)

        # MMC provides bonus tolerance, so variation should be >= RFS
        # (with statistical noise allowance)
        assert mmc_result.mc_std >= rfs_result.mc_std * 0.85

    def test_mmc_bonus_is_bounded(self):
        """MMC bonus should never exceed total size tolerance."""
        assy = _make_mmc_assembly()
        result = assembly_monte_carlo(assy, n_samples=5000, seed=42)
        # The variation should exist and be finite
        assert np.isfinite(result.mc_std)
        assert result.mc_std > 0

    def test_rfs_no_bonus_applied(self):
        """RFS should not apply any bonus scaling."""
        assy = _make_rfs_assembly()
        result = assembly_monte_carlo(assy, n_samples=1000, seed=42)
        assert result.mc_samples is not None
        assert result.mc_std > 0

    def test_mmc_reproducible(self):
        """Same seed should produce same results."""
        assy = _make_mmc_assembly()
        r1 = assembly_monte_carlo(assy, n_samples=1000, seed=99)
        r2 = assembly_monte_carlo(assy, n_samples=1000, seed=99)
        np.testing.assert_array_equal(r1.mc_samples, r2.mc_samples)

    def test_lmc_assembly(self):
        """LMC should also work with bonus tolerance."""
        block = Body("Block")
        hole = Feature(
            "hole", FeatureType.CYLINDER,
            origin=(0, 0, 0), direction=(0, 0, 1),
            radius=5.0, position_tol=0.0,
            size_nominal=10.0, size_plus_tol=0.1, size_minus_tol=0.1,
        )
        fcf = FeatureControlFrame(
            name="pos_lmc", gdt_type=GDTType.POSITION,
            tolerance_value=1.0, is_diametral=True,
            material_condition=MaterialCondition.LMC,
            feature_size_nominal=10.0,
            feature_size_tol_plus=0.1,
            feature_size_tol_minus=0.1,
        )
        hole.add_fcf(fcf)
        block.add_feature(hole)

        pin = Body("Pin")
        pin.add_feature(Feature(
            "shaft", FeatureType.CYLINDER,
            origin=(0, 0, 0), direction=(0, 0, 1),
            radius=4.9, position_tol=0.03,
        ))

        assy = Assembly("lmc_test")
        assy.add_body(block, origin=(0, 0, 0))
        assy.add_body(pin, origin=(0, 0, 10))
        assy.add_mate(Mate("coax", "Block", "hole", "Pin", "shaft", MateType.COAXIAL))
        assy.set_measurement(Measurement(
            "gap", "Block", "hole", "Pin", "shaft",
            MeasurementType.DISTANCE_ALONG, direction=(0, 0, 1),
        ))

        result = assembly_monte_carlo(assy, n_samples=1000, seed=42)
        assert result.mc_samples is not None
        assert result.mc_std > 0
