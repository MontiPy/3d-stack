"""Tests for the 3D rigid body assembly model and analysis."""

import math

import numpy as np
import pytest

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType, BodyPlacement,
    _make_transform, _rotation_matrix,
)
from tolerance_stack.assembly_analysis import (
    assembly_worst_case,
    assembly_rss,
    assembly_monte_carlo,
    analyze_assembly,
    _compute_jacobian,
)
from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_two_body_assembly() -> Assembly:
    """Two blocks stacked along Z with face tolerances.

    Block A: 20mm tall, top face at Z=20 with ±0.05 position tol.
    Block B: 10mm tall, bottom face at Z=0 with ±0.03 position tol.
    Measurement: distance along Z from A.bottom to B.top.
    """
    block_a = Body("BlockA")
    block_a.add_feature(Feature(
        "bottom", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
    ))
    block_a.add_feature(Feature(
        "top", FeatureType.PLANE,
        origin=(0, 0, 20), direction=(0, 0, 1),
        position_tol=0.10,
    ))

    block_b = Body("BlockB")
    block_b.add_feature(Feature(
        "bottom", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
        position_tol=0.06,
    ))
    block_b.add_feature(Feature(
        "top", FeatureType.PLANE,
        origin=(0, 0, 10), direction=(0, 0, 1),
        position_tol=0.06,
    ))

    assy = Assembly(name="TwoBlock")
    assy.add_body(block_a, origin=(0, 0, 0))
    assy.add_body(block_b, origin=(0, 0, 20))

    assy.add_mate(Mate(
        "stack", "BlockA", "top", "BlockB", "bottom",
        mate_type=MateType.COPLANAR,
    ))

    assy.set_measurement(Measurement(
        "height", "BlockA", "bottom", "BlockB", "top",
        measurement_type=MeasurementType.DISTANCE_ALONG,
        direction=(0, 0, 1),
    ))

    return assy


def _point_to_point_assembly() -> Assembly:
    """Two bodies with point features, measuring 3D distance."""
    body_a = Body("A")
    body_a.add_feature(Feature("p1", FeatureType.POINT, origin=(0, 0, 0)))

    body_b = Body("B")
    body_b.add_feature(Feature(
        "p2", FeatureType.POINT, origin=(0, 0, 0),
        position_tol=0.10,
    ))

    assy = Assembly(name="P2P")
    assy.add_body(body_a, origin=(0, 0, 0))
    assy.add_body(body_b, origin=(100, 0, 0))

    assy.set_measurement(Measurement(
        "dist", "A", "p1", "B", "p2",
        measurement_type=MeasurementType.DISTANCE,
    ))
    return assy


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_identity(self):
        T = _make_transform()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_translation(self):
        T = _make_transform(origin=(10, 20, 30))
        np.testing.assert_allclose(T[:3, 3], [10, 20, 30], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_rotation_z_90(self):
        R = _rotation_matrix(0, 0, 90)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

    def test_rotation_y_90(self):
        R = _rotation_matrix(0, 90, 0)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 0, -1], atol=1e-12)


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

class TestFeature:
    def test_direction_normalization(self):
        f = Feature("f", FeatureType.AXIS, direction=(3, 4, 0))
        assert math.isclose(sum(d**2 for d in f.direction), 1.0)

    def test_world_origin(self):
        f = Feature("f", FeatureType.POINT, origin=(10, 0, 0))
        T = _make_transform(origin=(5, 5, 5))
        pos = f.world_origin(T)
        np.testing.assert_allclose(pos, [15, 5, 5], atol=1e-12)

    def test_world_origin_rotated(self):
        f = Feature("f", FeatureType.POINT, origin=(10, 0, 0))
        T = _make_transform(rotation=(0, 0, 90))  # rotate 90 about Z
        pos = f.world_origin(T)
        np.testing.assert_allclose(pos, [0, 10, 0], atol=1e-12)

    def test_world_direction(self):
        f = Feature("f", FeatureType.AXIS, direction=(1, 0, 0))
        T = _make_transform(rotation=(0, 0, 90))
        d = f.world_direction(T)
        np.testing.assert_allclose(d, [0, 1, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Body tests
# ---------------------------------------------------------------------------

class TestBody:
    def test_add_duplicate_feature(self):
        body = Body("B")
        body.add_feature(Feature("f1", FeatureType.POINT))
        with pytest.raises(ValueError, match="already exists"):
            body.add_feature(Feature("f1", FeatureType.POINT))

    def test_get_missing_feature(self):
        body = Body("B")
        with pytest.raises(KeyError, match="not found"):
            body.get_feature("nonexistent")


# ---------------------------------------------------------------------------
# Assembly construction tests
# ---------------------------------------------------------------------------

class TestAssemblyConstruction:
    def test_duplicate_body(self):
        assy = Assembly(name="Test")
        body = Body("B1")
        assy.add_body(body)
        with pytest.raises(ValueError, match="already in assembly"):
            assy.add_body(body)

    def test_mate_invalid_body(self):
        assy = Assembly(name="Test")
        assy.add_body(Body("A"))
        with pytest.raises(ValueError, match="not in assembly"):
            assy.add_mate(Mate("m", "A", "f", "Missing", "f"))

    def test_measurement_invalid_body(self):
        assy = Assembly(name="Test")
        assy.add_body(Body("A"))
        with pytest.raises(ValueError, match="not in assembly"):
            assy.set_measurement(Measurement("m", "A", "f", "Missing", "f"))


# ---------------------------------------------------------------------------
# Measurement tests
# ---------------------------------------------------------------------------

class TestMeasurement:
    def test_distance_along_z(self):
        assy = _simple_two_body_assembly()
        val = assy.compute_measurement()
        # BlockA.bottom at Z=0, BlockB.top at Z=20+10=30
        assert val == pytest.approx(30.0)

    def test_point_distance(self):
        assy = _point_to_point_assembly()
        val = assy.compute_measurement()
        assert val == pytest.approx(100.0)

    def test_distance_along_x(self):
        body_a = Body("A")
        body_a.add_feature(Feature("p", FeatureType.POINT, origin=(0, 0, 0)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.POINT, origin=(0, 0, 0)))

        assy = Assembly(name="Test")
        assy.add_body(body_a, origin=(0, 0, 0))
        assy.add_body(body_b, origin=(50, 30, 0))

        assy.set_measurement(Measurement(
            "dx", "A", "p", "B", "p",
            measurement_type=MeasurementType.DISTANCE_ALONG,
            direction=(1, 0, 0),
        ))
        assert assy.compute_measurement() == pytest.approx(50.0)

    def test_angle_measurement(self):
        body_a = Body("A")
        body_a.add_feature(Feature(
            "axis", FeatureType.AXIS,
            origin=(0, 0, 0), direction=(1, 0, 0),
        ))
        body_b = Body("B")
        body_b.add_feature(Feature(
            "axis", FeatureType.AXIS,
            origin=(0, 0, 0), direction=(0, 1, 0),
        ))

        assy = Assembly(name="Angle")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.set_measurement(Measurement(
            "angle", "A", "axis", "B", "axis",
            measurement_type=MeasurementType.ANGLE,
        ))
        assert assy.compute_measurement() == pytest.approx(90.0)

    def test_feature_offset_changes_measurement(self):
        assy = _simple_two_body_assembly()
        nominal = assy.compute_measurement()
        shifted = assy.compute_measurement(
            feature_offsets={("BlockB", "top"): np.array([0, 0, 0.5])}
        )
        assert shifted == pytest.approx(nominal + 0.5)

    def test_mate_distance_offset(self):
        assy = _simple_two_body_assembly()
        nominal = assy.compute_measurement()
        shifted = assy.compute_measurement(
            mate_distance_offsets={"stack": 0.1}
        )
        # Mate shifts BlockB along Z by 0.1
        assert shifted == pytest.approx(nominal + 0.1)


# ---------------------------------------------------------------------------
# Jacobian tests
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_parameter_count(self):
        assy = _simple_two_body_assembly()
        J, names, params = _compute_jacobian(assy)
        # BlockA.top has position_tol -> 3 axes (x, y, z)
        # BlockB.bottom has position_tol -> 3 axes
        # BlockB.top has position_tol -> 3 axes
        # Total = 9
        assert len(names) == 9

    def test_z_sensitivity_dominant(self):
        """For a Z-direction measurement, Z tolerances on measurement features dominate.

        Measurement is from BlockA.bottom (no tol) to BlockB.top (has tol).
        BlockB.top.pos_z should have significant sensitivity since it's the
        measurement target. BlockA.top has tolerance but is not part of the
        measurement, only part of the mate, so it may have zero sensitivity.
        """
        assy = _simple_two_body_assembly()
        J, names, params = _compute_jacobian(assy)

        # X/Y components should have near-zero sensitivity for a Z measurement
        for i, name in enumerate(names):
            if name.endswith("pos_x") or name.endswith("pos_y"):
                assert abs(J[i]) < 0.01, f"{name} should have ~0 sensitivity"

        # BlockB.top is a measurement target — its Z tolerance should matter
        for i, name in enumerate(names):
            if name == "BlockB.top.pos_z":
                assert abs(J[i]) > 0.1, f"{name} should have significant sensitivity"


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestWorstCase:
    def test_nominal(self):
        assy = _simple_two_body_assembly()
        r = assembly_worst_case(assy)
        assert r.nominal_value == pytest.approx(30.0)

    def test_positive_tolerance(self):
        assy = _simple_two_body_assembly()
        r = assembly_worst_case(assy)
        assert r.plus_tolerance > 0
        assert r.minus_tolerance > 0

    def test_range_brackets_nominal(self):
        assy = _simple_two_body_assembly()
        r = assembly_worst_case(assy)
        assert r.value_min <= r.nominal_value
        assert r.value_max >= r.nominal_value

    def test_no_tolerance_zero_range(self):
        """Assembly with no tolerances should have zero tolerance."""
        body_a = Body("A")
        body_a.add_feature(Feature("p", FeatureType.POINT, origin=(0, 0, 0)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.POINT, origin=(0, 0, 0)))

        assy = Assembly(name="NoTol")
        assy.add_body(body_a)
        assy.add_body(body_b, origin=(100, 0, 0))
        assy.set_measurement(Measurement(
            "dist", "A", "p", "B", "p",
            measurement_type=MeasurementType.DISTANCE,
        ))
        r = assembly_worst_case(assy)
        assert r.plus_tolerance == pytest.approx(0.0, abs=1e-10)


class TestRSS:
    def test_rss_smaller_than_wc(self):
        assy = _simple_two_body_assembly()
        wc = assembly_worst_case(assy)
        rs = assembly_rss(assy, sigma=3.0)
        assert rs.plus_tolerance <= wc.plus_tolerance + 1e-10

    def test_sigma_scaling(self):
        assy = _simple_two_body_assembly()
        r3 = assembly_rss(assy, sigma=3.0)
        r6 = assembly_rss(assy, sigma=6.0)
        assert r6.plus_tolerance == pytest.approx(r3.plus_tolerance * 2.0, abs=1e-10)


class TestMonteCarlo:
    def test_mean_near_nominal(self):
        assy = _simple_two_body_assembly()
        r = assembly_monte_carlo(assy, n_samples=50_000, seed=42)
        assert r.mc_mean == pytest.approx(30.0, abs=0.05)

    def test_seed_reproducibility(self):
        assy = _simple_two_body_assembly()
        r1 = assembly_monte_carlo(assy, n_samples=1_000, seed=99)
        r2 = assembly_monte_carlo(assy, n_samples=1_000, seed=99)
        np.testing.assert_array_equal(r1.mc_samples, r2.mc_samples)


class TestAnalyzeAssembly:
    def test_all_methods(self):
        assy = _simple_two_body_assembly()
        results = analyze_assembly(assy, mc_seed=42)
        assert "wc" in results
        assert "rss" in results
        assert "mc" in results


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        assy = _simple_two_body_assembly()
        path = str(tmp_path / "test_assy.json")
        assy.save(path)
        loaded = Assembly.load(path)

        assert loaded.name == assy.name
        assert len(loaded.bodies) == len(assy.bodies)
        assert len(loaded.mates) == len(assy.mates)
        assert loaded.measurement is not None
        assert loaded.measurement.name == assy.measurement.name

    def test_loaded_measurement_matches(self, tmp_path):
        assy = _simple_two_body_assembly()
        path = str(tmp_path / "test_assy.json")
        assy.save(path)
        loaded = Assembly.load(path)

        val_orig = assy.compute_measurement()
        val_loaded = loaded.compute_measurement()
        assert val_orig == pytest.approx(val_loaded)


class TestToleranceParameters:
    def test_count(self):
        assy = _simple_two_body_assembly()
        params = assy.tolerance_parameters()
        # BlockA.top position_tol -> 3 axes
        # BlockB.bottom position_tol -> 3 axes
        # BlockB.top position_tol -> 3 axes
        assert len(params) == 9

    def test_mate_tolerance_counted(self):
        assy = _simple_two_body_assembly()
        # The existing mate has distance_tol=0, so add one with tolerance
        assy.mates[0].distance_tol = 0.05
        params = assy.tolerance_parameters()
        mate_params = [p for p in params if p["source"] == "mate_distance"]
        assert len(mate_params) == 1
