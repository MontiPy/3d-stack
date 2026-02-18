"""Tests for advanced measurement types (Phase D)."""

import math

import numpy as np
import pytest

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)


def _make_two_body_assembly(meas_type, **meas_kwargs):
    """Helper: two bodies with point/plane/axis features and a measurement."""
    body_a = Body("A")
    body_a.add_feature(Feature(
        "point", FeatureType.POINT, origin=(0, 0, 0),
    ))
    body_a.add_feature(Feature(
        "plane", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, 1),
    ))
    body_a.add_feature(Feature(
        "axis", FeatureType.AXIS,
        origin=(0, 0, 0), direction=(1, 0, 0),
    ))

    body_b = Body("B")
    body_b.add_feature(Feature(
        "point", FeatureType.POINT, origin=(0, 0, 0),
    ))
    body_b.add_feature(Feature(
        "plane", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, 1),
    ))
    body_b.add_feature(Feature(
        "axis", FeatureType.AXIS,
        origin=(0, 0, 0), direction=(0, 1, 0),
    ))

    assy = Assembly(name="Test")
    assy.add_body(body_a, origin=(0, 0, 0))
    assy.add_body(body_b, origin=(10, 20, 30))

    assy.set_measurement(Measurement(
        "meas", measurement_type=meas_type, **meas_kwargs,
    ))
    return assy


class TestPointToPlane:
    def test_above_plane(self):
        """Point above plane: positive distance."""
        assy = _make_two_body_assembly(
            MeasurementType.POINT_TO_PLANE,
            body_a="B", feature_a="point",
            body_b="A", feature_b="plane",
        )
        # B.point is at (10, 20, 30), A.plane at Z=0 with normal (0,0,1)
        val = assy.compute_measurement()
        assert val == pytest.approx(30.0)

    def test_below_plane(self):
        """Point below plane: negative distance."""
        body_a = Body("A")
        body_a.add_feature(Feature(
            "plane", FeatureType.PLANE,
            origin=(0, 0, 10), direction=(0, 0, 1),
        ))
        body_b = Body("B")
        body_b.add_feature(Feature(
            "point", FeatureType.POINT, origin=(0, 0, 0),
        ))
        assy = Assembly(name="Test")
        assy.add_body(body_a, origin=(0, 0, 0))
        assy.add_body(body_b, origin=(0, 0, 5))
        assy.set_measurement(Measurement(
            "m", body_a="B", feature_a="point",
            body_b="A", feature_b="plane",
            measurement_type=MeasurementType.POINT_TO_PLANE,
        ))
        val = assy.compute_measurement()
        assert val == pytest.approx(-5.0)


class TestPointToLine:
    def test_perpendicular_distance(self):
        """Point offset from an axis line."""
        body_a = Body("A")
        body_a.add_feature(Feature(
            "point", FeatureType.POINT, origin=(0, 5, 0),
        ))
        body_b = Body("B")
        body_b.add_feature(Feature(
            "axis", FeatureType.AXIS,
            origin=(0, 0, 0), direction=(1, 0, 0),
        ))
        assy = Assembly(name="Test")
        assy.add_body(body_a, origin=(0, 0, 0))
        assy.add_body(body_b, origin=(0, 0, 0))
        assy.set_measurement(Measurement(
            "m", body_a="A", feature_a="point",
            body_b="B", feature_b="axis",
            measurement_type=MeasurementType.POINT_TO_LINE,
        ))
        val = assy.compute_measurement()
        assert val == pytest.approx(5.0)


class TestPlaneToPlaneAngle:
    def test_parallel_planes(self):
        body_a = Body("A")
        body_a.add_feature(Feature("p", FeatureType.PLANE, direction=(0, 0, 1)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.PLANE, direction=(0, 0, 1)))
        assy = Assembly(name="Test")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.set_measurement(Measurement(
            "m", "A", "p", "B", "p",
            measurement_type=MeasurementType.PLANE_TO_PLANE_ANGLE,
        ))
        assert assy.compute_measurement() == pytest.approx(0.0)

    def test_perpendicular_planes(self):
        body_a = Body("A")
        body_a.add_feature(Feature("p", FeatureType.PLANE, direction=(0, 0, 1)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.PLANE, direction=(1, 0, 0)))
        assy = Assembly(name="Test")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.set_measurement(Measurement(
            "m", "A", "p", "B", "p",
            measurement_type=MeasurementType.PLANE_TO_PLANE_ANGLE,
        ))
        assert assy.compute_measurement() == pytest.approx(90.0)


class TestLinePlaneAngle:
    def test_line_parallel_to_plane(self):
        """Line in the plane: angle should be 0."""
        body_a = Body("A")
        body_a.add_feature(Feature("l", FeatureType.AXIS, direction=(1, 0, 0)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.PLANE, direction=(0, 0, 1)))
        assy = Assembly(name="Test")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.set_measurement(Measurement(
            "m", "A", "l", "B", "p",
            measurement_type=MeasurementType.LINE_TO_PLANE_ANGLE,
        ))
        assert assy.compute_measurement() == pytest.approx(0.0, abs=0.01)

    def test_line_normal_to_plane(self):
        """Line perpendicular to plane: angle should be 90."""
        body_a = Body("A")
        body_a.add_feature(Feature("l", FeatureType.AXIS, direction=(0, 0, 1)))
        body_b = Body("B")
        body_b.add_feature(Feature("p", FeatureType.PLANE, direction=(0, 0, 1)))
        assy = Assembly(name="Test")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.set_measurement(Measurement(
            "m", "A", "l", "B", "p",
            measurement_type=MeasurementType.LINE_TO_PLANE_ANGLE,
        ))
        assert assy.compute_measurement() == pytest.approx(90.0)


class TestGapFlush:
    def test_gap_positive(self):
        """Positive gap means B is above A along normal."""
        body_a = Body("A")
        body_a.add_feature(Feature(
            "surface", FeatureType.PLANE,
            origin=(0, 0, 10), direction=(0, 0, 1),
        ))
        body_b = Body("B")
        body_b.add_feature(Feature(
            "surface", FeatureType.PLANE,
            origin=(0, 0, 0), direction=(0, 0, -1),
        ))
        assy = Assembly(name="Test")
        assy.add_body(body_a, origin=(0, 0, 0))
        assy.add_body(body_b, origin=(0, 0, 15))
        assy.set_measurement(Measurement(
            "gap", "A", "surface", "B", "surface",
            measurement_type=MeasurementType.GAP,
        ))
        val = assy.compute_measurement()
        assert val == pytest.approx(5.0)  # B at Z=15, A at Z=10, gap=5

    def test_interference_positive(self):
        """Interference is negative of gap."""
        body_a = Body("A")
        body_a.add_feature(Feature(
            "surface", FeatureType.PLANE,
            origin=(0, 0, 10), direction=(0, 0, 1),
        ))
        body_b = Body("B")
        body_b.add_feature(Feature(
            "surface", FeatureType.PLANE,
            origin=(0, 0, 0), direction=(0, 0, -1),
        ))
        assy = Assembly(name="Test")
        assy.add_body(body_a, origin=(0, 0, 0))
        assy.add_body(body_b, origin=(0, 0, 15))
        assy.set_measurement(Measurement(
            "int", "A", "surface", "B", "surface",
            measurement_type=MeasurementType.INTERFERENCE,
        ))
        val = assy.compute_measurement()
        assert val == pytest.approx(-5.0)  # No interference
