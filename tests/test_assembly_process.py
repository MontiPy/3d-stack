"""Tests for multi-stage assembly process modeling."""

import pytest

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.assembly_process import (
    AssemblyProcess, AssemblyStation, Fixture,
    MoveOperation, MoveType, DOFStatus,
    compute_dof_status,
)


def _two_body_assembly():
    """Simple assembly for process tests."""
    body_a = Body("A")
    body_a.add_feature(Feature("top", FeatureType.PLANE,
                                origin=(0, 0, 10), direction=(0, 0, 1)))
    body_b = Body("B")
    body_b.add_feature(Feature("bottom", FeatureType.PLANE,
                                origin=(0, 0, 0), direction=(0, 0, -1)))

    assy = Assembly(name="Test")
    assy.add_body(body_a, origin=(0, 0, 0))
    assy.add_body(body_b, origin=(0, 0, 10))
    assy.add_mate(Mate("stack", "A", "top", "B", "bottom",
                        mate_type=MateType.COPLANAR))
    assy.set_measurement(Measurement(
        "height", "A", "top", "B", "bottom",
        measurement_type=MeasurementType.DISTANCE,
    ))
    return assy


class TestDOFStatus:
    def test_free_body(self):
        status = DOFStatus(body_name="B")
        assert status.total_free == 6
        assert not status.is_fully_constrained

    def test_constrained(self):
        status = DOFStatus(body_name="B", free_dof=[False] * 6)
        assert status.total_free == 0
        assert status.is_fully_constrained

    def test_summary(self):
        status = DOFStatus(body_name="B", free_dof=[True, True, False, False, False, True])
        s = status.summary()
        assert "B:" in s
        assert "FREE" in s
        assert "LOCKED" in s


class TestComputeDOF:
    def test_coplanar_mate(self):
        assy = _two_body_assembly()
        statuses = compute_dof_status(assy)
        # A has no mates constraining it (it's body_a in the mate)
        # B is body_b in the coplanar mate
        assert statuses["B"].free_dof[2] == False  # Tz locked
        assert statuses["B"].free_dof[3] == False  # Rx locked
        assert statuses["B"].free_dof[4] == False  # Ry locked
        assert statuses["B"].free_dof[0] == True   # Tx free
        assert statuses["B"].free_dof[1] == True   # Ty free
        assert statuses["B"].free_dof[5] == True   # Rz free

    def test_coaxial_mate(self):
        body_a = Body("A")
        body_a.add_feature(Feature("axis", FeatureType.CYLINDER,
                                    direction=(0, 0, 1), radius=5.0))
        body_b = Body("B")
        body_b.add_feature(Feature("axis", FeatureType.CYLINDER,
                                    direction=(0, 0, 1), radius=4.0))
        assy = Assembly(name="Test")
        assy.add_body(body_a)
        assy.add_body(body_b)
        assy.add_mate(Mate("coax", "A", "axis", "B", "axis",
                            mate_type=MateType.COAXIAL))
        statuses = compute_dof_status(assy)
        assert statuses["B"].total_constrained == 4  # Tx, Ty, Rx, Ry


class TestFixture:
    def test_basic(self):
        fixture = Fixture(name="Jig1", position_tol=0.05, repeatability=0.02)
        fixture.add_feature(Feature("pin1", FeatureType.CYLINDER,
                                     origin=(10, 0, 0), direction=(0, 0, 1),
                                     radius=3.0, position_tol=0.03))
        assert len(fixture.features) == 1
        assert fixture.position_tol == 0.05

    def test_duplicate_feature(self):
        fixture = Fixture(name="Jig1")
        fixture.add_feature(Feature("pin", FeatureType.POINT))
        with pytest.raises(ValueError, match="already on"):
            fixture.add_feature(Feature("pin", FeatureType.POINT))

    def test_serialization(self):
        fixture = Fixture(name="Jig1", position_tol=0.05)
        fixture.add_feature(Feature("pin", FeatureType.CYLINDER,
                                     origin=(10, 0, 0), direction=(0, 0, 1),
                                     radius=3.0))
        d = fixture.to_dict()
        loaded = Fixture.from_dict(d)
        assert loaded.name == "Jig1"
        assert loaded.position_tol == 0.05
        assert len(loaded.features) == 1


class TestAssemblyStation:
    def test_basic(self):
        station = AssemblyStation(name="Station1", description="Welding station")
        station.add_fixture(Fixture(name="Jig1", position_tol=0.05))
        station.add_operation(MoveOperation(
            name="place_A", move_type=MoveType.FEATURE_MOVE,
            body_name="A", target_body="Jig1",
        ))
        assert len(station.fixtures) == 1
        assert len(station.operations) == 1

    def test_serialization(self):
        station = AssemblyStation(name="S1", bodies_added=["A", "B"])
        d = station.to_dict()
        loaded = AssemblyStation.from_dict(d)
        assert loaded.name == "S1"
        assert loaded.bodies_added == ["A", "B"]


class TestAssemblyProcess:
    def test_multi_stage(self):
        assy = _two_body_assembly()
        process = AssemblyProcess(name="Test Process", assembly=assy)

        station1 = AssemblyStation(name="Station1", bodies_added=["A"])
        station1.add_fixture(Fixture(name="Base Jig", position_tol=0.03))
        process.add_station(station1)

        station2 = AssemblyStation(name="Station2", bodies_added=["B"])
        station2.add_fixture(Fixture(name="Top Jig", position_tol=0.05))
        process.add_station(station2)

        assert len(process.stations) == 2

    def test_fixture_parameters(self):
        assy = _two_body_assembly()
        process = AssemblyProcess(name="Test", assembly=assy)

        station = AssemblyStation(name="S1")
        station.add_fixture(Fixture(name="Jig", position_tol=0.10, repeatability=0.05))
        process.add_station(station)

        params = process.all_fixture_parameters()
        assert len(params) == 6  # 3 pos + 3 repeat

    def test_total_parameters(self):
        assy = _two_body_assembly()
        process = AssemblyProcess(name="Test", assembly=assy)

        station = AssemblyStation(name="S1")
        station.add_fixture(Fixture(name="Jig", position_tol=0.10))
        process.add_station(station)

        total = process.total_tolerance_parameters()
        fixture_params = [p for p in total if "fixture" in p.get("source", "")]
        assy_params = [p for p in total if "fixture" not in p.get("source", "")]
        assert len(fixture_params) == 3
