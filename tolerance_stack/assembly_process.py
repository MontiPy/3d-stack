"""Multi-stage assembly process modeling.

Supports assembly sequences, fixture/tooling variation, degrees of freedom
tracking, and move operations matching 3DCS and VisVSA process modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from tolerance_stack.assembly import Assembly, Body, Feature, FeatureType, Mate, MateType
from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Move Operations
# ---------------------------------------------------------------------------

class MoveType(Enum):
    """Types of assembly move operations."""
    FEATURE_MOVE = "feature_move"       # Align features on two parts
    RIGID_BODY_MOVE = "rigid_body_move" # Transfer between stations
    CLAMP = "clamp"                     # Clamp to fixture
    UNCLAMP = "unclamp"                 # Release from fixture


@dataclass
class MoveOperation:
    """An assembly move operation (positioning step).

    Attributes:
        name: Operation identifier.
        move_type: Type of move.
        body_name: Body being moved.
        target_body: Body or fixture being moved to.
        feature_pairs: List of (body_feature, target_feature) pairs for alignment.
        constrained_dof: Which DOFs are constrained by this move [tx, ty, tz, rx, ry, rz].
    """
    name: str
    move_type: MoveType
    body_name: str
    target_body: str = ""
    feature_pairs: list[tuple[str, str]] = field(default_factory=list)
    constrained_dof: list[bool] = field(default_factory=lambda: [False] * 6)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "move_type": self.move_type.value,
            "body_name": self.body_name,
            "target_body": self.target_body,
            "feature_pairs": self.feature_pairs,
            "constrained_dof": self.constrained_dof,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MoveOperation:
        return cls(
            name=d["name"],
            move_type=MoveType(d["move_type"]),
            body_name=d["body_name"],
            target_body=d.get("target_body", ""),
            feature_pairs=d.get("feature_pairs", []),
            constrained_dof=d.get("constrained_dof", [False] * 6),
        )


# ---------------------------------------------------------------------------
# Fixture / Tooling
# ---------------------------------------------------------------------------

@dataclass
class Fixture:
    """A fixture or tooling used in assembly.

    Fixtures have their own positional tolerances that contribute to
    the assembly variation.

    Attributes:
        name: Fixture identifier.
        features: Dict of fixture features (locating pins, pads, etc.).
        position_tol: Overall fixture position tolerance.
        repeatability: Fixture repeatability (variation across uses).
    """
    name: str
    features: dict[str, Feature] = field(default_factory=dict)
    position_tol: float = 0.0
    repeatability: float = 0.0

    def add_feature(self, feature: Feature) -> None:
        if feature.name in self.features:
            raise ValueError(f"Feature '{feature.name}' already on fixture '{self.name}'")
        self.features[feature.name] = feature

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "features": [f.to_dict() for f in self.features.values()],
            "position_tol": self.position_tol,
            "repeatability": self.repeatability,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fixture:
        fixture = cls(
            name=d["name"],
            position_tol=d.get("position_tol", 0.0),
            repeatability=d.get("repeatability", 0.0),
        )
        for fd in d.get("features", []):
            fixture.add_feature(Feature.from_dict(fd))
        return fixture


# ---------------------------------------------------------------------------
# Assembly Station
# ---------------------------------------------------------------------------

@dataclass
class AssemblyStation:
    """A station in a multi-stage assembly process.

    Each station can have its own fixtures, operations, and measurement points.

    Attributes:
        name: Station identifier.
        description: Optional description.
        fixtures: Fixtures used at this station.
        operations: Ordered list of move operations.
        bodies_added: Bodies introduced at this station.
    """
    name: str
    description: str = ""
    fixtures: list[Fixture] = field(default_factory=list)
    operations: list[MoveOperation] = field(default_factory=list)
    bodies_added: list[str] = field(default_factory=list)

    def add_fixture(self, fixture: Fixture) -> None:
        self.fixtures.append(fixture)

    def add_operation(self, op: MoveOperation) -> None:
        self.operations.append(op)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "fixtures": [f.to_dict() for f in self.fixtures],
            "operations": [op.to_dict() for op in self.operations],
            "bodies_added": self.bodies_added,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AssemblyStation:
        station = cls(
            name=d["name"],
            description=d.get("description", ""),
            bodies_added=d.get("bodies_added", []),
        )
        for fd in d.get("fixtures", []):
            station.add_fixture(Fixture.from_dict(fd))
        for od in d.get("operations", []):
            station.add_operation(MoveOperation.from_dict(od))
        return station


# ---------------------------------------------------------------------------
# DOF Tracker
# ---------------------------------------------------------------------------

@dataclass
class DOFStatus:
    """Degrees of freedom status for a body.

    Attributes:
        body_name: Body identifier.
        free_dof: [tx, ty, tz, rx, ry, rz] - True = free, False = constrained.
        total_free: Number of free DOFs.
        over_constrained: True if conflicting constraints detected.
    """
    body_name: str
    free_dof: list[bool] = field(default_factory=lambda: [True] * 6)
    over_constrained: bool = False

    @property
    def total_free(self) -> int:
        return sum(self.free_dof)

    @property
    def total_constrained(self) -> int:
        return 6 - self.total_free

    @property
    def is_fully_constrained(self) -> bool:
        return self.total_free == 0

    def summary(self) -> str:
        labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
        status = ["FREE" if f else "LOCKED" for f in self.free_dof]
        dof_str = "  ".join(f"{l}={s}" for l, s in zip(labels, status))
        oc = " [OVER-CONSTRAINED]" if self.over_constrained else ""
        return f"{self.body_name}: {dof_str}  ({self.total_free} free){oc}"


def compute_dof_status(assembly: Assembly) -> dict[str, DOFStatus]:
    """Compute DOF status for each body in the assembly.

    Analyzes mate constraints to determine which DOFs are constrained
    for each body. Detects over-constraint conditions.

    Args:
        assembly: The assembly to analyze.

    Returns:
        Dict of body_name -> DOFStatus.
    """
    statuses = {}
    for bname in assembly.bodies:
        statuses[bname] = DOFStatus(body_name=bname)

    for mate in assembly.mates:
        mt = mate.mate_type
        body_b = mate.body_b

        if body_b not in statuses:
            continue

        status = statuses[body_b]

        # Determine which DOFs are constrained by this mate type
        if mt == MateType.COINCIDENT:
            # Constrains 3 translations
            for i in range(3):
                if not status.free_dof[i]:
                    status.over_constrained = True
                status.free_dof[i] = False

        elif mt == MateType.COAXIAL:
            # Constrains 2 translations (radial) + 2 rotations (tilt)
            status.free_dof[0] = False  # Tx
            status.free_dof[1] = False  # Ty
            status.free_dof[3] = False  # Rx
            status.free_dof[4] = False  # Ry

        elif mt == MateType.COPLANAR:
            # Constrains 1 translation (normal) + 2 rotations (tilt)
            status.free_dof[2] = False  # Tz (normal)
            status.free_dof[3] = False  # Rx
            status.free_dof[4] = False  # Ry

        elif mt == MateType.AT_DISTANCE:
            # Same as coplanar but with offset
            status.free_dof[2] = False
            status.free_dof[3] = False
            status.free_dof[4] = False

        elif mt == MateType.PARALLEL:
            # Constrains 2 rotations only
            status.free_dof[3] = False
            status.free_dof[4] = False

        elif mt == MateType.CONCENTRIC:
            # Constrains 2 translations (radial)
            status.free_dof[0] = False
            status.free_dof[1] = False

    return statuses


# ---------------------------------------------------------------------------
# Assembly Process (multi-stage)
# ---------------------------------------------------------------------------

@dataclass
class AssemblyProcess:
    """A multi-stage assembly process.

    Models the real-world assembly sequence where parts are added at
    different stations, each with its own fixtures and operations.
    Fixture tolerances contribute to the overall variation.

    Attributes:
        name: Process identifier.
        assembly: The assembly being built.
        stations: Ordered list of assembly stations.
    """
    name: str
    assembly: Assembly
    stations: list[AssemblyStation] = field(default_factory=list)

    def add_station(self, station: AssemblyStation) -> None:
        self.stations.append(station)

    def all_fixture_parameters(self) -> list[dict]:
        """Collect tolerance parameters from all fixtures in the process.

        Returns:
            List of parameter dicts compatible with tolerance_parameters().
        """
        params = []
        for station in self.stations:
            for fixture in station.fixtures:
                if fixture.position_tol > 0:
                    for axis, label in enumerate(["x", "y", "z"]):
                        params.append({
                            "name": f"fixture.{fixture.name}.pos_{label}",
                            "source": "fixture_position",
                            "fixture": fixture.name,
                            "station": station.name,
                            "component": axis,
                            "nominal": 0.0,
                            "half_tol": fixture.position_tol / 2.0,
                            "sigma": 3.0,
                            "distribution": Distribution.NORMAL,
                        })
                if fixture.repeatability > 0:
                    for axis, label in enumerate(["x", "y", "z"]):
                        params.append({
                            "name": f"fixture.{fixture.name}.repeat_{label}",
                            "source": "fixture_repeatability",
                            "fixture": fixture.name,
                            "station": station.name,
                            "component": axis,
                            "nominal": 0.0,
                            "half_tol": fixture.repeatability / 2.0,
                            "sigma": 3.0,
                            "distribution": Distribution.NORMAL,
                        })
                for fname, feat in fixture.features.items():
                    if feat.position_tol > 0:
                        for axis, label in enumerate(["x", "y", "z"]):
                            params.append({
                                "name": f"fixture.{fixture.name}.{fname}.pos_{label}",
                                "source": "fixture_feature_position",
                                "fixture": fixture.name,
                                "feature": fname,
                                "station": station.name,
                                "component": axis,
                                "nominal": 0.0,
                                "half_tol": feat.position_tol / 2.0,
                                "sigma": feat.sigma,
                                "distribution": feat.distribution,
                            })
        return params

    def total_tolerance_parameters(self) -> list[dict]:
        """All tolerance parameters: assembly + fixtures."""
        params = self.assembly.tolerance_parameters()
        params.extend(self.all_fixture_parameters())
        return params

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stations": [s.to_dict() for s in self.stations],
        }

    @classmethod
    def from_dict(cls, d: dict, assembly: Assembly) -> AssemblyProcess:
        process = cls(name=d["name"], assembly=assembly)
        for sd in d.get("stations", []):
            process.add_station(AssemblyStation.from_dict(sd))
        return process
