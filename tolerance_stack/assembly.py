"""3D rigid body assembly model for tolerance stack analysis.

A body-based assembly model where rigid bodies carry geometric features
(points, planes, axes, cylinders) and are connected through mating
conditions. Tolerances on feature positions and orientations propagate
through the assembly graph to a measurement of interest.

Key concepts:
    Feature   — A geometric element on a body (point, plane, axis, cylinder)
                with a position and orientation in the body's local frame,
                plus optional tolerances.
    Body      — A rigid 3D part with a local coordinate frame and features.
    Mate      — A constraint between two features on different bodies
                (coincident, coaxial, coplanar, at-distance).
    Assembly  — A collection of bodies + mates + a measurement definition.
    Measurement — The quantity of interest (distance or angle between two
                  features after assembly).

Usage:
    body_a = Body("Housing")
    body_a.add_feature(Feature("bore_axis", FeatureType.AXIS,
                               origin=(0,0,0), direction=(0,0,1),
                               position_tol=0.05))
    body_b = Body("Shaft")
    body_b.add_feature(Feature("shaft_axis", FeatureType.AXIS,
                               origin=(0,0,0), direction=(0,0,1),
                               position_tol=0.03))

    assy = Assembly("Housing-Shaft")
    assy.add_body(body_a, placement=(0,0,0))
    assy.add_body(body_b, placement=(0,0,10))
    assy.add_mate(Mate("coaxial", "Housing", "bore_axis",
                        "Shaft", "shaft_axis", MateType.COAXIAL))
    assy.set_measurement("Housing", "top_face", "Shaft", "flange_face")

    results = analyze_assembly(assy)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Transform helpers (reuse from linkage where possible)
# ---------------------------------------------------------------------------

def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler ZYX angles (degrees)."""
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def _make_transform(
    origin: tuple[float, float, float] = (0, 0, 0),
    rotation: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from origin + Euler ZYX angles."""
    T = np.eye(4)
    T[:3, :3] = _rotation_matrix(*rotation)
    T[:3, 3] = origin
    return T


# ---------------------------------------------------------------------------
# Feature types
# ---------------------------------------------------------------------------

class FeatureType(Enum):
    """Geometric feature types."""
    POINT = "point"           # A single 3D point
    PLANE = "plane"           # A plane defined by origin + normal
    AXIS = "axis"             # A line defined by origin + direction
    CYLINDER = "cylinder"     # A cylinder (axis + radius)
    CIRCLE = "circle"         # A circle (axis + radius, on a plane)


class MateType(Enum):
    """Types of mating conditions between features."""
    COINCIDENT = "coincident"   # Two points, or point-on-plane, etc.
    COAXIAL = "coaxial"         # Two axes aligned
    COPLANAR = "coplanar"       # Two planes flush
    AT_DISTANCE = "at_distance" # Controlled distance between features
    PARALLEL = "parallel"       # Two directions parallel (no positional lock)
    CONCENTRIC = "concentric"   # Two circles/cylinders share center


class MeasurementType(Enum):
    """What to measure between two features."""
    DISTANCE = "distance"             # Point-to-point or feature-to-feature distance
    DISTANCE_ALONG = "distance_along" # Distance projected onto a direction
    ANGLE = "angle"                   # Angle between two directions


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """A geometric feature on a rigid body.

    Attributes:
        name: Feature identifier (unique within a body).
        feature_type: The geometric type.
        origin: Position in the body's local frame [x, y, z].
        direction: Direction vector (for plane normal, axis direction, etc.).
        radius: Radius (for CYLINDER and CIRCLE types).
        position_tol: Positional tolerance (bilateral, applied to origin).
        orientation_tol: Orientation tolerance in degrees (bilateral).
        form_tol: Form tolerance (flatness, cylindricity, etc.).
        position_tol_direction: Optional specific direction for the position
            tolerance (if None, tolerance applies radially in all directions).
        distribution: Statistical distribution for MC sampling.
        sigma: Sigma level the tolerance band represents.
    """
    name: str
    feature_type: FeatureType
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0
    position_tol: float = 0.0
    orientation_tol: float = 0.0
    form_tol: float = 0.0
    position_tol_direction: Optional[tuple[float, float, float]] = None
    distribution: Distribution = Distribution.NORMAL
    sigma: float = 3.0

    def __post_init__(self) -> None:
        d = np.array(self.direction, dtype=float)
        mag = np.linalg.norm(d)
        if mag > 1e-12:
            self.direction = tuple(d / mag)

    @property
    def has_tolerance(self) -> bool:
        return self.position_tol > 0 or self.orientation_tol > 0 or self.form_tol > 0

    def world_origin(self, body_transform: np.ndarray) -> np.ndarray:
        """Feature origin in world coordinates given the body's 4x4 transform."""
        pt = np.array([*self.origin, 1.0])
        return (body_transform @ pt)[:3]

    def world_direction(self, body_transform: np.ndarray) -> np.ndarray:
        """Feature direction in world coordinates (rotation only)."""
        d = np.array(self.direction, dtype=float)
        return body_transform[:3, :3] @ d

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "origin": list(self.origin),
            "direction": list(self.direction),
            "radius": self.radius,
            "position_tol": self.position_tol,
            "orientation_tol": self.orientation_tol,
            "form_tol": self.form_tol,
            "distribution": self.distribution.value,
            "sigma": self.sigma,
        }
        if self.position_tol_direction is not None:
            d["position_tol_direction"] = list(self.position_tol_direction)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        ptd = d.get("position_tol_direction")
        if ptd is not None:
            ptd = tuple(ptd)
        return cls(
            name=d["name"],
            feature_type=FeatureType(d["feature_type"]),
            origin=tuple(d.get("origin", [0, 0, 0])),
            direction=tuple(d.get("direction", [0, 0, 1])),
            radius=d.get("radius", 0.0),
            position_tol=d.get("position_tol", 0.0),
            orientation_tol=d.get("orientation_tol", 0.0),
            form_tol=d.get("form_tol", 0.0),
            position_tol_direction=ptd,
            distribution=Distribution(d.get("distribution", "normal")),
            sigma=d.get("sigma", 3.0),
        )


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

@dataclass
class Body:
    """A rigid 3D body (part) with geometric features.

    Attributes:
        name: Unique body identifier.
        features: Dict of features keyed by name.
        description: Optional description.
    """
    name: str
    features: dict[str, Feature] = field(default_factory=dict)
    description: str = ""

    def add_feature(self, feature: Feature) -> None:
        if feature.name in self.features:
            raise ValueError(f"Feature '{feature.name}' already exists on body '{self.name}'")
        self.features[feature.name] = feature

    def get_feature(self, name: str) -> Feature:
        if name not in self.features:
            raise KeyError(f"Feature '{name}' not found on body '{self.name}'")
        return self.features[name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "features": [f.to_dict() for f in self.features.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Body:
        body = cls(name=d["name"], description=d.get("description", ""))
        for fd in d.get("features", []):
            body.add_feature(Feature.from_dict(fd))
        return body


# ---------------------------------------------------------------------------
# Mate
# ---------------------------------------------------------------------------

@dataclass
class Mate:
    """A mating condition between features on two bodies.

    Attributes:
        name: Mate identifier.
        body_a: Name of the first body.
        feature_a: Feature name on body_a.
        body_b: Name of the second body.
        feature_b: Feature name on body_b.
        mate_type: Type of constraint (coincident, coaxial, etc.).
        distance: For AT_DISTANCE mates, the nominal separation.
        distance_tol: Tolerance on the distance (bilateral).
        distribution: Distribution for MC sampling of distance tolerance.
        sigma: Sigma level.
    """
    name: str
    body_a: str
    feature_a: str
    body_b: str
    feature_b: str
    mate_type: MateType = MateType.COINCIDENT
    distance: float = 0.0
    distance_tol: float = 0.0
    distribution: Distribution = Distribution.NORMAL
    sigma: float = 3.0

    @property
    def has_tolerance(self) -> bool:
        return self.distance_tol > 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "body_a": self.body_a,
            "feature_a": self.feature_a,
            "body_b": self.body_b,
            "feature_b": self.feature_b,
            "mate_type": self.mate_type.value,
            "distance": self.distance,
            "distance_tol": self.distance_tol,
            "distribution": self.distribution.value,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Mate:
        return cls(
            name=d["name"],
            body_a=d["body_a"],
            feature_a=d["feature_a"],
            body_b=d["body_b"],
            feature_b=d["feature_b"],
            mate_type=MateType(d.get("mate_type", "coincident")),
            distance=d.get("distance", 0.0),
            distance_tol=d.get("distance_tol", 0.0),
            distribution=Distribution(d.get("distribution", "normal")),
            sigma=d.get("sigma", 3.0),
        )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """Defines what to measure in the assembly.

    Attributes:
        name: Measurement identifier.
        body_a: First body name.
        feature_a: Feature on body_a.
        body_b: Second body name.
        feature_b: Feature on body_b.
        measurement_type: DISTANCE, DISTANCE_ALONG, or ANGLE.
        direction: For DISTANCE_ALONG, the projection direction.
    """
    name: str
    body_a: str
    feature_a: str
    body_b: str
    feature_b: str
    measurement_type: MeasurementType = MeasurementType.DISTANCE
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "body_a": self.body_a,
            "feature_a": self.feature_a,
            "body_b": self.body_b,
            "feature_b": self.feature_b,
            "measurement_type": self.measurement_type.value,
            "direction": list(self.direction),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Measurement:
        return cls(
            name=d["name"],
            body_a=d["body_a"],
            feature_a=d["feature_a"],
            body_b=d["body_b"],
            feature_b=d["feature_b"],
            measurement_type=MeasurementType(d.get("measurement_type", "distance")),
            direction=tuple(d.get("direction", [1, 0, 0])),
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class BodyPlacement:
    """A body placed in the assembly with a nominal position and orientation.

    Attributes:
        body: The body object.
        origin: Nominal position [x, y, z] in world frame.
        rotation: Nominal Euler ZYX rotation [rx, ry, rz] in degrees.
    """
    body: Body
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def transform(self) -> np.ndarray:
        return _make_transform(self.origin, self.rotation)


@dataclass
class Assembly:
    """A 3D assembly of rigid bodies connected by mates.

    Attributes:
        name: Assembly identifier.
        description: Optional description.
        bodies: Dict of body placements keyed by body name.
        mates: List of mating conditions.
        measurement: The measurement of interest (set after construction).
    """
    name: str
    description: str = ""
    bodies: dict[str, BodyPlacement] = field(default_factory=dict)
    mates: list[Mate] = field(default_factory=list)
    measurement: Optional[Measurement] = None

    def add_body(
        self,
        body: Body,
        origin: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        if body.name in self.bodies:
            raise ValueError(f"Body '{body.name}' already in assembly")
        self.bodies[body.name] = BodyPlacement(body, origin, rotation)

    def add_mate(self, mate: Mate) -> None:
        if mate.body_a not in self.bodies:
            raise ValueError(f"Body '{mate.body_a}' not in assembly")
        if mate.body_b not in self.bodies:
            raise ValueError(f"Body '{mate.body_b}' not in assembly")
        self.mates.append(mate)

    def set_measurement(self, measurement: Measurement) -> None:
        if measurement.body_a not in self.bodies:
            raise ValueError(f"Body '{measurement.body_a}' not in assembly")
        if measurement.body_b not in self.bodies:
            raise ValueError(f"Body '{measurement.body_b}' not in assembly")
        self.measurement = measurement

    def get_body(self, name: str) -> BodyPlacement:
        if name not in self.bodies:
            raise KeyError(f"Body '{name}' not in assembly")
        return self.bodies[name]

    def compute_measurement(
        self,
        feature_offsets: Optional[dict[tuple[str, str], np.ndarray]] = None,
        feature_angle_offsets: Optional[dict[tuple[str, str], np.ndarray]] = None,
        mate_distance_offsets: Optional[dict[str, float]] = None,
    ) -> float:
        """Compute the measurement value with optional perturbations.

        This is the core function for tolerance analysis. It:
        1. Places each body at its nominal transform.
        2. Applies feature position/orientation offsets (from tolerances).
        3. Evaluates the measurement between the two target features.

        For mating conditions, the assembly is treated as a stacked chain:
        body placements are nominal, and feature tolerances perturb the
        feature locations. Mate distance tolerances shift the effective
        position of body_b along the mate direction.

        Args:
            feature_offsets: {(body_name, feature_name): [dx, dy, dz]}
            feature_angle_offsets: {(body_name, feature_name): [drx, dry, drz] degrees}
            mate_distance_offsets: {mate_name: delta_distance}

        Returns:
            Scalar measurement value.
        """
        fo = feature_offsets or {}
        fao = feature_angle_offsets or {}
        mdo = mate_distance_offsets or {}

        # Build effective body transforms incorporating mate shifts
        body_transforms = self._compute_body_transforms(mdo)

        meas = self.measurement
        if meas is None:
            raise ValueError("No measurement defined on assembly")

        # Get feature world positions with offsets
        pos_a = self._feature_world_pos(meas.body_a, meas.feature_a,
                                         body_transforms, fo, fao)
        pos_b = self._feature_world_pos(meas.body_b, meas.feature_b,
                                         body_transforms, fo, fao)

        if meas.measurement_type == MeasurementType.DISTANCE:
            return float(np.linalg.norm(pos_b - pos_a))

        elif meas.measurement_type == MeasurementType.DISTANCE_ALONG:
            d = np.array(meas.direction, dtype=float)
            d = d / np.linalg.norm(d)
            return float(np.dot(pos_b - pos_a, d))

        elif meas.measurement_type == MeasurementType.ANGLE:
            dir_a = self._feature_world_dir(meas.body_a, meas.feature_a,
                                             body_transforms, fao)
            dir_b = self._feature_world_dir(meas.body_b, meas.feature_b,
                                             body_transforms, fao)
            cos_angle = np.clip(np.dot(dir_a, dir_b), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))

        raise ValueError(f"Unknown measurement type: {meas.measurement_type}")

    def _compute_body_transforms(
        self,
        mate_distance_offsets: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Compute world transforms for each body, incorporating mate shifts."""
        transforms = {}
        for bname, bp in self.bodies.items():
            transforms[bname] = bp.transform.copy()

        # Apply mate distance offsets: shift body_b along the mate direction
        for mate in self.mates:
            delta = mate_distance_offsets.get(mate.name, 0.0)
            if abs(delta) < 1e-15:
                continue

            bp_a = self.bodies[mate.body_a]
            feat_a = bp_a.body.get_feature(mate.feature_a)
            # Shift direction is the mate feature's world direction
            world_dir = feat_a.world_direction(transforms[mate.body_a])
            shift = world_dir * delta
            transforms[mate.body_b][:3, 3] += shift

        return transforms

    def _feature_world_pos(
        self,
        body_name: str,
        feature_name: str,
        body_transforms: dict[str, np.ndarray],
        feature_offsets: dict[tuple[str, str], np.ndarray],
        feature_angle_offsets: dict[tuple[str, str], np.ndarray],
    ) -> np.ndarray:
        """Compute a feature's world position with perturbations."""
        bp = self.bodies[body_name]
        feat = bp.body.get_feature(feature_name)
        T = body_transforms[body_name]

        # Base world position
        pos = feat.world_origin(T)

        # Apply position offset (in world frame)
        key = (body_name, feature_name)
        if key in feature_offsets:
            pos = pos + feature_offsets[key]

        # Apply orientation offset effect on position
        # (for features offset from body origin, rotation changes position)
        if key in feature_angle_offsets:
            angles = feature_angle_offsets[key]
            R_pert = _rotation_matrix(angles[0], angles[1], angles[2])
            local_origin = np.array(feat.origin, dtype=float)
            # The perturbation shifts the feature position by R_pert * local - local
            delta = R_pert @ local_origin - local_origin
            pos = pos + T[:3, :3] @ delta

        return pos

    def _feature_world_dir(
        self,
        body_name: str,
        feature_name: str,
        body_transforms: dict[str, np.ndarray],
        feature_angle_offsets: dict[tuple[str, str], np.ndarray],
    ) -> np.ndarray:
        """Compute a feature's world direction with orientation perturbation."""
        bp = self.bodies[body_name]
        feat = bp.body.get_feature(feature_name)
        T = body_transforms[body_name]

        d = feat.world_direction(T)

        key = (body_name, feature_name)
        if key in feature_angle_offsets:
            angles = feature_angle_offsets[key]
            R_pert = _rotation_matrix(angles[0], angles[1], angles[2])
            d = T[:3, :3] @ R_pert @ np.array(feat.direction, dtype=float)
            mag = np.linalg.norm(d)
            if mag > 1e-12:
                d = d / mag

        return d

    def tolerance_parameters(self) -> list[dict]:
        """List all tolerance-bearing parameters for analysis.

        Returns list of dicts with keys:
            name, source, kind, nominal, half_tol, sigma, distribution,
            body, feature, component
        """
        params = []

        for bname, bp in self.bodies.items():
            for fname, feat in bp.body.features.items():
                if feat.position_tol > 0:
                    if feat.position_tol_direction is not None:
                        # Single-direction positional tolerance
                        params.append({
                            "name": f"{bname}.{fname}.pos",
                            "source": "feature_position",
                            "body": bname,
                            "feature": fname,
                            "component": "directed",
                            "nominal": 0.0,
                            "half_tol": feat.position_tol / 2.0,
                            "sigma": feat.sigma,
                            "distribution": feat.distribution,
                        })
                    else:
                        # Per-axis positional tolerance
                        for axis, label in enumerate(["x", "y", "z"]):
                            params.append({
                                "name": f"{bname}.{fname}.pos_{label}",
                                "source": "feature_position",
                                "body": bname,
                                "feature": fname,
                                "component": axis,
                                "nominal": 0.0,
                                "half_tol": feat.position_tol / 2.0,
                                "sigma": feat.sigma,
                                "distribution": feat.distribution,
                            })

                if feat.orientation_tol > 0:
                    # Orientation tolerance as small-angle rotations
                    # Apply about the two axes perpendicular to the feature direction
                    perp_axes = _perpendicular_axes(np.array(feat.direction))
                    for i, (axis_label, _) in enumerate(zip(["u", "v"], perp_axes)):
                        params.append({
                            "name": f"{bname}.{fname}.orient_{axis_label}",
                            "source": "feature_orientation",
                            "body": bname,
                            "feature": fname,
                            "component": i,
                            "nominal": 0.0,
                            "half_tol": feat.orientation_tol / 2.0,
                            "sigma": feat.sigma,
                            "distribution": feat.distribution,
                        })

        for mate in self.mates:
            if mate.has_tolerance:
                params.append({
                    "name": f"mate.{mate.name}.dist",
                    "source": "mate_distance",
                    "mate": mate.name,
                    "component": "distance",
                    "nominal": mate.distance,
                    "half_tol": mate.distance_tol / 2.0,
                    "sigma": mate.sigma,
                    "distribution": mate.distribution,
                })

        return params

    def save(self, path: str) -> None:
        data = {
            "type": "assembly",
            "name": self.name,
            "description": self.description,
            "bodies": [],
            "mates": [m.to_dict() for m in self.mates],
            "measurement": self.measurement.to_dict() if self.measurement else None,
        }
        for bname, bp in self.bodies.items():
            data["bodies"].append({
                **bp.body.to_dict(),
                "placement_origin": list(bp.origin),
                "placement_rotation": list(bp.rotation),
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Assembly:
        with open(path) as f:
            data = json.load(f)
        assy = cls(name=data["name"], description=data.get("description", ""))

        for bd in data.get("bodies", []):
            body = Body.from_dict(bd)
            origin = tuple(bd.get("placement_origin", [0, 0, 0]))
            rotation = tuple(bd.get("placement_rotation", [0, 0, 0]))
            assy.add_body(body, origin=origin, rotation=rotation)

        for md in data.get("mates", []):
            assy.add_mate(Mate.from_dict(md))

        meas_d = data.get("measurement")
        if meas_d:
            assy.set_measurement(Measurement.from_dict(meas_d))

        return assy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perpendicular_axes(direction: np.ndarray) -> list[np.ndarray]:
    """Return two unit vectors perpendicular to the given direction."""
    d = direction / np.linalg.norm(direction)
    # Pick a seed vector not parallel to d
    if abs(d[0]) < 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    else:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, seed)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return [u, v]
