"""3D linkage model for tolerance stack analysis.

A linkage is a kinematic chain of rigid links connected by joints.
Each link has a nominal length and tolerances. Each joint has a nominal
angle/offset and tolerances. The linkage computes forward kinematics
using 4x4 homogeneous transformation matrices and derives sensitivities
via the geometric Jacobian.

Joint types:
    REVOLUTE_X/Y/Z  — rotation about a fixed axis
    PRISMATIC_X/Y/Z — translation along a fixed axis
    SPHERICAL        — 3-DOF rotation (Euler ZYX)
    FIXED            — rigid connection (no DOF, but transform applies)

Usage:
    linkage = Linkage("My Mechanism")
    linkage.add_joint(Joint("J0", JointType.FIXED))
    linkage.add_link(Link("Link1", length=100, plus_tol=0.1, minus_tol=0.1,
                          direction=(1, 0, 0)))
    linkage.add_joint(Joint("J1", JointType.REVOLUTE_Z, nominal=30.0,
                            plus_tol=0.5, minus_tol=0.5))
    linkage.add_link(Link("Link2", length=80, plus_tol=0.08, minus_tol=0.08,
                          direction=(1, 0, 0)))
    linkage.add_joint(Joint("J2", JointType.FIXED))

    result = analyze_linkage(linkage, target_point=(180, 0, 0))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Transformation matrix helpers
# ---------------------------------------------------------------------------

def _rotx(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation about X axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ])


def _roty(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation about Y axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ])


def _rotz(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation about Z axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _trans(dx: float, dy: float, dz: float) -> np.ndarray:
    """4x4 homogeneous pure translation."""
    T = np.eye(4)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T


# ---------------------------------------------------------------------------
# Joint and Link types
# ---------------------------------------------------------------------------

class JointType(Enum):
    """Types of joints in a 3D linkage."""
    FIXED = "fixed"
    REVOLUTE_X = "revolute_x"
    REVOLUTE_Y = "revolute_y"
    REVOLUTE_Z = "revolute_z"
    PRISMATIC_X = "prismatic_x"
    PRISMATIC_Y = "prismatic_y"
    PRISMATIC_Z = "prismatic_z"
    SPHERICAL = "spherical"  # 3-DOF (Euler ZYX angles)


@dataclass
class Joint:
    """A joint in a 3D linkage.

    Attributes:
        name: Descriptive name.
        joint_type: The type of joint (determines DOFs and transform).
        nominal: Nominal joint value (degrees for revolute, mm for prismatic).
                 For SPHERICAL: tuple of (rz, ry, rx) in degrees.
        plus_tol: Upper tolerance (degrees or mm).
        minus_tol: Lower tolerance (degrees or mm).
                   For SPHERICAL: applies equally to all 3 angles.
        distribution: Statistical distribution for Monte Carlo.
        sigma: Number of sigma the tolerance band represents.
    """
    name: str
    joint_type: JointType = JointType.FIXED
    nominal: float | tuple[float, float, float] = 0.0
    plus_tol: float = 0.0
    minus_tol: float = 0.0
    distribution: Distribution = Distribution.NORMAL
    sigma: float = 3.0

    @property
    def has_tolerance(self) -> bool:
        return self.plus_tol > 0 or self.minus_tol > 0

    @property
    def half_tolerance(self) -> float:
        return (self.plus_tol + self.minus_tol) / 2.0

    @property
    def midpoint_shift(self) -> float:
        return (self.plus_tol - self.minus_tol) / 2.0

    @property
    def dof_count(self) -> int:
        if self.joint_type == JointType.FIXED:
            return 0
        elif self.joint_type == JointType.SPHERICAL:
            return 3
        else:
            return 1

    def transform(self, value: float | tuple | None = None) -> np.ndarray:
        """Compute the 4x4 transform for this joint at the given value.

        Args:
            value: Joint parameter. If None, uses nominal.
        """
        jt = self.joint_type

        if jt == JointType.FIXED:
            return np.eye(4)

        if jt == JointType.SPHERICAL:
            if value is None:
                value = self.nominal if isinstance(self.nominal, tuple) else (self.nominal, 0.0, 0.0)
            rz, ry, rx = value
            return _rotz(rz) @ _roty(ry) @ _rotx(rx)

        if value is None:
            value = self.nominal if isinstance(self.nominal, (int, float)) else 0.0

        if jt == JointType.REVOLUTE_X:
            return _rotx(value)
        elif jt == JointType.REVOLUTE_Y:
            return _roty(value)
        elif jt == JointType.REVOLUTE_Z:
            return _rotz(value)
        elif jt == JointType.PRISMATIC_X:
            return _trans(value, 0, 0)
        elif jt == JointType.PRISMATIC_Y:
            return _trans(0, value, 0)
        elif jt == JointType.PRISMATIC_Z:
            return _trans(0, 0, value)

        raise ValueError(f"Unknown joint type: {jt}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "joint_type": self.joint_type.value,
            "nominal": list(self.nominal) if isinstance(self.nominal, tuple) else self.nominal,
            "plus_tol": self.plus_tol,
            "minus_tol": self.minus_tol,
            "distribution": self.distribution.value,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Joint:
        nom = d.get("nominal", 0.0)
        if isinstance(nom, list):
            nom = tuple(nom)
        return cls(
            name=d["name"],
            joint_type=JointType(d["joint_type"]),
            nominal=nom,
            plus_tol=d.get("plus_tol", 0.0),
            minus_tol=d.get("minus_tol", 0.0),
            distribution=Distribution(d.get("distribution", "normal")),
            sigma=d.get("sigma", 3.0),
        )


@dataclass
class Link:
    """A rigid link in a 3D linkage.

    Defines a translation from one joint to the next, with tolerances
    on the link length.

    Attributes:
        name: Descriptive name.
        length: Nominal link length.
        plus_tol: Upper tolerance on length.
        minus_tol: Lower tolerance on length.
        direction: Unit vector along the link in the local frame.
        distribution: Statistical distribution for Monte Carlo.
        sigma: Number of sigma the tolerance band represents.
    """
    name: str
    length: float
    plus_tol: float = 0.0
    minus_tol: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    distribution: Distribution = Distribution.NORMAL
    sigma: float = 3.0

    def __post_init__(self) -> None:
        d = np.array(self.direction, dtype=float)
        mag = np.linalg.norm(d)
        if mag < 1e-12:
            raise ValueError("Link direction must be non-zero")
        self.direction = tuple(d / mag)

    @property
    def has_tolerance(self) -> bool:
        return self.plus_tol > 0 or self.minus_tol > 0

    @property
    def half_tolerance(self) -> float:
        return (self.plus_tol + self.minus_tol) / 2.0

    @property
    def midpoint_shift(self) -> float:
        return (self.plus_tol - self.minus_tol) / 2.0

    def transform(self, length: float | None = None) -> np.ndarray:
        """4x4 translation along the link direction by the given length."""
        L = length if length is not None else self.length
        d = np.array(self.direction, dtype=float)
        return _trans(*(d * L))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "plus_tol": self.plus_tol,
            "minus_tol": self.minus_tol,
            "direction": list(self.direction),
            "distribution": self.distribution.value,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Link:
        return cls(
            name=d["name"],
            length=d["length"],
            plus_tol=d.get("plus_tol", 0.0),
            minus_tol=d.get("minus_tol", 0.0),
            direction=tuple(d.get("direction", [1, 0, 0])),
            distribution=Distribution(d.get("distribution", "normal")),
            sigma=d.get("sigma", 3.0),
        )


# ---------------------------------------------------------------------------
# Linkage: ordered chain of alternating joints and links
# ---------------------------------------------------------------------------

@dataclass
class Linkage:
    """A 3D kinematic linkage (chain of joints and links).

    The linkage is built by alternating add_joint() and add_link() calls.
    The chain always starts and ends with a joint:
        J0 -> L0 -> J1 -> L1 -> J2 -> ... -> Jn

    Attributes:
        name: Descriptive name.
        description: Optional description.
        joints: Ordered list of joints.
        links: Ordered list of links (len = len(joints) - 1 when complete).
    """
    name: str
    description: str = ""
    joints: list[Joint] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)

    def add_joint(self, joint: Joint) -> None:
        """Add the next joint. Must alternate with add_link."""
        if len(self.joints) != len(self.links):
            raise ValueError(
                f"Expected a link before adding another joint. "
                f"Have {len(self.joints)} joints and {len(self.links)} links."
            )
        self.joints.append(joint)

    def add_link(self, link: Link) -> None:
        """Add the next link. Must alternate with add_joint."""
        if len(self.joints) != len(self.links) + 1:
            raise ValueError(
                f"Expected a joint before adding a link. "
                f"Have {len(self.joints)} joints and {len(self.links)} links."
            )
        self.links.append(link)

    @property
    def is_complete(self) -> bool:
        """A complete linkage has n joints and n-1 links (n >= 2)."""
        return len(self.joints) >= 2 and len(self.joints) == len(self.links) + 1

    def validate(self) -> None:
        if not self.is_complete:
            raise ValueError(
                f"Linkage is incomplete: {len(self.joints)} joints, {len(self.links)} links. "
                f"Chain must end with a joint."
            )

    def forward_kinematics(
        self,
        joint_values: dict[str, float | tuple] | None = None,
        link_lengths: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute the end-effector 4x4 transform.

        Multiplies all joint and link transforms in chain order:
            T = J0 * L0 * J1 * L1 * ... * Jn

        Args:
            joint_values: Optional override dict {joint_name: value}.
            link_lengths: Optional override dict {link_name: length}.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        self.validate()
        jv = joint_values or {}
        ll = link_lengths or {}

        T = np.eye(4)

        for i, joint in enumerate(self.joints):
            val = jv.get(joint.name, None)
            T = T @ joint.transform(val)

            if i < len(self.links):
                link = self.links[i]
                length = ll.get(link.name, None)
                T = T @ link.transform(length)

        return T

    def end_effector_position(
        self,
        joint_values: dict[str, float | tuple] | None = None,
        link_lengths: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute the 3D position of the end-effector (chain tip)."""
        T = self.forward_kinematics(joint_values, link_lengths)
        return T[:3, 3]

    def all_joint_positions(
        self,
        joint_values: dict[str, float | tuple] | None = None,
        link_lengths: dict[str, float] | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Compute the 3D position of every joint in the chain.

        Returns:
            List of (joint_name, position_3d) tuples.
        """
        self.validate()
        jv = joint_values or {}
        ll = link_lengths or {}

        T = np.eye(4)
        positions = []

        for i, joint in enumerate(self.joints):
            val = jv.get(joint.name, None)
            T = T @ joint.transform(val)
            positions.append((joint.name, T[:3, 3].copy()))

            if i < len(self.links):
                link = self.links[i]
                length = ll.get(link.name, None)
                T = T @ link.transform(length)

        return positions

    def parameter_list(self) -> list[tuple[str, str, float, float, float]]:
        """List all tolerance-bearing parameters in chain order.

        Returns:
            List of (name, kind, nominal, plus_tol, minus_tol) tuples.
            kind is "joint" or "link".
        """
        params = []
        for j in self.joints:
            if j.has_tolerance:
                nom = j.nominal if isinstance(j.nominal, (int, float)) else 0.0
                params.append((j.name, "joint", float(nom), j.plus_tol, j.minus_tol))
        for lk in self.links:
            if lk.has_tolerance:
                params.append((lk.name, "link", lk.length, lk.plus_tol, lk.minus_tol))
        return params

    def save(self, path: str) -> None:
        """Save the linkage definition to a JSON file."""
        data = {
            "type": "linkage",
            "name": self.name,
            "description": self.description,
            "joints": [j.to_dict() for j in self.joints],
            "links": [lk.to_dict() for lk in self.links],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Linkage:
        """Load a linkage definition from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        linkage = cls(
            name=data["name"],
            description=data.get("description", ""),
        )
        joints = [Joint.from_dict(j) for j in data["joints"]]
        links = [Link.from_dict(lk) for lk in data["links"]]

        for i, j in enumerate(joints):
            linkage.add_joint(j)
            if i < len(links):
                linkage.add_link(links[i])

        return linkage
