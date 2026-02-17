"""Data models for 3D tolerance stack analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Distribution(Enum):
    """Statistical distribution for a tolerance contributor."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


class ContributorType(Enum):
    """Type of tolerance contributor."""
    LINEAR = "linear"
    ANGULAR = "angular"       # angular dimension (degrees)
    GEOMETRIC = "geometric"   # GD&T (flatness, position, etc.)


@dataclass
class Contributor:
    """A single contributor in a 3D tolerance stack.

    Each contributor has a nominal dimension, bilateral or unilateral
    tolerances, a 3D direction vector, and a sign (+1 or -1) indicating
    whether it adds to or subtracts from the gap.

    Attributes:
        name: Descriptive name for this contributor.
        nominal: Nominal dimension value.
        plus_tol: Upper tolerance (positive value).
        minus_tol: Lower tolerance (positive value, will be subtracted).
        direction: 3D unit vector [dx, dy, dz] for the contribution axis.
        sign: +1 if the dimension adds to the gap, -1 if it subtracts.
        distribution: Statistical distribution assumed for Monte Carlo.
        contributor_type: LINEAR, ANGULAR, or GEOMETRIC.
        sigma: Number of sigma the tolerance band represents (default 3).
    """
    name: str
    nominal: float
    plus_tol: float
    minus_tol: float
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    sign: int = 1
    distribution: Distribution = Distribution.NORMAL
    contributor_type: ContributorType = ContributorType.LINEAR
    sigma: float = 3.0

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")
        # Normalize direction to unit vector
        d = np.array(self.direction, dtype=float)
        mag = np.linalg.norm(d)
        if mag < 1e-12:
            raise ValueError("direction vector must be non-zero")
        self.direction = tuple(d / mag)

    @property
    def tolerance_band(self) -> float:
        """Total bilateral-equivalent tolerance band."""
        return self.plus_tol + self.minus_tol

    @property
    def midpoint_shift(self) -> float:
        """Shift of the mean from nominal due to asymmetric tolerances."""
        return (self.plus_tol - self.minus_tol) / 2.0

    @property
    def half_tolerance(self) -> float:
        """Half of the total tolerance band (used for RSS)."""
        return self.tolerance_band / 2.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nominal": self.nominal,
            "plus_tol": self.plus_tol,
            "minus_tol": self.minus_tol,
            "direction": list(self.direction),
            "sign": self.sign,
            "distribution": self.distribution.value,
            "contributor_type": self.contributor_type.value,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Contributor:
        return cls(
            name=d["name"],
            nominal=d["nominal"],
            plus_tol=d["plus_tol"],
            minus_tol=d["minus_tol"],
            direction=tuple(d.get("direction", [1, 0, 0])),
            sign=d.get("sign", 1),
            distribution=Distribution(d.get("distribution", "normal")),
            contributor_type=ContributorType(d.get("contributor_type", "linear")),
            sigma=d.get("sigma", 3.0),
        )


@dataclass
class ToleranceStack:
    """A complete 3D tolerance stack definition.

    Attributes:
        name: Descriptive name for the stack.
        description: Optional longer description.
        closure_direction: The 3D direction of the gap/closure dimension.
        contributors: List of Contributor objects in the loop.
        gap_nominal: If set, override the computed nominal gap.
    """
    name: str
    contributors: list[Contributor] = field(default_factory=list)
    description: str = ""
    closure_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gap_nominal: Optional[float] = None

    def __post_init__(self) -> None:
        d = np.array(self.closure_direction, dtype=float)
        mag = np.linalg.norm(d)
        if mag < 1e-12:
            raise ValueError("closure_direction must be non-zero")
        self.closure_direction = tuple(d / mag)

    def add(self, contributor: Contributor) -> None:
        """Add a contributor to the stack."""
        self.contributors.append(contributor)

    def save(self, path: str) -> None:
        """Save the stack definition to a JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "closure_direction": list(self.closure_direction),
            "gap_nominal": self.gap_nominal,
            "contributors": [c.to_dict() for c in self.contributors],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> ToleranceStack:
        """Load a stack definition from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        stack = cls(
            name=data["name"],
            description=data.get("description", ""),
            closure_direction=tuple(data.get("closure_direction", [1, 0, 0])),
            gap_nominal=data.get("gap_nominal"),
        )
        for c in data.get("contributors", []):
            stack.add(Contributor.from_dict(c))
        return stack
