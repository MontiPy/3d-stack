"""Full GD&T support per ASME Y14.5 / ISO 1101.

Provides tolerance types, datum reference frames, material condition
modifiers, and composite tolerance support for 3D assembly analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# GD&T tolerance categories
# ---------------------------------------------------------------------------

class GDTType(Enum):
    """GD&T tolerance types per ASME Y14.5."""
    # Form tolerances
    FLATNESS = "flatness"
    STRAIGHTNESS = "straightness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"

    # Orientation tolerances
    PERPENDICULARITY = "perpendicularity"
    ANGULARITY = "angularity"
    PARALLELISM = "parallelism"

    # Location tolerances
    POSITION = "position"
    CONCENTRICITY = "concentricity"
    SYMMETRY = "symmetry"

    # Profile tolerances
    PROFILE_SURFACE = "profile_surface"
    PROFILE_LINE = "profile_line"

    # Runout tolerances
    CIRCULAR_RUNOUT = "circular_runout"
    TOTAL_RUNOUT = "total_runout"

    # Size
    SIZE = "size"


class MaterialCondition(Enum):
    """Material condition modifiers."""
    NONE = "none"           # RFS (Regardless of Feature Size) — default
    MMC = "mmc"             # Maximum Material Condition
    LMC = "lmc"             # Least Material Condition


class DatumPrecedence(Enum):
    """Datum reference precedence."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


# ---------------------------------------------------------------------------
# Datum Reference Frame
# ---------------------------------------------------------------------------

@dataclass
class DatumFeature:
    """A feature designated as a datum.

    Attributes:
        label: Datum letter (A, B, C, ...).
        body_name: Body the datum is on.
        feature_name: Feature name on the body.
        precedence: Primary, secondary, or tertiary.
        material_condition: MMC, LMC, or None (RFS).
    """
    label: str
    body_name: str
    feature_name: str
    precedence: DatumPrecedence = DatumPrecedence.PRIMARY
    material_condition: MaterialCondition = MaterialCondition.NONE

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "body_name": self.body_name,
            "feature_name": self.feature_name,
            "precedence": self.precedence.value,
            "material_condition": self.material_condition.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatumFeature:
        return cls(
            label=d["label"],
            body_name=d["body_name"],
            feature_name=d["feature_name"],
            precedence=DatumPrecedence(d.get("precedence", "primary")),
            material_condition=MaterialCondition(d.get("material_condition", "none")),
        )


@dataclass
class DatumReferenceFrame:
    """A datum reference frame (DRF) defined by up to three datum features.

    The DRF constrains degrees of freedom for tolerance evaluation.
    Primary datum: constrains 3 DOF (e.g., plane locks translations + rotations).
    Secondary datum: constrains 2 additional DOF.
    Tertiary datum: constrains remaining DOF.

    Attributes:
        name: DRF identifier.
        datums: List of DatumFeature (1-3 features).
    """
    name: str
    datums: list[DatumFeature] = field(default_factory=list)

    def add_datum(self, datum: DatumFeature) -> None:
        if len(self.datums) >= 3:
            raise ValueError("DRF can have at most 3 datum features")
        # Validate no duplicate precedence
        for d in self.datums:
            if d.precedence == datum.precedence:
                raise ValueError(f"DRF already has a {datum.precedence.value} datum")
        self.datums.append(datum)
        # Sort by precedence order
        order = {DatumPrecedence.PRIMARY: 0, DatumPrecedence.SECONDARY: 1,
                 DatumPrecedence.TERTIARY: 2}
        self.datums.sort(key=lambda d: order[d.precedence])

    @property
    def primary(self) -> Optional[DatumFeature]:
        return next((d for d in self.datums if d.precedence == DatumPrecedence.PRIMARY), None)

    @property
    def secondary(self) -> Optional[DatumFeature]:
        return next((d for d in self.datums if d.precedence == DatumPrecedence.SECONDARY), None)

    @property
    def tertiary(self) -> Optional[DatumFeature]:
        return next((d for d in self.datums if d.precedence == DatumPrecedence.TERTIARY), None)

    @property
    def constrained_dof(self) -> int:
        """Number of DOF constrained by this DRF."""
        count = 0
        for d in self.datums:
            if d.precedence == DatumPrecedence.PRIMARY:
                count += 3
            elif d.precedence == DatumPrecedence.SECONDARY:
                count += 2
            else:
                count += 1
        return min(count, 6)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "datums": [d.to_dict() for d in self.datums],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatumReferenceFrame:
        drf = cls(name=d["name"])
        for datum_d in d.get("datums", []):
            drf.add_datum(DatumFeature.from_dict(datum_d))
        return drf


# ---------------------------------------------------------------------------
# GD&T Feature Control Frame
# ---------------------------------------------------------------------------

@dataclass
class FeatureControlFrame:
    """A GD&T Feature Control Frame (FCF).

    Represents a single GD&T callout on a feature, including the tolerance
    type, zone value, material condition, and datum references.

    Attributes:
        name: Identifier for this FCF.
        gdt_type: The type of GD&T tolerance.
        tolerance_value: The tolerance zone width/diameter.
        material_condition: Material condition modifier on the tolerance.
        datum_refs: List of datum reference labels (e.g., ["A", "B", "C"]).
        datum_conditions: Material conditions for each datum reference.
        is_diametral: True if zone is diametral (position), False if bilateral.
        composite_lower: For composite position/profile, the lower segment tolerance.
        feature_size_nominal: Nominal feature size (for MMC/LMC bonus).
        feature_size_tol_plus: Plus tolerance on feature size.
        feature_size_tol_minus: Minus tolerance on feature size.
    """
    name: str
    gdt_type: GDTType
    tolerance_value: float
    material_condition: MaterialCondition = MaterialCondition.NONE
    datum_refs: list[str] = field(default_factory=list)
    datum_conditions: list[MaterialCondition] = field(default_factory=list)
    is_diametral: bool = False
    composite_lower: Optional[float] = None
    feature_size_nominal: float = 0.0
    feature_size_tol_plus: float = 0.0
    feature_size_tol_minus: float = 0.0

    @property
    def half_tolerance(self) -> float:
        """Half of the tolerance zone (radius if diametral)."""
        if self.is_diametral:
            return self.tolerance_value / 2.0
        return self.tolerance_value / 2.0

    @property
    def bonus_tolerance(self) -> float:
        """MMC/LMC bonus tolerance (additional tolerance from departure).

        At MMC: bonus = |actual_size - MMC_size|
        Maximum bonus when feature is at LMC:
            bonus_max = feature_size_tol_plus + feature_size_tol_minus
        At LMC: bonus = |actual_size - LMC_size|
        Maximum bonus when feature is at MMC:
            bonus_max = feature_size_tol_plus + feature_size_tol_minus
        """
        if self.material_condition == MaterialCondition.NONE:
            return 0.0

        total_size_tol = self.feature_size_tol_plus + self.feature_size_tol_minus
        return total_size_tol

    @property
    def max_tolerance(self) -> float:
        """Maximum possible tolerance including bonus."""
        return self.tolerance_value + self.bonus_tolerance

    def effective_tolerance(self, actual_departure: float = 0.0) -> float:
        """Compute effective tolerance for a given departure from MMC/LMC.

        Args:
            actual_departure: How far the actual size departs from the
                material condition boundary.

        Returns:
            Effective tolerance zone value.
        """
        if self.material_condition == MaterialCondition.NONE:
            return self.tolerance_value
        # Bonus = departure from material condition boundary
        bonus = min(abs(actual_departure), self.bonus_tolerance)
        return self.tolerance_value + bonus

    def zone_axes(self) -> int:
        """Number of axes the tolerance zone constrains.

        Form: 1 (zone width in normal direction)
        Orientation: 1-2 depending on type
        Position (diametral): 2 (circular zone) or 3 (spherical)
        Profile: 1 (bilateral zone normal to surface)
        Runout: 1 (radial)
        """
        if self.gdt_type in (GDTType.FLATNESS, GDTType.STRAIGHTNESS,
                              GDTType.PROFILE_LINE, GDTType.PROFILE_SURFACE,
                              GDTType.CIRCULAR_RUNOUT, GDTType.TOTAL_RUNOUT):
            return 1
        elif self.gdt_type in (GDTType.PARALLELISM, GDTType.PERPENDICULARITY,
                                GDTType.ANGULARITY):
            return 1
        elif self.gdt_type == GDTType.POSITION:
            return 2 if self.is_diametral else 1
        elif self.gdt_type in (GDTType.CIRCULARITY, GDTType.CYLINDRICITY,
                                GDTType.CONCENTRICITY):
            return 2
        elif self.gdt_type == GDTType.SYMMETRY:
            return 1
        return 1

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "gdt_type": self.gdt_type.value,
            "tolerance_value": self.tolerance_value,
            "material_condition": self.material_condition.value,
            "datum_refs": self.datum_refs,
            "datum_conditions": [mc.value for mc in self.datum_conditions],
            "is_diametral": self.is_diametral,
            "feature_size_nominal": self.feature_size_nominal,
            "feature_size_tol_plus": self.feature_size_tol_plus,
            "feature_size_tol_minus": self.feature_size_tol_minus,
        }
        if self.composite_lower is not None:
            d["composite_lower"] = self.composite_lower
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureControlFrame:
        return cls(
            name=d["name"],
            gdt_type=GDTType(d["gdt_type"]),
            tolerance_value=d["tolerance_value"],
            material_condition=MaterialCondition(d.get("material_condition", "none")),
            datum_refs=d.get("datum_refs", []),
            datum_conditions=[MaterialCondition(mc) for mc in d.get("datum_conditions", [])],
            is_diametral=d.get("is_diametral", False),
            composite_lower=d.get("composite_lower"),
            feature_size_nominal=d.get("feature_size_nominal", 0.0),
            feature_size_tol_plus=d.get("feature_size_tol_plus", 0.0),
            feature_size_tol_minus=d.get("feature_size_tol_minus", 0.0),
        )


def fcf_to_tolerance_parameters(
    fcf: FeatureControlFrame,
    body_name: str,
    feature_name: str,
    feature_direction: tuple[float, float, float] = (0, 0, 1),
    sigma: float = 3.0,
) -> list[dict]:
    """Convert a FeatureControlFrame into tolerance parameter dicts.

    This bridges the FCF model into the assembly tolerance parameter system,
    so analysis engines can consume GD&T callouts directly.

    Returns:
        List of parameter dicts compatible with Assembly.tolerance_parameters().
    """
    from tolerance_stack.models import Distribution

    params = []
    half = fcf.half_tolerance

    gtype = fcf.gdt_type

    if gtype == GDTType.POSITION:
        if fcf.is_diametral:
            # Diametral position: 2-axis radial tolerance zone
            for axis, label in enumerate(["x", "y", "z"]):
                params.append({
                    "name": f"{body_name}.{feature_name}.{fcf.name}.pos_{label}",
                    "source": "feature_position",
                    "body": body_name,
                    "feature": feature_name,
                    "component": axis,
                    "nominal": 0.0,
                    "half_tol": half,
                    "sigma": sigma,
                    "distribution": Distribution.NORMAL,
                    "gdt_type": gtype.value,
                    "material_condition": fcf.material_condition.value,
                    "bonus_max": fcf.bonus_tolerance,
                })
        else:
            params.append({
                "name": f"{body_name}.{feature_name}.{fcf.name}.pos",
                "source": "feature_position",
                "body": body_name,
                "feature": feature_name,
                "component": "directed",
                "nominal": 0.0,
                "half_tol": half,
                "sigma": sigma,
                "distribution": Distribution.NORMAL,
                "gdt_type": gtype.value,
            })

    elif gtype in (GDTType.FLATNESS, GDTType.STRAIGHTNESS):
        # Form tolerance: zone normal to feature direction
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.form",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.NORMAL,
            "gdt_type": gtype.value,
        })

    elif gtype in (GDTType.CIRCULARITY, GDTType.CYLINDRICITY):
        # Radial form: affects radius
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.radial_form",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.RAYLEIGH,
            "gdt_type": gtype.value,
        })

    elif gtype in (GDTType.PERPENDICULARITY, GDTType.ANGULARITY, GDTType.PARALLELISM):
        # Orientation: angular zone about perpendicular axes
        from tolerance_stack.assembly import _perpendicular_axes
        d = np.array(feature_direction, dtype=float)
        if np.linalg.norm(d) > 1e-12:
            perp = _perpendicular_axes(d)
            for i, label in enumerate(["u", "v"]):
                params.append({
                    "name": f"{body_name}.{feature_name}.{fcf.name}.orient_{label}",
                    "source": "feature_orientation",
                    "body": body_name,
                    "feature": feature_name,
                    "component": i,
                    "nominal": 0.0,
                    "half_tol": half,
                    "sigma": sigma,
                    "distribution": Distribution.NORMAL,
                    "gdt_type": gtype.value,
                })

    elif gtype in (GDTType.PROFILE_SURFACE, GDTType.PROFILE_LINE):
        # Profile: bilateral zone normal to surface
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.profile",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.NORMAL,
            "gdt_type": gtype.value,
        })

    elif gtype in (GDTType.CIRCULAR_RUNOUT, GDTType.TOTAL_RUNOUT):
        # Runout: radial deviation from datum axis
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.runout",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.NORMAL,
            "gdt_type": gtype.value,
        })

    elif gtype == GDTType.CONCENTRICITY:
        # Concentricity: 2-axis position of median points
        for axis, label in enumerate(["x", "y"]):
            params.append({
                "name": f"{body_name}.{feature_name}.{fcf.name}.conc_{label}",
                "source": "feature_position",
                "body": body_name,
                "feature": feature_name,
                "component": axis,
                "nominal": 0.0,
                "half_tol": half,
                "sigma": sigma,
                "distribution": Distribution.NORMAL,
                "gdt_type": gtype.value,
            })

    elif gtype == GDTType.SYMMETRY:
        # Symmetry: single-axis position of median
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.symmetry",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.NORMAL,
            "gdt_type": gtype.value,
        })

    elif gtype == GDTType.SIZE:
        params.append({
            "name": f"{body_name}.{feature_name}.{fcf.name}.size",
            "source": "feature_position",
            "body": body_name,
            "feature": feature_name,
            "component": "directed",
            "nominal": 0.0,
            "half_tol": half,
            "sigma": sigma,
            "distribution": Distribution.NORMAL,
            "gdt_type": gtype.value,
        })

    return params
