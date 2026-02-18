"""STEP file import and PMI/GD&T extraction.

Parses STEP AP203/AP214/AP242 files to extract:
- Geometric entities (points, planes, axes, cylinders)
- Product Manufacturing Information (PMI)
- GD&T annotations (tolerance callouts)
- Assembly structure (component placement)

Uses a built-in lightweight STEP parser (no external CAD kernel required).
For full STEP B-rep geometry, an external library like pythonocc would be
needed, but this parser handles the PMI and assembly structure data that
tolerance analysis requires.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.gdt import (
    FeatureControlFrame, GDTType, MaterialCondition,
    DatumReferenceFrame, DatumFeature, DatumPrecedence,
)


# ---------------------------------------------------------------------------
# STEP entity parsing
# ---------------------------------------------------------------------------

@dataclass
class StepEntity:
    """A parsed STEP entity."""
    id: int
    type_name: str
    params: list
    raw: str = ""


def _parse_step_file(path: str) -> dict[int, StepEntity]:
    """Parse a STEP file into a dict of entities.

    Returns:
        Dict of entity_id -> StepEntity.
    """
    entities = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"STEP file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Find DATA section
    data_match = re.search(r'DATA\s*;(.*?)ENDSEC\s*;', content, re.DOTALL)
    if not data_match:
        return entities

    data_section = data_match.group(1)

    # Parse entity lines: #123 = TYPE_NAME(param1, param2, ...);
    # Handle multi-line entities
    entity_pattern = re.compile(r'#(\d+)\s*=\s*(\w+)\s*\((.*?)\)\s*;', re.DOTALL)
    for match in entity_pattern.finditer(data_section):
        eid = int(match.group(1))
        etype = match.group(2).upper()
        params_raw = match.group(3).strip()
        params = _parse_params(params_raw)
        entities[eid] = StepEntity(id=eid, type_name=etype, params=params,
                                    raw=match.group(0))

    return entities


def _parse_params(raw: str) -> list:
    """Parse STEP parameter list into Python objects."""
    params = []
    depth = 0
    current = ""

    for ch in raw:
        if ch == '(' and depth > 0:
            current += ch
            depth += 1
        elif ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            params.append(_parse_value(current.strip()))
            current = ""
        else:
            current += ch

    if current.strip():
        params.append(_parse_value(current.strip()))

    return params


def _parse_value(val: str):
    """Parse a single STEP parameter value."""
    if val == '$' or val == '*':
        return None
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    if val.startswith('#'):
        try:
            return int(val[1:])
        except ValueError:
            return val
    if val.startswith('.') and val.endswith('.'):
        return val[1:-1]  # Enum value
    if val.startswith('(') and val.endswith(')'):
        inner = val[1:-1]
        return [_parse_value(v.strip()) for v in inner.split(',') if v.strip()]
    try:
        if '.' in val or 'E' in val.upper():
            return float(val)
        return int(val)
    except ValueError:
        return val


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

def _extract_point(entities: dict[int, StepEntity], ref_id: int) -> Optional[tuple[float, float, float]]:
    """Extract a 3D point from CARTESIAN_POINT entity."""
    entity = entities.get(ref_id)
    if entity is None:
        return None
    if entity.type_name == 'CARTESIAN_POINT':
        coords = entity.params[1] if len(entity.params) > 1 else entity.params[0]
        if isinstance(coords, list) and len(coords) >= 3:
            return (float(coords[0]), float(coords[1]), float(coords[2]))
    return None


def _extract_direction(entities: dict[int, StepEntity], ref_id: int) -> Optional[tuple[float, float, float]]:
    """Extract a direction vector from DIRECTION entity."""
    entity = entities.get(ref_id)
    if entity is None:
        return None
    if entity.type_name == 'DIRECTION':
        ratios = entity.params[1] if len(entity.params) > 1 else entity.params[0]
        if isinstance(ratios, list) and len(ratios) >= 3:
            return (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    return None


def _extract_axis_placement(entities: dict[int, StepEntity], ref_id: int
                             ) -> Optional[tuple[tuple, tuple]]:
    """Extract origin and direction from AXIS2_PLACEMENT_3D."""
    entity = entities.get(ref_id)
    if entity is None:
        return None
    if entity.type_name in ('AXIS2_PLACEMENT_3D', 'AXIS1_PLACEMENT'):
        origin_ref = entity.params[1] if len(entity.params) > 1 else None
        dir_ref = entity.params[2] if len(entity.params) > 2 else None

        origin = _extract_point(entities, origin_ref) if isinstance(origin_ref, int) else (0, 0, 0)
        direction = _extract_direction(entities, dir_ref) if isinstance(dir_ref, int) else (0, 0, 1)

        return (origin or (0, 0, 0), direction or (0, 0, 1))
    return None


# ---------------------------------------------------------------------------
# PMI / GD&T extraction
# ---------------------------------------------------------------------------

# Map STEP tolerance entity types to our GDTType
_STEP_GDT_MAP = {
    'FLATNESS_TOLERANCE': GDTType.FLATNESS,
    'STRAIGHTNESS_TOLERANCE': GDTType.STRAIGHTNESS,
    'CIRCULARITY_TOLERANCE': GDTType.CIRCULARITY,
    'CYLINDRICITY_TOLERANCE': GDTType.CYLINDRICITY,
    'PERPENDICULARITY_TOLERANCE': GDTType.PERPENDICULARITY,
    'ANGULARITY_TOLERANCE': GDTType.ANGULARITY,
    'PARALLELISM_TOLERANCE': GDTType.PARALLELISM,
    'POSITION_TOLERANCE': GDTType.POSITION,
    'CONCENTRICITY_TOLERANCE': GDTType.CONCENTRICITY,
    'SYMMETRY_TOLERANCE': GDTType.SYMMETRY,
    'SURFACE_PROFILE_TOLERANCE': GDTType.PROFILE_SURFACE,
    'LINE_PROFILE_TOLERANCE': GDTType.PROFILE_LINE,
    'CIRCULAR_RUNOUT_TOLERANCE': GDTType.CIRCULAR_RUNOUT,
    'TOTAL_RUNOUT_TOLERANCE': GDTType.TOTAL_RUNOUT,
}


def _extract_gdt(entities: dict[int, StepEntity]) -> list[dict]:
    """Extract GD&T callouts from STEP entities.

    Returns list of dicts with keys: gdt_type, tolerance_value, datum_refs, etc.
    """
    gdt_list = []

    for eid, entity in entities.items():
        step_type = entity.type_name

        if step_type in _STEP_GDT_MAP:
            gdt_type = _STEP_GDT_MAP[step_type]
            tol_value = None
            datum_refs = []

            # Extract tolerance value (usually first numeric param after name)
            for p in entity.params:
                if isinstance(p, (int, float)) and p > 0:
                    tol_value = float(p)
                    break

            # Extract datum references
            for p in entity.params:
                if isinstance(p, int) and p in entities:
                    ref_entity = entities[p]
                    if ref_entity.type_name in ('DATUM_REFERENCE', 'DATUM'):
                        for dp in ref_entity.params:
                            if isinstance(dp, str) and len(dp) <= 2:
                                datum_refs.append(dp)

            if tol_value is not None:
                gdt_list.append({
                    "entity_id": eid,
                    "gdt_type": gdt_type,
                    "tolerance_value": tol_value,
                    "datum_refs": datum_refs,
                })

    return gdt_list


# ---------------------------------------------------------------------------
# Assembly extraction
# ---------------------------------------------------------------------------

def _extract_products(entities: dict[int, StepEntity]) -> list[dict]:
    """Extract product/component information from STEP."""
    products = []

    for eid, entity in entities.items():
        if entity.type_name == 'PRODUCT':
            name = entity.params[0] if entity.params else f"Part_{eid}"
            if isinstance(name, str):
                products.append({
                    "entity_id": eid,
                    "name": name,
                    "description": entity.params[1] if len(entity.params) > 1 and isinstance(entity.params[1], str) else "",
                })

    return products


def _extract_geometric_features(entities: dict[int, StepEntity]) -> list[dict]:
    """Extract geometric features (planes, cylinders, etc.) from STEP."""
    features = []

    for eid, entity in entities.items():
        if entity.type_name == 'PLANE':
            placement = entity.params[1] if len(entity.params) > 1 else None
            if isinstance(placement, int):
                ap = _extract_axis_placement(entities, placement)
                if ap:
                    origin, direction = ap
                    features.append({
                        "entity_id": eid,
                        "feature_type": FeatureType.PLANE,
                        "name": entity.params[0] if entity.params and isinstance(entity.params[0], str) else f"plane_{eid}",
                        "origin": origin,
                        "direction": direction,
                    })

        elif entity.type_name in ('CYLINDRICAL_SURFACE', 'CIRCLE'):
            placement = entity.params[1] if len(entity.params) > 1 else None
            radius = None
            for p in entity.params:
                if isinstance(p, (int, float)) and p > 0:
                    radius = float(p)

            if isinstance(placement, int):
                ap = _extract_axis_placement(entities, placement)
                if ap:
                    origin, direction = ap
                    features.append({
                        "entity_id": eid,
                        "feature_type": FeatureType.CYLINDER if entity.type_name == 'CYLINDRICAL_SURFACE' else FeatureType.CIRCLE,
                        "name": entity.params[0] if entity.params and isinstance(entity.params[0], str) else f"cyl_{eid}",
                        "origin": origin,
                        "direction": direction,
                        "radius": radius or 0.0,
                    })

    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class StepImportResult:
    """Result from importing a STEP file.

    Attributes:
        file_path: Path to the imported file.
        products: List of product/component info dicts.
        features: List of geometric feature dicts.
        gdt_callouts: List of GD&T callout dicts.
        n_entities: Total number of STEP entities parsed.
        assembly: Optional Assembly object constructed from the data.
        warnings: List of warning messages.
    """
    file_path: str = ""
    products: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)
    gdt_callouts: list[dict] = field(default_factory=list)
    n_entities: int = 0
    assembly: Optional[Assembly] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== STEP Import: {os.path.basename(self.file_path)} ===",
            f"  Total entities:    {self.n_entities}",
            f"  Products/parts:    {len(self.products)}",
            f"  Geometric features:{len(self.features)}",
            f"  GD&T callouts:     {len(self.gdt_callouts)}",
        ]
        if self.assembly:
            lines.append(f"  Assembly created:   {self.assembly.name} "
                         f"({len(self.assembly.bodies)} bodies)")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings[:10]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def import_step(path: str, assembly_name: Optional[str] = None) -> StepImportResult:
    """Import a STEP file and extract tolerance-relevant information.

    Parses the STEP file to extract:
    - Product structure (parts/components)
    - Geometric features (planes, cylinders, axes)
    - GD&T callouts (tolerance annotations)

    Constructs an Assembly object if sufficient data is found.

    Args:
        path: Path to the STEP (.stp/.step) file.
        assembly_name: Optional name override for the assembly.

    Returns:
        StepImportResult with all extracted data.
    """
    result = StepImportResult(file_path=path)

    try:
        entities = _parse_step_file(path)
    except FileNotFoundError:
        result.warnings.append(f"File not found: {path}")
        return result
    except Exception as e:
        result.warnings.append(f"Parse error: {e}")
        return result

    result.n_entities = len(entities)
    result.products = _extract_products(entities)
    result.features = _extract_geometric_features(entities)
    result.gdt_callouts = _extract_gdt(entities)

    # Attempt to construct an Assembly
    if result.products or result.features:
        name = assembly_name or os.path.splitext(os.path.basename(path))[0]
        assy = Assembly(name=name)

        if result.products:
            # Create bodies from products
            for prod in result.products:
                body = Body(name=prod["name"], description=prod.get("description", ""))
                # Add extracted features that belong to this product
                for feat_info in result.features:
                    feat = Feature(
                        name=feat_info["name"],
                        feature_type=feat_info["feature_type"],
                        origin=feat_info.get("origin", (0, 0, 0)),
                        direction=feat_info.get("direction", (0, 0, 1)),
                        radius=feat_info.get("radius", 0.0),
                    )
                    # Apply GD&T callouts
                    for gdt in result.gdt_callouts:
                        fcf = FeatureControlFrame(
                            name=f"gdt_{gdt['entity_id']}",
                            gdt_type=gdt["gdt_type"],
                            tolerance_value=gdt["tolerance_value"],
                            datum_refs=gdt.get("datum_refs", []),
                        )
                        feat.add_fcf(fcf)
                    try:
                        body.add_feature(feat)
                    except ValueError:
                        pass  # Duplicate feature name
                try:
                    assy.add_body(body)
                except ValueError:
                    pass  # Duplicate body
        else:
            # No products found - create a single body with all features
            body = Body(name=name)
            for feat_info in result.features:
                feat = Feature(
                    name=feat_info["name"],
                    feature_type=feat_info["feature_type"],
                    origin=feat_info.get("origin", (0, 0, 0)),
                    direction=feat_info.get("direction", (0, 0, 1)),
                    radius=feat_info.get("radius", 0.0),
                )
                try:
                    body.add_feature(feat)
                except ValueError:
                    pass
            if body.features:
                assy.add_body(body)

        if assy.bodies:
            result.assembly = assy

    return result


def import_step_multi(
    paths: list[str],
    assembly_name: Optional[str] = None,
) -> StepImportResult:
    """Import multiple STEP files and merge them into a single Assembly.

    Each STEP file becomes one or more bodies in the combined assembly.
    After import, users can define mates between features from different
    files and set a measurement for tolerance analysis.

    Args:
        paths: List of paths to STEP (.stp/.step) files.
        assembly_name: Optional name for the combined assembly.

    Returns:
        StepImportResult with merged data from all files.
    """
    combined = StepImportResult(
        file_path=", ".join(os.path.basename(p) for p in paths),
    )
    name = assembly_name or "Multi-Part Assembly"
    assy = Assembly(name=name)

    for path in paths:
        part_result = import_step(path)
        combined.n_entities += part_result.n_entities
        combined.products.extend(part_result.products)
        combined.features.extend(part_result.features)
        combined.gdt_callouts.extend(part_result.gdt_callouts)
        combined.warnings.extend(part_result.warnings)

        if part_result.assembly:
            for bp_name, bp in part_result.assembly.bodies.items():
                body_name = bp.body.name
                # Avoid duplicate body names across files
                if body_name in assy.bodies:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    body_name = f"{body_name}_{stem}"
                    bp.body.name = body_name
                try:
                    origin = bp.origin
                    rotation = bp.rotation
                    # Handle both tuple and ndarray origins
                    if hasattr(origin, 'tolist'):
                        origin = tuple(origin.tolist())
                    if hasattr(rotation, 'tolist'):
                        rotation = tuple(rotation.tolist())
                    assy.add_body(bp.body, origin=origin, rotation=rotation)
                except (ValueError, AttributeError) as exc:
                    combined.warnings.append(
                        f"Could not add body '{body_name}' from {os.path.basename(path)}: {exc}"
                    )
        else:
            # No assembly produced — create a body from the filename
            stem = os.path.splitext(os.path.basename(path))[0]
            body = Body(name=stem)
            for feat_info in part_result.features:
                feat = Feature(
                    name=feat_info["name"],
                    feature_type=feat_info["feature_type"],
                    origin=feat_info.get("origin", (0, 0, 0)),
                    direction=feat_info.get("direction", (0, 0, 1)),
                    radius=feat_info.get("radius", 0.0),
                )
                for gdt in part_result.gdt_callouts:
                    fcf = FeatureControlFrame(
                        name=f"gdt_{gdt['entity_id']}",
                        gdt_type=gdt["gdt_type"],
                        tolerance_value=gdt["tolerance_value"],
                        datum_refs=gdt.get("datum_refs", []),
                    )
                    feat.add_fcf(fcf)
                try:
                    body.add_feature(feat)
                except ValueError:
                    pass
            if body.features:
                if body.name in assy.bodies:
                    body.name = f"{body.name}_{len(assy.bodies)}"
                try:
                    assy.add_body(body)
                except ValueError:
                    combined.warnings.append(
                        f"Could not add body '{body.name}' from {os.path.basename(path)}"
                    )

    if assy.bodies:
        combined.assembly = assy

    return combined


def import_step_pmi(path: str) -> list[FeatureControlFrame]:
    """Import just the PMI (GD&T) callouts from a STEP file.

    Convenience function when you only need the tolerance annotations.

    Args:
        path: Path to the STEP file.

    Returns:
        List of FeatureControlFrame objects.
    """
    entities = _parse_step_file(path)
    gdt_list = _extract_gdt(entities)

    fcfs = []
    for gdt in gdt_list:
        fcf = FeatureControlFrame(
            name=f"gdt_{gdt['entity_id']}",
            gdt_type=gdt["gdt_type"],
            tolerance_value=gdt["tolerance_value"],
            datum_refs=gdt.get("datum_refs", []),
        )
        fcfs.append(fcf)

    return fcfs
