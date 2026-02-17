"""Built-in example assemblies for demonstration."""

from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.models import Distribution


def create_pin_in_hole_assembly() -> Assembly:
    """A pin inserted into a hole — classic GD&T tolerance stack.

    Two bodies:
        Block — has a hole (cylinder) and a top face (plane)
        Pin   — has a shaft (cylinder) and a head face (plane)

    The measurement is the gap between the top of the block and the
    bottom of the pin head, measured along Z.

    Block:
        - Height: 50mm along Z
        - Hole: cylinder at center, diameter 10.05mm ± 0.03 position tol
        - Top face: plane at Z=50

    Pin:
        - Shaft: cylinder diameter 10.00mm ± 0.02 position tol
        - Head face: plane at Z=shaft_length
        - Shaft length: 45mm
        - Head height: 8mm
    """
    # --- Block ---
    block = Body("Block", description="Block with a bore hole")
    block.add_feature(Feature(
        "bottom_face", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
    ))
    block.add_feature(Feature(
        "top_face", FeatureType.PLANE,
        origin=(0, 0, 50), direction=(0, 0, 1),
        position_tol=0.05,
    ))
    block.add_feature(Feature(
        "bore_axis", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=5.025,
        position_tol=0.03,
    ))

    # --- Pin ---
    pin = Body("Pin", description="Pin with shaft and head")
    pin.add_feature(Feature(
        "shaft_bottom", FeatureType.POINT,
        origin=(0, 0, 0), direction=(0, 0, -1),
    ))
    pin.add_feature(Feature(
        "shaft_axis", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=5.0,
        position_tol=0.02,
    ))
    pin.add_feature(Feature(
        "head_bottom", FeatureType.PLANE,
        origin=(0, 0, 45), direction=(0, 0, -1),
        position_tol=0.04,
    ))
    pin.add_feature(Feature(
        "head_top", FeatureType.PLANE,
        origin=(0, 0, 53), direction=(0, 0, 1),
        position_tol=0.04,
    ))

    # --- Assembly ---
    assy = Assembly(
        name="Pin-in-Hole Assembly",
        description="Gap between block top face and pin head bottom face",
    )

    # Block sits at origin
    assy.add_body(block, origin=(0, 0, 0))

    # Pin is inserted from above; shaft bottom at Z=0 (flush with block bottom)
    assy.add_body(pin, origin=(0, 0, 0))

    # Mate: coaxial bore and shaft
    assy.add_mate(Mate(
        "bore-shaft", "Block", "bore_axis", "Pin", "shaft_axis",
        mate_type=MateType.COAXIAL,
    ))

    # Measurement: gap = distance along Z from block top_face to pin head_bottom
    assy.set_measurement(Measurement(
        "gap", "Block", "top_face", "Pin", "head_bottom",
        measurement_type=MeasurementType.DISTANCE_ALONG,
        direction=(0, 0, -1),
    ))

    return assy


def create_stacked_plates_assembly() -> Assembly:
    """Three plates stacked along Z with face-to-face mates.

    Plate A (20mm thick) -> Plate B (15mm thick) -> Plate C (10mm thick)

    Each plate has top and bottom faces with position tolerances.
    Face mates have distance tolerances (representing gap or preload).

    Measurement: total stack height from bottom of A to top of C.
    """
    # --- Plate A ---
    plate_a = Body("PlateA", description="Bottom plate, 20mm thick")
    plate_a.add_feature(Feature(
        "bottom", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
    ))
    plate_a.add_feature(Feature(
        "top", FeatureType.PLANE,
        origin=(0, 0, 20), direction=(0, 0, 1),
        position_tol=0.06,
    ))

    # --- Plate B ---
    plate_b = Body("PlateB", description="Middle plate, 15mm thick")
    plate_b.add_feature(Feature(
        "bottom", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
        position_tol=0.04,
    ))
    plate_b.add_feature(Feature(
        "top", FeatureType.PLANE,
        origin=(0, 0, 15), direction=(0, 0, 1),
        position_tol=0.04,
    ))

    # --- Plate C ---
    plate_c = Body("PlateC", description="Top plate, 10mm thick")
    plate_c.add_feature(Feature(
        "bottom", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, -1),
        position_tol=0.03,
    ))
    plate_c.add_feature(Feature(
        "top", FeatureType.PLANE,
        origin=(0, 0, 10), direction=(0, 0, 1),
        position_tol=0.03,
    ))

    # --- Assembly ---
    assy = Assembly(
        name="Stacked Plates",
        description="Three plates stacked along Z, measuring total height",
    )

    assy.add_body(plate_a, origin=(0, 0, 0))
    assy.add_body(plate_b, origin=(0, 0, 20))
    assy.add_body(plate_c, origin=(0, 0, 35))

    # Face mates with small gap tolerances
    assy.add_mate(Mate(
        "A-B mate", "PlateA", "top", "PlateB", "bottom",
        mate_type=MateType.COPLANAR,
        distance=0.0, distance_tol=0.02,
    ))
    assy.add_mate(Mate(
        "B-C mate", "PlateB", "top", "PlateC", "bottom",
        mate_type=MateType.COPLANAR,
        distance=0.0, distance_tol=0.02,
    ))

    # Measurement: total height along Z
    assy.set_measurement(Measurement(
        "total_height", "PlateA", "bottom", "PlateC", "top",
        measurement_type=MeasurementType.DISTANCE_ALONG,
        direction=(0, 0, 1),
    ))

    return assy


def create_bracket_assembly() -> Assembly:
    """An L-bracket bolted to a base with a hole offset from the bolt.

    Base has a mounting surface and bolt hole.
    Bracket has a bolt hole and a target hole offset in X and Z.

    Measurement: distance from base datum to bracket target hole,
    along X. Demonstrates how angular and positional tolerances on
    the bracket's bend interact.

    Base:
        - Top surface at Z=0
        - Bolt hole at (50, 0, 0)

    Bracket:
        - Bolt hole at (0, 0, 0) in local frame
        - Bend at 90 degrees
        - Target hole at (0, 0, 80) in local frame
        - Tolerance on target hole position and bracket angle
    """
    base = Body("Base", description="Flat base plate")
    base.add_feature(Feature(
        "surface", FeatureType.PLANE,
        origin=(0, 0, 0), direction=(0, 0, 1),
    ))
    base.add_feature(Feature(
        "bolt_hole", FeatureType.CYLINDER,
        origin=(50, 0, 0), direction=(0, 0, 1),
        radius=4.0,
        position_tol=0.05,
    ))
    base.add_feature(Feature(
        "datum_edge", FeatureType.POINT,
        origin=(0, 0, 0),
    ))

    bracket = Body("Bracket", description="L-bracket with target hole")
    bracket.add_feature(Feature(
        "bolt_hole", FeatureType.CYLINDER,
        origin=(0, 0, 0), direction=(0, 0, 1),
        radius=4.1,
        position_tol=0.04,
    ))
    bracket.add_feature(Feature(
        "target_hole", FeatureType.CYLINDER,
        origin=(0, 0, 80), direction=(1, 0, 0),
        radius=3.0,
        position_tol=0.10,
        orientation_tol=0.5,
    ))

    assy = Assembly(
        name="Bracket Assembly",
        description="L-bracket bolted to base, measuring target hole position",
    )

    assy.add_body(base, origin=(0, 0, 0))
    assy.add_body(bracket, origin=(50, 0, 0))

    assy.add_mate(Mate(
        "bolt", "Base", "bolt_hole", "Bracket", "bolt_hole",
        mate_type=MateType.COAXIAL,
    ))

    # Measure distance from base datum to bracket target hole along X
    assy.set_measurement(Measurement(
        "target_position_x", "Base", "datum_edge", "Bracket", "target_hole",
        measurement_type=MeasurementType.DISTANCE_ALONG,
        direction=(1, 0, 0),
    ))

    return assy
