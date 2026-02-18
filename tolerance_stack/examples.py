"""Built-in example tolerance stacks for demonstration."""

from tolerance_stack.models import Contributor, ContributorType, Distribution, ToleranceStack


def create_shaft_housing_example() -> ToleranceStack:
    """Classic 1D shaft-in-housing tolerance stack along the X axis.

    Assembly: A shaft sits inside a housing. We want to know the gap
    between the end of the shaft and the inner wall of the housing.

    Dimension loop (all along X axis):
        +Housing length
        -Shaft length
        -Washer thickness
        -Retaining ring groove width
        = Gap
    """
    stack = ToleranceStack(
        name="Shaft-Housing Assembly",
        description="Gap between shaft end and housing inner wall",
        closure_direction=(1, 0, 0),
    )

    stack.add(Contributor(
        name="Housing bore depth",
        nominal=50.000,
        plus_tol=0.100,
        minus_tol=0.100,
        direction=(1, 0, 0),
        sign=+1,
        distribution=Distribution.NORMAL,
    ))

    stack.add(Contributor(
        name="Shaft length",
        nominal=45.000,
        plus_tol=0.050,
        minus_tol=0.050,
        direction=(1, 0, 0),
        sign=-1,
        distribution=Distribution.NORMAL,
    ))

    stack.add(Contributor(
        name="Washer thickness",
        nominal=2.000,
        plus_tol=0.025,
        minus_tol=0.025,
        direction=(1, 0, 0),
        sign=-1,
        distribution=Distribution.NORMAL,
    ))

    stack.add(Contributor(
        name="Retaining ring width",
        nominal=1.500,
        plus_tol=0.030,
        minus_tol=0.030,
        direction=(1, 0, 0),
        sign=-1,
        distribution=Distribution.NORMAL,
    ))

    stack.add(Contributor(
        name="Snap ring groove depth",
        nominal=0.800,
        plus_tol=0.020,
        minus_tol=0.020,
        direction=(1, 0, 0),
        sign=-1,
        distribution=Distribution.NORMAL,
    ))

    return stack


def create_multiaxis_example() -> ToleranceStack:
    """Multi-axis 3D tolerance stack.

    A bracket assembly where components contribute dimensions along
    different axes. The gap of interest is along the X axis, but some
    features are oriented in Y and Z (e.g., angled mounting surfaces).

    This demonstrates 3D direction vectors and angular contributions.
    """
    import math

    stack = ToleranceStack(
        name="Angled Bracket Assembly",
        description="Clearance gap at bracket tip, closure along X",
        closure_direction=(1, 0, 0),
    )

    # Base plate width along X
    stack.add(Contributor(
        name="Base plate width",
        nominal=100.000,
        plus_tol=0.150,
        minus_tol=0.150,
        direction=(1, 0, 0),
        sign=+1,
    ))

    # Angled strut (45 degrees in XZ plane) — projects cos(45) onto X
    angle = math.radians(45)
    stack.add(Contributor(
        name="Angled strut length",
        nominal=30.000,
        plus_tol=0.100,
        minus_tol=0.100,
        direction=(math.cos(angle), 0, math.sin(angle)),
        sign=-1,
    ))

    # Vertical spacer — only Z component, zero X projection
    stack.add(Contributor(
        name="Vertical spacer",
        nominal=15.000,
        plus_tol=0.050,
        minus_tol=0.050,
        direction=(0, 0, 1),
        sign=-1,
    ))

    # Top cap width along X
    stack.add(Contributor(
        name="Top cap width",
        nominal=70.000,
        plus_tol=0.120,
        minus_tol=0.120,
        direction=(1, 0, 0),
        sign=-1,
    ))

    # Angular mounting surface — 10 deg tilt introduces an angular tolerance
    tilt = math.radians(10)
    stack.add(Contributor(
        name="Mount surface angle",
        nominal=5.000,
        plus_tol=0.5,  # 0.5 degrees
        minus_tol=0.5,
        direction=(math.cos(tilt), math.sin(tilt), 0),
        sign=+1,
        contributor_type=ContributorType.ANGULAR,
    ))

    return stack
