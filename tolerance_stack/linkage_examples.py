"""Built-in example linkages for demonstration."""

from tolerance_stack.linkage import Joint, JointType, Link, Linkage
from tolerance_stack.models import Distribution


def create_planar_two_bar() -> Linkage:
    """A simple planar two-bar linkage in the XY plane.

    Two rigid links connected by revolute joints about Z:

        Ground -> [J0:fixed] -> Link1 (100mm @ 30deg) -> [J1:revolute_z 45deg]
               -> Link2 (80mm) -> [J2:fixed]

    Tolerances on both link lengths and the joint angle.
    This is the classic mechanism tolerance problem.
    """
    linkage = Linkage(
        name="Planar Two-Bar Linkage",
        description="Two rigid links with a revolute joint, in the XY plane",
    )

    # Base joint (fixed to ground at origin)
    linkage.add_joint(Joint("Base", JointType.REVOLUTE_Z, nominal=30.0))

    # First link — extends along local X
    linkage.add_link(Link(
        "Link1", length=100.0, plus_tol=0.10, minus_tol=0.10,
        direction=(1, 0, 0),
    ))

    # Revolute joint at elbow — rotates about Z
    linkage.add_joint(Joint(
        "Elbow", JointType.REVOLUTE_Z, nominal=45.0,
        plus_tol=0.5, minus_tol=0.5,
    ))

    # Second link
    linkage.add_link(Link(
        "Link2", length=80.0, plus_tol=0.08, minus_tol=0.08,
        direction=(1, 0, 0),
    ))

    # End joint (tip)
    linkage.add_joint(Joint("Tip", JointType.FIXED))

    return linkage


def create_spatial_robot_arm() -> Linkage:
    """A 3D robot arm with shoulder, elbow, and wrist joints.

    A 3-joint spatial linkage:
        Base -> [shoulder revolute_Y 45deg] -> upper arm (200mm, +Z)
             -> [elbow revolute_Y -30deg] -> forearm (150mm, +Z)
             -> [wrist revolute_X 0deg] -> hand (50mm, +Z)
             -> Tip

    Demonstrates full 3D kinematics with tolerances on all links and joints.
    """
    linkage = Linkage(
        name="3-DOF Spatial Robot Arm",
        description="Shoulder-elbow-wrist arm with upward links",
    )

    # Shoulder rotates about Y
    linkage.add_joint(Joint(
        "Shoulder", JointType.REVOLUTE_Y, nominal=45.0,
        plus_tol=0.3, minus_tol=0.3,
    ))

    # Upper arm extends along Z
    linkage.add_link(Link(
        "Upper Arm", length=200.0, plus_tol=0.15, minus_tol=0.15,
        direction=(0, 0, 1),
    ))

    # Elbow rotates about Y
    linkage.add_joint(Joint(
        "Elbow", JointType.REVOLUTE_Y, nominal=-30.0,
        plus_tol=0.4, minus_tol=0.4,
    ))

    # Forearm extends along Z
    linkage.add_link(Link(
        "Forearm", length=150.0, plus_tol=0.12, minus_tol=0.12,
        direction=(0, 0, 1),
    ))

    # Wrist rotates about X
    linkage.add_joint(Joint(
        "Wrist", JointType.REVOLUTE_X, nominal=0.0,
        plus_tol=0.2, minus_tol=0.2,
    ))

    # Hand
    linkage.add_link(Link(
        "Hand", length=50.0, plus_tol=0.05, minus_tol=0.05,
        direction=(0, 0, 1),
    ))

    # Tip
    linkage.add_joint(Joint("Tip", JointType.FIXED))

    return linkage


def create_four_bar_mechanism() -> Linkage:
    """A planar four-bar linkage approximated as an open chain.

    For tolerance analysis of the output link position, we model
    the kinematic path from ground to coupler point:

        Ground -> [crank angle] -> Crank (60mm)
               -> [coupler angle] -> Coupler (120mm)
               -> [rocker angle] -> (end)

    All revolute joints about Z axis (planar mechanism).
    """
    linkage = Linkage(
        name="Four-Bar Mechanism Path",
        description="Open-chain path through a four-bar linkage coupler point",
    )

    # Crank pivot
    linkage.add_joint(Joint(
        "Crank Pivot", JointType.REVOLUTE_Z, nominal=60.0,
        plus_tol=0.3, minus_tol=0.3,
    ))

    # Crank link
    linkage.add_link(Link(
        "Crank", length=60.0, plus_tol=0.05, minus_tol=0.05,
        direction=(1, 0, 0),
    ))

    # Coupler joint
    linkage.add_joint(Joint(
        "Coupler Joint", JointType.REVOLUTE_Z, nominal=-20.0,
        plus_tol=0.4, minus_tol=0.4,
    ))

    # Coupler link
    linkage.add_link(Link(
        "Coupler", length=120.0, plus_tol=0.10, minus_tol=0.10,
        direction=(1, 0, 0),
    ))

    # Output joint
    linkage.add_joint(Joint(
        "Output Joint", JointType.REVOLUTE_Z, nominal=15.0,
        plus_tol=0.3, minus_tol=0.3,
    ))

    # Rocker link
    linkage.add_link(Link(
        "Rocker", length=90.0, plus_tol=0.08, minus_tol=0.08,
        direction=(1, 0, 0),
    ))

    # End
    linkage.add_joint(Joint("End", JointType.FIXED))

    return linkage
