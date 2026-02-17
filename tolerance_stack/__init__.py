"""3D Tolerance Stack Analysis Tool.

Supports Worst-Case, RSS, and Monte Carlo analysis methods for:
- Linear tolerance stacks with 3D directional contributions
- 3D kinematic linkages with forward kinematics
- 3D rigid body assemblies with geometric features and mates
"""

from tolerance_stack.models import Contributor, ToleranceStack
from tolerance_stack.analysis import analyze_stack
from tolerance_stack.linkage import Joint, JointType, Link, Linkage
from tolerance_stack.linkage_analysis import analyze_linkage
from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.assembly_analysis import analyze_assembly

__all__ = [
    "Contributor", "ToleranceStack", "analyze_stack",
    "Joint", "JointType", "Link", "Linkage", "analyze_linkage",
    "Assembly", "Body", "Feature", "FeatureType",
    "Mate", "MateType", "Measurement", "MeasurementType",
    "analyze_assembly",
]
__version__ = "0.3.0"
