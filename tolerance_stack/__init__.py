"""3D Tolerance Stack Analysis Tool.

Supports Worst-Case, RSS, and Monte Carlo analysis methods
for tolerance stacks with full 3D directional contributions,
and 3D linkage tolerance analysis with forward kinematics.
"""

from tolerance_stack.models import Contributor, ToleranceStack
from tolerance_stack.analysis import analyze_stack
from tolerance_stack.linkage import Joint, JointType, Link, Linkage
from tolerance_stack.linkage_analysis import analyze_linkage

__all__ = [
    "Contributor", "ToleranceStack", "analyze_stack",
    "Joint", "JointType", "Link", "Linkage", "analyze_linkage",
]
__version__ = "0.2.0"
