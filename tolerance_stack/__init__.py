"""3D Tolerance Stack Analysis Tool.

Supports Worst-Case, RSS, and Monte Carlo analysis methods
for tolerance stacks with full 3D directional contributions.
"""

from tolerance_stack.models import Contributor, ToleranceStack
from tolerance_stack.analysis import analyze_stack

__all__ = ["Contributor", "ToleranceStack", "analyze_stack"]
__version__ = "0.1.0"
