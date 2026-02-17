"""3D Tolerance Stack Analysis Tool.

Supports Worst-Case, RSS, and Monte Carlo analysis methods for:
- Linear tolerance stacks with 3D directional contributions
- 3D kinematic linkages with forward kinematics
- 3D rigid body assemblies with geometric features and mates

Additional capabilities:
- Full GD&T per ASME Y14.5 (position, profile, runout, MMC/LMC)
- Datum reference frames
- Process capability metrics (Cp/Cpk/Pp/Ppk/PPM)
- Tolerance optimization and DOE
- Multi-stage assembly process modeling
- STEP file import with PMI extraction
- HTML/PDF report generation (APQP compliant)
"""

from tolerance_stack.models import Contributor, ToleranceStack, Distribution
from tolerance_stack.analysis import analyze_stack
from tolerance_stack.linkage import Joint, JointType, Link, Linkage
from tolerance_stack.linkage_analysis import analyze_linkage
from tolerance_stack.assembly import (
    Assembly, Body, Feature, FeatureType, Mate, MateType,
    Measurement, MeasurementType,
)
from tolerance_stack.assembly_analysis import analyze_assembly
from tolerance_stack.gdt import (
    GDTType, MaterialCondition, FeatureControlFrame,
    DatumReferenceFrame, DatumFeature,
)
from tolerance_stack.statistics import (
    ProcessCapability, compute_process_capability,
    percent_contribution, geo_factor,
)
from tolerance_stack.optimizer import (
    critical_tolerance_identifier, optimize_tolerances,
    hlm_sensitivity, full_factorial_doe,
    DOEFactor,
)
from tolerance_stack.assembly_process import (
    AssemblyProcess, AssemblyStation, Fixture, MoveOperation,
    compute_dof_status,
)
from tolerance_stack.reporting import (
    ReportConfig, generate_html_report, generate_text_report,
    generate_apqp_report, save_report,
)
from tolerance_stack.step_import import import_step, import_step_pmi

__all__ = [
    # Core models
    "Contributor", "ToleranceStack", "Distribution", "analyze_stack",
    # Linkage
    "Joint", "JointType", "Link", "Linkage", "analyze_linkage",
    # Assembly
    "Assembly", "Body", "Feature", "FeatureType",
    "Mate", "MateType", "Measurement", "MeasurementType",
    "analyze_assembly",
    # GD&T
    "GDTType", "MaterialCondition", "FeatureControlFrame",
    "DatumReferenceFrame", "DatumFeature",
    # Statistics
    "ProcessCapability", "compute_process_capability",
    "percent_contribution", "geo_factor",
    # Optimization
    "critical_tolerance_identifier", "optimize_tolerances",
    "hlm_sensitivity", "full_factorial_doe", "DOEFactor",
    # Assembly Process
    "AssemblyProcess", "AssemblyStation", "Fixture", "MoveOperation",
    "compute_dof_status",
    # Reporting
    "ReportConfig", "generate_html_report", "generate_text_report",
    "generate_apqp_report", "save_report",
    # STEP Import
    "import_step", "import_step_pmi",
]
__version__ = "0.4.0"
