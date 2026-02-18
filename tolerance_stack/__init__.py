"""3D Tolerance Stack Analysis Tool.

Supports Worst-Case, RSS, and Monte Carlo analysis methods for:
- Linear tolerance stacks with 3D directional contributions
- 3D kinematic linkages with forward kinematics
- 3D rigid body assemblies with geometric features and mates

Additional capabilities:
- Full GD&T per ASME Y14.5 (position, profile, runout, MMC/LMC)
- Composite GD&T and datum feature simulation
- Datum reference frames
- Process capability metrics (Cp/Cpk/Pp/Ppk/PPM)
- Tolerance optimization, DOE (LHS, RSM, Full Factorial), and Sobol' sensitivity
- Multi-stage assembly process modeling
- STEP file import with PMI extraction
- Interactive 3D visualization (Plotly)
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
    CompositeFCF, DatumShiftResult, compute_datum_shift,
    composite_fcf_to_tolerance_parameters,
)
from tolerance_stack.statistics import (
    ProcessCapability, compute_process_capability,
    percent_contribution, geo_factor,
)
from tolerance_stack.optimizer import (
    critical_tolerance_identifier, optimize_tolerances,
    hlm_sensitivity, full_factorial_doe,
    latin_hypercube_doe, response_surface_doe, sobol_sensitivity,
    DOEFactor, RSMResult, SobolResult,
)
from tolerance_stack.assembly_process import (
    AssemblyProcess, AssemblyStation, Fixture, MoveOperation,
    compute_dof_status,
)
from tolerance_stack.reporting import (
    ReportConfig, generate_html_report, generate_text_report,
    generate_apqp_report, generate_pdf_report, save_report,
    save_pdf_report,
)
from tolerance_stack.step_import import import_step, import_step_pmi
from tolerance_stack.visualization import (
    visualize_assembly, visualize_linkage, visualize_sensitivity,
    add_mc_cloud, VisualizationConfig, PLOTLY_AVAILABLE,
)

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
    "CompositeFCF", "DatumShiftResult", "compute_datum_shift",
    "composite_fcf_to_tolerance_parameters",
    # Statistics
    "ProcessCapability", "compute_process_capability",
    "percent_contribution", "geo_factor",
    # Optimization & DOE
    "critical_tolerance_identifier", "optimize_tolerances",
    "hlm_sensitivity", "full_factorial_doe",
    "latin_hypercube_doe", "response_surface_doe", "sobol_sensitivity",
    "DOEFactor", "RSMResult", "SobolResult",
    # Assembly Process
    "AssemblyProcess", "AssemblyStation", "Fixture", "MoveOperation",
    "compute_dof_status",
    # Reporting
    "ReportConfig", "generate_html_report", "generate_text_report",
    "generate_apqp_report", "generate_pdf_report",
    "save_report", "save_pdf_report",
    # STEP Import
    "import_step", "import_step_pmi",
    # Visualization
    "visualize_assembly", "visualize_linkage", "visualize_sensitivity",
    "add_mc_cloud", "VisualizationConfig", "PLOTLY_AVAILABLE",
]
__version__ = "0.5.0"
