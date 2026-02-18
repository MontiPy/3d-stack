# HOWTO: 3D Tolerance Stack Analysis Tool

## Is This Application Ready for Use?

**Yes.** The application (v0.6.0) passes all 267 tests and provides a fully
functional toolkit for tolerance analysis at three levels of complexity:

| Level | What it does | Status |
|-------|-------------|--------|
| 1D Tolerance Stacks | Classic dimension-loop gap analysis | Production-ready |
| 3D Kinematic Linkages | Joint/link chains with forward kinematics | Production-ready |
| 3D Rigid Body Assemblies | Multi-body assemblies with mates and GD&T | Production-ready |
| Reporting (HTML/PDF/APQP) | Professional tolerance analysis reports | Production-ready |
| STEP Import | CAD file geometry and PMI extraction | Requires `pythonocc-core` |
| 3D Visualization | Interactive Plotly plots | Requires `plotly` |
| Streamlit GUI | Browser-based interactive UI | Requires `streamlit` |

### Required dependencies

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

### Optional dependencies

```bash
pip install plotly          # Interactive 3D visualization
pip install streamlit       # GUI (future)
pip install pythonocc-core  # STEP file import
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd 3d-stack

# Install in development mode
pip install -e .

# Or install just the dependencies
pip install numpy matplotlib scipy

# Verify installation
python -c "from tolerance_stack import analyze_stack; print('OK')"
```

After installation the `tolstack` CLI command becomes available.

---

## User Workflow Overview

The typical workflow follows these steps:

```
1. Define your problem    (stack / linkage / assembly)
2. Run analysis           (worst-case, RSS, Monte Carlo)
3. Interpret results      (gap, tolerance range, sensitivity)
4. Optimize if needed     (DOE, genetic algorithm, contingency)
5. Generate a report      (HTML, PDF, or APQP)
```

---

## Workflow 1: One-Dimensional Tolerance Stack

This is the classic "dimension loop" analysis. Use it when parts stack
linearly along a single measurement direction.

### Step 1 — Create a stack definition

**Option A: Generate an example file**

```bash
tolstack example shaft -o shaft_stack.json
tolstack example multiaxis -o multiaxis_stack.json
```

**Option B: Use the interactive builder**

```bash
tolstack interactive
```

You will be prompted for each contributor's name, nominal value, tolerances,
direction vector, sign (+1 or -1), distribution, and type.

**Option C: Write JSON directly**

```json
{
  "name": "Shaft-Housing Gap",
  "description": "Clearance between shaft end and housing shoulder",
  "closure_direction": [1, 0, 0],
  "contributors": [
    {
      "name": "Housing length",
      "nominal": 50.0,
      "plus_tol": 0.10,
      "minus_tol": 0.10,
      "direction": [1, 0, 0],
      "sign": 1,
      "distribution": "normal",
      "contributor_type": "linear",
      "sigma": 3.0
    },
    {
      "name": "Shaft length",
      "nominal": 48.0,
      "plus_tol": 0.05,
      "minus_tol": 0.05,
      "direction": [1, 0, 0],
      "sign": -1,
      "distribution": "normal",
      "contributor_type": "linear",
      "sigma": 3.0
    }
  ]
}
```

**Option D: Python API**

```python
from tolerance_stack import Contributor, ToleranceStack, analyze_stack

stack = ToleranceStack(name="Shaft-Housing Gap", closure_direction=(1, 0, 0))
stack.add(Contributor("Housing length", nominal=50.0, plus_tol=0.10, minus_tol=0.10, sign=+1))
stack.add(Contributor("Shaft length",   nominal=48.0, plus_tol=0.05, minus_tol=0.05, sign=-1))
```

### Key concepts

- **`sign=+1`** means the dimension adds to the gap.
- **`sign=-1`** means the dimension subtracts from the gap.
- **`direction`** is a 3D vector. Its dot product with `closure_direction`
  determines how much the contributor affects the gap. A contributor
  perpendicular to the closure direction has zero effect.
- **`distribution`** can be: `normal`, `uniform`, `triangular`,
  `weibull_right`, `weibull_left`, `lognormal`, `rayleigh`, `bimodal`,
  or `empirical`.
- **`contributor_type`** can be: `linear`, `angular`, or `geometric`.

### Step 2 — Run analysis

```bash
# All three methods (worst-case, RSS, Monte Carlo)
tolstack analyze shaft_stack.json

# Specific methods only
tolstack analyze shaft_stack.json -m wc,rss

# Customize parameters
tolstack analyze shaft_stack.json --sigma 6 --mc-samples 500000 --seed 42

# With visualization plots
tolstack analyze shaft_stack.json --plot

# Save plots to files
tolstack analyze shaft_stack.json --save-plots output_plots

# With spec limits for Monte Carlo histogram
tolstack analyze shaft_stack.json --spec-limits 1.5,2.5
```

Or via Python:

```python
results = analyze_stack(stack, methods=["wc", "rss", "mc"], sigma=3.0,
                        mc_samples=100_000, mc_seed=42)

for method, result in results.items():
    print(result.summary())
```

### Step 3 — Read the results

Each analysis method returns:

| Field | Meaning |
|-------|---------|
| `nominal_gap` | Gap at nominal dimensions |
| `gap_max` | Largest possible/likely gap |
| `gap_min` | Smallest possible/likely gap |
| `plus_tolerance` | `gap_max - nominal_gap` |
| `minus_tolerance` | `nominal_gap - gap_min` |
| `sensitivity` | List of (contributor, factor) showing each part's influence |
| `mc_mean`, `mc_std` | Monte Carlo statistics (MC method only) |
| `percent_yield` | Estimated yield percentage (RSS/MC) |

### When to use each method

| Method | Best for | Assumption |
|--------|----------|------------|
| **Worst-Case (WC)** | Safety-critical assemblies that must never fail | Every dimension at its extreme simultaneously |
| **RSS** | Cost-sensitive manufacturing, typical production | Dimensions are independent and normally distributed |
| **Monte Carlo (MC)** | Complex distributions, high-confidence predictions | None — uses actual distribution shapes |

---

## Workflow 2: 3D Kinematic Linkage Analysis

Use this when your tolerance problem involves a chain of rigid links
connected by joints (robot arms, mechanisms, linkage assemblies).

### Step 1 — Define a linkage

**Generate an example:**

```bash
tolstack linkage-example two-bar -o two_bar.json
tolstack linkage-example robot-arm -o robot_arm.json
tolstack linkage-example four-bar -o four_bar.json
```

**Or use the interactive builder:**

```bash
tolstack linkage-interactive
```

**Or build in Python:**

```python
from tolerance_stack import Joint, JointType, Link, Linkage, analyze_linkage

linkage = Linkage(name="Two-Bar Mechanism")

# Chain structure: Joint -> Link -> Joint -> Link -> Joint
linkage.add_joint(Joint("Base",  JointType.REVOLUTE_Z, nominal=30.0,
                        plus_tol=0.5, minus_tol=0.5))
linkage.add_link(Link("Arm1", length=100.0, plus_tol=0.1, minus_tol=0.1,
                       direction=(1, 0, 0)))
linkage.add_joint(Joint("Elbow", JointType.REVOLUTE_Z, nominal=45.0,
                        plus_tol=0.5, minus_tol=0.5))
linkage.add_link(Link("Arm2", length=80.0, plus_tol=0.08, minus_tol=0.08,
                       direction=(1, 0, 0)))
linkage.add_joint(Joint("Tip",   JointType.FIXED))
```

### Joint types

| Type | Axis | Nominal value |
|------|------|---------------|
| `FIXED` | None | 0 (no motion) |
| `REVOLUTE_X/Y/Z` | Rotation about X/Y/Z | Angle in degrees |
| `PRISMATIC_X/Y/Z` | Translation along X/Y/Z | Distance |
| `SPHERICAL` | All rotation axes | Tuple of (rz, ry, rx) in degrees |

### Chain structure

A linkage alternates joints and links:

```
J0 -> L0 -> J1 -> L1 -> ... -> Jn
```

There is always one more joint than links. The first and last joints are
typically `FIXED`.

### Step 2 — Analyze

```bash
tolstack linkage-analyze two_bar.json
tolstack linkage-analyze robot_arm.json --plot
tolstack linkage-analyze robot_arm.json -m rss,mc --mc-samples 200000
```

Or in Python:

```python
results = analyze_linkage(linkage, methods=["wc", "rss", "mc"], mc_seed=42)

for method, result in results.items():
    print(result.summary())
```

### Step 3 — Read the results

Linkage analysis results are **3D** (X, Y, Z per axis):

| Field | Meaning |
|-------|---------|
| `nominal_position` | End-effector position at nominal values [X, Y, Z] |
| `position_max/min` | Bounding box of the tolerance zone per axis |
| `plus_tolerance/minus_tolerance` | Per-axis tolerance from nominal |
| `radial_tolerance` | Spherical tolerance radius |
| `sensitivity` | List of (parameter, [dX, dY, dZ]) influence vectors |
| `jacobian` | 3 x N matrix relating parameter changes to position changes |
| `mc_samples` | N x 3 array of sampled end-effector positions (MC only) |
| `mc_cov` | 3 x 3 covariance matrix (MC only) |

---

## Workflow 3: 3D Rigid Body Assembly Analysis

Use this for full 3D assemblies where multiple bodies are positioned in
space, connected by mates, and you measure a gap or alignment between
features.

### Step 1 — Define an assembly

**Generate an example:**

```bash
tolstack assembly-example pin-in-hole -o pin_hole.json
tolstack assembly-example stacked-plates -o plates.json
tolstack assembly-example bracket -o bracket.json
```

**Or build in Python:**

```python
from tolerance_stack import (
    Assembly, Body, Feature, FeatureType,
    Mate, MateType, Measurement, MeasurementType,
)

# Create bodies with features
housing = Body("Housing")
housing.add_feature(Feature("bore", FeatureType.CYLINDER,
                            origin=(0, 0, 0), direction=(0, 0, 1),
                            radius=5.0, position_tol=0.05))

pin = Body("Pin")
pin.add_feature(Feature("shaft", FeatureType.CYLINDER,
                        origin=(0, 0, 0), direction=(0, 0, 1),
                        radius=4.9, position_tol=0.03))

# Build the assembly
assy = Assembly("Pin-in-Hole")
assy.add_body(housing, origin=(0, 0, 0))
assy.add_body(pin, origin=(0, 0, 10))

# Add mates (constraints between features)
assy.add_mate(Mate("coaxial_fit", "Housing", "bore",
                   "Pin", "shaft", MateType.COAXIAL))

# Define what to measure
assy.set_measurement(Measurement("radial_clearance",
                                 "Housing", "bore",
                                 "Pin", "shaft",
                                 MeasurementType.DISTANCE))
```

### Feature types

| Type | Description | Key attributes |
|------|-------------|----------------|
| `POINT` | A point in space | `origin` |
| `PLANE` | A flat surface | `origin`, `direction` (normal) |
| `AXIS` | A centerline | `origin`, `direction` |
| `CYLINDER` | A cylindrical surface | `origin`, `direction`, `radius` |
| `CIRCLE` | A circular edge | `origin`, `direction`, `radius` |

### Mate types

| Type | Constrains |
|------|-----------|
| `COINCIDENT` | Two points at the same location |
| `COAXIAL` | Two axes aligned |
| `COPLANAR` | Two planes sharing a surface |
| `AT_DISTANCE` | Two features at a specified distance |
| `PARALLEL` | Two directions aligned |
| `CONCENTRIC` | Two circles/cylinders sharing a center |

### Measurement types

| Type | What it measures |
|------|-----------------|
| `DISTANCE` | Straight-line distance between features |
| `DISTANCE_ALONG` | Distance projected onto a direction |
| `ANGLE` | Angle between feature directions |
| `POINT_TO_PLANE` | Perpendicular distance from point to plane |
| `POINT_TO_LINE` | Perpendicular distance from point to line |
| `PLANE_TO_PLANE_ANGLE` | Angle between two planes |
| `LINE_TO_PLANE_ANGLE` | Angle between a line and a plane |
| `GAP` | Clearance (positive = no contact) |
| `FLUSH` | Flush condition between surfaces |
| `INTERFERENCE` | Overlap (negative clearance) |

### Feature tolerances

Each feature can carry position, orientation, and form tolerances:

```python
Feature("bore", FeatureType.CYLINDER,
        origin=(0, 0, 0),
        direction=(0, 0, 1),
        radius=5.0,
        position_tol=0.05,      # Position tolerance (mm)
        orientation_tol=0.01,    # Orientation tolerance (rad)
        form_tol=0.02,           # Form tolerance (mm)
        size_nominal=10.0,       # Size dimension
        size_plus_tol=0.1,       # Size + tolerance
        size_minus_tol=0.1,      # Size - tolerance
        distribution=Distribution.NORMAL,
        sigma=3.0)
```

### Step 2 — Analyze

```bash
tolstack assembly-analyze pin_hole.json
tolstack assembly-analyze pin_hole.json -m wc,mc --mc-samples 200000
```

Or in Python:

```python
from tolerance_stack import analyze_assembly

results = analyze_assembly(assy, methods=["wc", "rss", "mc"],
                           sigma=3.0, mc_samples=100_000, mc_seed=42)
for method, result in results.items():
    print(result.summary())
```

---

## Adding GD&T (Geometric Dimensioning and Tolerancing)

The tool supports full ASME Y14.5 / ISO 1101 GD&T through Feature Control
Frames (FCFs) attached to features.

### GD&T tolerance types

| Category | Types |
|----------|-------|
| Form | Flatness, Straightness, Circularity, Cylindricity |
| Orientation | Perpendicularity, Angularity, Parallelism |
| Location | Position, Concentricity, Symmetry |
| Profile | Profile of a Surface, Profile of a Line |
| Runout | Circular Runout, Total Runout |

### Adding GD&T to a feature

```python
from tolerance_stack import (
    Feature, FeatureType, FeatureControlFrame, GDTType,
    MaterialCondition, DatumFeature, DatumReferenceFrame, DatumPrecedence,
)

# Create a feature with a GD&T position callout
bore = Feature("bore", FeatureType.CYLINDER,
               origin=(0, 0, 0), direction=(0, 0, 1), radius=5.0)
bore.add_fcf(FeatureControlFrame(
    name="bore_position",
    gdt_type=GDTType.POSITION,
    tolerance_value=0.05,
    material_condition=MaterialCondition.MMC,
    datum_refs=["A", "B"],
))

# Set up a datum reference frame
drf = DatumReferenceFrame(
    primary=DatumFeature("A", "Housing", "bottom_face", DatumPrecedence.PRIMARY),
    secondary=DatumFeature("B", "Housing", "side_face", DatumPrecedence.SECONDARY),
)
```

### Material condition modifiers

| Modifier | Symbol | Effect |
|----------|--------|--------|
| `NONE` (RFS) | — | Tolerance applies regardless of feature size |
| `MMC` | (M) | Bonus tolerance when feature departs from MMC |
| `LMC` | (L) | Bonus tolerance when feature departs from LMC |

### Datum shift

When a datum feature is applied at MMC or LMC, the datum can shift,
providing additional tolerance:

```python
from tolerance_stack import compute_datum_shift

result = compute_datum_shift(
    datum_size_nominal=10.0,
    datum_size_tol=0.1,
    material_condition=MaterialCondition.MMC,
    actual_size=9.95,
)
print(f"Datum shift: {result.shift}")
```

---

## Process Capability Analysis

After running a Monte Carlo simulation, you can compute process capability
metrics:

```python
from tolerance_stack import compute_process_capability

mc_result = results["mc"]
capability = compute_process_capability(
    mc_result.mc_samples,
    lsl=1.5,   # Lower spec limit
    usl=2.5,   # Upper spec limit
)

print(f"Cp  = {capability.cp:.3f}")
print(f"Cpk = {capability.cpk:.3f}")
print(f"Pp  = {capability.pp:.3f}")
print(f"Ppk = {capability.ppk:.3f}")
print(f"PPM = {capability.ppm:.1f}")
```

---

## Tolerance Optimization

### Identify critical tolerances

```python
from tolerance_stack import critical_tolerance_identifier

critical = critical_tolerance_identifier(stack, results)
# Returns contributors ranked by their impact on the total tolerance
```

### Optimize tolerances to meet a target

```python
from tolerance_stack import optimize_tolerances

optimized = optimize_tolerances(stack, target_tolerance=0.5)
# Returns recommended tolerance values for each contributor
```

### Design of Experiments (DOE)

```python
from tolerance_stack import (
    DOEFactor, full_factorial_doe, latin_hypercube_doe,
    response_surface_doe, sobol_sensitivity, hlm_sensitivity,
)

# Define factors
factors = [
    DOEFactor("Part A tol", low=0.01, high=0.2),
    DOEFactor("Part B tol", low=0.01, high=0.1),
]

# Full factorial
ff_results = full_factorial_doe(stack, factors, levels=3)

# Latin Hypercube Sampling
lhs_results = latin_hypercube_doe(stack, factors, n_samples=50)

# Response Surface Method
rsm_results = response_surface_doe(stack, factors)

# Sobol' sensitivity indices
sobol = sobol_sensitivity(stack, factors, n_samples=1024)
print(f"First-order indices: {sobol.s1}")
print(f"Total-order indices: {sobol.st}")
```

### Genetic Algorithm Optimization

```python
from tolerance_stack import ga_optimize_tolerances, GAConfig

config = GAConfig(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
)
ga_result = ga_optimize_tolerances(stack, target_tolerance=0.5, config=config)
print(f"Best tolerances: {ga_result.best_tolerances}")
print(f"Best cost: {ga_result.best_cost}")
```

### Contingency Analysis

```python
from tolerance_stack import contingency_analysis

contingency = contingency_analysis(stack, results, target_yield=99.73)
for item in contingency.items:
    print(f"{item.name}: current={item.current_tol:.4f}, "
          f"recommended={item.recommended_tol:.4f}")
```

---

## Assembly Process Modeling

Model multi-stage assembly with fixtures and operations:

```python
from tolerance_stack import (
    AssemblyProcess, AssemblyStation, Fixture, MoveOperation,
    Feature, FeatureType, compute_dof_status,
)

# Define a fixture
fixture = Fixture("Base Fixture", position_tol=0.01, repeatability=0.005)
fixture.add_feature(Feature("loc_pin", FeatureType.CYLINDER,
                            origin=(0, 0, 0), direction=(0, 0, 1), radius=3.0))

# Define a station
station = AssemblyStation("Station 1", description="Place housing on fixture")
station.add_fixture(fixture)
station.bodies_added.append("Housing")

# Build the process
process = AssemblyProcess("Widget Assembly", assembly=assy)
process.add_station(station)

# Check degrees of freedom
dof_status = compute_dof_status(assy)
for body_name, status in dof_status.items():
    print(f"{body_name}: {status.summary()}")
```

---

## Report Generation

### HTML report

```python
from tolerance_stack import ReportConfig, generate_html_report, save_report

config = ReportConfig(
    title="Shaft-Housing Tolerance Analysis",
    project="Widget Product Line",
    author="J. Engineer",
    revision="B",
    include_sensitivity=True,
    include_histograms=True,
    include_capability=True,
)

html = generate_html_report(config, results)
save_report(html, "analysis_report.html")
```

### PDF report

```python
from tolerance_stack import generate_pdf_report, save_pdf_report

pdf_bytes = generate_pdf_report(config, results)
save_pdf_report(pdf_bytes, "analysis_report.pdf")
```

### APQP-compliant report

```python
from tolerance_stack import generate_apqp_report, save_report

apqp_html = generate_apqp_report(config, results,
                                  spec_limits=(1.5, 2.5))
save_report(apqp_html, "apqp_report.html")
```

### Plain text report

```python
from tolerance_stack import generate_text_report

text = generate_text_report(config, results)
print(text)
```

---

## STEP File Import

Import geometry and GD&T from STEP files (requires `pythonocc-core`):

```python
from tolerance_stack import import_step, import_step_pmi

# Full import: geometry + GD&T -> Assembly
result = import_step("part.step", assembly_name="Imported Assembly")
print(result.summary())

if result.assembly:
    # Analyze the imported assembly directly
    from tolerance_stack import analyze_assembly
    analysis = analyze_assembly(result.assembly)

# Extract only GD&T/PMI callouts
fcfs = import_step_pmi("part.step")
for fcf in fcfs:
    print(f"{fcf.name}: {fcf.gdt_type.value} = {fcf.tolerance_value}")
```

---

## 3D Visualization

Interactive Plotly visualization (requires `plotly`):

```python
from tolerance_stack import (
    visualize_assembly, visualize_linkage,
    visualize_sensitivity, add_mc_cloud, VisualizationConfig,
)

# Visualize an assembly
fig = visualize_assembly(assy)
fig.show()

# Visualize a linkage
fig = visualize_linkage(linkage)
fig.show()

# Sensitivity chart
fig = visualize_sensitivity(results["rss"])
fig.show()

# Add Monte Carlo scatter cloud to a linkage plot
fig = visualize_linkage(linkage)
add_mc_cloud(fig, results["mc"])
fig.show()
```

---

## Saving and Loading

All models support JSON persistence:

```python
# Tolerance stacks
stack.save("my_stack.json")
loaded_stack = ToleranceStack.load("my_stack.json")

# Linkages
linkage.save("my_linkage.json")
loaded_linkage = Linkage.load("my_linkage.json")

# Assemblies
assy.save("my_assembly.json")
loaded_assy = Assembly.load("my_assembly.json")
```

---

## Streamlit GUI: STEP-to-Stack Workflow

The application includes a browser-based GUI for the complete STEP-to-analysis
workflow. Launch it with:

```bash
pip install streamlit plotly   # one-time setup
streamlit run tolerance_stack/gui.py
```

### GUI Workflow: Upload STEP, Define Features, Analyze

The **Assembly** tab provides a guided 5-step workflow:

**Step 1 -- Import Geometry**

Choose from four import sources:
- **STEP file upload** -- drag-and-drop a `.stp` or `.step` file. The parser
  extracts bodies, geometric features (planes, cylinders), and any embedded
  GD&T/PMI callouts automatically.
- **JSON file upload** -- load a previously saved assembly definition.
- **Load example** -- start from a built-in example (pin-in-hole, stacked
  plates, bracket).
- **Build manually** -- create bodies from scratch.

**Step 2 -- Edit Feature Tolerances**

After import, expand each body to see its features. For every feature you can
edit:
- Position tolerance, orientation tolerance, form tolerance
- Sigma level (default 3.0)
- Size nominal, size +/- tolerance (for cylinders and circles)

Click **Apply** to save changes to each feature.
You can also **add new features** to any body.

**Step 3 -- Add GD&T**

Select any `Body.Feature` from a dropdown and attach a Feature Control Frame:
- Choose from all 14 GD&T types (Position, Flatness, Perpendicularity, etc.)
- Set tolerance value, material condition (RFS/MMC/LMC), and datum references

**Step 4 -- Define Mates**

Connect features between bodies:
- Select Feature A and Feature B from dropdowns
- Choose mate type (coincident, coaxial, coplanar, at_distance, parallel,
  concentric)
- Optionally set distance and distance tolerance

**Step 5 -- Set Measurement**

Define what the analysis should compute:
- Select "from" and "to" features
- Choose measurement type (distance, gap, flush, interference, angle, etc.)
- Set direction vector for directional measurements

**Run Analysis**

With a measurement defined, click **Run Analysis** to execute Worst-Case, RSS,
and Monte Carlo. The right panel shows:
- Numerical results (nominal, min, max, tolerance range)
- Monte Carlo histogram with sigma lines
- Process capability (Cp/Cpk) if spec limits are set
- Sensitivity tornado chart
- 3D visualization (if Plotly is installed)
- DOF status for each body

### Other GUI Tabs

| Tab | Description |
|-----|-------------|
| **Tolerance Stack** | 1D linear stack builder and analysis |
| **Linkage** | 3D kinematic chain builder with joint/link editor |
| **DOE / Optimizer** | HLM, Full Factorial, LHS, RSM, Sobol', GA, Contingency |
| **Reports** | Generate HTML, APQP, PDF, or text reports |

---

## CLI Command Reference

| Command | Description |
|---------|-------------|
| `tolstack analyze <file>` | Analyze a 1D tolerance stack JSON file |
| `tolstack interactive` | Interactively build and analyze a 1D stack |
| `tolstack example {shaft\|multiaxis}` | Generate example stack file |
| `tolstack linkage-analyze <file>` | Analyze a linkage JSON file |
| `tolstack linkage-interactive` | Interactively build a linkage |
| `tolstack linkage-example {two-bar\|robot-arm\|four-bar}` | Generate example linkage |
| `tolstack assembly-analyze <file>` | Analyze an assembly JSON file |
| `tolstack assembly-example {pin-in-hole\|stacked-plates\|bracket}` | Generate example assembly |

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --methods` | `wc,rss,mc` | Comma-separated analysis methods |
| `--sigma` | `3.0` | Sigma level for RSS analysis |
| `--mc-samples` | `100000` | Number of Monte Carlo samples |
| `--seed` | None | Random seed for reproducibility |
| `--plot` | off | Show visualization plots |
| `--save-plots <base>` | None | Save plots to files |
| `--spec-limits <lo,hi>` | None | Spec limits for MC histogram |
| `-o, --output` | None | Output file path (for examples) |

---

## Quick-Start Cheat Sheet

```bash
# Install
pip install -e .

# Run a complete analysis in 30 seconds
tolstack example shaft -o demo.json
tolstack analyze demo.json

# Try a linkage
tolstack linkage-example robot-arm -o arm.json
tolstack linkage-analyze arm.json

# Try an assembly
tolstack assembly-example pin-in-hole -o pin.json
tolstack assembly-analyze pin.json

# Full Python workflow
python -c "
from tolerance_stack import *

stack = ToleranceStack('Demo', closure_direction=(1,0,0))
stack.add(Contributor('A', nominal=50, plus_tol=0.1, minus_tol=0.1, sign=+1))
stack.add(Contributor('B', nominal=48, plus_tol=0.05, minus_tol=0.05, sign=-1))

for m, r in analyze_stack(stack, mc_seed=42).items():
    print(r.summary())
"
```
