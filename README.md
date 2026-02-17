# 3D Tolerance Stack Analysis Tool

A Python application for performing 3-dimensional tolerance stack analysis using
Worst-Case, Root Sum of Squares (RSS), and Monte Carlo simulation methods.

## Features

- **3D direction vectors** — each contributor has a direction in 3D space,
  projected onto the closure direction to compute its effect on the gap
- **Three analysis methods**:
  - **Worst-Case (WC)** — every contributor at its extreme simultaneously
  - **RSS** — statistical root-sum-of-squares assuming normal distributions
  - **Monte Carlo** — numerical simulation with normal, uniform, or triangular distributions
- **Asymmetric tolerances** — supports unequal +/- tolerance values
- **Contributor types** — linear, angular (degrees), and geometric (GD&T)
- **Visualization** — waterfall charts, Monte Carlo histograms, sensitivity bar charts, 3D vector plots
- **JSON persistence** — save and load stack definitions as JSON files
- **CLI and interactive mode** — command-line analysis or guided stack builder

## Installation

```bash
pip install numpy matplotlib
# Optional for RSS yield calculation:
pip install scipy
```

## Quick Start

### Analyze a predefined example

```bash
# Generate an example stack file
python -m tolerance_stack.cli example shaft -o shaft_stack.json

# Run all three analyses
python -m tolerance_stack.cli analyze shaft_stack.json

# Run with plots
python -m tolerance_stack.cli analyze shaft_stack.json --plot

# Run only worst-case and RSS at 6-sigma
python -m tolerance_stack.cli analyze shaft_stack.json -m wc,rss --sigma 6
```

### Interactive mode

```bash
python -m tolerance_stack.cli interactive
```

Prompts you to enter each contributor's name, nominal, tolerances, 3D direction,
sign, distribution, and type.

### Python API

```python
from tolerance_stack import Contributor, ToleranceStack, analyze_stack

stack = ToleranceStack(
    name="My Assembly",
    closure_direction=(1, 0, 0),
)

stack.add(Contributor(
    name="Housing length",
    nominal=50.0,
    plus_tol=0.10,
    minus_tol=0.10,
    direction=(1, 0, 0),
    sign=+1,
))

stack.add(Contributor(
    name="Shaft length",
    nominal=48.0,
    plus_tol=0.05,
    minus_tol=0.05,
    direction=(1, 0, 0),
    sign=-1,
))

results = analyze_stack(stack, sigma=3.0, mc_samples=100_000, mc_seed=42)

for method, result in results.items():
    print(result.summary())
```

## Concepts

### Dimension Loop

A tolerance stack is a closed loop of dimensions. Each contributor either
**adds to** (`sign=+1`) or **subtracts from** (`sign=-1`) the gap.

### 3D Direction Vectors

Each contributor has a `direction` vector in 3D space. Its contribution to
the gap is the dot product of its direction with the `closure_direction`.
A contributor perpendicular to the closure direction has zero effect.

### Analysis Methods

| Method | Description | Use When |
|--------|-------------|----------|
| Worst-Case | All tolerances at extreme | Safety-critical, must never fail |
| RSS | Statistical (assumes normal) | Typical manufacturing, cost-sensitive |
| Monte Carlo | Numerical simulation | Complex distributions, high confidence |

## Stack JSON Format

```json
{
  "name": "Assembly Name",
  "description": "Optional description",
  "closure_direction": [1, 0, 0],
  "contributors": [
    {
      "name": "Part A",
      "nominal": 10.0,
      "plus_tol": 0.1,
      "minus_tol": 0.1,
      "direction": [1, 0, 0],
      "sign": 1,
      "distribution": "normal",
      "contributor_type": "linear",
      "sigma": 3.0
    }
  ]
}
```

## 3D Linkage Analysis

In addition to linear tolerance stacks, the tool supports **3D kinematic
linkages** — chains of rigid links connected by joints. Tolerances on
link lengths and joint angles propagate to the end-effector position
through forward kinematics and Jacobian-based sensitivity analysis.

### Linkage Features

- **Joint types**: Revolute (X/Y/Z), Prismatic (X/Y/Z), Spherical, Fixed
- **Forward kinematics** using 4x4 homogeneous transformation matrices
- **Numerical Jacobian** for sensitivity of end-effector XYZ to each parameter
- **3D tolerance zones** — per-axis and radial tolerance at the end-effector
- **Covariance ellipsoids** from Monte Carlo showing the 3D scatter shape
- **Visualization** — 3D skeleton plots, sensitivity bar charts, MC scatter

### Linkage Quick Start

```bash
# Generate an example linkage
python -m tolerance_stack.cli linkage-example two-bar -o two_bar.json
python -m tolerance_stack.cli linkage-example robot-arm -o robot_arm.json
python -m tolerance_stack.cli linkage-example four-bar -o four_bar.json

# Analyze a linkage
python -m tolerance_stack.cli linkage-analyze two_bar.json

# With plots
python -m tolerance_stack.cli linkage-analyze robot_arm.json --plot

# Interactive linkage builder
python -m tolerance_stack.cli linkage-interactive
```

### Linkage Python API

```python
from tolerance_stack import Joint, JointType, Link, Linkage, analyze_linkage

linkage = Linkage(name="My Mechanism")

linkage.add_joint(Joint("Base", JointType.REVOLUTE_Z, nominal=30.0,
                        plus_tol=0.5, minus_tol=0.5))
linkage.add_link(Link("Arm1", length=100.0, plus_tol=0.1, minus_tol=0.1,
                       direction=(1, 0, 0)))
linkage.add_joint(Joint("Elbow", JointType.REVOLUTE_Z, nominal=45.0,
                        plus_tol=0.5, minus_tol=0.5))
linkage.add_link(Link("Arm2", length=80.0, plus_tol=0.08, minus_tol=0.08,
                       direction=(1, 0, 0)))
linkage.add_joint(Joint("Tip", JointType.FIXED))

results = analyze_linkage(linkage, mc_seed=42)

for method, result in results.items():
    print(result.summary())
```

### Linkage JSON Format

```json
{
  "type": "linkage",
  "name": "Two-Bar Linkage",
  "joints": [
    {"name": "Base", "joint_type": "revolute_z", "nominal": 30.0,
     "plus_tol": 0.5, "minus_tol": 0.5},
    {"name": "Elbow", "joint_type": "revolute_z", "nominal": 45.0,
     "plus_tol": 0.5, "minus_tol": 0.5},
    {"name": "Tip", "joint_type": "fixed", "nominal": 0.0}
  ],
  "links": [
    {"name": "Arm1", "length": 100.0, "plus_tol": 0.1, "minus_tol": 0.1,
     "direction": [1, 0, 0]},
    {"name": "Arm2", "length": 80.0, "plus_tol": 0.08, "minus_tol": 0.08,
     "direction": [1, 0, 0]}
  ]
}
```

### How Linkage Tolerance Propagation Works

1. **Build the chain**: Joints and links alternate: J0 -> L0 -> J1 -> L1 -> ... -> Jn
2. **Forward kinematics**: Multiply all 4x4 transforms to get end-effector position
3. **Numerical Jacobian**: Perturb each tolerance parameter and measure the
   end-effector shift (central difference). This captures lever arm amplification,
   angular coupling, and 3D geometry effects.
4. **Analysis**:
   - **Worst-Case**: |J_col| * half_tolerance, summed per axis
   - **RSS**: Variances add in quadrature through the Jacobian
   - **Monte Carlo**: Full nonlinear FK with sampled parameters, producing a
     3D position distribution with covariance matrix

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
