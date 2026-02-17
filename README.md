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

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
