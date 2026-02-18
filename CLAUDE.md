# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D Tolerance Stack Analysis Tool (v0.6.0) — a Python library and CLI for 3-dimensional tolerance stack analysis supporting Worst-Case, RSS, and Monte Carlo methods. Includes full GD&T (ASME Y14.5), DOE, optimization, STEP CAD import, 3D visualization, and APQP reporting.

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[gui,dev,cad,viz]"   # all optional deps

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_analysis.py -v          # single module
python -m pytest tests/test_analysis.py::TestName -v # single test class

# CLI
tolstack --help
tolstack analyze examples/shaft_housing.json --method mc --samples 10000
tolstack linkage-analyze examples/two_bar_linkage.json
tolstack assembly-analyze examples/pin_in_hole_assembly.json

# Streamlit GUI
streamlit run tolerance_stack/gui.py
```

## Architecture

The package lives entirely in `tolerance_stack/` with three analysis tiers:

### Three Analysis Tiers
1. **1D Linear Stacks** (`models.py` → `analysis.py`): Dimension-loop analysis with 3D direction vectors. Uses `Contributor` and `ToleranceStack` dataclasses.
2. **3D Kinematic Linkages** (`linkage.py` → `linkage_analysis.py`): Joint/link chains analyzed via 4×4 homogeneous transformation matrices and numerical Jacobian (central difference).
3. **3D Rigid Body Assemblies** (`assembly.py` → `assembly_analysis.py`): Multi-body systems with geometric features (`cylinder`, `plane`, `sphere`, `slot`, `tab`, `pattern`) and mates (`cylindrical_fit`, `planar_contact`, `spherical_fit`, `slot_fit`, `pattern_fit`).

### Supporting Modules
- `gdt.py` — GD&T feature control frames, datum reference frames, material condition modifiers (MMC/LMC), composite GD&T
- `statistics.py` — Process capability metrics (Cp, Cpk, Pp, Ppk, PPM)
- `optimizer.py` — Tolerance optimization (gradient-free + genetic algorithm), DOE (full factorial, LHS, RSM), Sobol' sensitivity, contingency analysis
- `assembly_process.py` — Multi-stage assembly process modeling with stations and fixtures
- `step_import.py` — STEP file import; `pythonocc-core` is optional and guarded with try/except
- `visualization.py` — Plotly 3D interactive plots (also optional)
- `visualize.py` — Matplotlib static plots (waterfall, histogram, sensitivity)
- `reporting.py` — HTML, PDF, APQP report generation
- `cli.py` — argparse-based CLI (`tolstack` entry point)
- `gui.py` — Streamlit web UI with 5 tabs: Tolerance Stack, Linkage, Assembly, DOE/Optimizer, Reports

### Data Flow
All analysis tiers follow the same pattern: build a dataclass model → call `analyze_*()` → get an `AnalysisResult` dict with keys like `nominal`, `wc_min/wc_max`, `rss_min/rss_max`, `mc_mean/mc_std`, `sensitivity`, `mc_samples`. JSON files in `examples/` define stacks, linkages, and assemblies; the CLI and GUI load these directly.

### Key Patterns
- All domain models are Python `@dataclass` classes; enums define valid types throughout
- Optional dependencies (Plotly, Streamlit, pythonocc) use guarded imports — modules work without them
- Monte Carlo uses 9 distribution types defined in `Distribution` enum (normal, uniform, triangular, beta, lognormal, weibull, skew_normal, truncated_normal, shifted_rayleigh)
- `__init__.py` re-exports the full public API; all public symbols are in `__all__`

## Testing

267 tests across 18 modules in `tests/`. Tests are pure pytest with no fixtures or plugins. Test file names mirror source modules (e.g., `test_assembly.py` tests `assembly.py` + `assembly_analysis.py`).

## JSON Data Formats

Three JSON schemas distinguished by a `"type"` field:
- No `type` field or `"type": "stack"` → 1D tolerance stack
- `"type": "linkage"` → 3D linkage (joints + links)
- `"type": "assembly"` → 3D assembly (bodies + features + mates + measurement)

Example files in `examples/` cover all three types.
