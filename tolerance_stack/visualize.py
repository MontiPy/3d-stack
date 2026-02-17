"""Visualization helpers for 3D tolerance stack results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tolerance_stack.analysis import AnalysisResult
from tolerance_stack.models import ToleranceStack


def plot_waterfall(
    stack: ToleranceStack,
    result: AnalysisResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Draw a waterfall chart showing each contributor's effect on the gap.

    The chart shows the nominal contribution and tolerance band for each
    contributor, projected onto the closure direction.
    """
    import matplotlib.pyplot as plt

    closure_dir = np.array(stack.closure_direction, dtype=float)

    names = []
    nominals = []
    plus_tols = []
    minus_tols = []

    for c in stack.contributors:
        from tolerance_stack.analysis import _projection_factor, _effective_tolerance
        pf = _projection_factor(c, closure_dir)
        sens = c.sign * pf

        half, shift = _effective_tolerance(c)
        names.append(c.name)
        nominals.append(sens * c.nominal)
        plus_tols.append(abs(sens) * half)
        minus_tols.append(abs(sens) * half)

    # Build the waterfall
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.5 + 2)))

    cumulative = 0.0
    y_positions = list(range(len(names)))

    for i, (name, nom, pt, mt) in enumerate(zip(names, nominals, plus_tols, minus_tols)):
        # Bar for the nominal contribution
        color = "#2196F3" if nom >= 0 else "#F44336"
        ax.barh(i, nom, left=cumulative, height=0.5, color=color, alpha=0.8,
                edgecolor="black", linewidth=0.5)

        # Error bars for tolerance band
        center = cumulative + nom
        ax.plot([center - mt, center + pt], [i, i], color="black", linewidth=2)
        ax.plot([center - mt, center - mt], [i - 0.15, i + 0.15], color="black", linewidth=2)
        ax.plot([center + pt, center + pt], [i - 0.15, i + 0.15], color="black", linewidth=2)

        cumulative += nom

    # Final gap marker
    ax.axvline(x=result.nominal_gap, color="green", linestyle="--", linewidth=1.5,
               label=f"Nominal gap = {result.nominal_gap:.4f}")
    ax.axvspan(result.gap_min, result.gap_max, alpha=0.15, color="green",
               label=f"Gap range [{result.gap_min:.4f}, {result.gap_max:.4f}]")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.set_xlabel("Dimension (projected onto closure direction)")
    ax.set_title(title or f"{stack.name} — {result.method} Waterfall")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved waterfall chart to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_monte_carlo_histogram(
    result: AnalysisResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    bins: int = 100,
    spec_limits: Optional[tuple[float, float]] = None,
) -> None:
    """Plot a histogram of Monte Carlo gap samples.

    Args:
        result: An AnalysisResult from Monte Carlo analysis.
        title: Optional title override.
        save_path: If given, save the plot to this path instead of showing.
        bins: Number of histogram bins.
        spec_limits: Optional (lower, upper) spec limits to overlay.
    """
    import matplotlib.pyplot as plt

    if result.mc_samples is None:
        raise ValueError("AnalysisResult does not contain Monte Carlo samples")

    samples = result.mc_samples
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(samples, bins=bins, density=True, alpha=0.7, color="#2196F3",
            edgecolor="black", linewidth=0.3)

    # Overlay normal fit
    from matplotlib.patches import FancyArrowPatch
    x = np.linspace(samples.min(), samples.max(), 300)
    pdf = (1.0 / (result.mc_std * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((x - result.mc_mean) / result.mc_std) ** 2)
    ax.plot(x, pdf, "r-", linewidth=2, label="Normal fit")

    # Mean and std lines
    ax.axvline(result.mc_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {result.mc_mean:.4f}")
    for k in [1, 2, 3]:
        ax.axvline(result.mc_mean + k * result.mc_std, color="orange",
                   linestyle=":", linewidth=0.8,
                   label=f"+{k}\u03c3 = {result.mc_mean + k * result.mc_std:.4f}" if k == 3 else None)
        ax.axvline(result.mc_mean - k * result.mc_std, color="orange",
                   linestyle=":", linewidth=0.8,
                   label=f"-{k}\u03c3 = {result.mc_mean - k * result.mc_std:.4f}" if k == 3 else None)

    # Spec limits
    if spec_limits:
        lo, hi = spec_limits
        ax.axvline(lo, color="red", linewidth=2, label=f"LSL = {lo:.4f}")
        ax.axvline(hi, color="red", linewidth=2, label=f"USL = {hi:.4f}")
        # Compute out-of-spec
        out_of_spec = np.sum((samples < lo) | (samples > hi)) / len(samples) * 100
        ax.text(0.02, 0.95, f"Out of spec: {out_of_spec:.3f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Gap Value")
    ax.set_ylabel("Probability Density")
    ax.set_title(title or f"Monte Carlo Distribution (n={len(samples):,})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sensitivity(
    result: AnalysisResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a bar chart of contributor sensitivity (Pareto-style)."""
    import matplotlib.pyplot as plt

    if not result.sensitivity:
        print("No sensitivity data to plot.")
        return

    # Sort by absolute sensitivity
    sorted_sens = sorted(result.sensitivity, key=lambda x: abs(x[1]), reverse=True)
    names = [s[0] for s in sorted_sens]
    values = [s[1] for s in sorted_sens]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4 + 1)))
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in values]
    y_pos = range(len(names))

    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel("Sensitivity (contribution per unit to gap)")
    ax.set_title(title or f"Sensitivity Analysis — {result.method}")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved sensitivity chart to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_3d_stack(
    stack: ToleranceStack,
    save_path: Optional[str] = None,
) -> None:
    """Plot a 3D arrow diagram showing each contributor's direction and magnitude."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    origin = np.array([0.0, 0.0, 0.0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(stack.contributors)))

    for i, c in enumerate(stack.contributors):
        d = np.array(c.direction, dtype=float)
        vec = c.sign * c.nominal * d
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=colors[i], arrow_length_ratio=0.1, linewidth=2,
            label=f"{c.name} ({c.sign * c.nominal:+.3f})",
        )
        origin = origin + vec

    # Draw closure direction
    cd = np.array(stack.closure_direction, dtype=float)
    max_dim = max(abs(origin).max(), 1.0) * 0.3
    ax.quiver(0, 0, 0, cd[0] * max_dim, cd[1] * max_dim, cd[2] * max_dim,
              color="black", arrow_length_ratio=0.15, linewidth=3,
              linestyle="dashed", label="Closure direction")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{stack.name} — 3D Stack Vectors")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)
