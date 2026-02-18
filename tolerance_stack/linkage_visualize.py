"""Visualization for 3D linkage tolerance analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tolerance_stack.linkage import Linkage
from tolerance_stack.linkage_analysis import LinkageAnalysisResult


def plot_linkage_3d(
    linkage: Linkage,
    result: Optional[LinkageAnalysisResult] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_tolerance_ellipsoid: bool = True,
) -> None:
    """Plot the 3D linkage skeleton with joint positions and tolerance zones.

    Shows the kinematic chain as connected line segments with joint markers.
    If a Monte Carlo result is provided, draws the tolerance ellipsoid at
    the end-effector and scatter of MC samples.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw nominal linkage skeleton
    joint_positions = linkage.all_joint_positions()
    xs = [p[0] for _, p in joint_positions]
    ys = [p[1] for _, p in joint_positions]
    zs = [p[2] for _, p in joint_positions]

    ax.plot(xs, ys, zs, "o-", color="#2196F3", linewidth=2.5, markersize=8,
            label="Nominal chain")

    # Label joints
    for name, pos in joint_positions:
        ax.text(pos[0], pos[1], pos[2], f"  {name}", fontsize=7, color="black")

    # Draw end-effector
    end_pos = linkage.end_effector_position()
    ax.scatter(*end_pos, color="red", s=100, zorder=5, label="End-effector")

    if result is not None:
        # Draw tolerance box at end-effector
        nom = result.nominal_position
        p_tol = result.plus_tolerance
        m_tol = result.minus_tolerance

        # Draw bounding box edges
        _draw_box(ax, nom - m_tol, nom + p_tol, color="green", alpha=0.15,
                  label="Tolerance zone")

        # If MC data, draw scatter
        if result.mc_samples is not None and show_tolerance_ellipsoid:
            samples = result.mc_samples
            # Subsample for performance
            n_plot = min(2000, len(samples))
            idx = np.random.default_rng(0).choice(len(samples), n_plot, replace=False)
            ax.scatter(
                samples[idx, 0], samples[idx, 1], samples[idx, 2],
                alpha=0.08, s=3, color="orange", label=f"MC samples (n={len(samples):,})",
            )

            # Draw covariance ellipsoid
            if result.mc_cov is not None and show_tolerance_ellipsoid:
                _draw_covariance_ellipsoid(ax, result.mc_mean, result.mc_cov,
                                           n_std=3.0, color="orange", alpha=0.1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or f"{linkage.name} — 3D Linkage")
    ax.legend(loc="best", fontsize=8)

    # Equal aspect ratio
    _set_equal_aspect_3d(ax, xs + [end_pos[0]], ys + [end_pos[1]], zs + [end_pos[2]])

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved linkage plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_linkage_sensitivity(
    result: LinkageAnalysisResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-parameter sensitivity as grouped horizontal bars (dX, dY, dZ)."""
    import matplotlib.pyplot as plt

    if not result.sensitivity:
        print("No sensitivity data to plot.")
        return

    names = [s[0] for s in result.sensitivity]
    dx = [s[1][0] for s in result.sensitivity]
    dy = [s[1][1] for s in result.sensitivity]
    dz = [s[1][2] for s in result.sensitivity]

    # Sort by total magnitude
    mag = [np.linalg.norm(s[1]) for s in result.sensitivity]
    order = np.argsort(mag)[::-1]
    names = [names[i] for i in order]
    dx = [dx[i] for i in order]
    dy = [dy[i] for i in order]
    dz = [dz[i] for i in order]

    y = np.arange(len(names))
    bar_h = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5 + 1)))
    ax.barh(y - bar_h, dx, height=bar_h, color="#F44336", label="dX")
    ax.barh(y, dy, height=bar_h, color="#4CAF50", label="dY")
    ax.barh(y + bar_h, dz, height=bar_h, color="#2196F3", label="dZ")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sensitivity (end-effector shift per unit)")
    ax.set_title(title or f"Linkage Sensitivity — {result.method}")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved linkage sensitivity plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_linkage_mc_scatter(
    result: LinkageAnalysisResult,
    axes: tuple[int, int] = (0, 1),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """2D scatter plot of Monte Carlo end-effector samples on two chosen axes."""
    import matplotlib.pyplot as plt

    if result.mc_samples is None:
        raise ValueError("No Monte Carlo samples in result")

    axis_labels = ["X", "Y", "Z"]
    a1, a2 = axes
    samples = result.mc_samples

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(samples[:, a1], samples[:, a2], alpha=0.05, s=1, color="#2196F3")

    # Draw nominal
    nom = result.nominal_position
    ax.scatter(nom[a1], nom[a2], color="red", s=80, zorder=5,
               marker="+", linewidths=2, label="Nominal")

    # Draw mean
    if result.mc_mean is not None:
        ax.scatter(result.mc_mean[a1], result.mc_mean[a2], color="green",
                   s=80, zorder=5, marker="x", linewidths=2, label="MC Mean")

    # Draw 3-sigma ellipse from covariance
    if result.mc_cov is not None:
        cov_2d = result.mc_cov[np.ix_([a1, a2], [a1, a2])]
        _draw_cov_ellipse_2d(ax, result.mc_mean[[a1, a2]], cov_2d, n_std=3.0)

    ax.set_xlabel(f"{axis_labels[a1]} position")
    ax.set_ylabel(f"{axis_labels[a2]} position")
    ax.set_title(title or f"MC Scatter — {axis_labels[a1]} vs {axis_labels[a2]}")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved MC scatter to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_box(ax, lo: np.ndarray, hi: np.ndarray, color="green", alpha=0.15,
              label=None) -> None:
    """Draw a wireframe box in 3D."""
    corners = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ]
    for i, (a, b) in enumerate(edges):
        ax.plot3D(
            *zip(corners[a], corners[b]),
            color=color, alpha=0.5,
            label=label if i == 0 else None,
        )


def _draw_covariance_ellipsoid(ax, center, cov, n_std=3.0, color="orange",
                                alpha=0.1, n_points=30) -> None:
    """Draw a 3D covariance ellipsoid on a matplotlib 3D axis."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Clamp small eigenvalues to avoid degenerate ellipsoids
    eigenvalues = np.maximum(eigenvalues, 1e-20)
    radii = n_std * np.sqrt(eigenvalues)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(u)):
        for j in range(len(v)):
            pt = eigenvectors @ np.array([x[i, j], y[i, j], z[i, j]]) + center
            x[i, j], y[i, j], z[i, j] = pt

    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def _draw_cov_ellipse_2d(ax, center, cov_2d, n_std=3.0, color="orange") -> None:
    """Draw a 2D covariance ellipse."""
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    eigenvalues = np.maximum(eigenvalues, 1e-20)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.15,
                      linewidth=2, label=f"{n_std:.0f}\u03c3 ellipse")
    ax.add_patch(ellipse)


def _set_equal_aspect_3d(ax, xs, ys, zs) -> None:
    """Set equal aspect ratio for a 3D matplotlib axis."""
    max_range = max(
        max(xs) - min(xs) if xs else 1,
        max(ys) - min(ys) if ys else 1,
        max(zs) - min(zs) if zs else 1,
    ) / 2.0
    if max_range < 1e-10:
        max_range = 1.0
    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    mid_z = (max(zs) + min(zs)) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
