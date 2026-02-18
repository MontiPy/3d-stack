"""3D assembly visualization using Plotly.

Generates interactive 3D visualizations of assemblies, linkages,
and tolerance analysis results for embedding in the Streamlit GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Plotly-based 3D visualization (soft dependency)
# ---------------------------------------------------------------------------

def _check_plotly():
    """Check if plotly is available."""
    try:
        import plotly.graph_objects as go  # noqa: F401
        return True
    except ImportError:
        return False


PLOTLY_AVAILABLE = _check_plotly()


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BODY_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#F44336", "#00BCD4", "#795548", "#607D8B",
    "#E91E63", "#3F51B5", "#8BC34A", "#FFC107",
]

FEATURE_COLORS = {
    "point": "#F44336",
    "plane": "#4CAF50",
    "axis": "#2196F3",
    "cylinder": "#FF9800",
    "circle": "#9C27B0",
}


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization.

    Attributes:
        width: Plot width in pixels.
        height: Plot height in pixels.
        show_features: Display feature geometry.
        show_mates: Display mate connections.
        show_measurement: Highlight measurement path.
        show_mc_cloud: Show Monte Carlo sample cloud.
        mc_sample_count: Number of MC samples to display.
        mc_alpha: Opacity for MC samples.
        feature_scale: Scale factor for feature indicators.
        body_opacity: Opacity for body representations.
    """
    width: int = 800
    height: int = 600
    show_features: bool = True
    show_mates: bool = True
    show_measurement: bool = True
    show_mc_cloud: bool = True
    mc_sample_count: int = 2000
    mc_alpha: float = 0.15
    feature_scale: float = 5.0
    body_opacity: float = 0.3


def visualize_assembly(assembly, config: Optional[VisualizationConfig] = None):
    """Create an interactive 3D Plotly figure for an assembly.

    Args:
        assembly: Assembly object.
        config: Visualization configuration.

    Returns:
        plotly.graph_objects.Figure if plotly is available, else None.
    """
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    if config is None:
        config = VisualizationConfig()

    fig = go.Figure()

    body_names = list(assembly.bodies.keys())

    for bi, (bname, bp) in enumerate(assembly.bodies.items()):
        color = BODY_COLORS[bi % len(BODY_COLORS)]
        T = bp.transform
        origin = T[:3, 3]

        # Body origin marker
        fig.add_trace(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="diamond"),
            text=[bname],
            textposition="top center",
            textfont=dict(size=10),
            name=f"Body: {bname}",
            legendgroup=bname,
        ))

        # Body local axes (small triad)
        axis_len = config.feature_scale * 0.5
        for ai, (ax_label, ax_color) in enumerate(zip(["X", "Y", "Z"], ["red", "green", "blue"])):
            ax_dir = T[:3, ai]
            end = origin + ax_dir * axis_len
            fig.add_trace(go.Scatter3d(
                x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=ax_color, width=3),
                name=f"{bname} {ax_label}",
                legendgroup=bname,
                showlegend=False,
            ))

        # Features
        if config.show_features:
            for fname, feat in bp.body.features.items():
                f_origin = feat.world_origin(T)
                f_dir = feat.world_direction(T)
                ft = feat.feature_type.value
                fc = FEATURE_COLORS.get(ft, "#999999")

                # Feature origin
                fig.add_trace(go.Scatter3d(
                    x=[f_origin[0]], y=[f_origin[1]], z=[f_origin[2]],
                    mode="markers+text",
                    marker=dict(size=5, color=fc, symbol="circle"),
                    text=[f"{fname} ({ft})"],
                    textposition="bottom center",
                    textfont=dict(size=8, color=fc),
                    name=f"{bname}.{fname}",
                    legendgroup=bname,
                    showlegend=False,
                ))

                # Direction indicator
                if ft in ("plane", "axis", "cylinder", "circle"):
                    d_end = f_origin + f_dir * config.feature_scale
                    fig.add_trace(go.Scatter3d(
                        x=[f_origin[0], d_end[0]],
                        y=[f_origin[1], d_end[1]],
                        z=[f_origin[2], d_end[2]],
                        mode="lines",
                        line=dict(color=fc, width=2, dash="dash"),
                        legendgroup=bname,
                        showlegend=False,
                    ))

                # Cylinder/circle radius ring
                if ft in ("cylinder", "circle") and feat.radius > 0:
                    _add_circle_trace(fig, f_origin, f_dir, feat.radius, fc, bname)

                # Tolerance zone indicator
                if feat.position_tol > 0:
                    ht = feat.position_tol / 2.0
                    _add_tolerance_box(fig, f_origin, ht, bname)

    # Mates
    if config.show_mates:
        for mate in assembly.mates:
            bp_a = assembly.bodies[mate.body_a]
            bp_b = assembly.bodies[mate.body_b]
            feat_a = bp_a.body.get_feature(mate.feature_a)
            feat_b = bp_b.body.get_feature(mate.feature_b)
            pos_a = feat_a.world_origin(bp_a.transform)
            pos_b = feat_b.world_origin(bp_b.transform)

            fig.add_trace(go.Scatter3d(
                x=[pos_a[0], pos_b[0]],
                y=[pos_a[1], pos_b[1]],
                z=[pos_a[2], pos_b[2]],
                mode="lines",
                line=dict(color="#999", width=2, dash="dot"),
                name=f"Mate: {mate.name}",
            ))

    # Measurement
    if config.show_measurement and assembly.measurement:
        meas = assembly.measurement
        bp_a = assembly.bodies[meas.body_a]
        bp_b = assembly.bodies[meas.body_b]
        feat_a = bp_a.body.get_feature(meas.feature_a)
        feat_b = bp_b.body.get_feature(meas.feature_b)
        pos_a = feat_a.world_origin(bp_a.transform)
        pos_b = feat_b.world_origin(bp_b.transform)

        fig.add_trace(go.Scatter3d(
            x=[pos_a[0], pos_b[0]],
            y=[pos_a[1], pos_b[1]],
            z=[pos_a[2], pos_b[2]],
            mode="lines+markers",
            line=dict(color="#E91E63", width=4),
            marker=dict(size=6, color="#E91E63", symbol="x"),
            name=f"Measurement: {meas.name}",
        ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=config.width,
        height=config.height,
        title=f"Assembly: {assembly.name}",
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def add_mc_cloud(fig, mc_samples, config: Optional[VisualizationConfig] = None):
    """Add Monte Carlo sample cloud to an existing assembly figure.

    Args:
        fig: Plotly figure from visualize_assembly().
        mc_samples: Array of MC measurement samples (1D for scalar, Nx3 for 3D).
        config: Visualization configuration.
    """
    if not PLOTLY_AVAILABLE or fig is None:
        return

    import plotly.graph_objects as go

    if config is None:
        config = VisualizationConfig()

    samples = np.asarray(mc_samples)
    n = min(config.mc_sample_count, len(samples))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), n, replace=False)

    if samples.ndim == 2 and samples.shape[1] >= 3:
        subset = samples[idx]
        fig.add_trace(go.Scatter3d(
            x=subset[:, 0], y=subset[:, 1], z=subset[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#FF9800", opacity=config.mc_alpha),
            name=f"MC Samples (n={n})",
        ))
    elif samples.ndim == 1:
        # For scalar measurements, show as histogram overlay (2D)
        pass  # Scalar MC handled separately in GUI


def visualize_linkage(linkage, mc_samples=None,
                      config: Optional[VisualizationConfig] = None):
    """Create an interactive 3D Plotly figure for a kinematic linkage.

    Args:
        linkage: Linkage object.
        mc_samples: Optional Nx3 array of MC end-effector positions.
        config: Visualization configuration.

    Returns:
        plotly.graph_objects.Figure if plotly is available, else None.
    """
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    if config is None:
        config = VisualizationConfig()

    fig = go.Figure()

    # Nominal chain
    positions = linkage.all_joint_positions()
    end_pos = linkage.end_effector_position()

    xs = [p[0] for _, p in positions]
    ys = [p[1] for _, p in positions]
    zs = [p[2] for _, p in positions]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines+markers+text",
        line=dict(color="#2196F3", width=5),
        marker=dict(size=8, color="#2196F3"),
        text=[name for name, _ in positions],
        textposition="top center",
        textfont=dict(size=9),
        name="Nominal Chain",
    ))

    # End-effector
    fig.add_trace(go.Scatter3d(
        x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name="End-effector",
    ))

    # MC scatter
    if mc_samples is not None and config.show_mc_cloud:
        samples = np.asarray(mc_samples)
        if samples.ndim == 2 and samples.shape[1] >= 3:
            n = min(config.mc_sample_count, len(samples))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(samples), n, replace=False)
            subset = samples[idx]
            fig.add_trace(go.Scatter3d(
                x=subset[:, 0], y=subset[:, 1], z=subset[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="#FF9800", opacity=config.mc_alpha),
                name=f"MC Samples (n={n})",
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=config.width,
        height=config.height,
        title=f"Linkage: {linkage.name}",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def visualize_sensitivity(sensitivity, title="Sensitivity Analysis",
                          config: Optional[VisualizationConfig] = None):
    """Create a Plotly horizontal bar chart for sensitivity data.

    Args:
        sensitivity: List of (name, value) tuples or (name, [x,y,z]) tuples.
        title: Chart title.
        config: Configuration.

    Returns:
        plotly.graph_objects.Figure or None.
    """
    if not PLOTLY_AVAILABLE or not sensitivity:
        return None

    import plotly.graph_objects as go

    if config is None:
        config = VisualizationConfig()

    # Determine if scalar or vector sensitivity
    first_val = sensitivity[0][1]
    is_vector = hasattr(first_val, '__len__')

    if is_vector:
        # Sort by magnitude
        sorted_sens = sorted(sensitivity, key=lambda x: np.linalg.norm(x[1]), reverse=True)
        names = [s[0] for s in sorted_sens]
        dx = [s[1][0] for s in sorted_sens]
        dy = [s[1][1] for s in sorted_sens]
        dz = [s[1][2] for s in sorted_sens]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=dx, name="dX", orientation="h",
                             marker_color="#F44336"))
        fig.add_trace(go.Bar(y=names, x=dy, name="dY", orientation="h",
                             marker_color="#4CAF50"))
        fig.add_trace(go.Bar(y=names, x=dz, name="dZ", orientation="h",
                             marker_color="#2196F3"))
        fig.update_layout(barmode="group")
    else:
        sorted_sens = sorted(sensitivity, key=lambda x: abs(x[1]), reverse=True)
        names = [s[0] for s in sorted_sens]
        vals = [s[1] for s in sorted_sens]
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in vals]

        fig = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker_color=colors,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Sensitivity",
        width=config.width,
        height=max(300, len(sensitivity) * 30),
        margin=dict(l=200, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_circle_trace(fig, center, normal, radius, color, group):
    """Add a circle trace to a 3D Plotly figure."""
    import plotly.graph_objects as go

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Find two perpendicular vectors
    if abs(normal[0]) < 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    else:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    theta = np.linspace(0, 2 * np.pi, 32)
    pts = center[:, None] + radius * (u[:, None] * np.cos(theta) + v[:, None] * np.sin(theta))

    fig.add_trace(go.Scatter3d(
        x=pts[0], y=pts[1], z=pts[2],
        mode="lines",
        line=dict(color=color, width=2),
        legendgroup=group,
        showlegend=False,
    ))


def _add_tolerance_box(fig, center, half_tol, group):
    """Add a small translucent tolerance zone indicator."""
    import plotly.graph_objects as go

    # Small cube wireframe
    h = half_tol
    c = center
    # Cube corners
    corners = np.array([
        [c[0]-h, c[1]-h, c[2]-h],
        [c[0]+h, c[1]-h, c[2]-h],
        [c[0]+h, c[1]+h, c[2]-h],
        [c[0]-h, c[1]+h, c[2]-h],
        [c[0]-h, c[1]-h, c[2]+h],
        [c[0]+h, c[1]-h, c[2]+h],
        [c[0]+h, c[1]+h, c[2]+h],
        [c[0]-h, c[1]+h, c[2]+h],
    ])

    # Draw edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i][0], corners[j][0]],
            y=[corners[i][1], corners[j][1]],
            z=[corners[i][2], corners[j][2]],
            mode="lines",
            line=dict(color="rgba(255,152,0,0.3)", width=1),
            legendgroup=group,
            showlegend=False,
        ))
