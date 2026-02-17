"""Report generation for tolerance analysis results.

Produces HTML, PDF (via HTML), and plain-text reports matching industrial
standards including APQP compliance templates.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        title: Report title.
        project: Project name.
        author: Author name.
        revision: Document revision.
        date: Report date (defaults to now).
        logo_path: Optional path to company logo image.
        include_sensitivity: Include sensitivity analysis section.
        include_histograms: Include MC histogram plots.
        include_capability: Include process capability section.
        include_contributors: Include contributor detail table.
        image_width: Width for embedded images (pixels).
        image_height: Height for embedded images (pixels).
    """
    title: str = "Tolerance Analysis Report"
    project: str = ""
    author: str = ""
    revision: str = "A"
    date: str = ""
    logo_path: Optional[str] = None
    include_sensitivity: bool = True
    include_histograms: bool = True
    include_capability: bool = True
    include_contributors: bool = True
    image_width: int = 700
    image_height: int = 400


def generate_html_report(
    config: ReportConfig,
    analysis_results: dict,
    assembly_info: Optional[dict] = None,
    capability_results: Optional[dict] = None,
    plot_images: Optional[dict[str, bytes]] = None,
) -> str:
    """Generate an HTML tolerance analysis report.

    Args:
        config: Report configuration.
        analysis_results: Dict of method -> result objects with .summary().
        assembly_info: Optional dict describing the assembly (bodies, mates, etc.).
        capability_results: Optional process capability results.
        plot_images: Optional dict of name -> PNG bytes for embedding.

    Returns:
        HTML string.
    """
    import base64

    date_str = config.date or datetime.now().strftime("%Y-%m-%d %H:%M")

    html = [_html_header(config, date_str)]

    # Cover section
    html.append(f"""
    <div class="cover">
        <h1>{_esc(config.title)}</h1>
        <table class="info-table">
            <tr><td><b>Project:</b></td><td>{_esc(config.project)}</td></tr>
            <tr><td><b>Author:</b></td><td>{_esc(config.author)}</td></tr>
            <tr><td><b>Revision:</b></td><td>{_esc(config.revision)}</td></tr>
            <tr><td><b>Date:</b></td><td>{_esc(date_str)}</td></tr>
        </table>
    </div>
    """)

    # Assembly description
    if assembly_info:
        html.append('<div class="section"><h2>Assembly Description</h2>')
        if "name" in assembly_info:
            html.append(f'<p><b>Assembly:</b> {_esc(str(assembly_info["name"]))}</p>')
        if "description" in assembly_info:
            html.append(f'<p>{_esc(str(assembly_info["description"]))}</p>')

        if "bodies" in assembly_info:
            html.append('<h3>Bodies</h3><table class="data-table">')
            html.append('<tr><th>Name</th><th>Features</th><th>Description</th></tr>')
            for b in assembly_info["bodies"]:
                n_feat = len(b.get("features", []))
                html.append(f'<tr><td>{_esc(b["name"])}</td>'
                           f'<td>{n_feat}</td>'
                           f'<td>{_esc(b.get("description", ""))}</td></tr>')
            html.append('</table>')

        if "mates" in assembly_info:
            html.append('<h3>Mates</h3><table class="data-table">')
            html.append('<tr><th>Name</th><th>Type</th><th>Bodies</th><th>Distance Tol</th></tr>')
            for m in assembly_info["mates"]:
                html.append(f'<tr><td>{_esc(m["name"])}</td>'
                           f'<td>{_esc(m.get("mate_type", ""))}</td>'
                           f'<td>{_esc(m["body_a"])} - {_esc(m["body_b"])}</td>'
                           f'<td>{m.get("distance_tol", 0):.4f}</td></tr>')
            html.append('</table>')

        if "measurement" in assembly_info and assembly_info["measurement"]:
            m = assembly_info["measurement"]
            html.append(f'<h3>Measurement</h3>')
            html.append(f'<p><b>{_esc(m["name"])}:</b> {_esc(m["body_a"])}.{_esc(m["feature_a"])} '
                        f'&rarr; {_esc(m["body_b"])}.{_esc(m["feature_b"])} '
                        f'({_esc(m.get("measurement_type", "distance"))})</p>')
        html.append('</div>')

    # Analysis results
    for method_key, result in analysis_results.items():
        html.append(f'<div class="section"><h2>{_esc(method_key.upper())} Analysis Results</h2>')
        html.append(f'<pre class="summary">{_esc(result.summary())}</pre>')

        # Sensitivity table
        if config.include_sensitivity and hasattr(result, 'sensitivity') and result.sensitivity:
            html.append('<h3>Sensitivity Analysis</h3>')
            html.append('<table class="data-table">')
            html.append('<tr><th>Parameter</th><th>Sensitivity</th><th>|Sensitivity|</th></tr>')
            sorted_sens = sorted(result.sensitivity, key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
            for name, sens in sorted_sens:
                if isinstance(sens, (int, float)):
                    if abs(sens) > 1e-10:
                        css = 'class="positive"' if sens >= 0 else 'class="negative"'
                        html.append(f'<tr><td>{_esc(name)}</td>'
                                   f'<td {css}>{sens:+.6f}</td>'
                                   f'<td>{abs(sens):.6f}</td></tr>')
            html.append('</table>')

        html.append('</div>')

    # Process capability
    if config.include_capability and capability_results:
        html.append('<div class="section"><h2>Process Capability</h2>')
        if hasattr(capability_results, 'summary'):
            html.append(f'<pre class="summary">{_esc(capability_results.summary())}</pre>')
        else:
            html.append(f'<pre class="summary">{_esc(str(capability_results))}</pre>')
        html.append('</div>')

    # Embedded plots
    if plot_images:
        html.append('<div class="section"><h2>Analysis Plots</h2>')
        for name, img_bytes in plot_images.items():
            b64 = base64.b64encode(img_bytes).decode('ascii')
            html.append(f'<h3>{_esc(name)}</h3>')
            html.append(f'<img src="data:image/png;base64,{b64}" '
                        f'width="{config.image_width}" alt="{_esc(name)}">')
        html.append('</div>')

    html.append(_html_footer())
    return "\n".join(html)


def generate_text_report(
    config: ReportConfig,
    analysis_results: dict,
    capability_results=None,
) -> str:
    """Generate a plain-text tolerance analysis report."""
    lines = [
        "=" * 70,
        config.title.center(70),
        "=" * 70,
        f"Project:  {config.project}",
        f"Author:   {config.author}",
        f"Revision: {config.revision}",
        f"Date:     {config.date or datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
    ]

    for method_key, result in analysis_results.items():
        lines.append(result.summary())
        lines.append("")

    if capability_results and hasattr(capability_results, 'summary'):
        lines.append(capability_results.summary())
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    return "\n".join(lines)


def generate_apqp_report(
    config: ReportConfig,
    analysis_results: dict,
    assembly_info: Optional[dict] = None,
    capability_results=None,
    plot_images: Optional[dict[str, bytes]] = None,
    spec_limits: Optional[tuple[float, float]] = None,
) -> str:
    """Generate an APQP-compliant tolerance analysis report (HTML).

    Follows AIAG APQP template structure:
    1. Design Records
    2. Dimensional Results
    3. Process Capability Study
    4. Measurement System Analysis reference

    Args:
        config: Report configuration.
        analysis_results: Analysis results by method.
        assembly_info: Assembly description dict.
        capability_results: Process capability results.
        plot_images: Embedded plot images.
        spec_limits: (LSL, USL) specification limits.

    Returns:
        HTML string with APQP formatting.
    """
    import base64

    date_str = config.date or datetime.now().strftime("%Y-%m-%d %H:%M")
    html = [_html_header(config, date_str)]

    html.append(f"""
    <div class="cover apqp">
        <h1>APQP Dimensional Analysis Report</h1>
        <h2>{_esc(config.title)}</h2>
        <table class="info-table">
            <tr><td><b>Project:</b></td><td>{_esc(config.project)}</td></tr>
            <tr><td><b>Author:</b></td><td>{_esc(config.author)}</td></tr>
            <tr><td><b>Revision:</b></td><td>{_esc(config.revision)}</td></tr>
            <tr><td><b>Date:</b></td><td>{_esc(date_str)}</td></tr>
            <tr><td><b>Phase:</b></td><td>Design Verification (APQP Phase 3)</td></tr>
        </table>
    </div>
    """)

    # Section 1: Design Records
    html.append('<div class="section apqp-section">')
    html.append('<h2>1. Design Records</h2>')
    if assembly_info:
        html.append(f'<p><b>Assembly:</b> {_esc(str(assembly_info.get("name", "N/A")))}</p>')
        if assembly_info.get("measurement"):
            m = assembly_info["measurement"]
            html.append(f'<p><b>Critical Dimension:</b> {_esc(m["name"])}</p>')
            html.append(f'<p><b>Measurement:</b> {_esc(m["body_a"])}.{_esc(m["feature_a"])} '
                        f'&rarr; {_esc(m["body_b"])}.{_esc(m["feature_b"])}</p>')
    if spec_limits:
        html.append(f'<p><b>Specification:</b> LSL={spec_limits[0]:.4f}, USL={spec_limits[1]:.4f}</p>')
    html.append('</div>')

    # Section 2: Dimensional Results
    html.append('<div class="section apqp-section">')
    html.append('<h2>2. Dimensional Analysis Results</h2>')
    html.append('<table class="data-table">')
    html.append('<tr><th>Method</th><th>Nominal</th><th>Min</th><th>Max</th>'
                '<th>+Tol</th><th>-Tol</th></tr>')
    for method_key, result in analysis_results.items():
        nom = getattr(result, 'nominal_value', getattr(result, 'nominal_gap', 0))
        vmin = getattr(result, 'value_min', getattr(result, 'gap_min', 0))
        vmax = getattr(result, 'value_max', getattr(result, 'gap_max', 0))
        html.append(f'<tr><td>{_esc(method_key.upper())}</td>'
                    f'<td>{nom:.6f}</td><td>{vmin:.6f}</td><td>{vmax:.6f}</td>'
                    f'<td>+{result.plus_tolerance:.6f}</td>'
                    f'<td>-{result.minus_tolerance:.6f}</td></tr>')
    html.append('</table>')
    html.append('</div>')

    # Section 3: Process Capability
    html.append('<div class="section apqp-section">')
    html.append('<h2>3. Process Capability Study</h2>')
    if capability_results and hasattr(capability_results, 'summary'):
        html.append(f'<pre class="summary">{_esc(capability_results.summary())}</pre>')
        if hasattr(capability_results, 'cpk'):
            cpk = capability_results.cpk
            status = "CAPABLE" if cpk >= 1.33 else ("MARGINAL" if cpk >= 1.0 else "NOT CAPABLE")
            color = "#4CAF50" if cpk >= 1.33 else ("#FF9800" if cpk >= 1.0 else "#F44336")
            html.append(f'<p style="font-size:18px;color:{color};font-weight:bold;">'
                        f'Cpk = {cpk:.3f} &mdash; {status}</p>')
    else:
        html.append('<p>Process capability data not available. '
                    'Run Monte Carlo analysis with specification limits to generate.</p>')
    html.append('</div>')

    # Section 4: Plots
    if plot_images:
        html.append('<div class="section apqp-section">')
        html.append('<h2>4. Graphical Analysis</h2>')
        for name, img_bytes in plot_images.items():
            b64 = base64.b64encode(img_bytes).decode('ascii')
            html.append(f'<h3>{_esc(name)}</h3>')
            html.append(f'<img src="data:image/png;base64,{b64}" '
                        f'width="{config.image_width}" alt="{_esc(name)}">')
        html.append('</div>')

    # Section 5: Approval block
    html.append("""
    <div class="section apqp-section">
        <h2>5. Approval</h2>
        <table class="data-table">
            <tr><th>Role</th><th>Name</th><th>Signature</th><th>Date</th></tr>
            <tr><td>Design Engineer</td><td></td><td></td><td></td></tr>
            <tr><td>Quality Engineer</td><td></td><td></td><td></td></tr>
            <tr><td>Manufacturing Engineer</td><td></td><td></td><td></td></tr>
            <tr><td>Customer Representative</td><td></td><td></td><td></td></tr>
        </table>
    </div>
    """)

    html.append(_html_footer())
    return "\n".join(html)


def save_report(html: str, path: str) -> None:
    """Save HTML report to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _html_header(config: ReportConfig, date_str: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(config.title)}</title>
<style>
    body {{
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 40px;
        color: #333;
        background: #fff;
        line-height: 1.6;
    }}
    h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }}
    h2 {{ color: #1976D2; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
    h3 {{ color: #1E88E5; }}
    .cover {{ text-align: center; margin-bottom: 40px; padding: 30px;
              background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
              border-radius: 8px; }}
    .info-table {{ margin: 15px auto; border-collapse: collapse; }}
    .info-table td {{ padding: 5px 15px; text-align: left; }}
    .section {{ margin: 20px 0; padding: 15px; }}
    .apqp-section {{ border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; padding: 20px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    .data-table th {{ background: #1976D2; color: white; padding: 8px 12px;
                      text-align: left; font-weight: 600; }}
    .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
    .data-table tr:hover {{ background: #F5F5F5; }}
    .data-table tr:nth-child(even) {{ background: #FAFAFA; }}
    .summary {{ background: #F5F5F5; padding: 15px; border-radius: 5px;
                font-family: 'Consolas', monospace; font-size: 13px;
                white-space: pre-wrap; overflow-x: auto; }}
    .positive {{ color: #2E7D32; }}
    .negative {{ color: #C62828; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
           margin: 10px 0; }}
    @media print {{
        body {{ margin: 20px; }}
        .section {{ page-break-inside: avoid; }}
        .cover {{ background: #fff; border: 2px solid #1565C0; }}
    }}
</style>
</head>
<body>
"""


def _html_footer() -> str:
    return """
<div style="text-align:center; margin-top:40px; padding:15px; border-top:1px solid #ddd; color:#999; font-size:12px;">
    Generated by 3D Tolerance Stack Analyzer
</div>
</body>
</html>"""
