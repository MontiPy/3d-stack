"""Streamlit GUI for 3D Tolerance Stack Analysis.

Launch with:
    streamlit run tolerance_stack/gui.py
    # or
    python -m streamlit run tolerance_stack/gui.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="3D Tolerance Stack Analyzer",
    page_icon="\U0001F527",
    layout="wide",
)

st.title("3D Tolerance Stack Analyzer")

ALL_DISTRIBUTIONS = [
    "normal", "uniform", "triangular",
    "weibull_right", "weibull_left", "lognormal",
    "rayleigh", "bimodal",
]

tab_stack, tab_linkage, tab_assembly, tab_tools, tab_reports = st.tabs([
    "Tolerance Stack",
    "Linkage",
    "Assembly",
    "DOE / Optimizer",
    "Reports",
])


# ===================================================================
# TAB 1: Tolerance Stack
# ===================================================================

with tab_stack:
    st.header("Linear Tolerance Stack")
    st.markdown("Define a dimension loop with 3D direction vectors. Each contributor adds to or subtracts from the gap.")

    col_def, col_results = st.columns([1, 1])

    with col_def:
        st.subheader("Stack Definition")
        stack_name = st.text_input("Stack name", "My Stack", key="ts_name")

        cd = st.text_input("Closure direction (x,y,z)", "1,0,0", key="ts_cd")
        try:
            closure_dir = tuple(float(x) for x in cd.split(","))
        except ValueError:
            closure_dir = (1.0, 0.0, 0.0)
            st.warning("Invalid closure direction, using (1,0,0)")

        # --- Upload JSON ---
        uploaded = st.file_uploader("Or load from JSON", type=["json"], key="ts_upload")
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read())
                if "contributors" in data:
                    st.session_state["ts_contributors"] = data["contributors"]
                    st.success(f"Loaded {len(data['contributors'])} contributors")
            except Exception as e:
                st.error(f"Failed to load: {e}")

        # --- Contributor table ---
        st.subheader("Contributors")

        if "ts_contributors" not in st.session_state:
            st.session_state["ts_contributors"] = []

        with st.expander("Add a contributor", expanded=len(st.session_state["ts_contributors"]) == 0):
            c_name = st.text_input("Name", key="ts_c_name")
            c1, c2 = st.columns(2)
            with c1:
                c_nom = st.number_input("Nominal", value=10.0, format="%.4f", key="ts_c_nom")
                c_plus = st.number_input("Plus tolerance", value=0.1, min_value=0.0, format="%.4f", key="ts_c_plus")
                c_minus = st.number_input("Minus tolerance", value=0.1, min_value=0.0, format="%.4f", key="ts_c_minus")
            with c2:
                c_dir = st.text_input("Direction (x,y,z)", "1,0,0", key="ts_c_dir")
                c_sign = st.selectbox("Sign", [+1, -1], key="ts_c_sign")
                c_dist = st.selectbox("Distribution", ALL_DISTRIBUTIONS, key="ts_c_dist")
                c_sigma = st.number_input("Sigma", value=3.0, min_value=0.1, key="ts_c_sigma")

            if st.button("Add contributor", key="ts_add"):
                try:
                    direction = [float(x) for x in c_dir.split(",")]
                except ValueError:
                    direction = [1, 0, 0]
                st.session_state["ts_contributors"].append({
                    "name": c_name,
                    "nominal": c_nom,
                    "plus_tol": c_plus,
                    "minus_tol": c_minus,
                    "direction": direction,
                    "sign": c_sign,
                    "distribution": c_dist,
                    "contributor_type": "linear",
                    "sigma": c_sigma,
                })
                st.rerun()

        # Display contributors
        contribs = st.session_state["ts_contributors"]
        if contribs:
            for i, c in enumerate(contribs):
                sign_str = "+" if c["sign"] == 1 else "-"
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.text(f"{sign_str} {c['name']}: {c['nominal']:.4f} +{c['plus_tol']:.4f}/-{c['minus_tol']:.4f}  dir={c['direction']}  dist={c['distribution']}")
                with col_del:
                    if st.button("Remove", key=f"ts_del_{i}"):
                        st.session_state["ts_contributors"].pop(i)
                        st.rerun()

            if st.button("Clear all contributors", key="ts_clear"):
                st.session_state["ts_contributors"] = []
                st.rerun()

        # --- Export JSON ---
        if contribs:
            export_data = {
                "name": stack_name,
                "closure_direction": list(closure_dir),
                "contributors": contribs,
            }
            st.download_button(
                "Download stack JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"{stack_name.replace(' ', '_').lower()}.json",
                mime="application/json",
                key="ts_export",
            )

    with col_results:
        st.subheader("Analysis")

        r1, r2, r3 = st.columns(3)
        with r1:
            ts_sigma = st.number_input("RSS sigma", value=3.0, min_value=0.1, key="ts_sigma")
        with r2:
            ts_mc = st.number_input("MC samples", value=100000, min_value=1000, step=10000, key="ts_mc")
        with r3:
            ts_seed = st.number_input("MC seed", value=42, key="ts_seed")

        ts_methods = st.multiselect("Methods", ["wc", "rss", "mc"], default=["wc", "rss", "mc"], key="ts_methods")

        # --- Spec limits for process capability ---
        with st.expander("Process Capability (Spec Limits)"):
            ts_usl = st.number_input("USL (Upper Spec Limit)", value=0.0, format="%.4f", key="ts_usl")
            ts_lsl = st.number_input("LSL (Lower Spec Limit)", value=0.0, format="%.4f", key="ts_lsl")
            ts_target = st.number_input("Target (optional, 0=midpoint)", value=0.0, format="%.4f", key="ts_target")
            ts_calc_cp = st.checkbox("Calculate Cp/Cpk after MC", value=False, key="ts_calc_cp")

        if st.button("Run Analysis", key="ts_run", type="primary", disabled=len(contribs) == 0):
            from tolerance_stack.models import Contributor, ToleranceStack, Distribution, ContributorType
            from tolerance_stack.analysis import analyze_stack

            stack = ToleranceStack(name=stack_name, closure_direction=closure_dir)
            for c in contribs:
                stack.add(Contributor(
                    name=c["name"],
                    nominal=c["nominal"],
                    plus_tol=c["plus_tol"],
                    minus_tol=c["minus_tol"],
                    direction=tuple(c["direction"]),
                    sign=c["sign"],
                    distribution=Distribution(c["distribution"]),
                    contributor_type=ContributorType(c.get("contributor_type", "linear")),
                    sigma=c.get("sigma", 3.0),
                ))

            results = analyze_stack(
                stack, methods=ts_methods, sigma=ts_sigma,
                mc_samples=int(ts_mc), mc_seed=int(ts_seed),
            )

            for key, result in results.items():
                st.text(result.summary())

            # --- Percent Contribution ---
            best_result = results.get("rss", results.get("wc", results.get("mc")))
            if best_result and best_result.sensitivity:
                from tolerance_stack.statistics import percent_contribution
                sens_list = best_result.sensitivity
                tols = [c["plus_tol"] for c in contribs]
                if len(sens_list) == len(tols):
                    pct = percent_contribution(sens_list, tols)
                    if pct:
                        st.markdown("**Percent Contribution (RSS)**")
                        fig_pc, ax_pc = plt.subplots(figsize=(6, max(3, len(pct) * 0.35)))
                        pct_sorted = sorted(pct, key=lambda x: x[1], reverse=True)
                        names_pc = [p[0] for p in pct_sorted]
                        vals_pc = [p[1] for p in pct_sorted]
                        bars = ax_pc.barh(range(len(names_pc)), vals_pc, color="#FF9800",
                                          edgecolor="black", linewidth=0.5)
                        ax_pc.set_yticks(range(len(names_pc)))
                        ax_pc.set_yticklabels(names_pc)
                        ax_pc.set_xlabel("Contribution (%)")
                        ax_pc.set_title("Percent Contribution")
                        ax_pc.invert_yaxis()
                        for bar, val in zip(bars, vals_pc):
                            ax_pc.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                                       f"{val:.1f}%", va="center", fontsize=8)
                        fig_pc.tight_layout()
                        st.pyplot(fig_pc)
                        plt.close(fig_pc)

            # --- Plots ---
            if "mc" in results and results["mc"].mc_samples is not None:
                mc_result = results["mc"]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(mc_result.mc_samples, bins=80, density=True, alpha=0.7, color="#2196F3",
                        edgecolor="black", linewidth=0.3)
                ax.axvline(mc_result.mc_mean, color="red", linestyle="--", label=f"Mean={mc_result.mc_mean:.4f}")
                for k in [1, 2, 3]:
                    ax.axvline(mc_result.mc_mean + k * mc_result.mc_std, color="orange", linestyle=":", linewidth=0.8)
                    ax.axvline(mc_result.mc_mean - k * mc_result.mc_std, color="orange", linestyle=":", linewidth=0.8)

                # Overlay spec limits if provided
                if ts_calc_cp and (ts_usl != 0 or ts_lsl != 0):
                    if ts_usl != 0:
                        ax.axvline(ts_usl, color="green", linestyle="-", linewidth=2, label=f"USL={ts_usl:.4f}")
                    if ts_lsl != 0:
                        ax.axvline(ts_lsl, color="green", linestyle="-", linewidth=2, label=f"LSL={ts_lsl:.4f}")

                ax.set_xlabel("Gap Value")
                ax.set_ylabel("Density")
                ax.set_title("Monte Carlo Distribution")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # --- Process Capability ---
                if ts_calc_cp and (ts_usl != 0 or ts_lsl != 0):
                    from tolerance_stack.statistics import compute_process_capability
                    usl = ts_usl if ts_usl != 0 else None
                    lsl = ts_lsl if ts_lsl != 0 else None
                    tgt = ts_target if ts_target != 0 else None
                    cp_result = compute_process_capability(
                        mc_result.mc_samples, usl=usl, lsl=lsl, target=tgt
                    )
                    st.markdown("**Process Capability**")
                    st.text(cp_result.summary())

            # Sensitivity chart
            best_result = results.get("wc", results.get("rss", results.get("mc")))
            if best_result and best_result.sensitivity:
                sorted_sens = sorted(best_result.sensitivity, key=lambda x: abs(x[1]), reverse=True)
                names_s = [s[0] for s in sorted_sens]
                vals_s = [s[1] for s in sorted_sens]
                fig2, ax2 = plt.subplots(figsize=(8, max(3, len(names_s) * 0.4)))
                colors = ["#2196F3" if v >= 0 else "#F44336" for v in vals_s]
                ax2.barh(range(len(names_s)), vals_s, color=colors, edgecolor="black", linewidth=0.5)
                ax2.set_yticks(range(len(names_s)))
                ax2.set_yticklabels(names_s)
                ax2.set_xlabel("Sensitivity")
                ax2.set_title("Contributor Sensitivity")
                ax2.invert_yaxis()
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)


# ===================================================================
# TAB 2: Linkage
# ===================================================================

with tab_linkage:
    st.header("3D Kinematic Linkage")
    st.markdown("Define a kinematic chain of joints and links. Tolerances propagate through forward kinematics.")

    col_def2, col_results2 = st.columns([1, 1])

    with col_def2:
        st.subheader("Linkage Definition")
        lk_name = st.text_input("Linkage name", "My Linkage", key="lk_name")

        # --- Upload JSON ---
        lk_uploaded = st.file_uploader("Or load from JSON", type=["json"], key="lk_upload")
        if lk_uploaded is not None:
            try:
                data = json.loads(lk_uploaded.read())
                if "joints" in data:
                    st.session_state["lk_joints"] = data["joints"]
                    st.session_state["lk_links"] = data.get("links", [])
                    st.success(f"Loaded {len(data['joints'])} joints, {len(data.get('links', []))} links")
            except Exception as e:
                st.error(f"Failed to load: {e}")

        # --- Load example ---
        lk_example = st.selectbox("Load example", ["(none)", "two-bar", "robot-arm", "four-bar"], key="lk_ex")
        if lk_example != "(none)" and st.button("Load example", key="lk_load_ex"):
            from tolerance_stack.linkage_examples import (
                create_planar_two_bar, create_spatial_robot_arm, create_four_bar_mechanism,
            )
            builders = {"two-bar": create_planar_two_bar, "robot-arm": create_spatial_robot_arm,
                        "four-bar": create_four_bar_mechanism}
            linkage_obj = builders[lk_example]()
            st.session_state["lk_joints"] = [j.to_dict() for j in linkage_obj.joints]
            st.session_state["lk_links"] = [lk.to_dict() for lk in linkage_obj.links]
            st.session_state["lk_name_val"] = linkage_obj.name
            st.rerun()

        if "lk_joints" not in st.session_state:
            st.session_state["lk_joints"] = []
            st.session_state["lk_links"] = []

        joint_types = ["fixed", "revolute_x", "revolute_y", "revolute_z",
                       "prismatic_x", "prismatic_y", "prismatic_z", "spherical"]

        # Add joint
        with st.expander("Add a joint"):
            j_name = st.text_input("Joint name", key="lk_j_name")
            j_type = st.selectbox("Joint type", joint_types, key="lk_j_type")
            j_nom = st.number_input("Nominal (deg/mm)", value=0.0, format="%.2f", key="lk_j_nom")
            jc1, jc2 = st.columns(2)
            with jc1:
                j_plus = st.number_input("Plus tol", value=0.0, min_value=0.0, format="%.4f", key="lk_j_plus")
            with jc2:
                j_minus = st.number_input("Minus tol", value=0.0, min_value=0.0, format="%.4f", key="lk_j_minus")
            j_dist = st.selectbox("Distribution", ALL_DISTRIBUTIONS, key="lk_j_dist")

            if st.button("Add joint", key="lk_add_j"):
                st.session_state["lk_joints"].append({
                    "name": j_name, "joint_type": j_type,
                    "nominal": j_nom, "plus_tol": j_plus, "minus_tol": j_minus,
                    "distribution": j_dist, "sigma": 3.0,
                })
                st.rerun()

        # Add link
        with st.expander("Add a link"):
            l_name = st.text_input("Link name", key="lk_l_name")
            l_len = st.number_input("Length", value=100.0, format="%.3f", key="lk_l_len")
            l_dir = st.text_input("Direction (x,y,z)", "1,0,0", key="lk_l_dir")
            lc1, lc2 = st.columns(2)
            with lc1:
                l_plus = st.number_input("Plus tol", value=0.0, min_value=0.0, format="%.4f", key="lk_l_plus")
            with lc2:
                l_minus = st.number_input("Minus tol", value=0.0, min_value=0.0, format="%.4f", key="lk_l_minus")
            l_dist = st.selectbox("Distribution", ALL_DISTRIBUTIONS, key="lk_l_dist")

            if st.button("Add link", key="lk_add_l"):
                try:
                    direction = [float(x) for x in l_dir.split(",")]
                except ValueError:
                    direction = [1, 0, 0]
                st.session_state["lk_links"].append({
                    "name": l_name, "length": l_len, "direction": direction,
                    "plus_tol": l_plus, "minus_tol": l_minus,
                    "distribution": l_dist, "sigma": 3.0,
                })
                st.rerun()

        # Display chain
        joints = st.session_state["lk_joints"]
        links = st.session_state["lk_links"]

        if joints or links:
            st.markdown("**Chain:**")
            chain_parts = []
            for i in range(max(len(joints), len(links))):
                if i < len(joints):
                    j = joints[i]
                    tol_str = f" \u00b1{j['plus_tol']}" if j['plus_tol'] > 0 else ""
                    chain_parts.append(f"[{j['name']} ({j['joint_type']} {j['nominal']:.1f}{tol_str})]")
                if i < len(links):
                    lk = links[i]
                    tol_str = f" \u00b1{lk['plus_tol']}" if lk['plus_tol'] > 0 else ""
                    chain_parts.append(f"--- {lk['name']} ({lk['length']:.1f}{tol_str}) ---")
            st.text(" -> ".join(chain_parts[:6]))
            if len(chain_parts) > 6:
                st.text(" -> ".join(chain_parts[6:]))

            if st.button("Clear chain", key="lk_clear"):
                st.session_state["lk_joints"] = []
                st.session_state["lk_links"] = []
                st.rerun()

        # Export
        if joints:
            export_data = {"type": "linkage", "name": lk_name,
                           "joints": joints, "links": links}
            st.download_button(
                "Download linkage JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"{lk_name.replace(' ', '_').lower()}.json",
                mime="application/json", key="lk_export",
            )

    with col_results2:
        st.subheader("Analysis")

        lr1, lr2, lr3 = st.columns(3)
        with lr1:
            lk_sigma = st.number_input("RSS sigma", value=3.0, min_value=0.1, key="lk_sigma")
        with lr2:
            lk_mc = st.number_input("MC samples", value=100000, min_value=1000, step=10000, key="lk_mc")
        with lr3:
            lk_seed = st.number_input("MC seed", value=42, key="lk_seed")

        lk_methods = st.multiselect("Methods", ["wc", "rss", "mc"], default=["wc", "rss", "mc"], key="lk_methods")

        can_run = len(joints) >= 2 and len(links) == len(joints) - 1
        if not can_run and (joints or links):
            st.info(f"Need at least 2 joints with n-1 links between them. Currently: {len(joints)} joints, {len(links)} links.")

        if st.button("Run Analysis", key="lk_run", type="primary", disabled=not can_run):
            from tolerance_stack.linkage import Joint, JointType, Link, Linkage
            from tolerance_stack.linkage_analysis import analyze_linkage
            from tolerance_stack.models import Distribution

            linkage = Linkage(name=lk_name)
            for i, jd in enumerate(joints):
                nom = jd["nominal"]
                if isinstance(nom, list):
                    nom = tuple(nom)
                linkage.add_joint(Joint(
                    jd["name"], JointType(jd["joint_type"]),
                    nominal=nom, plus_tol=jd["plus_tol"], minus_tol=jd["minus_tol"],
                    distribution=Distribution(jd.get("distribution", "normal")),
                    sigma=jd.get("sigma", 3.0),
                ))
                if i < len(links):
                    ld = links[i]
                    linkage.add_link(Link(
                        ld["name"], length=ld["length"],
                        plus_tol=ld["plus_tol"], minus_tol=ld["minus_tol"],
                        direction=tuple(ld["direction"]),
                        distribution=Distribution(ld.get("distribution", "normal")),
                        sigma=ld.get("sigma", 3.0),
                    ))

            results = analyze_linkage(
                linkage, methods=lk_methods, sigma=lk_sigma,
                mc_samples=int(lk_mc), mc_seed=int(lk_seed),
            )

            for key, result in results.items():
                st.text(result.summary())

            # 3D Interactive Plotly visualization
            try:
                from tolerance_stack.visualization import visualize_linkage as viz_lk, PLOTLY_AVAILABLE as _PL
                mc_result_3d = results.get("mc")
                mc_s = mc_result_3d.mc_samples if mc_result_3d and mc_result_3d.mc_samples is not None else None
                if _PL:
                    fig_pl = viz_lk(linkage, mc_samples=mc_s)
                    if fig_pl is not None:
                        st.plotly_chart(fig_pl, use_container_width=True)
            except Exception:
                pass

            # 3D linkage plot (matplotlib fallback)
            positions = linkage.all_joint_positions()
            end_pos = linkage.end_effector_position()
            xs = [p[0] for _, p in positions]
            ys = [p[1] for _, p in positions]
            zs = [p[2] for _, p in positions]

            fig3d = plt.figure(figsize=(8, 6))
            ax3d = fig3d.add_subplot(111, projection="3d")
            ax3d.plot(xs, ys, zs, "o-", color="#2196F3", linewidth=2.5, markersize=8, label="Nominal chain")
            ax3d.scatter(*end_pos, color="red", s=100, zorder=5, label="End-effector")
            for name, pos in positions:
                ax3d.text(pos[0], pos[1], pos[2], f"  {name}", fontsize=7)

            # MC scatter
            mc_result = results.get("mc")
            if mc_result and mc_result.mc_samples is not None:
                samples = mc_result.mc_samples
                n_plot = min(2000, len(samples))
                idx = np.random.default_rng(0).choice(len(samples), n_plot, replace=False)
                ax3d.scatter(samples[idx, 0], samples[idx, 1], samples[idx, 2],
                             alpha=0.08, s=3, color="orange", label="MC samples")

            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_title(f"{lk_name}")
            ax3d.legend(fontsize=7)
            fig3d.tight_layout()
            st.pyplot(fig3d)
            plt.close(fig3d)

            # Sensitivity chart
            best = results.get("wc", results.get("rss", results.get("mc")))
            if best and best.sensitivity:
                names_s = [s[0] for s in best.sensitivity]
                mag = [np.linalg.norm(s[1]) for s in best.sensitivity]
                order = np.argsort(mag)[::-1]

                fig_s, ax_s = plt.subplots(figsize=(8, max(3, len(names_s) * 0.5)))
                y = np.arange(len(names_s))
                bar_h = 0.25
                dx_vals = [best.sensitivity[i][1][0] for i in order]
                dy_vals = [best.sensitivity[i][1][1] for i in order]
                dz_vals = [best.sensitivity[i][1][2] for i in order]
                ax_s.barh(y - bar_h, dx_vals, height=bar_h, color="#F44336", label="dX")
                ax_s.barh(y, dy_vals, height=bar_h, color="#4CAF50", label="dY")
                ax_s.barh(y + bar_h, dz_vals, height=bar_h, color="#2196F3", label="dZ")
                ax_s.set_yticks(y)
                ax_s.set_yticklabels([names_s[i] for i in order])
                ax_s.set_xlabel("Sensitivity")
                ax_s.set_title("Parameter Sensitivity (XYZ)")
                ax_s.legend()
                ax_s.invert_yaxis()
                fig_s.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)


# ===================================================================
# TAB 3: Assembly
# ===================================================================

with tab_assembly:
    st.header("3D Rigid Body Assembly")
    st.markdown("Define bodies with geometric features and mating conditions. Measure distances or angles between features.")

    col_def3, col_results3 = st.columns([1, 1])

    with col_def3:
        st.subheader("Assembly Definition")
        assy_name = st.text_input("Assembly name", "My Assembly", key="assy_name")

        # --- Upload JSON ---
        assy_uploaded = st.file_uploader("Or load from JSON", type=["json"], key="assy_upload")
        if assy_uploaded is not None:
            try:
                data = json.loads(assy_uploaded.read())
                if "bodies" in data:
                    st.session_state["assy_data"] = data
                    st.success(f"Loaded assembly: {data.get('name', 'unknown')}")
            except Exception as e:
                st.error(f"Failed to load: {e}")

        # --- STEP file import ---
        with st.expander("Import from STEP file"):
            step_uploaded = st.file_uploader("Upload STEP file (.stp/.step)", type=["stp", "step"], key="step_upload")
            if step_uploaded is not None and st.button("Import STEP", key="step_import_btn"):
                from tolerance_stack.step_import import import_step
                with tempfile.NamedTemporaryFile(suffix=".stp", delete=False, mode="wb") as tf:
                    tf.write(step_uploaded.read())
                    tf.flush()
                    step_result = import_step(tf.name, assembly_name=assy_name)
                st.text(step_result.summary())
                if step_result.assembly:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as jf:
                        step_result.assembly.save(jf.name)
                        with open(jf.name) as rf:
                            st.session_state["assy_data"] = json.load(rf)
                    st.success(f"Assembly imported: {len(step_result.assembly.bodies)} bodies")
                    st.rerun()
                else:
                    st.warning("Could not construct assembly from STEP file.")

        # --- Load example ---
        assy_example = st.selectbox("Load example",
                                     ["(none)", "pin-in-hole", "stacked-plates", "bracket"],
                                     key="assy_ex")
        if assy_example != "(none)" and st.button("Load example", key="assy_load_ex"):
            from tolerance_stack.assembly_examples import (
                create_pin_in_hole_assembly, create_stacked_plates_assembly,
                create_bracket_assembly,
            )
            builders = {"pin-in-hole": create_pin_in_hole_assembly,
                        "stacked-plates": create_stacked_plates_assembly,
                        "bracket": create_bracket_assembly}
            assy_obj = builders[assy_example]()
            # Serialize to session state via save/load
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                assy_obj.save(f.name)
                with open(f.name) as rf:
                    st.session_state["assy_data"] = json.load(rf)
            st.rerun()

        if "assy_data" not in st.session_state:
            st.session_state["assy_data"] = None

        assy_data = st.session_state["assy_data"]

        if assy_data:
            st.markdown("**Bodies:**")
            for bd in assy_data.get("bodies", []):
                with st.expander(f"Body: {bd['name']}"):
                    origin = bd.get("placement_origin", [0, 0, 0])
                    rot = bd.get("placement_rotation", [0, 0, 0])
                    st.text(f"  Placement: origin={origin}, rotation={rot}")
                    for feat in bd.get("features", []):
                        tol_parts = []
                        if feat.get("position_tol", 0) > 0:
                            tol_parts.append(f"pos_tol={feat['position_tol']}")
                        if feat.get("orientation_tol", 0) > 0:
                            tol_parts.append(f"orient_tol={feat['orientation_tol']}")
                        tol_str = f"  [{', '.join(tol_parts)}]" if tol_parts else ""
                        st.text(f"  {feat['feature_type']:10s} {feat['name']}: origin={feat.get('origin', [0,0,0])}, dir={feat.get('direction', [0,0,1])}{tol_str}")

                        # Show GD&T if present
                        fcfs = feat.get("feature_control_frames", [])
                        if fcfs:
                            for fcf in fcfs:
                                datum_str = f"  datums: {fcf.get('datum_refs', [])}" if fcf.get('datum_refs') else ""
                                mc_str = f"  [{fcf.get('material_condition', 'NONE')}]" if fcf.get('material_condition', 'NONE') != 'NONE' else ""
                                st.text(f"    GD&T: {fcf.get('gdt_type', '?')} = {fcf.get('tolerance_value', 0):.4f}{mc_str}{datum_str}")

            if assy_data.get("mates"):
                st.markdown("**Mates:**")
                for m in assy_data["mates"]:
                    tol_str = f"  dist_tol={m['distance_tol']}" if m.get("distance_tol", 0) > 0 else ""
                    st.text(f"  {m['name']}: {m['body_a']}.{m['feature_a']} <-> {m['body_b']}.{m['feature_b']} ({m['mate_type']}){tol_str}")

            if assy_data.get("measurement"):
                m = assy_data["measurement"]
                st.markdown("**Measurement:**")
                st.text(f"  {m['name']}: {m['body_a']}.{m['feature_a']} -> {m['body_b']}.{m['feature_b']} ({m['measurement_type']})")

            # Export
            st.download_button(
                "Download assembly JSON",
                data=json.dumps(assy_data, indent=2),
                file_name=f"{assy_name.replace(' ', '_').lower()}.json",
                mime="application/json", key="assy_export",
            )
        else:
            st.info("Load an example, upload a JSON file, or import a STEP file to get started.")

    with col_results3:
        st.subheader("Analysis")

        ar1, ar2, ar3 = st.columns(3)
        with ar1:
            assy_sigma = st.number_input("RSS sigma", value=3.0, min_value=0.1, key="assy_sigma")
        with ar2:
            assy_mc = st.number_input("MC samples", value=100000, min_value=1000, step=10000, key="assy_mc")
        with ar3:
            assy_seed = st.number_input("MC seed", value=42, key="assy_seed")

        assy_methods = st.multiselect("Methods", ["wc", "rss", "mc"], default=["wc", "rss", "mc"], key="assy_methods")

        # --- Spec limits for assembly process capability ---
        with st.expander("Process Capability (Spec Limits)"):
            assy_usl = st.number_input("USL", value=0.0, format="%.4f", key="assy_usl")
            assy_lsl = st.number_input("LSL", value=0.0, format="%.4f", key="assy_lsl")
            assy_calc_cp = st.checkbox("Calculate Cp/Cpk after MC", value=False, key="assy_calc_cp")

        can_run_assy = assy_data is not None and assy_data.get("measurement") is not None
        if assy_data and not can_run_assy:
            st.info("Assembly needs a measurement defined to run analysis.")

        # 3D Visualization (before analysis, so user can always see the model)
        if assy_data is not None:
            try:
                from tolerance_stack.visualization import visualize_assembly, PLOTLY_AVAILABLE
                if PLOTLY_AVAILABLE:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                        json.dump(assy_data, f)
                        f.flush()
                        from tolerance_stack.assembly import Assembly as _Assy
                        _viz_assy = _Assy.load(f.name)
                    fig_3d = visualize_assembly(_viz_assy)
                    if fig_3d is not None:
                        st.plotly_chart(fig_3d, use_container_width=True)
            except Exception:
                pass  # Graceful fallback if visualization fails

        if st.button("Run Analysis", key="assy_run", type="primary", disabled=not can_run_assy):
            from tolerance_stack.assembly import Assembly
            from tolerance_stack.assembly_analysis import analyze_assembly

            # Save to temp file and reload (cleanest way to reconstruct)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump(assy_data, f)
                f.flush()
                assy_obj = Assembly.load(f.name)

            results = analyze_assembly(
                assy_obj, methods=assy_methods, sigma=assy_sigma,
                mc_samples=int(assy_mc), mc_seed=int(assy_seed),
            )

            for key, result in results.items():
                st.text(result.summary())

            # --- DOF Status ---
            with st.expander("DOF Status", expanded=False):
                from tolerance_stack.assembly_process import compute_dof_status
                dof = compute_dof_status(assy_obj)
                st.text(dof.summary())

            # MC histogram
            mc_result = results.get("mc")
            if mc_result and mc_result.mc_samples is not None:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(mc_result.mc_samples, bins=80, density=True, alpha=0.7,
                        color="#2196F3", edgecolor="black", linewidth=0.3)
                ax.axvline(mc_result.mc_mean, color="red", linestyle="--",
                           label=f"Mean={mc_result.mc_mean:.4f}")
                for k in [1, 2, 3]:
                    ax.axvline(mc_result.mc_mean + k * mc_result.mc_std, color="orange",
                               linestyle=":", linewidth=0.8)
                    ax.axvline(mc_result.mc_mean - k * mc_result.mc_std, color="orange",
                               linestyle=":", linewidth=0.8)

                if assy_calc_cp and (assy_usl != 0 or assy_lsl != 0):
                    if assy_usl != 0:
                        ax.axvline(assy_usl, color="green", linestyle="-", linewidth=2, label=f"USL={assy_usl:.4f}")
                    if assy_lsl != 0:
                        ax.axvline(assy_lsl, color="green", linestyle="-", linewidth=2, label=f"LSL={assy_lsl:.4f}")

                ax.set_xlabel("Measurement Value")
                ax.set_ylabel("Density")
                ax.set_title("Monte Carlo Distribution")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Process Capability
                if assy_calc_cp and (assy_usl != 0 or assy_lsl != 0):
                    from tolerance_stack.statistics import compute_process_capability
                    usl = assy_usl if assy_usl != 0 else None
                    lsl = assy_lsl if assy_lsl != 0 else None
                    cp_result = compute_process_capability(mc_result.mc_samples, usl=usl, lsl=lsl)
                    st.markdown("**Process Capability**")
                    st.text(cp_result.summary())

            # Sensitivity chart
            best = results.get("wc", results.get("rss", results.get("mc")))
            if best and best.sensitivity:
                sorted_sens = sorted(best.sensitivity, key=lambda x: abs(x[1]), reverse=True)
                # Filter out zero-sensitivity parameters
                sorted_sens = [(n, s) for n, s in sorted_sens if abs(s) > 1e-10]
                if sorted_sens:
                    names_s = [s[0] for s in sorted_sens]
                    vals_s = [s[1] for s in sorted_sens]
                    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(names_s) * 0.4)))
                    colors = ["#2196F3" if v >= 0 else "#F44336" for v in vals_s]
                    ax2.barh(range(len(names_s)), vals_s, color=colors,
                             edgecolor="black", linewidth=0.5)
                    ax2.set_yticks(range(len(names_s)))
                    ax2.set_yticklabels(names_s)
                    ax2.set_xlabel("Sensitivity")
                    ax2.set_title("Parameter Sensitivity")
                    ax2.invert_yaxis()
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)


# ===================================================================
# TAB 4: DOE / Optimizer
# ===================================================================

with tab_tools:
    st.header("Design of Experiments & Tolerance Optimization")

    tool_mode = st.radio("Tool", [
        "HLM Sensitivity", "Full Factorial DOE",
        "Latin Hypercube DOE", "Response Surface (RSM)",
        "Sobol' Sensitivity", "Tolerance Optimizer",
    ], horizontal=True, key="tool_mode")

    # --- HLM Sensitivity ---
    if tool_mode == "HLM Sensitivity":
        st.subheader("High-Low-Median Sensitivity Analysis")
        st.markdown("Tests each factor at its high, low, and nominal values while holding all others at nominal.")

        if "hlm_factors" not in st.session_state:
            st.session_state["hlm_factors"] = []

        with st.expander("Add a factor", expanded=len(st.session_state["hlm_factors"]) == 0):
            hf_name = st.text_input("Factor name", key="hlm_f_name")
            hf_low = st.number_input("Low level", value=-1.0, format="%.4f", key="hlm_f_low")
            hf_nom = st.number_input("Nominal", value=0.0, format="%.4f", key="hlm_f_nom")
            hf_high = st.number_input("High level", value=1.0, format="%.4f", key="hlm_f_high")

            if st.button("Add factor", key="hlm_add_f"):
                st.session_state["hlm_factors"].append({
                    "name": hf_name, "low": hf_low, "nominal": hf_nom, "high": hf_high,
                })
                st.rerun()

        factors_hlm = st.session_state["hlm_factors"]
        if factors_hlm:
            for i, f in enumerate(factors_hlm):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.text(f"  {f['name']}: low={f['low']}, nom={f['nominal']}, high={f['high']}")
                with fd:
                    if st.button("X", key=f"hlm_del_{i}"):
                        st.session_state["hlm_factors"].pop(i)
                        st.rerun()

        st.markdown("**Response function:** `sum(factor_value * weight)` (interactive demo)")
        hlm_weights_str = st.text_input(
            "Weights (comma-separated, one per factor)",
            value=",".join(["1.0"] * max(1, len(factors_hlm))),
            key="hlm_weights",
        )

        if st.button("Run HLM Analysis", key="hlm_run", type="primary", disabled=len(factors_hlm) < 1):
            from tolerance_stack.optimizer import hlm_sensitivity, DOEFactor

            try:
                weights = [float(w.strip()) for w in hlm_weights_str.split(",")]
            except ValueError:
                weights = [1.0] * len(factors_hlm)

            doe_factors = []
            for fi, fd in enumerate(factors_hlm):
                doe_factors.append(DOEFactor(
                    name=fd["name"],
                    levels=[fd["low"], fd["nominal"], fd["high"]],
                    nominal=fd["nominal"],
                ))

            def evaluate_hlm(inputs):
                total = 0.0
                for fi, fd in enumerate(factors_hlm):
                    w = weights[fi] if fi < len(weights) else 1.0
                    total += inputs[fd["name"]] * w
                return total

            result = hlm_sensitivity(evaluate_hlm, doe_factors)
            st.text(result.summary())

            # Main effects chart
            if result.main_effects:
                me_sorted = sorted(result.main_effects.items(), key=lambda x: abs(x[1]), reverse=True)
                names_me = [m[0] for m in me_sorted]
                vals_me = [m[1] for m in me_sorted]
                fig_me, ax_me = plt.subplots(figsize=(8, max(3, len(names_me) * 0.5)))
                ax_me.barh(range(len(names_me)), vals_me, color="#9C27B0",
                           edgecolor="black", linewidth=0.5)
                ax_me.set_yticks(range(len(names_me)))
                ax_me.set_yticklabels(names_me)
                ax_me.set_xlabel("Main Effect (range)")
                ax_me.set_title("HLM Main Effects")
                ax_me.invert_yaxis()
                fig_me.tight_layout()
                st.pyplot(fig_me)
                plt.close(fig_me)

    # --- Full Factorial DOE ---
    elif tool_mode == "Full Factorial DOE":
        st.subheader("Full Factorial DOE with Interaction Analysis")
        st.markdown("Tests all combinations of factor levels. Computes main effects and two-factor interactions.")

        if "doe_factors" not in st.session_state:
            st.session_state["doe_factors"] = []

        with st.expander("Add a factor", expanded=len(st.session_state["doe_factors"]) == 0):
            df_name = st.text_input("Factor name", key="doe_f_name")
            df_low = st.number_input("Low level", value=-1.0, format="%.4f", key="doe_f_low")
            df_high = st.number_input("High level", value=1.0, format="%.4f", key="doe_f_high")
            df_nom = st.number_input("Nominal", value=0.0, format="%.4f", key="doe_f_nom")

            if st.button("Add factor", key="doe_add_f"):
                st.session_state["doe_factors"].append({
                    "name": df_name, "low": df_low, "high": df_high, "nominal": df_nom,
                })
                st.rerun()

        factors_doe = st.session_state["doe_factors"]
        if factors_doe:
            n_runs_est = 2 ** len(factors_doe)
            st.text(f"Factors: {len(factors_doe)}, Estimated runs: {n_runs_est}")
            for i, f in enumerate(factors_doe):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.text(f"  {f['name']}: [{f['low']}, {f['high']}], nom={f['nominal']}")
                with fd:
                    if st.button("X", key=f"doe_del_{i}"):
                        st.session_state["doe_factors"].pop(i)
                        st.rerun()

        st.markdown("**Response function:** `sum(factor_value * weight) + interactions`")
        doe_weights_str = st.text_input(
            "Weights (comma-separated, one per factor)",
            value=",".join(["1.0"] * max(1, len(factors_doe))),
            key="doe_weights",
        )
        doe_interact = st.checkbox("Add product interaction term (f1*f2)", value=False, key="doe_interact")

        if st.button("Run Full Factorial DOE", key="doe_run", type="primary", disabled=len(factors_doe) < 2):
            from tolerance_stack.optimizer import full_factorial_doe, DOEFactor

            try:
                weights = [float(w.strip()) for w in doe_weights_str.split(",")]
            except ValueError:
                weights = [1.0] * len(factors_doe)

            doe_factor_objs = []
            for fd in factors_doe:
                doe_factor_objs.append(DOEFactor(
                    name=fd["name"],
                    levels=[fd["low"], fd["high"]],
                    nominal=fd["nominal"],
                ))

            def evaluate_doe(inputs):
                total = 0.0
                factor_names = [fd["name"] for fd in factors_doe]
                for fi, fn in enumerate(factor_names):
                    w = weights[fi] if fi < len(weights) else 1.0
                    total += inputs[fn] * w
                if doe_interact and len(factor_names) >= 2:
                    total += inputs[factor_names[0]] * inputs[factor_names[1]]
                return total

            result = full_factorial_doe(evaluate_doe, doe_factor_objs)
            st.text(result.summary())

            # Main effects chart
            if result.main_effects:
                me_sorted = sorted(result.main_effects.items(), key=lambda x: abs(x[1]), reverse=True)
                names_me = [m[0] for m in me_sorted]
                vals_me = [m[1] for m in me_sorted]
                fig_me, ax_me = plt.subplots(figsize=(8, max(3, len(names_me) * 0.5)))
                colors = ["#4CAF50" if v >= 0 else "#F44336" for v in vals_me]
                ax_me.barh(range(len(names_me)), vals_me, color=colors,
                           edgecolor="black", linewidth=0.5)
                ax_me.set_yticks(range(len(names_me)))
                ax_me.set_yticklabels(names_me)
                ax_me.set_xlabel("Main Effect")
                ax_me.set_title("DOE Main Effects")
                ax_me.invert_yaxis()
                fig_me.tight_layout()
                st.pyplot(fig_me)
                plt.close(fig_me)

            # Interactions chart
            if result.interactions:
                int_sorted = sorted(result.interactions.items(), key=lambda x: abs(x[1]), reverse=True)
                int_sorted = int_sorted[:10]
                names_int = [f"{a} x {b}" for (a, b), _ in int_sorted]
                vals_int = [v for _, v in int_sorted]
                fig_int, ax_int = plt.subplots(figsize=(8, max(3, len(names_int) * 0.5)))
                ax_int.barh(range(len(names_int)), vals_int, color="#FF9800",
                            edgecolor="black", linewidth=0.5)
                ax_int.set_yticks(range(len(names_int)))
                ax_int.set_yticklabels(names_int)
                ax_int.set_xlabel("Interaction Effect")
                ax_int.set_title("Two-Factor Interactions")
                ax_int.invert_yaxis()
                fig_int.tight_layout()
                st.pyplot(fig_int)
                plt.close(fig_int)

    # --- Latin Hypercube DOE ---
    elif tool_mode == "Latin Hypercube DOE":
        st.subheader("Latin Hypercube Sampling DOE")
        st.markdown("Space-filling sampling with better coverage than random, far fewer runs than full factorial.")

        if "lhs_factors" not in st.session_state:
            st.session_state["lhs_factors"] = []

        with st.expander("Add a factor", expanded=len(st.session_state["lhs_factors"]) == 0):
            lf_name = st.text_input("Factor name", key="lhs_f_name")
            lf_low = st.number_input("Low level", value=-1.0, format="%.4f", key="lhs_f_low")
            lf_high = st.number_input("High level", value=1.0, format="%.4f", key="lhs_f_high")
            lf_nom = st.number_input("Nominal", value=0.0, format="%.4f", key="lhs_f_nom")

            if st.button("Add factor", key="lhs_add_f"):
                st.session_state["lhs_factors"].append({
                    "name": lf_name, "low": lf_low, "high": lf_high, "nominal": lf_nom,
                })
                st.rerun()

        factors_lhs = st.session_state["lhs_factors"]
        if factors_lhs:
            for i, f in enumerate(factors_lhs):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.text(f"  {f['name']}: [{f['low']}, {f['high']}], nom={f['nominal']}")
                with fd:
                    if st.button("X", key=f"lhs_del_{i}"):
                        st.session_state["lhs_factors"].pop(i)
                        st.rerun()

        lhs_n = st.number_input("Number of samples", value=100, min_value=10, key="lhs_n")
        lhs_weights_str = st.text_input(
            "Weights (comma-separated)",
            value=",".join(["1.0"] * max(1, len(factors_lhs))),
            key="lhs_weights",
        )

        if st.button("Run LHS DOE", key="lhs_run", type="primary", disabled=len(factors_lhs) < 1):
            from tolerance_stack.optimizer import latin_hypercube_doe, DOEFactor

            try:
                weights = [float(w.strip()) for w in lhs_weights_str.split(",")]
            except ValueError:
                weights = [1.0] * len(factors_lhs)

            lhs_factor_objs = [DOEFactor(name=f["name"], levels=[f["low"], f["high"]],
                                          nominal=f["nominal"]) for f in factors_lhs]

            def evaluate_lhs(inputs):
                return sum(inputs[f["name"]] * (weights[i] if i < len(weights) else 1.0)
                           for i, f in enumerate(factors_lhs))

            result = latin_hypercube_doe(evaluate_lhs, lhs_factor_objs, n_samples=int(lhs_n))
            st.text(result.summary())

            if result.main_effects:
                me_sorted = sorted(result.main_effects.items(), key=lambda x: abs(x[1]), reverse=True)
                names_me = [m[0] for m in me_sorted]
                vals_me = [m[1] for m in me_sorted]
                fig_me, ax_me = plt.subplots(figsize=(8, max(3, len(names_me) * 0.5)))
                ax_me.barh(range(len(names_me)), vals_me, color="#00BCD4",
                           edgecolor="black", linewidth=0.5)
                ax_me.set_yticks(range(len(names_me)))
                ax_me.set_yticklabels(names_me)
                ax_me.set_xlabel("Correlation-based Main Effect")
                ax_me.set_title("LHS Main Effects")
                ax_me.invert_yaxis()
                fig_me.tight_layout()
                st.pyplot(fig_me)
                plt.close(fig_me)

    # --- Response Surface Methodology ---
    elif tool_mode == "Response Surface (RSM)":
        st.subheader("Response Surface Methodology")
        st.markdown("Fits a second-order polynomial model. Identifies main effects, quadratic effects, and interactions.")

        if "rsm_factors" not in st.session_state:
            st.session_state["rsm_factors"] = []

        with st.expander("Add a factor", expanded=len(st.session_state["rsm_factors"]) == 0):
            rf_name = st.text_input("Factor name", key="rsm_f_name")
            rf_low = st.number_input("Low level", value=-1.0, format="%.4f", key="rsm_f_low")
            rf_high = st.number_input("High level", value=1.0, format="%.4f", key="rsm_f_high")
            rf_nom = st.number_input("Nominal", value=0.0, format="%.4f", key="rsm_f_nom")

            if st.button("Add factor", key="rsm_add_f"):
                st.session_state["rsm_factors"].append({
                    "name": rf_name, "low": rf_low, "high": rf_high, "nominal": rf_nom,
                })
                st.rerun()

        factors_rsm = st.session_state["rsm_factors"]
        if factors_rsm:
            for i, f in enumerate(factors_rsm):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.text(f"  {f['name']}: [{f['low']}, {f['high']}], nom={f['nominal']}")
                with fd:
                    if st.button("X", key=f"rsm_del_{i}"):
                        st.session_state["rsm_factors"].pop(i)
                        st.rerun()

        rsm_weights_str = st.text_input(
            "Weights (comma-separated)",
            value=",".join(["1.0"] * max(1, len(factors_rsm))),
            key="rsm_weights",
        )
        rsm_quad = st.checkbox("Include x*y interaction", value=True, key="rsm_interact_term")

        if st.button("Run RSM Analysis", key="rsm_run", type="primary", disabled=len(factors_rsm) < 2):
            from tolerance_stack.optimizer import response_surface_doe, DOEFactor

            try:
                weights = [float(w.strip()) for w in rsm_weights_str.split(",")]
            except ValueError:
                weights = [1.0] * len(factors_rsm)

            rsm_factor_objs = [DOEFactor(name=f["name"], levels=[f["low"], f["high"]],
                                          nominal=f["nominal"]) for f in factors_rsm]

            def evaluate_rsm(inputs):
                total = sum(inputs[f["name"]] * (weights[i] if i < len(weights) else 1.0)
                            for i, f in enumerate(factors_rsm))
                if rsm_quad and len(factors_rsm) >= 2:
                    total += inputs[factors_rsm[0]["name"]] * inputs[factors_rsm[1]["name"]]
                return total

            result = response_surface_doe(evaluate_rsm, rsm_factor_objs)
            st.text(result.summary())

            # R-squared metric
            st.metric("R-squared", f"{result.r_squared:.4f}")

    # --- Sobol' Sensitivity ---
    elif tool_mode == "Sobol' Sensitivity":
        st.subheader("Sobol' Global Sensitivity Analysis")
        st.markdown("Computes first-order (S_i) and total-effect (S_Ti) Sobol' indices for variance decomposition.")

        if "sobol_factors" not in st.session_state:
            st.session_state["sobol_factors"] = []

        with st.expander("Add a factor", expanded=len(st.session_state["sobol_factors"]) == 0):
            sf_name = st.text_input("Factor name", key="sobol_f_name")
            sf_low = st.number_input("Low level", value=-1.0, format="%.4f", key="sobol_f_low")
            sf_high = st.number_input("High level", value=1.0, format="%.4f", key="sobol_f_high")
            sf_nom = st.number_input("Nominal", value=0.0, format="%.4f", key="sobol_f_nom")

            if st.button("Add factor", key="sobol_add_f"):
                st.session_state["sobol_factors"].append({
                    "name": sf_name, "low": sf_low, "high": sf_high, "nominal": sf_nom,
                })
                st.rerun()

        factors_sobol = st.session_state["sobol_factors"]
        if factors_sobol:
            for i, f in enumerate(factors_sobol):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.text(f"  {f['name']}: [{f['low']}, {f['high']}]")
                with fd:
                    if st.button("X", key=f"sobol_del_{i}"):
                        st.session_state["sobol_factors"].pop(i)
                        st.rerun()

        sobol_n = st.number_input("Base samples", value=1024, min_value=128, step=256, key="sobol_n")
        sobol_weights_str = st.text_input(
            "Weights (comma-separated)",
            value=",".join(["1.0"] * max(1, len(factors_sobol))),
            key="sobol_weights",
        )
        sobol_interact = st.checkbox("Add x1*x2 interaction", value=False, key="sobol_interact_term")

        if st.button("Run Sobol' Analysis", key="sobol_run", type="primary", disabled=len(factors_sobol) < 2):
            from tolerance_stack.optimizer import sobol_sensitivity, DOEFactor

            try:
                weights = [float(w.strip()) for w in sobol_weights_str.split(",")]
            except ValueError:
                weights = [1.0] * len(factors_sobol)

            sobol_factor_objs = [DOEFactor(name=f["name"], levels=[f["low"], f["high"]],
                                            nominal=f["nominal"]) for f in factors_sobol]

            def evaluate_sobol(inputs):
                total = sum(inputs[f["name"]] * (weights[i] if i < len(weights) else 1.0)
                            for i, f in enumerate(factors_sobol))
                if sobol_interact and len(factors_sobol) >= 2:
                    total += inputs[factors_sobol[0]["name"]] * inputs[factors_sobol[1]["name"]]
                return total

            result = sobol_sensitivity(evaluate_sobol, sobol_factor_objs, n_samples=int(sobol_n))
            st.text(result.summary())

            # Sobol' indices chart
            names_s = sorted(result.factor_names,
                              key=lambda n: result.total_order.get(n, 0), reverse=True)
            si_vals = [result.first_order.get(n, 0) for n in names_s]
            sti_vals = [result.total_order.get(n, 0) for n in names_s]

            fig_sob, ax_sob = plt.subplots(figsize=(8, max(3, len(names_s) * 0.5)))
            y = np.arange(len(names_s))
            ax_sob.barh(y - 0.15, si_vals, height=0.3, color="#2196F3", label="First-order S_i")
            ax_sob.barh(y + 0.15, sti_vals, height=0.3, color="#F44336", label="Total-effect S_Ti")
            ax_sob.set_yticks(y)
            ax_sob.set_yticklabels(names_s)
            ax_sob.set_xlabel("Sensitivity Index")
            ax_sob.set_title("Sobol' Indices")
            ax_sob.legend()
            ax_sob.invert_yaxis()
            ax_sob.set_xlim(0, 1.1)
            fig_sob.tight_layout()
            st.pyplot(fig_sob)
            plt.close(fig_sob)

    # --- Tolerance Optimizer ---
    elif tool_mode == "Tolerance Optimizer":
        st.subheader("Tolerance Optimization")
        st.markdown("Optimize tolerance allocations to minimize cost while meeting a variation target.")

        if "opt_params" not in st.session_state:
            st.session_state["opt_params"] = []

        with st.expander("Add a parameter", expanded=len(st.session_state["opt_params"]) == 0):
            op_name = st.text_input("Parameter name", key="opt_p_name")
            op_sens = st.number_input("Sensitivity", value=1.0, format="%.4f", key="opt_p_sens")
            op_tol = st.number_input("Current half-tolerance", value=0.1, min_value=0.001, format="%.4f", key="opt_p_tol")

            if st.button("Add parameter", key="opt_add_p"):
                st.session_state["opt_params"].append({
                    "name": op_name, "sensitivity": op_sens, "tolerance": op_tol,
                })
                st.rerun()

        opt_params = st.session_state["opt_params"]
        if opt_params:
            for i, p in enumerate(opt_params):
                pc, pd = st.columns([5, 1])
                with pc:
                    st.text(f"  {p['name']}: sens={p['sensitivity']}, tol={p['tolerance']}")
                with pd:
                    if st.button("X", key=f"opt_del_{i}"):
                        st.session_state["opt_params"].pop(i)
                        st.rerun()

        opt_target = st.number_input("Target variation (RSS)", value=0.05, min_value=0.001, format="%.4f", key="opt_target")
        opt_max_iter = st.number_input("Max iterations", value=100, min_value=10, key="opt_max_iter")

        if st.button("Optimize", key="opt_run", type="primary", disabled=len(opt_params) < 1):
            from tolerance_stack.optimizer import optimize_tolerances

            sens = [(p["name"], p["sensitivity"]) for p in opt_params]
            tols = {p["name"]: p["tolerance"] for p in opt_params}

            result = optimize_tolerances(
                sens, tols,
                target_variation=opt_target,
                max_iterations=int(opt_max_iter),
            )
            st.text(result.summary())

            # Before/after chart
            names_opt = sorted(result.original_tolerances.keys())
            orig_vals = [result.original_tolerances[n] for n in names_opt]
            opt_vals = [result.optimized_tolerances.get(n, result.original_tolerances[n]) for n in names_opt]

            fig_opt, ax_opt = plt.subplots(figsize=(8, max(3, len(names_opt) * 0.5)))
            y = np.arange(len(names_opt))
            ax_opt.barh(y - 0.2, orig_vals, height=0.35, color="#BDBDBD", label="Original", edgecolor="black", linewidth=0.5)
            ax_opt.barh(y + 0.2, opt_vals, height=0.35, color="#2196F3", label="Optimized", edgecolor="black", linewidth=0.5)
            ax_opt.set_yticks(y)
            ax_opt.set_yticklabels(names_opt)
            ax_opt.set_xlabel("Half-Tolerance")
            ax_opt.set_title("Tolerance Optimization Results")
            ax_opt.legend()
            ax_opt.invert_yaxis()
            fig_opt.tight_layout()
            st.pyplot(fig_opt)
            plt.close(fig_opt)


# ===================================================================
# TAB 5: Reports
# ===================================================================

with tab_reports:
    st.header("Report Generation")
    st.markdown("Generate HTML or APQP-compliant reports from analysis results.")

    st.subheader("Report Configuration")
    rc1, rc2 = st.columns(2)
    with rc1:
        rpt_title = st.text_input("Report title", "Tolerance Stack Analysis Report", key="rpt_title")
        rpt_project = st.text_input("Project", "3D Stack Project", key="rpt_project")
        rpt_author = st.text_input("Author", "", key="rpt_author")
    with rc2:
        rpt_revision = st.text_input("Revision", "A", key="rpt_revision")
        rpt_format = st.selectbox("Report format", ["HTML", "APQP (HTML)", "PDF", "Plain Text"], key="rpt_format")
        rpt_include_plots = st.checkbox("Include plots", value=True, key="rpt_plots")

    st.markdown("---")
    st.subheader("Run Analysis and Generate Report")
    st.markdown("Select an analysis source to include in the report.")

    rpt_source = st.radio("Analysis source", [
        "Tolerance Stack (Tab 1)",
        "Assembly (Tab 3)",
        "Manual input",
    ], key="rpt_source")

    if rpt_source == "Manual input":
        st.markdown("Enter results manually as JSON:")
        rpt_manual = st.text_area(
            "Results JSON",
            value='{"nominal": 10.0, "wc_variation": 0.5, "rss_variation": 0.25, "mc_mean": 10.01, "mc_std": 0.08}',
            key="rpt_manual",
        )

    # Spec limits for APQP
    if rpt_format == "APQP (HTML)":
        with st.expander("APQP Spec Limits"):
            rpt_usl = st.number_input("USL", value=0.0, format="%.4f", key="rpt_usl")
            rpt_lsl = st.number_input("LSL", value=0.0, format="%.4f", key="rpt_lsl")

    if st.button("Generate Report", key="rpt_gen", type="primary"):
        from tolerance_stack.reporting import ReportConfig, generate_html_report, generate_text_report, generate_apqp_report

        config = ReportConfig(
            title=rpt_title,
            project=rpt_project,
            author=rpt_author,
            revision=rpt_revision,
            include_plots=rpt_include_plots,
        )

        # Collect results dict
        results_dict = {}
        assembly_info = None
        capability_results = None
        plot_images = []
        spec_limits_dict = None

        if rpt_source == "Tolerance Stack (Tab 1)":
            # Re-run analysis if we have contributors
            ts_contribs = st.session_state.get("ts_contributors", [])
            if ts_contribs:
                from tolerance_stack.models import Contributor, ToleranceStack, Distribution, ContributorType
                from tolerance_stack.analysis import analyze_stack

                stack = ToleranceStack(name=st.session_state.get("ts_name", "Stack"))
                for c in ts_contribs:
                    stack.add(Contributor(
                        name=c["name"], nominal=c["nominal"],
                        plus_tol=c["plus_tol"], minus_tol=c["minus_tol"],
                        direction=tuple(c["direction"]), sign=c["sign"],
                        distribution=Distribution(c["distribution"]),
                        contributor_type=ContributorType(c.get("contributor_type", "linear")),
                        sigma=c.get("sigma", 3.0),
                    ))
                ana_results = analyze_stack(stack, methods=["wc", "rss", "mc"], sigma=3.0, mc_samples=100000, mc_seed=42)
                for k, v in ana_results.items():
                    results_dict[k] = {
                        "method": k,
                        "nominal": v.nominal,
                        "variation": v.variation,
                        "sensitivity": v.sensitivity,
                    }
                    if v.mc_mean is not None:
                        results_dict[k]["mc_mean"] = v.mc_mean
                        results_dict[k]["mc_std"] = v.mc_std
            else:
                st.warning("No tolerance stack contributors defined in Tab 1.")
                results_dict = {"info": "No data"}

        elif rpt_source == "Assembly (Tab 3)":
            ad = st.session_state.get("assy_data")
            if ad and ad.get("measurement"):
                from tolerance_stack.assembly import Assembly
                from tolerance_stack.assembly_analysis import analyze_assembly
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                    json.dump(ad, f)
                    f.flush()
                    assy_obj = Assembly.load(f.name)
                ana_results = analyze_assembly(assy_obj, methods=["wc", "rss", "mc"], sigma=3.0, mc_samples=100000, mc_seed=42)
                assembly_info = {
                    "name": ad.get("name", "Assembly"),
                    "bodies": len(ad.get("bodies", [])),
                    "mates": len(ad.get("mates", [])),
                }
                for k, v in ana_results.items():
                    results_dict[k] = {
                        "method": k,
                        "nominal": v.nominal,
                        "variation": v.variation,
                        "sensitivity": v.sensitivity,
                    }
                    if v.mc_mean is not None:
                        results_dict[k]["mc_mean"] = v.mc_mean
                        results_dict[k]["mc_std"] = v.mc_std
            else:
                st.warning("No assembly with measurement defined in Tab 3.")
                results_dict = {"info": "No data"}

        elif rpt_source == "Manual input":
            try:
                results_dict = {"manual": json.loads(rpt_manual)}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                results_dict = {"error": str(e)}

        if rpt_format == "APQP (HTML)":
            spec_limits_dict = {}
            if hasattr(st.session_state, "rpt_usl") or "rpt_usl" in st.session_state:
                usl_v = st.session_state.get("rpt_usl", 0)
                lsl_v = st.session_state.get("rpt_lsl", 0)
                if usl_v != 0 or lsl_v != 0:
                    spec_limits_dict["measurement"] = {}
                    if usl_v != 0:
                        spec_limits_dict["measurement"]["usl"] = usl_v
                    if lsl_v != 0:
                        spec_limits_dict["measurement"]["lsl"] = lsl_v

        # Generate report
        if rpt_format == "HTML":
            html = generate_html_report(config, results_dict, assembly_info=assembly_info,
                                        capability_results=capability_results, plot_images=plot_images)
            st.download_button("Download HTML Report", data=html,
                               file_name="tolerance_report.html", mime="text/html", key="rpt_dl_html")
            st.success("HTML report generated!")
            with st.expander("Preview"):
                st.components.v1.html(html, height=600, scrolling=True)

        elif rpt_format == "APQP (HTML)":
            html = generate_apqp_report(config, results_dict, assembly_info=assembly_info,
                                        capability_results=capability_results,
                                        plot_images=plot_images,
                                        spec_limits=spec_limits_dict)
            st.download_button("Download APQP Report", data=html,
                               file_name="apqp_report.html", mime="text/html", key="rpt_dl_apqp")
            st.success("APQP report generated!")
            with st.expander("Preview"):
                st.components.v1.html(html, height=600, scrolling=True)

        elif rpt_format == "PDF":
            from tolerance_stack.reporting import generate_pdf_report
            pdf_bytes = generate_pdf_report(config, results_dict, assembly_info=assembly_info,
                                             capability_results=capability_results)
            st.download_button("Download PDF Report", data=pdf_bytes,
                               file_name="tolerance_report.pdf", mime="application/pdf", key="rpt_dl_pdf")
            st.success("PDF report generated!")

        elif rpt_format == "Plain Text":
            text = generate_text_report(config, results_dict, capability_results=capability_results)
            st.download_button("Download Text Report", data=text,
                               file_name="tolerance_report.txt", mime="text/plain", key="rpt_dl_txt")
            st.success("Text report generated!")
            st.text(text)


# ===================================================================
# Sidebar info
# ===================================================================

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **3D Tolerance Stack Analyzer** supports:

    1. **Tolerance Stack** -- Linear dimension chains with 3D direction vectors
    2. **Linkage** -- Kinematic chains of joints and links
    3. **Assembly** -- Rigid bodies with geometric features, mates, and GD&T
    4. **DOE / Optimizer** -- Design of Experiments and tolerance optimization
    5. **Reports** -- HTML, APQP-compliant, and text report generation

    Each analysis supports **Worst-Case**, **RSS**, and **Monte Carlo** methods.
    """)
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("""
    - 9 statistical distributions (Normal, Uniform, Triangular, Weibull, Lognormal, Rayleigh, Bimodal)
    - Process capability (Cp/Cpk/Pp/Ppk/PPM)
    - Full GD&T per ASME Y14.5 with composite FCF support
    - Datum reference frames, DOF tracking, datum shift
    - Interactive 3D visualization (Plotly)
    - STEP file import with PMI extraction
    - DOE: HLM, Full Factorial, Latin Hypercube, Response Surface (RSM)
    - Sobol' global sensitivity analysis
    - Tolerance optimization with cost model
    - APQP-compliant reporting (HTML, PDF, Text)
    - Gap, flush, and interference measurements
    """)
    st.markdown("---")
    st.markdown("**Analysis Methods**")
    st.markdown("""
    - **Worst-Case**: Every tolerance at its extreme simultaneously
    - **RSS**: Statistical root-sum-of-squares
    - **Monte Carlo**: Numerical simulation with configurable distributions
    """)
    st.markdown("---")
    st.markdown(f"Version: {__import__('tolerance_stack').__version__}")
