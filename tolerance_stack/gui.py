"""Streamlit GUI for 3D Tolerance Stack Analysis.

Launch with:
    streamlit run tolerance_stack/gui.py
    # or
    python -m streamlit run tolerance_stack/gui.py
"""

from __future__ import annotations

import io
import json
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

tab_stack, tab_linkage, tab_assembly = st.tabs([
    "Tolerance Stack",
    "Linkage",
    "Assembly",
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
                c_dist = st.selectbox("Distribution", ["normal", "uniform", "triangular"], key="ts_c_dist")
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
                    st.text(f"{sign_str} {c['name']}: {c['nominal']:.4f} +{c['plus_tol']:.4f}/-{c['minus_tol']:.4f}  dir={c['direction']}")
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
                ax.set_xlabel("Gap Value")
                ax.set_ylabel("Density")
                ax.set_title("Monte Carlo Distribution")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

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

            if st.button("Add joint", key="lk_add_j"):
                st.session_state["lk_joints"].append({
                    "name": j_name, "joint_type": j_type,
                    "nominal": j_nom, "plus_tol": j_plus, "minus_tol": j_minus,
                    "distribution": "normal", "sigma": 3.0,
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

            if st.button("Add link", key="lk_add_l"):
                try:
                    direction = [float(x) for x in l_dir.split(",")]
                except ValueError:
                    direction = [1, 0, 0]
                st.session_state["lk_links"].append({
                    "name": l_name, "length": l_len, "direction": direction,
                    "plus_tol": l_plus, "minus_tol": l_minus,
                    "distribution": "normal", "sigma": 3.0,
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

            # 3D linkage plot
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
            st.info("Load an example or upload a JSON file to get started.")

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

        can_run_assy = assy_data is not None and assy_data.get("measurement") is not None
        if assy_data and not can_run_assy:
            st.info("Assembly needs a measurement defined to run analysis.")

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
                ax.set_xlabel("Measurement Value")
                ax.set_ylabel("Density")
                ax.set_title("Monte Carlo Distribution")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

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
# Sidebar info
# ===================================================================

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **3D Tolerance Stack Analyzer** supports three levels of analysis:

    1. **Tolerance Stack** — Linear dimension chains with 3D direction vectors
    2. **Linkage** — Kinematic chains of joints and links
    3. **Assembly** — Rigid bodies with geometric features and mates

    Each supports **Worst-Case**, **RSS**, and **Monte Carlo** analysis.
    """)
    st.markdown("---")
    st.markdown("**Analysis Methods**")
    st.markdown("""
    - **Worst-Case**: Every tolerance at its extreme simultaneously
    - **RSS**: Statistical root-sum-of-squares (normal distribution)
    - **Monte Carlo**: Numerical simulation with configurable distributions
    """)
    st.markdown("---")
    st.markdown(f"Version: {__import__('tolerance_stack').__version__}")
