"""Command-line interface for 3D tolerance stack analysis."""

from __future__ import annotations

import argparse
import sys

from tolerance_stack.models import Contributor, ContributorType, Distribution, ToleranceStack
from tolerance_stack.analysis import analyze_stack


def _parse_direction(s: str) -> tuple[float, float, float]:
    """Parse a direction string like '1,0,0' into a tuple."""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Direction must be 3 comma-separated numbers, got: {s!r}")
    return tuple(float(x) for x in parts)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run analysis on a JSON stack file."""
    stack = ToleranceStack.load(args.file)
    methods = args.methods.split(",") if args.methods else None

    results = analyze_stack(
        stack,
        methods=methods,
        sigma=args.sigma,
        mc_samples=args.mc_samples,
        mc_seed=args.seed,
    )

    for key, result in results.items():
        print(result.summary())
        print()

    if args.plot:
        from tolerance_stack.visualize import (
            plot_waterfall,
            plot_monte_carlo_histogram,
            plot_sensitivity,
        )
        for key, result in results.items():
            if key == "mc" and result.mc_samples is not None:
                spec = None
                if args.spec_limits:
                    parts = args.spec_limits.split(",")
                    spec = (float(parts[0]), float(parts[1]))
                plot_monte_carlo_histogram(result, spec_limits=spec,
                                           save_path=args.save_plots)
            plot_waterfall(stack, result,
                          save_path=args.save_plots.replace(".png", f"_{key}_waterfall.png")
                          if args.save_plots else None)
            plot_sensitivity(result,
                             save_path=args.save_plots.replace(".png", f"_{key}_sensitivity.png")
                             if args.save_plots else None)


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive stack builder."""
    print("=== 3D Tolerance Stack Builder ===")
    print()

    name = input("Stack name: ").strip() or "My Stack"
    desc = input("Description (optional): ").strip()

    cd_str = input("Closure direction [1,0,0]: ").strip() or "1,0,0"
    closure_dir = _parse_direction(cd_str)

    stack = ToleranceStack(name=name, description=desc, closure_direction=closure_dir)

    print("\nAdd contributors (enter empty name to finish):\n")
    idx = 1
    while True:
        print(f"--- Contributor #{idx} ---")
        cname = input("  Name: ").strip()
        if not cname:
            break

        nominal = float(input("  Nominal dimension: "))
        plus_tol = float(input("  Plus tolerance (+): "))
        minus_tol = float(input("  Minus tolerance (+value, will be subtracted): "))

        dir_str = input("  Direction [1,0,0]: ").strip() or "1,0,0"
        direction = _parse_direction(dir_str)

        sign_str = input("  Sign (+1 or -1) [+1]: ").strip() or "+1"
        sign = int(sign_str)

        dist_str = input("  Distribution (normal/uniform/triangular) [normal]: ").strip() or "normal"
        distribution = Distribution(dist_str)

        type_str = input("  Type (linear/angular/geometric) [linear]: ").strip() or "linear"
        ctype = ContributorType(type_str)

        sigma_str = input("  Sigma (how many sigma the tol band represents) [3]: ").strip() or "3"
        sigma = float(sigma_str)

        contributor = Contributor(
            name=cname,
            nominal=nominal,
            plus_tol=plus_tol,
            minus_tol=minus_tol,
            direction=direction,
            sign=sign,
            distribution=distribution,
            contributor_type=ctype,
            sigma=sigma,
        )
        stack.add(contributor)
        print(f"  Added: {cname}\n")
        idx += 1

    if not stack.contributors:
        print("No contributors added. Exiting.")
        return

    # Save option
    save_path = input("\nSave stack to JSON file? (path or empty to skip): ").strip()
    if save_path:
        stack.save(save_path)
        print(f"Saved to {save_path}")

    # Analyze
    print("\n--- Running Analysis ---\n")
    results = analyze_stack(stack, sigma=args.sigma, mc_samples=args.mc_samples,
                            mc_seed=args.seed)

    for key, result in results.items():
        print(result.summary())
        print()


def cmd_create_example(args: argparse.Namespace) -> None:
    """Create an example stack file."""
    from tolerance_stack.examples import create_shaft_housing_example, create_multiaxis_example

    if args.example == "shaft":
        stack = create_shaft_housing_example()
    elif args.example == "multiaxis":
        stack = create_multiaxis_example()
    else:
        print(f"Unknown example: {args.example}")
        sys.exit(1)

    path = args.output or f"{args.example}_example.json"
    stack.save(path)
    print(f"Created example stack: {path}")


# ---------------------------------------------------------------------------
# Linkage commands
# ---------------------------------------------------------------------------

def cmd_linkage_analyze(args: argparse.Namespace) -> None:
    """Run analysis on a linkage JSON file."""
    from tolerance_stack.linkage import Linkage
    from tolerance_stack.linkage_analysis import analyze_linkage

    linkage = Linkage.load(args.file)
    methods = args.methods.split(",") if args.methods else None

    results = analyze_linkage(
        linkage,
        methods=methods,
        sigma=args.sigma,
        mc_samples=args.mc_samples,
        mc_seed=args.seed,
    )

    for key, result in results.items():
        print(result.summary())
        print()

    if args.plot:
        from tolerance_stack.linkage_visualize import (
            plot_linkage_3d,
            plot_linkage_sensitivity,
            plot_linkage_mc_scatter,
        )
        # Use MC result for the richest 3D plot, fall back to WC
        best = results.get("mc", results.get("wc", results.get("rss")))
        plot_linkage_3d(
            linkage, best,
            save_path=args.save_plots.replace(".png", "_linkage3d.png")
            if args.save_plots else None,
        )
        for key, result in results.items():
            plot_linkage_sensitivity(
                result,
                save_path=args.save_plots.replace(".png", f"_{key}_sensitivity.png")
                if args.save_plots else None,
            )
            if key == "mc" and result.mc_samples is not None:
                plot_linkage_mc_scatter(
                    result,
                    save_path=args.save_plots.replace(".png", f"_mc_scatter.png")
                    if args.save_plots else None,
                )


def cmd_linkage_example(args: argparse.Namespace) -> None:
    """Create an example linkage JSON file."""
    from tolerance_stack.linkage_examples import (
        create_planar_two_bar,
        create_spatial_robot_arm,
        create_four_bar_mechanism,
    )

    builders = {
        "two-bar": create_planar_two_bar,
        "robot-arm": create_spatial_robot_arm,
        "four-bar": create_four_bar_mechanism,
    }

    builder = builders.get(args.example)
    if builder is None:
        print(f"Unknown example: {args.example}")
        sys.exit(1)

    linkage = builder()
    path = args.output or f"{args.example}_linkage.json"
    linkage.save(path)
    print(f"Created example linkage: {path}")


def cmd_linkage_interactive(args: argparse.Namespace) -> None:
    """Interactive linkage builder."""
    from tolerance_stack.linkage import Joint, JointType, Link, Linkage
    from tolerance_stack.linkage_analysis import analyze_linkage
    from tolerance_stack.models import Distribution

    print("=== 3D Linkage Builder ===")
    print()
    print("Build a kinematic chain by alternating joints and links.")
    print("The chain must start and end with a joint.")
    print()

    name = input("Linkage name: ").strip() or "My Linkage"
    desc = input("Description (optional): ").strip()

    linkage = Linkage(name=name, description=desc)

    joint_types = {
        "fixed": JointType.FIXED,
        "revolute_x": JointType.REVOLUTE_X,
        "revolute_y": JointType.REVOLUTE_Y,
        "revolute_z": JointType.REVOLUTE_Z,
        "prismatic_x": JointType.PRISMATIC_X,
        "prismatic_y": JointType.PRISMATIC_Y,
        "prismatic_z": JointType.PRISMATIC_Z,
        "spherical": JointType.SPHERICAL,
    }

    print("\nStep 1: Add the first joint (base).\n")
    step = 1

    while True:
        is_joint_turn = len(linkage.joints) == len(linkage.links)

        if is_joint_turn:
            print(f"--- Joint #{len(linkage.joints) + 1} ---")
            jname = input("  Name (empty to finish): ").strip()
            if not jname:
                if not linkage.is_complete:
                    print("  Linkage must end with a joint. Please add one more joint.")
                    continue
                break

            type_str = input(f"  Type ({'/'.join(joint_types.keys())}) [fixed]: ").strip() or "fixed"
            jtype = joint_types.get(type_str)
            if jtype is None:
                print(f"  Unknown joint type: {type_str}")
                continue

            nominal = 0.0
            if jtype != JointType.FIXED:
                nominal = float(input("  Nominal value (deg or mm): "))

            pt = float(input("  Plus tolerance [0]: ").strip() or "0")
            mt = float(input("  Minus tolerance [0]: ").strip() or "0")

            joint = Joint(jname, jtype, nominal=nominal, plus_tol=pt, minus_tol=mt)
            linkage.add_joint(joint)
            print(f"  Added joint: {jname}\n")
        else:
            print(f"--- Link #{len(linkage.links) + 1} ---")
            lname = input("  Name: ").strip()
            if not lname:
                print("  A link is required between joints.")
                continue

            length = float(input("  Length: "))
            pt = float(input("  Plus tolerance [0]: ").strip() or "0")
            mt = float(input("  Minus tolerance [0]: ").strip() or "0")
            dir_str = input("  Direction [1,0,0]: ").strip() or "1,0,0"
            direction = _parse_direction(dir_str)

            link = Link(lname, length=length, plus_tol=pt, minus_tol=mt,
                        direction=direction)
            linkage.add_link(link)
            print(f"  Added link: {lname}\n")

    if not linkage.is_complete:
        print("Linkage is not complete. Exiting.")
        return

    # Save
    save_path = input("\nSave linkage to JSON? (path or empty to skip): ").strip()
    if save_path:
        linkage.save(save_path)
        print(f"Saved to {save_path}")

    # Analyze
    print("\n--- Running Linkage Analysis ---\n")
    results = analyze_linkage(linkage, sigma=args.sigma, mc_samples=args.mc_samples,
                              mc_seed=args.seed)

    for key, result in results.items():
        print(result.summary())
        print()


# ---------------------------------------------------------------------------
# Assembly commands
# ---------------------------------------------------------------------------

def cmd_assembly_analyze(args: argparse.Namespace) -> None:
    """Run analysis on an assembly JSON file."""
    from tolerance_stack.assembly import Assembly
    from tolerance_stack.assembly_analysis import analyze_assembly

    assy = Assembly.load(args.file)
    methods = args.methods.split(",") if args.methods else None

    results = analyze_assembly(
        assy,
        methods=methods,
        sigma=args.sigma,
        mc_samples=args.mc_samples,
        mc_seed=args.seed,
    )

    for key, result in results.items():
        print(result.summary())
        print()


def cmd_assembly_example(args: argparse.Namespace) -> None:
    """Create an example assembly JSON file."""
    from tolerance_stack.assembly_examples import (
        create_pin_in_hole_assembly,
        create_stacked_plates_assembly,
        create_bracket_assembly,
    )

    builders = {
        "pin-in-hole": create_pin_in_hole_assembly,
        "stacked-plates": create_stacked_plates_assembly,
        "bracket": create_bracket_assembly,
    }

    builder = builders.get(args.example)
    if builder is None:
        print(f"Unknown example: {args.example}")
        sys.exit(1)

    assy = builder()
    path = args.output or f"{args.example}_assembly.json"
    assy.save(path)
    print(f"Created example assembly: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tolstack",
        description="3D Tolerance Stack Analysis Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze a tolerance stack from a JSON file")
    p_analyze.add_argument("file", help="Path to stack JSON file")
    p_analyze.add_argument("-m", "--methods", default=None,
                           help="Comma-separated methods: wc,rss,mc (default: all)")
    p_analyze.add_argument("--sigma", type=float, default=3.0,
                           help="Sigma level for RSS (default: 3.0)")
    p_analyze.add_argument("--mc-samples", type=int, default=100_000,
                           help="Number of Monte Carlo samples (default: 100000)")
    p_analyze.add_argument("--seed", type=int, default=None,
                           help="Random seed for Monte Carlo")
    p_analyze.add_argument("--plot", action="store_true",
                           help="Show visualization plots")
    p_analyze.add_argument("--save-plots", default=None,
                           help="Save plots to file (base path, e.g. output.png)")
    p_analyze.add_argument("--spec-limits", default=None,
                           help="Spec limits for MC histogram, e.g. '0.1,0.5'")
    p_analyze.set_defaults(func=cmd_analyze)

    # --- interactive ---
    p_inter = subparsers.add_parser("interactive", help="Interactively build and analyze a stack")
    p_inter.add_argument("--sigma", type=float, default=3.0)
    p_inter.add_argument("--mc-samples", type=int, default=100_000)
    p_inter.add_argument("--seed", type=int, default=None)
    p_inter.set_defaults(func=cmd_interactive)

    # --- example ---
    p_example = subparsers.add_parser("example", help="Create an example stack file")
    p_example.add_argument("example", choices=["shaft", "multiaxis"],
                           help="Which example to create")
    p_example.add_argument("-o", "--output", default=None,
                           help="Output file path")
    p_example.set_defaults(func=cmd_create_example)

    # --- linkage analyze ---
    p_la = subparsers.add_parser("linkage-analyze",
                                  help="Analyze a linkage from a JSON file")
    p_la.add_argument("file", help="Path to linkage JSON file")
    p_la.add_argument("-m", "--methods", default=None,
                      help="Comma-separated methods: wc,rss,mc (default: all)")
    p_la.add_argument("--sigma", type=float, default=3.0)
    p_la.add_argument("--mc-samples", type=int, default=100_000)
    p_la.add_argument("--seed", type=int, default=None)
    p_la.add_argument("--plot", action="store_true",
                      help="Show visualization plots")
    p_la.add_argument("--save-plots", default=None,
                      help="Save plots to file (base path)")
    p_la.set_defaults(func=cmd_linkage_analyze)

    # --- linkage example ---
    p_le = subparsers.add_parser("linkage-example",
                                  help="Create an example linkage JSON file")
    p_le.add_argument("example", choices=["two-bar", "robot-arm", "four-bar"],
                      help="Which example to create")
    p_le.add_argument("-o", "--output", default=None,
                      help="Output file path")
    p_le.set_defaults(func=cmd_linkage_example)

    # --- linkage interactive ---
    p_li = subparsers.add_parser("linkage-interactive",
                                  help="Interactively build and analyze a linkage")
    p_li.add_argument("--sigma", type=float, default=3.0)
    p_li.add_argument("--mc-samples", type=int, default=100_000)
    p_li.add_argument("--seed", type=int, default=None)
    p_li.set_defaults(func=cmd_linkage_interactive)

    # --- assembly analyze ---
    p_aa = subparsers.add_parser("assembly-analyze",
                                  help="Analyze an assembly from a JSON file")
    p_aa.add_argument("file", help="Path to assembly JSON file")
    p_aa.add_argument("-m", "--methods", default=None,
                      help="Comma-separated methods: wc,rss,mc (default: all)")
    p_aa.add_argument("--sigma", type=float, default=3.0)
    p_aa.add_argument("--mc-samples", type=int, default=100_000)
    p_aa.add_argument("--seed", type=int, default=None)
    p_aa.set_defaults(func=cmd_assembly_analyze)

    # --- assembly example ---
    p_ae = subparsers.add_parser("assembly-example",
                                  help="Create an example assembly JSON file")
    p_ae.add_argument("example", choices=["pin-in-hole", "stacked-plates", "bracket"],
                      help="Which example to create")
    p_ae.add_argument("-o", "--output", default=None,
                      help="Output file path")
    p_ae.set_defaults(func=cmd_assembly_example)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
