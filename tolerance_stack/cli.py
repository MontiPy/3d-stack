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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
