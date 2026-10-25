#!/usr/bin/env python3
"""Unified CLI tool for the CNN vs Exact Pattern Matching project.

Commands:
    test         Run unit tests with pytest.
    adversarial  Run adversarial checks on learned representations and identities.
    experiments  Run empirical experiments (experiment1.py through experiment6.py).
    figures      Generate all publication figures (make_figures.py).
    scaling      Run scaling benchmark (scaling_benchmark.py if present, else experiment5.py).
    stats        Run statistical analysis (statistical_analysis.py if present).
    run-all      Run full pipeline: tests, experiments, scaling, and adversarial check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# ANSI colors for readable terminal output
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m" if USE_COLOR else text


def bold(text: str) -> str:
    return _c(text, "1")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def red(text: str) -> str:
    return _c(text, "31")


def cyan(text: str) -> str:
    return _c(text, "36")


def dim(text: str) -> str:
    return _c(text, "2")


def print_banner(title: str) -> None:
    sep = "=" * 72
    print(f"\n{cyan(sep)}")
    print(f" {bold(title)}")
    print(f"{cyan(sep)}\n")


def print_step(name: str) -> None:
    print(f"{cyan('==>')} {bold(name)}")


def get_process_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}:{existing_pp}" if existing_pp else str(PROJECT_ROOT)
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    return env


def run_command(
    cmd: list[str],
    desc: str | None = None,
    check: bool = True,
) -> int:
    """Execute a command list in a subprocess with unified environment."""
    if desc:
        print_step(desc)
    cmd_str = " ".join(cmd)
    print(dim(f"$ {cmd_str}"))

    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=get_process_env())
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(green(f"✓ Success ({elapsed:.2f}s)"))
        else:
            print(red(f"✗ Failed with exit code {res.returncode} ({elapsed:.2f}s)"))
            if check:
                sys.exit(res.returncode)
        return res.returncode
    except KeyboardInterrupt:
        print(yellow("\nExecution interrupted by user."))
        sys.exit(130)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> int:
    """Run pytest suite."""
    print_banner("Running Test Suite")
    cmd = [sys.executable, "-m", "pytest"]
    if args.verbose:
        cmd.append("-v")
    if args.exitfirst:
        cmd.append("-x")
    if args.expression:
        cmd.extend(["-k", args.expression])
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    return run_command(cmd, desc="Running unit tests via pytest")


def cmd_adversarial(args: argparse.Namespace) -> int:
    """Run adversarial check."""
    print_banner("Adversarial Representation Checks")
    script = PROJECT_ROOT / "adversarial_check.py"
    if not script.exists():
        print(red(f"Error: {script} not found."))
        return 1
    return run_command([sys.executable, str(script)], desc="Running adversarial_check.py")


def cmd_experiments(args: argparse.Namespace) -> int:
    """Run experiments 1 through 6."""
    target_exps = (
        [int(args.experiment)]
        if args.experiment is not None
        else list(range(1, 7))
    )

    print_banner(f"Running Experiments ({', '.join(f'exp{i}' for i in target_exps)})")
    results: list[tuple[int, int, float]] = []

    for num in target_exps:
        script = PROJECT_ROOT / f"experiment{num}.py"
        if not script.exists():
            print(yellow(f"Warning: {script} does not exist. Skipping."))
            continue

        desc = f"Running Experiment {num} ({script.name})"
        t0 = time.time()
        ret = run_command([sys.executable, str(script)], desc=desc, check=False)
        elapsed = time.time() - t0
        results.append((num, ret, elapsed))

        if ret != 0 and args.stop_on_error:
            print(red(f"Aborting experiments due to error in experiment {num}."))
            return ret

    print("\n" + "-" * 40)
    print("Experiments Summary:")
    all_ok = True
    for num, ret, elap in results:
        status = green("PASS") if ret == 0 else red("FAIL")
        if ret != 0:
            all_ok = False
        print(f"  Experiment {num}: {status} ({elap:.2f}s)")
    print("-" * 40)
    return 0 if all_ok else 1


def cmd_figures(args: argparse.Namespace) -> int:
    """Generate figures."""
    print_banner("Generating Publication Figures")
    script = PROJECT_ROOT / "make_figures.py"
    if not script.exists():
        print(red(f"Error: {script} not found."))
        return 1
    return run_command([sys.executable, str(script)], desc="Running make_figures.py")


def cmd_scaling(args: argparse.Namespace) -> int:
    """Run scaling benchmark or fallback to experiment5."""
    print_banner("Running Scaling Benchmark")
    scaling_script = PROJECT_ROOT / "scaling_benchmark.py"
    if scaling_script.exists():
        return run_command(
            [sys.executable, str(scaling_script)],
            desc="Running scaling_benchmark.py",
        )
    else:
        print(yellow("scaling_benchmark.py not found; falling back to Direction 2 prototype (experiment5.py)."))
        exp5 = PROJECT_ROOT / "experiment5.py"
        if not exp5.exists():
            print(red(f"Error: Neither scaling_benchmark.py nor {exp5} found."))
            return 1
        return run_command([sys.executable, str(exp5)], desc="Running experiment5.py (scaling prototype)")


def cmd_stats(args: argparse.Namespace) -> int:
    """Run statistical analysis if present."""
    print_banner("Running Statistical Analysis")
    stats_script = PROJECT_ROOT / "statistical_analysis.py"
    if stats_script.exists():
        return run_command(
            [sys.executable, str(stats_script)],
            desc="Running statistical_analysis.py",
        )
    else:
        print(yellow("statistical_analysis.py is not present in this repository."))
        print(dim("No statistical analysis to run."))
        return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run full pipeline: tests, experiments, figures, scaling, and adversarial check."""
    print_banner("Executing Full Project Pipeline (run-all)")
    pipeline: list[tuple[str, list[str]]] = []

    if not args.skip_tests:
        pipeline.append(("Unit Tests", [sys.executable, "-m", "pytest", "-v"]))
    if not args.skip_experiments:
        for i in range(1, 7):
            script = PROJECT_ROOT / f"experiment{i}.py"
            if script.exists():
                pipeline.append((f"Experiment {i}", [sys.executable, str(script)]))
    if not args.skip_figures:
        script = PROJECT_ROOT / "make_figures.py"
        if script.exists():
            pipeline.append(("Make Figures", [sys.executable, str(script)]))
    if not args.skip_scaling:
        scaling_script = PROJECT_ROOT / "scaling_benchmark.py"
        target_scaling = scaling_script if scaling_script.exists() else PROJECT_ROOT / "experiment5.py"
        if target_scaling.exists():
            pipeline.append(("Scaling Benchmark", [sys.executable, str(target_scaling)]))
    if not args.skip_adversarial:
        adv_script = PROJECT_ROOT / "adversarial_check.py"
        if adv_script.exists():
            pipeline.append(("Adversarial Check", [sys.executable, str(adv_script)]))

    overall_t0 = time.time()
    results: list[tuple[str, int, float]] = []
    failed = False

    for name, cmd in pipeline:
        t0 = time.time()
        ret = run_command(cmd, desc=f"Pipeline Stage: {name}", check=False)
        elap = time.time() - t0
        results.append((name, ret, elap))
        if ret != 0:
            failed = True
            if args.stop_on_error:
                print(red(f"Stopping pipeline execution due to failure in stage: {name}"))
                break

    print_banner("Pipeline Execution Summary")
    for name, ret, elap in results:
        status = green("PASS") if ret == 0 else red("FAIL")
        print(f"  {name:<25} : {status} ({elap:.2f}s)")
    total_elapsed = time.time() - overall_t0
    print(f"\nTotal Pipeline Time: {total_elapsed:.2f}s")

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI Argument Parser Construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified CLI for CNN vs Exact Pattern Matching research project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python main.py test
  uv run python main.py test -k test_identity
  uv run python main.py experiments --experiment 1
  uv run python main.py experiments
  uv run python main.py adversarial
  uv run python main.py figures
  uv run python main.py scaling
  uv run python main.py stats
  uv run python main.py run-all
""",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="Available Subcommands",
        metavar="<command>",
    )

    # 1. test
    p_test = subparsers.add_parser(
        "test",
        help="Run unit tests via pytest.",
        description="Run automated pytest test suite in tests/.",
    )
    p_test.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Run pytest in verbose mode (default: True).",
    )
    p_test.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Exit pytest immediately on first error or failure.",
    )
    p_test.add_argument(
        "-k", "--expression",
        type=str,
        default=None,
        help="Filter tests by substring or expression (passed to pytest -k).",
    )
    p_test.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments forwarded directly to pytest.",
    )
    p_test.set_defaults(func=cmd_test)

    # 2. adversarial
    p_adv = subparsers.add_parser(
        "adversarial",
        help="Run adversarial checks on learned representations.",
        description="Run adversarial_check.py: verifies structural identities and claims.",
    )
    p_adv.set_defaults(func=cmd_adversarial)

    # 3. experiments
    p_exp = subparsers.add_parser(
        "experiments",
        help="Run empirical experiments (experiment1.py to experiment6.py).",
        description="Run one or all experiments in the study.",
    )
    p_exp.add_argument(
        "-e", "--experiment",
        type=int,
        choices=range(1, 7),
        default=None,
        help="Specific experiment number to run (1-6). Defaults to running all.",
    )
    p_exp.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop sequence if any experiment fails (default: True).",
    )
    p_exp.set_defaults(func=cmd_experiments)

    # 4. figures
    p_fig = subparsers.add_parser(
        "figures",
        help="Generate publication figures (make_figures.py).",
        description="Run make_figures.py to produce all publication plots in figures/.",
    )
    p_fig.set_defaults(func=cmd_figures)

    # 5. scaling
    p_scale = subparsers.add_parser(
        "scaling",
        help="Run scaling benchmark (scaling_benchmark.py or experiment5.py).",
        description="Benchmark operations and scaling for pattern matching.",
    )
    p_scale.set_defaults(func=cmd_scaling)

    # 6. stats
    p_stats = subparsers.add_parser(
        "stats",
        help="Run statistical analysis (statistical_analysis.py if present).",
        description="Run statistical_analysis.py if present in the workspace.",
    )
    p_stats.set_defaults(func=cmd_stats)

    # 7. run-all
    p_all = subparsers.add_parser(
        "run-all",
        help="Run complete pipeline: tests, experiments, scaling, adversarial checks.",
        description="Execute the complete research pipeline end-to-end.",
    )
    p_all.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running unit tests.",
    )
    p_all.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments.",
    )
    p_all.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip generating figures.",
    )
    p_all.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip scaling benchmark.",
    )
    p_all.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip adversarial checks.",
    )
    p_all.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop immediately if any pipeline stage fails (default: True).",
    )
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        sys.exit(1)

    exit_code = args.func(args)
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
