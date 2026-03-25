#!/usr/bin/env python
"""Run the full manuscript replication stack from one entry point.

Usage:
    python replications/run_all.py
    python replications/run_all.py --skip-simulation
    python replications/run_all.py --full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _run_script(script_name: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(BASE_DIR / script_name)]
    if extra_args:
        command.extend(extra_args)

    start = time.perf_counter()
    print(f"\n=== Running {script_name} ===\n")
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - start
    print(f"\n=== Completed {script_name} in {elapsed:.1f}s ===\n")


def _run_reporting_bundle() -> None:
    command = [sys.executable, "-m", "causal_lens.cli"]
    start = time.perf_counter()
    print("\n=== Generating manuscript-ready report artifacts ===\n")
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - start
    print(f"\n=== Completed manuscript-ready report artifacts in {elapsed:.1f}s ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CausalLens replication scripts from one command."
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip the Monte Carlo simulation script.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full simulation study instead of the quick mode.",
    )
    args = parser.parse_args()

    _run_script("replicate_lalonde.py")
    _run_script("replicate_nhefs.py")
    _run_script("replicate_cross_design.py")

    if not args.skip_simulation:
        simulation_args = ["--full"] if args.full else None
        _run_script("replicate_simulation.py", simulation_args)

    _run_reporting_bundle()

    print("Replication outputs are available in replications/outputs/.")
    print("Manuscript-ready tables and figures are available in outputs/paper/.")


if __name__ == "__main__":
    main()