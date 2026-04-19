from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


DEFAULT_NOTEBOOKS = [
    Path("examples/notebooks/beginner_workflow.ipynb"),
    Path("examples/notebooks/geopandas_migration.ipynb"),
    Path("examples/notebooks/network_resilience_walkthrough.ipynb"),
    Path("examples/notebooks/benchmark_proof_walkthrough.ipynb"),
    Path("examples/notebooks/remote_service_walkthrough.ipynb"),
]


def execute_notebook(path: Path, timeout: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3")
    client.execute()

    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute GeoPrompt notebooks with a hard timeout.")
    parser.add_argument("--timeout", type=int, default=20, help="Per-cell timeout in seconds")
    parser.add_argument("paths", nargs="*", help="Optional notebook paths")
    args = parser.parse_args()

    notebook_paths = [Path(p) for p in args.paths] if args.paths else DEFAULT_NOTEBOOKS
    failures: list[str] = []

    for path in notebook_paths:
        try:
            execute_notebook(path, timeout=args.timeout)
            print(f"OK {path}")
        except Exception as exc:
            failures.append(f"FAIL {path}: {exc}")
            print(failures[-1])

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
