from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Sequence


def _version() -> str:
    try:
        return importlib.metadata.version("geoprompt")
    except importlib.metadata.PackageNotFoundError:
        return "local-dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoprompt",
        description="Unified CLI for GeoPrompt demos, comparisons, and package info.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Show package summary and install guidance")
    subparsers.add_parser("version", help="Print the installed version")

    demo_parser = subparsers.add_parser("demo", help="Run the built-in demo workflow")
    demo_parser.add_argument("demo_args", nargs=argparse.REMAINDER)

    compare_parser = subparsers.add_parser("compare", help="Run the comparison and benchmark workflow")
    compare_parser.add_argument("compare_args", nargs=argparse.REMAINDER)

    history_parser = subparsers.add_parser("history", help="Export a benchmark history summary page")
    history_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory containing JSON benchmark reports")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service template")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def _print_info() -> None:
    print("GeoPrompt")
    print(f"Version: {_version()}")
    print("Install profiles: core, viz, io, excel, db, overlay, compare, raster, service, all")
    print("Commands: info, version, demo, compare, history, serve")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in {None, "info"}:
        _print_info()
        return 0

    if args.command == "version":
        print(_version())
        return 0

    if args.command == "demo":
        from . import demo as demo_module

        original = list(sys.argv)
        try:
            sys.argv = ["geoprompt-demo", *list(getattr(args, "demo_args", []))]
            demo_module.main()
        finally:
            sys.argv = original
        return 0

    if args.command == "compare":
        from . import compare as compare_module

        original = list(sys.argv)
        try:
            sys.argv = ["geoprompt-compare", *list(getattr(args, "compare_args", []))]
            compare_module.main()
        finally:
            sys.argv = original
        return 0

    if args.command == "history":
        from .compare import export_benchmark_history

        written = export_benchmark_history(args.output_dir)
        print(f"History HTML: {written['html']}")
        print(f"History JSON: {written['json']}")
        return 0

    if args.command == "serve":
        from .service import app
        if app is None:
            print("FastAPI support is not installed. Install the service extra to run the API.")
            return 1
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
