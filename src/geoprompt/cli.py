from __future__ import annotations

import argparse
import importlib.metadata
import sys
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

    return parser


def _print_info() -> None:
    print("GeoPrompt")
    print(f"Version: {_version()}")
    print("Install profiles: core, viz, io, excel, db, overlay, compare, all")
    print("Commands: info, version, demo, compare")


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

    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
