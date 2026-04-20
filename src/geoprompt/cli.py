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


def shell_completion_script(program: str = "geoprompt", *, shell: str = "bash") -> str:
    """Return a starter shell-completion script for the GeoPrompt CLI."""
    commands = "info version wizard demo compare history serve plugins recipes doctor completion"
    if shell == "powershell":
        return (
            f"Register-ArgumentCompleter -Native -CommandName {program} -ScriptBlock {{\n"
            f"    param($wordToComplete, $commandAst, $cursorPosition)\n"
            f"    '{commands}'.Split(' ') | Where-Object {{ $_ -like \"$wordToComplete*\" }} | ForEach-Object {{\n"
            f"        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)\n"
            f"    }}\n"
            f"}}"
        )
    return (
        f"_{program}_complete() {{\n"
        f"    local commands=\"{commands}\"\n"
        f"    COMPREPLY=( $(compgen -W \"$commands\" -- \"${{COMP_WORDS[1]}}\") )\n"
        f"}}\n"
        f"complete -F _{program}_complete {program}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level GeoPrompt command-line parser."""
    parser = argparse.ArgumentParser(
        prog="geoprompt",
        description="Unified CLI for GeoPrompt demos, comparisons, workflows, and package diagnostics.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Show package summary and install guidance")
    subparsers.add_parser("version", help="Print the installed version")
    subparsers.add_parser("plugins", help="List registered plugins and extension hooks")
    subparsers.add_parser("recipes", help="List built-in workflow recipes")
    subparsers.add_parser("doctor", help="Run a lightweight release-readiness audit")

    completion_parser = subparsers.add_parser("completion", help="Print a starter shell completion script")
    completion_parser.add_argument("--shell", choices=["bash", "powershell"], default="bash")

    wizard_parser = subparsers.add_parser("wizard", help="Suggest a no-code workflow plan")
    wizard_parser.add_argument("goal", nargs="+", help="Plain-English workflow goal")
    wizard_parser.add_argument("--persona", default="analyst")
    wizard_parser.add_argument("--industry", default="general")

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
    print("Commands: info, version, wizard, demo, compare, history, serve, plugins, recipes, doctor, completion")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GeoPrompt command-line interface."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in {None, "info"}:
        _print_info()
        return 0

    if args.command == "version":
        print(_version())
        return 0

    if args.command == "plugins":
        from .ecosystem import list_plugins

        plugins = list_plugins()
        if not plugins:
            print("No plugins are registered yet. Use register_plugin(...) to add one.")
            return 0
        for plugin in plugins:
            print(f"- {plugin['name']}: {plugin.get('description', '').strip()}")
        return 0

    if args.command == "recipes":
        from .ecosystem import list_recipes

        for recipe in list_recipes():
            print(f"- {recipe['name']}: {', '.join(recipe.get('steps', []))}")
        return 0

    if args.command == "doctor":
        from .quality import release_readiness_report

        report = release_readiness_report([Path(__file__), Path(__file__).with_name("ecosystem.py")])
        print(f"Release stage: {report['release_stage']}")
        print(f"Quality passed: {report['quality'].get('passed')}")
        print(f"Smoke profiles: {', '.join(item['profile'] for item in report['packaging_smoke_matrix'])}")
        return 0

    if args.command == "completion":
        print(shell_completion_script(shell=args.shell))
        return 0

    if args.command == "wizard":
        from .ecosystem import build_workflow_wizard

        plan = build_workflow_wizard(
            " ".join(args.goal),
            persona=args.persona,
            industry=args.industry,
        )
        print(plan["message"])
        if plan["recommended_recipes"]:
            print("Recipes:")
            for recipe in plan["recommended_recipes"]:
                print(f"- {recipe['name']}: {', '.join(recipe['steps'])}")
        else:
            print("Steps:")
            for step in plan["suggested_steps"]:
                print(f"- {step}")
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


__all__ = ["build_parser", "main", "shell_completion_script"]


if __name__ == "__main__":
    raise SystemExit(main())
