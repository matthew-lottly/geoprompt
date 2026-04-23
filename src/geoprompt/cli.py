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

    # I10 — raster AI model management commands
    model_register = subparsers.add_parser("model-register", help="Register a local model asset into the GeoPrompt model registry")
    model_register.add_argument("model_path", type=Path, help="Path to the model file")
    model_register.add_argument("--model-id", required=True, help="Unique model identifier")
    model_register.add_argument("--version", default="1.0.0", help="Model version string")
    model_register.add_argument("--trust-level", default="internal", choices=["public", "internal", "restricted", "classified"])

    model_validate = subparsers.add_parser("model-validate", help="Validate a model against the RasterModelContract")
    model_validate.add_argument("model_id", help="Registered model ID to validate")
    model_validate.add_argument("--connector", default="auto", help="Runtime connector to use for validation")

    infer_raster = subparsers.add_parser("infer-raster", help="Run raster inference with a registered model")
    infer_raster.add_argument("raster_path", type=Path, help="Path to input raster")
    infer_raster.add_argument("--model-id", required=True, help="Registered model ID")
    infer_raster.add_argument("--connector", default="auto", help="Inference connector")
    infer_raster.add_argument("--output", type=Path, default=None, help="Output path for inference result")

    subparsers.add_parser("benchmark-run", help="Run the raster AI benchmark suite and report results")

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

    if args.command == "model-register":
        from .ai import feature_tier_labels
        from .security import model_registry_manifest
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: model file not found: {model_path}")
            return 1
        import hashlib
        artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
        manifest = model_registry_manifest(
            args.model_id,
            args.version,
            artifact_hash=artifact_hash,
            trust_level=args.trust_level,
            metadata={"source_path": str(model_path)},
        )
        print(f"Model registered: {manifest['model_id']} v{manifest['model_version']}")
        print(f"  Manifest hash : {manifest['manifest_hash']}")
        print(f"  Artifact hash : {manifest['artifact_hash']}")
        print(f"  Trust level   : {manifest['trust_level']}")
        return 0

    if args.command == "model-validate":
        from .ai import runtime_doctor
        report = runtime_doctor()
        print(f"Runtime doctor: {report['overall']}")
        for rec in report.get("recommendations", []):
            print(f"  [!] {rec}")
        print(f"Connector requested: {args.connector}")
        print("Model validation requires a running connector — run `geoprompt doctor` first.")
        return 0

    if args.command == "infer-raster":
        from .ai import raster_ai_pipeline
        raster_path = Path(args.raster_path)
        if not raster_path.exists():
            print(f"Error: raster file not found: {raster_path}")
            return 1
        plan = raster_ai_pipeline(
            [
                {"type": "load", "config": {"path": str(raster_path)}},
                {"type": "preprocess", "config": {}},
                {"type": "infer", "config": {"model_id": args.model_id, "connector": args.connector}},
                {"type": "export", "config": {"output": str(args.output) if args.output else "output.tif"}},
            ],
            backend=args.connector if args.connector != "auto" else "local",
        )
        print(f"Raster AI pipeline planned: {plan['step_count']} steps, backend={plan['backend']}")
        if plan["validation_issues"]:
            for issue in plan["validation_issues"]:
                print(f"  [!] {issue}")
        return 0

    if args.command == "benchmark-run":
        from .quality import raster_ai_golden_benchmark, throughput_benchmark_matrix
        bench = raster_ai_golden_benchmark()
        matrix = throughput_benchmark_matrix()
        print(f"Golden benchmark corpus: {bench['corpus_id']}")
        print(f"Locked metrics: {bench['locked_metrics']}")
        print(f"Throughput matrix: {matrix['matrix_size']} combinations")
        print("Run benchmarks against a configured environment to collect results.")
        return 0

    parser.print_help()
    return 0


__all__ = ["build_parser", "main", "shell_completion_script"]


if __name__ == "__main__":
    raise SystemExit(main())
