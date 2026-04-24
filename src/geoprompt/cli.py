from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Sequence


def _version() -> str:
    try:
        return importlib.metadata.version("geoprompt")
    except importlib.metadata.PackageNotFoundError:
        return "local-dev"


def _cli_require_capability(name: str, command: str) -> int | None:
    """Check a capability and print an actionable error if missing.

    Returns ``1`` (exit code) if the capability is absent so callers can
    ``return _cli_require_capability(...)`` directly, or ``None`` on success.
    """
    from ._capabilities import CAPABILITY_REGISTRY, _is_importable

    spec = CAPABILITY_REGISTRY.get(name)
    if spec is None:
        return None  # unknown capability – let the command handle it
    if not _is_importable(spec.import_name):
        print(
            f"Error: '{command}' requires '{spec.import_name}' which is not installed.\n"
            f"  {spec.description}\n"
            f"  Install with: pip install {spec.pip_extra}"
        )
        return 1
    return None


def shell_completion_script(program: str = "geoprompt", *, shell: str = "bash") -> str:
    """Return a starter shell-completion script for the GeoPrompt CLI."""
    commands = "info version wizard demo compare history docs-artifacts serve plugins recipes doctor completion capability-report model-register model-validate infer-raster benchmark-run"
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
    capability_parser = subparsers.add_parser("capability-report", help="Print startup capability availability and dependency status")
    capability_parser.add_argument("--format", choices=["text", "json"], default="text")

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
    history_parser.add_argument("--dashboard", action="store_true", help="Also export benchmark dashboard bundle")
    history_parser.add_argument("--min-speedup-ratio", type=float, default=1.05, help="Alert threshold for benchmark dashboard speedup ratio")

    docs_artifacts_parser = subparsers.add_parser("docs-artifacts", help="Rebuild or validate generated docs/report artifacts")
    docs_artifacts_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Target artifact directory")
    docs_artifacts_parser.add_argument("--check", action="store_true", help="Fail if artifacts are stale or mismatched")
    docs_artifacts_parser.add_argument("--clean", action="store_true", help="Clear output directory before rebuilding artifacts")

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
    print("Commands: info, version, wizard, demo, compare, history, docs-artifacts, serve, plugins, recipes, doctor, completion, capability-report, model-register, model-validate, infer-raster, benchmark-run")


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

    if args.command == "capability-report":
        from . import capability_report

        report = capability_report()
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        print(f"GeoPrompt capability report v{report.get('schema_version', 'unknown')}")
        print(f"Package version: {report.get('package_version', 'unknown')}")
        print(f"Checked at (UTC): {report.get('checked_at_utc', 'unknown')}")
        print(f"Fallback policy: {report.get('fallback_policy', 'warn')}")
        print(f"Enabled ({len(report.get('enabled', []))}): {', '.join(report.get('enabled', []))}")
        print(f"Disabled ({len(report.get('disabled', []))}): {', '.join(report.get('disabled', [])) or 'none'}")
        print(f"Degraded ({len(report.get('degraded', []))}): {', '.join(report.get('degraded', [])) or 'none'}")
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
        from .compare import export_benchmark_dashboard_bundle, export_benchmark_history

        written = export_benchmark_history(args.output_dir)
        print(f"History HTML: {written['html']}")
        print(f"History JSON: {written['json']}")
        if args.dashboard:
            dashboard = export_benchmark_dashboard_bundle(
                args.output_dir,
                min_speedup_ratio=args.min_speedup_ratio,
            )
            print(f"Dashboard HTML: {dashboard['html']}")
            print(f"Dashboard JSON: {dashboard['json']}")
            print(f"Dashboard Markdown: {dashboard['markdown']}")
        return 0

    if args.command == "docs-artifacts":
        from .artifacts import check_docs_artifacts_freshness, generate_docs_artifacts

        if args.check:
            result = check_docs_artifacts_freshness(args.output_dir)
            if result.get("ok"):
                print(f"Artifacts are fresh: {result.get('manifest', 'unknown manifest')}")
                print(f"Checked files: {result.get('checked_files', 0)}")
                return 0
            print("Artifacts are stale or invalid.")
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1

        written = generate_docs_artifacts(args.output_dir, clean_output_dir=args.clean)
        print(f"Artifacts rebuilt in: {written['output_dir']}")
        print(f"Manifest: {written['manifest']}")
        print(f"Files tracked: {written['file_count']}")
        return 0

    if args.command == "serve":
        rc = _cli_require_capability("fastapi", "serve")
        if rc is not None:
            return rc
        rc = _cli_require_capability("uvicorn", "serve")
        if rc is not None:
            return rc
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
        rc = _cli_require_capability("rasterio", "infer-raster")
        if rc is not None:
            return rc
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
