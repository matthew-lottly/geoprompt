from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compare import (
    build_comparison_report,
    export_benchmark_dashboard_bundle,
    export_benchmark_history,
    export_comparison_bundle,
)
from .tools import (
    build_multi_scenario_report,
    build_resilience_portfolio_report,
    build_resilience_summary_report,
    build_scenario_report,
    export_multi_scenario_report,
    export_resilience_portfolio_report,
    export_resilience_summary_report,
    export_scenario_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROVENANCE_MANIFEST_NAME = "provenance_manifest.json"

DEFAULT_SOURCE_INPUTS = [
    PROJECT_ROOT / "data" / "sample_features.json",
    PROJECT_ROOT / "data" / "sample_assets.csv",
    PROJECT_ROOT / "data" / "benchmark_features.json",
    PROJECT_ROOT / "data" / "benchmark_regions.json",
    PROJECT_ROOT / "docs" / "figures-manifest.json",
    PROJECT_ROOT / "examples" / "benchmark_report_bundle.py",
    PROJECT_ROOT / "generate_assets.py",
    PROJECT_ROOT / "src" / "geoprompt" / "compare.py",
    PROJECT_ROOT / "src" / "geoprompt" / "tools.py",
    PROJECT_ROOT / "src" / "geoprompt" / "viz.py",
]


def _package_version() -> str:
    try:
        return importlib.metadata.version("geoprompt")
    except importlib.metadata.PackageNotFoundError:
        return "local-dev"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(source_inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_inputs):
        digest.update(path.as_posix().encode("utf-8"))
        if path.exists():
            digest.update(_file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


def _scenario_samples() -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    baseline = {"deficit": 0.22, "served": 100.0, "cost": 980.0}
    candidate = {"deficit": 0.09, "served": 118.0, "cost": 940.0}
    uncertainty = {
        "served": {"lower": 111.0, "upper": 123.0, "observed": 118.0},
        "cost": {"lower": 920.0, "upper": 980.0, "observed": 940.0},
    }
    return baseline, candidate, uncertainty


def _write_report_outputs(output_dir: Path) -> list[Path]:
    generated: list[Path] = []
    baseline, candidate, uncertainty = _scenario_samples()

    scenario_report = build_scenario_report(
        baseline,
        candidate,
        higher_is_better=["served"],
        uncertainty=uncertainty,
        metadata={"workflow": "docs-artifacts", "dataset": "sample-features"},
    )
    generated.append(Path(export_scenario_report(scenario_report, output_dir / "scenario-report.json")))
    generated.append(Path(export_scenario_report(scenario_report, output_dir / "scenario-report.csv")))
    generated.append(Path(export_scenario_report(scenario_report, output_dir / "scenario-report.md")))
    generated.append(Path(export_scenario_report(scenario_report, output_dir / "scenario-report.html")))

    multi_report = build_multi_scenario_report(
        {
            "baseline": baseline,
            "candidate_a": candidate,
            "candidate_b": {"deficit": 0.12, "served": 112.0, "cost": 900.0},
        },
        baseline_name="baseline",
        higher_is_better=["served"],
    )
    generated.append(Path(export_multi_scenario_report(multi_report, output_dir / "multi-scenario-report.json")))
    generated.append(Path(export_multi_scenario_report(multi_report, output_dir / "multi-scenario-report.csv")))
    generated.append(Path(export_multi_scenario_report(multi_report, output_dir / "multi-scenario-report.md")))
    generated.append(
        Path(
            export_multi_scenario_report(
                multi_report,
                output_dir / "multi-scenario-report.html",
                chart_output_path=output_dir / "multi-scenario-report-chart.html",
            )
        )
    )

    resilience_a = build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": True, "is_critical": True, "resilience_tier": "low"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 3,
            "impacted_customer_count": 165,
            "estimated_cost": 980.0,
            "severity_tier": "high",
        },
        restoration_report={
            "total_steps": 2,
            "stages": [
                {"step": 1, "repair_edge_id": "e1", "cumulative_restored_demand": 48.0},
                {"step": 2, "repair_edge_id": "e2", "cumulative_restored_demand": 103.0},
            ],
        },
        metadata={"scenario": "storm_a"},
    )
    resilience_b = build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": False, "is_critical": True, "resilience_tier": "high"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 2,
            "impacted_customer_count": 72,
            "estimated_cost": 640.0,
            "severity_tier": "medium",
        },
        restoration_report={
            "total_steps": 2,
            "stages": [
                {"step": 1, "repair_edge_id": "e3", "cumulative_restored_demand": 64.0},
                {"step": 2, "repair_edge_id": "e4", "cumulative_restored_demand": 120.0},
            ],
        },
        metadata={"scenario": "storm_b"},
    )

    generated.append(Path(export_resilience_summary_report(resilience_a, output_dir / "resilience-summary.json")))
    generated.append(Path(export_resilience_summary_report(resilience_a, output_dir / "resilience-summary.md")))
    generated.append(Path(export_resilience_summary_report(resilience_a, output_dir / "resilience-summary.html")))

    portfolio = build_resilience_portfolio_report({"storm_a": resilience_a, "storm_b": resilience_b})
    generated.append(Path(export_resilience_portfolio_report(portfolio, output_dir / "resilience-portfolio.json")))
    generated.append(Path(export_resilience_portfolio_report(portfolio, output_dir / "resilience-portfolio.csv")))
    generated.append(Path(export_resilience_portfolio_report(portfolio, output_dir / "resilience-portfolio.md")))
    generated.append(Path(export_resilience_portfolio_report(portfolio, output_dir / "resilience-portfolio.html")))

    comparison = build_comparison_report(output_dir=output_dir)
    comparison_paths = export_comparison_bundle(comparison, output_dir)
    generated.extend(Path(path) for path in comparison_paths.values())

    history_paths = export_benchmark_history(output_dir)
    generated.extend(Path(path) for path in history_paths.values())

    dashboard_paths = export_benchmark_dashboard_bundle(output_dir, min_speedup_ratio=1.05)
    generated.extend(Path(path) for path in dashboard_paths.values())
    return generated


def _stamp_text_provenance(path: Path, metadata: dict[str, Any]) -> None:
    marker_start = "<!-- geoprompt-provenance:start -->"
    marker_end = "<!-- geoprompt-provenance:end -->"
    text = path.read_text(encoding="utf-8")
    if marker_start in text and marker_end in text:
        prefix = text.split(marker_start, 1)[0].rstrip()
        text = prefix + "\n"
    payload = (
        f"{marker_start}\n"
        f"generated_at_utc: {metadata['generated_at_utc']}\n"
        f"package_version: {metadata['package_version']}\n"
        f"git_sha: {metadata['git_sha']}\n"
        f"source_digest: {metadata['source_digest']}\n"
        f"dataset_digest: {metadata['dataset_digest']}\n"
        f"{marker_end}\n"
    )
    path.write_text(text.rstrip() + "\n\n" + payload, encoding="utf-8")


def _stamp_json_provenance(path: Path, metadata: dict[str, Any]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(payload, dict):
        payload["_provenance"] = dict(metadata)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_manifest(output_dir: Path, metadata: dict[str, Any], source_inputs: list[Path]) -> dict[str, Any]:
    file_rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == PROVENANCE_MANIFEST_NAME:
            continue
        file_rows.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "metadata": dict(metadata),
        "source_inputs": [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in source_inputs if path.exists()],
        "files": file_rows,
    }


def generate_docs_artifacts(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    clean_output_dir: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if clean_output_dir and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_inputs = [path for path in DEFAULT_SOURCE_INPUTS if path.exists()]
    dataset_inputs = [path for path in source_inputs if path.parent.name == "data"]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": _package_version(),
        "git_sha": _git_sha(),
        "source_digest": _source_digest(source_inputs),
        "dataset_digest": _source_digest(dataset_inputs) if dataset_inputs else "none",
        "generator": "geoprompt.artifacts.generate_docs_artifacts",
    }

    _write_report_outputs(output_path)

    for path in output_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".md", ".html"}:
            _stamp_text_provenance(path, metadata)
        elif path.suffix.lower() == ".json":
            _stamp_json_provenance(path, metadata)

    manifest = _build_manifest(output_path, metadata, source_inputs)
    manifest_path = output_path / PROVENANCE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": str(output_path), "manifest": str(manifest_path), "file_count": len(manifest["files"])}


def check_docs_artifacts_freshness(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    manifest_path = output_path / PROVENANCE_MANIFEST_NAME
    if not manifest_path.exists():
        return {"ok": False, "reason": "missing_manifest", "manifest": str(manifest_path)}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": "invalid_manifest", "error": str(exc)}

    source_inputs = [path for path in DEFAULT_SOURCE_INPUTS if path.exists()]
    current_source_digest = _source_digest(source_inputs)
    recorded_digest = str(manifest.get("metadata", {}).get("source_digest", ""))
    if current_source_digest != recorded_digest:
        return {
            "ok": False,
            "reason": "source_digest_mismatch",
            "recorded": recorded_digest,
            "current": current_source_digest,
        }

    mismatches: list[str] = []
    for row in manifest.get("files", []):
        rel_path = Path(str(row.get("path", "")))
        file_path = output_path / rel_path
        if not file_path.exists():
            mismatches.append(f"missing:{rel_path.as_posix()}")
            continue
        current_hash = _file_sha256(file_path)
        if current_hash != str(row.get("sha256", "")):
            mismatches.append(f"hash:{rel_path.as_posix()}")

    if mismatches:
        return {
            "ok": False,
            "reason": "file_mismatch",
            "mismatches": mismatches,
        }
    return {"ok": True, "checked_files": len(manifest.get("files", [])), "manifest": str(manifest_path)}


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "PROVENANCE_MANIFEST_NAME",
    "check_docs_artifacts_freshness",
    "generate_docs_artifacts",
]
