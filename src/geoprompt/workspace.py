"""Workspace manifest and provenance helpers for GeoPrompt.

These utilities support lightweight lineage tracking for repeatable spatial
analysis runs without introducing a heavy project system.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def build_workspace_manifest(
    *,
    name: str,
    datasets: Sequence[dict[str, Any]] | None = None,
    steps: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight workspace manifest describing inputs and outputs."""
    dataset_list = [dict(item) for item in (datasets or [])]
    step_list = [str(step) for step in (steps or [])]
    output_list = [str(item) for item in (outputs or [])]
    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(dataset_list),
        "datasets": dataset_list,
        "steps": step_list,
        "outputs": output_list,
        "metadata": dict(metadata or {}),
    }


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render a workspace manifest as Markdown."""
    lines = [
        f"# Workspace Manifest — {manifest.get('name', 'unknown')}",
        "",
        f"- Created: {manifest.get('created_at', '')}",
        f"- Dataset count: {manifest.get('dataset_count', 0)}",
        "",
        "## Datasets",
    ]
    for dataset in manifest.get("datasets", []):
        lines.append(f"- {dataset.get('name', 'unknown')} — {dataset.get('path', '')} ({dataset.get('crs', 'n/a')})")
    lines.extend(["", "## Steps"])
    for step in manifest.get("steps", []):
        lines.append(f"- {step}")
    lines.extend(["", "## Outputs"])
    for output in manifest.get("outputs", []):
        lines.append(f"- {output}")
    return "\n".join(lines).strip() + "\n"


def export_provenance_bundle(output_dir: str | Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Write provenance artifacts for a workspace manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "manifest.json"
    md_path = out / "manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md_path.write_text(render_manifest_markdown(manifest), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


__all__ = ["build_workspace_manifest", "export_provenance_bundle", "render_manifest_markdown"]
