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


class GeoPromptWorkspace:
    """Small registry class for datasets, outputs, and provenance artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._datasets: list[dict[str, Any]] = []

    def register_layer(
        self,
        name: str,
        *,
        path: str,
        crs: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset = {
            "name": str(name),
            "path": str(path),
            "crs": crs,
            "metadata": dict(metadata or {}),
        }
        self._datasets.append(dataset)
        return dataset

    def build_manifest(
        self,
        *,
        steps: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        return build_workspace_manifest(
            name=name or self.root.name,
            datasets=self._datasets,
            steps=steps,
            outputs=outputs,
            metadata=metadata,
        )

    def save_manifest(
        self,
        manifest: dict[str, Any] | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        payload = manifest or self.build_manifest()
        target = Path(output_dir) if output_dir is not None else self.root / "provenance"
        return export_provenance_bundle(target, payload)


class LineageTracker:
    """Track step-by-step lineage for a multi-step analysis run.

    Records each processing step with inputs, outputs, and parameters so that
    the full provenance chain can be reconstructed.

    Usage::

        tracker = LineageTracker()
        tracker.add_step("buffer", inputs=["parcels.shp"], outputs=["buffered.geojson"], params={"distance": 100})
        report = tracker.report()
    """

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        *,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Record a processing step."""
        self._steps.append({
            "step": len(self._steps) + 1,
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": list(inputs or []),
            "outputs": list(outputs or []),
            "params": dict(params or {}),
        })

    def report(self) -> list[dict[str, Any]]:
        """Return the full lineage log."""
        return list(self._steps)

    def to_markdown(self) -> str:
        """Render the lineage as Markdown."""
        lines = ["# Lineage Report", ""]
        for step in self._steps:
            lines.append(f"## Step {step['step']}: {step['name']}")
            lines.append(f"- Timestamp: {step['timestamp']}")
            if step["inputs"]:
                lines.append(f"- Inputs: {', '.join(step['inputs'])}")
            if step["outputs"]:
                lines.append(f"- Outputs: {', '.join(step['outputs'])}")
            if step["params"]:
                lines.append(f"- Params: {step['params']}")
            lines.append("")
        return "\n".join(lines)


class JobSpec:
    """A reusable, parameterized job specification for batch execution.

    Usage::

        spec = JobSpec("buffer_and_clip", params={"distance": 100})
        spec.add_step("buffer", callable_name="buffer_geometries")
        spec.add_step("clip", callable_name="clip_geometries")
        manifest = spec.to_manifest()
    """

    def __init__(self, name: str, *, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = dict(params or {})
        self._steps: list[dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        *,
        callable_name: str | None = None,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Add a step to the job specification."""
        self._steps.append({
            "name": name,
            "callable": callable_name,
            "inputs": list(inputs or []),
            "outputs": list(outputs or []),
            "params": dict(params or {}),
        })

    def to_manifest(self) -> dict[str, Any]:
        """Export the job spec as a serializable manifest dict."""
        return {
            "job_name": self.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "global_params": self.params,
            "steps": list(self._steps),
            "step_count": len(self._steps),
        }

    def save(self, path: str | Path) -> str:
        """Write the job spec to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_manifest(), indent=2), encoding="utf-8")
        return str(p)


geopromptworkspace = GeoPromptWorkspace
lineagetracker = LineageTracker
jobspec = JobSpec


__all__ = [
    "GeoPromptWorkspace",
    "JobSpec",
    "LineageTracker",
    "build_workspace_manifest",
    "export_provenance_bundle",
    "geopromptworkspace",
    "jobspec",
    "lineagetracker",
    "render_manifest_markdown",
]
