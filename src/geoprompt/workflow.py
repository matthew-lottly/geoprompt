"""Task-oriented workflow facades and fluent configuration objects.

These helpers provide a composable object-oriented surface for common
GeoPrompt workflows while preserving the underlying function APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .frame import GeoPromptFrame
from .io import read_data


@dataclass
class IOConfig:
    """Fluent configuration for I/O-driven workflow steps."""

    source: Any
    where: Any | None = None
    geometry: str = "geometry"
    crs: str | None = None

    def with_where(self, where: Any) -> "IOConfig":
        self.where = where
        return self

    def with_geometry(self, geometry: str) -> "IOConfig":
        self.geometry = geometry
        return self

    def with_crs(self, crs: str | None) -> "IOConfig":
        self.crs = crs
        return self


@dataclass
class JoinConfig:
    """Fluent configuration for frame join workflows."""

    on: str | None = None
    predicate: str = "intersects"
    how: str = "inner"

    def with_predicate(self, predicate: str) -> "JoinConfig":
        self.predicate = predicate
        return self

    def with_how(self, how: str) -> "JoinConfig":
        self.how = how
        return self


@dataclass
class ReportConfig:
    """Fluent configuration for report output."""

    title: str = "GeoPrompt Workflow Report"
    format: str = "markdown"
    output_path: str | Path | None = None

    def with_title(self, title: str) -> "ReportConfig":
        self.title = title
        return self

    def with_format(self, format: str) -> "ReportConfig":
        self.format = format
        return self

    def with_output(self, output_path: str | Path | None) -> "ReportConfig":
        self.output_path = output_path
        return self


class DataPipeline:
    """Composable data workflow facade for read/filter/join operations."""

    def __init__(self) -> None:
        self._frame: GeoPromptFrame | None = None

    def load(self, source: Any, *, config: IOConfig | None = None) -> "DataPipeline":
        cfg = config or IOConfig(source=source)
        self._frame = read_data(
            cfg.source,
            where=cfg.where,
            geometry=cfg.geometry,
            crs=cfg.crs,
        )
        return self

    def filter(self, expression: str) -> "DataPipeline":
        if self._frame is None:
            raise ValueError("pipeline has no loaded frame; call load(...) first")
        self._frame = self._frame.query(expression)
        return self

    def join(self, other: GeoPromptFrame, *, config: JoinConfig | None = None) -> "DataPipeline":
        if self._frame is None:
            raise ValueError("pipeline has no loaded frame; call load(...) first")
        cfg = config or JoinConfig()
        self._frame = self._frame.spatial_join(other, predicate=cfg.predicate, how=cfg.how)
        return self

    def frame(self) -> GeoPromptFrame:
        if self._frame is None:
            raise ValueError("pipeline has no loaded frame; call load(...) first")
        return self._frame


class ScenarioRunner:
    """Task-oriented scenario facade with deterministic result collection."""

    def __init__(self) -> None:
        self._scenarios: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []

    def add(self, name: str, runner: Callable[..., Any], **params: Any) -> "ScenarioRunner":
        if not callable(runner):
            raise TypeError("runner must be callable")
        self._scenarios.append((name, runner, params))
        return self

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, runner, params in self._scenarios:
            results[name] = runner(**params)
        return results


class ReportBuilder:
    """Facade for building portable workflow summaries in markdown/json/text."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()

    def build(self, data: dict[str, Any]) -> str:
        if self.config.format == "json":
            text = json.dumps(data, indent=2, sort_keys=True)
        elif self.config.format == "text":
            text = "\n".join(f"{k}: {v}" for k, v in sorted(data.items()))
        else:
            lines = [f"# {self.config.title}", ""]
            for key, value in sorted(data.items()):
                lines.append(f"- **{key}**: {value}")
            text = "\n".join(lines)

        if self.config.output_path is not None:
            Path(self.config.output_path).write_text(text, encoding="utf-8")
        return text


__all__ = [
    "DataPipeline",
    "IOConfig",
    "JoinConfig",
    "ReportBuilder",
    "ReportConfig",
    "ScenarioRunner",
]
