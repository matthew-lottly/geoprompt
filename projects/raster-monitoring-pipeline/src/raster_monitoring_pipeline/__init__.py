from __future__ import annotations

from typing import Any

__all__ = ["build_change_report", "export_change_report", "load_grid"]


def __getattr__(name: str) -> Any:
	if name == "build_change_report":
		from raster_monitoring_pipeline.pipeline import build_change_report

		return build_change_report
	if name == "export_change_report":
		from raster_monitoring_pipeline.pipeline import export_change_report

		return export_change_report
	if name == "load_grid":
		from raster_monitoring_pipeline.pipeline import load_grid

		return load_grid
	raise AttributeError(f"module 'raster_monitoring_pipeline' has no attribute {name!r}")
