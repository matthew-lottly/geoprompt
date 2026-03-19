from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from environmental_time_series_lab.lab import TimeSeriesLab

__all__ = ["TimeSeriesLab", "build_time_series_report", "export_time_series_report", "load_histories"]


def __getattr__(name: str) -> Any:
	if name == "TimeSeriesLab":
		from environmental_time_series_lab.lab import TimeSeriesLab

		return TimeSeriesLab
	if name == "build_time_series_report":
		from environmental_time_series_lab.lab import build_time_series_report

		return build_time_series_report
	if name == "export_time_series_report":
		from environmental_time_series_lab.lab import export_time_series_report

		return export_time_series_report
	if name == "load_histories":
		from environmental_time_series_lab.lab import load_histories

		return load_histories
	raise AttributeError(f"module 'environmental_time_series_lab' has no attribute {name!r}")