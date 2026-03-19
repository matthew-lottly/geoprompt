from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["__version__", "build_html_report", "build_markdown_report", "compute_summary", "export_reports"]


def __getattr__(name: str) -> Any:
	if name == "build_html_report":
		from environmental_monitoring_analytics.reporting import build_html_report

		return build_html_report
	if name == "build_markdown_report":
		from environmental_monitoring_analytics.reporting import build_markdown_report

		return build_markdown_report
	if name == "compute_summary":
		from environmental_monitoring_analytics.reporting import compute_summary

		return compute_summary
	if name == "export_reports":
		from environmental_monitoring_analytics.reporting import export_reports

		return export_reports
	raise AttributeError(f"module 'environmental_monitoring_analytics' has no attribute {name!r}")