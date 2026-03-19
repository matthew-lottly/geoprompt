from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from monitoring_anomaly_detection.pipeline import AnomalyDetectionWorkflow

__all__ = ["AnomalyDetectionWorkflow", "build_anomaly_report", "export_anomaly_report", "load_observations"]


def __getattr__(name: str) -> Any:
	if name == "AnomalyDetectionWorkflow":
		from monitoring_anomaly_detection.pipeline import AnomalyDetectionWorkflow

		return AnomalyDetectionWorkflow
	if name == "build_anomaly_report":
		from monitoring_anomaly_detection.pipeline import build_anomaly_report

		return build_anomaly_report
	if name == "export_anomaly_report":
		from monitoring_anomaly_detection.pipeline import export_anomaly_report

		return export_anomaly_report
	if name == "load_observations":
		from monitoring_anomaly_detection.pipeline import load_observations

		return load_observations
	raise AttributeError(f"module 'monitoring_anomaly_detection' has no attribute {name!r}")