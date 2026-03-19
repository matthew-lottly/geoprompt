from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from station_forecasting_workbench.workbench import ForecastWorkbench

__all__ = ["ForecastWorkbench", "build_forecast_report", "export_forecast_report", "load_histories"]


def __getattr__(name: str) -> Any:
	if name == "ForecastWorkbench":
		from station_forecasting_workbench.workbench import ForecastWorkbench

		return ForecastWorkbench
	if name == "build_forecast_report":
		from station_forecasting_workbench.workbench import build_forecast_report

		return build_forecast_report
	if name == "export_forecast_report":
		from station_forecasting_workbench.workbench import export_forecast_report

		return export_forecast_report
	if name == "load_histories":
		from station_forecasting_workbench.workbench import load_histories

		return load_histories
	raise AttributeError(f"module 'station_forecasting_workbench' has no attribute {name!r}")