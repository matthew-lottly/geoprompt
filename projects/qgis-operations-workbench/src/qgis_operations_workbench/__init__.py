from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from qgis_operations_workbench.workbench import (
		build_workbench_pack,
		export_geopackage,
		export_workbench_pack,
		load_inspection_routes,
		load_station_features,
	)

__all__ = [
	"build_workbench_pack",
	"export_geopackage",
	"export_workbench_pack",
	"load_inspection_routes",
	"load_station_features",
]


def __getattr__(name: str) -> Any:
	if name == "build_workbench_pack":
		from qgis_operations_workbench.workbench import build_workbench_pack

		return build_workbench_pack
	if name == "export_geopackage":
		from qgis_operations_workbench.workbench import export_geopackage

		return export_geopackage
	if name == "export_workbench_pack":
		from qgis_operations_workbench.workbench import export_workbench_pack

		return export_workbench_pack
	if name == "load_inspection_routes":
		from qgis_operations_workbench.workbench import load_inspection_routes

		return load_inspection_routes
	if name == "load_station_features":
		from qgis_operations_workbench.workbench import load_station_features

		return load_station_features
	raise AttributeError(f"module 'qgis_operations_workbench' has no attribute {name!r}")
