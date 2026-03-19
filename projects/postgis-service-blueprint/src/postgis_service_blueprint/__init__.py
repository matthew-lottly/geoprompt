from __future__ import annotations

from typing import Any

__all__ = ["build_service_blueprint", "export_seed_sql", "export_service_blueprint", "load_layer_features"]


def __getattr__(name: str) -> Any:
	if name == "build_service_blueprint":
		from postgis_service_blueprint.blueprint import build_service_blueprint

		return build_service_blueprint
	if name == "export_seed_sql":
		from postgis_service_blueprint.blueprint import export_seed_sql

		return export_seed_sql
	if name == "export_service_blueprint":
		from postgis_service_blueprint.blueprint import export_service_blueprint

		return export_service_blueprint
	if name == "load_layer_features":
		from postgis_service_blueprint.blueprint import load_layer_features

		return load_layer_features
	raise AttributeError(f"module 'postgis_service_blueprint' has no attribute {name!r}")
