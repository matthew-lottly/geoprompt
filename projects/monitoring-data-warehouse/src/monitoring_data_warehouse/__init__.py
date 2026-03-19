from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["__version__", "build_warehouse"]


def __getattr__(name: str) -> Any:
	if name == "build_warehouse":
		from monitoring_data_warehouse.builder import build_warehouse

		return build_warehouse
	raise AttributeError(f"module 'monitoring_data_warehouse' has no attribute {name!r}")