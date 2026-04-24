from __future__ import annotations

from pathlib import Path

import geoprompt

STABLE_MODULES = [
    "frame",
    "geometry",
    "io",
    "equations",
    "table",
    "tools",
    "compare",
    "interop",
]

ADVANCED_MODULES = [
    "raster",
    "stats",
    "temporal",
    "topology",
    "workspace",
    "service",
    "enterprise",
    "geocoding",
    "cartography",
    "ecosystem",
    "ai",
    "data_management",
]

PUBLIC_SYMBOLS = [
    "GeoPromptFrame",
    "GroupedGeoPromptFrame",
    "PromptTable",
    "capability_report",
]


def _module_path_exists(module_name: str) -> bool:
    root = Path("src/geoprompt")
    return (root / f"{module_name}.py").exists() or (root / module_name).is_dir()


def test_reference_api_modules_exist_in_source_tree() -> None:
    for name in STABLE_MODULES + ADVANCED_MODULES:
        assert _module_path_exists(name), f"Module listed in docs/reference-api.md is missing: {name}"


def test_reference_api_symbols_are_exported() -> None:
    for symbol in PUBLIC_SYMBOLS:
        assert hasattr(geoprompt, symbol), f"Public API symbol missing from geoprompt: {symbol}"


def test_capability_report_contract_shape() -> None:
    report = geoprompt.capability_report()
    expected = {
        "schema_version",
        "enabled",
        "disabled",
        "degraded",
        "disabled_reasons",
        "degraded_reasons",
        "fallback_policy",
        "package_version",
        "optional_dependency_versions",
        "checked_at_utc",
    }
    assert expected.issubset(set(report.keys()))
