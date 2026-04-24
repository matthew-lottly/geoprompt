"""Public API contract snapshot and tier-resolution helpers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ._tier_metadata import TIER_METADATA, TIER_STABLE, TierLevel

_MODULE_DEFAULT_TIERS: dict[str, TierLevel] = {
    "ai": "beta",
    "ecosystem": "beta",
    "enterprise": "beta",
    "geoprocessing": "beta",
    "raster": "beta",
    "service": "beta",
    "visualization": "beta",
}


def _default_tier_for_module(module_name: str) -> TierLevel:
    return _MODULE_DEFAULT_TIERS.get(module_name, TIER_STABLE)


def resolve_public_symbol_tier(name: str, module_name: str) -> tuple[TierLevel, str]:
    """Resolve the maturity tier for a public symbol and its declaration source."""
    exact = f"{module_name}.{name}"
    if exact in TIER_METADATA:
        return TIER_METADATA[exact], exact

    candidates = [key for key in TIER_METADATA if key.endswith(f".{name}")]
    if len(candidates) == 1:
        key = candidates[0]
        return TIER_METADATA[key], key
    if len(candidates) > 1:
        raise ValueError(f"ambiguous tier declarations for {name}: {sorted(candidates)}")

    default_tier = _default_tier_for_module(module_name)
    return default_tier, f"{module_name}.*"


def build_public_api_contract() -> dict[str, Any]:
    """Build a machine-readable contract for the package public surface."""
    gp = importlib.import_module("geoprompt")

    entries: list[dict[str, Any]] = []
    for name in sorted(set(getattr(gp, "__all__", []))):
        if not hasattr(gp, name):
            continue
        symbol = getattr(gp, name)
        module = getattr(symbol, "__module__", "geoprompt")
        module_name = module.removeprefix("geoprompt.").split(".")[0]
        tier, tier_source = resolve_public_symbol_tier(name, module_name)
        deprecation = getattr(symbol, "_geoprompt_deprecation", None)
        entries.append(
            {
                "name": name,
                "module": module_name,
                "qualname": getattr(symbol, "__qualname__", name),
                "tier": tier,
                "tier_source": tier_source,
                "deprecated": bool(deprecation),
                "deprecation": deprecation or None,
            }
        )

    return {
        "schema_version": "1.0",
        "symbol_count": len(entries),
        "symbols": entries,
    }


def load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_snapshot(path: str | Path) -> dict[str, Any]:
    contract = build_public_api_contract()
    Path(path).write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


__all__ = [
    "build_public_api_contract",
    "load_snapshot",
    "resolve_public_symbol_tier",
    "save_snapshot",
]
