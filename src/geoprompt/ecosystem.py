"""Plugin, recipe, migration, and workflow-wizard helpers for GeoPrompt.

This module supports third-party extension registration, reusable recipe
catalogs, and guided no-code workflow planning from a simple goal.
"""
from __future__ import annotations

import textwrap
import warnings
from functools import wraps
from typing import Any, Callable, Sequence


PluginFunc = Callable[..., Any]
Record = dict[str, Any]

_PLUGIN_REGISTRY: dict[str, dict[str, Any]] = {}
_RECIPE_REGISTRY: dict[str, dict[str, Any]] = {}
_MIGRATION_REGISTRY: dict[str, dict[str, Any]] = {}


def register_plugin(
    name: str,
    func: PluginFunc,
    *,
    description: str = "",
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Register a third-party tool or domain-specific extension."""
    if not callable(func):
        raise TypeError("func must be callable")
    entry = {
        "name": name,
        "func": func,
        "description": description or (func.__doc__ or "").strip().splitlines()[0] if (func.__doc__ or "").strip() else "",
        "tags": sorted(set(str(t).strip().lower() for t in (tags or []))),
    }
    _PLUGIN_REGISTRY[name] = entry
    return {k: v for k, v in entry.items() if k != "func"}


def get_plugin(name: str) -> PluginFunc:
    """Return a registered plugin callable by name."""
    if name not in _PLUGIN_REGISTRY:
        raise KeyError(name)
    return _PLUGIN_REGISTRY[name]["func"]


def list_plugins(*, tag: str | None = None) -> list[dict[str, Any]]:
    """List registered plugin metadata."""
    items = []
    wanted = tag.strip().lower() if tag else None
    for entry in _PLUGIN_REGISTRY.values():
        if wanted and wanted not in entry["tags"]:
            continue
        items.append({k: v for k, v in entry.items() if k != "func"})
    return sorted(items, key=lambda item: item["name"])


def run_plugin(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute a registered plugin."""
    return get_plugin(name)(*args, **kwargs)


def register_recipe(
    name: str,
    steps: Sequence[str],
    *,
    persona: str = "general",
    industry: str = "general",
    description: str = "",
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Register a reusable analysis recipe."""
    entry = {
        "name": name,
        "steps": list(steps),
        "persona": persona,
        "industry": industry,
        "description": description,
        "tags": sorted(set(str(t).strip().lower() for t in (tags or []))),
    }
    _RECIPE_REGISTRY[name] = entry
    return dict(entry)


def get_recipe(name: str) -> dict[str, Any]:
    """Fetch a registered recipe by name."""
    if name not in _RECIPE_REGISTRY:
        raise KeyError(name)
    return dict(_RECIPE_REGISTRY[name])


def list_recipes(
    *,
    persona: str | None = None,
    industry: str | None = None,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    """List recipes with optional persona and industry filtering."""
    results: list[dict[str, Any]] = []
    wanted_tag = tag.strip().lower() if tag else None
    for recipe in _RECIPE_REGISTRY.values():
        if persona and recipe["persona"] not in {persona, "general"}:
            continue
        if industry and recipe["industry"] not in {industry, "general"}:
            continue
        if wanted_tag and wanted_tag not in recipe["tags"]:
            continue
        results.append(dict(recipe))
    return sorted(results, key=lambda item: item["name"])


def run_recipe(name: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an execution-ready recipe plan."""
    recipe = get_recipe(name)
    return {
        "name": recipe["name"],
        "persona": recipe["persona"],
        "industry": recipe["industry"],
        "steps": recipe["steps"],
        "context": dict(context or {}),
    }


def register_migration(
    old_name: str,
    new_name: str,
    *,
    remove_in: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Record a migration path from an old API name to a new one."""
    entry = {
        "old_name": old_name,
        "new_name": new_name,
        "remove_in": remove_in,
        "note": note,
    }
    _MIGRATION_REGISTRY[old_name] = entry
    return dict(entry)


def get_migration_registry() -> dict[str, dict[str, Any]]:
    """Return all registered migration aliases."""
    return {k: dict(v) for k, v in _MIGRATION_REGISTRY.items()}


def deprecated_alias(
    new_name: str,
    *,
    remove_in: str | None = None,
    note: str = "",
) -> Callable[[PluginFunc], PluginFunc]:
    """Decorator that warns callers and records a migration alias."""

    def decorator(func: PluginFunc) -> PluginFunc:
        register_migration(func.__name__, new_name, remove_in=remove_in, note=note)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = f"{func.__name__} is deprecated; use {new_name} instead."
            if remove_in:
                msg += f" Planned removal in {remove_in}."
            if note:
                msg += f" {note}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def recommend_recipes(
    goal: str,
    *,
    persona: str = "general",
    industry: str = "general",
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Recommend recipes based on a free-text goal plus persona and industry."""
    goal_tokens = {
        token.strip().lower()
        for token in goal.replace("-", " ").replace("_", " ").split()
        if token.strip()
    }

    scored: list[tuple[float, dict[str, Any]]] = []
    for recipe in list_recipes(persona=persona, industry=industry):
        text = " ".join(
            [recipe["name"], recipe.get("description", ""), recipe.get("persona", ""), recipe.get("industry", ""), " ".join(recipe.get("steps", []))]
        ).lower()
        score = sum(1.0 for token in goal_tokens if token in text)
        if recipe.get("persona") == persona:
            score += 2.0
        if recipe.get("industry") == industry:
            score += 2.0
        scored.append((score, recipe))

    scored.sort(key=lambda pair: (pair[0], pair[1]["name"]), reverse=True)
    return [dict(recipe) for score, recipe in scored[:max_results] if score > 0] or list_recipes(persona=persona, industry=industry)[:max_results]


def build_workflow_wizard(
    goal: str | dict[str, Any],
    *,
    persona: str = "analyst",
    industry: str = "general",
) -> dict[str, Any]:
    """Build a no-code workflow suggestion from a plain-English goal."""
    context: dict[str, Any] = dict(goal) if isinstance(goal, dict) else {}
    goal_text = str(context.get("goal") or context.get("question") or goal)
    persona_value = str(context.get("persona") or persona)
    industry_value = str(context.get("industry") or industry)

    recommended = recommend_recipes(goal_text, persona=persona_value, industry=industry_value)
    if recommended:
        suggested_steps = []
        seen: set[str] = set()
        for step in recommended[0].get("steps", []):
            if step not in seen:
                suggested_steps.append(step)
                seen.add(step)
    else:
        suggested_steps = _fallback_steps(goal_text)

    summary = textwrap.shorten(goal_text.strip(), width=120, placeholder="...")
    return {
        "goal": summary,
        "persona": persona_value,
        "industry": industry_value,
        "context": context,
        "recommended_recipes": recommended,
        "suggested_steps": suggested_steps,
        "message": f"Suggested workflow for {persona_value} work in {industry_value}: {' -> '.join(suggested_steps)}",
    }


def _fallback_steps(goal: str) -> list[str]:
    goal_l = goal.lower()
    if any(token in goal_l for token in ["risk", "screen", "compare", "scenario"]):
        return ["read_data", "profile_dataset", "scenario_comparison_engine", "prompt_to_report"]
    if any(token in goal_l for token in ["route", "travel", "fleet"]):
        return ["read_data", "get_travel_mode", "stop_sequence_optimize", "route_directions_narrative"]
    if any(token in goal_l for token in ["map", "dashboard", "report"]):
        return ["read_data", "quickplot", "interactive_dashboard", "export_map_image"]
    return ["read_data", "quality_control_scan", "write_data"]


def _seed_default_recipes() -> None:
    defaults = [
        {
            "name": "resilience_screening",
            "steps": ["read_data", "profile_dataset", "scenario_comparison_engine", "prompt_to_report"],
            "persona": "analyst",
            "industry": "infrastructure",
            "description": "Baseline resilience screening and scenario comparison.",
            "tags": ["resilience", "scenario"],
        },
        {
            "name": "field_routing",
            "steps": ["read_data", "get_travel_mode", "stop_sequence_optimize", "route_directions_narrative"],
            "persona": "operations",
            "industry": "utilities",
            "description": "Crew routing and stop sequencing for field operations.",
            "tags": ["routing", "mobility"],
        },
        {
            "name": "map_briefing_pack",
            "steps": ["read_data", "quickplot", "interactive_dashboard", "export_map_image"],
            "persona": "executive",
            "industry": "general",
            "description": "Fast map and dashboard outputs for briefings.",
            "tags": ["maps", "reporting"],
        },
    ]
    for item in defaults:
        if item["name"] not in _RECIPE_REGISTRY:
            register_recipe(**item)


_seed_default_recipes()


__all__ = [
    "build_workflow_wizard",
    "deprecated_alias",
    "get_migration_registry",
    "get_plugin",
    "get_recipe",
    "list_plugins",
    "list_recipes",
    "recommend_recipes",
    "register_migration",
    "register_plugin",
    "register_recipe",
    "run_plugin",
    "run_recipe",
]
