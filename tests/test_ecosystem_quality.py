from __future__ import annotations

import pytest

from geoprompt.cli import build_parser
from geoprompt.ecosystem import (
    build_workflow_wizard,
    deprecated_alias,
    get_migration_registry,
    get_plugin,
    get_recipe,
    list_plugins,
    register_plugin,
    register_recipe,
)


def test_plugin_registry_roundtrip() -> None:
    def sample_tool(value: int) -> int:
        return value + 1

    register_plugin(
        "sample_increment_tool",
        sample_tool,
        description="Increment a number",
        tags=["demo", "test"],
    )

    assert get_plugin("sample_increment_tool") is sample_tool
    names = [item["name"] for item in list_plugins()]
    assert "sample_increment_tool" in names


def test_recipe_registry_and_wizard() -> None:
    register_recipe(
        "stormwater_screening",
        ["read_data", "buffer_geometries", "scenario_comparison_engine"],
        persona="analyst",
        industry="water",
        description="Quick stormwater screening workflow",
    )

    recipe = get_recipe("stormwater_screening")
    assert recipe["persona"] == "analyst"

    wizard = build_workflow_wizard(
        "screen flooding risk near assets",
        persona="analyst",
        industry="water",
    )
    assert wizard["recommended_recipes"]
    assert "read_data" in wizard["suggested_steps"]

    contextual = build_workflow_wizard({"goal": "compare utility scenarios", "offline": True})
    assert contextual["goal"] == "compare utility scenarios"
    assert contextual["persona"] == "analyst"


def test_deprecated_alias_tracks_migration() -> None:
    @deprecated_alias("new_tool_name", remove_in="0.2.0")
    def old_tool_name(value: int) -> int:
        return value * 2

    with pytest.warns(DeprecationWarning):
        assert old_tool_name(4) == 8

    migrations = get_migration_registry()
    assert migrations["old_tool_name"]["new_name"] == "new_tool_name"


def test_cli_supports_wizard_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["wizard", "build a resilience workflow", "--persona", "analyst"])
    assert args.command == "wizard"
    assert args.persona == "analyst"
