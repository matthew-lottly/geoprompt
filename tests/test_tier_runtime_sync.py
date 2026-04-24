from __future__ import annotations

from pathlib import Path

import pytest

from geoprompt._tier_metadata import (
    TIER_BETA,
    TIER_EXPERIMENTAL,
    TIER_SIMULATION,
    add_tier_info_to_docstring,
    warn_if_non_stable,
)


def test_maturity_terms_exist_in_docs_for_runtime_tiers() -> None:
    text = Path("docs/MATURITY_MATRIX.md").read_text(encoding="utf-8").lower()
    for token in ("stable", "beta", "experimental", "simulation-only"):
        assert token in text


def test_runtime_warning_categories_match_tier_contract() -> None:
    with pytest.warns(FutureWarning):
        warn_if_non_stable("gwr")

    with pytest.warns(FutureWarning):
        warn_if_non_stable("explain_pipeline")

    with pytest.warns(UserWarning):
        warn_if_non_stable("wms_capabilities_document")


def test_docstring_tier_badges_include_declared_tier() -> None:
    doc = add_tier_info_to_docstring("Example", TIER_BETA)
    assert "[BETA]" in doc

    doc = add_tier_info_to_docstring("Example", TIER_EXPERIMENTAL)
    assert "[EXPERIMENTAL]" in doc

    doc = add_tier_info_to_docstring("Example", TIER_SIMULATION)
    assert "[SIMULATION_ONLY]" in doc
