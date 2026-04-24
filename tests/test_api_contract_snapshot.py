from __future__ import annotations

from pathlib import Path

from geoprompt.api_contract import build_public_api_contract, load_snapshot


_SNAPSHOT = Path("data/public_api_contract.snapshot.json")


def test_public_api_contract_snapshot_matches_generated_contract() -> None:
    assert _SNAPSHOT.exists(), "API contract snapshot file is missing"
    generated = build_public_api_contract()
    snapshot = load_snapshot(_SNAPSHOT)
    assert generated == snapshot


def test_public_api_contract_entries_have_resolved_tiers() -> None:
    contract = build_public_api_contract()
    valid_tiers = {"stable", "beta", "experimental", "simulation_only"}
    for item in contract["symbols"]:
        assert item["tier"] in valid_tiers
        assert isinstance(item["tier_source"], str)
        assert item["tier_source"].strip() != ""
