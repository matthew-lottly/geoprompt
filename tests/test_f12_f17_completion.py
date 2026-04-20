from __future__ import annotations

import json
from pathlib import Path

from geoprompt.quality import (
    documentation_asset_manifest,
    placeholder_inventory,
    release_readiness_report,
    subsystem_maturity_matrix,
)
from geoprompt.security import release_security_gates, support_window_policy


def test_f12_security_and_supply_chain_trust_material_is_published() -> None:
    gates = release_security_gates()
    policy = support_window_policy()
    security_doc = Path("SECURITY.md").read_text(encoding="utf-8")
    threat_model = Path("docs/threat-model.md").read_text(encoding="utf-8")

    assert {"sbom", "signing", "provenance", "secrets_scan"}.issubset(gates["release_gates"])
    assert policy["lts_style_support"] is True
    assert "CVE triage" in security_doc
    assert "attack surfaces" in threat_model.lower()


def test_f13_public_proof_and_adoption_story_are_visible() -> None:
    scorecard = Path("docs/competitive-scorecard.md").read_text(encoding="utf-8")
    release_evidence = Path("docs/release-evidence.md").read_text(encoding="utf-8")
    case_study = Path("docs/case-studies/utility-resilience.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "GeoPandas" in scorecard and "ArcPy" in scorecard
    assert "evidence" in release_evidence.lower()
    assert "outage" in case_study.lower() or "resilience" in case_study.lower()
    assert "Why choose GeoPrompt" in readme


def test_f14_f17_documentation_assets_and_realism_manifest_are_tracked() -> None:
    manifest = documentation_asset_manifest("docs/figures-manifest.json")
    style_guide = Path("docs/STYLE_GUIDE.md").read_text(encoding="utf-8")
    manifest_json = json.loads(Path("docs/figures-manifest.json").read_text(encoding="utf-8"))

    assert manifest["passed"] is True
    assert manifest["count"] >= 3
    assert "figure sizing" in style_guide.lower()
    assert all("generation_script" in item for item in manifest_json["figures"])


def test_f15_f16_architecture_and_placeholder_audits_are_available() -> None:
    matrix = subsystem_maturity_matrix()
    placeholders = placeholder_inventory(["src/geoprompt/geoprocessing.py"])
    architecture_doc = Path("docs/API_ARCHITECTURE.md").read_text(encoding="utf-8")
    maturity_doc = Path("docs/MATURITY_MATRIX.md").read_text(encoding="utf-8")

    assert "stable_core" in matrix and "experimental_surface" in matrix
    names = {item["name"] for item in placeholders["functions"]}
    assert "notify_email_stub" in names
    assert "serverless_endpoint_stub" in names
    assert "module boundaries" in architecture_doc.lower()
    assert "simulation-only" in maturity_doc.lower()


def test_release_readiness_report_includes_trust_and_realism_checks() -> None:
    report = release_readiness_report(["src/geoprompt/quality.py", "src/geoprompt/security.py"])

    assert "placeholder_audit" in report
    assert "docs_assets" in report
    assert "evidence_realism" in report
    assert report["release_stage"] in {"beta", "release-candidate"}
