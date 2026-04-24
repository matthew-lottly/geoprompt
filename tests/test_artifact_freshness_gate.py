from __future__ import annotations

import json
from pathlib import Path

from geoprompt.artifacts import check_docs_artifacts_freshness, generate_docs_artifacts


def test_docs_artifacts_generation_writes_manifest_and_passes_check(tmp_path: Path) -> None:
    result = generate_docs_artifacts(tmp_path, clean_output_dir=True)
    assert Path(result["manifest"]).exists()
    check = check_docs_artifacts_freshness(tmp_path)
    assert check["ok"] is True
    assert check["checked_files"] > 0


def test_docs_artifacts_check_fails_after_file_tamper(tmp_path: Path) -> None:
    generate_docs_artifacts(tmp_path, clean_output_dir=True)
    target = tmp_path / "scenario-report.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nmanual drift\n", encoding="utf-8")

    check = check_docs_artifacts_freshness(tmp_path)
    assert check["ok"] is False
    assert check["reason"] == "file_mismatch"


def test_provenance_is_embedded_in_json_outputs(tmp_path: Path) -> None:
    generate_docs_artifacts(tmp_path, clean_output_dir=True)
    payload = json.loads((tmp_path / "scenario-report.json").read_text(encoding="utf-8"))
    provenance = payload.get("_provenance", {})
    assert provenance.get("generator") == "geoprompt.artifacts.generate_docs_artifacts"
    assert "source_digest" in provenance
