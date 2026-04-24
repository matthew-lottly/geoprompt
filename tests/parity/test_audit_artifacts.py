from __future__ import annotations

from pathlib import Path

from .build_parity_audit_artifacts import build_audit_artifacts


def test_build_audit_artifacts_generates_tables_and_graphs(tmp_path: Path) -> None:
    payload = build_audit_artifacts(private_dir=tmp_path / "parity-audit", run_tests=False)

    assert payload["symbol_count"] > 0
    assert (tmp_path / "parity-audit" / "parity-class-counts.csv").exists()
    assert (tmp_path / "parity-audit" / "parity-assignment-details.csv").exists()
    assert (tmp_path / "parity-audit" / "parity-class-counts.svg").exists()
    assert (tmp_path / "parity-audit" / "parity-audit-summary.md").exists()
    assert (tmp_path / "parity-audit" / "parity-audit-summary.json").exists()
