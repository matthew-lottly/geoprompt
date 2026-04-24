from __future__ import annotations

from pathlib import Path

from geoprompt.quality import export_simulation_symbol_labels, simulation_symbol_labels_manifest


def test_simulation_symbol_manifest_includes_ml_and_standards() -> None:
    manifest = simulation_symbol_labels_manifest()
    symbols = manifest["symbols"]
    assert manifest["count"] > 0
    modules = {row["module"] for row in symbols}
    assert "ml" in modules
    assert "standards" in modules


def test_export_simulation_symbol_labels_writes_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "simulation-labels.md"
    written = export_simulation_symbol_labels(output_path)
    text = Path(written).read_text(encoding="utf-8")
    assert text.startswith("# Simulation and Deprecation Labels\n")
    assert "| Module | Symbol | Label | Deprecation | Guidance |" in text
