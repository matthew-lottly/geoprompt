from __future__ import annotations

from pathlib import Path


def test_dependency_compatibility_matrix_documents_warning_watchlist() -> None:
    matrix = Path("docs/dependency-compatibility-matrix.md")
    text = matrix.read_text(encoding="utf-8")

    assert "np.find_common_type is deprecated" in text
    assert "matplotlib" in text
    assert "Monitoring Tasks" in text
