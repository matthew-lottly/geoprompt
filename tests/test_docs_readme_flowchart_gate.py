from pathlib import Path


def test_readme_has_end_to_end_flow_heading() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "## End-to-End Flow" in text


def test_readme_has_mermaid_flowchart_block() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "```mermaid" in text
    assert "flowchart" in text


def test_readme_flowchart_has_required_nodes() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    required = [
        "Inputs",
        "Policy and Safety Gates",
        "safe expression",
        "Processing Modules",
        "Outputs",
    ]
    lowered = text.lower()
    for token in required:
        assert token.lower() in lowered, f"Missing flowchart token: {token}"
