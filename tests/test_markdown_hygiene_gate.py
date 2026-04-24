from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_private_has_only_single_tracker_markdown() -> None:
    private_md = sorted((ROOT / "private").glob("*.md"))
    names = [path.name for path in private_md]
    assert names == ["GEOPROMPT_PLATFORM_PARITY.private.md"], (
        "private/ must only contain the active tracker markdown file; found: "
        + ", ".join(names)
    )


def test_outputs_contains_no_committed_markdown_files() -> None:
    output_md = sorted((ROOT / "outputs").rglob("*.md"))
    assert not output_md, (
        "outputs/ should not contain committed markdown artifacts; found:\n"
        + "\n".join(str(path.relative_to(ROOT)) for path in output_md)
    )
