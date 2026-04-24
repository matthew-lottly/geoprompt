from pathlib import Path


TRACKED_DOCS = [
    Path("README.md"),
    Path("PUBLISHING.md"),
    Path("CHANGELOG.md"),
    Path("docs/api-stability.md"),
    Path("docs/architecture.md"),
    Path("docs/environment-and-optional-dependencies.md"),
    Path("docs/troubleshooting.md"),
    Path("docs/governance-and-support.md"),
    Path("docs/trust-profiles-and-migration.md"),
    Path("docs/trust-governance-slos.md"),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_tracked_docs_exist() -> None:
    for path in TRACKED_DOCS:
        assert path.exists(), f"Tracked doc missing: {path}"


def test_markdown_no_trailing_whitespace() -> None:
    offenders: list[str] = []
    for path in TRACKED_DOCS:
        for idx, line in enumerate(_read(path).splitlines(), start=1):
            if line.rstrip(" ") != line:
                offenders.append(f"{path}:{idx}")
    assert not offenders, "Trailing whitespace found:\n" + "\n".join(offenders)


def test_markdown_no_tabs() -> None:
    offenders: list[str] = []
    for path in TRACKED_DOCS:
        for idx, line in enumerate(_read(path).splitlines(), start=1):
            if "\t" in line:
                offenders.append(f"{path}:{idx}")
    assert not offenders, "Tab characters found:\n" + "\n".join(offenders)


def test_docs_have_primary_title() -> None:
    for path in TRACKED_DOCS:
        text = _read(path)
        assert "# " in text, f"Missing level-1 heading in {path}"
