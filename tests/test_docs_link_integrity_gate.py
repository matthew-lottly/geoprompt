from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_ROOT = ROOT / "docs"

_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted(DOC_ROOT.rglob("*.md")))
    return files


def _iter_local_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    links = [match.group(1).strip() for match in _LINK_RE.finditer(text)]
    local: list[str] = []
    for link in links:
        if not link:
            continue
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        if link.startswith("#"):
            continue
        target = link.split("#", 1)[0].strip()
        if not target:
            continue
        if target.startswith("<") and target.endswith(">"):
            continue
        local.append(target)
    return local


def test_readme_and_docs_links_resolve_locally() -> None:
    missing: list[str] = []
    for doc in _markdown_files():
        for link in _iter_local_links(doc):
            resolved = (doc.parent / link).resolve()
            if not resolved.exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {link}")

    assert not missing, "Broken local markdown links found:\n" + "\n".join(sorted(missing))
