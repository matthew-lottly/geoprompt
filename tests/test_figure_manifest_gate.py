from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST_PATH = Path("docs/figures-manifest.json")
IMAGE_REGEX = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _iter_doc_images() -> list[tuple[Path, str, str]]:
    matches: list[tuple[Path, str, str]] = []
    for doc_path in Path("docs").rglob("*.md"):
        text = doc_path.read_text(encoding="utf-8")
        for alt_text, raw_target in IMAGE_REGEX.findall(text):
            if raw_target.startswith("http://") or raw_target.startswith("https://"):
                continue
            target = raw_target.split("#", 1)[0].strip()
            if not target:
                continue
            matches.append((doc_path, alt_text.strip(), target))
    return matches


def test_figure_manifest_paths_exist_and_have_alt_text() -> None:
    manifest = _load_manifest()
    figures = manifest.get("figures", [])
    assert isinstance(figures, list)
    assert figures

    for idx, row in enumerate(figures, start=1):
        assert isinstance(row, dict), f"Manifest entry {idx} is not an object"
        path = str(row.get("path", "")).strip()
        assert path, f"Manifest entry {idx} missing path"
        assert Path(path).exists(), f"Manifest path does not exist: {path}"
        alt_text = str(row.get("alt_text", "")).strip()
        assert alt_text, f"Manifest entry {idx} missing alt_text"


def test_docs_images_resolve_and_have_nonempty_alt_text() -> None:
    images = _iter_doc_images()
    assert images, "No markdown images found under docs/"

    for doc_path, alt_text, target in images:
        assert alt_text, f"Image alt text missing in {doc_path}: {target}"
        resolved = (doc_path.parent / target).resolve()
        assert resolved.exists(), f"Missing image referenced by {doc_path}: {target}"


def test_doc_images_are_cataloged_in_figure_manifest() -> None:
    manifest = _load_manifest()
    manifest_paths = {
        str(Path(str(row.get("path", ""))).as_posix())
        for row in manifest.get("figures", [])
        if isinstance(row, dict)
    }

    uncataloged: list[str] = []
    repo_root = Path.cwd().resolve()
    for doc_path, _alt, target in _iter_doc_images():
        normalized = str((doc_path.parent / target).resolve().relative_to(repo_root).as_posix())
        if normalized.startswith("docs/"):
            normalized = normalized[len("docs/") :]
        if normalized not in manifest_paths:
            uncataloged.append(f"{doc_path}:{target}")

    assert not uncataloged, "Doc images missing from manifest:\n" + "\n".join(sorted(uncataloged))
