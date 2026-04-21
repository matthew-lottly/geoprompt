"""Code-quality audit helpers for GeoPrompt.

These utilities provide lightweight AST-based checks used to validate the
B1 code-quality cleanup work across selected files.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class QualityAuditResult:
    """Structured result for a single audit pass."""

    path: str
    check: str
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a dictionary."""
        return {"path": self.path, "check": self.check, "issues": list(self.issues), "count": len(self.issues)}


def _to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _module_tree(path: str | Path) -> tuple[Path, ast.AST]:
    resolved = _to_path(path)
    return resolved, ast.parse(resolved.read_text(encoding="utf-8"))


def _public_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append(node)
    return functions


def audit_public_docstrings(path: str | Path) -> dict[str, Any]:
    """Report public functions missing docstrings."""
    resolved, tree = _module_tree(path)
    missing = [node.name for node in _public_functions(tree) if ast.get_docstring(node) is None]
    return QualityAuditResult(str(resolved), "docstrings", tuple(missing)).as_dict() | {"missing": missing}


def audit_type_annotations(path: str | Path) -> dict[str, Any]:
    """Report public functions missing argument or return annotations."""
    resolved, tree = _module_tree(path)
    missing: list[str] = []
    for node in _public_functions(tree):
        args = [arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
        has_all_annotations = all(arg.annotation is not None for arg in args) and node.returns is not None
        if not has_all_annotations:
            missing.append(node.name)
    return QualityAuditResult(str(resolved), "annotations", tuple(missing)).as_dict() | {"missing": missing}


def audit_mutable_defaults(path: str | Path) -> dict[str, Any]:
    """Report public functions with mutable default argument values."""
    resolved, tree = _module_tree(path)
    issues: list[str] = []
    for node in _public_functions(tree):
        defaults = list(node.args.defaults) + list(node.args.kw_defaults)
        if any(isinstance(default, (ast.List, ast.Dict, ast.Set)) for default in defaults if default is not None):
            issues.append(node.name)
    return QualityAuditResult(str(resolved), "mutable-defaults", tuple(issues)).as_dict()


def audit_pathlib_usage(path: str | Path) -> dict[str, Any]:
    """Report actual remaining use of os.path in executable code."""
    resolved, tree = _module_tree(path)
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path":
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                issues.append("os.path")
                break
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "path":
                    issues.append("from os import path")
                    break
    return QualityAuditResult(str(resolved), "pathlib", tuple(issues)).as_dict()


def audit_print_debugging(path: str | Path) -> dict[str, Any]:
    """Report public functions containing direct print calls."""
    resolved, tree = _module_tree(path)
    issues: list[str] = []
    for node in _public_functions(tree):
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "print":
                issues.append(node.name)
                break
    return QualityAuditResult(str(resolved), "print-debugging", tuple(issues)).as_dict()


def quality_scorecard(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Aggregate core code-quality checks across multiple files."""
    file_summaries: list[dict[str, Any]] = []
    total_issues = 0
    for path in paths:
        docstrings = audit_public_docstrings(path)
        annotations = audit_type_annotations(path)
        mutable = audit_mutable_defaults(path)
        pathlib_result = audit_pathlib_usage(path)
        prints = audit_print_debugging(path)
        issue_count = (
            len(docstrings.get("missing", []))
            + len(annotations.get("missing", []))
            + len(mutable.get("issues", []))
            + len(pathlib_result.get("issues", []))
            + len(prints.get("issues", []))
        )
        file_summaries.append(
            {
                "path": str(_to_path(path)),
                "docstrings": docstrings,
                "annotations": annotations,
                "mutable_defaults": mutable,
                "pathlib": pathlib_result,
                "prints": prints,
                "issue_count": issue_count,
            }
        )
        total_issues += issue_count
    return {"files": file_summaries, "total_issues": total_issues, "passed": total_issues == 0}


def packaging_smoke_matrix(pyproject_path: str | Path = "pyproject.toml") -> list[dict[str, Any]]:
    """Build a simple packaging smoke-test matrix from the project extras."""
    path = _to_path(pyproject_path)
    extras: dict[str, Any] = {}
    if tomllib is not None and path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        extras = data.get("project", {}).get("optional-dependencies", {})

    ordered_profiles = ["core", *sorted(extras.keys())]
    matrix: list[dict[str, Any]] = []
    for profile in ordered_profiles:
        install_cmd = "pip install -e ." if profile == "core" else f"pip install -e .[{profile}]"
        matrix.append({
            "profile": profile,
            "install": install_cmd,
            "purpose": "base package" if profile == "core" else "optional extra validation",
        })
    return matrix


def placeholder_inventory(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Inventory public functions that are explicitly marked as stubs or placeholders."""
    markers = ("stub", "placeholder", "simulation-only", "demo-only")
    found: list[dict[str, Any]] = []
    for path in paths:
        resolved, tree = _module_tree(path)
        for node in _public_functions(tree):
            doc = (ast.get_docstring(node) or "").lower()
            reasons = [marker for marker in markers if marker in node.name.lower() or marker in doc]
            if reasons:
                found.append({"path": str(resolved), "name": node.name, "markers": sorted(set(reasons))})
    return {"functions": found, "count": len(found), "passed": True}


def subsystem_maturity_matrix() -> dict[str, Any]:
    """Return a concise maturity map for the major GeoPrompt subsystems."""
    return {
        "stable_core": ["frame", "geometry", "network", "equations", "io"],
        "supported_optional": ["service", "viz", "db", "raster", "compare"],
        "experimental_surface": ["domain", "ml", "imagery", "geoprocessing extras"],
        "simulation_only": [
            "mypy_plugin_stub",
            "notify_email_stub",
            "notify_slack_stub",
            "sso_saml_metadata_stub",
            "serverless_endpoint_stub",
        ],
    }


def documentation_asset_manifest(manifest_path: str | Path = "docs/figures-manifest.json") -> dict[str, Any]:
    """Validate the committed figure manifest and its provenance fields."""
    path = _to_path(manifest_path)
    if not path.exists():
        return {"passed": False, "count": 0, "issues": [f"missing manifest: {path}"]}

    data = json.loads(path.read_text(encoding="utf-8"))
    figures = list(data.get("figures", []))
    required = {"path", "caption", "data_source", "generation_script", "last_verified"}
    issues: list[str] = []
    for item in figures:
        missing = sorted(required - set(item.keys()))
        if missing:
            issues.append(f"{item.get('path', 'unknown')}: missing {', '.join(missing)}")
    return {"passed": not issues, "count": len(figures), "issues": issues, "figures": figures}


def release_readiness_report(
    paths: Iterable[str | Path],
    *,
    pyproject_path: str | Path = "pyproject.toml",
    api_stability_path: str | Path = "docs/api-stability.md",
) -> dict[str, Any]:
    """Summarize whether the package has enough proof to be treated as beta-ready."""
    quality = quality_scorecard(paths)
    packaging = packaging_smoke_matrix(pyproject_path)
    api_doc = _to_path(api_stability_path)
    api_stability = {
        "documented": api_doc.exists(),
        "path": str(api_doc),
        "status": "tracked" if api_doc.exists() else "missing",
    }
    benchmark_evidence = {
        "comparison_bundle_support": Path("src/geoprompt/compare.py").exists(),
        "benchmarks_doc": Path("docs/benchmarks.md").exists(),
    }
    placeholder_audit = placeholder_inventory([
        "src/geoprompt/geoprocessing.py",
        "src/geoprompt/domain.py",
        "src/geoprompt/ml.py",
        "src/geoprompt/imagery.py",
    ])
    docs_assets = documentation_asset_manifest("docs/figures-manifest.json")
    evidence_realism = {
        "manifest_tracked": docs_assets.get("passed", False),
        "figures_count": docs_assets.get("count", 0),
        "release_evidence_doc": Path("docs/release-evidence.md").exists(),
    }

    if quality["passed"] and api_stability["documented"] and len(packaging) >= 6 and docs_assets.get("passed"):
        release_stage = "release-candidate"
    elif api_stability["documented"] and len(packaging) >= 3:
        release_stage = "beta"
    else:
        release_stage = "alpha"

    return {
        "release_stage": release_stage,
        "quality": quality,
        "packaging_smoke_matrix": packaging,
        "api_stability": api_stability,
        "benchmark_evidence": benchmark_evidence,
        "placeholder_audit": placeholder_audit,
        "docs_assets": docs_assets,
        "evidence_realism": evidence_realism,
        "trust_milestone": release_stage in {"beta", "release-candidate"},
    }


__all__ = [
    "QualityAuditResult",
    "audit_public_docstrings",
    "audit_type_annotations",
    "audit_mutable_defaults",
    "audit_pathlib_usage",
    "audit_print_debugging",
    "documentation_asset_manifest",
    "packaging_smoke_matrix",
    "placeholder_inventory",
    "quality_scorecard",
    "release_readiness_report",
    "subsystem_maturity_matrix",
]
