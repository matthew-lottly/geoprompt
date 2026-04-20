"""Code-quality audit helpers for GeoPrompt.

These utilities provide lightweight AST-based checks used to validate the
B1 code-quality cleanup work across selected files.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
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


__all__ = [
    "QualityAuditResult",
    "audit_public_docstrings",
    "audit_type_annotations",
    "audit_mutable_defaults",
    "audit_pathlib_usage",
    "audit_print_debugging",
    "quality_scorecard",
]
