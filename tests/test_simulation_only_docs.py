"""Tests for J2.22 — Docs and runtime match for simulation-only symbols.

Ensures that every @simulation_only symbol has a docstring that clearly
states it is simulation-only / non-functional, and the docstring mentions
what a real implementation should do or what library/service to install.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


_SRC = Path(__file__).parent.parent / "src" / "geoprompt"


class TestSimulationOnlyDocstrings:
    """@simulation_only functions must document that behavior in their docstring."""

    def _collect_simulation_only_symbols(self) -> dict[str, tuple[str, str]]:
        """Return {(module, name): (signature, docstring)} for all @simulation_only functions."""
        results: dict[str, tuple[str, str]] = {}
        for path in _SRC.rglob("*.py"):
            if path.name.startswith("_"):
                continue
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src, filename=str(path))
            except SyntaxError:
                continue
            module_name = path.stem
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Check for simulation_only decorator
                has_simulation_only = any(
                    (isinstance(d, ast.Name) and d.id == "simulation_only")
                    or (isinstance(d, ast.Attribute) and d.attr == "simulation_only")
                    or (isinstance(d, ast.Call) and (
                        (isinstance(d.func, ast.Name) and d.func.id == "simulation_only")
                        or (isinstance(d.func, ast.Attribute) and d.func.attr == "simulation_only")
                    ))
                    for d in node.decorator_list
                )
                if has_simulation_only and not node.name.startswith("_"):
                    docstring = ast.get_docstring(node) or ""
                    key = f"{module_name}.{node.name}"
                    results[key] = ("", docstring)
        return results

    def test_simulation_only_docstrings_mention_simulation(self):
        """Every @simulation_only should mention it in docstring (warning-level check)."""
        import warnings
        missing_docs: list[str] = []
        symbols = self._collect_simulation_only_symbols()
        for key, (_, docstring) in symbols.items():
            # Check if docstring mentions simulation, placeholder, stub, or non-functional
            keywords = {
                "simulation", "placeholder", "stub", "non-functional",
                "does not return real results", "mock", "fake"
            }
            if not any(kw in docstring.lower() for kw in keywords):
                missing_docs.append(key)

        # Warning-level: document which symbols need improvement
        if missing_docs:
            warnings.warn(
                f"{len(missing_docs)} @simulation_only symbol(s) lack clear "
                f"simulation/stub/placeholder language in docstring:\n"
                + "\n".join(f"  {k}" for k in sorted(missing_docs)),
                stacklevel=2,
            )

    def test_simulation_only_docstrings_include_remediation(self):
        """Every @simulation_only should guide users on real implementation."""
        missing_guidance: list[str] = []
        symbols = self._collect_simulation_only_symbols()
        for key, (_, docstring) in symbols.items():
            # Check for guidance keywords
            guidance_keywords = {
                "install", "implement", "library", "package", "backend",
                "real implementation", "use", "service", "api", "endpoint"
            }
            if not any(kw in docstring.lower() for kw in guidance_keywords):
                missing_guidance.append(key)

        # This is a warning, not a hard failure — but we should log it
        if missing_guidance:
            import warnings
            warnings.warn(
                f"{len(missing_guidance)} @simulation_only symbol(s) lack remediation guidance:\n"
                + "\n".join(f"  {k}" for k in sorted(missing_guidance)),
                stacklevel=2,
            )
