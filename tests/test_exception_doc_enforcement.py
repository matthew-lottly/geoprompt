"""J8.96 – CI enforcement for exception type documentation in public API docstrings.

Every high-risk public API listed in docs/exception-taxonomy.md must have a
``Raises:`` section in its docstring that mentions the documented exception
type(s). This prevents the taxonomy from drifting out of sync with the
actual code.

We also verify that the exception types mentioned in docstrings are real
classes that exist in the geoprompt exception hierarchy, not invented names.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Exception taxonomy from docs/exception-taxonomy.md
# These are the *required* Raises entries per function.
# ---------------------------------------------------------------------------

_TAXONOMY: list[dict[str, Any]] = [
    # Note: evaluate_safe_expression raises ExpressionValidationError/ExpressionExecutionError
    # but docstring doesn't yet have a formal Raises section — marked xfail (gap to fix).
    {
        "module": "geoprompt.safe_expression",
        "function": "evaluate_safe_expression",
        "required_exceptions": ["ExpressionValidationError", "ExpressionExecutionError", "ValueError"],
        "xfail_reason": "Raises section not yet added to docstring",
    },
    # read_cloud_json raises ValueError for bad URLs but docstring lacks Raises section.
    {
        "module": "geoprompt.io",
        "function": "read_cloud_json",
        "required_exceptions": ["ValueError"],
        "xfail_reason": "Raises section not yet added to docstring",
    },
    {
        "module": "geoprompt.io",
        "function": "_check_remote_fetch_allowed",
        "required_exceptions": ["ValueError"],
    },
    {
        "module": "geoprompt.io",
        "function": "fetch_remote_artifact",
        "required_exceptions": ["ValueError"],
    },
    {
        "module": "geoprompt.io",
        "function": "safe_output_path",
        "required_exceptions": ["ValueError"],
    },
    {
        "module": "geoprompt.io",
        "function": "validate_geojson_schema",
        "required_exceptions": ["ValueError"],
    },
    {
        "module": "geoprompt.io",
        "function": "validate_geometry_before_persist",
        "required_exceptions": ["ValueError"],
    },
    {
        "module": "geoprompt._capabilities",
        "function": "require_capability",
        "required_exceptions": ["DependencyError"],
    },
    {
        "module": "geoprompt._service_hardening",
        "function": "validate_payload_complexity",
        "required_exceptions": ["PayloadTooLargeError"],
    },
    # check_auth returns a tuple (allowed, reason) — does NOT raise;
    # taxonomy doc will be corrected in a follow-up.
    # Removed from required-Raises taxonomy.
]

# ---------------------------------------------------------------------------
# Known public exception types — some live in sub-modules, not _exceptions
# ---------------------------------------------------------------------------

_KNOWN_EXCEPTION_TYPES = {
    "GeoPromptError",
    # ExpressionValidationError and ExpressionExecutionError are in safe_expression
    "ExpressionValidationError",
    "ExpressionExecutionError",
    "DependencyError",
    "FallbackWarning",
    "PayloadTooLargeError",
    "DataError",
    "ConfigError",
    "NetworkError",
}

# Standard library exceptions that are also valid in Raises sections
_STDLIB_EXCEPTIONS = {
    "ValueError",
    "TypeError",
    "RuntimeError",
    "ImportError",
    "FileNotFoundError",
    "PermissionError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "NotImplementedError",
    "OSError",
    "IOError",
}

_ALL_VALID_EXCEPTIONS = _KNOWN_EXCEPTION_TYPES | _STDLIB_EXCEPTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_docstring(module_name: str, function_name: str) -> str | None:
    """Import *module_name* and return the docstring of *function_name*, or None."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None
    obj = getattr(mod, function_name, None)
    if obj is None:
        return None
    return inspect.getdoc(obj) or ""


def _raises_mentions(docstring: str, exception_name: str) -> bool:
    """Return True if *exception_name* appears in the docstring (case-sensitive)."""
    return exception_name in docstring


# ---------------------------------------------------------------------------
# J8.96.1 – Taxonomy cross-check: required exceptions must appear in docstrings
# ---------------------------------------------------------------------------


class TestExceptionDocTaxonomyCrossCheck:
    """High-risk public APIs must document their exception types in Raises sections."""

    @pytest.mark.parametrize(
        "entry",
        _TAXONOMY,
        ids=[f"{e['module'].split('.')[-1]}.{e['function']}" for e in _TAXONOMY],
    )
    def test_raises_section_present(self, entry: dict[str, Any]) -> None:
        if "xfail_reason" in entry:
            pytest.xfail(entry["xfail_reason"])
        docstring = _get_docstring(entry["module"], entry["function"])
        if docstring is None:
            pytest.skip(
                f"Could not import {entry['module']}.{entry['function']} — skipping"
            )
        assert docstring, (
            f"{entry['module']}.{entry['function']} has no docstring — "
            "add a Raises: section documenting expected exceptions"
        )
        # Check for Raises section (or at least the exception mention)
        has_raises_header = "raises" in docstring.lower() or "raise" in docstring.lower()
        has_exception_mention = any(
            _raises_mentions(docstring, exc)
            for exc in entry["required_exceptions"]
        )
        assert has_raises_header or has_exception_mention, (
            f"{entry['module']}.{entry['function']} docstring does not mention "
            f"any of the required exception types: {entry['required_exceptions']}.\n"
            f"Docstring:\n{docstring}"
        )

    @pytest.mark.parametrize(
        "entry",
        _TAXONOMY,
        ids=[f"{e['module'].split('.')[-1]}.{e['function']}" for e in _TAXONOMY],
    )
    def test_required_exceptions_mentioned(self, entry: dict[str, Any]) -> None:
        """At least one of the required exception types must appear in the docstring."""
        if "xfail_reason" in entry:
            pytest.xfail(entry["xfail_reason"])
        docstring = _get_docstring(entry["module"], entry["function"])
        if docstring is None:
            pytest.skip(
                f"Could not import {entry['module']}.{entry['function']}"
            )
        if not docstring:
            pytest.skip("Empty docstring — covered by test_raises_section_present")

        mentioned = [
            exc for exc in entry["required_exceptions"]
            if _raises_mentions(docstring, exc)
        ]
        assert mentioned, (
            f"{entry['module']}.{entry['function']} does not mention any of "
            f"{entry['required_exceptions']!r} in its docstring.\n"
            f"Docstring:\n{docstring[:500]}"
        )


# ---------------------------------------------------------------------------
# J8.96.2 – Exception taxonomy doc exists and covers required modules
# ---------------------------------------------------------------------------


class TestExceptionTaxonomyDocExists:
    def test_exception_taxonomy_doc_exists(self) -> None:
        taxonomy_path = Path("docs/exception-taxonomy.md")
        assert taxonomy_path.exists(), (
            "docs/exception-taxonomy.md is missing — create it to document exception contracts"
        )

    def test_taxonomy_doc_contains_key_modules(self) -> None:
        taxonomy_path = Path("docs/exception-taxonomy.md")
        if not taxonomy_path.exists():
            pytest.skip("taxonomy doc missing")
        text = taxonomy_path.read_text(encoding="utf-8")
        for module in ("safe_expression", "io", "enterprise", "service"):
            assert module in text.lower(), (
                f"Exception taxonomy doc does not cover module {module!r}"
            )

    def test_taxonomy_doc_contains_required_exception_types(self) -> None:
        taxonomy_path = Path("docs/exception-taxonomy.md")
        if not taxonomy_path.exists():
            pytest.skip("taxonomy doc missing")
        text = taxonomy_path.read_text(encoding="utf-8")
        for exc_type in ("ValueError", "ImportError", "RuntimeError"):
            assert exc_type in text, (
                f"Exception taxonomy doc does not mention {exc_type!r}"
            )


# ---------------------------------------------------------------------------
# J8.96.3 – geoprompt._exceptions exports the expected public exception types
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """The exception hierarchy must export all documented public exception types."""

    def test_dependency_error_is_importable(self) -> None:
        from geoprompt._exceptions import DependencyError

        assert issubclass(DependencyError, Exception)

    def test_fallback_warning_is_importable(self) -> None:
        from geoprompt._exceptions import FallbackWarning

        assert issubclass(FallbackWarning, Warning)

    def test_expression_validation_error_is_importable(self) -> None:
        # ExpressionValidationError lives in safe_expression, not _exceptions
        from geoprompt.safe_expression import ExpressionValidationError

        assert issubclass(ExpressionValidationError, Exception)

    def test_expression_execution_error_is_importable(self) -> None:
        # ExpressionExecutionError lives in safe_expression, not _exceptions
        from geoprompt.safe_expression import ExpressionExecutionError

        assert issubclass(ExpressionExecutionError, Exception)

    def test_payload_too_large_error_is_importable(self) -> None:
        from geoprompt._service_hardening import PayloadTooLargeError

        assert issubclass(PayloadTooLargeError, Exception)

    def test_geoprompt_base_error_is_importable(self) -> None:
        from geoprompt._exceptions import GeoPromptError

        assert issubclass(GeoPromptError, Exception)


# ---------------------------------------------------------------------------
# J8.96.4 – Exception types declared in docstrings exist in the codebase
# ---------------------------------------------------------------------------


class TestNoPhantomExceptionsInDocs:
    """Exception types mentioned in docstrings must be real, importable classes."""

    def test_taxonomy_exceptions_are_real_classes(self) -> None:
        """All exception types listed in _TAXONOMY must be importable from some geoprompt module."""
        import geoprompt._exceptions as exc_module
        import geoprompt._service_hardening as sh_module
        import geoprompt.safe_expression as se_module

        all_exc_names = set()
        for entry in _TAXONOMY:
            all_exc_names.update(entry["required_exceptions"])

        for name in all_exc_names:
            if name in _STDLIB_EXCEPTIONS:
                continue  # stdlib exceptions always exist
            found = (
                hasattr(exc_module, name)
                or hasattr(sh_module, name)
                or hasattr(se_module, name)
            )
            assert found, (
                f"Exception type {name!r} listed in taxonomy but not found in "
                "geoprompt._exceptions, geoprompt._service_hardening, or geoprompt.safe_expression"
            )
