"""CI lint gate tests (J8.94-J8.96).

These tests statically scan source files to enforce code-level policies:
  J8.94 — no new bare ``eval(`` calls in production code.
  J8.95 — no new bare ``except Exception:`` or ``except Exception as`` patterns.
  J8.96 — no new public ``@simulation_only`` symbols added without a tier entry.

The checks operate on the source tree under ``src/geoprompt/`` and intentionally
avoid running the code.  They are fast, safe for CI, and fail hard on violations.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
import pytest

_SRC = Path(__file__).parent.parent / "src" / "geoprompt"

# Source files excluded from the eval-ban (e.g. the safe expression engine itself
# which must reference the ``ast`` module — but it still must not call eval()).
_EVAL_EXEMPT_FILES: frozenset[str] = frozenset({
    "safe_expression.py",   # the safe evaluator; references to 'eval' only in strings
})

# Files excluded from the broad-except check (legacy, known, and tracked entries).
_BROAD_EXCEPT_EXEMPT_FILES: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# J8.94 — No bare eval() calls in production source
# ---------------------------------------------------------------------------

class TestNoNewEvalCalls:
    """Ensure no production module calls the builtin eval()."""

    def _source_files(self) -> list[Path]:
        return [p for p in _SRC.rglob("*.py") if p.name not in _EVAL_EXEMPT_FILES]

    def test_no_eval_in_source(self):
        violations: list[str] = []
        # Match `eval(` as a call but NOT `def eval(` (method definition) or `.eval(`
        # (attribute method call on another object) or string literals.
        eval_call_pattern = re.compile(r'(?<!def )(?<![.\w])eval\s*\(')
        for path in self._source_files():
            src = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if eval_call_pattern.search(line):
                    violations.append(f"{path.relative_to(_SRC)}:{lineno}: {line.strip()!r}")

        if violations:
            listing = "\n".join(f"  {v}" for v in violations)
            pytest.fail(
                f"Found {len(violations)} bare eval() call(s) in production source.\n"
                f"Use evaluate_safe_expression() from safe_expression.py instead:\n{listing}"
            )


# ---------------------------------------------------------------------------
# J8.95 — No new bare except Exception patterns
# ---------------------------------------------------------------------------

# Ratchet baseline: current known broad-except count in the codebase.
# This test fails only when the count EXCEEDS this baseline, acting as a
# forward-looking gate that prevents regression while accepting existing debt.
# Decrement this number as existing broad-excepts are replaced.
_BROAD_EXCEPT_BASELINE: int = 57


class TestNoBroadExceptException:
    """Catch patterns like ``except Exception:`` and ``except Exception as e:``.

    Uses a ratchet baseline — fails only when violations exceed the baseline.
    The baseline must be reduced (never increased) as debt is paid down.
    """

    _BROAD_EXCEPT_RE = re.compile(r'except\s+Exception\s*(?:as\s+\w+)?\s*:')

    def _source_files(self) -> list[Path]:
        return [
            p for p in _SRC.rglob("*.py")
            if p.name not in _BROAD_EXCEPT_EXEMPT_FILES
        ]

    def test_broad_except_count_does_not_increase(self):
        """Fail if new broad 'except Exception' patterns are added above the baseline."""
        violations: list[str] = []
        for path in self._source_files():
            src = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if self._BROAD_EXCEPT_RE.search(line):
                    violations.append(
                        f"{path.relative_to(_SRC)}:{lineno}: {line.strip()!r}"
                    )

        count = len(violations)
        if count > _BROAD_EXCEPT_BASELINE:
            new_violations = violations[_BROAD_EXCEPT_BASELINE:]
            listing = "\n".join(f"  {v}" for v in new_violations)
            pytest.fail(
                textwrap.dedent(f"""\
                    Broad 'except Exception' count increased from {_BROAD_EXCEPT_BASELINE} to {count}.
                    Do NOT add new broad-except patterns. Replace with specific exception types.
                    New violations above baseline:
                    {listing}
                """)
            )
        if count < _BROAD_EXCEPT_BASELINE:
            # Violations decreased — update the baseline in the test file to lock in progress.
            import warnings
            warnings.warn(
                f"Broad-except count dropped to {count} (baseline={_BROAD_EXCEPT_BASELINE}). "
                "Update _BROAD_EXCEPT_BASELINE in test_ci_lint_gates.py to lock in progress.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# J8.96 — simulation_only symbols must have TIER_METADATA entries
# ---------------------------------------------------------------------------

class TestSimulationOnlySymbolsHaveTierEntries:
    """Every public function decorated with @simulation_only must have a TIER_METADATA entry."""

    def _collect_simulation_only_symbols(self) -> list[tuple[str, str]]:
        """Return list of (module_name, symbol_name) for all @simulation_only public fns."""
        results: list[tuple[str, str]] = []
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
                    results.append((module_name, node.name))
        return results

    def test_simulation_only_symbols_have_tier_metadata(self):
        from geoprompt._tier_metadata import TIER_METADATA
        simulation_symbols = self._collect_simulation_only_symbols()
        missing: list[str] = []
        for module_name, symbol_name in simulation_symbols:
            key = f"{module_name}.{symbol_name}"
            if key not in TIER_METADATA:
                missing.append(key)

        if missing:
            listing = "\n".join(f"  {k}" for k in sorted(missing))
            pytest.fail(
                f"{len(missing)} @simulation_only symbol(s) lack TIER_METADATA entries.\n"
                f"Add each to TIER_METADATA with TIER_SIMULATION:\n{listing}"
            )
