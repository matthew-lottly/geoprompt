"""Public API and tier metadata integrity checks.

These tests act as CI gates for:
- duplicate names in module ``__all__`` declarations
- unresolved names in module ``__all__`` declarations
- unresolved keys in ``TIER_METADATA``
"""

from __future__ import annotations

from collections import Counter
import importlib
import pkgutil

import geoprompt
from geoprompt._tier_metadata import TIER_METADATA


def _iter_geoprompt_modules() -> list[str]:
    names: list[str] = []
    for module_info in pkgutil.iter_modules(geoprompt.__path__):
        if module_info.ispkg:
            continue
        names.append(f"geoprompt.{module_info.name}")
    return sorted(names)


def test_module_all_has_no_duplicates() -> None:
    """Fail if any module declares duplicate names in __all__."""
    failures: list[str] = []

    for module_name in _iter_geoprompt_modules():
        module = importlib.import_module(module_name)
        all_names = getattr(module, "__all__", None)
        if all_names is None:
            continue

        counts = Counter(all_names)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            failures.append(f"{module_name}: duplicate __all__ names {duplicates}")

    assert not failures, "\n".join(failures)


def test_module_all_has_only_resolvable_names() -> None:
    """Fail if any __all__ entry does not resolve to a module attribute."""
    failures: list[str] = []

    for module_name in _iter_geoprompt_modules():
        module = importlib.import_module(module_name)
        all_names = getattr(module, "__all__", None)
        if all_names is None:
            continue

        unresolved = sorted(name for name in all_names if not hasattr(module, name))
        if unresolved:
            failures.append(f"{module_name}: unresolved __all__ names {unresolved}")

    assert not failures, "\n".join(failures)


def test_tier_metadata_keys_resolve_to_symbols() -> None:
    """Fail if any TIER_METADATA key does not map to a real symbol."""
    unresolved: list[str] = []

    for key in sorted(TIER_METADATA):
        module_name, symbol_name = key.split(".", 1)
        module = importlib.import_module(f"geoprompt.{module_name}")
        if not hasattr(module, symbol_name):
            unresolved.append(key)

    assert not unresolved, f"Unresolved tier metadata keys: {unresolved}"
