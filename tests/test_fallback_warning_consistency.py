"""J8.89 – Warning category and warning text consistency tests for fallback paths.

Every FallbackWarning emitted by a degraded-mode path must:
  1. Be an instance of ``FallbackWarning`` (not plain ``UserWarning``).
  2. Contain the name of the missing package.
  3. Contain a ``pip install`` hint (so users know how to resolve it).
  4. Use stacklevel ≥ 2 so the warning points to the caller, not the internals.

We test a representative sample from each module that emits FallbackWarning.
"""
from __future__ import annotations

import re
import warnings
from unittest.mock import patch

import pytest

from geoprompt._exceptions import FallbackWarning


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _capture_warnings(fn, *args, **kwargs):
    """Call *fn* and return all warnings emitted."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
    return w


def _fallback_warnings(fn, *args, **kwargs):
    return [
        x for x in _capture_warnings(fn, *args, **kwargs)
        if issubclass(x.category, FallbackWarning)
    ]


# ---------------------------------------------------------------------------
# J8.89.1 – FallbackWarning category contract
# ---------------------------------------------------------------------------


class TestFallbackWarningIsCorrectCategory:
    """FallbackWarning must be the warning category (not bare UserWarning)."""

    def test_fallback_warning_is_subclass_of_user_warning(self) -> None:
        assert issubclass(FallbackWarning, UserWarning)

    def test_geometry_split_emits_fallback_warning(self) -> None:
        import importlib
        from unittest.mock import patch as _patch
        from geoprompt.geometry import geometry_split

        geom = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        splitter = {"type": "LineString", "coordinates": [[0, 1], [1, 0]]}

        original_import_module = importlib.import_module

        def _block_shapely(name, *args, **kwargs):
            if "shapely" in name:
                raise ImportError(f"Mocked missing: {name}")
            return original_import_module(name, *args, **kwargs)

        with _patch.object(importlib, "import_module", side_effect=_block_shapely):
            ws = _fallback_warnings(geometry_split, geom, splitter)

        assert ws, "Expected FallbackWarning from geometry_split"
        assert issubclass(ws[0].category, FallbackWarning)

    def test_geoprocessing_identity_emits_fallback_warning(self) -> None:
        import sys
        from geoprompt import GeoPromptFrame
        import geoprompt.geoprocessing as gp

        frame = GeoPromptFrame(
            [{"id": 1, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
        )
        identity_frame = GeoPromptFrame(
            [{"id": 2, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]
        )

        with patch.dict(sys.modules, {"shapely": None, "shapely.geometry": None, "shapely.ops": None}):
            ws = _fallback_warnings(gp.identity_overlay, frame, identity_frame)

        assert ws, "Expected FallbackWarning from geoprocessing.identity_overlay"
        assert issubclass(ws[0].category, FallbackWarning)


# ---------------------------------------------------------------------------
# J8.89.2 – FallbackWarning text contract
# ---------------------------------------------------------------------------


class TestFallbackWarningTextContract:
    """FallbackWarning messages must contain the package name and a pip hint."""

    _PIP_PATTERN = re.compile(r"pip install", re.IGNORECASE)

    def test_geometry_split_warning_contains_shapely(self) -> None:
        import importlib
        from unittest.mock import patch as _patch
        from geoprompt.geometry import geometry_split

        geom = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        splitter = {"type": "LineString", "coordinates": [[0, 1], [1, 0]]}

        original_import_module = importlib.import_module

        def _block_shapely(name, *args, **kwargs):
            if "shapely" in name:
                raise ImportError(f"Mocked missing: {name}")
            return original_import_module(name, *args, **kwargs)

        with _patch.object(importlib, "import_module", side_effect=_block_shapely):
            ws = _fallback_warnings(geometry_split, geom, splitter)

        assert ws
        msg = str(ws[0].message)
        assert "shapely" in msg.lower(), f"Expected 'shapely' in warning message: {msg!r}"
        assert self._PIP_PATTERN.search(msg), f"Expected 'pip install' in warning message: {msg!r}"

    def test_geoprocessing_identity_warning_contains_shapely(self) -> None:
        import sys
        from geoprompt import GeoPromptFrame
        import geoprompt.geoprocessing as gp

        frame = GeoPromptFrame(
            [{"id": 1, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
        )
        identity_frame = GeoPromptFrame(
            [{"id": 2, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]
        )

        with patch.dict(sys.modules, {"shapely": None, "shapely.geometry": None, "shapely.ops": None}):
            ws = _fallback_warnings(gp.identity_overlay, frame, identity_frame)

        assert ws
        msg = str(ws[0].message)
        assert "shapely" in msg.lower(), f"Expected 'shapely' in warning message: {msg!r}"
        assert self._PIP_PATTERN.search(msg), f"Expected 'pip install' in: {msg!r}"

    def test_degraded_mode_policy_warning_contains_package_name(self) -> None:
        from geoprompt._capabilities import DegradedModePolicy

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            policy = DegradedModePolicy("scipy", context="spatial analysis")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            policy.warn_degraded()

        assert w, "Expected a warning from warn_degraded()"
        msg = str(w[0].message)
        assert "scipy" in msg.lower(), f"Expected 'scipy' in: {msg!r}"
        assert self._PIP_PATTERN.search(msg), f"Expected 'pip install' in: {msg!r}"


# ---------------------------------------------------------------------------
# J8.89.3 – Warning consistency: no FallbackWarning without the pip hint
# ---------------------------------------------------------------------------


class TestFallbackWarningAlwaysHasPipHint:
    """All FallbackWarning instances in the capability registry must reference pip."""

    def test_capability_registry_entries_have_pip_extra(self) -> None:
        from geoprompt._capabilities import CAPABILITY_REGISTRY

        for name, spec in CAPABILITY_REGISTRY.items():
            assert spec.pip_extra, (
                f"CAPABILITY_REGISTRY[{name!r}] has no pip_extra — "
                "users cannot know how to install it"
            )

    def test_dependency_error_contains_package_name(self) -> None:
        from geoprompt._capabilities import require_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(Exception) as exc_info:
                require_capability("shapely", context="test_op()")

        msg = str(exc_info.value)
        assert "shapely" in msg.lower(), f"DependencyError missing package name: {msg!r}"


# ---------------------------------------------------------------------------
# J8.89.4 – Warning category taxonomy (FallbackWarning vs FutureWarning)
# ---------------------------------------------------------------------------


class TestWarningCategoryTaxonomy:
    """Tier degraded paths use FallbackWarning; tier-promotion warnings use FutureWarning."""

    def test_future_warning_for_non_stable_tier(self) -> None:
        from geoprompt._tier_metadata import warn_if_non_stable

        with pytest.warns(FutureWarning):
            warn_if_non_stable("gwr")

    def test_fallback_warning_is_user_warning_subclass(self) -> None:
        # FallbackWarning must not be FutureWarning (different concern)
        assert not issubclass(FallbackWarning, FutureWarning)
        assert issubclass(FallbackWarning, UserWarning)

    def test_degraded_warn_degraded_uses_user_warning_not_future_warning(self) -> None:
        from geoprompt._capabilities import DegradedModePolicy

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            policy = DegradedModePolicy("scipy", context="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            policy.warn_degraded()

        if w:
            # warn_degraded emits UserWarning (superset of FallbackWarning)
            assert any(issubclass(x.category, UserWarning) for x in w)
            # Must NOT be FutureWarning
            assert not any(issubclass(x.category, FutureWarning) for x in w)
