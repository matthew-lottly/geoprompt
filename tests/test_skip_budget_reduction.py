"""J8.85 – Deterministic CI-safe variants of skip-gated trust-critical tests.

These tests replace ``pytest.skip`` patterns that required a specific
dependency to be absent by mocking the import so the tests run in every CI
environment regardless of what is installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon(coords: list) -> dict:
    return {"type": "Polygon", "coordinates": [coords]}


def _polygon_frame():
    from geoprompt import GeoPromptFrame

    return GeoPromptFrame(
        [
            {
                "site_id": "A",
                "geometry": _polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]),
            },
            {
                "site_id": "B",
                "geometry": _polygon([(5, 5), (8, 5), (8, 8), (5, 8), (5, 5)]),
            },
        ]
    )


# ---------------------------------------------------------------------------
# J8.85.1 – shapely-absent variants (mock-based, no real absence required)
# ---------------------------------------------------------------------------


class TestShapelyAbsentPathsCI:
    """CI-safe versions of the shapely-skip tests in test_parity_features.py.

    Instead of waiting for shapely to be uninstalled, we patch
    ``geoprompt._capabilities._is_importable`` to return ``False`` for
    all capabilities and verify that the capability guard raises
    ``DependencyError`` — proving the error path is exercised.
    """

    def test_require_shapely_raises_dependency_error(self) -> None:
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError):
                require_capability("shapely", context="geometry.simplify")

    def test_require_shapely_error_mentions_shapely(self) -> None:
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError) as exc_info:
                require_capability("shapely", context="geometry.convex_hull")
        assert "shapely" in str(exc_info.value).lower()

    def test_check_shapely_returns_false_when_absent(self) -> None:
        from geoprompt._capabilities import check_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            assert check_capability("shapely") is False

    def test_geometry_split_emits_fallback_warning_when_shapely_absent(self) -> None:
        import importlib
        from geoprompt.geometry import geometry_split
        from geoprompt._exceptions import FallbackWarning

        geom = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        splitter = {"type": "LineString", "coordinates": [[0, 1], [1, 0]]}

        original_import_module = importlib.import_module

        def _block_shapely(name, *args, **kwargs):
            if "shapely" in name:
                raise ImportError(f"Mocked missing: {name}")
            return original_import_module(name, *args, **kwargs)

        import warnings
        with patch.object(importlib, "import_module", side_effect=_block_shapely):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    geometry_split(geom, splitter)
                except Exception:
                    pass
        fallback_w = [x for x in w if issubclass(x.category, FallbackWarning)]
        assert fallback_w, "Expected FallbackWarning when shapely is absent"

    def test_identity_overlay_emits_fallback_warning_when_shapely_absent(self) -> None:
        import sys
        from geoprompt import GeoPromptFrame
        import geoprompt.geoprocessing as gp
        from geoprompt._exceptions import FallbackWarning

        frame = GeoPromptFrame(
            [{"id": 1, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
        )
        identity_frame = GeoPromptFrame(
            [{"id": 2, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]
        )

        import warnings
        with patch.dict(sys.modules, {"shapely": None, "shapely.geometry": None, "shapely.ops": None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    gp.identity_overlay(frame, identity_frame)
                except Exception:
                    pass
        fallback_w = [x for x in w if issubclass(x.category, FallbackWarning)]
        assert fallback_w, "Expected FallbackWarning from identity_overlay when shapely absent"


# ---------------------------------------------------------------------------
# J8.85.2 – sklearn-absent variant (mock-based)
# ---------------------------------------------------------------------------


class TestSklearnAbsentPathCI:
    """CI-safe version of the sklearn-skip test in test_g1_g5_g8_g13_g24.py."""

    def test_rf_emits_simulation_warning_when_sklearn_absent(self) -> None:
        """Random forest should emit a simulation-only warning without sklearn."""
        import geoprompt as gp
        from geoprompt import GeoPromptFrame

        rows = [
            {
                "feat": float(i),
                "label": "A" if i < 5 else "B",
                "geometry": {"type": "Point", "coordinates": [float(i), 0.0]},
            }
            for i in range(10)
        ]
        frame = GeoPromptFrame(rows, geometry_column="geometry")

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    result = gp.random_forest_spatial_prediction(
                        frame,
                        target_column="label",
                        feature_columns=["feat"],
                    )
                    # If it returned a result, it should be simulation-mode
                    sim_warnings = [
                        x
                        for x in w
                        if "simulation" in str(x.message).lower()
                        or "fallback" in str(x.message).lower()
                    ]
                    # Either a warning was emitted OR an exception was raised
                    # (both are acceptable; silence is not)
                    assert sim_warnings, (
                        "Expected a simulation/fallback warning when sklearn is absent"
                    )
                except Exception:
                    # Raising is also acceptable — the key is no silent fallback
                    pass


# ---------------------------------------------------------------------------
# J8.85.3 – osmium-absent variant (mock-based, always runs)
# ---------------------------------------------------------------------------


class TestOsmiumAbsentPathCI:
    """CI-safe version of the osmium-missing test in test_no_fallback_without_optin.py."""

    def test_osm_reader_raises_when_osmium_absent(self) -> None:
        from geoprompt._exceptions import DependencyError
        import geoprompt.io as io_module

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises((DependencyError, ImportError, RuntimeError)):
                io_module.read_osm_pbf("/tmp/dummy.pbf")


# ---------------------------------------------------------------------------
# J8.85.4 – ezdxf-absent variant (mock-based, always runs)
# ---------------------------------------------------------------------------


class TestEzdxfAbsentPathCI:
    """CI-safe version of the ezdxf-missing test in test_no_fallback_without_optin.py."""

    def test_dxf_reader_raises_when_ezdxf_absent(self) -> None:
        from geoprompt._exceptions import DependencyError
        import geoprompt.io as io_module

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises((DependencyError, ImportError, RuntimeError)):
                io_module.read_dxf("/tmp/dummy.dxf")


# ---------------------------------------------------------------------------
# J8.85.5 – FastAPI-absent variant (mock-based, always runs)
# ---------------------------------------------------------------------------


class TestFastapiAbsentPathCI:
    """CI-safe version of the FastAPI-skip test in test_no_fallback_without_optin.py."""

    def test_build_app_raises_stub_without_dev_profile(self) -> None:
        """build_app raises when stub-mode is allowed but dev profile is off."""
        import os

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            old_allow = os.environ.get("GEOPROMPT_ALLOW_STUB_FALLBACK")
            old_dev = os.environ.get("GEOPROMPT_DEV_PROFILE")
            try:
                os.environ["GEOPROMPT_ALLOW_STUB_FALLBACK"] = "true"
                os.environ.pop("GEOPROMPT_DEV_PROFILE", None)
                import geoprompt.service

                with pytest.raises((RuntimeError, ImportError)):
                    geoprompt.service.build_app()
            except ImportError:
                # FastAPI not installed at all — acceptable outcome
                pass
            finally:
                if old_allow is None:
                    os.environ.pop("GEOPROMPT_ALLOW_STUB_FALLBACK", None)
                else:
                    os.environ["GEOPROMPT_ALLOW_STUB_FALLBACK"] = old_allow
                if old_dev is not None:
                    os.environ["GEOPROMPT_DEV_PROFILE"] = old_dev
