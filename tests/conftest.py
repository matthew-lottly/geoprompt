"""Shared test fixtures for the geoprompt test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Common geometries
# ---------------------------------------------------------------------------


@pytest.fixture()
def point_geometry() -> dict:
    """A simple Point geometry."""
    return {"type": "Point", "coordinates": (-111.93, 40.77)}


@pytest.fixture()
def line_geometry() -> dict:
    """A simple LineString geometry."""
    return {
        "type": "LineString",
        "coordinates": ((-111.95, 40.75), (-111.93, 40.77), (-111.91, 40.79)),
    }


@pytest.fixture()
def polygon_geometry() -> dict:
    """A simple Polygon geometry (triangle)."""
    return {
        "type": "Polygon",
        "coordinates": (
            ((-111.95, 40.75), (-111.91, 40.75), (-111.93, 40.79), (-111.95, 40.75)),
        ),
    }


@pytest.fixture()
def square_polygon() -> dict:
    """A unit-square Polygon at the origin."""
    return {
        "type": "Polygon",
        "coordinates": (((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)),),
    }


@pytest.fixture()
def multipoint_geometry() -> dict:
    """A MultiPoint with three points."""
    return {
        "type": "MultiPoint",
        "coordinates": ((-111.95, 40.75), (-111.93, 40.77), (-111.91, 40.79)),
    }


# ---------------------------------------------------------------------------
# Feature records
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_features() -> list[dict]:
    """A small set of point-feature records with numeric attributes."""
    return [
        {"id": 1, "name": "A", "value": 10, "geometry": {"type": "Point", "coordinates": (0, 0)}},
        {"id": 2, "name": "B", "value": 20, "geometry": {"type": "Point", "coordinates": (1, 0)}},
        {"id": 3, "name": "C", "value": 30, "geometry": {"type": "Point", "coordinates": (0.5, 1)}},
    ]


@pytest.fixture()
def sample_polygon_features() -> list[dict]:
    """Three adjacent square polygons sharing edges."""
    return [
        {
            "id": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": (((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)),),
            },
        },
        {
            "id": 2,
            "geometry": {
                "type": "Polygon",
                "coordinates": (((1, 0), (2, 0), (2, 1), (1, 1), (1, 0)),),
            },
        },
        {
            "id": 3,
            "geometry": {
                "type": "Polygon",
                "coordinates": (((2, 0), (3, 0), (3, 1), (2, 1), (2, 0)),),
            },
        },
    ]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir() -> Path:
    """Path to the test-data directory."""
    return DATA_DIR


@pytest.fixture()
def sample_points_path() -> Path:
    """Path to sample_points.json."""
    return DATA_DIR / "sample_points.json"


@pytest.fixture()
def sample_features_path() -> Path:
    """Path to sample_features.json."""
    return DATA_DIR / "sample_features.json"
