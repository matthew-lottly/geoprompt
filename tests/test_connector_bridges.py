from __future__ import annotations

from pathlib import Path
import sqlite3

from geoprompt.io import read_cloud_json, write_cloud_json
from geoprompt.db import read_spatialite, write_spatialite


def test_cloud_json_roundtrip_local_path(tmp_path: Path) -> None:
    payload = {"status": "ok", "items": [1, 2, 3]}
    out = tmp_path / "payload.json"
    write_cloud_json(str(out), payload)
    loaded = read_cloud_json(str(out))
    assert loaded == payload


def test_cloud_json_accepts_path_object(tmp_path: Path) -> None:
    payload = [{"site_id": "a", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
    out = tmp_path / "payload-path.json"
    write_cloud_json(out, payload)
    loaded = read_cloud_json(out)
    assert loaded == payload


def test_spatialite_roundtrip(tmp_path: Path) -> None:
    database = tmp_path / "assets.sqlite"
    rows = [
        {"id": "a", "status": "active", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
        {"id": "b", "status": "planned", "geometry": {"type": "Point", "coordinates": [1.0, 1.0]}},
    ]
    count = write_spatialite(rows, "assets", str(database))
    assert count == 2

    read_back = read_spatialite("SELECT * FROM assets", str(database))
    assert len(read_back) == 2
    assert read_back[0]["geometry"]["type"] == "Point"


def test_spatialite_writes_metadata_and_index(tmp_path: Path) -> None:
    database = tmp_path / "assets_indexed.sqlite"
    rows = [
        {"id": "a", "status": "active", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
    ]

    count = write_spatialite(
        rows,
        "assets",
        str(database),
        create_spatial_index=True,
        create_metadata_table=True,
        srid=3857,
    )
    assert count == 1

    with sqlite3.connect(database) as conn:
        metadata = conn.execute(
            "SELECT geometry_column, srid FROM geoprompt_spatial_metadata WHERE table_name = 'assets'"
        ).fetchone()
        assert metadata == ("geometry", 3857)

        index_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_assets_geometry'"
        ).fetchall()
        assert index_rows
