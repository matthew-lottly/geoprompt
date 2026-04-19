from __future__ import annotations

import os

import pytest

from geoprompt.enterprise import paginated_request


@pytest.mark.skipif(not os.getenv("GEOPROMPT_RUN_REMOTE_INTEGRATION"), reason="Remote integration environment not configured")
def test_remote_service_healthcheck() -> None:
    import requests

    url = os.environ["GEOPROMPT_REMOTE_HEALTH_URL"]
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    assert response.status_code == 200


@pytest.mark.skipif(not os.getenv("GEOPROMPT_RUN_DB_INTEGRATION"), reason="Database integration environment not configured")
def test_database_connector_roundtrip() -> None:
    from geoprompt.db import read_duckdb, write_duckdb

    rows = [
        {"id": "a", "value": "1", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"id": "b", "value": "2", "geometry": {"type": "Point", "coordinates": [1, 1]}},
    ]
    written = write_duckdb(rows, "sample_points")
    assert written == 2
    read_back = read_duckdb("SELECT * FROM sample_points")
    assert len(read_back) == 2


def test_pagination_helper_is_importable() -> None:
    assert callable(paginated_request)
