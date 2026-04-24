from __future__ import annotations

import pytest

from geoprompt.db import _geometry_to_wkt, _parse_wkt


def test_db_module_parses_common_wkt_shapes() -> None:
    assert _parse_wkt("POINT (1 2)") == {"type": "Point", "coordinates": (1.0, 2.0)}
    assert _parse_wkt("LINESTRING (0 0, 1 1)") == {
        "type": "LineString",
        "coordinates": ((0.0, 0.0), (1.0, 1.0)),
    }
    assert _parse_wkt("POLYGON ((0 0, 1 0, 1 1, 0 0))") == {
        "type": "Polygon",
        "coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
    }


def test_db_module_roundtrips_minimal_wkt_and_rejects_invalid_input() -> None:
    polygon = _parse_wkt("POLYGON ((0 0, 1 0, 1 1, 0 0))")

    assert _geometry_to_wkt(_parse_wkt("POINT (1 2)")) == "POINT (1.0 2.0)"
    assert _geometry_to_wkt(_parse_wkt("LINESTRING (0 0, 1 1)")) == "LINESTRING (0.0 0.0, 1.0 1.0)"
    assert _geometry_to_wkt(polygon) == "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 0.0))"

    with pytest.raises(ValueError, match="cannot parse WKT"):
        _parse_wkt("NOT_A_WKT")