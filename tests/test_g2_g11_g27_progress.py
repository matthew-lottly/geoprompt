from __future__ import annotations

import json
from pathlib import Path

import geoprompt as gp
from geoprompt import geocoding as geocoding_module


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": (x, y)}


def _polygon(coords: list[tuple[float, float]]) -> dict:
    ring = list(coords)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": (tuple(ring),)}


def test_g2_frame_progress_features_work() -> None:
    frame = gp.GeoPromptFrame([
        {"name": "a", "score": 2, "geometry": _point(0, 0)},
        {"name": "b", "score": 5, "geometry": _point(2, 2)},
        {"name": "c", "score": 9, "geometry": _point(5, 5)},
    ])

    selected = frame.cx[0:3, 0:3]
    assert len(selected) == 2
    assert frame.total_bounds == (0.0, 0.0, 5.0, 5.0)
    assert frame.active_geometry_name == "geometry"
    assert frame.area == [0.0, 0.0, 0.0]
    assert frame.length == [0.0, 0.0, 0.0]
    assert frame.is_valid == [True, True, True]
    assert frame.estimate_utm_crs().startswith("EPSG:")
    assert list(frame.iterfeatures())[0]["type"] == "Feature"
    assert frame.__geo_interface__["type"] == "FeatureCollection"


def test_g2_plot_returns_axes_when_matplotlib_available() -> None:
    matplotlib = __import__("pytest").importorskip("matplotlib.pyplot")
    _, axes = matplotlib.subplots()
    frame = gp.GeoPromptFrame([
        {"name": "a", "geometry": _point(0, 0)},
        {"name": "b", "geometry": _polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
    ])
    rendered = frame.plot(ax=axes)
    assert rendered is axes


def test_g11_batch_geocode_supports_fallback_and_cache(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_forward_geocode(address: str, *, provider: str = "nominatim", api_key=None, limit: int = 1):
        calls.append((address, provider))
        if provider == "nominatim":
            raise RuntimeError("temporary failure")
        return [{"address": address, "lat": 47.0, "lon": -122.0, "display_name": address, "type": "house"}]

    monkeypatch.setattr(geocoding_module, "forward_geocode", fake_forward_geocode)

    cache: dict[str, dict[str, object]] = {}
    results = gp.batch_geocode(["Seattle, WA"], providers=["nominatim", "arcgis"], delay=0, retries=1, cache=cache)
    cached_results = gp.batch_geocode(["Seattle, WA"], providers=["nominatim", "arcgis"], delay=0, cache=cache)

    assert results[0]["provider"] == "arcgis"
    assert results[0]["score"] > 0.0
    assert cached_results[0]["provider"] == "arcgis"
    assert calls.count(("Seattle, WA", "arcgis")) == 1
    assert gp.geocode_quality_score({"lat": 1, "lon": 2, "type": "house", "score": 0.9}) == 0.9


def test_g27_remote_listing_metadata_and_stac_traversal(tmp_path: Path) -> None:
    item_path = tmp_path / "item.json"
    item_path.write_text(json.dumps({"type": "Feature", "id": "item-1", "assets": {"data": {"href": "data.tif"}}}), encoding="utf-8")

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "stac_version": "1.0.0",
        "type": "Catalog",
        "id": "demo",
        "links": [{"rel": "item", "href": "item.json"}],
    }), encoding="utf-8")

    entries = gp.list_remote_dataset_entries(tmp_path)
    metadata = gp.inspect_remote_dataset_metadata(catalog_path)
    catalog = gp.read_stac_catalog(catalog_path)

    assert any(entry["name"] == "catalog.json" for entry in entries)
    assert metadata["scheme"] == "file"
    assert metadata["exists"] is True
    assert len(catalog["resolved_links"]) == 1
    assert catalog["resolved_links"][0]["payload"]["id"] == "item-1"