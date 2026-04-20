from __future__ import annotations

import pytest

from geoprompt.frame import GeoPromptFrame


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": (x, y)}


def _frame() -> GeoPromptFrame:
    return GeoPromptFrame(
        [
            {"id": "a", "region": "north", "score": 10.0, "flag": True, "geometry": _point(0, 0)},
            {"id": "b", "region": "north", "score": None, "flag": False, "geometry": _point(1, 1)},
            {"id": "c", "region": "south", "score": 30.0, "flag": True, "geometry": _point(2, 2)},
        ],
        geometry_column="geometry",
        crs="EPSG:4326",
    )


def test_getitem_column_subset_and_mask_selection() -> None:
    frame = _frame()

    assert frame["region"] == ["north", "north", "south"]

    subset = frame[["id", "score"]]
    assert isinstance(subset, GeoPromptFrame)
    assert subset.columns == ["id", "score", "geometry"]

    masked = frame[[True, False, True]]
    assert isinstance(masked, GeoPromptFrame)
    assert [row["id"] for row in masked.to_records()] == ["a", "c"]


def test_setitem_and_assign_preserve_frame_integrity() -> None:
    frame = _frame().copy()
    frame["status"] = ["ok", "hold", "ok"]
    frame["priority"] = "normal"

    rows = frame.to_records()
    assert rows[0]["status"] == "ok"
    assert rows[1]["priority"] == "normal"
    assert frame.geometry_column == "geometry"


def test_index_column_survives_common_analyst_transforms() -> None:
    frame = _frame().set_index("id")

    filtered = frame.where(region="north")
    sorted_frame = frame.sort_values("region")
    filled = frame.fillna({"score": 0.0})

    assert filtered._index_column == "id"
    assert sorted_frame._index_column == "id"
    assert filled._index_column == "id"
    assert filtered.loc("a")["region"] == "north"


def test_tail_profile_and_null_summary_outputs() -> None:
    frame = _frame()

    assert [row["id"] for row in frame.tail(2)] == ["b", "c"]

    profile = frame.profile()
    assert profile["row_count"] == 3
    assert profile["geometry_column"] == "geometry"
    assert profile["null_counts"]["score"] == 1

    missing = frame.isna()
    assert missing.to_records()[1]["score"] is True
    assert missing.to_records()[0]["score"] is False


def test_concat_preserves_union_of_columns_and_order() -> None:
    left = GeoPromptFrame([
        {"id": "a", "score": 10, "geometry": _point(0, 0)},
    ], geometry_column="geometry")
    right = GeoPromptFrame([
        {"id": "b", "category": "priority", "geometry": _point(1, 1)},
    ], geometry_column="geometry")

    combined = GeoPromptFrame.concat([left, right])
    rows = combined.to_records()

    assert combined.columns == ["id", "score", "geometry", "category"]
    assert rows[0].get("category") is None
    assert rows[1].get("score") is None


def test_groupby_multi_aggregation_and_crs_preservation() -> None:
    frame = _frame()

    grouped = frame.groupby("region").agg({"score": ["mean", "max"], "id": "count"})
    records = grouped.sort_values("region").to_records()

    assert grouped.crs == "EPSG:4326"
    assert records[0]["id_count"] == 2
    assert pytest.approx(records[1]["score_mean"], rel=1e-9) == 30.0


def test_tabular_roundtrips_preserve_geometry_metadata() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    frame = _frame()

    pandas_df = frame.to_pandas()
    assert pandas_df.attrs["crs"] == "EPSG:4326"
    assert pandas_df.attrs["geometry_column"] == "geometry"

    restored_from_pandas = GeoPromptFrame.from_pandas(pandas_df)
    assert restored_from_pandas.crs == "EPSG:4326"
    assert restored_from_pandas.geometry_column == "geometry"

    arrow_table = frame.to_arrow()
    metadata = arrow_table.schema.metadata or {}
    assert metadata.get(b"geoprompt.crs") == b"EPSG:4326"
    assert metadata.get(b"geoprompt.geometry_column") == b"geometry"

    restored_from_arrow = GeoPromptFrame.from_arrow(arrow_table)
    assert restored_from_arrow.crs == "EPSG:4326"
    assert restored_from_arrow.geometry_column == "geometry"
