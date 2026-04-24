from __future__ import annotations

from geoprompt.query import calculate_field, data_dictionary, parse_expression, select_by_attributes, where_clause


def _rows() -> list[dict[str, object]]:
    return [
        {"name": "hospital", "tier": "high", "score": 3, "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"name": "pump", "tier": "medium", "score": 1, "geometry": {"type": "Point", "coordinates": [1, 1]}},
        {"name": "hydrant", "tier": None, "score": 2, "geometry": {"type": "Point", "coordinates": [2, 2]}},
    ]


def test_query_module_sql_like_selection_and_calculation() -> None:
    rows = _rows()

    hospital_predicate = parse_expression("name LIKE 'h%'")
    between_predicate = parse_expression("score BETWEEN 2 AND 3")
    composite = where_clause(["score >= 2", "tier IS NOT NULL"])

    assert hospital_predicate(rows[0]) is True
    assert between_predicate(rows[2]) is True
    assert composite(rows[0]) is True
    assert composite(rows[2]) is False
    assert select_by_attributes(rows, "score >= 2") == {0, 2}
    assert select_by_attributes(rows, "name LIKE 'p%'", mode="add", existing_selection={0, 2}) == {0, 1, 2}
    assert calculate_field(rows, "priority", "score * 10") == 3
    assert [row["priority"] for row in rows] == [30, 10, 20]


def test_query_module_data_dictionary_reports_nullable_types() -> None:
    dictionary = data_dictionary(_rows())
    by_name = {row["name"]: row for row in dictionary}

    assert by_name["name"]["sample_values"] == ["hospital", "pump", "hydrant"]
    assert by_name["tier"]["nullable"] is True
    assert by_name["score"]["type"] == "int"