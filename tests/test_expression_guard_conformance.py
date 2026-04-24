from __future__ import annotations

import pytest

from geoprompt.data_management import field_calculate
from geoprompt.frame import GeoPromptFrame
from geoprompt.geoprocessing import select_by_attributes
from geoprompt.io import _row_matches_filter
from geoprompt.query import calculate_field as query_calculate_field
from geoprompt.raster import raster_lazy_algebra
from geoprompt.safe_expression import ExpressionValidationError


def _sample_frame() -> GeoPromptFrame:
    return GeoPromptFrame.from_records([
        {"a": 1, "b": 2, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
        {"a": 2, "b": 3, "geometry": {"type": "Point", "coordinates": [1.0, 1.0]}},
    ])


def _oversized_expression() -> str:
    return " + ".join(["1"] * 1200)


def _deep_expression() -> str:
    depth = 40
    expr = "1"
    for _ in range(depth):
        expr = f"(1 + {expr})"
    return expr


@pytest.mark.parametrize("expression", [_oversized_expression(), _deep_expression()])
def test_expression_limits_fail_with_typed_contracts_across_modules(expression: str) -> None:
    frame = _sample_frame()

    with pytest.raises((ValueError, ExpressionValidationError)):
        _row_matches_filter({"a": 1}, expression)

    with pytest.raises((ValueError, ExpressionValidationError)):
        frame.query(expression)

    with pytest.raises((ValueError, ExpressionValidationError)):
        query_calculate_field([{"a": 1}], "x", expression)

    with pytest.raises((ValueError, ExpressionValidationError)):
        field_calculate([{"a": 1}], "x", expression)

    with pytest.raises((ValueError, ExpressionValidationError)):
        select_by_attributes(frame, expression)

    with pytest.raises((ValueError, ExpressionValidationError)):
        raster_lazy_algebra(expression, {"a": {"data": [[1.0]], "transform": (0.0, 1.0, 1.0, 1.0), "nodata": None}})
