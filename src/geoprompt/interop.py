from __future__ import annotations

import importlib
import json
from typing import Any

from .frame import GeoPromptFrame
from .overlay import geometry_to_shapely


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def geopandas_available() -> bool:
    """Return True when GeoPandas is importable in the current environment."""
    return _module_available("geopandas")


def pandas_available() -> bool:
    """Return True when Pandas is importable in the current environment."""
    return _module_available("pandas")


def polars_available() -> bool:
    """Return True when Polars is importable in the current environment."""
    return _module_available("polars")


def arrow_available() -> bool:
    """Return True when PyArrow is importable in the current environment."""
    return _module_available("pyarrow")


def _load_geopandas() -> Any:
    try:
        return importlib.import_module("geopandas")
    except ImportError as exc:  # pragma: no cover - exercised by error path tests
        raise RuntimeError("Install GeoPandas support with 'pip install -e .[io]' or 'pip install geoprompt[all]'.") from exc


def _load_pandas() -> Any:
    try:
        return importlib.import_module("pandas")
    except ImportError as exc:  # pragma: no cover - exercised by error path tests
        raise RuntimeError("Install Pandas support with 'pip install -e .[io]' or 'pip install geoprompt[all]'.") from exc


def _load_polars() -> Any:
    try:
        return importlib.import_module("polars")
    except ImportError as exc:  # pragma: no cover - exercised by error path tests
        raise RuntimeError("Install Polars support with 'pip install polars'.") from exc


def _load_pyarrow() -> Any:
    try:
        return importlib.import_module("pyarrow")
    except ImportError as exc:  # pragma: no cover - exercised by error path tests
        raise RuntimeError("Install PyArrow support with 'pip install -e .[io]' or 'pip install geoprompt[all]'.") from exc


def _coerce_geometry(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__geo_interface__"):
        return value.__geo_interface__
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _records_from_tabular(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict(orient="records")
            if isinstance(records, list):
                return [dict(record) for record in records]
        except TypeError:
            pass
    if hasattr(data, "to_dicts"):
        return [dict(record) for record in data.to_dicts()]
    if hasattr(data, "to_pylist"):
        return [dict(record) for record in data.to_pylist()]
    if isinstance(data, list):
        return [dict(record) for record in data]
    return [dict(record) for record in data]


def to_geopandas(frame: GeoPromptFrame) -> Any:
    """Convert a GeoPromptFrame to a GeoPandas GeoDataFrame."""
    geopandas = _load_geopandas()
    rows = frame.to_records()
    geometry_values = [geometry_to_shapely(row[frame.geometry_column]) for row in rows]
    records = []
    for row in rows:
        copied = dict(row)
        copied.pop(frame.geometry_column, None)
        records.append(copied)
    return geopandas.GeoDataFrame(records, geometry=geometry_values, crs=frame.crs)


def from_geopandas(dataframe: Any, geometry_column: str | None = None) -> GeoPromptFrame:
    """Convert a GeoPandas GeoDataFrame into a GeoPromptFrame."""
    geometry_accessor = getattr(dataframe, "geometry", None)
    geometry_name = geometry_column or getattr(geometry_accessor, "name", None)
    if geometry_name is None:
        raise ValueError("geometry column could not be determined")

    rows: list[dict[str, object]] = []
    for _, row in dataframe.iterrows():
        record = dict(row)
        geometry = record[geometry_name]
        if hasattr(geometry, "geom_type"):
            if geometry.geom_type == "Point":
                record[geometry_name] = {"type": "Point", "coordinates": (float(geometry.x), float(geometry.y))}
            elif geometry.geom_type == "LineString":
                record[geometry_name] = {
                    "type": "LineString",
                    "coordinates": tuple((float(x), float(y)) for x, y in geometry.coords),
                }
            elif geometry.geom_type == "Polygon":
                record[geometry_name] = {
                    "type": "Polygon",
                    "coordinates": tuple((float(x), float(y)) for x, y in geometry.exterior.coords),
                }
            else:
                raise TypeError(f"unsupported geometry type: {geometry.geom_type}")
        rows.append(record)
    return GeoPromptFrame.from_records(rows, geometry=geometry_name, crs=getattr(dataframe, "crs", None))


def to_pandas(frame: GeoPromptFrame) -> Any:
    """Convert a GeoPromptFrame to a Pandas DataFrame."""
    pandas = _load_pandas()
    dataframe = pandas.DataFrame(frame.to_records())
    dataframe.attrs["crs"] = frame.crs
    dataframe.attrs["geometry_column"] = frame.geometry_column
    return dataframe


def from_pandas(
    dataframe: Any,
    *,
    geometry_column: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> GeoPromptFrame:
    """Convert a Pandas-like DataFrame into a GeoPromptFrame."""
    rows = _records_from_tabular(dataframe)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        if geometry_column in record and record[geometry_column] is not None:
            record[geometry_column] = _coerce_geometry(record[geometry_column])
        elif x_column is not None and y_column is not None:
            record[geometry_column] = {
                "type": "Point",
                "coordinates": [float(record.pop(x_column)), float(record.pop(y_column))],
            }
        else:
            raise ValueError("dataframe must include a geometry column or x/y columns")
        normalized_rows.append(record)
    resolved_crs = crs
    if resolved_crs is None:
        attrs = getattr(dataframe, "attrs", None)
        if isinstance(attrs, dict):
            resolved_crs = attrs.get("crs")
        if resolved_crs is None:
            resolved_crs = getattr(dataframe, "crs", None)
    return GeoPromptFrame.from_records(normalized_rows, geometry=geometry_column, crs=resolved_crs)


def to_polars(frame: GeoPromptFrame) -> Any:
    """Convert a GeoPromptFrame to a Polars DataFrame."""
    polars = _load_polars()
    return polars.DataFrame(frame.to_records())


def from_polars(
    dataframe: Any,
    *,
    geometry_column: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> GeoPromptFrame:
    """Convert a Polars DataFrame into a GeoPromptFrame."""
    return from_pandas(
        dataframe,
        geometry_column=geometry_column,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
    )


def to_arrow(frame: GeoPromptFrame) -> Any:
    """Convert a GeoPromptFrame to a PyArrow table."""
    pyarrow = _load_pyarrow()
    table = pyarrow.Table.from_pylist(frame.to_records())
    metadata = dict(table.schema.metadata or {})
    metadata[b"geoprompt.geometry_column"] = frame.geometry_column.encode("utf-8")
    metadata[b"geoprompt.crs"] = (frame.crs or "").encode("utf-8")
    return table.replace_schema_metadata(metadata)


def from_arrow(
    table: Any,
    *,
    geometry_column: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> GeoPromptFrame:
    """Convert a PyArrow table into a GeoPromptFrame."""
    metadata = getattr(getattr(table, "schema", None), "metadata", None) or {}
    resolved_geometry = geometry_column
    if geometry_column == "geometry":
        resolved_geometry = metadata.get(b"geoprompt.geometry_column", b"geometry").decode("utf-8")
    resolved_crs = crs
    if resolved_crs is None:
        raw_crs = metadata.get(b"geoprompt.crs", b"").decode("utf-8")
        resolved_crs = raw_crs or None
    return from_pandas(
        table,
        geometry_column=resolved_geometry,
        crs=resolved_crs,
        x_column=x_column,
        y_column=y_column,
    )


def dataframe_protocol(
    frame: GeoPromptFrame,
    *,
    nan_as_null: bool = False,
    allow_copy: bool = True,
) -> Any:
    """Return a dataframe interchange object for libraries supporting the protocol."""
    dataframe = to_pandas(frame)
    protocol = getattr(dataframe, "__dataframe__", None)
    if protocol is None:
        return dataframe
    return protocol(nan_as_null=nan_as_null, allow_copy=allow_copy)


__all__ = [
    "arrow_available",
    "dataframe_protocol",
    "from_arrow",
    "from_geopandas",
    "from_pandas",
    "from_polars",
    "geopandas_available",
    "pandas_available",
    "polars_available",
    "to_arrow",
    "to_geopandas",
    "to_pandas",
    "to_polars",
]
