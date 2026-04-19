"""Database connectors for PostGIS and DuckDB spatial data.

All database dependencies are lazily imported and optional.  Install the
``db`` extra for PostGIS support or ``duckdb`` for DuckDB support.
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Sequence

from .geometry import Geometry, normalize_geometry


Record = dict[str, Any]


# ---------------------------------------------------------------------------
# PostGIS helpers
# ---------------------------------------------------------------------------

def read_postgis(
    query: str,
    connection_string: str,
    geometry_column: str = "geom",
    crs: str | None = None,
) -> list[Record]:
    """Read spatial data from a PostGIS query.

    Requires ``sqlalchemy`` and ``psycopg2`` (or ``psycopg``).

    Args:
        query: SQL query returning rows with a geometry column.
        connection_string: SQLAlchemy connection string
            (e.g. ``"postgresql://user:pass@host/dbname"``).
        geometry_column: Name of the geometry column in the result set.
        crs: Optional CRS string to attach to results.

    Returns:
        List of row dicts suitable for :class:`~geoprompt.frame.GeoPromptFrame`.
    """
    sa = importlib.import_module("sqlalchemy")
    engine = sa.create_engine(connection_string)

    with engine.connect() as conn:
        result = conn.execute(sa.text(query))
        columns = list(result.keys())
        rows: list[Record] = []
        for db_row in result:
            row_dict: Record = {}
            for col, val in zip(columns, db_row):
                if col == geometry_column:
                    row_dict["geometry"] = _parse_postgis_geometry(val)
                else:
                    row_dict[col] = val
            rows.append(row_dict)

    return rows


def write_postgis(
    records: Sequence[Record],
    table_name: str,
    connection_string: str,
    geometry_column: str = "geometry",
    srid: int = 4326,
    if_exists: str = "replace",
) -> int:
    """Write records to a PostGIS table.

    Creates or replaces the table with the correct geometry column.
    Requires ``sqlalchemy`` and ``psycopg2``.

    Args:
        records: Row dicts with a geometry column.
        table_name: Target table name.
        connection_string: SQLAlchemy connection string.
        geometry_column: Name of the geometry key in the records.
        srid: Spatial reference ID for the geometry column.
        if_exists: ``"replace"`` (drop and recreate) or ``"append"``.

    Returns:
        Number of rows written.
    """
    if not records:
        return 0

    sa = importlib.import_module("sqlalchemy")
    engine = sa.create_engine(connection_string)

    non_geom_cols = [k for k in records[0] if k != geometry_column]

    with engine.begin() as conn:
        if if_exists == "replace":
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {table_name}"))
            col_defs = ", ".join(f"{col} TEXT" for col in non_geom_cols)
            conn.execute(sa.text(
                f"CREATE TABLE {table_name} ({col_defs}, {geometry_column} geometry(Geometry, {srid}))"
            ))

        for record in records:
            geojson_str = json.dumps(_geometry_to_geojson(record[geometry_column]))
            values = {col: str(record.get(col, "")) for col in non_geom_cols}
            cols = ", ".join(non_geom_cols + [geometry_column])
            placeholders = ", ".join(f":{col}" for col in non_geom_cols)
            values["__geom__"] = geojson_str
            insert_sql = (
                f"INSERT INTO {table_name} ({cols}) "
                f"VALUES ({placeholders}, ST_SetSRID(ST_GeomFromGeoJSON(:__geom__), {srid}))"
            )
            conn.execute(sa.text(insert_sql), values)

    return len(records)


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def read_duckdb(
    query: str,
    database: str | None = None,
    geometry_column: str = "geom",
) -> list[Record]:
    """Read spatial data from DuckDB.

    Requires the ``duckdb`` package with the spatial extension loaded.

    Args:
        query: SQL query returning rows with a geometry column.
        database: Path to a DuckDB file, or ``None`` for in-memory.
        geometry_column: Name of the geometry column.

    Returns:
        List of row dicts.
    """
    duckdb = importlib.import_module("duckdb")
    conn = duckdb.connect(database or ":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    result = conn.execute(query)
    columns = [desc[0] for desc in result.description]
    rows: list[Record] = []
    for db_row in result.fetchall():
        row_dict: Record = {}
        for col, val in zip(columns, db_row):
            if col == geometry_column:
                row_dict["geometry"] = _parse_wkt_or_geojson(val)
            else:
                row_dict[col] = val
        rows.append(row_dict)

    conn.close()
    return rows


def write_duckdb(
    records: Sequence[Record],
    table_name: str,
    database: str | None = None,
    geometry_column: str = "geometry",
) -> int:
    """Write records to a DuckDB table with spatial support.

    Args:
        records: Row dicts with a geometry column.
        table_name: Target table name.
        database: Path to DuckDB file, or ``None`` for in-memory.
        geometry_column: Geometry key in the records.

    Returns:
        Number of rows written.
    """
    if not records:
        return 0

    duckdb = importlib.import_module("duckdb")
    conn = duckdb.connect(database or ":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    non_geom_cols = [k for k in records[0] if k != geometry_column]
    col_defs = ", ".join(f"{col} VARCHAR" for col in non_geom_cols)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} ({col_defs}, {geometry_column} GEOMETRY)")

    for record in records:
        geojson_str = json.dumps(_geometry_to_geojson(record[geometry_column]))
        vals = [str(record.get(col, "")) for col in non_geom_cols]
        placeholders = ", ".join(["?"] * len(non_geom_cols))
        conn.execute(
            f"INSERT INTO {table_name} VALUES ({placeholders}, ST_GeomFromGeoJSON(?))",
            vals + [geojson_str],
        )

    conn.close()
    return len(records)


# ---------------------------------------------------------------------------
# SpatiaLite / SQLite helpers
# ---------------------------------------------------------------------------


def read_spatialite(
    query: str,
    database: str,
    geometry_column: str = "geometry",
) -> list[Record]:
    """Read rows from a SQLite or SpatiaLite-style database.

    The geometry column may contain WKT or GeoJSON text.
    """
    import sqlite3

    conn = sqlite3.connect(database)
    try:
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows: list[Record] = []
        for db_row in cursor.fetchall():
            item: Record = {}
            for col, val in zip(columns, db_row):
                if col == geometry_column:
                    item["geometry"] = _parse_wkt_or_geojson(val)
                else:
                    item[col] = val
            rows.append(item)
        return rows
    finally:
        conn.close()


def write_spatialite(
    records: Sequence[Record],
    table_name: str,
    database: str,
    geometry_column: str = "geometry",
    if_exists: str = "replace",
) -> int:
    """Write rows to a SQLite or SpatiaLite-style database using WKT text."""
    if not records:
        return 0

    import sqlite3

    conn = sqlite3.connect(database)
    try:
        non_geom_cols = [k for k in records[0] if k != geometry_column]
        with conn:
            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                col_defs = ", ".join(f'"{col}" TEXT' for col in non_geom_cols)
                conn.execute(f"CREATE TABLE {table_name} ({col_defs}, " + geometry_column + " TEXT)")

            for record in records:
                values = [str(record.get(col, "")) for col in non_geom_cols]
                wkt = _geometry_to_wkt(record[geometry_column])
                placeholders = ", ".join(["?"] * (len(non_geom_cols) + 1))
                conn.execute(
                    f"INSERT INTO {table_name} VALUES ({placeholders})",
                    values + [wkt],
                )
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_postgis_geometry(value: Any) -> Geometry:
    """Parse a PostGIS geometry value (WKB hex, WKT, or GeoJSON) to internal format."""
    if isinstance(value, dict):
        return normalize_geometry(value)

    text = str(value)

    if text.startswith("{"):
        return normalize_geometry(json.loads(text))

    if any(text.upper().startswith(prefix) for prefix in ("POINT", "LINESTRING", "POLYGON", "MULTI")):
        return _parse_wkt(text)

    try:
        shapely_wkb = importlib.import_module("shapely.wkb")
        shape = shapely_wkb.loads(bytes.fromhex(text))
        geojson = json.loads(shape.__geo_interface__.__str__().replace("'", '"'))
        return normalize_geometry(geojson)
    except Exception:
        pass

    raise ValueError(f"cannot parse geometry value: {text[:100]}")


def _parse_wkt_or_geojson(value: Any) -> Geometry:
    """Parse WKT, GeoJSON string, or dict to internal geometry."""
    if isinstance(value, dict):
        return normalize_geometry(value)
    text = str(value)
    if text.startswith("{"):
        return normalize_geometry(json.loads(text))
    return _parse_wkt(text)


def _parse_wkt(wkt: str) -> Geometry:
    """Minimal WKT parser for common geometry types."""
    text = wkt.strip().upper()

    if text.startswith("POINT"):
        coords_str = text.replace("POINT", "").strip().strip("()")
        parts = coords_str.split()
        return normalize_geometry({"type": "Point", "coordinates": [float(parts[0]), float(parts[1])]})

    if text.startswith("LINESTRING"):
        coords_str = text.replace("LINESTRING", "").strip().strip("()")
        coords = [tuple(float(v) for v in pt.split()) for pt in coords_str.split(",")]
        return normalize_geometry({"type": "LineString", "coordinates": coords})

    if text.startswith("POLYGON"):
        body = text.replace("POLYGON", "").strip()
        # Remove outer parens
        body = body.strip("()")
        # Handle single ring
        coords = [tuple(float(v) for v in pt.strip().split()) for pt in body.split(",")]
        return normalize_geometry({"type": "Polygon", "coordinates": [coords]})

    try:
        shapely_wkt = importlib.import_module("shapely.wkt")
        shape = shapely_wkt.loads(wkt)
        geojson = shape.__geo_interface__
        return normalize_geometry(geojson)
    except Exception:
        raise ValueError(f"cannot parse WKT: {wkt[:100]}")


def _geometry_to_geojson(geometry: Geometry) -> dict[str, Any]:
    """Convert internal geometry to GeoJSON dict."""
    from .overlay import geometry_to_geojson
    return geometry_to_geojson(geometry)


def _geometry_to_wkt(geometry: Geometry) -> str:
    """Convert internal geometry to a minimal WKT representation."""
    geom = normalize_geometry(geometry)
    gtype = geom["type"]
    coords = geom["coordinates"]
    if gtype == "Point":
        return f"POINT ({coords[0]} {coords[1]})"
    if gtype == "LineString":
        body = ", ".join(f"{x} {y}" for x, y in coords)
        return f"LINESTRING ({body})"
    if gtype == "Polygon":
        body = ", ".join(f"{x} {y}" for x, y in coords)
        return f"POLYGON (({body}))"
    return json.dumps(_geometry_to_geojson(geom))


__all__ = [
    "read_duckdb",
    "read_postgis",
    "read_spatialite",
    "write_duckdb",
    "write_postgis",
    "write_spatialite",
]
