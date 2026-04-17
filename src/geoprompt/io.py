"""Unified spatial data I/O with workload presets and progress callbacks.

Supports GeoJSON, CSV, GeoParquet, and other geographic data formats.
Workload presets (small/medium/large/huge) provide sampling and batching tuning
for different dataset sizes. Progress callbacks enable real-time monitoring of
read/write operations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .frame import GeoPromptFrame
from .geometry import geometry_type


WORKLOAD_PRESETS: dict[str, dict[str, int | None]] = {
    "small": {"chunk_size": 5000, "sample_step": 1, "limit_rows": 100000},
    "medium": {"chunk_size": 20000, "sample_step": 1, "limit_rows": None},
    "large": {"chunk_size": 50000, "sample_step": 1, "limit_rows": None},
    "huge": {"chunk_size": 100000, "sample_step": 2, "limit_rows": None},
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_read_options(
    *,
    data_path: Path,
    sample_step: int,
    limit_rows: int | None,
    chunk_size: int | None = None,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"input path does not exist: {data_path}")
    if sample_step <= 0:
        raise ValueError("sample_step must be >= 1")
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be >= 1 when provided")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")


def _parse_coordinate_pairs(body: str) -> list[list[float]] | None:
    coordinates: list[list[float]] = []
    for pair in body.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            return None
        try:
            coordinates.append([float(parts[0]), float(parts[1])])
        except ValueError:
            return None
    return coordinates or None


def _split_wkt_groups(body: str) -> list[str]:
    groups: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(body):
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(body[start:index].strip())
                start = None
    return groups


def _split_top_level_parts(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    token: list[str] = []
    for char in body:
        if char == "," and depth == 0:
            part = "".join(token).strip()
            if part:
                parts.append(part)
            token = []
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        token.append(char)
    part = "".join(token).strip()
    if part:
        parts.append(part)
    return parts


def _strip_outer_group(body: str) -> str:
    text = body.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wrapped = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and index < len(text) - 1:
                wrapped = False
                break
        if not wrapped:
            break
        text = text[1:-1].strip()
    return text


def _parse_point_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("POINT"):
        return None
    if "(" not in text or ")" not in text:
        return None
    body = text[text.find("(") + 1 : text.rfind(")")].strip()
    coordinates = _parse_coordinate_pairs(body)
    if not coordinates:
        return None
    x_value, y_value = coordinates[0]
    return {"type": "Point", "coordinates": [x_value, y_value]}


def _parse_linestring_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("LINESTRING"):
        return None
    if "(" not in text or ")" not in text:
        return None
    body = text[text.find("(") + 1 : text.rfind(")")].strip()
    coordinates = _parse_coordinate_pairs(body)
    if not coordinates or len(coordinates) < 2:
        return None
    return {"type": "LineString", "coordinates": coordinates}


def _parse_polygon_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("POLYGON"):
        return None
    if "((" not in text or "))" not in text:
        return None
    body = text[text.find("((") + 2 : text.rfind("))")].strip()
    ring_text = body.split("),(", 1)[0]
    coordinates = _parse_coordinate_pairs(ring_text)
    if not coordinates or len(coordinates) < 3:
        return None
    return {"type": "Polygon", "coordinates": [coordinates]}


def _parse_multipoint_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("MULTIPOINT"):
        return None
    if "(" not in text or ")" not in text:
        return None
    body = text[text.find("(") + 1 : text.rfind(")")].replace("(", "").replace(")", "").strip()
    coordinates = _parse_coordinate_pairs(body)
    if not coordinates:
        return None
    return {"type": "MultiPoint", "coordinates": coordinates}


def _parse_multilinestring_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("MULTILINESTRING"):
        return None
    if "(" not in text or ")" not in text:
        return None
    body = _strip_outer_group(text[text.find("(") : text.rfind(")") + 1])
    lines = [_parse_coordinate_pairs(_strip_outer_group(group)) for group in _split_top_level_parts(body)]
    if not lines or any(line is None or len(line) < 2 for line in lines):
        return None
    return {"type": "MultiLineString", "coordinates": lines}


def _parse_multipolygon_wkt(value: str) -> dict[str, Any] | None:
    text = value.strip()
    upper = text.upper()
    if not upper.startswith("MULTIPOLYGON"):
        return None
    if "(" not in text or ")" not in text:
        return None
    body = _strip_outer_group(text[text.find("(") : text.rfind(")") + 1])
    polygons: list[list[list[float]]] = []
    for group in _split_top_level_parts(body):
        ring_text = _strip_outer_group(group)
        coordinates = _parse_coordinate_pairs(ring_text)
        if not coordinates or len(coordinates) < 3:
            return None
        polygons.append(coordinates)
    if not polygons:
        return None
    return {"type": "MultiPolygon", "coordinates": [[polygon] for polygon in polygons]}


def _coerce_geometry_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and "type" in value and "coordinates" in value:
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and "type" in parsed and "coordinates" in parsed:
            return parsed
    return (
        _parse_point_wkt(text)
        or _parse_multipoint_wkt(text)
        or _parse_linestring_wkt(text)
        or _parse_multilinestring_wkt(text)
        or _parse_polygon_wkt(text)
        or _parse_multipolygon_wkt(text)
    )


def _feature_to_record(feature: dict[str, Any], geometry: str) -> dict[str, Any]:
    if feature.get("type") != "Feature":
        raise TypeError("GeoJSON input must contain Feature objects")
    properties = dict(feature.get("properties") or {})
    properties[geometry] = feature.get("geometry")
    if "id" in feature and "site_id" not in properties:
        properties["site_id"] = str(feature["id"])
    return properties


def _extract_crs(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("crs"), str):
        return str(payload["crs"])
    crs_mapping = payload.get("crs")
    if isinstance(crs_mapping, dict):
        properties = crs_mapping.get("properties")
        if isinstance(properties, dict) and "name" in properties:
            return str(properties["name"])
    return None


def _records_from_payload(payload: Any, geometry: str = "geometry") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise TypeError("feature records must be mappings")
        return [dict(item) for item in payload]

    if isinstance(payload, dict) and "records" in payload:
        records = payload.get("records")
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise TypeError("record wrapper payload must contain a record list")
        return [dict(item) for item in records]

    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list):
            raise TypeError("FeatureCollection must contain a feature list")
        return [_feature_to_record(feature, geometry=geometry) for feature in features]

    if isinstance(payload, dict) and payload.get("type") == "Feature":
        return [_feature_to_record(payload, geometry=geometry)]

    raise TypeError("input must be a record list, a GeoJSON Feature, or a GeoJSON FeatureCollection")


def _apply_row_limits(
    rows: Iterable[dict[str, Any]],
    limit_rows: int | None,
    sample_step: int,
) -> list[dict[str, Any]]:
    if sample_step <= 0:
        raise ValueError("sample_step must be >= 1")
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if sample_step > 1 and index % sample_step != 0:
            continue
        selected.append(dict(row))
        if limit_rows is not None and len(selected) >= limit_rows:
            break
    return selected


def _records_from_csv(
    path: str | Path,
    *,
    geometry: str,
    x_column: str | None,
    y_column: str | None,
    geometry_column: str | None,
    use_columns: Sequence[str] | None,
    limit_rows: int | None,
    sample_step: int,
    delimiter: str,
    encoding: str,
) -> list[dict[str, Any]]:
    selected_columns = set(use_columns or [])
    records: list[dict[str, Any]] = []

    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for raw_index, raw_row in enumerate(reader):
            if sample_step > 1 and raw_index % sample_step != 0:
                continue

            row = dict(raw_row)
            if selected_columns:
                row = {key: row[key] for key in row.keys() if key in selected_columns}

            resolved_geometry: dict[str, Any] | None = None
            if x_column is not None and y_column is not None:
                if x_column not in raw_row or y_column not in raw_row:
                    raise KeyError(f"CSV is missing required columns '{x_column}' and '{y_column}'")
                x_value = float(raw_row[x_column])
                y_value = float(raw_row[y_column])
                resolved_geometry = {"type": "Point", "coordinates": [x_value, y_value]}
            elif geometry_column is not None:
                resolved_geometry = _coerce_geometry_value(raw_row.get(geometry_column))
            elif geometry in row:
                resolved_geometry = _coerce_geometry_value(row.get(geometry))

            if resolved_geometry is None:
                raise ValueError(
                    "tabular spatial reads require x_column/y_column or geometry_column"
                )

            row[geometry] = resolved_geometry
            records.append(row)

            if limit_rows is not None and len(records) >= limit_rows:
                break

    return records


def _iter_csv_records(
    path: str | Path,
    *,
    geometry: str,
    x_column: str | None,
    y_column: str | None,
    geometry_column: str | None,
    use_columns: Sequence[str] | None,
    sample_step: int,
    delimiter: str,
    encoding: str,
) -> Iterable[dict[str, Any]]:
    selected_columns = set(use_columns or [])
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for raw_index, raw_row in enumerate(reader):
            if sample_step > 1 and raw_index % sample_step != 0:
                continue

            row = dict(raw_row)
            if selected_columns:
                row = {key: row[key] for key in row.keys() if key in selected_columns}

            resolved_geometry: dict[str, Any] | None = None
            if x_column is not None and y_column is not None:
                if x_column not in raw_row or y_column not in raw_row:
                    raise KeyError(f"CSV is missing required columns '{x_column}' and '{y_column}'")
                x_value = float(raw_row[x_column])
                y_value = float(raw_row[y_column])
                resolved_geometry = {"type": "Point", "coordinates": [x_value, y_value]}
            elif geometry_column is not None:
                resolved_geometry = _coerce_geometry_value(raw_row.get(geometry_column))
            elif geometry in row:
                resolved_geometry = _coerce_geometry_value(row.get(geometry))

            if resolved_geometry is None:
                raise ValueError(
                    "tabular spatial reads require x_column/y_column or geometry_column"
                )

            row[geometry] = resolved_geometry
            yield row


def _iter_frame_chunks(
    rows: Iterable[dict[str, Any]],
    *,
    geometry: str,
    crs: str | None,
    chunk_size: int,
    limit_rows: int | None,
) -> Iterable[GeoPromptFrame]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")

    emitted = 0
    bucket: list[dict[str, Any]] = []
    for row in rows:
        if limit_rows is not None and emitted >= limit_rows:
            break
        bucket.append(dict(row))
        emitted += 1
        if len(bucket) >= chunk_size:
            yield GeoPromptFrame.from_records(bucket, geometry=geometry, crs=crs)
            bucket = []
    if bucket:
        yield GeoPromptFrame.from_records(bucket, geometry=geometry, crs=crs)


def _read_with_geopandas(
    path: str | Path,
    *,
    geometry: str,
    crs: str | None,
    layer: str | None,
    bbox: tuple[float, float, float, float] | None,
    use_columns: Sequence[str] | None,
    limit_rows: int | None,
    sample_step: int,
) -> GeoPromptFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for this format. Install extras: pip install geoprompt[io,compare]"
        ) from exc

    data_path = Path(path)
    if data_path.suffix.lower() == ".parquet":
        if layer is not None:
            raise ValueError("layer is not supported for parquet inputs")
        if bbox is not None:
            raise ValueError("bbox is not supported for parquet inputs")
        parquet_kwargs: dict[str, Any] = {}
        if use_columns:
            parquet_kwargs["columns"] = list(use_columns)
        frame = gpd.read_parquet(data_path, **parquet_kwargs)
        if limit_rows is not None:
            frame = frame.iloc[:limit_rows]
    else:
        read_kwargs: dict[str, Any] = {}
        if layer is not None:
            read_kwargs["layer"] = layer
        if bbox is not None:
            read_kwargs["bbox"] = bbox
        if use_columns:
            read_kwargs["columns"] = list(use_columns)
        if limit_rows is not None:
            read_kwargs["rows"] = slice(0, limit_rows)

        frame = gpd.read_file(data_path, **read_kwargs)
    if sample_step > 1:
        frame = frame.iloc[::sample_step]
    if crs is not None and frame.crs is not None and str(frame.crs) != crs:
        frame = frame.to_crs(crs)

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = row.drop(labels=[frame.geometry.name]).to_dict()
        geom = row[frame.geometry.name]
        record[geometry] = geom.__geo_interface__ if geom is not None else None
        rows.append(record)

    filtered = [r for r in rows if r.get(geometry) is not None]
    resolved_crs = crs or (str(frame.crs) if frame.crs is not None else None)
    return GeoPromptFrame.from_records(filtered, geometry=geometry, crs=resolved_crs)


def _frame_from_path(path: str | Path, geometry: str = "geometry", crs: str | None = None) -> GeoPromptFrame:
    payload = _read_json(path)
    return GeoPromptFrame.from_records(
        _records_from_payload(payload, geometry=geometry),
        geometry=geometry,
        crs=crs or _extract_crs(payload),
    )


def read_data(
    path: str | Path,
    *,
    geometry: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_column: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int = 1,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> GeoPromptFrame:
    """Unified spatial reader for GeoJSON/JSON, tabular files, and geofiles.

    Large-dataset knobs:
    - ``limit_rows``: cap row count
    - ``sample_step``: keep every Nth row
    - ``use_columns``: read selected attributes only
    - ``bbox`` and ``layer`` for geospatial stores
    """
    data_path = Path(path)
    _validate_read_options(
        data_path=data_path,
        sample_step=sample_step,
        limit_rows=limit_rows,
    )
    suffix = data_path.suffix.lower()

    if suffix in {".json", ".geojson"}:
        payload = _read_json(data_path)
        rows = _records_from_payload(payload, geometry=geometry)
        rows = _apply_row_limits(rows, limit_rows=limit_rows, sample_step=sample_step)
        if use_columns:
            keep = set(use_columns)
            rows = [
                {**{k: v for k, v in row.items() if k in keep}, geometry: row[geometry]}
                for row in rows
            ]
        return GeoPromptFrame.from_records(rows, geometry=geometry, crs=crs or _extract_crs(payload))

    if suffix in {".csv", ".tsv", ".txt"}:
        delim = "\t" if suffix == ".tsv" else delimiter
        rows = _records_from_csv(
            data_path,
            geometry=geometry,
            x_column=x_column,
            y_column=y_column,
            geometry_column=geometry_column,
            use_columns=use_columns,
            limit_rows=limit_rows,
            sample_step=sample_step,
            delimiter=delim,
            encoding=encoding,
        )
        return GeoPromptFrame.from_records(rows, geometry=geometry, crs=crs)

    if suffix in {".shp", ".gpkg", ".fgb", ".gdb", ".parquet"}:
        return _read_with_geopandas(
            data_path,
            geometry=geometry,
            crs=crs,
            layer=layer,
            bbox=bbox,
            use_columns=use_columns,
            limit_rows=limit_rows,
            sample_step=sample_step,
        )

    raise ValueError(f"unsupported input format for path: {data_path}")


def get_workload_preset(name: str) -> dict[str, int | None]:
    """Return the default IO controls for a named workload preset."""
    preset = WORKLOAD_PRESETS.get(name.lower())
    if preset is None:
        valid = ", ".join(sorted(WORKLOAD_PRESETS.keys()))
        raise ValueError(f"unknown workload preset '{name}'. expected one of: {valid}")
    return dict(preset)


def read_data_with_preset(
    path: str | Path,
    *,
    preset: str = "large",
    geometry: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_column: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int | None = None,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> GeoPromptFrame:
    """Read data with workload defaults and explicit per-call overrides."""
    settings = get_workload_preset(preset)
    resolved_limit = limit_rows if limit_rows is not None else settings["limit_rows"]
    resolved_sample = sample_step if sample_step is not None else int(settings["sample_step"] or 1)
    return read_data(
        path,
        geometry=geometry,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
        geometry_column=geometry_column,
        use_columns=use_columns,
        limit_rows=resolved_limit,
        sample_step=resolved_sample,
        layer=layer,
        bbox=bbox,
        delimiter=delimiter,
        encoding=encoding,
    )


def iter_data(
    path: str | Path,
    *,
    geometry: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_column: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int = 1,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_size: int = 50000,
    delimiter: str = ",",
    encoding: str = "utf-8",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Iterable[GeoPromptFrame]:
    """Yield data as chunked ``GeoPromptFrame`` batches for large datasets."""
    data_path = Path(path)
    _validate_read_options(
        data_path=data_path,
        sample_step=sample_step,
        limit_rows=limit_rows,
        chunk_size=chunk_size,
    )
    suffix = data_path.suffix.lower()

    emitted_chunks = 0
    emitted_rows = 0

    def _notify(chunk: GeoPromptFrame) -> None:
        nonlocal emitted_chunks, emitted_rows
        emitted_chunks += 1
        emitted_rows += len(chunk)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "chunk",
                    "path": str(data_path),
                    "chunk_index": emitted_chunks,
                    "chunk_rows": len(chunk),
                    "rows_emitted": emitted_rows,
                }
            )

    if suffix in {".json", ".geojson"}:
        payload = _read_json(data_path)
        rows = _records_from_payload(payload, geometry=geometry)
        if use_columns:
            keep = set(use_columns)
            rows = [
                {**{k: v for k, v in row.items() if k in keep}, geometry: row[geometry]}
                for row in rows
            ]
        for chunk in _iter_frame_chunks(
            rows,
            geometry=geometry,
            crs=crs or _extract_crs(payload),
            chunk_size=chunk_size,
            limit_rows=limit_rows,
        ):
            _notify(chunk)
            yield chunk
        return

    if suffix in {".csv", ".tsv", ".txt"}:
        delim = "\t" if suffix == ".tsv" else delimiter
        rows_iter = _iter_csv_records(
            data_path,
            geometry=geometry,
            x_column=x_column,
            y_column=y_column,
            geometry_column=geometry_column,
            use_columns=use_columns,
            sample_step=sample_step,
            delimiter=delim,
            encoding=encoding,
        )
        for chunk in _iter_frame_chunks(
            rows_iter,
            geometry=geometry,
            crs=crs,
            chunk_size=chunk_size,
            limit_rows=limit_rows,
        ):
            _notify(chunk)
            yield chunk
        return

    if suffix in {".shp", ".gpkg", ".fgb", ".gdb", ".parquet"}:
        frame = _read_with_geopandas(
            data_path,
            geometry=geometry,
            crs=crs,
            layer=layer,
            bbox=bbox,
            use_columns=use_columns,
            limit_rows=limit_rows,
            sample_step=sample_step,
        )
        for chunk in _iter_frame_chunks(
            frame.to_records(),
            geometry=geometry,
            crs=frame.crs,
            chunk_size=chunk_size,
            limit_rows=limit_rows,
        ):
            _notify(chunk)
            yield chunk
        return

    raise ValueError(f"unsupported input format for path: {data_path}")


def iter_data_with_preset(
    path: str | Path,
    *,
    preset: str = "large",
    geometry: str = "geometry",
    crs: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_column: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int | None = None,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_size: int | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Iterable[GeoPromptFrame]:
    """Iterate data in chunks with workload defaults and explicit overrides."""
    settings = get_workload_preset(preset)
    resolved_limit = limit_rows if limit_rows is not None else settings["limit_rows"]
    resolved_sample = sample_step if sample_step is not None else int(settings["sample_step"] or 1)
    resolved_chunk = chunk_size if chunk_size is not None else int(settings["chunk_size"] or 50000)
    return iter_data(
        path,
        geometry=geometry,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
        geometry_column=geometry_column,
        use_columns=use_columns,
        limit_rows=resolved_limit,
        sample_step=resolved_sample,
        layer=layer,
        bbox=bbox,
        chunk_size=resolved_chunk,
        delimiter=delimiter,
        encoding=encoding,
        progress_callback=progress_callback,
    )


def read_csv_points(
    path: str | Path,
    *,
    x_column: str,
    y_column: str,
    preset: str = "large",
    geometry: str = "geometry",
    crs: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> GeoPromptFrame:
    """Convenience wrapper for point CSV reads with workload presets."""
    return read_data_with_preset(
        path,
        preset=preset,
        geometry=geometry,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
        use_columns=use_columns,
        limit_rows=limit_rows,
        sample_step=sample_step,
        delimiter=delimiter,
        encoding=encoding,
    )


def iter_csv_points(
    path: str | Path,
    *,
    x_column: str,
    y_column: str,
    preset: str = "large",
    geometry: str = "geometry",
    crs: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int | None = None,
    chunk_size: int | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Iterable[GeoPromptFrame]:
    """Convenience wrapper for chunked point CSV reads with workload presets."""
    return iter_data_with_preset(
        path,
        preset=preset,
        geometry=geometry,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
        use_columns=use_columns,
        limit_rows=limit_rows,
        sample_step=sample_step,
        chunk_size=chunk_size,
        delimiter=delimiter,
        encoding=encoding,
        progress_callback=progress_callback,
    )


def read_table(
    path: str | Path,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_column: str | None = None,
    geometry: str = "geometry",
    crs: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int = 1,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> GeoPromptFrame:
    """Simple tabular reader wrapper for point-based or geometry-column CSV/TSV data."""
    return read_data(
        path,
        geometry=geometry,
        crs=crs,
        x_column=x_column,
        y_column=y_column,
        geometry_column=geometry_column,
        use_columns=use_columns,
        limit_rows=limit_rows,
        sample_step=sample_step,
        delimiter=delimiter,
        encoding=encoding,
    )


def read_points(path: str | Path, geometry: str = "geometry", crs: str | None = None) -> GeoPromptFrame:
    frame = read_data(path, geometry=geometry, crs=crs)
    if any(geometry_type(row[geometry]) != "Point" for row in frame):
        raise TypeError("read_points only accepts point geometry inputs")
    return frame


def read_features(path: str | Path, geometry: str = "geometry", crs: str | None = None) -> GeoPromptFrame:
    return read_data(path, geometry=geometry, crs=crs)


def read_geojson(path: str | Path, geometry: str = "geometry", crs: str | None = None) -> GeoPromptFrame:
    return read_data(path, geometry=geometry, crs=crs)


def _as_geojson_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    geometry_kind = str(geometry["type"])
    coordinates = geometry["coordinates"]
    if geometry_kind == "Point":
        return {"type": "Point", "coordinates": list(coordinates)}
    if geometry_kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": [list(coord) for coord in coordinates]}
    if geometry_kind == "LineString":
        return {"type": "LineString", "coordinates": [list(coord) for coord in coordinates]}
    if geometry_kind == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [[list(coord) for coord in line] for line in coordinates]}
    if geometry_kind == "Polygon":
        return {"type": "Polygon", "coordinates": [[list(coord) for coord in coordinates]]}
    if geometry_kind == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[[list(coord) for coord in polygon]] for polygon in coordinates]}
    raise TypeError(f"unsupported geometry type: {geometry_kind}")


def frame_to_geojson(frame: GeoPromptFrame, geometry: str = "geometry", id_column: str = "site_id") -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for row in frame.to_records():
        properties = {key: value for key, value in row.items() if key != geometry}
        feature: dict[str, Any] = {
            "type": "Feature",
            "properties": properties,
            "geometry": _as_geojson_geometry(row[geometry]),
        }
        if id_column in properties:
            feature["id"] = str(properties[id_column])
        features.append(feature)
    collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if frame.crs is not None:
        collection["crs"] = {"type": "name", "properties": {"name": frame.crs}}
    return collection


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_geojson(path: str | Path, frame: GeoPromptFrame, geometry: str = "geometry", id_column: str = "site_id") -> Path:
    return write_json(path, frame_to_geojson(frame, geometry=geometry, id_column=id_column))


def write_data(
    path: str | Path,
    frame: GeoPromptFrame,
    *,
    geometry: str = "geometry",
    id_column: str = "site_id",
    delimiter: str = ",",
    encoding: str = "utf-8",
    layer: str | None = None,
    driver: str | None = None,
) -> Path:
    """Unified writer for GeoJSON/JSON/CSV and optional geospatial file formats."""
    output_path = Path(path)
    suffix = output_path.suffix.lower()

    if suffix in {".geojson", ".json"}:
        return write_geojson(output_path, frame, geometry=geometry, id_column=id_column)

    if suffix in {".csv", ".tsv", ".txt"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = frame.to_records()
        if not rows:
            output_path.write_text("", encoding=encoding)
            return output_path
        fieldnames = list(rows[0].keys())
        delim = "\t" if suffix == ".tsv" else delimiter
        with output_path.open("w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delim)
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                if geometry in serialized:
                    serialized[geometry] = json.dumps(serialized[geometry], separators=(",", ":"))
                writer.writerow(serialized)
        return output_path

    if suffix in {".shp", ".gpkg", ".fgb", ".gdb", ".parquet"}:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "geopandas is required to write this format. Install extras: pip install geoprompt[io,compare]"
            ) from exc

        collection = frame_to_geojson(frame, geometry=geometry, id_column=id_column)
        gdf = gpd.GeoDataFrame.from_features(collection["features"], crs=frame.crs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_file_kwargs: dict[str, Any] = {}
        if layer is not None:
            to_file_kwargs["layer"] = layer
        if driver is not None:
            to_file_kwargs["driver"] = driver
        if suffix == ".parquet":
            if layer is not None:
                raise ValueError("layer is not supported for parquet outputs")
            if driver is not None:
                raise ValueError("driver is not supported for parquet outputs")
            gdf.to_parquet(output_path)
        else:
            gdf.to_file(output_path, **to_file_kwargs)
        return output_path

    raise ValueError(f"unsupported output format for path: {output_path}")


def discover_layers(path: str | Path) -> list[dict[str, Any]]:
    """List available layers and their schemas in a geospatial file.

    Supports GeoPackage, Shapefile, GDB, and other GDAL-backed formats.
    Returns a list of dicts with ``layer``, ``feature_count``, ``geometry_type``,
    ``crs``, and ``columns`` keys.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"input path does not exist: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix in {".json", ".geojson"}:
        payload = _read_json(data_path)
        records = _records_from_payload(payload)
        geom_types: set[str] = set()
        columns: set[str] = set()
        for record in records:
            geom = record.get("geometry")
            if isinstance(geom, dict):
                geom_types.add(str(geom.get("type", "Unknown")))
            columns.update(k for k in record if k != "geometry")
        return [
            {
                "layer": data_path.stem,
                "feature_count": len(records),
                "geometry_type": sorted(geom_types)[0] if geom_types else "Unknown",
                "crs": _extract_crs(payload),
                "columns": sorted(columns),
            }
        ]

    try:
        import fiona
    except ImportError:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "fiona or geopandas is required for layer discovery. "
                "Install extras: pip install geoprompt[io,compare]"
            ) from exc

        if suffix == ".parquet":
            gdf = gpd.read_parquet(data_path)
            geom_col = gdf.geometry.name if gdf.geometry is not None else "geometry"
            gt = str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else "Unknown"
            crs_str = str(gdf.crs) if gdf.crs is not None else None
            cols = [c for c in gdf.columns if c != geom_col]
            return [
                {
                    "layer": data_path.stem,
                    "feature_count": len(gdf),
                    "geometry_type": gt,
                    "crs": crs_str,
                    "columns": cols,
                }
            ]

        layer_names = gpd.list_layers(data_path)["name"].tolist() if hasattr(gpd, "list_layers") else [None]
        layers: list[dict[str, Any]] = []
        for layer_name in layer_names:
            read_kw: dict[str, Any] = {}
            if layer_name is not None:
                read_kw["layer"] = layer_name
            gdf = gpd.read_file(data_path, rows=slice(0, 1), **read_kw)
            geom_col = gdf.geometry.name if gdf.geometry is not None else "geometry"
            gt = str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else "Unknown"
            crs_str = str(gdf.crs) if gdf.crs is not None else None
            cols = [c for c in gdf.columns if c != geom_col]
            count_gdf = gpd.read_file(data_path, **read_kw)
            layers.append(
                {
                    "layer": layer_name or data_path.stem,
                    "feature_count": len(count_gdf),
                    "geometry_type": gt,
                    "crs": crs_str,
                    "columns": cols,
                }
            )
        return layers

    layer_names = fiona.listlayers(str(data_path))
    layers = []
    for layer_name in layer_names:
        with fiona.open(str(data_path), layer=layer_name) as src:
            schema = src.schema
            gt = schema.get("geometry", "Unknown")
            crs_str = str(src.crs) if src.crs else None
            cols = list(schema.get("properties", {}).keys())
            layers.append(
                {
                    "layer": layer_name,
                    "feature_count": len(src),
                    "geometry_type": gt,
                    "crs": crs_str,
                    "columns": cols,
                }
            )
    return layers


def write_geoparquet(
    path: str | Path,
    frame: GeoPromptFrame,
    *,
    geometry: str = "geometry",
    id_column: str = "site_id",
    schema_version: str = "1.1.0",
    primary_column: str | None = None,
) -> Path:
    """Write a GeoPromptFrame to GeoParquet with enriched metadata.

    Embeds ``geo`` metadata compliant with the GeoParquet specification,
    including CRS, geometry types, bounding box, and encoding information.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required to write GeoParquet. Install extras: pip install geoprompt[io,compare]"
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection = frame_to_geojson(frame, geometry=geometry, id_column=id_column)
    gdf = gpd.GeoDataFrame.from_features(collection["features"], crs=frame.crs)

    geom_col = primary_column or gdf.geometry.name
    geom_types = sorted({str(g.geom_type) for g in gdf.geometry if g is not None})
    bbox = list(gdf.total_bounds) if len(gdf) > 0 else None
    crs_json = gdf.crs.to_json_dict() if gdf.crs is not None else None

    geo_metadata = {
        "version": schema_version,
        "primary_column": geom_col,
        "columns": {
            geom_col: {
                "encoding": "WKB",
                "geometry_types": geom_types,
                "crs": crs_json,
                "bbox": bbox,
            }
        },
    }

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        table = pa.Table.from_pandas(gdf)
        existing_meta = table.schema.metadata or {}
        existing_meta[b"geo"] = json.dumps(geo_metadata).encode("utf-8")
        table = table.replace_schema_metadata(existing_meta)
        pq.write_table(table, str(output_path))
    except ImportError:
        gdf.to_parquet(output_path)

    return output_path


def read_geoparquet_metadata(path: str | Path) -> dict[str, Any]:
    """Read GeoParquet ``geo`` metadata without loading the full dataset."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"input path does not exist: {data_path}")

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(data_path))
        schema_meta = pf.schema_arrow.metadata or {}
        geo_bytes = schema_meta.get(b"geo")
        if geo_bytes is not None:
            return json.loads(geo_bytes.decode("utf-8"))
        return {}
    except ImportError:
        try:
            import geopandas as gpd

            gdf = gpd.read_parquet(data_path)
            return {
                "primary_column": gdf.geometry.name,
                "crs": str(gdf.crs) if gdf.crs is not None else None,
                "feature_count": len(gdf),
            }
        except ImportError as exc:
            raise ImportError(
                "pyarrow or geopandas is required to read GeoParquet metadata"
            ) from exc


__all__ = [
    "discover_layers",
    "frame_to_geojson",
    "get_workload_preset",
    "iter_data",
    "iter_data_with_preset",
    "iter_csv_points",
    "read_csv_points",
    "read_data",
    "read_data_with_preset",
    "read_features",
    "read_geojson",
    "read_geoparquet_metadata",
    "read_points",
    "read_table",
    "WORKLOAD_PRESETS",
    "write_data",
    "write_geojson",
    "write_geoparquet",
    "write_json",
]