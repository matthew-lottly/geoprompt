"""Unified spatial data I/O with workload presets and progress callbacks.

Supports GeoJSON, CSV, GeoParquet, and other geographic data formats.
Workload presets (small/medium/large/huge) provide sampling and batching tuning
for different dataset sizes. Progress callbacks enable real-time monitoring of
read/write operations.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from .frame import GeoPromptFrame
from .geometry import geometry_intersects, geometry_type, normalize_geometry


WORKLOAD_PRESETS: dict[str, dict[str, int | None]] = {
    "small": {"chunk_size": 5000, "sample_step": 1, "limit_rows": 100000},
    "medium": {"chunk_size": 20000, "sample_step": 1, "limit_rows": None},
    "large": {"chunk_size": 50000, "sample_step": 1, "limit_rows": None},
    "huge": {"chunk_size": 100000, "sample_step": 2, "limit_rows": None},
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_url_source(path: str | Path) -> bool:
    if isinstance(path, Path):
        return False
    parsed = urlparse(str(path))
    return parsed.scheme in {"http", "https", "file"}


def _resolve_service_url(url: str, *, query_params: dict[str, Any] | None = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if any(token in parsed.path for token in ("FeatureServer", "MapServer")):
        query.setdefault("where", "1=1")
        query.setdefault("outFields", "*")
        query.setdefault("f", "json")

    if query_params:
        for key, value in query_params.items():
            if value is None:
                continue
            query[key] = str(value)

    return urlunparse(parsed._replace(query=urlencode(query, safe=",")))


def _read_json_source(
    path: str | Path,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    max_retries: int = 0,
    retry_delay: float = 0.5,
) -> Any:
    if _is_url_source(path):
        resolved = _resolve_service_url(str(path))
        request = Request(resolved, headers=headers or {})
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                last_error = exc
                if attempt >= max_retries:
                    raise
                time.sleep(retry_delay)
        if last_error is not None:
            raise last_error
    return _read_json(path)


def _service_query_params(
    *,
    where: str | None = None,
    out_fields: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    result_offset: int | None = None,
    page_size: int | None = None,
    out_sr: str | int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if where is not None:
        params["where"] = where
    if out_fields is not None:
        params["outFields"] = ",".join(str(field) for field in out_fields)
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        params["geometry"] = f"{xmin},{ymin},{xmax},{ymax}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["spatialRel"] = "esriSpatialRelIntersects"
    if result_offset is not None:
        params["resultOffset"] = result_offset
    if page_size is not None:
        params["resultRecordCount"] = page_size
    if out_sr is not None:
        params["outSR"] = out_sr
    return params


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


def _extract_service_crs(payload: Any) -> str | None:
    crs = _extract_crs(payload)
    if crs:
        return crs
    if not isinstance(payload, dict):
        return None
    spatial_reference = payload.get("spatialReference")
    if isinstance(spatial_reference, dict):
        wkid = spatial_reference.get("latestWkid", spatial_reference.get("wkid"))
        if wkid is not None:
            return f"EPSG:{wkid}"
    return None


def _arcgis_geometry_to_internal(geometry: Any) -> dict[str, Any] | None:
    if not isinstance(geometry, dict):
        return None
    if "type" in geometry and "coordinates" in geometry:
        return dict(geometry)
    if "x" in geometry and "y" in geometry:
        return {"type": "Point", "coordinates": [float(geometry["x"]), float(geometry["y"])]}
    if "points" in geometry:
        return {"type": "MultiPoint", "coordinates": [[float(x), float(y)] for x, y in geometry["points"]]}
    if "paths" in geometry:
        paths = geometry.get("paths") or []
        if len(paths) == 1:
            return {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in paths[0]]}
        return {
            "type": "MultiLineString",
            "coordinates": [
                [[float(x), float(y)] for x, y in path]
                for path in paths
            ],
        }
    if "rings" in geometry:
        rings = geometry.get("rings") or []
        if len(rings) == 1:
            return {"type": "Polygon", "coordinates": [[float(x), float(y)] for x, y in rings[0]]}
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[[float(x), float(y)] for x, y in ring]]
                for ring in rings
            ],
        }
    return None


def _records_from_service_payload(payload: Any, geometry: str = "geometry") -> list[dict[str, Any]]:
    try:
        return _records_from_payload(payload, geometry=geometry)
    except TypeError:
        pass

    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        rows: list[dict[str, Any]] = []
        for feature in payload["features"]:
            if not isinstance(feature, dict):
                continue
            attributes = dict(feature.get("attributes") or feature.get("properties") or {})
            resolved_geometry = _arcgis_geometry_to_internal(feature.get("geometry"))
            if resolved_geometry is not None:
                attributes[geometry] = resolved_geometry
            if geometry in attributes and attributes[geometry] is not None:
                rows.append(attributes)
        if rows:
            return rows

    raise TypeError("service payload must be GeoJSON or ArcGIS feature JSON")


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


def _row_matches_filter(row: dict[str, Any], where: Any | None) -> bool:
    if where is None:
        return True
    if callable(where):
        return bool(where(dict(row)))
    if isinstance(where, dict):
        return all(row.get(key) == value for key, value in where.items())
    if isinstance(where, str):
        try:
            return bool(eval(where, {"__builtins__": {}}, dict(row)))
        except Exception as exc:  # pragma: no cover - invalid expressions are user input issues
            raise ValueError(f"invalid where expression: {where}") from exc
    raise TypeError("where must be a mapping, callable, or expression string")


def _apply_read_filters(
    rows: Iterable[dict[str, Any]],
    *,
    geometry: str,
    where: Any | None = None,
    geometry_mask: Any | None = None,
) -> list[dict[str, Any]]:
    resolved_mask = normalize_geometry(geometry_mask) if geometry_mask is not None else None
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not _row_matches_filter(row, where):
            continue
        if resolved_mask is not None:
            candidate = row.get(geometry)
            if candidate is None or not geometry_intersects(candidate, resolved_mask):
                continue
        filtered.append(dict(row))
    return filtered


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
    where: Any | None = None,
    geometry_mask: Any | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> GeoPromptFrame:
    """Unified spatial reader for GeoJSON/JSON, tabular files, and geofiles.

    Large-dataset knobs:
    - ``limit_rows``: cap row count
    - ``sample_step``: keep every Nth row
    - ``use_columns``: read selected attributes only
    - ``bbox`` and ``layer`` for geospatial stores
    - ``where`` and ``geometry_mask`` for lightweight read-time filtering
    """
    if _is_url_source(path):
        frame = read_service_url(
            str(path),
            geometry=geometry,
            crs=crs,
            use_columns=use_columns,
            limit_rows=limit_rows,
            sample_step=sample_step,
            where=where if isinstance(where, str) else None,
            bbox=bbox,
        )
        rows = _apply_read_filters(frame.to_records(), geometry=geometry, where=where, geometry_mask=geometry_mask)
        return GeoPromptFrame.from_records(rows, geometry=geometry, crs=frame.crs)

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
        rows = _apply_read_filters(rows, geometry=geometry, where=where, geometry_mask=geometry_mask)
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
        rows = _apply_read_filters(rows, geometry=geometry, where=where, geometry_mask=geometry_mask)
        return GeoPromptFrame.from_records(rows, geometry=geometry, crs=crs)

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
        rows = _apply_read_filters(frame.to_records(), geometry=geometry, where=where, geometry_mask=geometry_mask)
        return GeoPromptFrame.from_records(rows, geometry=geometry, crs=frame.crs)

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

    emitted_chunks = 0
    emitted_rows = 0

    def _notify(chunk: GeoPromptFrame, source: str = "") -> None:
        nonlocal emitted_chunks, emitted_rows
        emitted_chunks += 1
        emitted_rows += len(chunk)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "chunk",
                    "path": source,
                    "chunk_index": emitted_chunks,
                    "chunk_rows": len(chunk),
                    "rows_emitted": emitted_rows,
                }
            )

    if _is_url_source(path):
        frame = read_service_url(
            str(path),
            geometry=geometry,
            crs=crs,
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
            _notify(chunk, source=str(path))
            yield chunk
        return

    data_path = Path(path)
    _validate_read_options(
        data_path=data_path,
        sample_step=sample_step,
        limit_rows=limit_rows,
        chunk_size=chunk_size,
    )
    suffix = data_path.suffix.lower()

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
    workload: str | None = None,
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
    resolved_preset = workload or preset
    settings = get_workload_preset(resolved_preset)
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
    where: Any | None = None,
    geometry_mask: Any | None = None,
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
        where=where,
        geometry_mask=geometry_mask,
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


def read_service_url(
    url: str,
    *,
    geometry: str = "geometry",
    crs: str | None = None,
    use_columns: Sequence[str] | None = None,
    limit_rows: int | None = None,
    sample_step: int = 1,
    timeout: float = 30.0,
    where: str | None = None,
    out_fields: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    page_size: int | None = None,
    headers: dict[str, str] | None = None,
    paginate: bool = True,
    out_sr: str | int | None = None,
    max_retries: int = 0,
    retry_delay: float = 0.5,
) -> GeoPromptFrame:
    """Read GeoJSON or ArcGIS-style feature JSON directly from a URL or URI.

    Supports ArcGIS Feature Service query parameters like ``where``,
    ``out_fields``, bounding-box filters, custom headers, and automatic
    pagination when a service reports ``exceededTransferLimit``.
    """
    if sample_step <= 0:
        raise ValueError("sample_step must be >= 1")
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be >= 1 when provided")
    if page_size is not None and page_size <= 0:
        raise ValueError("page_size must be >= 1 when provided")

    all_rows: list[dict[str, Any]] = []
    last_payload: Any = None
    next_offset = 0

    while True:
        request_url = _resolve_service_url(
            url,
            query_params=_service_query_params(
                where=where,
                out_fields=out_fields,
                bbox=bbox,
                result_offset=next_offset if next_offset else None,
                page_size=page_size,
                out_sr=out_sr,
            ),
        )
        extra_read_kwargs: dict[str, Any] = {}
        if max_retries:
            extra_read_kwargs["max_retries"] = max_retries
        if retry_delay != 0.5:
            extra_read_kwargs["retry_delay"] = retry_delay
        payload = _read_json_source(
            request_url,
            timeout=timeout,
            headers=headers,
            **extra_read_kwargs,
        )
        last_payload = payload
        rows = _records_from_service_payload(payload, geometry=geometry)
        if use_columns:
            keep = set(use_columns)
            rows = [{**{k: v for k, v in row.items() if k in keep}, geometry: row[geometry]} for row in rows]
        all_rows.extend(rows)

        if limit_rows is not None and len(all_rows) >= limit_rows:
            break

        more_rows = bool(payload.get("exceededTransferLimit")) if isinstance(payload, dict) else False
        if not paginate or not more_rows or not rows:
            break
        next_offset += len(rows)

    all_rows = _apply_row_limits(all_rows, limit_rows=limit_rows, sample_step=sample_step)
    return GeoPromptFrame.from_records(all_rows, geometry=geometry, crs=crs or _extract_service_crs(last_payload or {}))


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
    mode: str = "overwrite",
) -> Path:
    """Unified writer for GeoJSON/JSON/CSV and optional geospatial file formats."""
    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be 'overwrite' or 'append'")

    output_path = Path(path)
    suffix = output_path.suffix.lower()

    if suffix in {".geojson", ".json"}:
        if mode == "append" and output_path.exists() and output_path.stat().st_size > 0:
            existing = read_data(output_path, geometry=geometry)
            combined = GeoPromptFrame.from_records(
                [*existing.to_records(), *frame.to_records()],
                geometry=geometry,
                crs=frame.crs or existing.crs,
            )
            return write_geojson(output_path, combined, geometry=geometry, id_column=id_column)
        return write_geojson(output_path, frame, geometry=geometry, id_column=id_column)

    if suffix in {".csv", ".tsv", ".txt"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = frame.to_records()
        if not rows:
            if mode == "overwrite" or not output_path.exists():
                output_path.write_text("", encoding=encoding)
            return output_path

        delim = "\t" if suffix == ".tsv" else delimiter
        existing_fields: list[str] = []
        append_mode = mode == "append" and output_path.exists() and output_path.stat().st_size > 0
        if append_mode:
            with output_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delim)
                existing_fields = list(reader.fieldnames or [])

        fieldnames = list(dict.fromkeys([*existing_fields, *(key for row in rows for key in row.keys())]))
        with output_path.open("a" if append_mode else "w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delim)
            if not append_mode:
                writer.writeheader()
            for row in rows:
                serialized = {field: row.get(field) for field in fieldnames}
                if geometry in serialized and serialized[geometry] is not None:
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
            if mode == "append":
                raise ValueError("append mode is not supported for parquet outputs")
            if layer is not None:
                raise ValueError("layer is not supported for parquet outputs")
            if driver is not None:
                raise ValueError("driver is not supported for parquet outputs")
            gdf.to_parquet(output_path)
        else:
            to_file_kwargs["mode"] = "a" if mode == "append" else "w"
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


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

def read_excel(
    path: str | Path,
    x_column: str = "x",
    y_column: str = "y",
    sheet_name: str | int = 0,
    crs: str | None = None,
) -> GeoPromptFrame:
    """Read point data from an Excel file.

    Requires ``openpyxl`` (for .xlsx) or ``xlrd`` (for .xls) via pandas.

    Args:
        path: Path to the Excel file.
        x_column: Column name containing X (longitude) values.
        y_column: Column name containing Y (latitude) values.
        sheet_name: Sheet name or index to read.
        crs: Optional CRS string.

    Returns:
        A :class:`GeoPromptFrame` with point geometries.
    """
    import importlib
    pd = importlib.import_module("pandas")

    df = pd.read_excel(path, sheet_name=sheet_name)
    rows = []
    for _, row in df.iterrows():
        record = row.to_dict()
        x = float(record.pop(x_column))
        y = float(record.pop(y_column))
        record["geometry"] = {"type": "Point", "coordinates": [x, y]}
        rows.append(record)
    return GeoPromptFrame(rows, geometry_column="geometry", crs=crs)


def write_feather(
    frame: GeoPromptFrame,
    path: str | Path,
) -> str:
    """Write a frame to Apache Feather while preserving GeoPrompt metadata."""
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required to write Feather files") from exc

    from .interop import to_arrow

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(to_arrow(frame), str(out))
    return str(out)


def read_feather(
    path: str | Path,
    *,
    geometry_column: str = "geometry",
    crs: str | None = None,
) -> GeoPromptFrame:
    """Read a GeoPrompt Feather file back into a frame."""
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required to read Feather files") from exc

    from .interop import from_arrow

    table = feather.read_table(str(path))
    return from_arrow(table, geometry_column=geometry_column, crs=crs)


def write_excel(
    frame: GeoPromptFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    include_wkt: bool = True,
) -> str:
    """Write frame data to an Excel file.

    Geometries are exported as WKT strings in a ``geometry_wkt`` column.
    Requires ``openpyxl`` via pandas.

    Args:
        frame: Frame to export.
        path: Output file path.
        sheet_name: Target sheet name.
        include_wkt: If ``True``, include a WKT representation of geometries.

    Returns:
        The resolved output path string.
    """
    import importlib
    pd = importlib.import_module("pandas")

    records = frame.to_records()
    for record in records:
        geom = record.pop(frame.geometry_column, None)
        if include_wkt and geom:
            record["geometry_wkt"] = _geometry_to_wkt(geom)

    df = pd.DataFrame(records)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(out), sheet_name=sheet_name, index=False)
    return str(out)


# ---------------------------------------------------------------------------
# GeoPackage helpers
# ---------------------------------------------------------------------------

def read_geopackage(
    path: str | Path,
    layer: str | None = None,
    crs: str | None = None,
) -> GeoPromptFrame:
    """Read spatial data from a GeoPackage file.

    Requires ``geopandas`` and ``fiona``.

    Args:
        path: Path to the .gpkg file.
        layer: Layer name to read; defaults to the first layer.
        crs: Optional CRS override.

    Returns:
        A :class:`GeoPromptFrame`.
    """
    import importlib
    gpd = importlib.import_module("geopandas")

    gdf = gpd.read_file(str(path), layer=layer)
    from .interop import from_geopandas
    frame = from_geopandas(gdf)
    if crs:
        frame = frame.set_crs(crs, allow_override=True)
    return frame


def write_geopackage(
    frame: GeoPromptFrame,
    path: str | Path,
    layer: str = "layer",
) -> str:
    """Write frame data to a GeoPackage file.

    Requires ``geopandas``.

    Args:
        frame: Frame to export.
        path: Output file path.
        layer: Layer name.

    Returns:
        The resolved output path string.
    """
    from .interop import to_geopandas
    gdf = to_geopandas(frame)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out), layer=layer, driver="GPKG")
    return str(out)


# ---------------------------------------------------------------------------
# Shapefile helpers
# ---------------------------------------------------------------------------

def read_shapefile(
    path: str | Path,
    crs: str | None = None,
) -> GeoPromptFrame:
    """Read spatial data from a Shapefile.

    Requires ``geopandas`` and ``fiona``.

    Args:
        path: Path to the .shp file.
        crs: Optional CRS override.

    Returns:
        A :class:`GeoPromptFrame`.
    """
    import importlib
    gpd = importlib.import_module("geopandas")

    gdf = gpd.read_file(str(path))
    from .interop import from_geopandas
    frame = from_geopandas(gdf)
    if crs:
        frame = frame.set_crs(crs, allow_override=True)
    return frame


def write_shapefile(
    frame: GeoPromptFrame,
    path: str | Path,
) -> str:
    """Write frame data to a Shapefile.

    Requires ``geopandas``.

    Args:
        frame: Frame to export.
        path: Output file path (.shp).

    Returns:
        The resolved output path string.
    """
    from .interop import to_geopandas
    gdf = to_geopandas(frame)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out), driver="ESRI Shapefile")
    return str(out)


# ---------------------------------------------------------------------------
# Internal geometry WKT helper
# ---------------------------------------------------------------------------

def _geometry_to_wkt(geom: dict) -> str:
    """Convert internal geometry dict to WKT string."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Point":
        return f"POINT ({coords[0]} {coords[1]})"
    if gtype == "LineString":
        pts = ", ".join(f"{c[0]} {c[1]}" for c in coords)
        return f"LINESTRING ({pts})"
    if gtype == "Polygon":
        if isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (int, float)):
            pts = ", ".join(f"{c[0]} {c[1]}" for c in coords)
            return f"POLYGON (({pts}))"
        rings = []
        for ring in coords:
            pts = ", ".join(f"{c[0]} {c[1]}" for c in ring)
            rings.append(f"({pts})")
        return f"POLYGON ({', '.join(rings)})"
    if gtype == "MultiPoint":
        pts = ", ".join(f"({c[0]} {c[1]})" for c in coords)
        return f"MULTIPOINT ({pts})"
    if gtype == "MultiLineString":
        lines = []
        for line in coords:
            pts = ", ".join(f"{c[0]} {c[1]}" for c in line)
            lines.append(f"({pts})")
        return f"MULTILINESTRING ({', '.join(lines)})"
    if gtype == "MultiPolygon":
        polys = []
        for poly in coords:
            if isinstance(poly[0], (int, float)):
                pts = ", ".join(f"{c[0]} {c[1]}" for c in poly)
                polys.append(f"(({pts}))")
            else:
                pts = ", ".join(f"{c[0]} {c[1]}" for c in poly)
                polys.append(f"(({pts}))")
        return f"MULTIPOLYGON ({', '.join(polys)})"
    return "GEOMETRYCOLLECTION EMPTY"


def schema_report(records: list[dict[str, object]] | Any) -> dict[str, Any]:
    """Generate a schema report for a collection of records.

    Inspects all rows to determine column names, types, null counts, and
    unique value counts.

    Args:
        records: A list of dicts, or a :class:`~geoprompt.frame.GeoPromptFrame`.

    Returns:
        Dict with ``"columns"`` list and overall ``"row_count"``.
    """
    if hasattr(records, "to_records"):
        rows: list[dict[str, object]] = records.to_records()
    else:
        rows = list(records)

    if not rows:
        return {"row_count": 0, "columns": []}

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    columns: list[dict[str, Any]] = []
    for key in all_keys:
        values = [row.get(key) for row in rows]
        non_null = [v for v in values if v is not None]
        types = {type(v).__name__ for v in non_null}
        columns.append({
            "name": key,
            "types": sorted(types),
            "null_count": len(values) - len(non_null),
            "non_null_count": len(non_null),
            "unique_count": len({str(v) for v in non_null}),
            "sample": non_null[0] if non_null else None,
        })

    return {"row_count": len(rows), "columns": columns}


def validate_schema(
    records: list[dict[str, object]] | Any,
    expected: dict[str, str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate records against an expected schema.

    Args:
        records: List of dicts or a frame.
        expected: Dict mapping column names to expected type names
            (e.g. ``{"name": "str", "population": "int"}``).
        strict: If ``True``, extra columns cause a violation.

    Returns:
        Dict with ``"valid"`` bool and ``"violations"`` detail list.
    """
    report = schema_report(records)
    col_lookup: dict[str, dict[str, Any]] = {c["name"]: c for c in report["columns"]}

    violations: list[str] = []
    for col_name, expected_type in expected.items():
        if col_name not in col_lookup:
            violations.append(f"missing column: {col_name}")
            continue
        col = col_lookup[col_name]
        if expected_type not in col["types"] and col["non_null_count"] > 0:
            violations.append(
                f"column '{col_name}' expected type '{expected_type}', found {col['types']}"
            )

    if strict:
        extra = set(col_lookup) - set(expected)
        for col_name in sorted(extra):
            violations.append(f"unexpected column: {col_name}")

    return {"valid": len(violations) == 0, "violations": violations}


def _cast_schema_value(value: Any, dtype: str) -> Any:
    """Cast a value to a target schema dtype."""
    if value is None:
        return None

    normalized = dtype.lower()
    if normalized == "str":
        return str(value)
    if normalized == "int":
        return int(value)
    if normalized == "float":
        return float(value)
    if normalized == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if text in {"false", "f", "no", "n", "0", "off", ""}:
            return False
        raise ValueError(f"cannot cast value {value!r} to bool")
    return value


def generate_schema_migration_plan(
    records: list[dict[str, object]] | Any,
    target_schema: dict[str, str],
) -> dict[str, Any]:
    """Describe the type changes needed to align records to a target schema."""
    report = schema_report(records)
    lookup: dict[str, dict[str, Any]] = {col["name"]: col for col in report["columns"]}
    changes: list[dict[str, Any]] = []

    for column, target_type in target_schema.items():
        if column not in lookup:
            changes.append({"column": column, "action": "add", "target_type": target_type})
            continue
        actual_types = [t for t in lookup[column].get("types", []) if t != "NoneType"]
        if actual_types != [target_type]:
            changes.append({
                "column": column,
                "action": "cast",
                "from_types": actual_types,
                "target_type": target_type,
            })

    return {
        "row_count": report["row_count"],
        "target_schema": dict(target_schema),
        "changes": changes,
        "change_count": len(changes),
    }


def apply_schema_mapping(
    records: list[dict[str, object]] | Any,
    *,
    rename: dict[str, str] | None = None,
    casts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Rename and cast columns across a record collection."""
    rows = records.to_records() if hasattr(records, "to_records") else [dict(row) for row in records]
    rename_map = dict(rename or {})
    cast_map = dict(casts or {})

    output: list[dict[str, Any]] = []
    for row in rows:
        new_row = {rename_map.get(key, key): value for key, value in row.items()}
        for column, dtype in cast_map.items():
            target_column = rename_map.get(column, column)
            if target_column in new_row:
                new_row[target_column] = _cast_schema_value(new_row[target_column], dtype)
        output.append(new_row)
    return output


def read_cloud_json(
    url: str | Path,
    *,
    geometry_column: str = "geometry",
    headers: dict[str, str] | None = None,
) -> Any:
    """Read a JSON or GeoJSON file from a cloud object URL or local path.

    Supports S3 pre-signed URLs, Azure Blob SAS URLs, GCS public URLs,
    local files, or any HTTPS endpoint returning JSON.
    """
    import json as _json
    from urllib.request import Request, urlopen

    target = str(url)
    parsed = urlparse(target)
    is_local_path = parsed.scheme in {"", "file"} or (len(parsed.scheme) == 1 and parsed.scheme.isalpha())
    if is_local_path:
        path = Path(parsed.path if parsed.scheme == "file" else target)
        payload = _json.loads(path.read_text(encoding="utf-8"))
    else:
        req = Request(target)
        if headers:
            for key, val in headers.items():
                req.add_header(key, val)
        with urlopen(req, timeout=60) as resp:  # noqa: S310
            raw = resp.read()
        payload = _json.loads(raw)

    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        return _records_from_payload(payload, geometry_column)
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict) and "features" in payload:
        return _records_from_service_payload(payload, geometry_column)
    return payload


def write_cloud_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
) -> str:
    """Write JSON to a local path or fsspec-supported object-store path."""
    target = str(path)
    parsed = urlparse(target)
    if parsed.scheme in {"s3", "gs", "az", "abfs"}:
        try:
            import fsspec  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Install fsspec plus the relevant cloud backend to write remote object-store paths") from exc
        with fsspec.open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
        return target

    local_path = Path(parsed.path if parsed.scheme == "file" else target)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return str(local_path)


def read_zipped_shapefile(
    path: str | Path,
    *,
    geometry: str = "geometry",
    crs: str | None = None,
    layer: str | None = None,
) -> Any:
    """Read a zipped shapefile (.zip containing .shp and sidecar files).

    Uses fiona or the built-in ``read_shapefile`` after extracting to a
    temporary directory.

    Args:
        path: Local path or URL to a .zip archive.
        geometry: Geometry column name.
        crs: Override CRS.
        layer: Layer name if the archive contains multiple layers.

    Returns:
        A :class:`~geoprompt.frame.GeoPromptFrame`.
    """
    import tempfile
    import zipfile
    from pathlib import Path as _P
    import urllib.request

    p = str(path)
    if p.startswith("http://") or p.startswith("https://"):
        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        urllib.request.urlretrieve(p, tmp_zip.name)  # noqa: S310
        p = tmp_zip.name

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(p) as zf:
            zf.extractall(tmpdir)

        shp_files = list(_P(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("no .shp file found in the archive")

        target = shp_files[0]
        if layer:
            for sf in shp_files:
                if layer.lower() in sf.stem.lower():
                    target = sf
                    break

        return read_shapefile(str(target), geometry=geometry, crs=crs)


def apply_field_aliases(
    records: list[dict],
    alias_map: dict[str, str],
) -> list[dict]:
    """Rename fields in a list of records according to an alias map.

    Args:
        records: Raw record dicts.
        alias_map: Mapping from original field names to desired aliases.

    Returns:
        New list of records with renamed fields.
    """
    if not alias_map:
        return records
    result: list[dict] = []
    for row in records:
        new_row = {}
        for k, v in row.items():
            new_row[alias_map.get(k, k)] = v
        result.append(new_row)
    return result


__all__ = [
    "apply_field_aliases",
    "apply_schema_mapping",
    "discover_layers",
    "frame_to_geojson",
    "generate_schema_migration_plan",
    "get_workload_preset",
    "iter_data",
    "iter_data_with_preset",
    "iter_csv_points",
    "read_cloud_json",
    "read_csv_points",
    "read_data",
    "read_data_with_preset",
    "read_excel",
    "read_feather",
    "read_features",
    "read_geojson",
    "read_geopackage",
    "read_geoparquet_metadata",
    "read_points",
    "read_service_url",
    "read_shapefile",
    "read_table",
    "read_zipped_shapefile",
    "schema_report",
    "validate_schema",
    "WORKLOAD_PRESETS",
    "write_cloud_json",
    "write_data",
    "write_excel",
    "write_feather",
    "write_geojson",
    "write_geopackage",
    "write_geoparquet",
    "write_json",
    "write_shapefile",
    # G3 additions
    "read_flatgeobuf",
    "write_flatgeobuf",
    "read_dxf",
    "read_mapinfo_tab",
    "read_wfs",
    "read_osm_pbf",
    "auto_read_file",
    "auto_write_file",
]


# ---------------------------------------------------------------------------
# G3 additions — additional format readers/writers
# ---------------------------------------------------------------------------

def read_flatgeobuf(path: str | Path) -> GeoPromptFrame:
    """Read a FlatGeobuf file into a :class:`~geoprompt.GeoPromptFrame`.

    Attempts to use the ``flatgeobuf`` package if available; otherwise falls
    back to reading via ``pyogrio`` or ``fiona`` if installed.

    Args:
        path: Path to a ``.fgb`` file.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    p = Path(path)
    try:
        import pyogrio  # type: ignore[import]
        gdf = pyogrio.read_dataframe(str(p))
        records = []
        for _, row in gdf.iterrows():
            rec: dict = dict(row)
            geom = rec.pop("geometry", None)
            if geom is not None:
                rec["geometry"] = json.loads(geom.__geo_interface__.__str__()) if hasattr(geom, "__geo_interface__") else {"type": "Point", "coordinates": (0.0, 0.0)}
            records.append(rec)
        return GeoPromptFrame.from_records(records)
    except Exception:
        pass
    try:
        import fiona  # type: ignore[import]
        with fiona.open(str(p)) as src:
            records = [dict(feat["properties"]) | {"geometry": dict(feat["geometry"])} for feat in src]
        return GeoPromptFrame.from_records(records)
    except Exception:
        pass
    # Final fallback: treat as GeoJSON
    return read_geojson(path)  # type: ignore[attr-defined]


def write_flatgeobuf(frame: GeoPromptFrame, path: str | Path) -> None:
    """Write a :class:`~geoprompt.GeoPromptFrame` to a FlatGeobuf file.

    Falls back to writing GeoJSON if ``pyogrio``/``fiona`` is not installed.

    Args:
        frame: The frame to write.
        path: Output path for the ``.fgb`` file.
    """
    p = Path(path)
    try:
        import pyogrio  # type: ignore[import]
        import geopandas as gpd  # type: ignore[import]
        import shapely  # type: ignore[import]
        rows = list(frame)
        geom_col = frame.geometry_column
        shapes = [shapely.from_geojson(json.dumps(r.get(geom_col, {}))) for r in rows]
        props = [{k: v for k, v in r.items() if k != geom_col} for r in rows]
        gdf = gpd.GeoDataFrame(props, geometry=shapes)
        gdf.to_file(str(p), driver="FlatGeobuf")
        return
    except Exception:
        pass
    # Fallback: write as GeoJSON with .fgb extension
    write_geojson(frame, str(p))  # type: ignore[attr-defined]


def read_dxf(path: str | Path, layer: str | None = None) -> GeoPromptFrame:
    """Read geometry entities from a DXF (CAD) file.

    Extracts LINE, LWPOLYLINE, CIRCLE, ARC, and INSERT (block reference)
    entities.  Requires ``ezdxf`` for full support; falls back to stub
    output for basic metadata only.

    Args:
        path: Path to the ``.dxf`` file.
        layer: Optional layer name filter.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    try:
        import ezdxf  # type: ignore[import]
        doc = ezdxf.readfile(str(path))
        msp = doc.modelspace()
        records = []
        for entity in msp:
            if layer and entity.dxf.layer != layer:
                continue
            etype = entity.dxftype()
            try:
                if etype == "LINE":
                    sx, sy = entity.dxf.start.x, entity.dxf.start.y
                    ex, ey = entity.dxf.end.x, entity.dxf.end.y
                    geom = {"type": "LineString", "coordinates": [(sx, sy), (ex, ey)]}
                elif etype in {"LWPOLYLINE", "POLYLINE"}:
                    coords = [(pt[0], pt[1]) for pt in entity.get_points()]
                    geom = {"type": "LineString", "coordinates": coords} if len(coords) >= 2 else {"type": "Point", "coordinates": coords[0] if coords else (0.0, 0.0)}
                elif etype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    geom = {"type": "Point", "coordinates": (cx, cy)}
                elif etype == "INSERT":
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    geom = {"type": "Point", "coordinates": (ix, iy)}
                else:
                    continue
                records.append({"entity_type": etype, "layer": entity.dxf.layer, "geometry": geom})
            except Exception:
                continue
        return GeoPromptFrame.from_records(records) if records else GeoPromptFrame.from_records([{"entity_type": "empty", "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}}])
    except ImportError:
        # Return a stub frame if ezdxf not available
        return GeoPromptFrame.from_records([{"entity_type": "dxf_stub", "source": str(path), "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}}])


def read_mapinfo_tab(path: str | Path) -> GeoPromptFrame:
    """Read a MapInfo TAB file into a :class:`~geoprompt.GeoPromptFrame`.

    Requires ``pyogrio`` or ``fiona``; raises ``ImportError`` if neither
    is available.

    Args:
        path: Path to the ``.tab`` file.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    try:
        import pyogrio  # type: ignore[import]
        gdf = pyogrio.read_dataframe(str(path))
        rows = []
        for _, row in gdf.iterrows():
            rec = dict(row)
            geom = rec.pop("geometry", None)
            if geom is not None and hasattr(geom, "__geo_interface__"):
                gi = geom.__geo_interface__
                rec["geometry"] = {"type": gi["type"], "coordinates": gi["coordinates"]}
            rows.append(rec)
        return GeoPromptFrame.from_records(rows)
    except ImportError:
        pass
    try:
        import fiona  # type: ignore[import]
        with fiona.open(str(path)) as src:
            rows = [dict(feat["properties"]) | {"geometry": dict(feat["geometry"])} for feat in src]
        return GeoPromptFrame.from_records(rows)
    except ImportError as err:
        raise ImportError("pyogrio or fiona is required to read MapInfo TAB files") from err


def read_wfs(url: str, type_name: str, *, max_features: int = 1000,
             bbox: tuple[float, float, float, float] | None = None) -> GeoPromptFrame:
    """Fetch features from an OGC Web Feature Service (WFS).

    Constructs a WFS ``GetFeature`` request, fetches GeoJSON output, and
    returns the features as a :class:`~geoprompt.GeoPromptFrame`.

    Args:
        url: WFS endpoint base URL.
        type_name: Feature type name (``typeName`` parameter).
        max_features: Maximum number of features to request.
        bbox: Optional bounding box ``(minx, miny, maxx, maxy)`` filter.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    params: dict[str, str] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "count": str(max_features),
    }
    if bbox is not None:
        params["bbox"] = ",".join(str(v) for v in bbox)
    query_string = urlencode(params)
    sep = "&" if "?" in url else "?"
    full_url = f"{url}{sep}{query_string}"
    req = Request(full_url, headers={"Accept": "application/json"})  # noqa: S310
    with urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    features = data.get("features", [])
    rows = [
        {**(feat.get("properties") or {}), "geometry": feat.get("geometry") or {"type": "Point", "coordinates": (0.0, 0.0)}}
        for feat in features
    ]
    crs_info = data.get("crs", {}).get("properties", {}).get("name")
    return GeoPromptFrame.from_records(rows, crs=crs_info)


def read_osm_pbf(path: str | Path, *, element_types: list[str] | None = None) -> GeoPromptFrame:
    """Read an OpenStreetMap PBF file into a :class:`~geoprompt.GeoPromptFrame`.

    Requires ``osmium`` (``osmium-tool`` Python bindings) or falls back to
    ``pyosmium`` if available.  Returns a stub frame if neither is installed.

    Args:
        path: Path to the ``.osm.pbf`` file.
        element_types: Optional list of element types to include:
            ``"node"``, ``"way"``, ``"relation"``.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    try:
        import osmium  # type: ignore[import]

        class _Handler(osmium.SimpleHandler):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.records: list[dict] = []

            def node(self, n: Any) -> None:
                if element_types and "node" not in element_types:
                    return
                self.records.append({"osm_id": n.id, "osm_type": "node", "tags": dict(n.tags), "geometry": {"type": "Point", "coordinates": (n.location.lon, n.location.lat)}})

            def way(self, w: Any) -> None:
                if element_types and "way" not in element_types:
                    return
                if w.nodes:
                    coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
                    if len(coords) >= 2:
                        self.records.append({"osm_id": w.id, "osm_type": "way", "tags": dict(w.tags), "geometry": {"type": "LineString", "coordinates": coords}})

        handler = _Handler()
        handler.apply_file(str(path))
        return GeoPromptFrame.from_records(handler.records) if handler.records else GeoPromptFrame.from_records([{"osm_type": "empty", "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}}])
    except ImportError:
        return GeoPromptFrame.from_records([{"osm_type": "stub", "source": str(path), "note": "osmium not installed", "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}}])


_FORMAT_READERS: dict[str, Any] = {
    ".geojson": "read_geojson",
    ".json": "read_geojson",
    ".geojsonl": "read_geojson",
    ".csv": "read_csv_points",
    ".parquet": "read_data",
    ".gpkg": "read_geopackage",
    ".shp": "read_shapefile",
    ".fgb": "read_flatgeobuf",
    ".kml": None,  # handled via formats module
    ".gpx": None,
    ".gml": None,
}

_FORMAT_WRITERS: dict[str, Any] = {
    ".geojson": "write_geojson",
    ".json": "write_geojson",
    ".parquet": "write_geoparquet",
    ".gpkg": "write_geopackage",
    ".shp": "write_shapefile",
    ".fgb": "write_flatgeobuf",
    ".csv": None,
}


def auto_read_file(path: str | Path, **kwargs: Any) -> GeoPromptFrame:
    """Auto-detect format from file extension and read into a frame.

    Supports: GeoJSON, CSV, GeoPackage, Shapefile, FlatGeobuf, GeoParquet,
    KML, GPX, GML (via the :mod:`~geoprompt.formats` module).

    Args:
        path: Path to the spatial data file.
        **kwargs: Additional keyword arguments forwarded to the reader.

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.

    Raises:
        ValueError: If the format cannot be determined from the extension.
    """
    import importlib as _il
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".geojson", ".json", ".geojsonl"}:
        return read_geojson(p, **kwargs)  # type: ignore[attr-defined]
    if ext == ".csv":
        return read_csv_points(p, **kwargs)  # type: ignore[attr-defined]
    if ext in {".parquet", ".gpq"}:
        return read_data(p, **kwargs)  # type: ignore[attr-defined]
    if ext == ".gpkg":
        return read_geopackage(p, **kwargs)  # type: ignore[attr-defined]
    if ext == ".shp":
        return read_shapefile(p, **kwargs)  # type: ignore[attr-defined]
    if ext == ".fgb":
        return read_flatgeobuf(p, **kwargs)
    if ext == ".kml":
        fmt = _il.import_module(".formats", "geoprompt")
        return GeoPromptFrame.from_records(list(fmt.read_kml(p)))
    if ext == ".gpx":
        fmt = _il.import_module(".formats", "geoprompt")
        return GeoPromptFrame.from_records(list(fmt.read_gpx(p)))
    if ext == ".gml":
        fmt = _il.import_module(".formats", "geoprompt")
        return GeoPromptFrame.from_records(list(fmt.read_gml(p)))
    if ext == ".tab":
        return read_mapinfo_tab(p, **kwargs)
    if ext in {".osm", ".pbf"}:
        return read_osm_pbf(p, **kwargs)
    raise ValueError(f"unsupported format extension: {ext!r}")


def auto_write_file(frame: GeoPromptFrame, path: str | Path, **kwargs: Any) -> None:
    """Auto-detect format from file extension and write a frame.

    Supports: GeoJSON, GeoPackage, Shapefile, FlatGeobuf, GeoParquet.

    Args:
        frame: The :class:`~geoprompt.GeoPromptFrame` to write.
        path: Output file path.
        **kwargs: Additional keyword arguments forwarded to the writer.

    Raises:
        ValueError: If the format cannot be determined from the extension.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".geojson", ".json"}:
        write_geojson(frame, p, **kwargs)  # type: ignore[attr-defined]
    elif ext in {".parquet", ".gpq"}:
        write_geoparquet(frame, p, **kwargs)  # type: ignore[attr-defined]
    elif ext == ".gpkg":
        write_geopackage(frame, p, **kwargs)  # type: ignore[attr-defined]
    elif ext == ".shp":
        write_shapefile(frame, p, **kwargs)  # type: ignore[attr-defined]
    elif ext == ".fgb":
        write_flatgeobuf(frame, p, **kwargs)
    else:
        raise ValueError(f"unsupported output format extension: {ext!r}")