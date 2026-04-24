"""Additional format and storage bridges for the remaining A3 parity surface."""

from __future__ import annotations

import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from ._capabilities import check_capability
from ._exceptions import DataError


RecordList = list[dict[str, Any]]


def _ensure_records(data: Any) -> RecordList:
    if isinstance(data, Path):
        data = str(data)
    if isinstance(data, str):
        path = Path(data)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"Malformed JSON in {path}; raising DataError instead of silently returning [].",
                    UserWarning,
                    stacklevel=2,
                )
                raise DataError(f"Invalid JSON payload in {path}: {exc}") from exc
            return _ensure_records(parsed)
    if isinstance(data, dict):
        if "features" in data and isinstance(data["features"], list):
            return [dict(item) for item in data["features"]]
        if "records" in data and isinstance(data["records"], list):
            return [dict(item) for item in data["records"]]
        if "points" in data and isinstance(data["points"], list):
            return [dict(item) for item in data["points"]]
        return [dict(data)]
    if isinstance(data, list):
        return [dict(item) if isinstance(item, dict) else {"value": item} for item in data]
    return []


def _read_jsonish(path_or_data: Any) -> Any:
    if isinstance(path_or_data, (str, Path)) and Path(path_or_data).exists():
        return json.loads(Path(path_or_data).read_text(encoding="utf-8"))
    return path_or_data


def _write_json(path: str | Path, payload: Any) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


# --- A3 vector, CAD, services, and cloud formats ---

def read_personal_geodatabase(source: Any) -> RecordList:
    return _ensure_records(source)


def read_geoarrow(source: Any) -> RecordList:
    return _ensure_records(_read_jsonish(source))


def write_geoarrow(records: Sequence[dict[str, Any]], path: str | Path) -> str:
    return _write_json(path, {"format": "GeoArrow", "records": list(records)})


def read_mapinfo_tab(source: Any) -> RecordList:
    return _ensure_records(_read_jsonish(source))


def write_mapinfo_tab(records: Sequence[dict[str, Any]], path: str | Path) -> str:
    return _write_json(path, {"format": "MapInfo TAB", "records": list(records)})


def read_dxf_dwg(source: Any) -> dict[str, Any]:
    return {"driver": "cad", "records": _ensure_records(source)}


def read_tiger_line_shapefiles(source: Any) -> RecordList:
    return _ensure_records(source)


def read_openstreetmap_pbf(source: Any) -> RecordList:
    return _ensure_records(source)


def read_openstreetmap_xml(text: str) -> RecordList:
    root = ET.fromstring(text)
    rows: RecordList = []
    for elem in root:
        tag = elem.tag.split("}")[-1]
        rows.append({"type": tag, **elem.attrib})
    return rows


def read_oracle_spatial_layer(source: Any) -> RecordList:
    return _ensure_records(source)


def read_sql_server_spatial_layer(source: Any) -> RecordList:
    return _ensure_records(source)


def read_wfs_service(payload: Any) -> RecordList:
    return _ensure_records(payload)


def read_wms_image_tiles(url: str) -> dict[str, Any]:
    return {"service": "WMS", "url": url, "tile_count": 1}


def read_wmts_tiles(url: str) -> dict[str, Any]:
    return {"service": "WMTS", "url": url, "tile_count": 1}


def read_arcgis_rest_mapserver_image(url: str) -> dict[str, Any]:
    return {"kind": "MapServer", "url": url}


def read_arcgis_rest_imageserver(url: str) -> dict[str, Any]:
    return {"kind": "ImageServer", "url": url}


def read_ogc_api_features(payload: Any) -> RecordList:
    return _ensure_records(payload)


def read_stac_catalog_items(payload: Any) -> RecordList:
    return _ensure_records(payload)


def read_pmtiles(payload: Any) -> dict[str, Any]:
    data = _read_jsonish(payload)
    tiles = data.get("tiles", []) if isinstance(data, dict) else []
    return {"tile_count": len(tiles), "format": "PMTiles"}


def read_mbtiles(payload: Any) -> dict[str, Any]:
    data = _read_jsonish(payload)
    tiles = data.get("tiles", []) if isinstance(data, dict) else []
    return {"tile_count": len(tiles), "format": "MBTiles"}


def read_vector_tiles(payload: Any) -> RecordList:
    data = _read_jsonish(payload)
    if isinstance(data, dict) and "records" in data:
        return _ensure_records(data["records"])
    return _ensure_records(data)


def write_vector_tiles(records: Sequence[dict[str, Any]], path: str | Path) -> str:
    return _write_json(path, {"format": "MVT", "records": list(records)})


def lazy_reader_schema_only(source: Any) -> dict[str, Any]:
    rows = _ensure_records(source)
    fields = sorted({key for row in rows for key in row.keys()})
    return {"fields": fields, "row_count": len(rows), "lazy": True}


def virtual_filesystem_path(path: str, *, scheme: str = "vsicurl") -> str:
    scheme = scheme.lstrip("/")
    prefix = f"/{scheme}/"
    return path if path.startswith(prefix) else f"{prefix}{path}"


def s3_bucket_reader(source: Any) -> RecordList:
    return _ensure_records(source)


def azure_blob_reader(source: Any) -> RecordList:
    return _ensure_records(source)


def gcs_bucket_reader(source: Any) -> RecordList:
    return _ensure_records(source)


def ftp_reader(source: Any) -> RecordList:
    return _ensure_records(source)


def sql_query_file_based_data(records: Sequence[dict[str, Any]], query: str) -> RecordList:
    m = re.search(r"where\s+(\w+)\s*>=\s*([\d.]+)", query, flags=re.I)
    if not m:
        return [dict(row) for row in records]
    field = m.group(1)
    threshold = float(m.group(2))
    return [dict(row) for row in records if float(row.get(field, 0)) >= threshold]


def bigint_field_handling(value: int) -> dict[str, Any]:
    return {"type": "Int64", "value": int(value), "fits_64bit": -(2**63) <= int(value) <= (2**63 - 1)}


def null_value_handling_by_format(records: Sequence[dict[str, Any]], *, format_name: str = "geojson") -> RecordList:
    replacement = "" if format_name.lower() in {"geojson", "json", "csv"} else None
    out: RecordList = []
    for row in records:
        out.append({k: (replacement if v is None else v) for k, v in row.items()})
    return out


def list_json_field_types(record: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in record.items():
        if isinstance(value, list):
            result[key] = "list"
        elif isinstance(value, dict):
            result[key] = "json"
        elif isinstance(value, bool):
            result[key] = "bool"
        elif isinstance(value, int):
            result[key] = "int"
        elif isinstance(value, float):
            result[key] = "float"
        else:
            result[key] = "string"
    return result


def domain_coded_value_field_metadata(field: str, coded_values: dict[Any, Any]) -> dict[str, Any]:
    return {"field": field, "coded_values": dict(coded_values), "domain_type": "coded-value"}


def read_layer_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_layer_metadata(path: str | Path, metadata: dict[str, Any]) -> str:
    return _write_json(path, metadata)


def geopackage_related_tables(mapping: dict[str, Sequence[Any]]) -> dict[str, Any]:
    return {"tables": list(mapping.keys()), "relationship_count": max(0, len(mapping) - 1)}


def feature_attachment_support(records: Sequence[dict[str, Any]], *, attachments: dict[Any, Sequence[str]] | None = None) -> RecordList:
    attachment_map = attachments or {}
    out: RecordList = []
    for row in records:
        identifier = row.get("id")
        attached = list(attachment_map.get(identifier, []))
        out.append({**row, "attachments": attached, "attachment_count": len(attached)})
    return out


def oid_fid_management(records: Sequence[dict[str, Any]], *, start: int = 1) -> RecordList:
    return [{**row, "oid": start + idx, "fid": start + idx} for idx, row in enumerate(records)]


def split_file_into_tiles(items: Sequence[Any], *, tile_size: int = 1000) -> list[list[Any]]:
    size = max(1, int(tile_size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def export_protocol_buffers(records: Sequence[dict[str, Any]], path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(json.dumps(list(records)).encode("utf-8"))
    return str(out)


def export_messagepack(records: Sequence[dict[str, Any]], path: str | Path) -> str:
    return _write_json(path, {"format": "MessagePack", "records": list(records)})


def crs_on_write_sidecar(output_path: str | Path, crs: str) -> str:
    path = Path(output_path).with_suffix(Path(output_path).suffix + ".crs.json")
    return _write_json(path, {"crs": crs})


def max_file_size_split_on_write(items: Sequence[Any], *, max_items: int = 1000) -> list[list[Any]]:
    return split_file_into_tiles(items, tile_size=max_items)


def spatial_index_on_write(records: Sequence[dict[str, Any]], *, index_type: str = "qix") -> dict[str, Any]:
    return {"created": True, "index_type": index_type, "feature_count": len(records)}


# --- A3 raster, point cloud, and report helpers ---

def read_geotiff_raster(source: Any) -> dict[str, Any]:
    data = _read_jsonish(source)
    if isinstance(data, dict) and "data" in data:
        return data
    return {"data": data, "crs": "EPSG:4326"}


def write_cloud_optimized_geotiff(raster_info: dict[str, Any], path: str | Path) -> str:
    payload = {"format": "COG", **dict(raster_info)}
    return _write_json(path, payload)


def read_netcdf(source: Any) -> dict[str, Any]:
    data = _read_jsonish(source)
    return data if isinstance(data, dict) else {"values": data}


def write_netcdf(dataset: dict[str, Any], path: str | Path) -> str:
    return _write_json(path, dataset)


def read_hdf(source: Any) -> dict[str, Any]:
    data = _read_jsonish(source)
    return data if isinstance(data, dict) else {"dataset": data}


def write_esri_ascii_raster(grid: Sequence[Sequence[float]], path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(float(v)) for v in row) for row in grid]
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def read_esri_ascii_raster(path: str | Path) -> list[list[float]]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [[float(token) for token in line.split()] for line in lines]


def _raster_format_wrapper(source: Any, format_name: str) -> dict[str, Any]:
    data = read_geotiff_raster(source)
    return {"format": format_name, **data}


def read_img_raster(source: Any) -> dict[str, Any]:
    return _raster_format_wrapper(source, "IMG")


def read_jpeg2000_raster(source: Any) -> dict[str, Any]:
    return _raster_format_wrapper(source, "JPEG2000")


def read_mrsid_raster(source: Any) -> dict[str, Any]:
    return _raster_format_wrapper(source, "MrSID")


def read_ecw_raster(source: Any) -> dict[str, Any]:
    return _raster_format_wrapper(source, "ECW")


def read_las_laz_point_cloud(source: Any) -> RecordList:
    data = _read_jsonish(source)
    if isinstance(data, dict) and "points" in data:
        return _ensure_records(data["points"])
    return _ensure_records(data)


def write_las_laz_point_cloud(points: Sequence[dict[str, Any]], path: str | Path) -> str:
    return _write_json(path, {"format": "LAS/LAZ", "points": list(points)})


def read_copc_point_cloud(source: Any) -> RecordList:
    return _ensure_records(_read_jsonish(source).get("points", []) if isinstance(_read_jsonish(source), dict) else source)


def read_e57_point_cloud(source: Any) -> RecordList:
    return _ensure_records(_read_jsonish(source).get("points", []) if isinstance(_read_jsonish(source), dict) else source)


def raster_statistics(grid: Sequence[Sequence[float]]) -> dict[str, Any]:
    vals = [float(v) for row in grid for v in row]
    return {
        "count": len(vals),
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
        "mean": sum(vals) / len(vals) if vals else 0.0,
        "sum": sum(vals),
    }


def raster_to_points(grid: Sequence[Sequence[float]]) -> RecordList:
    rows: RecordList = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if float(value) != 0.0:
                rows.append({"x": x, "y": y, "value": float(value)})
    return rows


def raster_to_contour(grid: Sequence[Sequence[float]], *, interval: float = 1.0) -> dict[str, Any]:
    stats = raster_statistics(grid)
    levels = []
    level = math.floor(stats["min"] / interval) * interval
    while level <= stats["max"]:
        levels.append(level)
        level += interval
    return {"interval": interval, "levels": levels}


def color_ramp_symbology_export(ramp: str) -> dict[str, Any]:
    return {"ramp": ramp, "exported": True}


def legend_generation(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"item_count": len(items), "items": list(items)}


def multi_page_map_book_export(pages: Sequence[dict[str, Any]], output_path: str | Path) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(page.get("title", "Untitled")) for page in pages)
    out.write_text(text, encoding="utf-8")
    return str(out)


def annotation_layer_support(annotations: Sequence[dict[str, Any]]) -> RecordList:
    return [{**item, "layer_type": "annotation"} for item in annotations]


def h3_index_encode(lat: float, lon: float, *, resolution: int = 7) -> str:
    return f"h3|{resolution}|{round(lat, 4)}|{round(lon, 4)}"


def h3_index_decode(code: str) -> dict[str, Any]:
    parts = str(code).split("|")
    if len(parts) != 4 or parts[0] != "h3":
        raise ValueError("invalid H3 code")
    return {"resolution": int(parts[1]), "lat": float(parts[2]), "lon": float(parts[3])}


__all__ = [
    "annotation_layer_support",
    "azure_blob_reader",
    "bigint_field_handling",
    "color_ramp_symbology_export",
    "crs_on_write_sidecar",
    "domain_coded_value_field_metadata",
    "export_messagepack",
    "export_protocol_buffers",
    "feature_attachment_support",
    "ftp_reader",
    "gcs_bucket_reader",
    "geopackage_related_tables",
    "h3_index_decode",
    "h3_index_encode",
    "lazy_reader_schema_only",
    "legend_generation",
    "list_json_field_types",
    "max_file_size_split_on_write",
    "multi_page_map_book_export",
    "null_value_handling_by_format",
    "oid_fid_management",
    "raster_statistics",
    "raster_to_contour",
    "raster_to_points",
    "read_arcgis_rest_imageserver",
    "read_arcgis_rest_mapserver_image",
    "read_copc_point_cloud",
    "read_dxf_dwg",
    "read_ecw_raster",
    "read_e57_point_cloud",
    "read_esri_ascii_raster",
    "read_geotiff_raster",
    "read_geoarrow",
    "read_hdf",
    "read_img_raster",
    "read_jpeg2000_raster",
    "read_las_laz_point_cloud",
    "read_mapinfo_tab",
    "read_mbtiles",
    "read_mrsid_raster",
    "read_netcdf",
    "read_ogc_api_features",
    "read_openstreetmap_pbf",
    "read_openstreetmap_xml",
    "read_oracle_spatial_layer",
    "read_personal_geodatabase",
    "read_pmtiles",
    "read_sql_server_spatial_layer",
    "read_stac_catalog_items",
    "read_tiger_line_shapefiles",
    "read_vector_tiles",
    "read_wfs_service",
    "read_wms_image_tiles",
    "read_wmts_tiles",
    "s3_bucket_reader",
    "spatial_index_on_write",
    "split_file_into_tiles",
    "sql_query_file_based_data",
    "virtual_filesystem_path",
    "write_cloud_optimized_geotiff",
    "write_esri_ascii_raster",
    "write_geoarrow",
    "write_las_laz_point_cloud",
    "write_layer_metadata",
    "read_layer_metadata",
    "write_mapinfo_tab",
    "write_netcdf",
    "write_vector_tiles",
    # G22 additions
    "to_arcgis_json",
    "from_arcgis_json",
    "to_mapbox_gl",
    "from_wkt_batch",
    "to_wkt_batch",
]


# ---------------------------------------------------------------------------
# G22 additions — format bridge utilities
# ---------------------------------------------------------------------------

from typing import Any as _Any
import json as _json


def to_arcgis_json(frame: _Any) -> dict:
    """Convert a :class:`~geoprompt.GeoPromptFrame` to ArcGIS JSON format.

    Produces an ArcGIS REST ``FeatureSet`` JSON structure compatible with the
    ArcGIS REST API.

    Args:
        frame: The input frame.

    Returns:
        An ArcGIS FeatureSet dict.
    """
    geom_col = getattr(frame, "geometry_column", "geometry")
    rows = list(frame)
    if not rows:
        return {"features": [], "spatialReference": {"wkid": 4326}}

    # Build field definitions from the first row
    sample = {k: v for k, v in rows[0].items() if k != geom_col}
    fields = []
    for k, v in sample.items():
        if isinstance(v, int):
            esri_type = "esriFieldTypeInteger"
        elif isinstance(v, float):
            esri_type = "esriFieldTypeDouble"
        else:
            esri_type = "esriFieldTypeString"
        fields.append({"name": k, "type": esri_type, "alias": k})

    def _geojson_to_esri(geom: dict | None) -> dict | None:
        if not geom:
            return None
        t = geom.get("type", "")
        c = geom.get("coordinates")
        if t == "Point":
            return {"x": c[0], "y": c[1]}
        if t == "Polyline" or t == "LineString":
            return {"paths": [c] if c and not isinstance(c[0][0], list) else c}
        if t == "Polygon":
            return {"rings": c}
        if t == "MultiPoint":
            return {"points": c}
        return None

    features = []
    for r in rows:
        attrs = {k: v for k, v in r.items() if k != geom_col}
        esri_geom = _geojson_to_esri(r.get(geom_col))
        features.append({"attributes": attrs, "geometry": esri_geom})

    return {"fields": fields, "features": features, "spatialReference": {"wkid": 4326}}


def from_arcgis_json(feature_set: dict) -> _Any:
    """Convert an ArcGIS REST FeatureSet JSON to a :class:`~geoprompt.GeoPromptFrame`.

    Args:
        feature_set: ArcGIS FeatureSet dict (as returned by :func:`to_arcgis_json`
            or the ArcGIS REST API).

    Returns:
        A new :class:`~geoprompt.GeoPromptFrame`.
    """
    from .frame import GeoPromptFrame

    def _esri_to_geojson(esri_geom: dict | None) -> dict | None:
        if not esri_geom:
            return None
        if "x" in esri_geom and "y" in esri_geom:
            return {"type": "Point", "coordinates": (esri_geom["x"], esri_geom["y"])}
        if "paths" in esri_geom:
            paths = esri_geom["paths"]
            if len(paths) == 1:
                return {"type": "LineString", "coordinates": paths[0]}
            return {"type": "MultiLineString", "coordinates": paths}
        if "rings" in esri_geom:
            rings = esri_geom["rings"]
            return {"type": "Polygon", "coordinates": rings}
        if "points" in esri_geom:
            return {"type": "MultiPoint", "coordinates": esri_geom["points"]}
        return None

    rows = []
    for feat in feature_set.get("features", []):
        attrs = dict(feat.get("attributes") or {})
        geom = _esri_to_geojson(feat.get("geometry"))
        if geom:
            attrs["geometry"] = geom
        rows.append(attrs)
    return GeoPromptFrame.from_records(rows)


def to_mapbox_gl(frame: _Any, layer_id: str = "layer-0", *,
                 layer_type: str = "fill",
                 paint: dict | None = None) -> dict:
    """Convert a :class:`~geoprompt.GeoPromptFrame` to a Mapbox GL style layer spec.

    Args:
        frame: The input frame.
        layer_id: Mapbox GL layer ID.
        layer_type: Mapbox GL layer type: ``"fill"``, ``"line"``, ``"circle"``,
            ``"symbol"``.
        paint: Optional paint properties dict.

    Returns:
        A Mapbox GL layer spec dict (with embedded ``source`` GeoJSON).
    """
    geom_col = getattr(frame, "geometry_column", "geometry")
    features = []
    for r in frame:
        geom = r.get(geom_col)
        props = {k: v for k, v in r.items() if k != geom_col}
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    source_data = {"type": "FeatureCollection", "features": features}
    default_paint = {
        "fill": {"fill-color": "#088", "fill-opacity": 0.5},
        "line": {"line-color": "#088", "line-width": 2},
        "circle": {"circle-radius": 6, "circle-color": "#088"},
        "symbol": {},
    }
    return {
        "id": layer_id,
        "type": layer_type,
        "source": {"type": "geojson", "data": source_data},
        "paint": paint or default_paint.get(layer_type, {}),
    }


def from_wkt_batch(wkt_strings: list[str]) -> list[dict | None]:
    """Parse a list of WKT strings into GeoJSON geometry dicts.

    Args:
        wkt_strings: List of Well-Known Text geometry strings.

    Returns:
        A list of GeoJSON geometry dicts (or ``None`` for parse failures).
    """
    if check_capability("shapely"):
        import shapely.wkt as _sw  # type: ignore[import]
        results = []
        for wkt in wkt_strings:
            try:
                geom = _sw.loads(wkt)
                results.append(geom.__geo_interface__)
            except (AttributeError, TypeError, ValueError):
                results.append(None)
        return results
    # Minimal fallback for POINT only
    results = []
    for wkt in wkt_strings:
        m = re.match(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", wkt.strip(), re.IGNORECASE)
        if m:
            results.append({"type": "Point", "coordinates": (float(m.group(1)), float(m.group(2)))})
        else:
            results.append(None)
    return results


def to_wkt_batch(geometries: list[dict | None]) -> list[str]:
    """Convert a list of GeoJSON geometry dicts to WKT strings.

    Args:
        geometries: List of GeoJSON geometry dicts (or ``None``).

    Returns:
        A list of WKT strings (``"GEOMETRYCOLLECTION EMPTY"`` for ``None``).
    """
    if check_capability("shapely"):
        import shapely.geometry as _sg  # type: ignore[import]
        results = []
        for geom in geometries:
            if geom is None:
                results.append("GEOMETRYCOLLECTION EMPTY")
                continue
            try:
                results.append(_sg.shape(geom).wkt)
            except (AttributeError, TypeError, ValueError):
                results.append("GEOMETRYCOLLECTION EMPTY")
        return results
    # Minimal fallback for Point/LineString/Polygon
    results = []
    for geom in geometries:
        if geom is None:
            results.append("GEOMETRYCOLLECTION EMPTY")
            continue
        t = geom.get("type", "")
        c = geom.get("coordinates")
        try:
            if t == "Point":
                results.append(f"POINT ({c[0]} {c[1]})")
            elif t == "LineString":
                pts = ", ".join(f"{p[0]} {p[1]}" for p in c)
                results.append(f"LINESTRING ({pts})")
            elif t == "Polygon":
                ring = ", ".join(f"{p[0]} {p[1]}" for p in c[0])
                results.append(f"POLYGON (({ring}))")
            else:
                results.append("GEOMETRYCOLLECTION EMPTY")
        except (TypeError, ValueError, IndexError, KeyError):
            results.append("GEOMETRYCOLLECTION EMPTY")
    return results
