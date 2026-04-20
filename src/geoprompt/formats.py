"""Extended I/O format support: KML, GPX, TopoJSON, streaming GeoJSON, and utilities.

Pure-Python implementations that read/write common geospatial exchange formats
without heavy binary dependencies.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from zipfile import ZipFile


# ---------------------------------------------------------------------------
# KML / KMZ  (items 310, 311)
# ---------------------------------------------------------------------------

_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _parse_kml_coordinates(text: str) -> list[list[float]]:
    coords: list[list[float]] = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            coords.append([float(p) for p in parts])
    return coords


def _kml_geometry(placemark: ET.Element) -> dict[str, Any] | None:
    """Extract GeoJSON geometry from a KML Placemark element."""
    for tag in ("Point", "LineString", "Polygon", "MultiGeometry"):
        elem = placemark.find(f"kml:{tag}", _KML_NS)
        if elem is None:
            elem = placemark.find(tag)
        if elem is not None:
            return _kml_element_to_geojson(elem)
    return None


def _kml_element_to_geojson(elem: ET.Element) -> dict[str, Any] | None:
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    if tag == "Point":
        coord_el = elem.find("kml:coordinates", _KML_NS)
        if coord_el is None:
            coord_el = elem.find("coordinates")
        if coord_el is not None and coord_el.text:
            coords = _parse_kml_coordinates(coord_el.text)
            if coords:
                return {"type": "Point", "coordinates": coords[0][:2]}
    elif tag == "LineString":
        coord_el = elem.find("kml:coordinates", _KML_NS)
        if coord_el is None:
            coord_el = elem.find("coordinates")
        if coord_el is not None and coord_el.text:
            coords = _parse_kml_coordinates(coord_el.text)
            return {"type": "LineString", "coordinates": [c[:2] for c in coords]}
    elif tag == "Polygon":
        rings: list[list[list[float]]] = []
        for boundary_tag in ("outerBoundaryIs", "innerBoundaryIs"):
            boundary = elem.find(f"kml:{boundary_tag}", _KML_NS)
            if boundary is None:
                boundary = elem.find(boundary_tag)
            if boundary is not None:
                lr = boundary.find("kml:LinearRing", _KML_NS)
                if lr is None:
                    lr = boundary.find("LinearRing")
                if lr is not None:
                    coord_el = lr.find("kml:coordinates", _KML_NS)
                    if coord_el is None:
                        coord_el = lr.find("coordinates")
                    if coord_el is not None and coord_el.text:
                        coords = _parse_kml_coordinates(coord_el.text)
                        rings.append([c[:2] for c in coords])
        if rings:
            return {"type": "Polygon", "coordinates": rings}
    elif tag == "MultiGeometry":
        geometries: list[dict[str, Any]] = []
        for child in elem:
            g = _kml_element_to_geojson(child)
            if g:
                geometries.append(g)
        if geometries:
            return {"type": "GeometryCollection", "coordinates": [], "geometries": geometries}

    return None


def _extract_simple_data(placemark: ET.Element) -> dict[str, Any]:
    """Extract SimpleData / Data fields from a KML Placemark."""
    props: dict[str, Any] = {}

    # <name>
    name_el = placemark.find("kml:name", _KML_NS)
    if name_el is None:
        name_el = placemark.find("name")
    if name_el is not None and name_el.text:
        props["name"] = name_el.text

    # <description>
    desc_el = placemark.find("kml:description", _KML_NS)
    if desc_el is None:
        desc_el = placemark.find("description")
    if desc_el is not None and desc_el.text:
        props["description"] = desc_el.text

    # <ExtendedData>
    ext = placemark.find("kml:ExtendedData", _KML_NS)
    if ext is None:
        ext = placemark.find("ExtendedData")
    if ext is not None:
        for sd in ext.iter():
            tag = sd.tag.split("}")[-1] if "}" in sd.tag else sd.tag
            if tag == "SimpleData":
                field_name = sd.get("name", "")
                if field_name and sd.text:
                    props[field_name] = sd.text
            elif tag == "Data":
                field_name = sd.get("name", "")
                val_el = sd.find("kml:value", _KML_NS)
                if val_el is None:
                    val_el = sd.find("value")
                if field_name and val_el is not None and val_el.text:
                    props[field_name] = val_el.text

    return props


def read_kml(path: str | Path) -> list[dict[str, Any]]:
    """Read a KML file and return a list of GeoJSON-like feature dicts.

    Also handles KMZ (compressed KML) files transparently.
    """
    path = Path(path)
    if path.suffix.lower() == ".kmz":
        with ZipFile(path) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("no KML file found inside KMZ archive")
            kml_data = zf.read(kml_names[0])
            root = ET.fromstring(kml_data)
    else:
        tree = ET.parse(path)
        root = tree.getroot()

    features: list[dict[str, Any]] = []
    # Find all Placemarks recursively
    for pm in root.iter():
        tag = pm.tag.split("}")[-1] if "}" in pm.tag else pm.tag
        if tag != "Placemark":
            continue
        geom = _kml_geometry(pm)
        props = _extract_simple_data(pm)
        if geom:
            features.append({"type": "Feature", "geometry": geom, "properties": props})

    return features


def write_kml(features: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write features to a KML file.

    Each feature must have a 'geometry' dict and optional 'properties' dict.
    """
    path = Path(path)
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")

    for feat in features:
        pm = ET.SubElement(doc, "Placemark")
        props = feat.get("properties", {})
        if "name" in props:
            ET.SubElement(pm, "name").text = str(props["name"])
        if "description" in props:
            ET.SubElement(pm, "description").text = str(props["description"])

        # Extended data
        extra = {k: v for k, v in props.items() if k not in ("name", "description")}
        if extra:
            ext = ET.SubElement(pm, "ExtendedData")
            for k, v in extra.items():
                data = ET.SubElement(ext, "Data", name=k)
                ET.SubElement(data, "value").text = str(v)

        geom = feat.get("geometry")
        if geom:
            _write_kml_geometry(pm, geom)

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


def _coords_to_kml_text(coords: list[list[float]] | list[float]) -> str:
    """Convert coordinate list to KML coordinates text."""
    if not coords:
        return ""
    if isinstance(coords[0], (int, float)):
        return ",".join(str(c) for c in coords)
    return " ".join(",".join(str(c) for c in pt) for pt in coords)


def _write_kml_geometry(parent: ET.Element, geom: dict[str, Any]) -> None:
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Point":
        pt = ET.SubElement(parent, "Point")
        ET.SubElement(pt, "coordinates").text = _coords_to_kml_text(coords)
    elif gtype == "LineString":
        ls = ET.SubElement(parent, "LineString")
        ET.SubElement(ls, "coordinates").text = _coords_to_kml_text(coords)
    elif gtype == "Polygon":
        poly = ET.SubElement(parent, "Polygon")
        for i, ring in enumerate(coords):
            tag = "outerBoundaryIs" if i == 0 else "innerBoundaryIs"
            boundary = ET.SubElement(poly, tag)
            lr = ET.SubElement(boundary, "LinearRing")
            ET.SubElement(lr, "coordinates").text = _coords_to_kml_text(ring)
    elif gtype == "GeometryCollection":
        mg = ET.SubElement(parent, "MultiGeometry")
        for g in geom.get("geometries", []):
            _write_kml_geometry(mg, g)


# ---------------------------------------------------------------------------
# GPX  (items 325, 326)
# ---------------------------------------------------------------------------

_GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def read_gpx(path: str | Path) -> list[dict[str, Any]]:
    """Read a GPX file and return a list of GeoJSON-like feature dicts.

    Reads waypoints, tracks, and routes.
    """
    tree = ET.parse(Path(path))
    root = tree.getroot()
    features: list[dict[str, Any]] = []

    # Waypoints
    for wpt in root.iter():
        tag = wpt.tag.split("}")[-1] if "}" in wpt.tag else wpt.tag
        if tag != "wpt":
            continue
        lon = float(wpt.get("lon", "0"))
        lat = float(wpt.get("lat", "0"))
        props: dict[str, Any] = {"type": "waypoint"}
        name_el = wpt.find("gpx:name", _GPX_NS)
        if name_el is None:
            name_el = wpt.find("name")
        if name_el is not None and name_el.text:
            props["name"] = name_el.text
        ele_el = wpt.find("gpx:ele", _GPX_NS)
        if ele_el is None:
            ele_el = wpt.find("ele")
        if ele_el is not None and ele_el.text:
            props["elevation"] = float(ele_el.text)
        time_el = wpt.find("gpx:time", _GPX_NS)
        if time_el is None:
            time_el = wpt.find("time")
        if time_el is not None and time_el.text:
            props["time"] = time_el.text
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    # Tracks
    for trk in root.iter():
        tag = trk.tag.split("}")[-1] if "}" in trk.tag else trk.tag
        if tag != "trk":
            continue
        props = {"type": "track"}
        name_el = trk.find("gpx:name", _GPX_NS)
        if name_el is None:
            name_el = trk.find("name")
        if name_el is not None and name_el.text:
            props["name"] = name_el.text

        segments: list[list[list[float]]] = []
        for seg in trk.iter():
            seg_tag = seg.tag.split("}")[-1] if "}" in seg.tag else seg.tag
            if seg_tag != "trkseg":
                continue
            coords: list[list[float]] = []
            for pt in seg.iter():
                pt_tag = pt.tag.split("}")[-1] if "}" in pt.tag else pt.tag
                if pt_tag != "trkpt":
                    continue
                lon = float(pt.get("lon", "0"))
                lat = float(pt.get("lat", "0"))
                coords.append([lon, lat])
            if coords:
                segments.append(coords)

        if len(segments) == 1:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": segments[0]},
                "properties": props,
            })
        elif segments:
            features.append({
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": segments},
                "properties": props,
            })

    # Routes
    for rte in root.iter():
        tag = rte.tag.split("}")[-1] if "}" in rte.tag else rte.tag
        if tag != "rte":
            continue
        props = {"type": "route"}
        name_el = rte.find("gpx:name", _GPX_NS)
        if name_el is None:
            name_el = rte.find("name")
        if name_el is not None and name_el.text:
            props["name"] = name_el.text
        coords = []
        for pt in rte.iter():
            pt_tag = pt.tag.split("}")[-1] if "}" in pt.tag else pt.tag
            if pt_tag != "rtept":
                continue
            lon = float(pt.get("lon", "0"))
            lat = float(pt.get("lat", "0"))
            coords.append([lon, lat])
        if coords:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            })

    return features


def write_gpx(features: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write features to a GPX file.

    Points become waypoints, LineStrings become tracks.
    """
    path = Path(path)
    gpx = ET.Element("gpx", version="1.1", creator="geoprompt",
                     xmlns="http://www.topografix.com/GPX/1/1")

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        gtype = geom.get("type", "")

        if gtype == "Point":
            coords = geom["coordinates"]
            wpt = ET.SubElement(gpx, "wpt", lat=str(coords[1]), lon=str(coords[0]))
            if "name" in props:
                ET.SubElement(wpt, "name").text = str(props["name"])
            if "elevation" in props:
                ET.SubElement(wpt, "ele").text = str(props["elevation"])
        elif gtype == "LineString":
            trk = ET.SubElement(gpx, "trk")
            if "name" in props:
                ET.SubElement(trk, "name").text = str(props["name"])
            seg = ET.SubElement(trk, "trkseg")
            for coord in geom["coordinates"]:
                ET.SubElement(seg, "trkpt", lat=str(coord[1]), lon=str(coord[0]))
        elif gtype == "MultiLineString":
            trk = ET.SubElement(gpx, "trk")
            if "name" in props:
                ET.SubElement(trk, "name").text = str(props["name"])
            for line in geom["coordinates"]:
                seg = ET.SubElement(trk, "trkseg")
                for coord in line:
                    ET.SubElement(seg, "trkpt", lat=str(coord[1]), lon=str(coord[0]))

    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


# ---------------------------------------------------------------------------
# TopoJSON  (items 323, 324)
# ---------------------------------------------------------------------------

def read_topojson(path: str | Path) -> list[dict[str, Any]]:
    """Read a TopoJSON file and return GeoJSON-like features.

    Decodes arc references back to coordinates.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") != "Topology":
        raise ValueError("not a valid TopoJSON file")

    arcs = data.get("arcs", [])
    transform = data.get("transform")

    def decode_arc(arc_coords: list[list[int | float]]) -> list[list[float]]:
        if transform:
            sx, sy = transform["scale"]
            tx, ty = transform["translate"]
            decoded: list[list[float]] = []
            x, y = 0.0, 0.0
            for dx, dy in arc_coords:
                x += dx
                y += dy
                decoded.append([x * sx + tx, y * sy + ty])
            return decoded
        return [[float(c) for c in pt] for pt in arc_coords]

    decoded_arcs = [decode_arc(a) for a in arcs]

    def resolve_arc_index(idx: int) -> list[list[float]]:
        if idx >= 0:
            return decoded_arcs[idx]
        return list(reversed(decoded_arcs[~idx]))

    def resolve_arcs(arc_indices: list[int]) -> list[list[float]]:
        coords: list[list[float]] = []
        for idx in arc_indices:
            resolved = resolve_arc_index(idx)
            if coords:
                coords.extend(resolved[1:])
            else:
                coords.extend(resolved)
        return coords

    def topo_geom_to_geojson(geo: dict[str, Any]) -> dict[str, Any] | None:
        gtype = geo.get("type", "")
        arc_data = geo.get("arcs", [])

        if gtype == "Point":
            return {"type": "Point", "coordinates": geo.get("coordinates", [])}
        elif gtype == "MultiPoint":
            return {"type": "MultiPoint", "coordinates": geo.get("coordinates", [])}
        elif gtype == "LineString":
            return {"type": "LineString", "coordinates": resolve_arcs(arc_data)}
        elif gtype == "MultiLineString":
            return {"type": "MultiLineString", "coordinates": [resolve_arcs(a) for a in arc_data]}
        elif gtype == "Polygon":
            return {"type": "Polygon", "coordinates": [resolve_arcs(ring) for ring in arc_data]}
        elif gtype == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "coordinates": [[resolve_arcs(ring) for ring in polygon] for polygon in arc_data],
            }
        return None

    features: list[dict[str, Any]] = []
    for obj_name, obj in data.get("objects", {}).items():
        if obj.get("type") == "GeometryCollection":
            for geo in obj.get("geometries", []):
                geom = topo_geom_to_geojson(geo)
                props = geo.get("properties", {})
                if geom:
                    features.append({"type": "Feature", "geometry": geom, "properties": props})
        else:
            geom = topo_geom_to_geojson(obj)
            props = obj.get("properties", {})
            if geom:
                features.append({"type": "Feature", "geometry": geom, "properties": props})

    return features


def write_topojson(
    features: Sequence[dict[str, Any]],
    path: str | Path,
    object_name: str = "data",
    quantization: int | None = None,
) -> Path:
    """Write features to a TopoJSON file.

    Extracts shared arcs from geometries.  If *quantization* is given,
    coordinates are delta-encoded with a transform.
    """
    path = Path(path)
    arcs: list[list[list[float]]] = []
    arc_map: dict[str, int] = {}  # (stringified coords) → arc index

    def register_arc(coords: list[list[float]]) -> int:
        key = json.dumps(coords)
        if key in arc_map:
            return arc_map[key]
        idx = len(arcs)
        arcs.append(coords)
        arc_map[key] = idx
        return idx

    def geojson_to_topo(geom: dict[str, Any]) -> dict[str, Any]:
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Point":
            return {"type": "Point", "coordinates": coords}
        elif gtype == "MultiPoint":
            return {"type": "MultiPoint", "coordinates": coords}
        elif gtype == "LineString":
            return {"type": "LineString", "arcs": [register_arc(coords)]}
        elif gtype == "MultiLineString":
            return {"type": "MultiLineString", "arcs": [[register_arc(line)] for line in coords]}
        elif gtype == "Polygon":
            return {"type": "Polygon", "arcs": [[register_arc(ring)] for ring in coords]}
        elif gtype == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "arcs": [[[register_arc(ring)] for ring in polygon] for polygon in coords],
            }
        return {"type": gtype}

    geometries: list[dict[str, Any]] = []
    for feat in features:
        geom = feat.get("geometry")
        if geom:
            topo = geojson_to_topo(geom)
            topo["properties"] = feat.get("properties", {})
            geometries.append(topo)

    topology: dict[str, Any] = {
        "type": "Topology",
        "objects": {object_name: {"type": "GeometryCollection", "geometries": geometries}},
        "arcs": arcs,
    }

    if quantization and quantization > 0:
        # Compute bounds
        all_coords = [c for arc in arcs for c in arc]
        if all_coords:
            min_x = min(c[0] for c in all_coords)
            min_y = min(c[1] for c in all_coords)
            max_x = max(c[0] for c in all_coords)
            max_y = max(c[1] for c in all_coords)
            kx = (max_x - min_x) / max(quantization - 1, 1) or 1
            ky = (max_y - min_y) / max(quantization - 1, 1) or 1
            topology["transform"] = {
                "scale": [kx, ky],
                "translate": [min_x, min_y],
            }
            quantized_arcs: list[list[list[int]]] = []
            for arc in arcs:
                q_arc: list[list[int]] = []
                px, py = 0, 0
                for c in arc:
                    qx = round((c[0] - min_x) / kx)
                    qy = round((c[1] - min_y) / ky)
                    q_arc.append([qx - px, qy - py])
                    px, py = qx, qy
                quantized_arcs.append(q_arc)
            topology["arcs"] = quantized_arcs

    path.write_text(json.dumps(topology, separators=(",", ":")), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# GeoJSON streaming — newline-delimited  (items 1754, 1755)
# ---------------------------------------------------------------------------

def iter_geojson_features(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily iterate features from a GeoJSON or newline-delimited GeoJSON file.

    For standard GeoJSON FeatureCollections, features are yielded one at a time.
    For newline-delimited GeoJSON (.geojsonl / .geojsonseq), each line is a Feature.
    """
    path = Path(path)
    open_fn: Any = gzip.open if path.suffix.lower() in (".gz",) else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        first_char = ""
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not first_char:
                first_char = stripped[0]

            if first_char == "{":
                # Could be FeatureCollection or NDJSON
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "FeatureCollection":
                    for feat in obj.get("features", []):
                        yield feat
                    return
                elif obj.get("type") == "Feature":
                    yield obj
                    break  # Switch to NDJSON mode
                else:
                    continue

        # Continue reading remaining NDJSON lines
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if obj.get("type") == "Feature":
                    yield obj
            except json.JSONDecodeError:
                continue


def write_geojsonl(
    features: Sequence[dict[str, Any]] | Iterator[dict[str, Any]],
    path: str | Path,
    *,
    compress: bool = False,
) -> Path:
    """Write features as newline-delimited GeoJSON.

    If *compress* is True, output is gzip-compressed.
    """
    path = Path(path)
    open_fn: Any = gzip.open if compress else open
    mode = "wt"
    with open_fn(path, mode, encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat, separators=(",", ":")) + "\n")
    return path


# ---------------------------------------------------------------------------
# Multi-file glob reader  (item 370)
# ---------------------------------------------------------------------------

def read_glob(
    pattern: str,
    reader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Read multiple files matching a wildcard pattern and merge results.

    *reader* defaults to reading GeoJSON files. Pass a custom reader for
    other formats.
    """
    if reader is None:
        reader = lambda p: json.loads(Path(p).read_text(encoding="utf-8")).get("features", [])  # noqa: E731

    features: list[dict[str, Any]] = []
    for filepath in sorted(glob(pattern, recursive=True)):
        features.extend(reader(filepath))
    return features


# ---------------------------------------------------------------------------
# Row count without loading data  (item 383)
# ---------------------------------------------------------------------------

def feature_count(path: str | Path) -> int:
    """Return the feature/row count of a GeoJSON file without fully loading it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                # Count occurrences of "Feature" type — fast heuristic
                count += line.count('"type":"Feature"') + line.count('"type": "Feature"')
        return count
    elif suffix in (".csv",):
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # subtract header
    elif suffix in (".geojsonl", ".geojsonseq"):
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    raise ValueError(f"unsupported format for feature count: {suffix}")


# ---------------------------------------------------------------------------
# Extent / bounds without loading data  (item 384)
# ---------------------------------------------------------------------------

def quick_bounds(path: str | Path) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) from a GeoJSON file with minimal parsing."""
    path = Path(path)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    coord_pattern = re.compile(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)")
    with open(path, encoding="utf-8") as f:
        in_coords = False
        for line in f:
            if '"coordinates"' in line:
                in_coords = True
            if in_coords:
                for m in coord_pattern.finditer(line):
                    x, y = float(m.group(1)), float(m.group(2))
                    if -180 <= x <= 180 and -90 <= y <= 90:
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
                if "]" in line and "[" not in line:
                    in_coords = False

    if min_x == float("inf"):
        return (0.0, 0.0, 0.0, 0.0)
    return (min_x, min_y, max_x, max_y)


# ---------------------------------------------------------------------------
# Coordinate precision on write  (item 423)
# ---------------------------------------------------------------------------

def round_coordinates(geom: dict[str, Any], precision: int = 6) -> dict[str, Any]:
    """Round all coordinates in a GeoJSON geometry to *precision* decimal places."""
    def _round_coords(coords: Any) -> Any:
        if isinstance(coords, (int, float)):
            return round(coords, precision)
        if isinstance(coords, (list, tuple)):
            return [_round_coords(c) for c in coords]
        return coords

    result = dict(geom)
    if "coordinates" in result:
        result["coordinates"] = _round_coords(result["coordinates"])
    return result


# ---------------------------------------------------------------------------
# Compression on write  (item 425)
# ---------------------------------------------------------------------------

def write_compressed(data: str | bytes, path: str | Path, method: str = "gzip") -> Path:
    """Write data to a compressed file.

    Supported methods: gzip, none.
    """
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if method == "gzip":
        with gzip.open(path, "wb") as f:
            f.write(data)
    elif method == "none":
        path.write_bytes(data)
    else:
        raise ValueError(f"unsupported compression method: {method}")
    return path


# ---------------------------------------------------------------------------
# Feature-count metadata on write  (item 427)
# ---------------------------------------------------------------------------

def write_geojson_with_metadata(
    features: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    precision: int | None = None,
    include_bbox: bool = True,
) -> Path:
    """Write a GeoJSON FeatureCollection with metadata (count, bbox)."""
    path = Path(path)
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": list(features),
        "metadata": {
            "feature_count": len(features),
            "created": datetime.utcnow().isoformat() + "Z",
        },
    }

    if include_bbox and features:
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            for c in _flatten_coords(coords):
                min_x = min(min_x, c[0])
                max_x = max(max_x, c[0])
                min_y = min(min_y, c[1])
                max_y = max(max_y, c[1])
        if min_x != float("inf"):
            fc["bbox"] = [min_x, min_y, max_x, max_y]

    if precision is not None:
        fc["features"] = [
            {**f, "geometry": round_coordinates(f["geometry"], precision)} if f.get("geometry") else f
            for f in fc["features"]
        ]

    path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return path


def _flatten_coords(coords: Any) -> list[list[float]]:
    """Recursively flatten nested coordinate arrays to point-level."""
    if not coords:
        return []
    if isinstance(coords[0], (int, float)):
        return [coords]
    result: list[list[float]] = []
    for c in coords:
        result.extend(_flatten_coords(c))
    return result


# ---------------------------------------------------------------------------
# Write empty layer / copy schema  (items 429, 430)
# ---------------------------------------------------------------------------

def write_empty_layer(path: str | Path, schema: dict[str, str]) -> Path:
    """Write an empty GeoJSON layer with schema metadata."""
    path = Path(path)
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {"schema": schema, "feature_count": 0},
    }
    path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return path


def copy_schema(source_features: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Extract the schema (field name → type string) from a feature set."""
    schema: dict[str, str] = {}
    for feat in source_features:
        for k, v in feat.get("properties", {}).items():
            if k not in schema:
                if isinstance(v, int):
                    schema[k] = "integer"
                elif isinstance(v, float):
                    schema[k] = "float"
                elif isinstance(v, bool):
                    schema[k] = "boolean"
                else:
                    schema[k] = "string"
    return schema


# ---------------------------------------------------------------------------
# __geo_interface__ protocol  (item 421)
# ---------------------------------------------------------------------------

def to_geo_interface(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return a __geo_interface__-compatible dict (GeoJSON FeatureCollection)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": f.get("geometry"),
                "properties": f.get("properties", {}),
            }
            for f in features
        ],
    }


def from_geo_interface(obj: Any) -> list[dict[str, Any]]:
    """Create features from an object implementing __geo_interface__."""
    if hasattr(obj, "__geo_interface__"):
        data = obj.__geo_interface__
    elif isinstance(obj, dict):
        data = obj
    else:
        raise TypeError("object does not implement __geo_interface__")

    if data.get("type") == "FeatureCollection":
        return data.get("features", [])
    elif data.get("type") == "Feature":
        return [data]
    else:
        return [{"type": "Feature", "geometry": data, "properties": {}}]


# ---------------------------------------------------------------------------
# GML read/write  (items 312, 313)
# ---------------------------------------------------------------------------

_GML_NS = {"gml": "http://www.opengis.net/gml/3.2", "gml31": "http://www.opengis.net/gml"}


def read_gml(path: str | Path) -> list[dict[str, Any]]:
    """Read a GML file and return GeoJSON-like features (basic support)."""
    tree = ET.parse(Path(path))
    root = tree.getroot()
    features: list[dict[str, Any]] = []

    for member in root.iter():
        tag = member.tag.split("}")[-1] if "}" in member.tag else member.tag
        if tag != "featureMember" and tag != "member":
            continue
        for child in member:
            props: dict[str, Any] = {}
            geom: dict[str, Any] | None = None
            for elem in child:
                elem_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if elem_tag in ("Point", "LineString", "Polygon"):
                    geom = _gml_geom_to_geojson(elem)
                elif len(elem) == 0 and elem.text:
                    props[elem_tag] = elem.text
                else:
                    # Check for nested geometry
                    for sub in elem:
                        sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if sub_tag in ("Point", "LineString", "Polygon"):
                            geom = _gml_geom_to_geojson(sub)
            if geom:
                features.append({"type": "Feature", "geometry": geom, "properties": props})
    return features


def _gml_geom_to_geojson(elem: ET.Element) -> dict[str, Any] | None:
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
    if tag == "Point":
        for child in elem.iter():
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag in ("pos", "coordinates") and child.text:
                parts = child.text.strip().replace(",", " ").split()
                if len(parts) >= 2:
                    return {"type": "Point", "coordinates": [float(parts[0]), float(parts[1])]}
    elif tag == "LineString":
        for child in elem.iter():
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag in ("posList", "coordinates") and child.text:
                return {"type": "LineString", "coordinates": _parse_poslist(child.text)}
    elif tag == "Polygon":
        rings: list[list[list[float]]] = []
        for child in elem.iter():
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag in ("posList", "coordinates") and child.text:
                rings.append(_parse_poslist(child.text))
        if rings:
            return {"type": "Polygon", "coordinates": rings}
    return None


def _parse_poslist(text: str) -> list[list[float]]:
    values = text.strip().split()
    coords: list[list[float]] = []
    for i in range(0, len(values) - 1, 2):
        coords.append([float(values[i]), float(values[i + 1])])
    return coords


def write_gml(features: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write features to a simple GML 3.2 file."""
    path = Path(path)
    gml = ET.Element("FeatureCollection",
                     xmlns="http://www.opengis.net/gml/3.2")
    for feat in features:
        member = ET.SubElement(gml, "featureMember")
        feature_el = ET.SubElement(member, "Feature")
        geom = feat.get("geometry")
        if geom:
            _write_gml_geometry(feature_el, geom)
        for k, v in feat.get("properties", {}).items():
            prop = ET.SubElement(feature_el, k)
            prop.text = str(v)
    tree = ET.ElementTree(gml)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


def _write_gml_geometry(parent: ET.Element, geom: dict[str, Any]) -> None:
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Point":
        pt = ET.SubElement(parent, "Point")
        ET.SubElement(pt, "pos").text = " ".join(str(c) for c in coords[:2])
    elif gtype == "LineString":
        ls = ET.SubElement(parent, "LineString")
        ET.SubElement(ls, "posList").text = " ".join(
            f"{c[0]} {c[1]}" for c in coords
        )
    elif gtype == "Polygon":
        poly = ET.SubElement(parent, "Polygon")
        for i, ring in enumerate(coords):
            tag = "exterior" if i == 0 else "interior"
            boundary = ET.SubElement(poly, tag)
            lr = ET.SubElement(boundary, "LinearRing")
            ET.SubElement(lr, "posList").text = " ".join(
                f"{c[0]} {c[1]}" for c in ring
            )


# ---------------------------------------------------------------------------
# Date/datetime handling  (item 391)
# ---------------------------------------------------------------------------

_ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
]


def parse_datetime_field(value: Any) -> datetime | None:
    """Attempt to parse a value as a datetime using common formats."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _ISO_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Encoding detection  (item 390)
# ---------------------------------------------------------------------------

def detect_encoding(path: str | Path, sample_size: int = 8192) -> str:
    """Detect the text encoding of a file (basic heuristic)."""
    path = Path(path)
    raw = path.read_bytes()[:sample_size]

    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"

    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        raw.decode("latin-1")
        return "latin-1"
    except UnicodeDecodeError:
        pass

    return "utf-8"


# ---------------------------------------------------------------------------
# Merge multiple files  (item 404)
# ---------------------------------------------------------------------------

def merge_geojson_files(paths: Sequence[str | Path], output: str | Path) -> Path:
    """Merge multiple GeoJSON files into a single FeatureCollection."""
    output = Path(output)
    all_features: list[dict[str, Any]] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if data.get("type") == "FeatureCollection":
            all_features.extend(data.get("features", []))
        elif data.get("type") == "Feature":
            all_features.append(data)
    fc = {"type": "FeatureCollection", "features": all_features}
    output.write_text(json.dumps(fc), encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # KML
    "read_kml",
    "write_kml",
    # GPX
    "read_gpx",
    "write_gpx",
    # TopoJSON
    "read_topojson",
    "write_topojson",
    # GML
    "read_gml",
    "write_gml",
    # Streaming GeoJSON
    "iter_geojson_features",
    "write_geojsonl",
    # Glob reader
    "read_glob",
    # Quick inspection
    "feature_count",
    "quick_bounds",
    # Precision & compression
    "round_coordinates",
    "write_compressed",
    "write_geojson_with_metadata",
    # Empty layer / schema
    "write_empty_layer",
    "copy_schema",
    # Geo interface
    "to_geo_interface",
    "from_geo_interface",
    # Datetime / encoding
    "parse_datetime_field",
    "detect_encoding",
    # Merge
    "merge_geojson_files",
]
