"""Data management, conversion, schema tools, and enterprise data helpers.

Covers geodatabase inspection, domain validation, field calculators,
append/upsert workflows, format conversion, schema diffing, coordinate
cleaning, lineage stamps, and workspace cataloging.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------

def _try_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ── 1. Geodatabase schema inspection and export ──────────────────────────────

def inspect_geodatabase(path: str | Path) -> dict[str, Any]:
    """Inspect a file geodatabase (.gdb) or GeoPackage and return schema info.

    Returns ``layers`` list with name, geometry_type, field_count, feature_count.
    """
    import fiona  # type: ignore[import-untyped]
    gdb_path = Path(path)
    layers: list[dict[str, Any]] = []
    for layer_name in fiona.listlayers(str(gdb_path)):
        with fiona.open(str(gdb_path), layer=layer_name) as src:
            layers.append({
                "name": layer_name,
                "geometry_type": src.schema.get("geometry", "Unknown"),
                "fields": list(src.schema.get("properties", {}).keys()),
                "field_count": len(src.schema.get("properties", {})),
                "feature_count": len(src),
                "crs": str(src.crs) if src.crs else None,
            })
    return {"path": str(gdb_path), "layer_count": len(layers), "layers": layers}


def export_schema(records: Sequence[dict[str, Any]], *, name: str = "dataset") -> dict[str, Any]:
    """Export a lightweight schema descriptor from a record list."""
    if not records:
        return {"name": name, "fields": [], "row_count": 0}
    sample = records[0]
    fields = []
    for k, v in sample.items():
        dtype = type(v).__name__ if v is not None else "unknown"
        fields.append({"name": k, "type": dtype})
    return {"name": name, "fields": fields, "row_count": len(records)}


# ── 2. Domains and subtype validation ────────────────────────────────────────

class FieldDomain:
    """Enterprise-style coded-value or range domain for field validation."""

    def __init__(
        self,
        name: str,
        domain_type: str = "coded",
        values: dict[Any, str] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        self.name = name
        self.domain_type = domain_type
        self.values = values or {}
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> bool:
        if value is None:
            return True
        if self.domain_type == "coded":
            return value in self.values
        if self.domain_type == "range":
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.domain_type, "values": self.values, "min": self.min_value, "max": self.max_value}


def validate_domains(
    records: Sequence[dict[str, Any]],
    domain_map: dict[str, FieldDomain],
) -> list[dict[str, Any]]:
    """Validate records against field domains; return violations."""
    violations: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        for field, domain in domain_map.items():
            val = r.get(field)
            if not domain.validate(val):
                violations.append({"row": i, "field": field, "value": val, "domain": domain.name})
    return violations


# ── 3. Attribute rules and constraint-checking ───────────────────────────────

def check_constraints(
    records: Sequence[dict[str, Any]],
    constraints: dict[str, Callable[[Any], bool]],
) -> list[dict[str, Any]]:
    """Check each record against per-field constraint functions.

    *constraints* maps field names to callables that return True if valid.
    """
    violations: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        for field, fn in constraints.items():
            val = r.get(field)
            try:
                if not fn(val):
                    violations.append({"row": i, "field": field, "value": val})
            except Exception as exc:
                violations.append({"row": i, "field": field, "value": val, "error": str(exc)})
    return violations


# ── 4. Field calculator ──────────────────────────────────────────────────────

_SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "round": round, "len": len, "int": int, "float": float, "str": str, "bool": bool}


def field_calculate(
    records: list[dict[str, Any]],
    target_field: str,
    expression: str,
    *,
    safe: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate *expression* per row and store result in *target_field*.

    Uses a restricted eval sandbox by default.  ``row`` is the current record dict.
    """
    for r in records:
        ns = dict(_SAFE_BUILTINS) if safe else {}
        ns["row"] = r
        ns["__builtins__"] = _SAFE_BUILTINS if safe else {}
        r[target_field] = eval(expression, ns)  # noqa: S307
    return records


# ── 5. Append, upsert, truncate, overwrite workflows ────────────────────────

def append_records(
    target: list[dict[str, Any]],
    source: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append source records to target, returning the combined list."""
    target.extend(source)
    return target


def upsert_records(
    target: list[dict[str, Any]],
    source: Sequence[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Insert or update records in *target* from *source* by *key*."""
    index = {r[key]: i for i, r in enumerate(target) if key in r}
    for s in source:
        k = s.get(key)
        if k is not None and k in index:
            target[index[k]].update(s)
        else:
            target.append(s)
    return target


def truncate_records(target: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove all records from *target* in place."""
    target.clear()
    return target


def overwrite_records(
    target: list[dict[str, Any]],
    source: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace all records in *target* with *source*."""
    target.clear()
    target.extend(source)
    return target


# ── 6. Batch rename, reorder, alias ──────────────────────────────────────────

def batch_rename_fields(
    records: list[dict[str, Any]],
    rename_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Rename fields across all records per *rename_map*."""
    for r in records:
        for old, new in rename_map.items():
            if old in r:
                r[new] = r.pop(old)
    return records


def reorder_fields(
    records: Sequence[dict[str, Any]],
    order: Sequence[str],
) -> list[dict[str, Any]]:
    """Return records with fields reordered per *order*.  Extra fields appended."""
    result: list[dict[str, Any]] = []
    for r in records:
        ordered = {k: r[k] for k in order if k in r}
        for k, v in r.items():
            if k not in ordered:
                ordered[k] = v
        result.append(ordered)
    return result


# ── 7. Relationship-class style helpers ──────────────────────────────────────

def one_to_many_join(
    parent: Sequence[dict[str, Any]],
    child: Sequence[dict[str, Any]],
    parent_key: str,
    child_key: str,
    children_field: str = "_children",
) -> list[dict[str, Any]]:
    """Attach child records to parents as nested lists."""
    child_index: dict[Any, list[dict[str, Any]]] = {}
    for c in child:
        k = c.get(child_key)
        child_index.setdefault(k, []).append(c)
    result = []
    for p in parent:
        row = dict(p)
        row[children_field] = child_index.get(p.get(parent_key), [])
        result.append(row)
    return result


def many_to_many_join(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    bridge: Sequence[dict[str, str]],
    left_key: str,
    right_key: str,
    bridge_left: str,
    bridge_right: str,
) -> list[dict[str, Any]]:
    """Resolve a many-to-many relationship through a bridge table."""
    right_idx = {r[right_key]: r for r in right if right_key in r}
    bridge_map: dict[Any, list[Any]] = {}
    for b in bridge:
        bridge_map.setdefault(b.get(bridge_left), []).append(b.get(bridge_right))
    results: list[dict[str, Any]] = []
    for left_row in left:
        lk = left_row.get(left_key)
        for rk in bridge_map.get(lk, []):
            matched = right_idx.get(rk)
            if matched:
                merged = dict(left_row)
                for mk, mv in matched.items():
                    if mk != right_key:
                        merged[f"right_{mk}"] = mv
                results.append(merged)
    return results


# ── 8. Feature class copy, move, archive, snapshot ───────────────────────────

def copy_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy a record list."""
    return copy.deepcopy(list(records))


def archive_records(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    fmt: str = "json",
) -> str:
    """Write records to an archive file and return the path."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        p.write_text(json.dumps(list(records), default=str), encoding="utf-8")
    elif fmt == "csv":
        if not records:
            p.write_text("", encoding="utf-8")
        else:
            keys = list(records[0].keys())
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(records)
    return str(p)


def snapshot_records(
    records: Sequence[dict[str, Any]],
    *,
    label: str = "",
) -> dict[str, Any]:
    """Create a versioned in-memory snapshot with timestamp and hash."""
    payload = json.dumps(list(records), default=str, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hash": digest,
        "row_count": len(records),
        "data": list(records),
    }


# ── 9. Metadata read/write/clone ─────────────────────────────────────────────

def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read JSON metadata sidecar (.meta.json) for a dataset."""
    meta_path = Path(str(path) + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> str:
    """Write JSON metadata sidecar for a dataset."""
    meta_path = Path(str(path) + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return str(meta_path)


def clone_metadata(source_path: str | Path, target_path: str | Path) -> str:
    """Clone the metadata sidecar from one dataset to another."""
    meta = read_metadata(source_path)
    meta["cloned_from"] = str(source_path)
    meta["cloned_at"] = datetime.now(timezone.utc).isoformat()
    return write_metadata(target_path, meta)


# ── 10. Template-driven project scaffolding ──────────────────────────────────

def scaffold_project(
    root: str | Path,
    *,
    name: str = "project",
    folders: Sequence[str] = ("data", "outputs", "docs", "scripts"),
) -> dict[str, Any]:
    """Create a project directory structure with starter files."""
    root_path = Path(root) / name
    root_path.mkdir(parents=True, exist_ok=True)
    created: list[str] = [str(root_path)]
    for folder in folders:
        fp = root_path / folder
        fp.mkdir(exist_ok=True)
        created.append(str(fp))
    readme = root_path / "README.md"
    if not readme.exists():
        readme.write_text(f"# {name}\n\nGeoPrompt project.\n", encoding="utf-8")
        created.append(str(readme))
    return {"root": str(root_path), "created": created}


# ── 11. Batch format conversion ──────────────────────────────────────────────

def batch_convert(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    target_format: str = "geojson",
) -> list[str]:
    """Convert multiple files to *target_format* (geojson, csv, json).

    Uses the geoprompt IO layer internally.
    """
    from geoprompt.io import read_data, write_data
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for inp in input_paths:
        frame = read_data(str(inp))
        stem = Path(inp).stem
        ext = {"geojson": ".geojson", "csv": ".csv", "json": ".json"}.get(target_format, f".{target_format}")
        out_path = out / f"{stem}{ext}"
        write_data(str(out_path), frame)
        outputs.append(str(out_path))
    return outputs


# ── 12. Schema diff and migration planner ────────────────────────────────────

def schema_diff(
    schema_a: dict[str, str],
    schema_b: dict[str, str],
) -> dict[str, Any]:
    """Compare two field-name→type dicts and report differences."""
    a_keys = set(schema_a)
    b_keys = set(schema_b)
    added = {k: schema_b[k] for k in b_keys - a_keys}
    removed = {k: schema_a[k] for k in a_keys - b_keys}
    type_changed = {}
    for k in a_keys & b_keys:
        if schema_a[k] != schema_b[k]:
            type_changed[k] = {"from": schema_a[k], "to": schema_b[k]}
    return {"added": added, "removed": removed, "type_changed": type_changed, "compatible": not (removed or type_changed)}


def migration_plan(diff: dict[str, Any]) -> list[str]:
    """Generate migration steps from a schema diff."""
    steps: list[str] = []
    for field, dtype in diff.get("added", {}).items():
        steps.append(f"ADD COLUMN {field} {dtype}")
    for field in diff.get("removed", {}):
        steps.append(f"DROP COLUMN {field}")
    for field, change in diff.get("type_changed", {}).items():
        steps.append(f"ALTER COLUMN {field} FROM {change['from']} TO {change['to']}")
    return steps


# ── 13. Feature comparison and change-detection ──────────────────────────────

def feature_change_report(
    old_records: Sequence[dict[str, Any]],
    new_records: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Detect added, removed, and modified features between two snapshots."""
    old_idx = {r[key]: r for r in old_records if key in r}
    new_idx = {r[key]: r for r in new_records if key in r}
    added = [k for k in new_idx if k not in old_idx]
    removed = [k for k in old_idx if k not in new_idx]
    modified: list[Any] = []
    for k in set(old_idx) & set(new_idx):
        if json.dumps(old_idx[k], sort_keys=True, default=str) != json.dumps(new_idx[k], sort_keys=True, default=str):
            modified.append(k)
    return {"added": added, "removed": removed, "modified": modified, "unchanged": len(old_idx) - len(removed) - len(modified)}


# ── 14. Coordinate cleaning and standardization ─────────────────────────────

def clean_coordinates(
    records: list[dict[str, Any]],
    x_field: str = "longitude",
    y_field: str = "latitude",
) -> list[dict[str, Any]]:
    """Clamp coordinates to valid WGS-84 ranges and strip whitespace."""
    for r in records:
        try:
            x = float(str(r.get(x_field, "")).strip())
            y = float(str(r.get(y_field, "")).strip())
            r[x_field] = max(-180.0, min(180.0, x))
            r[y_field] = max(-90.0, min(90.0, y))
        except (ValueError, TypeError):
            pass
    return records


def standardize_columns(
    records: list[dict[str, Any]],
    *,
    lower: bool = True,
    strip: bool = True,
    replace_spaces: str = "_",
) -> list[dict[str, Any]]:
    """Normalize column names across all records."""
    result: list[dict[str, Any]] = []
    for r in records:
        new_r: dict[str, Any] = {}
        for k, v in r.items():
            nk = k
            if strip:
                nk = nk.strip()
            if replace_spaces:
                nk = nk.replace(" ", replace_spaces)
            if lower:
                nk = nk.lower()
            new_r[nk] = v
        result.append(new_r)
    return result


# ── 15. Workspace catalog browser ────────────────────────────────────────────

_DATA_EXTENSIONS = {
    ".geojson", ".json", ".csv", ".tsv", ".shp", ".gpkg", ".gdb",
    ".parquet", ".fgb", ".xlsx", ".xls", ".tif", ".tiff",
}


def catalog_workspace(root: str | Path) -> list[dict[str, Any]]:
    """Scan a directory tree and return a catalog of recognized datasets."""
    root_path = Path(root)
    entries: list[dict[str, Any]] = []
    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in _DATA_EXTENSIONS:
            entries.append({
                "path": str(p),
                "name": p.name,
                "format": p.suffix.lower().lstrip("."),
                "size_bytes": p.stat().st_size,
            })
    return entries


# ── 16. Richer Parquet / Arrow optimization ──────────────────────────────────

def read_parquet_filtered(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    row_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Read a Parquet file with optional column and row predicate pushdown."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    table = pq.read_table(str(path), columns=list(columns) if columns else None)
    records = table.to_pydict()
    rows: list[dict[str, Any]] = []
    n = len(next(iter(records.values()))) if records else 0
    for i in range(n):
        row = {k: v[i] for k, v in records.items()}
        rows.append(row)
    if row_filter:
        ns = dict(_SAFE_BUILTINS)
        ns["__builtins__"] = _SAFE_BUILTINS
        rows = [r for r in rows if eval(row_filter, ns, {"row": r})]  # noqa: S307
    return rows


# ── 17. Data lineage stamps ──────────────────────────────────────────────────

def stamp_lineage(
    records: list[dict[str, Any]],
    *,
    operation: str,
    source: str = "",
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Embed lineage metadata into each record."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    for r in records:
        r.setdefault("_lineage", [])
        r["_lineage"].append({"operation": operation, "source": source, "timestamp": ts})
    return records


# ── 18. Attachment and related-media management ──────────────────────────────

def attach_media(
    record: dict[str, Any],
    media_path: str | Path,
    *,
    field: str = "_attachments",
) -> dict[str, Any]:
    """Register a media file path as an attachment on a record."""
    record.setdefault(field, [])
    record[field].append({"path": str(media_path), "attached_at": datetime.now(timezone.utc).isoformat()})
    return record


def list_attachments(record: dict[str, Any], *, field: str = "_attachments") -> list[dict[str, Any]]:
    """Return attachment metadata from a record."""
    return record.get(field, [])


# ── 19. Zip package import / export ──────────────────────────────────────────

def export_zip_bundle(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    extras: Sequence[str | Path] | None = None,
) -> str:
    """Package records as GeoJSON plus extra files into a .zip deliverable."""
    import zipfile
    out = Path(output_path)
    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {k: v for k, v in r.items() if k != "geometry"}, "geometry": r.get("geometry")}
            for r in records
        ]}
        zf.writestr("data.geojson", json.dumps(fc, default=str))
        for extra in extras or []:
            ep = Path(extra)
            if ep.exists():
                zf.write(str(ep), ep.name)
    return str(out)


def import_zip_bundle(zip_path: str | Path) -> dict[str, Any]:
    """Unpack a .zip and return contained GeoJSON records plus file listing."""
    import zipfile
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        records: list[dict[str, Any]] = []
        for n in names:
            if n.endswith(".geojson") or n.endswith(".json"):
                data = json.loads(zf.read(n))
                if isinstance(data, dict) and "features" in data:
                    for f in data["features"]:
                        row = dict(f.get("properties", {}))
                        row["geometry"] = f.get("geometry")
                        records.append(row)
        return {"files": names, "records": records}


# ── 20. Richer Excel roundtrip ───────────────────────────────────────────────

def write_excel_styled(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    sheet_name: str = "Data",
    header_color: str = "4472C4",
) -> str:
    """Write records to Excel with styled headers using openpyxl."""
    from openpyxl import Workbook  # type: ignore[import-untyped]
    from openpyxl.styles import Font, PatternFill  # type: ignore[import-untyped]
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    if not records:
        wb.save(str(output_path))
        return str(output_path)
    headers = list(records[0].keys())
    fill = PatternFill(start_color=header_color, end_color=header_color, fill_type="solid")
    font = Font(color="FFFFFF", bold=True)
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.fill = fill
        cell.font = font
    for ri, rec in enumerate(records, 2):
        for ci, h in enumerate(headers, 1):
            v = rec.get(h)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, default=str)
            ws.cell(row=ri, column=ci, value=v)
    wb.save(str(output_path))
    return str(output_path)


# ---------------------------------------------------------------------------
# Additional data management utilities (A7 items)
# ---------------------------------------------------------------------------


def frequency_analysis(
    records: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Count the frequency of unique value combinations across *fields*.

    Parameters
    ----------
    records : sequence of dicts
    fields : sequence of str
        Field names to group by.

    Returns
    -------
    list[dict]
        Records with the grouped field values plus a ``"FREQUENCY"`` key.
    """
    from collections import Counter

    counter: Counter[tuple[Any, ...]] = Counter()
    for rec in records:
        key = tuple(rec.get(f) for f in fields)
        counter[key] += 1

    result: list[dict[str, Any]] = []
    for key, count in counter.most_common():
        row: dict[str, Any] = {f: v for f, v in zip(fields, key)}
        row["FREQUENCY"] = count
        result.append(row)
    return result


def find_identical(
    records: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> list[list[int]]:
    """Find groups of records that are identical across *fields*.

    Parameters
    ----------
    records : sequence of dicts
    fields : sequence of str

    Returns
    -------
    list[list[int]]
        Groups of record indices that share the same values.
        Only groups with 2+ members are included.
    """
    from collections import defaultdict

    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        key = tuple(rec.get(f) for f in fields)
        groups[key].append(idx)
    return [idxs for idxs in groups.values() if len(idxs) >= 2]


def delete_identical(
    records: list[dict[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Remove duplicate records, keeping first occurrence per unique field combo.

    Parameters
    ----------
    records : list of dicts
    fields : sequence of str

    Returns
    -------
    list[dict]
        De-duplicated records.
    """
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for rec in records:
        key = tuple(rec.get(f) for f in fields)
        if key not in seen:
            seen.add(key)
            result.append(rec)
    return result


def transpose_table(
    records: Sequence[dict[str, Any]],
    id_field: str,
) -> list[dict[str, Any]]:
    """Transpose a table, turning row values into columns.

    The *id_field* values become column headers; other fields become rows.

    Parameters
    ----------
    records : sequence of dicts
    id_field : str
        Field whose values become the new column names.

    Returns
    -------
    list[dict]
        Transposed records.
    """
    if not records:
        return []
    other_fields = [f for f in records[0] if f != id_field]
    result: list[dict[str, Any]] = []
    for field in other_fields:
        row: dict[str, Any] = {"field": field}
        for rec in records:
            col_name = str(rec.get(id_field, ""))
            row[col_name] = rec.get(field)
        result.append(row)
    return result


def table_to_csv(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    fields: Sequence[str] | None = None,
) -> str:
    """Write records to a CSV file.

    Parameters
    ----------
    records : sequence of dicts
    output_path : str or Path
    fields : sequence of str, optional
        Columns to include. If omitted, all columns from the first record are used.

    Returns
    -------
    str
        The path written.
    """
    if not records:
        Path(output_path).write_text("", encoding="utf-8")
        return str(output_path)
    cols = list(fields) if fields else list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return str(output_path)


def get_count(records: Sequence[dict[str, Any]]) -> int:
    """Return the number of records.

    Parameters
    ----------
    records : sequence of dicts

    Returns
    -------
    int
    """
    return len(records)


def list_fields(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """List the field names and inferred types of a record set.

    Parameters
    ----------
    records : sequence of dicts

    Returns
    -------
    list[dict]
        Each dict has ``"name"`` and ``"type"`` keys.
    """
    if not records:
        return []
    sample = records[0]
    result: list[dict[str, str]] = []
    for key, val in sample.items():
        t = type(val).__name__ if val is not None else "NoneType"
        result.append({"name": key, "type": t})
    return result


def table_compare(
    table_a: Sequence[dict[str, Any]],
    table_b: Sequence[dict[str, Any]],
    *,
    key_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare two tables and report differences.

    Parameters
    ----------
    table_a, table_b : sequences of dicts
    key_fields : sequence of str, optional
        Fields that identify matching rows. If omitted, comparison is by index.

    Returns
    -------
    dict
        Keys: ``"added"`` (in B not A), ``"removed"`` (in A not B),
        ``"modified"`` (matching key but different values),
        ``"identical_count"`` (matching rows).
    """
    if key_fields:
        def _key(rec: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(rec.get(f) for f in key_fields)

        map_a = {_key(r): r for r in table_a}
        map_b = {_key(r): r for r in table_b}
        added = [map_b[k] for k in map_b if k not in map_a]
        removed = [map_a[k] for k in map_a if k not in map_b]
        modified: list[dict[str, Any]] = []
        identical = 0
        for k in map_a:
            if k in map_b:
                if map_a[k] == map_b[k]:
                    identical += 1
                else:
                    modified.append({"key": k, "a": map_a[k], "b": map_b[k]})
    else:
        added_list: list[dict[str, Any]] = []
        removed_list: list[dict[str, Any]] = []
        modified = []
        identical = 0
        max_len = max(len(table_a), len(table_b))
        for i in range(max_len):
            if i >= len(table_a):
                added_list.append(table_b[i])
            elif i >= len(table_b):
                removed_list.append(table_a[i])
            elif table_a[i] == table_b[i]:
                identical += 1
            else:
                modified.append({"index": i, "a": table_a[i], "b": table_b[i]})
        added = added_list
        removed = removed_list

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "identical_count": identical,
    }


def split_layer_by_attribute(
    records: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into groups by unique values of *field*.

    Parameters
    ----------
    records : sequence of dicts
    field : str
        Field to group by.

    Returns
    -------
    dict[str, list[dict]]
        Keys are the unique field values (stringified); values are record lists.
    """
    from collections import defaultdict

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = str(rec.get(field, ""))
        groups[key].append(rec)
    return dict(groups)


def validate_table_name(name: str) -> dict[str, Any]:
    """Validate a table name against common database naming rules.

    Returns
    -------
    dict
        Keys: ``"valid"`` (bool), ``"issues"`` (list of str).
    """
    import re

    issues: list[str] = []
    if not name:
        issues.append("name is empty")
    elif not re.match(r"^[A-Za-z_]", name):
        issues.append("must start with a letter or underscore")
    if re.search(r"[^A-Za-z0-9_]", name):
        issues.append("contains invalid characters (only letters, digits, _ allowed)")
    if len(name) > 128:
        issues.append("exceeds 128 characters")
    reserved = {"SELECT", "FROM", "WHERE", "TABLE", "DROP", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "INDEX"}
    if name.upper() in reserved:
        issues.append(f"'{name}' is a reserved SQL keyword")
    return {"valid": len(issues) == 0, "issues": issues}


def validate_field_name(name: str) -> dict[str, Any]:
    """Validate a field name against common database naming rules.

    Returns
    -------
    dict
        Keys: ``"valid"`` (bool), ``"issues"`` (list of str).
    """
    import re

    issues: list[str] = []
    if not name:
        issues.append("name is empty")
    elif not re.match(r"^[A-Za-z_]", name):
        issues.append("must start with a letter or underscore")
    if re.search(r"[^A-Za-z0-9_]", name):
        issues.append("contains invalid characters")
    if len(name) > 64:
        issues.append("exceeds 64 characters")
    return {"valid": len(issues) == 0, "issues": issues}


def write_csv_with_xy(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    geometry_column: str = "geometry",
    x_field: str = "X",
    y_field: str = "Y",
) -> str:
    """Write records to CSV with X/Y coordinate columns extracted from geometry.

    Parameters
    ----------
    records : sequence of dicts
    output_path : str or Path
    geometry_column : str
    x_field, y_field : str
        Column names for the coordinates in the output.

    Returns
    -------
    str
        Path written.
    """
    from .geometry import geometry_centroid

    if not records:
        Path(output_path).write_text("", encoding="utf-8")
        return str(output_path)

    non_geom_fields = [k for k in records[0] if k != geometry_column]
    fieldnames = non_geom_fields + [x_field, y_field]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in non_geom_fields}
            geom = rec.get(geometry_column)
            if geom:
                centroid = geometry_centroid(geom)
                row[x_field] = centroid[0]
                row[y_field] = centroid[1]
            else:
                row[x_field] = None
                row[y_field] = None
            writer.writerow(row)
    return str(output_path)


def pretty_print_feature_table(
    records: Sequence[dict[str, Any]],
    *,
    max_rows: int = 20,
    max_col_width: int = 30,
    geometry_column: str = "geometry",
) -> str:
    """Format records as a human-readable ASCII table string.

    Geometry fields are summarized as their type rather than printed in full.

    Parameters
    ----------
    records : sequence of dicts
    max_rows : int
        Maximum rows to display. Default 20.
    max_col_width : int
        Maximum column width. Default 30.
    geometry_column : str

    Returns
    -------
    str
        Formatted table string.
    """
    if not records:
        return "(empty table)"
    display = records[:max_rows]
    cols = list(display[0].keys())

    def _fmt(val: Any, col: str) -> str:
        if col == geometry_column and isinstance(val, dict):
            return f"<{val.get('type', 'Geometry')}>"
        s = str(val) if val is not None else ""
        return s[:max_col_width] if len(s) > max_col_width else s

    rows = [[_fmt(rec.get(c), c) for c in cols] for rec in display]
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    if len(records) > max_rows:
        lines.append(f"... ({len(records) - max_rows} more rows)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
__all__ = [
    "append_records",
    "archive_records",
    "attach_media",
    "batch_convert",
    "batch_rename_fields",
    "catalog_workspace",
    "check_constraints",
    "clean_coordinates",
    "clone_metadata",
    "copy_records",
    "delete_identical",
    "export_schema",
    "export_zip_bundle",
    "feature_change_report",
    "field_calculate",
    "FieldDomain",
    "find_identical",
    "frequency_analysis",
    "get_count",
    "import_zip_bundle",
    "inspect_geodatabase",
    "list_attachments",
    "list_fields",
    "many_to_many_join",
    "migration_plan",
    "one_to_many_join",
    "overwrite_records",
    "pretty_print_feature_table",
    "read_metadata",
    "read_parquet_filtered",
    "reorder_fields",
    "scaffold_project",
    "schema_diff",
    "snapshot_records",
    "split_layer_by_attribute",
    "stamp_lineage",
    "standardize_columns",
    "table_compare",
    "table_to_csv",
    "transpose_table",
    "truncate_records",
    "upsert_records",
    "validate_domains",
    "validate_field_name",
    "validate_table_name",
    "write_csv_with_xy",
    "write_excel_styled",
    "write_metadata",
]
