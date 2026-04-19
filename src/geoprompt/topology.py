"""Topology and snapping helpers for GeoPrompt.

This module provides lightweight QA and repair utilities for datasets that
need more structure than basic geometry validation alone.
"""
from __future__ import annotations

from typing import Any, Sequence

from .frame import GeoPromptFrame
from .geometry import (
    geometry_distance,
    geometry_intersects,
    geometry_touches,
    geometry_type,
    geometry_within,
    validate_geometry,
)


def snap_points(
    frame: GeoPromptFrame,
    *,
    tolerance: float,
    target_frame: GeoPromptFrame | None = None,
) -> GeoPromptFrame:
    """Snap point geometries to nearby target points within a tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    targets = target_frame.to_records() if target_frame is not None else []
    rows = frame.to_records()
    snapped_rows: list[dict[str, Any]] = []

    cluster_targets: list[tuple[float, float]] = []
    for row in rows:
        geom = row[frame.geometry_column]
        if geometry_type(geom) != "Point":
            raise TypeError("snap_points only supports point geometries")
        x, y = geom["coordinates"]
        snapped = (x, y)

        if target_frame is not None:
            best = None
            best_distance = None
            for target in targets:
                tgt_geom = target[target_frame.geometry_column]
                if geometry_type(tgt_geom) != "Point":
                    continue
                dist = geometry_distance(geom, tgt_geom)
                if dist <= tolerance and (best_distance is None or dist < best_distance):
                    best = tgt_geom["coordinates"]
                    best_distance = dist
            if best is not None:
                snapped = best
        else:
            for candidate in cluster_targets:
                dist = ((x - candidate[0]) ** 2 + (y - candidate[1]) ** 2) ** 0.5
                if dist <= tolerance:
                    snapped = candidate
                    break
            else:
                cluster_targets.append((x, y))

        new_row = dict(row)
        new_row[frame.geometry_column] = {"type": "Point", "coordinates": snapped}
        snapped_rows.append(new_row)

    return GeoPromptFrame.from_records(snapped_rows, geometry=frame.geometry_column, crs=frame.crs)


def validate_topology_rules(
    frame: GeoPromptFrame,
    *,
    rule: str = "must_not_overlap",
    other: GeoPromptFrame | None = None,
) -> dict[str, Any]:
    """Validate a simple topology rule across a frame."""
    rows = frame.to_records()
    violations: list[dict[str, Any]] = []

    if rule == "must_not_overlap":
        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                right = rows[right_index]
                left_geom = left[frame.geometry_column]
                right_geom = right[frame.geometry_column]
                if geometry_intersects(left_geom, right_geom) and not geometry_touches(left_geom, right_geom):
                    violations.append({
                        "left_index": left_index,
                        "right_index": right_index,
                        "left_id": left.get("id", left.get("site_id")),
                        "right_id": right.get("id", right.get("site_id")),
                    })
    elif rule == "endpoints_must_connect":
        endpoint_counts: dict[tuple[float, float], int] = {}
        for row in rows:
            geom = row[frame.geometry_column]
            if geometry_type(geom) != "LineString":
                continue
            coords = list(geom["coordinates"])
            for endpoint in (tuple(coords[0]), tuple(coords[-1])):
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        for endpoint, count in endpoint_counts.items():
            if count == 1:
                violations.append({"endpoint": endpoint, "issue": "dangling_endpoint"})
    elif rule == "must_not_have_gaps":
        # Check for gaps between polygon features by looking for dangling edges
        # in the boundary adjacency.  A simplified heuristic: each polygon boundary
        # edge should be shared with another polygon or lie on the convex hull.
        for idx, row in enumerate(rows):
            geom = row[frame.geometry_column]
            if geometry_type(geom) != "Polygon":
                continue
            has_neighbor = False
            for other_idx, other_row in enumerate(rows):
                if other_idx == idx:
                    continue
                other_geom = other_row[frame.geometry_column]
                if geometry_touches(geom, other_geom) or geometry_intersects(geom, other_geom):
                    has_neighbor = True
                    break
            if not has_neighbor and len(rows) > 1:
                violations.append({
                    "row_index": idx,
                    "feature_id": row.get("id", row.get("site_id", idx)),
                    "issue": "potential_gap",
                })
    elif rule == "must_not_self_intersect":
        for idx, row in enumerate(rows):
            report = validate_geometry(row[frame.geometry_column])
            if "self_intersection" in report.get("issues", []):
                violations.append({
                    "row_index": idx,
                    "feature_id": row.get("id", row.get("site_id", idx)),
                    "issue": "self_intersection",
                })
    elif rule == "must_be_covered_by":
        if other is None:
            raise ValueError("'other' frame is required for must_be_covered_by")
        other_rows = other.to_records()
        for idx, row in enumerate(rows):
            geom = row[frame.geometry_column]
            if not any(geometry_within(geom, other_row[other.geometry_column]) for other_row in other_rows):
                violations.append({
                    "row_index": idx,
                    "feature_id": row.get("id", row.get("site_id", idx)),
                    "issue": "not_covered",
                })
    else:
        raise ValueError(f"unsupported topology rule: {rule}")

    return {
        "rule": rule,
        "valid": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
    }


def topology_diff_report(
    before: GeoPromptFrame,
    after: GeoPromptFrame,
    *,
    id_column: str = "id",
) -> list[dict[str, Any]]:
    """Compare two feature sets and report changed, added, and removed rows."""
    before_map = {row.get(id_column): row for row in before.to_records()}
    after_map = {row.get(id_column): row for row in after.to_records()}

    ids = set(before_map) | set(after_map)
    report: list[dict[str, Any]] = []
    for fid in sorted(ids, key=str):
        if fid not in before_map:
            report.append({id_column: fid, "status": "added"})
            continue
        if fid not in after_map:
            report.append({id_column: fid, "status": "removed"})
            continue
        before_row = before_map[fid]
        after_row = after_map[fid]
        geometry_changed = before_row.get(before.geometry_column) != after_row.get(after.geometry_column)
        attribute_changes = [
            key for key in set(before_row) | set(after_row)
            if key not in {before.geometry_column, after.geometry_column} and before_row.get(key) != after_row.get(key)
        ]
        if geometry_changed or attribute_changes:
            report.append({
                id_column: fid,
                "status": "changed",
                "geometry_changed": geometry_changed,
                "attribute_changes": attribute_changes,
            })
    return report


def feature_validation_pipeline(
    frame: GeoPromptFrame,
    *,
    rules: Sequence[str] | None = None,
    other: GeoPromptFrame | None = None,
) -> dict[str, Any]:
    """Run multiple topology rules as a validation pipeline.

    Args:
        frame: Features to validate.
        rules: List of rule names (default: ``["must_not_overlap", "must_not_self_intersect"]``).
        other: Optional reference frame for coverage rules.

    Returns:
        Dict with ``"valid"``, ``"rule_results"`` (list of per-rule results),
        and ``"total_violations"``.
    """
    if rules is None:
        rules = ["must_not_overlap", "must_not_self_intersect"]

    results: list[dict[str, Any]] = []
    total_violations = 0
    for rule in rules:
        result = validate_topology_rules(frame, rule=rule, other=other)
        results.append(result)
        total_violations += result["violation_count"]

    return {
        "valid": total_violations == 0,
        "rule_results": results,
        "total_violations": total_violations,
    }


def repair_suggestions(
    frame: GeoPromptFrame,
    *,
    rules: Sequence[str] | None = None,
    other: GeoPromptFrame | None = None,
) -> list[dict[str, Any]]:
    """Generate repair suggestions for topology violations.

    Runs a validation pipeline and returns a suggestion per violation
    describing what action might fix the issue.

    Args:
        frame: Features to validate.
        rules: Rule names to check.
        other: Optional reference frame.

    Returns:
        List of suggestion dicts with ``"rule"``, ``"feature_id"``, ``"suggestion"``.
    """
    pipeline = feature_validation_pipeline(frame, rules=rules, other=other)
    suggestions: list[dict[str, Any]] = []

    repair_map: dict[str, str] = {
        "must_not_overlap": "Consider clipping or dissolving overlapping features.",
        "endpoints_must_connect": "Extend or snap the dangling endpoint to the nearest feature.",
        "must_not_self_intersect": "Run geometry repair (buffer(0) or repair_geometry) on the feature.",
        "must_be_covered_by": "Move or extend the feature so it falls within the reference layer.",
        "must_not_have_gaps": "Add missing features or extend boundaries to close the gap.",
    }

    for rule_result in pipeline["rule_results"]:
        rule_name = rule_result["rule"]
        suggestion_text = repair_map.get(rule_name, "Review the violation manually.")
        for violation in rule_result["violations"]:
            fid = violation.get("feature_id", violation.get("left_id", violation.get("endpoint", "unknown")))
            suggestions.append({
                "rule": rule_name,
                "feature_id": fid,
                "suggestion": suggestion_text,
                "violation": violation,
            })

    return suggestions


__all__ = [
    "feature_validation_pipeline",
    "repair_suggestions",
    "snap_points",
    "topology_diff_report",
    "validate_topology_rules",
]
