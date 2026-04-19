from __future__ import annotations

from geoprompt.data_management import export_schema, scaffold_project, stamp_lineage


if __name__ == "__main__":
    records = [
        {"asset_id": "A-1", "status": "active", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"asset_id": "A-2", "status": "planned", "geometry": {"type": "Point", "coordinates": [1, 1]}},
    ]
    project = scaffold_project("outputs", name="roundtrip-demo")
    stamped = stamp_lineage(records, operation="roundtrip_demo", source="database")
    schema = export_schema(stamped, name="asset_inventory")
    print(project)
    print(schema)
