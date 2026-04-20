"""Regression tests for remaining A7 data-management items."""

from pathlib import Path

import geoprompt as gp


class TestA7DataManagement:
    def test_table_to_dbase(self, tmp_path: Path):
        out = tmp_path / "sample.dbf"
        result = gp.table_to_dbase([{"id": 1, "name": "A"}], out)
        assert Path(result).exists()
        assert Path(result).suffix.lower() == ".dbf"

    def test_select_by_polygon(self):
        records = [{"x": 1, "y": 1}, {"x": 10, "y": 10}]
        poly = [(0, 0), (5, 0), (5, 5), (0, 5)]
        selected = gp.select_by_polygon(records, poly)
        assert len(selected) == 1

    def test_compact_and_compress_database(self, tmp_path: Path):
        folder = tmp_path / "my.gdb"
        folder.mkdir()
        compact = gp.compact_database(folder)
        compress = gp.compress_database(folder)
        assert compact["exists"] is True
        assert compress["compressed"] is True

    def test_raster_compare(self):
        result = gp.raster_compare([[1, 2], [3, 4]], [[1, 0], [3, 5]])
        assert result["changed_cells"] == 2

    def test_schema_lock_and_versioning(self):
        locked = gp.schema_lock_management("roads", action="lock")
        unlocked = gp.schema_lock_management("roads", action="unlock")
        reg = gp.versioning_register_unregister("roads", register=True)
        unreg = gp.versioning_register_unregister("roads", register=False)
        assert locked["locked"] is True
        assert unlocked["locked"] is False
        assert reg["versioned"] is True
        assert unreg["versioned"] is False

    def test_workspace_creation_helpers(self, tmp_path: Path):
        ds = gp.create_feature_dataset(tmp_path, "transport")
        gdb = gp.create_file_geodatabase(tmp_path, "demo")
        mgdb = gp.create_mobile_geodatabase(tmp_path, "mobile")
        assert Path(ds["path"]).exists()
        assert Path(gdb["path"]).exists()
        assert Path(mgdb["path"]).exists()

    def test_domain_management(self):
        record = {"field_domains": {"status": {"open": "Open"}}}
        removed = gp.remove_domain_from_field(record, "status")
        altered = gp.alter_domain({"name": "status", "values": {"open": "Open"}}, values={"closed": "Closed"})
        deleted = gp.delete_domain({"status": {"values": {}}}, "status")
        assert "status" not in removed["field_domains"]
        assert altered["values"] == {"closed": "Closed"}
        assert deleted == {}

    def test_subtypes(self):
        table = {"subtypes": {}, "subtype_field": None}
        table = gp.create_subtype(table, 1, "Primary")
        table = gp.set_subtype_field(table, "type_code")
        table = gp.set_default_subtype(table, 1)
        table = gp.remove_subtype(table, 1)
        assert table["subtype_field"] == "type_code"
        assert table["subtypes"] == {}

    def test_relationship_and_attachment_helpers(self):
        catalog = {}
        catalog = gp.create_relationship_class(catalog, "parcel_owner", "parcel", "owner")
        catalog = gp.delete_relationship_class(catalog, "parcel_owner")
        assets = [{"id": 1}]
        assets = gp.create_attachment_table(assets)
        assets = gp.enable_attachments(assets)
        assets = gp.disable_attachments(assets)
        assets = gp.remove_attachments(assets)
        assert catalog["relationship_classes"] == {}
        assert assets[0]["attachments_enabled"] is False
        assert assets[0]["attachments"] == []

    def test_topology_helpers(self, tmp_path: Path):
        topo = gp.create_topology("network")
        topo = gp.add_rule_to_topology(topo, "Must Not Overlap")
        fixed = gp.fix_topology_errors([{"id": 1, "error": "overlap"}])
        out = tmp_path / "topology_errors.json"
        result = gp.export_topology_errors(fixed, out)
        assert topo["rules"] == ["Must Not Overlap"]
        assert fixed[0]["fixed"] is True
        assert Path(result).exists()

    def test_3d_helpers(self):
        feats = [{"height": 10, "x": 0, "y": 0}, {"x": 1, "y": 1}]
        f3d = gp.feature_to_3d_by_attribute(feats, "height")
        interp = gp.interpolate_shape_3d_from_surface([{"x": 0, "y": 0}], [[5]])
        info = gp.add_surface_information([{"elev": 5}, {"elev": 7}], "elev")
        layer = gp.layer_3d_to_feature_class([{"x": 0, "y": 0, "z": 10}])
        assert f3d[0]["z"] == 10
        assert interp[0]["z"] == 5
        assert info["mean_surface"] == 6
        assert layer[0]["geometry"]["type"] == "Point"

    def test_feature_class_to_gdb_batch(self, tmp_path: Path):
        a = tmp_path / "roads.geojson"
        b = tmp_path / "parcels.geojson"
        a.write_text("{}", encoding="utf-8")
        b.write_text("{}", encoding="utf-8")
        gdb = tmp_path / "out.gdb"
        result = gp.feature_class_to_gdb_batch([a, b], gdb)
        assert len(result["copied"]) == 2

    def test_token_based_field_access(self):
        row = {"name": "Parcel 1", "value": 10}
        ctx = {"scale": 24000}
        out = gp.token_based_field_access("$FEATURE.name + ' @ ' + str($MAP.scale)", row, ctx)
        assert out == "Parcel 1 @ 24000"

    def test_transformer_chain(self):
        rows = [{"name": " Main ", "value": 5}]
        chain = [
            {"op": "strip", "field": "name"},
            {"op": "rename", "from": "name", "to": "label"},
            {"op": "calculate", "field": "score", "expression": "row['value'] * 2"},
        ]
        out = gp.transformer_chain(rows, chain)
        assert out[0]["label"] == "Main"
        assert out[0]["score"] == 10
