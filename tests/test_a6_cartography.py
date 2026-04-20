"""Regression tests for remaining A6 cartography and visualisation items."""

from pathlib import Path

import geoprompt as gp


class TestA6Cartography:
    def test_interactive_map_specs(self):
        features = [{"id": 1, "geometry": {"type": "Point", "coordinates": [0, 0]}}]
        mb = gp.interactive_web_map_mapbox_gl_js(features)
        deck = gp.interactive_web_map_deck_gl(features)
        ipy = gp.interactive_web_map_ipyleaflet(features)
        assert mb["engine"] == "mapbox-gl-js"
        assert deck["engine"] == "deck.gl"
        assert ipy["engine"] == "ipyleaflet"

    def test_layer_and_3d_specs(self):
        wms = gp.wms_overlay_layer("https://example.com/wms", "roads")
        wmts = gp.wmts_overlay_layer("https://example.com/wmts", "tiles")
        vt = gp.vector_tile_layer("https://tiles/{z}/{x}/{y}.pbf")
        ext = gp.extrusion_3d_layer([{"height": 10}])
        sc3d = gp.scatter_3d_layer([{"x": 1, "y": 2, "z": 3}])
        terr = gp.terrain_surface_3d([[1, 2], [3, 4]])
        assert wms["type"] == "wms"
        assert wmts["type"] == "wmts"
        assert vt["type"] == "vector-tile"
        assert ext[0]["extrusion_height"] == 10
        assert sc3d[0]["z"] == 3
        assert terr["rows"] == 2

    def test_bivariate_and_label_helpers(self):
        out = gp.bivariate_choropleth([
            {"x": 1, "y": 2},
            {"x": 2, "y": 1},
        ], "x", "y")
        leaders = gp.label_leader_lines([{"x": 0, "y": 0, "label_x": 1, "label_y": 1}])
        coord = gp.coordinate_display_on_click_hover(10, 20)
        assert len(out) == 2
        assert leaders[0]["from"] == [0, 0]
        assert coord["lon"] == 10

    def test_time_and_navigation_helpers(self):
        timed = gp.time_enabled_layer_animation([{"time": "2026-01-01", "value": 1}])
        linked = gp.linked_view_navigation(["map1", "map2"])
        slider = gp.time_slider_control(["2026-01-01", "2026-01-02"])
        date_filter = gp.date_filter_control("date")
        zoom = gp.zoom_to_feature_control({"id": 1})
        pan = gp.pan_to_coordinates_control(1, 2)
        assert timed["frame_count"] == 1
        assert linked["views"] == 2
        assert slider["type"] == "time-slider"
        assert date_filter["field"] == "date"
        assert zoom["type"] == "zoom-to-feature"
        assert pan["coordinates"] == [1, 2]

    def test_layout_and_legend(self):
        tmpl = gp.layout_template_management("executive", {"title": "Demo"})
        legend = gp.legend_patch_shape_customisation([{"label": "Road"}], patch_shape="circle")
        assert tmpl["template_name"] == "executive"
        assert legend[0]["patch_shape"] == "circle"

    def test_chart_and_network_views(self):
        pc = gp.parallel_coordinates([{"a": 1, "b": 2}, {"a": 2, "b": 1}], ["a", "b"])
        sankey = gp.sankey_flow_diagram([{"source": "A", "target": "B", "value": 5}])
        net = gp.network_graph_visualisation({"nodes": [1, 2], "edges": [(1, 2)]})
        sm = gp.small_multiples_faceted_maps([{"grp": "x"}, {"grp": "y"}], "grp")
        assert pc["dimensions"] == ["a", "b"]
        assert sankey["links"][0]["value"] == 5
        assert net["node_count"] == 2
        assert len(sm) == 2

    def test_report_and_preview_outputs(self, tmp_path: Path):
        docx = gp.report_generation_word_docx([{"title": "Summary", "content": "Done"}], tmp_path / "report.docx")
        html = gp.geojson_preview_in_browser({"type": "FeatureCollection", "features": []}, title="Preview")
        assert Path(docx).exists()
        assert "FeatureCollection" in html

    def test_symbol_and_effect_helpers(self):
        svg = gp.svg_marker_library("pin", color="#ff0000")
        custom = gp.custom_marker_from_png_svg("icon.svg")
        pattern = gp.pattern_fill_symbology("diagonal")
        hatch = gp.cross_hatch_fill_symbology()
        picture = gp.picture_fill_symbology("texture.png")
        deco = gp.cartographic_line_decoration("arrows")
        cased = gp.cased_line_symbology("#000", "#fff")
        tapered = gp.tapered_line(2, 8)
        offset = gp.offset_line_symbology(5)
        multi = gp.multi_layer_symbology([{"color": "red"}, {"color": "blue"}])
        trans = gp.transparency_mask(0.4)
        blend = gp.blend_modes("multiply")
        shadow = gp.drop_shadow_effect(3)
        glow = gp.glow_effect("outer", 4)
        assert "<svg" in svg
        assert custom["path"] == "icon.svg"
        assert pattern["pattern"] == "diagonal"
        assert hatch["pattern"] == "cross-hatch"
        assert picture["image"] == "texture.png"
        assert deco["decoration"] == "arrows"
        assert cased["outer_color"] == "#000"
        assert tapered["end_width"] == 8
        assert offset["offset"] == 5
        assert len(multi["layers"]) == 2
        assert trans["opacity"] == 0.4
        assert blend["mode"] == "multiply"
        assert shadow["blur"] == 3
        assert glow["style"] == "outer"

    def test_flow_and_animation_helpers(self):
        flows = [{"origin": [0, 0], "destination": [1, 1], "value": 10}]
        rings = gp.buffer_zone_display([(0, 0)], [100, 200])
        prop = gp.proportional_flow_lines(flows)
        desire = gp.desire_lines(flows)
        spider = gp.spider_diagram(flows)
        anim = gp.animated_path_moving_marker([[0, 0], [1, 1]])
        assert len(rings) == 2
        assert prop[0]["width"] > 0
        assert desire[0]["type"] == "desire-line"
        assert spider[0]["type"] == "spider-line"
        assert anim["frames"] == 2

    def test_controls_and_export_helpers(self):
        search = gp.search_geocode_control()
        draw = gp.draw_edit_control()
        pr = gp.print_export_control()
        attr = gp.attribution_control("GeoPrompt")
        shot = gp.screenshot_to_clipboard("map123")
        thumb = gp.thumbnail_generation([{"id": 1}], width=200, height=100)
        qr = gp.qr_code_with_spatial_link("https://example.com/map")
        cache = gp.map_offline_cache(["tile1", "tile2"])
        dark = gp.dark_mode_basemap()
        print_base = gp.print_optimised_basemap()
        sat = gp.satellite_basemap()
        terr = gp.terrain_basemap()
        street = gp.street_basemap()
        custom = gp.custom_basemap_from_url_template("https://tiles/{z}/{x}/{y}.png")
        assert search["type"] == "search-control"
        assert draw["type"] == "draw-edit-control"
        assert pr["type"] == "print-export-control"
        assert attr["text"] == "GeoPrompt"
        assert shot["copied"] is True
        assert thumb["width"] == 200
        assert qr["url"] == "https://example.com/map"
        assert cache["tile_count"] == 2
        assert dark["theme"] == "dark"
        assert print_base["theme"] == "print"
        assert sat["theme"] == "satellite"
        assert terr["theme"] == "terrain"
        assert street["theme"] == "street"
        assert custom["url_template"].startswith("https://tiles")

    def test_label_and_surround_helpers(self):
        multi = gp.multi_language_label_support([{"name_en": "River", "name_es": "Río"}], preferred_languages=["es", "en"])
        rtl = gp.right_to_left_label_support("مرحبا")
        cjk = gp.cjk_label_support("東京")
        expr = gp.label_expression_engine({"name": "A", "value": 5}, "{name}: {value}")
        classes = gp.label_class_management([{"where": "POP > 1000"}])
        maplex = gp.maplex_style_label_placement_engine([{"x": 0, "y": 0, "text": "A"}])
        ann = gp.annotation_conversion_label_to_annotation([{"text": "A", "x": 1, "y": 2}])
        dims = gp.dimension_lines([(0, 0), (3, 4)])
        surround = gp.map_surround_elements([{"type": "logo"}, {"type": "table"}])
        chart = gp.dynamic_chart_surround([1, 2, 3])
        grid = gp.grid_graticule_labelling((0, 0, 10, 10), interval=5)
        assert multi[0]["label"] == "Río"
        assert rtl["direction"] == "rtl"
        assert cjk["script"] == "cjk"
        assert expr == "A: 5"
        assert classes["class_count"] == 1
        assert maplex[0]["placement"] == "best"
        assert ann[0]["geometry"]["type"] == "Point"
        assert dims["distance"] == 5.0
        assert surround["element_count"] == 2
        assert chart["type"] == "chart-surround"
        assert len(grid["labels"]) > 0
