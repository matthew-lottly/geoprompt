from pathlib import Path

import geoprompt as gp


def test_a2_advanced_crs_helpers(tmp_path):
    geotiff_meta = {"crs": "EPSG:32633", "width": 512, "height": 512}

    shifted = gp.datum_transformation_ntv2((10.0, 45.0), "EPSG:4326", "EPSG:4269")
    crs = gp.crs_from_geotiff_metadata(geotiff_meta)
    geoid = gp.geoid_height_interpolation(10.0, 45.0)
    egm = gp.egm_vertical_support(100.0, model="EGM2008")
    itrf = gp.itrf_frame_transformation((500000.0, 5100000.0), "ITRF2014", "ITRF2008", epoch=2025.0)
    nadcon = gp.nadcon_grid_shift((-74.0, 40.7))
    timed = gp.time_dependent_coordinate_operation((100.0, 200.0), (0.01, -0.02), years=5)
    dynamic = gp.dynamic_crs_epoch_support("EPSG:7899", epoch=2025.5)
    plate = gp.plate_motion_model_transformation((100.0, 200.0), (0.003, 0.004), years=10)
    decl = gp.geomagnetic_declination_lookup(-74.0, 40.7, date="2026-04-19")
    words = gp.what3words_encode(-74.0, 40.7)
    decoded = gp.what3words_decode(words)

    cache = gp.CRSRegistryCache()
    cache.add("EPSG:4326", {"name": "WGS 84"})
    path = gp.proj_grid_download_helper("ca_nrc_ntv2", tmp_path)
    cache_path = tmp_path / "crs_cache.json"
    cache.save(cache_path)
    loaded = gp.CRSRegistryCache.load(cache_path)

    assert len(shifted) == 2 and crs["name"] == "EPSG:32633"
    assert "offset_m" in geoid and egm["model"] == "EGM2008"
    assert itrf["target_frame"] == "ITRF2008" and len(nadcon["coordinates"]) == 2
    assert timed[0] != 100.0 and dynamic["epoch"] == 2025.5 and plate[1] > 200.0
    assert isinstance(decl["declination_deg"], float)
    assert words.count(".") == 2 and len(decoded) == 2
    assert cache.lookup("EPSG:4326")["name"] == "WGS 84"
    assert Path(path).exists() and loaded.lookup("EPSG:4326")["name"] == "WGS 84"


def test_a3_vector_and_service_format_bridges(tmp_path):
    table = [{"id": 1, "x": 1.0, "y": 2.0}, {"id": 2, "x": 3.0, "y": 4.0}]
    geoarrow_path = tmp_path / "sample.arrow.json"
    tab_path = tmp_path / "sample.tab"
    vt_path = tmp_path / "tiles.mvt.json"

    gp.write_geoarrow(table, geoarrow_path)
    gp.write_mapinfo_tab(table, tab_path)
    gp.write_vector_tiles(table, vt_path)

    mdb = gp.read_personal_geodatabase(table)
    geoarrow = gp.read_geoarrow(geoarrow_path)
    tab = gp.read_mapinfo_tab(tab_path)
    cad = gp.read_dxf_dwg([{"layer": "roads"}])
    tiger = gp.read_tiger_line_shapefiles([{"state": "NY"}])
    osm_pbf = gp.read_openstreetmap_pbf([{"osm_id": 1}])
    osm_xml = gp.read_openstreetmap_xml("<osm><node id='1' lat='1' lon='2'/></osm>")
    oracle = gp.read_oracle_spatial_layer(table)
    sqls = gp.read_sql_server_spatial_layer(table)
    wfs = gp.read_wfs_service({"features": table})
    wms = gp.read_wms_image_tiles("https://example.com/wms")
    wmts = gp.read_wmts_tiles("https://example.com/wmts")
    mapserver = gp.read_arcgis_rest_mapserver_image("https://example.com/MapServer")
    imageserver = gp.read_arcgis_rest_imageserver("https://example.com/ImageServer")
    ogc = gp.read_ogc_api_features({"features": table})
    stac = gp.read_stac_catalog_items({"features": [{"id": "scene-1"}]})
    pmtiles = gp.read_pmtiles({"tiles": [1, 2]})
    mbtiles = gp.read_mbtiles({"tiles": [1, 2, 3]})
    vector_tiles = gp.read_vector_tiles(vt_path)
    lazy = gp.lazy_reader_schema_only(table)
    vsi = gp.virtual_filesystem_path("https://example.com/data.geojson", scheme="vsicurl")
    s3 = gp.s3_bucket_reader({"features": table})
    azure = gp.azure_blob_reader({"features": table})
    gcs = gp.gcs_bucket_reader({"features": table})
    ftp = gp.ftp_reader({"features": table})

    assert len(mdb) == 2 and len(geoarrow) == 2 and len(tab) == 2
    assert cad["driver"] == "cad" and tiger[0]["state"] == "NY"
    assert osm_pbf[0]["osm_id"] == 1 and osm_xml[0]["type"] == "node"
    assert len(oracle) == 2 and len(sqls) == 2 and len(wfs) == 2
    assert wms["service"] == "WMS" and wmts["service"] == "WMTS"
    assert mapserver["kind"] == "MapServer" and imageserver["kind"] == "ImageServer"
    assert len(ogc) == 2 and stac[0]["id"] == "scene-1"
    assert pmtiles["tile_count"] == 2 and mbtiles["tile_count"] == 3 and len(vector_tiles) == 2
    assert set(lazy["fields"]) >= {"id", "x", "y"}
    assert vsi.startswith("/vsicurl/")
    assert len(s3) == 2 and len(azure) == 2 and len(gcs) == 2 and len(ftp) == 2


def test_a3_metadata_and_export_helpers(tmp_path):
    records = [{"id": 1, "name": "alpha"}, {"id": 2, "name": None}]
    metadata_path = tmp_path / "layer.metadata.json"
    proto_path = tmp_path / "sample.pb"
    pack_path = tmp_path / "sample.msgpack.json"
    out_geojson = tmp_path / "out.geojson"

    meta_written = gp.write_layer_metadata(metadata_path, {"description": "demo", "tags": ["a", "b"]})
    meta_read = gp.read_layer_metadata(metadata_path)
    sql = gp.sql_query_file_based_data(records, "select * where id >= 2")
    bigint = gp.bigint_field_handling(2**40)
    nulls = gp.null_value_handling_by_format(records, format_name="geojson")
    list_json = gp.list_json_field_types({"name": "demo", "tags": ["a", "b"], "meta": {"a": 1}})
    domain = gp.domain_coded_value_field_metadata("status", {1: "Open", 2: "Closed"})
    related = gp.geopackage_related_tables({"assets": [1, 2], "photos": [3]})
    attachments = gp.feature_attachment_support(records, attachments={1: ["a.jpg"]})
    oid = gp.oid_fid_management(records, start=10)
    tiles = gp.split_file_into_tiles(list(range(10)), tile_size=3)
    proto = gp.export_protocol_buffers(records, proto_path)
    pack = gp.export_messagepack(records, pack_path)
    sidecar = gp.crs_on_write_sidecar(out_geojson, "EPSG:4326")
    chunks = gp.max_file_size_split_on_write(list(range(8)), max_items=3)
    spatial_index = gp.spatial_index_on_write(records, index_type="qix")

    assert Path(meta_written).exists() and meta_read["description"] == "demo"
    assert len(sql) == 1 and bigint["type"] == "Int64"
    assert nulls[1]["name"] == "" and list_json["tags"] == "list"
    assert domain["field"] == "status" and related["relationship_count"] == 1
    assert attachments[0]["attachment_count"] == 1 and oid[0]["oid"] == 10
    assert len(tiles) == 4 and Path(proto).exists() and Path(pack).exists() and Path(sidecar).exists()
    assert len(chunks) == 3 and spatial_index["created"]


def test_a3_raster_pointcloud_and_reporting_helpers(tmp_path):
    raster_path = tmp_path / "sample.tif.json"
    nc_path = tmp_path / "sample.nc.json"
    asc_path = tmp_path / "sample.asc"
    las_path = tmp_path / "sample.las.json"

    raster_info = {"data": [[1, 2], [3, 4]], "crs": "EPSG:4326"}
    gp.write_cloud_optimized_geotiff(raster_info, raster_path)
    gp.write_netcdf({"temperature": [20, 21]}, nc_path)
    gp.write_esri_ascii_raster([[1, 2], [3, 4]], asc_path)
    gp.write_las_laz_point_cloud([{"x": 1, "y": 2, "z": 3}], las_path)

    geotiff = gp.read_geotiff_raster(raster_path)
    netcdf = gp.read_netcdf(nc_path)
    hdf = gp.read_hdf({"dataset": "band1"})
    ascii_raster = gp.read_esri_ascii_raster(asc_path)
    img = gp.read_img_raster(raster_info)
    jp2 = gp.read_jpeg2000_raster(raster_info)
    mrsid = gp.read_mrsid_raster(raster_info)
    ecw = gp.read_ecw_raster(raster_info)
    las = gp.read_las_laz_point_cloud(las_path)
    copc = gp.read_copc_point_cloud({"points": [{"x": 1}]})
    e57 = gp.read_e57_point_cloud({"points": [{"x": 1}, {"x": 2}]})
    stats = gp.raster_statistics([[1, 2], [3, 4]])
    pts = gp.raster_to_points([[1, 0], [0, 2]])
    contour = gp.raster_to_contour([[1, 2], [3, 4]], interval=1)
    sym = gp.color_ramp_symbology_export("viridis")
    legend = gp.legend_generation([{"label": "A", "color": "#f00"}])
    mapbook = gp.multi_page_map_book_export([{"title": "Page 1"}, {"title": "Page 2"}], tmp_path / "mapbook.txt")
    annotation = gp.annotation_layer_support([{"text": "Site", "x": 1, "y": 2}])
    h3code = gp.h3_index_encode(40.7, -74.0, resolution=7)
    decoded = gp.h3_index_decode(h3code)

    assert geotiff["crs"] == "EPSG:4326" and netcdf["temperature"][0] == 20
    assert hdf["dataset"] == "band1" and ascii_raster[0][0] == 1.0
    assert img["format"] == "IMG" and jp2["format"] == "JPEG2000"
    assert mrsid["format"] == "MrSID" and ecw["format"] == "ECW"
    assert len(las) == 1 and len(copc) == 1 and len(e57) == 2
    assert stats["mean"] == 2.5 and len(pts) == 2 and contour["interval"] == 1
    assert sym["ramp"] == "viridis" and legend["item_count"] == 1 and Path(mapbook).exists()
    assert annotation[0]["text"] == "Site" and decoded["resolution"] == 7
