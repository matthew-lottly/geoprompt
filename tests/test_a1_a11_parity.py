from pathlib import Path

import geoprompt as gp


A1_FUNCTIONS = [
    "true_curve_storage",
    "multi_patch_3d_solid_geometry",
    "extrude_polygon_to_3d_block",
    "intersection_3d",
    "difference_3d",
    "buffer_3d_sphere",
    "near_3d_distance",
    "skyline_analysis",
    "line_of_sight_analysis",
    "shadow_volume_computation",
    "cross_section_profile_from_3d_line",
    "surface_volume_cut_fill",
    "contour_polygon_fill_generation",
]

A11_FUNCTIONS = [
    "ogc_geopackage_compliance",
    "ogc_api_features_implementation",
    "ogc_api_processes_implementation",
    "ogc_api_records_implementation",
    "ogc_api_tiles_implementation",
    "ogc_api_maps_implementation",
    "ogc_wfs_client",
    "ogc_wms_client",
    "ogc_wmts_client",
    "ogc_wps_client",
    "ogc_csw_client",
    "ogc_sensorthings_api_client",
    "ogc_sos_client",
    "ogc_sld_se_symbology_read",
    "ogc_sld_se_symbology_write",
    "iso_19139_xml_metadata",
    "stac_catalogue_support_write_publish",
    "cog_best_practices",
    "cloud_native_geoprocessing_patterns",
    "zarr_store_read_write",
    "kerchunk_reference_file_support",
    "xarray_integration",
    "rioxarray_integration",
    "dask_geopandas_bridge",
    "dask_array_raster_bridge",
    "spark_spatial_bridge_sedona",
    "bigquery_gis_bridge",
    "snowflake_geospatial_bridge",
    "databricks_mosaic_bridge",
    "arcgis_online_rest_api_read",
    "arcgis_online_rest_api_publish",
    "arcgis_enterprise_rest_api",
    "arcgis_pro_project_file_read",
    "qgis_project_file_read",
    "mapinfo_workspace_file_read",
    "google_earth_engine_bridge",
    "microsoft_planetary_computer_bridge",
    "aws_earth_search_bridge",
    "copernicus_data_space_bridge",
    "usgs_earth_explorer_bridge",
    "nasa_earthdata_bridge",
    "mapbox_api_integration",
    "google_maps_platform_integration",
    "here_platform_integration",
    "tomtom_api_integration",
    "openrouteservice_integration",
    "osrm_integration",
    "valhalla_integration",
    "graphhopper_integration",
    "pelias_integration_geocoding",
    "census_bureau_api",
    "bls_api",
    "noaa_weather_api",
    "usgs_earthquake_api",
    "epa_facility_registry_api",
    "fda_recall_api_spatial_join",
    "faa_nasr_data",
    "nhd_dataset",
    "nlcd_dataset",
    "ssurgo_gssurgo_soil_data",
    "ned_3dep_elevation_data",
    "tiger_line_census_boundaries",
    "fema_nfhl_flood_data",
    "hifld_infrastructure_data",
    "gbif_species_occurrence_data",
    "ebird_observation_data",
    "who_health_data_api",
    "world_bank_indicators_api",
    "un_sdg_indicators",
    "humanitarian_data_exchange_hdx",
    "acled_conflict_data",
    "global_forest_watch_data",
    "global_fishing_watch_data",
    "geojson_ld_linked_data",
    "rdf_sparql_geo_queries",
    "geosparql_compliance",
    "schema_org_place_type_support",
    "json_fg_support",
    "cityjson_support",
    "ifc_bim_spatial_integration",
    "tiles_3d_support",
    "i3s_indexed_3d_scene_layers",
    "czml_export",
    "gltf_export",
    "usd_export",
    "maplibre_gl_style_spec_support",
    "mapbox_style_spec_support",
    "sld_to_maplibre_conversion",
    "qgis_style_to_maplibre_conversion",
    "arcgis_symbology_to_maplibre_conversion",
    "geopdf_export",
    "protobuf_spatial_types",
    "capn_proto_spatial_types",
    "flatbuffers_spatial_types",
    "cbor_geospatial_tags",
    "messagepack_spatial_extension",
    "avro_geospatial_schema",
    "thrift_geospatial_schema",
    "binder_jupyterhub_launch_config",
    "google_colab_compatibility_layer",
    "vscode_jupyter_integration_helpers",
    "streamlit_spatial_app_template",
    "panel_holoviz_spatial_app_template",
    "dash_plotly_spatial_app_template",
    "django_gis_integration_geodjango_bridge",
    "flask_based_gis_api_template",
    "serverless_spatial_api_template",
    "webhook_receiver_for_spatial_events",
    "mqtt_spatial_message_handler",
    "kafka_geo_event_consumer",
    "redis_geospatial_index_integration",
    "elasticsearch_geo_query_integration",
    "mongodb_geospatial_integration",
]


def test_a1_public_surface_and_geometry_helpers():
    for name in A1_FUNCTIONS:
        assert hasattr(gp, name), name

    arc = gp.true_curve_storage((0, 0), radius=5, start_angle=0, end_angle=90)
    solid = gp.multi_patch_3d_solid_geometry([{"ring": [(0, 0, 0), (1, 0, 0), (1, 1, 0)]}])
    block = gp.extrude_polygon_to_3d_block([(0, 0), (2, 0), (2, 2), (0, 2)], height=3)
    inter = gp.intersection_3d((0, 0, 0, 4, 4, 4), (2, 2, 2, 6, 6, 6))
    diff = gp.difference_3d((0, 0, 0, 4, 4, 4), (2, 2, 2, 6, 6, 6))
    sphere = gp.buffer_3d_sphere((0, 0, 0), radius=5)
    near = gp.near_3d_distance((0, 0, 0), [(1, 1, 1), (10, 10, 10)])
    skyline = gp.skyline_analysis([{"name": "A", "height": 10}, {"name": "B", "height": 15}], observer=(0, 0, 2))
    los = gp.line_of_sight_analysis((0, 0, 2), (10, 0, 2), [{"height": 1.0}, {"height": 1.5}])
    shadow = gp.shadow_volume_computation([(0, 0), (1, 0), (1, 1), (0, 1)], height=4, sun_altitude=30, sun_azimuth=135)
    section = gp.cross_section_profile_from_3d_line([(0, 0, 0), (3, 4, 5), (6, 8, 8)])
    cut_fill = gp.surface_volume_cut_fill([[1, 2], [2, 3]], [[2, 1], [3, 4]], cell_size=1.0)
    contours = gp.contour_polygon_fill_generation([0, 10, 20])

    assert arc["type"] == "ParametricArc" and solid["patch_count"] == 1
    assert block["height"] == 3 and inter["volume"] > 0 and diff["remaining_volume"] >= 0
    assert sphere["radius"] == 5 and near["distance"] > 0
    assert skyline[0]["name"] == "B" and los["visible"]
    assert shadow["shadow_length"] > 0 and len(section["profile"]) == 3
    assert cut_fill["net_volume"] != 0 and len(contours["bands"]) == 2


def test_a11_public_surface_presence():
    for name in A11_FUNCTIONS:
        assert hasattr(gp, name), name


def test_a11_ogc_and_cloud_interop_helpers(tmp_path):
    gpkg = gp.ogc_geopackage_compliance({"tables": ["features"]})
    features = gp.ogc_api_features_implementation([{"id": 1}])
    processes = gp.ogc_api_processes_implementation([{"id": "buffer"}])
    records = gp.ogc_api_records_implementation([{"id": "rec-1"}])
    tiles = gp.ogc_api_tiles_implementation(["0/0/0"])
    maps = gp.ogc_api_maps_implementation("demo")
    wfs = gp.ogc_wfs_client("https://example.com/wfs")
    sld = gp.ogc_sld_se_symbology_read("<StyledLayerDescriptor />")
    sld_path = gp.ogc_sld_se_symbology_write({"rules": 2}, tmp_path / "style.sld")
    iso_path = gp.iso_19139_xml_metadata({"title": "Demo"}, tmp_path / "meta.xml")
    stac = gp.stac_catalogue_support_write_publish({"id": "item-1"}, tmp_path / "stac.json")
    zarr = gp.zarr_store_read_write({"temperature": [1, 2]}, tmp_path / "store.zarr.json")
    kerchunk = gp.kerchunk_reference_file_support({"chunks": 4}, tmp_path / "refs.json")
    xarray = gp.xarray_integration({"band1": [1, 2]})
    dask = gp.dask_geopandas_bridge([{"id": 1}], partitions=2)
    spark = gp.spark_spatial_bridge_sedona([{"id": 1}])
    bigquery = gp.bigquery_gis_bridge("select 1")
    arcgis = gp.arcgis_online_rest_api_read("https://example.com/FeatureServer/0")
    gee = gp.google_earth_engine_bridge("LANDSAT")
    mapbox = gp.mapbox_api_integration("mapbox://styles/demo")
    pelias = gp.pelias_integration_geocoding("Seattle")

    assert gpkg["compliant"] and features["service"] == "OGC API - Features"
    assert processes["process_count"] == 1 and records["record_count"] == 1
    assert tiles["tile_count"] == 1 and maps["service"] == "OGC API - Maps"
    assert wfs["service"] == "WFS" and wfs["status"] == "configured" and sld["format"] == "SLD/SE"
    assert Path(sld_path).exists() and Path(iso_path).exists() and Path(stac["path"]).exists()
    assert Path(zarr["path"]).exists() and Path(kerchunk["path"]).exists()
    assert xarray["backend"] == "xarray" and dask["partitions"] == 2 and spark["engine"] == "sedona"
    assert bigquery["platform"] == "BigQuery GIS" and arcgis["platform"] == "ArcGIS Online"
    assert gee["catalog"] == "LANDSAT" and mapbox["provider"] == "Mapbox" and pelias["query"] == "Seattle"


def test_a11_data_sources_styles_and_app_templates(tmp_path):
    noaa = gp.noaa_weather_api("Seattle")
    usgs = gp.usgs_earthquake_api("7d")
    gbif = gp.gbif_species_occurrence_data("ursus arctos")
    hdx = gp.humanitarian_data_exchange_hdx("floods")
    geojson_ld = gp.geojson_ld_linked_data({"type": "FeatureCollection", "features": []})
    geosparql = gp.geosparql_compliance({"endpoint": "https://example.com/sparql"})
    cityjson = gp.cityjson_support({"CityObjects": {}})
    czml = gp.czml_export([{"id": "path"}], tmp_path / "scene.czml")
    gltf = gp.gltf_export({"meshes": []}, tmp_path / "scene.gltf")
    style = gp.maplibre_gl_style_spec_support({"version": 8})
    converted = gp.sld_to_maplibre_conversion({"rules": 3})
    binder = gp.binder_jupyterhub_launch_config("geoprompt/demo")
    colab = gp.google_colab_compatibility_layer()
    vscode = gp.vscode_jupyter_integration_helpers()
    streamlit = gp.streamlit_spatial_app_template("demo-app")
    webhook = gp.webhook_receiver_for_spatial_events("asset.updated")
    redis_geo = gp.redis_geospatial_index_integration([{"id": 1, "x": 1.0, "y": 2.0}])
    mongo = gp.mongodb_geospatial_integration([{"id": 1, "geometry": {"type": "Point", "coordinates": [1, 2]}}])

    assert noaa["source"] == "NOAA" and usgs["source"] == "USGS"
    assert gbif["dataset"] == "GBIF" and hdx["dataset"] == "HDX"
    assert geojson_ld["format"] == "GeoJSON-LD" and geosparql["compliant"]
    assert cityjson["format"] == "CityJSON"
    assert Path(czml).exists() and Path(gltf).exists()
    assert style["style_spec"] == "MapLibre" and converted["target"] == "MapLibre"
    assert binder["platform"] == "Binder" and colab["platform"] == "Colab" and vscode["platform"] == "VS Code"
    assert streamlit["framework"] == "Streamlit" and webhook["event"] == "asset.updated"
    assert redis_geo["index_type"] == "redis-geo" and mongo["backend"] == "MongoDB"
