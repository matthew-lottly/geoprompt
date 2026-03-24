"""Smoke tests for backfill tools 169, 171 and new tools 523-530."""

import pytest

from geoprompt import GeoPromptFrame


@pytest.fixture()
def grid_frame():
    """4x4 grid with elevation, value, income, education, time columns."""
    records = []
    for idx in range(16):
        x = float(idx % 4)
        y = float(idx // 4)
        records.append({
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "site_id": f"s{idx}",
            "elevation": 100.0 + 10.0 * x - 5.0 * y + (idx % 3),
            "val": 2.0 * x + 3.0 * y + (idx % 5),
            "cases": float(5 + (idx % 4)),
            "pop": 100.0,
            "income": 1000.0 + 200.0 * x,
            "education": 8.0 + 0.5 * y,
            "time_val": float(idx % 3),
            "supply": 3.0 + (idx % 4),
            "capacity": 10.0,
        })
    return GeoPromptFrame.from_records(records, crs="EPSG:4326")


# ---- Tool 169: stream_power_index ----

def test_stream_power_index_smoke(grid_frame):
    result = grid_frame.stream_power_index("elevation")
    recs = result.to_records()
    assert len(recs) == 16
    assert "spi_spi" in recs[0]
    assert "contributing_area_spi" in recs[0]
    assert "slope_rad_spi" in recs[0]


def test_stream_power_index_nonnegative(grid_frame):
    result = grid_frame.stream_power_index("elevation")
    for r in result.to_records():
        assert r["spi_spi"] >= 0


# ---- Tool 171: network_betweenness ----

def test_network_betweenness_smoke(grid_frame):
    result = grid_frame.network_betweenness(k=4)
    recs = result.to_records()
    assert len(recs) == 16
    assert "betweenness_btw" in recs[0]
    assert "betweenness_norm_btw" in recs[0]


def test_network_betweenness_norm_range(grid_frame):
    result = grid_frame.network_betweenness(k=4)
    for r in result.to_records():
        assert 0.0 <= r["betweenness_norm_btw"] <= 1.0 + 1e-9


# ---- Tool 523: topologically_regularized_spatial_scan ----

def test_topo_scan_smoke(grid_frame):
    result = grid_frame.topologically_regularized_spatial_scan("cases", "pop", n_simulations=9, seed=42)
    recs = result.to_records()
    assert len(recs) == 16
    assert "in_cluster_trscan" in recs[0]
    assert "llr_trscan" in recs[0]
    assert "topo_score_trscan" in recs[0]
    assert "p_value_trscan" in recs[0]


def test_topo_scan_has_cluster(grid_frame):
    result = grid_frame.topologically_regularized_spatial_scan("cases", "pop", n_simulations=9, seed=42)
    in_cluster = [r["in_cluster_trscan"] for r in result.to_records()]
    assert any(in_cluster), "Expected at least one feature in cluster"


# ---- Tool 524: graph_coupled_space_time_regionalization ----

def test_gstr_smoke(grid_frame):
    result = grid_frame.graph_coupled_space_time_regionalization(
        value_columns=["val", "income"],
        time_column="time_val",
        n_regions=3,
        seed=42,
    )
    recs = result.to_records()
    assert len(recs) == 16
    assert "region_gstr" in recs[0]
    assert "region_size_gstr" in recs[0]


def test_gstr_region_count(grid_frame):
    result = grid_frame.graph_coupled_space_time_regionalization(
        value_columns=["val"],
        time_column="time_val",
        n_regions=4,
        seed=42,
    )
    labels = {r["region_gstr"] for r in result.to_records()}
    assert len(labels) <= 4


# ---- Tool 525: anisotropic_kernel_density ----

def test_akde_smoke(grid_frame):
    result = grid_frame.anisotropic_kernel_density(value_column="val", grid_resolution=5)
    recs = result.to_records()
    assert len(recs) == 16
    assert "density_akde" in recs[0]
    assert "angle_deg_akde" in recs[0]
    assert "h_major_akde" in recs[0]


def test_akde_density_positive(grid_frame):
    result = grid_frame.anisotropic_kernel_density(value_column="val", grid_resolution=5)
    for r in result.to_records():
        assert r["density_akde"] >= 0


# ---- Tool 526: spatial_durbin_error_model ----

def test_sdem_smoke(grid_frame):
    result = grid_frame.spatial_durbin_error_model(
        dependent_column="val",
        independent_columns=["income", "education"],
        k=4,
    )
    recs = result.to_records()
    assert len(recs) == 16
    assert "predicted_sdem" in recs[0]
    assert "residual_sdem" in recs[0]
    assert "lambda_sdem" in recs[0]
    assert "coeff_income_sdem" in recs[0]
    assert "lag_coeff_income_sdem" in recs[0]


# ---- Tool 527: wavelet_spatial_autocorrelation ----

def test_wsa_smoke(grid_frame):
    result = grid_frame.wavelet_spatial_autocorrelation("val", n_scales=3)
    recs = result.to_records()
    assert len(recs) == 16
    assert "wavelet_s0_wsa" in recs[0]
    assert "autocorr_s0_wsa" in recs[0]
    assert "dominant_scale_wsa" in recs[0]
    assert "dominant_autocorr_wsa" in recs[0]


# ---- Tool 528: multiscale_getis_ord ----

def test_msgi_smoke(grid_frame):
    result = grid_frame.multiscale_getis_ord("val", n_scales=3)
    recs = result.to_records()
    assert len(recs) == 16
    assert "gi_z_s0_msgi" in recs[0]
    assert "optimal_scale_idx_msgi" in recs[0]
    assert "optimal_gi_z_msgi" in recs[0]


def test_msgi_with_explicit_scales(grid_frame):
    result = grid_frame.multiscale_getis_ord("val", scales=[0.5, 1.0, 2.0])
    recs = result.to_records()
    assert "gi_z_s0_msgi" in recs[0]
    assert "gi_z_s2_msgi" in recs[0]


# ---- Tool 529: network_constrained_clustering ----

def test_ncc_smoke(grid_frame):
    result = grid_frame.network_constrained_clustering(
        "val", k=4, n_clusters=3, seed=42,
    )
    recs = result.to_records()
    assert len(recs) == 16
    assert "cluster_ncc" in recs[0]
    assert "medoid_ncc" in recs[0]
    assert "dist_to_medoid_ncc" in recs[0]


def test_ncc_cluster_count(grid_frame):
    result = grid_frame.network_constrained_clustering(
        "val", k=4, n_clusters=3, seed=42,
    )
    labels = {r["cluster_ncc"] for r in result.to_records()}
    assert len(labels) <= 3


# ---- Tool 530: spatial_envelope_anomaly ----

def test_sea_smoke(grid_frame):
    result = grid_frame.spatial_envelope_anomaly(
        value_columns=["val", "income", "education"],
        k=4,
        contamination=0.1,
    )
    recs = result.to_records()
    assert len(recs) == 16
    assert "anomaly_score_sea" in recs[0]
    assert "is_anomaly_sea" in recs[0]


def test_sea_anomaly_flags_present(grid_frame):
    result = grid_frame.spatial_envelope_anomaly(
        value_columns=["val", "income"],
        k=4,
        contamination=0.25,
    )
    flags = [r["is_anomaly_sea"] for r in result.to_records()]
    assert any(f == 1 for f in flags), "Expected at least one anomaly"


# ---- Parametrized smoke test for all tools ----

TOOL_CASES = [
    {
        "method": "stream_power_index",
        "args": ("elevation",),
        "kwargs": {},
        "cols": ["spi_spi"],
        "len": 16,
    },
    {
        "method": "network_betweenness",
        "args": (),
        "kwargs": {"k": 4},
        "cols": ["betweenness_btw", "betweenness_norm_btw"],
        "len": 16,
    },
    {
        "method": "topologically_regularized_spatial_scan",
        "args": ("cases",),
        "kwargs": {"population_column": "pop", "n_simulations": 5, "seed": 1},
        "cols": ["in_cluster_trscan", "llr_trscan", "topo_score_trscan", "p_value_trscan"],
        "len": 16,
    },
    {
        "method": "graph_coupled_space_time_regionalization",
        "args": (["val", "income"], "time_val"),
        "kwargs": {"n_regions": 3, "seed": 1},
        "cols": ["region_gstr", "region_size_gstr"],
        "len": 16,
    },
    {
        "method": "anisotropic_kernel_density",
        "args": (),
        "kwargs": {"value_column": "val", "grid_resolution": 4},
        "cols": ["density_akde", "angle_deg_akde"],
        "len": 16,
    },
    {
        "method": "spatial_durbin_error_model",
        "args": ("val", ["income", "education"]),
        "kwargs": {"k": 4},
        "cols": ["predicted_sdem", "lambda_sdem", "coeff_income_sdem", "lag_coeff_income_sdem"],
        "len": 16,
    },
    {
        "method": "wavelet_spatial_autocorrelation",
        "args": ("val",),
        "kwargs": {"n_scales": 2},
        "cols": ["wavelet_s0_wsa", "dominant_scale_wsa"],
        "len": 16,
    },
    {
        "method": "multiscale_getis_ord",
        "args": ("val",),
        "kwargs": {"n_scales": 3},
        "cols": ["gi_z_s0_msgi", "optimal_scale_idx_msgi"],
        "len": 16,
    },
    {
        "method": "network_constrained_clustering",
        "args": ("val",),
        "kwargs": {"k": 4, "n_clusters": 3, "seed": 1},
        "cols": ["cluster_ncc", "medoid_ncc"],
        "len": 16,
    },
    {
        "method": "spatial_envelope_anomaly",
        "args": (["val", "income"],),
        "kwargs": {"k": 4},
        "cols": ["anomaly_score_sea", "is_anomaly_sea"],
        "len": 16,
    },
]


@pytest.mark.parametrize(
    "case",
    TOOL_CASES,
    ids=[c["method"] for c in TOOL_CASES],
)
def test_parametrized_smoke(grid_frame, case):
    method = getattr(grid_frame, case["method"])
    result = method(*case.get("args", ()), **case.get("kwargs", {}))
    records = result.to_records()
    assert len(records) == case["len"]
    for col in case["cols"]:
        assert col in records[0], f"Missing column {col}"
