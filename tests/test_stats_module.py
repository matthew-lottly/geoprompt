from __future__ import annotations

import pytest

from geoprompt.stats import morans_i, semivariogram, spatial_lag, spatial_weights_matrix


def test_stats_module_weights_and_lag_are_deterministic() -> None:
    centroids = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    values = [1.0, 2.0, 3.0]

    weights = spatial_weights_matrix(centroids, k=1, row_standardize=True)

    assert weights == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert spatial_lag(values, centroids, k=1) == pytest.approx([2.0, 1.0, 1.0])


def test_stats_module_global_metrics_and_semivariogram_shapes() -> None:
    centroids = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    values = [1.0, 2.0, 3.0]

    moran = morans_i(values, centroids, k=1)
    variogram = semivariogram(centroids, values, bins=2)

    assert moran["morans_i"] == pytest.approx(-0.5)
    assert moran["expected_i"] == pytest.approx(-0.5)
    assert moran["z_score"] == pytest.approx(0.0)
    assert len(variogram) == 1
    assert variogram[0]["bin"] == pytest.approx(2.0)
    assert variogram[0]["pair_count"] == pytest.approx(3.0)
    assert variogram[0]["mean_distance"] == pytest.approx(1.1380711874576983)
    assert variogram[0]["semivariance"] == pytest.approx(1.0)