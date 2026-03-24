import math

import pytest

from geoprompt import (
    accessibility_index,
    directional_alignment,
    directional_bearing,
    gravity_model,
    min_max_scale,
    sigmoid,
    thin_plate_spline_basis,
    variogram_exponential,
    variogram_gaussian_model,
    variogram_spherical,
)


def test_gravity_model_rejects_nonpositive_friction_even_at_zero_distance() -> None:
    with pytest.raises(ValueError, match="friction must be greater than zero"):
        gravity_model(1.0, 1.0, distance_value=0.0, friction=0.0)


def test_accessibility_index_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        accessibility_index(weights=[1.0, 2.0], distances=[1.0])


def test_min_max_scale_is_linear_not_clamped() -> None:
    assert min_max_scale(5.0, 0.0, 10.0) == 0.5
    assert min_max_scale(12.0, 0.0, 10.0) == 1.2
    assert min_max_scale(-2.0, 0.0, 10.0) == -0.2


def test_min_max_scale_returns_zero_for_degenerate_range() -> None:
    assert min_max_scale(7.0, 3.0, 3.0) == 0.0


def test_variogram_spherical_matches_piecewise_definition() -> None:
    assert variogram_spherical(0.0, nugget=0.2, sill=1.3, range_param=4.0) == 0.0
    expected_mid = 0.2 + 1.3 * (1.5 * (2.0 / 4.0) - 0.5 * (2.0 / 4.0) ** 3)
    assert abs(variogram_spherical(2.0, nugget=0.2, sill=1.3, range_param=4.0) - expected_mid) < 1e-12
    assert variogram_spherical(4.0, nugget=0.2, sill=1.3, range_param=4.0) == 1.5
    assert variogram_spherical(5.0, nugget=0.2, sill=1.3, range_param=4.0) == 1.5


def test_variogram_exponential_matches_closed_form() -> None:
    expected = 0.1 + 2.0 * (1.0 - math.exp(-3.0 * 3.0 / 6.0))
    assert abs(variogram_exponential(3.0, nugget=0.1, sill=2.0, range_param=6.0) - expected) < 1e-12


def test_variogram_gaussian_matches_closed_form() -> None:
    expected = 0.1 + 2.0 * (1.0 - math.exp(-3.0 * (3.0 / 6.0) ** 2))
    assert abs(variogram_gaussian_model(3.0, nugget=0.1, sill=2.0, range_param=6.0) - expected) < 1e-12


def test_thin_plate_spline_basis_handles_zero_and_positive_radius() -> None:
    assert thin_plate_spline_basis(0.0) == 0.0
    expected = 2.0 * 2.0 * math.log(2.0)
    assert abs(thin_plate_spline_basis(2.0) - expected) < 1e-12


def test_sigmoid_is_bounded_and_symmetric() -> None:
    assert 0.0 < sigmoid(-1000.0) < 0.5
    assert 0.5 < sigmoid(1000.0) <= 1.0
    assert abs(sigmoid(1.75) + sigmoid(-1.75) - 1.0) < 1e-12


def test_directional_bearing_cardinal_directions() -> None:
    assert directional_bearing((0.0, 0.0), (0.0, 1.0)) == 0.0
    assert directional_bearing((0.0, 0.0), (1.0, 0.0)) == 90.0
    assert directional_bearing((0.0, 0.0), (0.0, -1.0)) == 180.0
    assert directional_bearing((0.0, 0.0), (-1.0, 0.0)) == 270.0


def test_directional_alignment_matches_cosine_of_bearing_offset() -> None:
    assert abs(directional_alignment((0.0, 0.0), (0.0, 1.0), preferred_bearing=0.0) - 1.0) < 1e-12
    assert abs(directional_alignment((0.0, 0.0), (1.0, 0.0), preferred_bearing=0.0)) < 1e-12
    assert abs(directional_alignment((0.0, 0.0), (0.0, -1.0), preferred_bearing=0.0) + 1.0) < 1e-12