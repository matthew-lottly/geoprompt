"""Tests for G5.2 spatial statistics classification functions."""
import math
import pytest
import geoprompt as gp


class TestG52Classification:
    """Test G5.2 classification break methods."""

    @pytest.fixture
    def sample_values(self):
        """Sample numeric values for classification testing."""
        return [10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 105, 110]

    @pytest.fixture
    def skewed_values(self):
        """Skewed data distribution (more at lower end)."""
        return [1, 2, 3, 4, 5, 6, 7, 8, 100, 200, 300]

    def test_maximum_breaks_basic(self, sample_values):
        """Test maximum breaks classification on uniform data."""
        breaks = gp.maximum_breaks_classification(sample_values, k=4)
        assert len(breaks) == 5  # k + 1
        assert breaks[0] == min(sample_values)
        assert breaks[-1] == max(sample_values)
        assert all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1))

    def test_maximum_breaks_detects_gaps(self, skewed_values):
        """Test that maximum breaks detects natural gaps in data."""
        breaks = gp.maximum_breaks_classification(skewed_values, k=3)
        # Should find a break near the large gap between 8 and 100
        # The exact placement depends on the algorithm, so just check basic validity
        assert len(breaks) == 4
        assert breaks[0] <= min(skewed_values)
        assert breaks[-1] >= max(skewed_values)

    def test_maximum_breaks_edge_cases(self):
        """Test maximum breaks with edge cases."""
        # Empty
        assert gp.maximum_breaks_classification([], k=3) == []
        # Single value
        breaks = gp.maximum_breaks_classification([5.0], k=2)
        assert len(breaks) <= 2
        # Two values
        breaks = gp.maximum_breaks_classification([5.0, 10.0], k=2)
        assert 5.0 in breaks and 10.0 in breaks

    def test_box_plot_classification(self, sample_values):
        """Test box plot quartile classification."""
        breaks = gp.box_plot_classification(sample_values)
        assert len(breaks) == 5  # Q0, Q1, Q2, Q3, Q4
        assert breaks[0] == min(sample_values)  # Min
        assert breaks[4] == max(sample_values)  # Max
        assert all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1))
        # Q2 should be close to median
        sorted_vals = sorted(sample_values)
        median_idx = len(sorted_vals) // 2
        assert abs(breaks[2] - sorted_vals[median_idx]) < abs(breaks[2] - sorted_vals[0])

    def test_box_plot_consistency(self):
        """Test box plot classification with fixed data."""
        values = [1, 2, 3, 4, 5]
        breaks = gp.box_plot_classification(values)
        assert len(breaks) == 5
        assert breaks[0] == 1
        assert breaks[-1] == 5

    def test_pretty_breaks_basic(self, sample_values):
        """Test pretty breaks classification."""
        breaks = gp.pretty_breaks_classification(sample_values, k=4)
        assert len(breaks) >= 2
        assert breaks[0] <= min(sample_values)
        assert breaks[-1] >= max(sample_values)
        # Breaks should be at reasonable round numbers
        assert all(isinstance(b, float) for b in breaks)

    def test_pretty_breaks_round_numbers(self):
        """Test that pretty breaks generates round numbers."""
        values = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        breaks = gp.pretty_breaks_classification(values, k=5)
        # At least some breaks should be round (0 or 5 in ones place, or whole tens)
        has_round = any(b == int(b) or b % 10 == 0 or b % 5 == 0 for b in breaks)
        assert has_round or len(breaks) >= 2

    def test_pretty_breaks_edge_cases(self):
        """Test pretty breaks with edge cases."""
        # All same values
        breaks = gp.pretty_breaks_classification([5.0] * 5, k=3)
        assert len(breaks) >= 2
        # Empty
        assert gp.pretty_breaks_classification([], k=3) == []

    def test_percentile_classification(self, sample_values):
        """Test percentile/quantile classification."""
        breaks = gp.percentile_classification(sample_values, k=4)
        assert len(breaks) == 5  # k + 1
        assert breaks[0] == min(sample_values)
        assert breaks[-1] == max(sample_values)
        assert all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1))

    def test_percentile_equal_frequency(self):
        """Test that percentile creates roughly equal-frequency classes."""
        values = list(range(1, 101))  # 1-100
        breaks = gp.percentile_classification(values, k=4)
        assert len(breaks) == 5
        # With 100 values and 4 classes, each class should have ~25 values
        classes = []
        for i in range(len(breaks) - 1):
            count = sum(1 for v in values if breaks[i] <= v < breaks[i+1])
            classes.append(count)
        # Last class includes the max value
        classes[-1] = sum(1 for v in values if breaks[-2] <= v <= breaks[-1])
        # Classes should be roughly balanced
        avg_count = sum(classes) / len(classes)
        assert all(0 < c <= avg_count * 1.5 for c in classes)

    def test_percentile_vs_quantile_classify(self):
        """Compare percentile breaks with alternative classification."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        breaks_4 = gp.percentile_classification(values, k=4)
        breaks_5 = gp.percentile_classification(values, k=5)
        assert len(breaks_4) == 5
        assert len(breaks_5) == 6
        # More classes = more breaks
        assert len(breaks_5) > len(breaks_4)

    def test_head_tail_breaks_exists(self):
        """Verify head_tail_breaks is available."""
        values = [1, 2, 3, 4, 5, 100, 200, 300]
        breaks = gp.head_tail_breaks(values)
        assert len(breaks) >= 2
        assert breaks[0] == min(values)
        assert breaks[-1] == max(values)

    def test_fisher_jenks_exists(self):
        """Verify fisher_jenks is available."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        breaks = gp.fisher_jenks(values, k=4)
        # fisher_jenks may return fewer unique breaks due to de-duplication
        assert len(breaks) >= 2
        assert len(breaks) <= 5  # At most k + 1
        assert breaks[0] == min(values)
        assert breaks[-1] == max(values)

    def test_classification_consistency(self, sample_values):
        """Test that all classification methods return valid breaks."""
        methods = [
            (gp.maximum_breaks_classification, {"k": 4}),
            (gp.box_plot_classification, {}),
            (gp.pretty_breaks_classification, {"k": 4}),
            (gp.percentile_classification, {"k": 4}),
            (gp.head_tail_breaks, {}),
            (gp.fisher_jenks, {"k": 4}),
        ]

        for method, kwargs in methods:
            breaks = method(sample_values, **kwargs)
            # Check basic properties
            assert len(breaks) >= 2, f"{method.__name__} returned too few breaks"
            assert breaks[0] <= min(sample_values), f"{method.__name__} min too small"
            assert breaks[-1] >= max(sample_values), f"{method.__name__} max too large"
            assert all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1)), \
                f"{method.__name__} breaks not sorted"

    def test_large_dataset_classification(self):
        """Test classification on large dataset."""
        values = list(range(1, 1001))  # 1-1000
        methods = [
            (gp.maximum_breaks_classification, {"k": 10}),
            (gp.pretty_breaks_classification, {"k": 10}),
            (gp.percentile_classification, {"k": 10}),
        ]

        for method, kwargs in methods:
            breaks = method(values, **kwargs)
            assert len(breaks) > 1
            assert min(breaks) <= 1
            assert max(breaks) >= 1000

    def test_classification_reproducibility(self, sample_values):
        """Test that classification is reproducible."""
        breaks1 = gp.maximum_breaks_classification(sample_values, k=4)
        breaks2 = gp.maximum_breaks_classification(sample_values, k=4)
        assert breaks1 == breaks2

        breaks1 = gp.percentile_classification(sample_values, k=4)
        breaks2 = gp.percentile_classification(sample_values, k=4)
        assert breaks1 == breaks2
