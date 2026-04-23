"""Tests for audit improvements: tier system, deduplication, exception handling."""

import pytest
import warnings
from geoprompt._tier_metadata import (
    TIER_METADATA, TIER_STABLE, TIER_BETA, TIER_EXPERIMENTAL, TIER_SIMULATION,
    get_tier, warn_if_non_stable, tier_description, add_tier_info_to_docstring
)
from geoprompt._deduplication import (
    FUNCTION_DUPLICATES, get_canonical_function_name, get_duplicate_info, list_duplicates
)
from geoprompt._exceptions import (
    safe_operation, ParameterError, DataError, FallbackWarning, 
    validate_parameter, validate_range, validate_not_empty, OperationMetadata
)


class TestTierMetadata:
    """Test API tier system."""
    
    def test_tier_constants_defined(self):
        """Test that tier level constants exist."""
        assert TIER_STABLE == "stable"
        assert TIER_BETA == "beta"
        assert TIER_EXPERIMENTAL == "experimental"
        assert TIER_SIMULATION == "simulation_only"
    
    def test_tier_metadata_populated(self):
        """Test that tier metadata has entries."""
        assert len(TIER_METADATA) > 50
        assert "frame.GeoPromptFrame" in TIER_METADATA
        assert TIER_METADATA["frame.GeoPromptFrame"] == TIER_STABLE
    
    def test_get_tier_stable(self):
        """Test getting tier for stable function."""
        tier = get_tier("moran_i")
        assert tier == TIER_STABLE
    
    def test_get_tier_beta(self):
        """Test getting tier for beta function."""
        tier = get_tier("gwr")
        assert tier == TIER_BETA
    
    def test_get_tier_simulation(self):
        """Test getting tier for simulation function."""
        tier = get_tier("wms_capabilities_document")
        assert tier == TIER_SIMULATION
    
    def test_get_tier_not_found(self):
        """Test getting tier for unknown function."""
        tier = get_tier("nonexistent_function_xyz")
        assert tier is None
    
    def test_warn_if_non_stable_stable_no_warning(self):
        """Test that stable functions don't trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_non_stable("moran_i")
            assert len(w) == 0
    
    def test_warn_if_non_stable_beta_triggers_warning(self):
        """Test that beta functions trigger FutureWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_non_stable("gwr", stacklevel=1)
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "BETA" in str(w[0].message)
    
    def test_warn_if_non_stable_experimental_triggers_warning(self):
        """Test that experimental functions trigger FutureWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_non_stable("scale_analysis", stacklevel=1)
            assert len(w) == 1
            assert "EXPERIMENTAL" in str(w[0].message)
    
    def test_warn_if_non_stable_simulation_triggers_warning(self):
        """Test that simulation-only functions trigger UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_non_stable("wms_capabilities_document", stacklevel=1)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "SIMULATION" in str(w[0].message)
    
    def test_tier_description(self):
        """Test tier descriptions."""
        assert "Production-ready" in tier_description(TIER_STABLE)
        assert "subject to change" in tier_description(TIER_BETA)
        assert "development" in tier_description(TIER_EXPERIMENTAL)
        assert "Stub" in tier_description(TIER_SIMULATION)
    
    def test_add_tier_info_to_docstring(self):
        """Test adding tier info to docstring."""
        original = "Do something useful."
        updated = add_tier_info_to_docstring(original, TIER_BETA)
        assert "[BETA]" in updated
        assert "subject to change" in updated
        assert "Do something useful." in updated


class TestDeduplication:
    """Test deduplication tracking."""
    
    def test_duplicates_dict_populated(self):
        """Test that duplicates dictionary has entries."""
        assert len(FUNCTION_DUPLICATES) > 0
        assert "natural_neighbor_interpolation" in FUNCTION_DUPLICATES
    
    def test_get_canonical_function_name_already_canonical(self):
        """Test getting canonical name when input is already canonical."""
        canonical = get_canonical_function_name("natural_neighbor_interpolation")
        assert canonical == "natural_neighbor_interpolation"
    
    def test_get_canonical_function_name_from_alias(self):
        """Test getting canonical name from an alias."""
        # spatial_analysis.natural_neighbor_interpolation is an alias
        canonical = get_canonical_function_name("natural_neighbor_interpolation")
        assert canonical == "natural_neighbor_interpolation"
    
    def test_get_duplicate_info_found(self):
        """Test getting duplicate info for a known duplicate."""
        info = get_duplicate_info("natural_neighbor_interpolation")
        assert info is not None
        assert "canonical" in info
        assert "aliases" in info
    
    def test_get_duplicate_info_not_found(self):
        """Test getting duplicate info for non-duplicate."""
        info = get_duplicate_info("moran_i")
        assert info is None
    
    def test_list_duplicates_returns_dict(self):
        """Test listing all duplicates."""
        dupes = list_duplicates()
        assert isinstance(dupes, dict)
        assert "natural_neighbor_interpolation" in dupes
        assert isinstance(dupes["natural_neighbor_interpolation"], list)


class TestExceptionHandling:
    """Test improved exception handling."""
    
    def test_safe_operation_success(self):
        """Test safe_operation with successful function."""
        def add(a, b):
            return a + b
        
        result_meta = safe_operation(add, args=(2, 3))
        assert isinstance(result_meta, OperationMetadata)
        assert result_meta.result == 5
        assert result_meta.is_fallback is False
    
    def test_safe_operation_with_fallback(self):
        """Test safe_operation with exception and fallback."""
        def failing_func():
            raise ValueError("expected error")
        
        result_meta = safe_operation(
            failing_func,
            fallback_result=-1,
            warn_on_fallback=False,
        )
        assert result_meta.result == -1
        assert result_meta.is_fallback is True
        assert "ValueError" in result_meta.reason
    
    def test_safe_operation_fallback_warning(self):
        """Test that fallback triggers warning."""
        def failing_func():
            raise ValueError("test error")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_meta = safe_operation(
                failing_func,
                fallback_result=-1,
                warn_on_fallback=True,
                operation_name="test_op",
            )
            assert len(w) == 1
            assert issubclass(w[0].category, FallbackWarning)
            assert "test_op" in str(w[0].message)
            assert result_meta.is_fallback is True
    
    def test_safe_operation_preserves_exception(self):
        """Test that original exception is preserved in metadata."""
        original_error = ValueError("my error")
        def failing_func():
            raise original_error
        
        result_meta = safe_operation(
            failing_func,
            fallback_result=None,
            warn_on_fallback=False,
        )
        assert result_meta.exception is original_error
    
    def test_validate_parameter_success(self):
        """Test parameter validation succeeds for correct type."""
        validate_parameter(5, param_name="count", expected_type=int)  # Should not raise
    
    def test_validate_parameter_wrong_type(self):
        """Test parameter validation fails for wrong type."""
        with pytest.raises(ParameterError) as exc_info:
            validate_parameter("5", param_name="count", expected_type=int)
        assert "count" in str(exc_info.value)
        assert "int" in str(exc_info.value)
    
    def test_validate_parameter_multiple_types(self):
        """Test parameter validation with multiple allowed types."""
        validate_parameter(5, param_name="value", expected_type=(int, float))
        validate_parameter(5.5, param_name="value", expected_type=(int, float))
    
    def test_validate_range_success(self):
        """Test range validation succeeds within range."""
        validate_range(5, param_name="count", min_val=0, max_val=10)  # Should not raise
    
    def test_validate_range_too_small(self):
        """Test range validation fails when too small."""
        with pytest.raises(ParameterError) as exc_info:
            validate_range(5, param_name="count", min_val=10)
        assert "count" in str(exc_info.value)
    
    def test_validate_range_too_large(self):
        """Test range validation fails when too large."""
        with pytest.raises(ParameterError) as exc_info:
            validate_range(15, param_name="count", max_val=10)
        assert "count" in str(exc_info.value)
    
    def test_validate_not_empty_success(self):
        """Test empty validation succeeds for non-empty."""
        validate_not_empty([1, 2, 3], param_name="items")  # Should not raise
    
    def test_validate_not_empty_fails_on_empty_list(self):
        """Test empty validation fails for empty list."""
        with pytest.raises(ParameterError) as exc_info:
            validate_not_empty([], param_name="items")
        assert "items" in str(exc_info.value)
        assert "empty" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
