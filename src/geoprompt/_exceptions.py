"""Improved exception handling for GeoPrompt.

Provides fallback-aware exception handling that tracks when operations
degrade to fallbacks, enabling better debugging and user awareness.

Usage:
    from ._exceptions import safe_operation, FallbackWarning
    
    result = safe_operation(
        expensive_operation,
        args=(data,),
        fallback_result=default_result,
        warn_on_fallback=True,
        operation_name="compute_kriging"
    )
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class GeoPromptError(Exception):
    """Base exception for GeoPrompt."""
    pass


class ParameterError(GeoPromptError):
    """Invalid function parameters."""
    pass


class DataError(GeoPromptError):
    """Malformed or invalid input data."""
    pass


class ImplementationError(GeoPromptError):
    """Software bug or unimplemented feature."""
    pass


class ResourceError(GeoPromptError):
    """Out of memory or resource exhaustion."""
    pass


class DependencyError(GeoPromptError):
    """Missing optional dependency."""
    pass


class FallbackWarning(UserWarning):
    """Warning issued when operation degraded to fallback implementation."""
    pass


class OperationMetadata:
    """Metadata about an operation execution.
    
    Attributes:
        result: The result value
        is_fallback: Whether result came from fallback (True) or primary (False)
        reason: Human-readable reason if fallback was used
        exception: Original exception if fallback was triggered
        operation_name: Name of operation for logging
    """
    
    def __init__(
        self,
        result: Any,
        *,
        is_fallback: bool = False,
        reason: str | None = None,
        exception: Exception | None = None,
        operation_name: str | None = None,
    ):
        self.result = result
        self.is_fallback = is_fallback
        self.reason = reason
        self.exception = exception
        self.operation_name = operation_name
    
    def __repr__(self) -> str:
        status = "FALLBACK" if self.is_fallback else "SUCCESS"
        reason_str = f" ({self.reason})" if self.reason else ""
        return f"OperationMetadata({status}{reason_str})"


def safe_operation(
    func: Callable[..., T],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    fallback_result: T | None = None,
    warn_on_fallback: bool = True,
    operation_name: str | None = None,
    exception_types: tuple[type[Exception], ...] = (Exception,),
) -> OperationMetadata:
    """Execute operation with fallback handling.
    
    Args:
        func: Callable to execute
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        fallback_result: Result to return if func raises exception
        warn_on_fallback: If True, issue FallbackWarning when fallback triggered
        operation_name: Name for logging/warnings (defaults to func.__name__)
        exception_types: Tuple of exception types to catch (default: all)
    
    Returns:
        OperationMetadata with result and status information
    
    Example:
        >>> result_meta = safe_operation(
        ...     expensive_kriging,
        ...     args=(points, values),
        ...     fallback_result=idw_result,
        ...     operation_name="kriging_interpolation"
        ... )
        >>> if result_meta.is_fallback:
        ...     print(f"Used fallback: {result_meta.reason}")
    """
    if kwargs is None:
        kwargs = {}
    
    op_name = operation_name or getattr(func, "__name__", "unknown_operation")
    
    try:
        result = func(*args, **kwargs)
        return OperationMetadata(
            result,
            is_fallback=False,
            operation_name=op_name,
        )
    except exception_types as e:
        if warn_on_fallback:
            reason = f"degraded to fallback: {type(e).__name__}: {str(e)[:100]}"
            warnings.warn(
                f"{op_name}: {reason}",
                FallbackWarning,
                stacklevel=2,
            )
        else:
            reason = f"degraded to fallback: {type(e).__name__}"
        
        return OperationMetadata(
            fallback_result,
            is_fallback=True,
            reason=reason,
            exception=e,
            operation_name=op_name,
        )


def validate_parameter(value: Any, *, param_name: str, expected_type: type | tuple[type, ...]) -> None:
    """Validate a parameter's type.
    
    Args:
        value: Value to validate
        param_name: Parameter name (for error message)
        expected_type: Expected type(s)
    
    Raises:
        ParameterError: If type doesn't match
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_str = " or ".join(t.__name__ for t in expected_type)
        else:
            type_str = expected_type.__name__
        raise ParameterError(
            f"Parameter '{param_name}' must be {type_str}, got {type(value).__name__}"
        )


def validate_range(value: float | int, *, param_name: str, min_val: float | None = None, max_val: float | None = None) -> None:
    """Validate a parameter is within range.
    
    Args:
        value: Numeric value to validate
        param_name: Parameter name (for error message)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    
    Raises:
        ParameterError: If value outside range
    """
    if min_val is not None and value < min_val:
        raise ParameterError(
            f"Parameter '{param_name}' must be >= {min_val}, got {value}"
        )
    if max_val is not None and value > max_val:
        raise ParameterError(
            f"Parameter '{param_name}' must be <= {max_val}, got {value}"
        )


def validate_not_empty(sequence: Any, *, param_name: str) -> None:
    """Validate a sequence is not empty.
    
    Args:
        sequence: Sequence to check
        param_name: Parameter name (for error message)
    
    Raises:
        ParameterError: If sequence is empty
    """
    if not sequence:
        raise ParameterError(f"Parameter '{param_name}' cannot be empty")
