"""Adversarial and hardening tests for the safe expression engine.

Tests cover:
- Forbidden dunder / builtin names
- Max length enforcement
- Max node count enforcement
- Max depth enforcement
- Attribute escape attempts
- Disallowed function calls
- Valid expression baseline
"""

from __future__ import annotations

import pytest

from geoprompt.safe_expression import (
    ExpressionExecutionError,
    ExpressionValidationError,
    evaluate_safe_expression,
    get_expression_audit_metrics,
    reset_expression_audit_metrics,
)


# ---------------------------------------------------------------------------
# Baseline – valid expressions should evaluate correctly
# ---------------------------------------------------------------------------

class TestValidExpressions:
    def test_arithmetic(self):
        assert evaluate_safe_expression("2 + 3 * 4") == 14

    def test_comparison(self):
        assert evaluate_safe_expression("x > 5", {"x": 10}) is True

    def test_boolean_logic(self):
        assert evaluate_safe_expression("a and b", {"a": True, "b": True}) is True

    def test_allowlisted_function(self):
        assert evaluate_safe_expression("abs(-7)") == 7

    def test_string_variable(self):
        assert evaluate_safe_expression("name == 'alice'", {"name": "alice"}) is True

    def test_list_literal(self):
        result = evaluate_safe_expression("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_subscript(self):
        assert evaluate_safe_expression("items[0]", {"items": [10, 20]}) == 10


# ---------------------------------------------------------------------------
# Forbidden dunder / builtin name blocking (J1.9)
# ---------------------------------------------------------------------------

class TestForbiddenNames:
    @pytest.mark.parametrize("expr", [
        "__import__('os')",
        "__builtins__",
        "__class__",
        "__subclasses__()",
        "__dict__",
        "exec('pass')",
        "eval('1+1')",
        "compile('pass','<>','exec')",
        "open('/etc/passwd')",
    ])
    def test_forbidden_name_blocked(self, expr: str):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression(expr, {})

    def test_dunder_prefix_blocked(self):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("__name__", {})


# ---------------------------------------------------------------------------
# Attribute escape blocking
# ---------------------------------------------------------------------------

class TestAttributeEscape:
    def test_arbitrary_attribute_blocked(self):
        """Accessing attributes on names not in allowed_attribute_roots must fail."""
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("x.dangerous_method()", {"x": object()})

    def test_private_attribute_on_allowed_root_blocked(self):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression(
                "row.__class__",
                {"row": {}},
                allowed_attribute_roots=["row"],
            )

    def test_allowed_attribute_access(self):
        """Attributes on declared roots must work."""
        result = evaluate_safe_expression(
            "row.value > 5",
            {"row": {"value": 10}},
            allowed_attribute_roots=["row"],
        )
        assert result is True


# ---------------------------------------------------------------------------
# Disallowed function calls
# ---------------------------------------------------------------------------

class TestDisallowedCalls:
    def test_print_blocked(self):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("print('hello')", {})

    def test_type_blocked(self):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("type(x)", {"x": 1})

    def test_getattr_blocked(self):
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("getattr(x, '__class__')", {"x": 1})

    def test_lambda_blocked(self):
        with pytest.raises((ExpressionValidationError, SyntaxError)):
            evaluate_safe_expression("lambda: None", {})


# ---------------------------------------------------------------------------
# Limit enforcement (J1.7)
# ---------------------------------------------------------------------------

class TestLimits:
    def test_max_length_blocked(self):
        long_expr = "1 + " * 1000 + "1"
        with pytest.raises(ExpressionValidationError, match="length"):
            evaluate_safe_expression(long_expr, {}, max_length=100)

    def test_max_length_custom_accepted(self):
        expr = "1 + 1"
        result = evaluate_safe_expression(expr, {}, max_length=20)
        assert result == 2

    def test_max_nodes_blocked(self):
        # Build a valid but very deep expression
        expr = " + ".join(str(i) for i in range(50))
        with pytest.raises(ExpressionValidationError, match="complex"):
            evaluate_safe_expression(expr, {}, max_nodes=10)

    def test_max_depth_blocked(self):
        # Build deeply nested addition expression
        expr = "1" + " + (1" * 12 + ")" * 12
        with pytest.raises(ExpressionValidationError, match="nested"):
            evaluate_safe_expression(expr, {}, max_depth=4)


# ---------------------------------------------------------------------------
# Execution error path
# ---------------------------------------------------------------------------

class TestExecutionErrors:
    def test_unknown_symbol_raises_execution_error(self):
        with pytest.raises(ExpressionExecutionError, match="unknown symbol"):
            evaluate_safe_expression("nonexistent_name > 0", {})

    def test_type_error_wrapped(self):
        with pytest.raises(ExpressionExecutionError):
            evaluate_safe_expression("'a' + 1", {})


class TestAuditMetricsAndCircuitBreaker:
    def test_metrics_track_module_validation_and_execution_errors(self):
        reset_expression_audit_metrics()

        assert evaluate_safe_expression("1 + 1", module="query") == 2
        with pytest.raises(ExpressionValidationError):
            evaluate_safe_expression("__import__('os')", module="query")
        with pytest.raises(ExpressionExecutionError):
            evaluate_safe_expression("unknown_name + 1", module="query")

        metrics = get_expression_audit_metrics()
        assert metrics["total_evaluated"] >= 3
        assert metrics["validation_errors"] >= 1
        assert metrics["execution_errors"] >= 1
        assert metrics["module_metrics"]["query"]["evaluated"] >= 3

    def test_caller_rejection_circuit_breaker_blocks_after_threshold(self):
        reset_expression_audit_metrics()

        caller = "127.0.0.1"
        for _ in range(2):
            with pytest.raises(ExpressionValidationError):
                evaluate_safe_expression(
                    "__import__('os')",
                    module="service",
                    caller_id=caller,
                    rejection_threshold=2,
                    rejection_window_seconds=600,
                )

        with pytest.raises(ExpressionValidationError, match="temporarily blocked"):
            evaluate_safe_expression(
                "1 + 1",
                module="service",
                caller_id=caller,
                rejection_threshold=2,
                rejection_window_seconds=600,
            )
