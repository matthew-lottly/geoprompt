from __future__ import annotations

import ast
import logging
import operator
import time
from collections import deque
from typing import Any, Iterable

_logger = logging.getLogger(__name__)


class ExpressionValidationError(ValueError):
    """Raised when an expression contains unsupported syntax."""


class ExpressionExecutionError(ValueError):
    """Raised when a valid expression fails at runtime."""


_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "len": len,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

# Names that must never appear as identifiers in user-supplied expressions.
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "__import__",
    "__builtins__",
    "__globals__",
    "__locals__",
    "__class__",
    "__base__",
    "__subclasses__",
    "__mro__",
    "__init__",
    "__new__",
    "__dict__",
    "__code__",
    "__func__",
    "__closure__",
    "__reduce__",
    "__reduce_ex__",
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
})

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_ALLOWED_CMPOPS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_EXPRESSION_METRICS: dict[str, Any] = {
    "total_evaluated": 0,
    "validation_errors": 0,
    "execution_errors": 0,
    "module_metrics": {},
}

_REJECTION_WINDOW_SECONDS = 60
_REJECTION_THRESHOLD = 5
_CALLER_REJECTIONS: dict[str, deque[float]] = {}


def _metric_bucket(module: str) -> dict[str, int]:
    modules = _EXPRESSION_METRICS.setdefault("module_metrics", {})
    bucket = modules.setdefault(module, {"evaluated": 0, "validation_errors": 0, "execution_errors": 0})
    return bucket


def _record_metric(module: str, key: str) -> None:
    _EXPRESSION_METRICS[key] = int(_EXPRESSION_METRICS.get(key, 0)) + 1
    bucket = _metric_bucket(module)
    bucket[key if key != "total_evaluated" else "evaluated"] = int(bucket.get(key if key != "total_evaluated" else "evaluated", 0)) + 1


def _register_rejection(caller_id: str, now: float, *, window_seconds: int) -> int:
    history = _CALLER_REJECTIONS.setdefault(caller_id, deque())
    history.append(now)
    cutoff = now - window_seconds
    while history and history[0] < cutoff:
        history.popleft()
    return len(history)


def _is_caller_blocked(caller_id: str, now: float, *, window_seconds: int, threshold: int) -> bool:
    history = _CALLER_REJECTIONS.get(caller_id)
    if not history:
        return False
    cutoff = now - window_seconds
    while history and history[0] < cutoff:
        history.popleft()
    return len(history) >= threshold


def get_expression_audit_metrics() -> dict[str, Any]:
    """Return a snapshot of expression-evaluation metrics and rejection counters."""
    module_metrics = {
        name: {
            "evaluated": int(values.get("evaluated", 0)),
            "validation_errors": int(values.get("validation_errors", 0)),
            "execution_errors": int(values.get("execution_errors", 0)),
        }
        for name, values in dict(_EXPRESSION_METRICS.get("module_metrics", {})).items()
    }
    now = time.time()
    blocked_callers: dict[str, int] = {}
    for caller_id, history in list(_CALLER_REJECTIONS.items()):
        cutoff = now - _REJECTION_WINDOW_SECONDS
        while history and history[0] < cutoff:
            history.popleft()
        if history:
            blocked_callers[caller_id] = len(history)
    return {
        "total_evaluated": int(_EXPRESSION_METRICS.get("total_evaluated", 0)),
        "validation_errors": int(_EXPRESSION_METRICS.get("validation_errors", 0)),
        "execution_errors": int(_EXPRESSION_METRICS.get("execution_errors", 0)),
        "module_metrics": module_metrics,
        "recent_rejections_by_caller": blocked_callers,
        "rejection_window_seconds": _REJECTION_WINDOW_SECONDS,
        "rejection_threshold": _REJECTION_THRESHOLD,
    }


def reset_expression_audit_metrics() -> None:
    """Reset expression metrics and caller rejection history (for tests/admin tooling)."""
    _EXPRESSION_METRICS["total_evaluated"] = 0
    _EXPRESSION_METRICS["validation_errors"] = 0
    _EXPRESSION_METRICS["execution_errors"] = 0
    _EXPRESSION_METRICS["module_metrics"] = {}
    _CALLER_REJECTIONS.clear()


def _check_depth(node: ast.AST, *, max_depth: int, _depth: int = 0) -> None:
    if _depth > max_depth:
        raise ExpressionValidationError("expression is too deeply nested")
    for child in ast.iter_child_nodes(node):
        _check_depth(child, max_depth=max_depth, _depth=_depth + 1)


def _validate_tree(
    tree: ast.AST,
    *,
    max_nodes: int,
    max_depth: int,
    allowed_attribute_roots: set[str] | None = None,
) -> None:
    allowed_attribute_roots = allowed_attribute_roots or set()
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > max_nodes:
            raise ExpressionValidationError("expression is too complex")

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in allowed_attribute_roots:
                raise ExpressionValidationError("attribute access is restricted")
            if node.attr.startswith("_"):
                raise ExpressionValidationError("private/dunder attributes are not allowed")
            continue

        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ExpressionValidationError(f"name '{node.id}' is not allowed in expressions")
            continue

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCTIONS:
                    raise ExpressionValidationError(f"function '{node.func.id}' is not allowed")
            elif isinstance(node.func, ast.Attribute):
                if not isinstance(node.func.value, ast.Name) or node.func.value.id not in allowed_attribute_roots:
                    raise ExpressionValidationError("method calls are restricted")
                if node.func.attr.startswith("_"):
                    raise ExpressionValidationError("private methods are not allowed")
            else:
                raise ExpressionValidationError("unsupported call target")
            continue

        allowed = (
            ast.Expression,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
            ast.Subscript,
            ast.Slice,
            ast.Index,
            ast.And,
            ast.Or,
            ast.keyword,
        )
        if not isinstance(node, allowed + tuple(_ALLOWED_BINOPS) + tuple(_ALLOWED_UNARYOPS) + tuple(_ALLOWED_CMPOPS)):
            raise ExpressionValidationError("unsupported expression syntax")

    _check_depth(tree, max_depth=max_depth)


def _eval_node(
    node: ast.AST,
    context: dict[str, Any],
    *,
    allowed_attribute_roots: set[str] | None = None,
) -> Any:
    allowed_attribute_roots = allowed_attribute_roots or set()

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context, allowed_attribute_roots=allowed_attribute_roots)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]
        raise ExpressionExecutionError(f"unknown symbol '{node.id}'")
    if isinstance(node, ast.List):
        return [_eval_node(e, context, allowed_attribute_roots=allowed_attribute_roots) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, context, allowed_attribute_roots=allowed_attribute_roots) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_node(e, context, allowed_attribute_roots=allowed_attribute_roots) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _eval_node(k, context, allowed_attribute_roots=allowed_attribute_roots): _eval_node(v, context, allowed_attribute_roots=allowed_attribute_roots)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, context, allowed_attribute_roots=allowed_attribute_roots)
        idx = _eval_node(node.slice, context, allowed_attribute_roots=allowed_attribute_roots)
        return target[idx]
    if isinstance(node, ast.Slice):
        lower = _eval_node(node.lower, context, allowed_attribute_roots=allowed_attribute_roots) if node.lower else None
        upper = _eval_node(node.upper, context, allowed_attribute_roots=allowed_attribute_roots) if node.upper else None
        step = _eval_node(node.step, context, allowed_attribute_roots=allowed_attribute_roots) if node.step else None
        return slice(lower, upper, step)
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ExpressionValidationError("unsupported unary operator")
        return op(_eval_node(node.operand, context, allowed_attribute_roots=allowed_attribute_roots))
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionValidationError("unsupported binary operator")
        return op(
            _eval_node(node.left, context, allowed_attribute_roots=allowed_attribute_roots),
            _eval_node(node.right, context, allowed_attribute_roots=allowed_attribute_roots),
        )
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, context, allowed_attribute_roots=allowed_attribute_roots) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ExpressionValidationError("unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context, allowed_attribute_roots=allowed_attribute_roots)
        for op, comp in zip(node.ops, node.comparators):
            fn = _ALLOWED_CMPOPS.get(type(op))
            if fn is None:
                raise ExpressionValidationError("unsupported comparison operator")
            right = _eval_node(comp, context, allowed_attribute_roots=allowed_attribute_roots)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id not in allowed_attribute_roots:
            raise ExpressionValidationError("attribute access is restricted")
        obj = _eval_node(node.value, context, allowed_attribute_roots=allowed_attribute_roots)
        if node.attr.startswith("_"):
            raise ExpressionValidationError("private attributes are not allowed")
        if isinstance(obj, dict):
            return obj.get(node.attr)
        return getattr(obj, node.attr)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ExpressionValidationError(f"function '{node.func.id}' is not allowed")
        elif isinstance(node.func, ast.Attribute):
            fn = _eval_node(node.func, context, allowed_attribute_roots=allowed_attribute_roots)
            if not callable(fn):
                raise ExpressionExecutionError("attribute is not callable")
        else:
            raise ExpressionValidationError("unsupported function call")

        args = [_eval_node(a, context, allowed_attribute_roots=allowed_attribute_roots) for a in node.args]
        kwargs = {
            kw.arg: _eval_node(kw.value, context, allowed_attribute_roots=allowed_attribute_roots)
            for kw in node.keywords
            if kw.arg is not None
        }
        return fn(*args, **kwargs)

    raise ExpressionValidationError("unsupported expression node")


def evaluate_safe_expression(
    expression: str,
    context: dict[str, Any] | None = None,
    *,
    max_length: int = 2000,
    max_nodes: int = 120,
    max_depth: int = 16,
    allowed_attribute_roots: Iterable[str] | None = None,
    module: str = "unknown",
    caller_id: str | None = None,
    rejection_window_seconds: int = _REJECTION_WINDOW_SECONDS,
    rejection_threshold: int = _REJECTION_THRESHOLD,
) -> Any:
    """Validate and evaluate a restricted expression with no eval/exec.

    The evaluator supports arithmetic, comparisons, boolean logic, literals,
    indexing/slicing, and a small allowlist of functions (abs, len, min, max,
    round, int, float, str, bool).

    Forbidden: all dunder names, exec/eval/compile/open, attribute traversal
    beyond declared roots, function calls outside the allowlist, and any AST
    node that could be used to escape the sandbox.

    Args:
        expression: User-supplied expression string.
        context: Mapping of symbol names to values available in the expression.
        max_length: Maximum allowed character length of the expression string.
        max_nodes: Maximum number of AST nodes allowed in the parsed tree.
        max_depth: Maximum nesting depth of the AST.
        allowed_attribute_roots: Symbol names whose attributes may be accessed.
    """
    def _redact_expression(expr: str, max_display: int = 100) -> str:
        """Redact long expressions for logging."""
        if len(expr) > max_display:
            return expr[:max_display] + "..."
        return expr

    module_name = module.strip() if isinstance(module, str) and module.strip() else "unknown"
    _record_metric(module_name, "total_evaluated")

    inferred_caller = caller_id
    if inferred_caller is None and isinstance(context, dict):
        inferred_caller = str(context.get("__caller_id") or context.get("__caller_ip") or "").strip() or None

    now = time.time()
    if inferred_caller and _is_caller_blocked(
        inferred_caller,
        now,
        window_seconds=rejection_window_seconds,
        threshold=rejection_threshold,
    ):
        _logger.warning(
            "Rejected expression (caller circuit breaker): caller=%s module=%s",
            inferred_caller,
            module_name,
        )
        _record_metric(module_name, "validation_errors")
        raise ExpressionValidationError(
            "expression evaluations temporarily blocked for caller due to repeated validation failures"
        )

    if not isinstance(expression, str) or not expression.strip():
        error_msg = "expression must be a non-empty string"
        if inferred_caller:
            _register_rejection(inferred_caller, now, window_seconds=rejection_window_seconds)
        _record_metric(module_name, "validation_errors")
        _logger.warning("Rejected expression: %s (module=%s, caller=%s)", error_msg, module_name, inferred_caller or "unknown")
        raise ExpressionValidationError(error_msg)

    if len(expression) > max_length:
        error_msg = f"expression length {len(expression)} exceeds maximum {max_length}"
        if inferred_caller:
            _register_rejection(inferred_caller, now, window_seconds=rejection_window_seconds)
        _record_metric(module_name, "validation_errors")
        _logger.warning(
            "Rejected expression (length exceeded): %s (module=%s, caller=%s)",
            _redact_expression(expression),
            module_name,
            inferred_caller or "unknown",
        )
        raise ExpressionValidationError(error_msg)

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        if inferred_caller:
            _register_rejection(inferred_caller, now, window_seconds=rejection_window_seconds)
        _record_metric(module_name, "validation_errors")
        _logger.warning(
            "Rejected expression (syntax error): %s - %s (module=%s, caller=%s)",
            _redact_expression(expression),
            exc.msg,
            module_name,
            inferred_caller or "unknown",
        )
        raise ExpressionValidationError(f"invalid expression syntax: {exc.msg}") from exc

    roots = set(allowed_attribute_roots or [])
    try:
        _validate_tree(tree, max_nodes=max_nodes, max_depth=max_depth, allowed_attribute_roots=roots)
    except ExpressionValidationError as exc:
        if inferred_caller:
            count = _register_rejection(inferred_caller, now, window_seconds=rejection_window_seconds)
        else:
            count = None
        _record_metric(module_name, "validation_errors")
        _logger.warning(
            "Rejected expression (validation failed): %s - %s (module=%s, caller=%s, rejection_count=%s)",
            _redact_expression(expression),
            str(exc),
            module_name,
            inferred_caller or "unknown",
            count if count is not None else "n/a",
        )
        raise

    local_ctx = dict(context or {})
    for name, fn in _ALLOWED_FUNCTIONS.items():
        local_ctx.setdefault(name, fn)

    try:
        return _eval_node(tree, local_ctx, allowed_attribute_roots=roots)
    except ExpressionValidationError:
        _record_metric(module_name, "validation_errors")
        raise
    except Exception as exc:
        _record_metric(module_name, "execution_errors")
        _logger.warning(
            "Rejected expression (execution error): %s - %s (module=%s, caller=%s)",
            _redact_expression(expression),
            str(exc),
            module_name,
            inferred_caller or "unknown",
        )
        raise ExpressionExecutionError(str(exc)) from exc


__all__ = [
    "ExpressionExecutionError",
    "ExpressionValidationError",
    "evaluate_safe_expression",
    "get_expression_audit_metrics",
    "reset_expression_audit_metrics",
]
