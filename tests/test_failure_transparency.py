"""Failure transparency tests (J3.36).

These tests ensure failures are explicit and not silently downgraded to
success-like outputs.
"""

from __future__ import annotations

import pytest

from geoprompt import io
from geoprompt._exceptions import failure_payload
from geoprompt.db import _parse_wkt
from geoprompt.geoprocessing import notify_webhook
from geoprompt.safe_expression import ExpressionValidationError, evaluate_safe_expression


def test_safe_expression_rejects_forbidden_symbol_explicitly() -> None:
    with pytest.raises(ExpressionValidationError):
        evaluate_safe_expression("__import__('os').system('echo x')", {})


def test_read_cloud_json_rejects_disallowed_scheme_explicitly() -> None:
    with pytest.raises(ValueError, match="scheme"):
        io.read_cloud_json("ftp://example.com/not_allowed.json")


def test_notify_webhook_returns_explicit_failure_payload() -> None:
    payload = notify_webhook(
        "http://127.0.0.1:1/webhook",
        {"event": "unit-test"},
        timeout=0.05,
    )

    assert isinstance(payload, dict)
    assert payload.get("sent") is False
    assert payload.get("status") == 0
    assert payload.get("code") == "WEBHOOK_DELIVERY_FAILED"
    assert payload.get("category") == "network"
    assert isinstance(payload.get("remediation"), str)
    assert payload["remediation"].strip() != ""
    assert isinstance(payload.get("error"), str)
    assert payload["error"].strip() != ""


def test_notify_webhook_emits_structured_failure_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        notify_webhook("http://127.0.0.1:1/webhook", {"event": "unit-test"}, timeout=0.05)

    record = next(record for record in caplog.records if record.message == "webhook_notification_failed")
    assert getattr(record, "event", None) == "webhook_notification_failed"
    assert getattr(record, "url", None) == "http://127.0.0.1:1/webhook"
    assert isinstance(getattr(record, "error_type", None), str)
    assert getattr(record, "error", None)


def test_failure_payload_helper_preserves_standard_error_contract() -> None:
    payload = failure_payload(
        code="EXAMPLE_FAILURE",
        category="validation",
        remediation="Retry with a supported input value.",
        error="invalid field",
        status=400,
    )

    assert payload == {
        "code": "EXAMPLE_FAILURE",
        "category": "validation",
        "remediation": "Retry with a supported input value.",
        "error": "invalid field",
        "status": 400,
    }


def test_parse_wkt_invalid_input_raises_explicit_exception() -> None:
    with pytest.raises(ValueError, match="cannot parse WKT"):
        _parse_wkt("NOT_A_WKT")
