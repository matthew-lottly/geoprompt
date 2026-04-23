# Exception Taxonomy and Failure Modes

This document defines expected failure modes for high-risk public GeoPrompt APIs.

## Principles

- Fail fast on unsafe input and missing required dependencies.
- Preserve exception cause chains when rewrapping (`raise ... from exc`).
- Do not return success-like payloads when an operation fails.
- Use typed exceptions to make caller handling deterministic.

## API Failure Map

| Module | Public API | Expected failure mode(s) | Caller remediation |
| --- | --- | --- | --- |
| `safe_expression` | `evaluate_safe_expression` | `ExpressionValidationError`, `ExpressionExecutionError`, `ValueError` (limits) | Reject user expression; show safe-expression guidance and simplify expression. |
| `io` | `read_cloud_json` | `ValueError` (disallowed URL scheme, oversized payload), `json.JSONDecodeError`, `URLError`/`HTTPError` | Use `http/https` only; verify payload size/JSON format; retry or correct endpoint. |
| `io` | `read_dxf` | `ImportError` when `ezdxf` is unavailable | Install required dependency (`pip install ezdxf`) or choose a supported source format. |
| `db` | WKT/WKB parsing paths | `ValueError` for invalid geometry text; import-related fallback failures remain typed | Validate geometry before write/read; sanitize upstream geometry payloads. |
| `geoprocessing` | `post_http_notification` | Explicit failure payload (`sent=False`, `status=0`, `error`) on network errors | Retry/backoff and inspect error text; do not treat as successful notification. |
| `enterprise` | `enterprise_geodatabase_connect`, `versioned_edit`, `replica_sync`, `portal_publish` | `ImportError` by default without explicit stub opt-in | Enable real backend dependencies or explicitly opt in to stub mode for dev/test only. |
| `service` | `build_app` | `RuntimeError` for unsafe stub env combinations in non-dev profile | Set `GEOPROMPT_DEV_PROFILE=true` only in dev; remove conflicting stub env settings in production. |

## Notes on Explicit Failure

- Failure payloads should include actionable detail and never imply success.
- Where APIs return dictionaries on failure, standardization work is tracked under J3.33 and J3.34.
- Behavior above is enforced by tests in `tests/test_failure_transparency.py`.
