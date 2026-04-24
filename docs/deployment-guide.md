# Deployment Guide

This guide covers production-minded GeoPrompt deployment on containers and cloud platforms.

## Production operations runbook

Use this as the minimum bar for a hosted GeoPrompt service:

- require an API key or portal-backed token flow for non-health endpoints
- run health, schema, nearest, and scenario comparison smoke checks on every deployment
- enable structured request logging with request IDs so failures can be traced quickly
- store long-running job state in a persistent job file or external store so polling and resumption stay predictable
- keep least-privilege secrets in environment variables or the platform secret manager, never in notebooks or example scripts

## Container Deployment

Use the included container template for repeatable builds:

```bash
docker build -t geoprompt .
docker run --rm geoprompt geoprompt info
```

## Azure, AWS, and GCP

GeoPrompt includes deployment template helpers for:

- Docker-based application containers
- Azure web app style deployments
- AWS function or service packaging
- GCP runtime configuration stubs

## Secret Management

Recommended patterns:

- inject service tokens through environment variables
- keep credentials outside notebooks and example scripts
- use CI secret stores for publishing and hosted service checks
- rotate tokens used for feature-service editing workflows

## Production Configuration Example

```python
from geoprompt.enterprise import AuthProfile, deployment_template, service_resilience_profile

profile = AuthProfile(portal_url="https://www.arcgis.com", username="service-account")
config = deployment_template("docker", app_name="geoprompt-service", port=8000)
ops = service_resilience_profile(
    "https://example.internal/geoprompt",
    auth_profile=profile,
    roles=["analyst", "operator"],
    retry_count=4,
    rate_limit_per_minute=120,
)
print(config)
print(ops)
```

## Hosted Service Editing Guidance

Use:

- AuthProfile for secure token handling
- paginated_request for retry-aware paging
- feature_service_query, add, update, delete, and sync helpers for controlled edits
- AuditLog for traceability of who ran what and when

## Smoke checks

Before promoting a deployment, run at least these checks:

1. health endpoint returns 200 and the expected version
2. schema report responds on a small feature payload
3. nearest endpoint returns deterministic output on a tiny test frame
4. a queued job can be submitted, polled, and resumed without losing state

Example local command sequence:

```bash
python -m geoprompt.cli info
python -m geoprompt.cli capability-report --format table
python -m pytest tests/test_service_hardening.py -q
python -m pytest tests/test_capability_report_contract.py -q
```

Expected pass criteria:

- CLI exits with code 0
- capability report completes without crashing in the current environment
- service hardening tests pass for auth, payload, and smoke-check workflows
- no unexpected configuration secrets or placeholder endpoints are used for the rehearsal

## Monitoring and failure recovery

Recommended defaults:

- publish structured logs to a central sink
- capture request rate, failure rate, and job backlog size
- alert on repeated 401, 429, and 5xx responses
- keep a rollback-ready image for the previous known-good release
- benchmark remote job throughput and failure recovery during release rehearsal, not only after incidents

## Rollback drill expectations

1. keep the previous container image or wheel available before promotion
2. retain the prior release evidence bundle and provenance manifest
3. verify that a rollback still passes health and capability checks
4. record who approved the rollback or hotfix and which artifact version was restored

## Database Roundtrips

See the persona example for moving records through DuckDB or PostGIS-style flows and then back into GeoPrompt for reporting.
