# Deployment Guide

This guide covers production-minded GeoPrompt deployment on containers and cloud platforms.

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
from geoprompt.enterprise import AuthProfile, deployment_template

profile = AuthProfile(portal_url="https://www.arcgis.com", username="service-account")
config = deployment_template("docker", app_name="geoprompt-service", port=8000)
print(config)
```

## Hosted Service Editing Guidance

Use:
- AuthProfile for secure token handling
- paginated_request for retry-aware paging
- feature_service_query, add, update, delete, and sync helpers for controlled edits
- AuditLog for traceability of who ran what and when

## Database Roundtrips

See the persona example for moving records through DuckDB or PostGIS-style flows and then back into GeoPrompt for reporting.
