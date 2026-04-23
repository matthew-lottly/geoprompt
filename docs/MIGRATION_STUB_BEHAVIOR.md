# Stub and Simulation-Only Behavior Migration Guide

GeoPrompt has transitioned to a **production-safe by default** stance on stub and simulation-only functions. This guide helps users who may have been relying on stub behavior.

## What Changed

### Previous Behavior (Pre-2026-Q2)
- Some missing optional dependencies would return "fake data" or placeholder outputs
- Simulation-only functions would return plausible-looking results
- Enterprise operations would fail silently or return stubs without clear notification

### New Behavior (2026-Q2+)
- Missing optional dependencies raise explicit `ImportError` with clear remediation
- Simulation-only functions raise `ImportError` by default; require explicit `allow_stub_fallback=True` to enable
- Enterprise operations require explicit opt-in; stub mode emits `UserWarning` when enabled
- Service deployments block stub-mode by default; set `GEOPROMPT_DEV_PROFILE=true` to enable

## Migration Scenarios

### Scenario 1: I Was Using Stub Data for Testing

**Old Code:**
```python
import geoprompt as gp
# This would return fake data if arcpy was not installed
result = gp.enterprise_geodatabase_connect("localhost", "mydb")
```

**New Code — Option A: Install the Backend**
```python
pip install arcpy
# Now real backend is available
result = gp.enterprise_geodatabase_connect("localhost", "mydb")
```

**New Code — Option B: Explicitly Enable Stubs (Testing Only)**
```python
import warnings
import geoprompt as gp

# Suppress or catch the warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    result = gp.enterprise_geodatabase_connect("localhost", "mydb", allow_stub_fallback=True)

# Or handle the warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = gp.enterprise_geodatabase_connect("localhost", "mydb", allow_stub_fallback=True)
    if w:
        print(f"Using stub: {w[0].message}")
```

### Scenario 2: I'm Using Cloud I/O but FastSpec is Missing

**Old Code:**
```python
import geoprompt as gp
# This might return fake data or fail silently
records = gp.read_cloud_json("s3://my-bucket/data.geojson")
```

**New Code:**
```python
# Solution: Install fsspec and the cloud backend
pip install fsspec s3fs  # For S3
pip install fsspec google-cloud-storage  # For GCS
pip install fsspec adlfs  # For Azure Blob Storage

# Now the call works
import geoprompt as gp
records = gp.read_cloud_json("s3://my-bucket/data.geojson")
```

### Scenario 3: I'm Running in a Service and Don't Want Stubs

**Old Code:**
```python
# No way to prevent stub fallbacks in service mode
from geoprompt.service import build_app
app = build_app()
```

**New Code:**
```python
# Stubs are blocked by default in service; production-safe
from geoprompt.service import build_app
app = build_app()  # Fails if stubs are enabled; raise RuntimeError

# To enable stubs (dev/testing only):
import os
os.environ["GEOPROMPT_DEV_PROFILE"] = "true"
app = build_app()  # Now stubs are allowed
```

### Scenario 4: I Want to Know What Features Are Available at Startup

**Old Code:**
```python
# No built-in way to check capability; had to try/except everything
```

**New Code:**
```python
import geoprompt as gp

report = gp.capability_report()
print(f"Enabled features: {report['enabled']}")
print(f"Disabled features: {report['disabled']}")
print(f"Degraded features: {report['degraded']}")
print(f"Fallback policy: {report['fallback_policy']}")
```

## Environment Variables for Stubbing Control

Set these environment variables to control stub/fallback behavior:

| Variable | Values | Purpose |
|----------|--------|---------|
| `GEOPROMPT_FALLBACK_POLICY` | `error`, `warn`, `allow` | Control how missing dependencies are handled (default: `error`) |
| `GEOPROMPT_DEV_PROFILE` | `true`, `false` | Enable development mode (allows stubs in service) |
| `GEOPROMPT_ALLOW_STUB_FALLBACK` | `true`, `false` | Explicitly allow stub fallbacks (service only) |

**Examples:**
```bash
# Production: fail fast on missing backends
export GEOPROMPT_FALLBACK_POLICY=error
python my_analysis.py

# Development: warn on missing backends, use stubs
export GEOPROMPT_FALLBACK_POLICY=warn
export GEOPROMPT_DEV_PROFILE=true
python my_script.py

# Testing: silently allow all fallbacks (not recommended for prod)
export GEOPROMPT_FALLBACK_POLICY=allow
pytest tests/
```

## Breaking Changes Summary

| Feature | Old Behavior | New Behavior | Migration |
|---------|-------------|--------------|-----------|
| Enterprise GDB ops | Silent stub fallback | Explicit ImportError | Use `allow_stub_fallback=True` or install `arcpy` |
| DXF I/O | Fake geometry rows returned | Explicit ImportError | Install `fiona` |
| OSM I/O | Implicit fallback | Explicit ImportError | Install `osmium` or use `allow_stub_fallback=True` |
| Cloud I/O (S3/GCS/Azure) | Might silently fail | Explicit payload size/scheme validation | Install `fsspec` + backend |
| Expression evaluation | Bare `eval()` | Safe sandbox with guards | No action needed (internal) |
| Service endpoints | Might use stubs | Blocked by default | Set `GEOPROMPT_DEV_PROFILE=true` if needed |

## Getting Help

If you encounter an `ImportError` or `FallbackWarning`:

1. **Check `capability_report()`** to see what's available
2. **Install the missing backend** using the error message's recommendation
3. **Read the docstring** of the function you're calling (e.g., `help(gp.enterprise_geodatabase_connect)`)
4. **Check environment variables** if running in a service or CI environment

## Questions?

- File an issue on GitHub with details about your use case
- Check the docs at [GeoPrompt Docs](https://docs.example.com/geoprompt)
- Read the [API reference](https://docs.example.com/api) for tier levels and maturity status
