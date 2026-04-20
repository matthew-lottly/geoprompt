# Threat Model for GeoPrompt Service Deployments

This page summarizes the main attack surfaces for service-facing and remote execution workflows.

## Attack surfaces

- uploaded files and remote dataset inputs
- service authentication tokens and API keys
- webhook and notification endpoints
- deployment configuration and secrets management
- generated report artifacts that may contain sensitive metadata

## Main controls

- validate file uploads and allowed extensions
- use RBAC or scoped API keys for service endpoints
- redact sensitive fields before logs or stakeholder exports
- run secrets scanning and configuration review before release
- keep provenance and benchmark evidence bundles for rollback and audit

## Remote execution notes

GeoPrompt is strongest when execution is constrained to trusted infrastructure, documented inputs, and auditable release artifacts.