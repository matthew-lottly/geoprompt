# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

GeoPrompt follows an LTS-style support window for the current released line and the latest patch line used in published evidence bundles.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it **privately** by
emailing the maintainer rather than opening a public issue.

1. Email a description of the vulnerability, steps to reproduce, and any
   relevant logs or proof-of-concept code.
2. You should receive an acknowledgement within **72 hours**.
3. We will work with you to understand and resolve the issue before any public
   disclosure.

## CVE triage and supply-chain policy

- CVE triage starts within 72 hours for confirmed dependency or package issues.
- Security-sensitive releases should review SBOM generation, provenance evidence, and artifact integrity before publishing.
- Secrets scanning and unsafe configuration review are release gates for service-facing changes.
- Critical fixes should include rollback guidance and a hotfix note in the changelog.

Thank you for helping keep GeoPrompt and its users safe.
