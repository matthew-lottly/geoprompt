# Extending GeoPrompt

GeoPrompt now publishes a stable extension story for advanced users who want to build on the package without forking it.

## Supported extension lanes

- plugin registration for reusable analyst or domain helpers
- connector starters for external services and data systems
- report starters for HTML and Markdown delivery
- domain module starters for organization-specific scoring logic

## Compatibility guidance

| Area | Status |
| --- | --- |
| Plugin registration helpers | stable within minor releases |
| Recipe and workflow guidance | stable and documented |
| Optional backends | supported when installed |
| Jupyter and IDE use | recommended and documented |

## Starter workflow

1. Generate a starter template with the extension template helper.
2. Register the plugin with the plugin registry.
3. Add a short test and one recipe entry.
4. Publish the workflow through the CLI or notebook gallery.

## Accessibility expectations

Generated HTML and dashboard artifacts should include:
- a document title
- a main landmark
- visible headings
- alt text for any images
- clear metric labels and table headers
