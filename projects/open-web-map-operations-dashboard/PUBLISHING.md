# Publishing Guide

## Recommended Standalone Repository Name

- open-web-map-operations-dashboard

## Recommended Description

- React and TypeScript open web map dashboard for operational layer review, map filtering, and feature inspection patterns.

## Suggested Topics

- react
- typescript
- maplibre
- openlayers
- gis
- geospatial
- frontend

## Split Steps

1. Create a new empty repository named `open-web-map-operations-dashboard`.
2. Copy this project folder into the new repository root.
3. Preserve `data/`, `src/`, `tests/`, and the frontend build files.
4. Use the committed screenshot in `assets/dashboard-live-screenshot.png` or refresh it with `npm run capture:demo`. Do not use placeholder illustration art.
5. Reference [docs/architecture.md](docs/architecture.md), [docs/demo-storyboard.md](docs/demo-storyboard.md), and [docs/site-map.md](docs/site-map.md) from the README when polishing the public pitch.

## First Public Polish Pass

- Connect the layer list to a real service response from the PostGIS lane
- Refresh the browser screenshot with `npm run capture:demo` and optionally add a short GIF from the same flow
- Do not add any placeholder SVG or illustration-style preview while that capture is still missing
