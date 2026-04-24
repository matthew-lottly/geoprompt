# Documentation Style Guide

This guide keeps the docs, figures, and examples consistent.

## Headings and tone

- use concise headings that match the workflow or problem type
- keep paragraphs short and plain-English first
- explain why an output matters, not only what it is

## Figure sizing and captions

- figure sizing should target readable widths for GitHub and PyPI pages
- prefer widths around 800 to 900 pixels for major figures
- keep aspect ratios consistent for before-and-after comparisons
- every figure should have a caption, a data source note, and a process note

## Chart styling rules

- use a high-contrast palette with a neutral background and direct labels where possible
- include axis units for every quantitative chart
- include a legend whenever color or symbol meaning would not be obvious to a first-time reader
- prefer comparable scales across before-and-after charts so deltas are visually honest

## Alt-text rubric

- state the chart or image type first
- name the main variables or comparison being shown
- mention the decision-relevant takeaway if it is visible in the figure
- avoid generic alt text such as "chart" or "screenshot"

## Markdown table readability

- keep column names short and explicit
- prefer one business question per table
- add units or value interpretation in nearby prose when raw column labels are not enough
- avoid tables that require horizontal scrolling for the core story

## Code examples

- use repository sample data or a clearly labeled representative dataset
- prefer copy-paste examples that end in a visible output or report artifact
- keep examples synchronized with the stable public API surface

## Update cadence

- review the docs and figure manifest before each release
- remove or archive stale screenshots and outputs when the package behavior changes
