# Architecture

## Overview

This project models the frontend layer of an open web mapping stack.

## Flow

1. Sample dashboard layer metadata is loaded from checked-in JSON.
2. Filter state narrows the visible map layers by region and status.
3. Summary cards and layer cards reflect the filtered set.
4. A conceptual map surface visualizes how the frontend can evolve toward a real map library integration.

## Why It Works Publicly

- Demonstrates frontend GIS interaction without relying on proprietary products.
- Keeps the repo runnable in a standard Vite toolchain.
- Leaves a clear upgrade path toward MapLibre or OpenLayers while preserving the current reviewable UI structure.
