# Performance and Scale

GeoPrompt is optimized first for clear analyst workflows, then for larger-scale batch processing with optional acceleration.

## Practical Scale Ceilings

| Workload | Recommended path | Guidance |
| --- | --- | --- |
| Under 100K features | Core frame methods | Default settings are usually enough |
| 100K to 1M features | Large preset plus spatial index | Prefer indexed joins and chunked reads |
| Multi-million rows | Huge preset plus chunking and optional deps | Stream inputs, avoid full dense matrices |

## Stress and Benchmark Coverage

The repository includes benchmark regression tests and larger synthetic corpora. Use them to validate changes before release.

## Profiling the Expensive Parts

Focus on these first:
- repeated joins without a reusable spatial index
- pairwise distance matrices over very large frames
- overlay-heavy polygon workflows without optional geometry engines
- raster-wide operations without chunked windows

## Memory-Aware Chunking

Use adaptive chunk sizing for remote or large local sources:

```python
import geoprompt as gp

size = gp.adaptive_chunk_size(2_000_000, available_memory_mb=4096)
for chunk in gp.iter_data("big.csv", x_column="x", y_column="y", chunk_size=size):
    _ = chunk.head(1)
```

## Optional Speedups

- NumPy helps vector math and dense numeric operations.
- Shapely improves overlay and geometry-heavy paths.
- GeoPandas helps roundtrip interoperability and high-level geospatial I/O.
- Rasterio improves raster reads, windows, resampling, and overviews.

## Multi-Million Row Guidance

1. Read only the columns you need.
2. Reuse the same spatial index across repeated windows or joins.
3. Write intermediate artifacts to GeoParquet or JSON bundles.
4. Prefer summary outputs over full pairwise joins when possible.
5. Keep benchmark snapshots with each release candidate.
