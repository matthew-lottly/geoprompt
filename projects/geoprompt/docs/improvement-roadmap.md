# GeoPrompt Spatial Tools — Improvement Roadmap

> **Baseline**: 85 tools, 338 tests (280 unit + 58 cross-validation), 0 failures, 0 warnings  
> **Audit date**: 2026-03-21  
> **Reference packages**: Shapely 2.1.2, SciPy 1.17.1, NumPy 2.4.3

---

## Bug Fixed This Session

| Tool | Bug | Fix |
|------|-----|-----|
| **77 (jenks_natural_breaks)** | Class assignment inverted — values between breaks mapped to wrong class | Rewrote classification loop: default to class k, scan breaks ascending, break on first `val < break[i]` |

---

## Priority 1 — Performance (O(n²) → O(n log n))

These affect ~30 tools that use pairwise distance calculations.

| # | Improvement | Tools Affected | Current | Target | Impact |
|---|-------------|----------------|---------|--------|--------|
| 1 | **Add spatial indexing (R-tree / k-d tree)** | 12, 13, 14, 15, 16, 17, 18, 22, 41, 42, 43, 47, 52, 53, 58, 61, 72, 74, 75 | O(n²) brute-force `_pairwise_distance_matrix` | scipy.spatial.KDTree pre-filter with `query_ball_point` / `query(k)` | 10-100× faster for n > 1000 |
| 2 | **Add `search_radius` and `k_neighbors` to IDW** (Tool 12) | 12, 53, 80, 84 | Uses all n points for every grid cell | Only k-nearest or within radius | Major speedup + avoids distant-point noise |
| 3 | **Lazy distance matrix** | 41, 42, 47, 52, 72, 73, 74, 75 | Full n×n matrix computed upfront | Compute on-demand or use KDTree | Memory: O(n) vs O(n²) |
| 4 | **Cache frequently-used centroids** | All spatial tools | `_centroids()` re-computes each call | Cache in `_cache` dict | Minor but cumulative |

---

## Priority 2 — Algorithmic Accuracy

| # | Improvement | Tool(s) | Current Algorithm | Gold Standard | Notes |
|---|-------------|---------|-------------------|---------------|-------|
| 5 | **Auto-fit variogram for Kriging** | 13 | User-supplied sill/range/nugget | Fit empirical variogram → best-fit model selection (spherical/exponential/gaussian) via SSE | What PyKrige, ArcGIS, and QGIS all do |
| 6 | **Kriging variance output** | 13 | Returns only predicted values | Return prediction variance per cell | Essential for uncertainty quantification |
| 7 | **Kriging cross-validation (LOOCV)** | 13 | None | Leave-one-out cross-validation → RMSE, MAE | Standard QA metric |
| 8 | **Silverman bandwidth for KDE** | 18 | `h = extent/sqrt(n)` (arbitrary) | Silverman rule: `h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)` | Proper statistical bandwidth |
| 9 | **Adaptive bandwidth KDE** | 18 | Fixed bandwidth | Pilot-density adaptive: `h_i ∝ f(x_i)^(-1/2)` | Better for multi-scale patterns |
| 10 | **FDR correction for hotspot p-values** | 16 | Raw p-values | Benjamini-Hochberg FDR correction | Critical for multiple comparison: testing 1000 points at α=0.05 yields ~50 false positives |
| 11 | **Ripley's K edge correction** | 52 | No edge correction | Ripley isotropic / translation / toroidal correction | Bias near study area boundaries |
| 12 | **True Natural Neighbor Interpolation** | 53 | Actually uses modified IDW with k-nearest | Sibson weights via Voronoi area-stealing (use Shapely Voronoi) | Algorithm name doesn't match implementation |
| 13 | **GWR bandwidth cross-validation** | 22 | `h = extent / 3` (arbitrary) | AICc-minimising golden section search | What GWR4 / MGWR use |
| 14 | **Spatial outlier z-score → local z-score** | 76 | Global z-score (no spatial component) | Local Moran's I or local z-score vs spatial neighbors | Name implies spatial, implementation is aspatial |

---

## Priority 3 — New Capabilities

| # | Feature | Category | Description | Reference |
|---|---------|----------|-------------|-----------|
| 15 | **Getis-Ord Gi (global statistic)** | Hotspot | Complement to local Gi* | PySAL esda |
| 16 | **Bivariate Moran's I** | Autocorrelation | Spatial correlation between two variables | GeoDa, PySAL |
| 17 | **Local Geary's C** | Autocorrelation | Local version complements global Geary | PySAL |
| 18 | **Spatial Durbin Model** | Regression | Regression with spatial lag of X and Y | PySAL spreg |
| 19 | **Kernel regression (LOESS)** | Regression | Non-parametric local regression | statsmodels |
| 20 | **Empirical Bayesian Kriging** | Interpolation | Auto-fits multiple variograms, more robust | ArcGIS Pro |
| 21 | **Co-Kriging** | Interpolation | Use correlated secondary variable | PyKrige |
| 22 | **Anisotropic IDW / Kriging** | Interpolation | Direction-dependent distance weights | ArcGIS Pro |
| 23 | **Space-time Kriging** | Interpolation | Spatiotemporal prediction | R gstat |
| 24 | **Spatial scan statistic (SaTScan)** | Cluster detection | Kulldorff circular/elliptical scan | SaTScan, PySAL |
| 25 | **OPTICS clustering** | Clustering | Density-based with variable epsilon | scikit-learn |
| 26 | **Spatial K-means (constrained)** | Clustering | K-means with contiguity constraint | max-p regions (PySAL) |
| 27 | **Point pattern intensity function** | Pattern | λ(s) estimation, first-order effects | spatstat (R) |
| 28 | **Mark correlation function** | Pattern | Correlation of marks as function of distance | spatstat (R) |
| 29 | **Network kernel density** | Network | KDE along network (not Euclidean) | SANET, PySAL |
| 30 | **Morphological spatial pattern analysis** | Landscape | Forest fragmentation metrics (MSPA) | GuidosToolbox |
| 31 | **Geographic detector (GeoDetector)** | Statistics | Factor detection, interaction detection | geodetector R package |
| 32 | **Flow mapping / OD analysis** | Network | Origin-destination flow visualization + analysis | flowmap.blue |
| 33 | **Reachability / service area** | Network | Network-based service areas with barriers | ArcGIS Network Analyst |
| 34 | **Location-allocation (p-median)** | Optimization | Optimal facility placement minimizing travel | PySAL spopt |
| 35 | **Spatial weights matrix (W)** | Foundation | Queen/Rook/K-NN/distance-band W matrix object | PySAL libpysal |
| 36 | **Tobler flow estimation** | Migration | Estimate migration flows from stock data | Tobler package |
| 37 | **Terrain ruggedness index (TRI)** | Raster | Neighborhood elevation variability | GDAL DEM |
| 38 | **Topographic position index (TPI)** | Raster | Relative elevation vs neighbors | GDAL DEM |
| 39 | **Curvature (profile + plan)** | Raster | Surface curvature for hydrology | ArcGIS Spatial Analyst |
| 40 | **Flow direction / accumulation** | Hydrology | D8/D-inf flow routing | WhiteboxTools |
| 41 | **Watershed delineation** | Hydrology | Pour point → contributing area | WhiteboxTools |
| 42 | **Stream network extraction** | Hydrology | Threshold-based stream from flow accumulation | GRASS r.stream.extract |
| 43 | **Visibility graph analysis** | Viewshed | Graph of inter-visible points | Space Syntax |
| 44 | **Solar radiation modeling** | Raster | Incoming solar radiation from DEM | GRASS r.sun |

---

## Priority 4 — API & Usability

| # | Improvement | Description |
|---|-------------|-------------|
| 45 | **Chaining API** | Allow `frame.buffer(100).dissolve("zone").hotspot_analysis()` fluent syntax |
| 46 | **Progress callbacks** | For long-running tools, emit progress (% complete) |
| 47 | **Parallel execution** | Use `concurrent.futures` for embarrassingly parallel grid operations |
| 48 | **GeoParquet I/O** | Read/write GeoParquet for large datasets |
| 49 | **CRS-aware distance** | Auto-detect EPSG:4326 and switch to Haversine; auto-choose projected CRS for area/length |
| 50 | **Streaming / chunk mode** | Process large datasets in chunks to limit memory |
| 51 | **Type-safe result objects** | Return typed result objects instead of raw dicts for better IDE support |
| 52 | **Docstrings for all 85 tools** | Full docstrings with parameter descriptions, examples, references |
| 53 | **Spatial weights matrix as first-class object** | Decouple W from individual tools; share across Moran, Geary, Gi*, GWR |
| 54 | **Validation decorators** | Standardize input validation (geometry type, required columns, CRS) |

---

## Priority 5 — Testing & Quality

| # | Improvement | Description |
|---|-------------|-------------|
| 55 | **Property-based testing** | Use Hypothesis to generate random geometries and verify invariants |
| 56 | **Benchmark suite** | Automated performance tracking (pytest-benchmark) for key tools |
| 57 | **Large-dataset stress tests** | 10k/100k/1M point tests to find scaling cliffs |
| 58 | **Edge-case test coverage** | Empty frames, single-point frames, collinear points, duplicate coords |
| 59 | **GeoPandas comparison suite** | Automate end-to-end comparison: same input → same output for all overlapping tools |
| 60 | **Numerical stability tests** | Test with very large/small coordinates, near-zero areas, near-degenerate polygons |

---

## Implementation Order (Recommended)

### Phase 1: Foundation (highest leverage)
1. Spatial indexing (KDTree wrapper) → unlocks performance for all distance-based tools
2. Spatial weights matrix object → unlocks correct Moran/Geary/Gi*/GWR
3. CRS-aware distance auto-switching → correctness for geographic coords
4. Silverman bandwidth for KDE → immediate accuracy win

### Phase 2: Interpolation excellence
5. Kriging auto-fit variogram + variance + LOOCV
6. True Natural Neighbor Interpolation
7. Anisotropic distance support
8. IDW search_radius + k_neighbors

### Phase 3: Statistical rigor  
9. FDR correction for hotspot/cluster p-values
10. Ripley's K edge correction
11. GWR bandwidth cross-validation
12. Local spatial outlier detection (replace global z-score)

### Phase 4: New analysis domains
13. Hydrology tools (flow direction, watershed, stream extraction)
14. Advanced clustering (OPTICS, spatial k-means, spatial scan)
15. Network analysis extensions (service areas, location-allocation)
16. Landscape ecology metrics (MSPA, connectivity)

### Phase 5: Scale & polish
17. Parallel grid operations
18. GeoParquet I/O
19. Streaming/chunk mode
20. Full documentation + benchmarks
