# Prior Work Comparison & Novelty Analysis

## 1. Closest Prior Work

### 1a. Conformal Prediction on Graphs

| Paper | Venue | Task | Key Idea | Differs from STRATA |
|-------|-------|------|----------|---------------------|
| **DAPS** (Zargarbashi et al., 2023) | ICML'23 | Node classification | Diffusion-based conformal prediction sets; propagates softmax scores through graph topology to produce smaller prediction sets | Classification only; homogeneous graphs; no heterogeneous types, no regression intervals, no heteroscedastic normalization |
| **NAPS** (Clarkson & Sherr, 2023) | ICML'23 | Node classification | Distribution-free prediction sets using neighborhood aggregation of p-values | Classification sets (not intervals); homogeneous graphs; no per-type Mondrian coverage |
| **SNAPS** (Song et al., 2024) | NeurIPS'24 | Node classification | Similarity-navigated conformal prediction; uses node similarity to weight calibration scores | Classification; homogeneous; no multi-layer infrastructure modeling |
| **CF-GNN** (Huang et al., 2024) | NeurIPS'23 | Node classification | Conformalized GNN with learned score functions | Classification; single graph type; no propagation-aware normalization from residuals |
| **BayesianCP** (Cha et al., 2023) | NeurIPS'23 Workshop | Node classification | Temperature scaling of Bayesian GNNs for conformal prediction | Classification; Bayesian approach; homogeneous; no infrastructure application |
| **TorchCP** (Huang et al., 2024) | Library | Classification + Regression + Graph | Implements DAPS, NAPS, SNAPS, CF-GNN; comprehensive toolbox | Library, not method; graph module is classification-only; no heterogeneous graph support, no regression intervals on graphs |

### 1b. Conformal Regression (Non-Graph)

| Paper | Key Idea | Differs from STRATA |
|-------|----------|---------------------|
| **CQR** (Romano et al., 2019) | Conformalized quantile regression | Tabular/time-series; no graph structure; STRATA extends CQR to heterogeneous graphs with propagation normalization |
| **Normalized CP** (Papadopoulos et al., 2002; Barber et al., 2023) | Heteroscedastic normalization of conformal scores | Tabular; σ from model uncertainty, not graph-propagated training residuals |
| **ACI** (Gibbs & Candès, 2021) | Adaptive conformal for distribution shift | Online/streaming; STRATA is offline split conformal |

### 1c. Heterogeneous GNNs

| Paper | Key Idea | Differs from STRATA |
|-------|----------|---------------------|
| **R-GCN** (Schlichtkrull et al., 2018) | Per-relation weight matrices | Foundational architecture; no conformal prediction, no UQ |
| **HAN** (Wang et al., 2019) | Hierarchical attention for heterogeneous graphs | Attention-based; no conformal calibration |
| **HGT** (Hu et al., 2020) | Transformer-style heterogeneous attention | More complex architecture; no UQ or conformal |

### 1d. Infrastructure Network UQ

| Paper | Key Idea | Differs from STRATA |
|-------|----------|---------------------|
| **Arderne et al. (2020)** Sci Data | Global power system mapping from open data | Topology inference, not UQ; no conformal prediction |
| **Birchfield et al. (2017)** | ACTIVSg synthetic grid validation criteria | Dataset provenance; no UQ method |
| **OpenGridMap** (Rivera et al., 2016) | Crowdsourced power grid topology | Data collection, not prediction or UQ |

---

## 2. STRATA's Novel Contributions

### Confirmed Novel (no direct prior):

1. **CHMP (Conformal Heterogeneous Message Passing)**: Propagation-aware normalization of conformal scores using *frozen training-set neighbor residuals* on *heterogeneous* graphs. No prior work combines (a) graph-propagated residuals for σ_i with (b) split-conformal guarantees on (c) heterogeneous multi-type graphs for (d) regression intervals.

2. **Mondrian conformal regression on heterogeneous infrastructure graphs**: Per-type coverage guarantees via node-type-conditional quantile estimation for regression. All prior graph CP work targets classification.

3. **Suite of learnable calibrators for graph conformal regression**: MetaCalibrator (heteroscedastic NLL), AttentionCalibrator (learned neighbor weights), LearnableLambdaCalibrator (per-type λ search), CQR+propagation, Ensemble epistemic variance—all applied to heterogeneous graph regression.

4. **Comprehensive diagnostics for graph conformal**: σ-vs-hitrate, width-decile coverage, degree-conditional coverage, bootstrap CIs, Wilcoxon/Friedman tests, runs test (non-exchangeability), Moran's I spatial autocorrelation—assembled as a diagnostic suite for graph conformal prediction.

5. **Multi-utility infrastructure benchmark**: Synthetic heterogeneous graph generator with power (tree), water (grid-mesh), telecom (star-hub) topologies, cross-utility coupling, and cascading-failure risk labels.

### Partially Novel (extends existing ideas to new domain):

6. **CQR on heterogeneous graphs**: CQR (Romano et al., 2019) exists for tabular; STRATA extends it with propagation normalization to heterogeneous graphs.

7. **Ensemble epistemic variance for conformal calibration on graphs**: Deep ensembles (Lakshminarayanan et al., 2017) + conformal prediction exist separately; combining ensemble variance as σ_i for graph conformal is new.

### Honest Limitation:

8. **CHMP's empirical gains are modest**: Width redistribution, not aggregate reduction. The ensemble variant is the only one with significant width improvement (6.4%, p=0.019). CHMP does not consistently beat Mondrian on real data.

---

## 3. Keyword Map for Literature Search

### Core Method Keywords
- "conformal prediction graph neural network"
- "conformal prediction heterogeneous graph"
- "split conformal regression graph"
- "mondrian conformal prediction node type"
- "propagation-aware conformal prediction"
- "normalized conformal scores graph"
- "heteroscedastic conformal graph"
- "CHMP conformal heterogeneous message passing"

### Extended Method Keywords
- "conformalized quantile regression graph"
- "graph neural network uncertainty quantification regression"
- "ensemble conformal prediction graph"
- "calibration conformal prediction GNN"
- "spatial conformal prediction infrastructure"
- "distribution-free prediction intervals graph"

### Application Keywords
- "infrastructure risk assessment graph neural network"
- "cascading failure prediction uncertainty"
- "multi-utility interdependent network risk"
- "power grid uncertainty quantification"
- "heterogeneous infrastructure graph learning"

### Diagnostic / Statistical Keywords
- "conditional coverage conformal prediction"
- "Moran's I conformal scores"
- "non-exchangeability test conformal graph"
- "calibration error ECE conformal intervals"

---

## 4. Gap Assessment & Recommendations

### Gaps That Could Strengthen the Paper

| Gap | Risk Level | Recommendation |
|-----|-----------|----------------|
| All prior graph CP is classification; no direct regression competitor exists | LOW (positive for novelty) | Emphasize this gap explicitly in Related Work; position STRATA as the first graph conformal regression framework |
| No independent multi-utility dataset with real lat/lon | MEDIUM | Add a recognized geolocated dataset (e.g., OSM substations, OPSD power plants) to strengthen external validity |
| CHMP's empirical gains are modest on real data | HIGH (reviewers will press this) | (a) Reframe contribution as "width redistribution" + "methodology framework" rather than "width reduction"; (b) Add conditional coverage analysis showing CHMP improves coverage for high-difficulty nodes even if aggregate is unchanged; (c) Add spatial visualization of width redistribution |
| Lambda insensitivity suggests weak signal | MEDIUM | Discuss this honestly; consider adding an experiment varying graph density/coupling to show when λ matters more |
| CQR underperforms | LOW | Already discussed; consider removing CQR from main table and moving to supplementary |
| Only ~400-node graphs | MEDIUM | Add scalability experiment with 1000+ node synthetic graphs |
| No comparison to TorchCP's graph methods | LOW | TorchCP graph module is classification-only; note this in Related Work |

### Suggested Method Improvements (if novelty feels thin)

1. **Conditional coverage analysis**: Show that CHMP improves coverage specifically for high-difficulty nodes (top quintile by σ_i) even when marginal coverage is identical—this would demonstrate the redistribution value.

2. **Topology perturbation sensitivity**: Show CHMP is more robust to edge dropout / rewiring than Mondrian CP.

3. **Larger-scale experiment**: Run on a 2000-node synthetic graph to demonstrate scalability.

4. **Cross-dataset transfer**: Train on ACTIVSg200, calibrate on IEEE 118 (or vice versa) to test domain transfer.

---

## 5. Summary Verdict

**STRATA is novel.** No prior work applies conformal prediction to heterogeneous graph regression with propagation-aware normalization. The closest competitors (DAPS, NAPS, SNAPS, CF-GNN) are all classification-only and homogeneous-graph-only. The key risk is the modest empirical improvement—but the theoretical contribution (CHMP preserves coverage guarantees while enabling width redistribution) and the comprehensive framework (5 calibrators, full diagnostics, spatial analysis) are substantial.

**Recommended framing**: Position STRATA as a *methodological framework* for conformal prediction on heterogeneous graphs, not as a method that achieves large empirical gains over baselines. Emphasize: (1) the first graph conformal regression system, (2) the coverage guarantee preservation, (3) the diagnostic suite, (4) the spatial analysis integration.
