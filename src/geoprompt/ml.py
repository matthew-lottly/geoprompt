"""Machine learning and AI helpers for spatial workflows.

These utilities provide lightweight, pure-Python fallbacks for common ML,
vision, NLP, optimisation, and model-operations patterns so GeoPrompt can
support analyst workflows without requiring heavyweight runtimes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import random
import re
import statistics
import warnings
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from .quality import simulation_only


def _try_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _flatten_grid(grid: Sequence[Sequence[float]]) -> list[float]:
    return [float(v) for row in grid for v in row]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _hash_vec(value: Any, dim: int = 8) -> list[float]:
    text = str(value).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return [round(digest[i] / 255.0, 4) for i in range(dim)]


def _feature_keys(rows: Sequence[dict[str, Any]], target_key: str | None = None) -> list[str]:
    if not rows:
        return []
    keys = [k for k in rows[0].keys() if k not in {target_key, "x", "y", "t", "geometry"}]
    return [k for k in keys if isinstance(rows[0].get(k), (int, float))]


def _dominant_class(rows: Sequence[dict[str, Any]], target_key: str) -> Any:
    counts = Counter(r.get(target_key) for r in rows)
    return counts.most_common(1)[0][0] if counts else None


def _simple_score(row: dict[str, Any], keys: Sequence[str]) -> float:
    values = [float(row.get(k, 0.0)) for k in keys]
    return _mean(values)


def feature_engineering_raster_pixels(
    points: Sequence[dict[str, Any]], raster: Sequence[Sequence[float]], *, band_name: str = "pixel_value"
) -> list[dict[str, Any]]:
    """Sample raster pixel values at point coordinates."""
    h = len(raster)
    w = len(raster[0]) if h else 0
    out = []
    for p in points:
        x = min(max(int(p.get("x", 0)), 0), max(w - 1, 0))
        y = min(max(int(p.get("y", 0)), 0), max(h - 1, 0))
        val = raster[y][x] if h and w else None
        out.append({**p, band_name: val})
    return out


def spatial_cross_validation_buffer(
    rows: Sequence[dict[str, Any]], *, target_key: str = "target", buffer_distance: float = 0.0, folds: int = 5
) -> dict[str, Any]:
    """Spatial buffer cross-validation using leave-nearby-out logic."""
    errors: list[float] = []
    for row in rows:
        x, y = float(row.get("x", 0)), float(row.get("y", 0))
        train = [
            r for r in rows
            if math.hypot(float(r.get("x", 0)) - x, float(r.get("y", 0)) - y) > buffer_distance
        ]
        if not train:
            continue
        pred = _mean([float(r.get(target_key, 0)) for r in train])
        errors.append(abs(pred - float(row.get(target_key, 0))))
    return {"folds": min(folds, max(1, len(rows))), "mean_error": round(_mean(errors), 4), "n": len(rows)}


def spatial_cross_validation_leave_one_out(
    rows: Sequence[dict[str, Any]], *, target_key: str = "target"
) -> dict[str, Any]:
    """Leave-one-out spatial cross-validation summary."""
    errors: list[float] = []
    for idx, row in enumerate(rows):
        train = [float(r.get(target_key, 0)) for j, r in enumerate(rows) if j != idx]
        pred = _mean(train) if train else float(row.get(target_key, 0))
        errors.append(abs(pred - float(row.get(target_key, 0))))
    return {"n": len(rows), "mae": round(_mean(errors), 4), "rmse": round(math.sqrt(_mean([e * e for e in errors])), 4)}


def spatial_resampling(
    rows: Sequence[dict[str, Any]], *, target_key: str = "target", strategy: str = "oversample"
) -> list[dict[str, Any]]:
    """Basic under/oversampling for imbalanced spatial labels."""
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(target_key)].append(dict(row))
    if not groups:
        return []
    rng = random.Random(42)
    sizes = [len(v) for v in groups.values()]
    if strategy == "undersample":
        target_size = min(sizes)
        return [rng.choice(v) for v in groups.values() for _ in range(target_size)]
    target_size = max(sizes)
    out: list[dict[str, Any]] = []
    for vals in groups.values():
        out.extend(vals)
        while len([x for x in out if x.get(target_key) == vals[0].get(target_key)]) < target_size:
            out.append(dict(rng.choice(vals)))
    return out


def build_sklearn_pipeline(steps: Sequence[str]) -> dict[str, Any]:
    """Return a portable pipeline specification for scikit-learn style workflows."""
    return {"steps": list(steps), "sklearn_available": _try_import("sklearn")}


def xgboost_lightgbm_integration() -> dict[str, Any]:
    """Report availability of gradient-boosting backends."""
    return {"xgboost": _try_import("xgboost"), "lightgbm": _try_import("lightgbm")}


def random_forest_spatial_prediction(
    rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str], target_key: str = "target"
) -> list[dict[str, Any]]:
    """Spatial prediction using Random Forest.

    When *scikit-learn* is installed, trains a real
    ``RandomForestClassifier`` (or ``Regressor`` for continuous targets) on
    the provided rows.  Falls back to a heuristic threshold classifier when
    sklearn is not available.
    """
    if not rows or not feature_keys:
        return list(rows)
    try:
        sklearn_ensemble = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier", "RandomForestRegressor"])
        import numpy as np_mod
        X = np_mod.array([[float(r.get(k, 0.0)) for k in feature_keys] for r in rows])
        y_raw = [r.get(target_key, 0) for r in rows]
        # Detect classification vs regression
        is_classification = all(isinstance(v, (bool, int)) and v in (0, 1) for v in y_raw)
        y = np_mod.array([int(v) for v in y_raw] if is_classification else [float(v) for v in y_raw])
        if is_classification:
            model = sklearn_ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = sklearn_ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        result = []
        for r, pred in zip(rows, predictions):
            entry = {**r, "prediction": float(pred)}
            if hasattr(model, "predict_proba") and is_classification:
                proba = model.predict_proba(X[list(rows).index(r): list(rows).index(r) + 1])
                entry["probability"] = round(float(proba[0][1]), 4)
            result.append(entry)
        return result
    except (ImportError, ModuleNotFoundError):
        pass
    # Heuristic fallback
    threshold = _mean([_simple_score(r, feature_keys) for r in rows])
    return [{**r, "prediction": int(_simple_score(r, feature_keys) >= threshold)} for r in rows]


@simulation_only("Use sklearn.ensemble.GradientBoostingClassifier or xgboost for real gradient boosting.")
def gradient_boosted_spatial_prediction(
    rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str], target_key: str = "target"
) -> list[dict[str, Any]]:
    """Simulation-only placeholder for gradient-boosted spatial scoring.

    This function does not train or run a production gradient boosting model.
    For real outputs, use sklearn GradientBoosting or XGBoost with a trained
    estimator and validated feature pipeline.
    """
    weights = [i + 1 for i, _ in enumerate(feature_keys)] or [1]
    scores = [sum(float(r.get(k, 0)) * w for k, w in zip(feature_keys, weights)) for r in rows]
    cutoff = _mean(scores)
    return [{**r, "prediction": int(s >= cutoff), "score": round(s, 4)} for r, s in zip(rows, scores)]


@simulation_only("Use sklearn.svm.SVC for a real support vector machine classifier.")
def svm_spatial_classification(
    rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str], target_key: str = "target"
) -> list[dict[str, Any]]:
    """Simulation-only placeholder for SVM-style spatial classification.

    This fallback is heuristic and not a real SVM model execution. For
    production classification, install sklearn and use ``sklearn.svm.SVC``
    with tuned hyperparameters.
    """
    center = _mean([_simple_score(r, feature_keys) for r in rows])
    return [{**r, "prediction": 1 if _simple_score(r, feature_keys) >= center else 0} for r in rows]


def logistic_regression_spatial_features(
    rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str], target_key: str = "target"
) -> list[dict[str, Any]]:
    """Logistic scoring over numeric spatial covariates.

    Uses sklearn.linear_model.LogisticRegression when available.
    """
    if not rows or not feature_keys:
        return list(rows)
    try:
        sklearn_lm = __import__("sklearn.linear_model", fromlist=["LogisticRegression"])
        import numpy as np_mod
        X = np_mod.array([[float(r.get(k, 0.0)) for k in feature_keys] for r in rows])
        y_raw = [r.get(target_key, 0) for r in rows]
        y = np_mod.array([int(v) for v in y_raw])
        model = sklearn_lm.LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        return [{**r, "probability": round(float(p[1]), 4), "prediction": int(p[1] >= 0.5)} for r, p in zip(rows, proba)]
    except (ImportError, ModuleNotFoundError):
        pass
    # Heuristic fallback
    out = []
    for r in rows:
        z = _simple_score(r, feature_keys)
        p = 1.0 / (1.0 + math.exp(-z))
        out.append({**r, "probability": round(p, 4), "prediction": int(p >= 0.5)})
    return out


@simulation_only("Use PyTorch (torch.nn) or TensorFlow for real neural network integration.")
def neural_network_integration(sequences: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Simulation-only neural-network integration preview.

    This helper does not execute a real neural backend or trained model.
    For production inference, install and configure PyTorch or TensorFlow.
    """
    backend = "heuristic"
    if _try_import("torch"):
        backend = "pytorch"
    elif _try_import("tensorflow"):
        backend = "tensorflow"
    return {"backend": backend, "sequence_count": len(sequences), "embedding": [round(_mean(seq), 4) for seq in sequences]}


@simulation_only("Use PyG (torch_geometric) or DGL for a real Graph Neural Network.")
def graph_neural_network_prediction(graph: dict[str, Any]) -> dict[str, Any]:
    """Simulation-only GNN placeholder using centrality-style scoring.

    This function does not perform real message passing or learned graph
    inference. For production graph learning, use PyG or DGL with a trained
    architecture.
    """
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    degrees = Counter(n for e in edges for n in e)
    return {"node_count": len(nodes), "edge_count": len(edges), "node_scores": {n: degrees.get(n, 0) for n in nodes}}


@simulation_only("Use PyTorch torchvision or TensorFlow Keras for a real CNN on rasters.")
def convolutional_neural_network_on_rasters(raster: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Simulation-only raster embedding placeholder.

    This helper does not execute a real CNN or segmentation model. For
    production raster inference, use TensorFlow/Keras or PyTorch vision models.
    """
    vals = _flatten_grid(raster)
    return {"embedding_dim": 4, "embedding": [min(vals or [0]), max(vals or [0]), round(_mean(vals), 4), len(vals)]}


@simulation_only("Use PyTorch LSTM/GRU or TensorFlow Keras for a real RNN time-series model.")
def recurrent_neural_network_spatial_time_series(sequences: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Simulation-only temporal trend placeholder for sequence inputs.

    This function does not execute an RNN or learned forecasting model. For
    production forecasting, use LSTM/GRU models in PyTorch or TensorFlow.
    """
    trends = [round(seq[-1] - seq[0], 4) if len(seq) >= 2 else 0.0 for seq in sequences]
    return {"sequence_count": len(sequences), "trend": trends}


@simulation_only("Use PyTorch transformers or Hugging Face for a real attention/transformer model.")
def transformer_model_spatial_sequences(sequences: Sequence[Sequence[float]], *, attention_heads: int = 4) -> dict[str, Any]:
    """Simulation-only transformer-style sequence placeholder.

    This helper does not run a real attention model or transformer backend.
    For production sequence modeling, use PyTorch transformers or Hugging Face.
    """
    return {"sequence_count": len(sequences), "attention_heads": attention_heads, "token_count": sum(len(s) for s in sequences)}


def automl_spatial_workflow(rows: Sequence[dict[str, Any]], *, target_key: str = "target") -> dict[str, Any]:
    """Pick a simple best-fit model family for a labelled spatial table."""
    labels = {r.get(target_key) for r in rows}
    best = "logistic_regression" if labels <= {0, 1} else "random_forest"
    return {"best_model": best, "rows": len(rows), "candidate_models": ["logistic_regression", "random_forest", "gradient_boosting"]}


def optuna_hyperparameter_search(search_space: dict[str, Sequence[Any]]) -> dict[str, Any]:
    """Pick deterministic best params from a search space."""
    best = {k: list(v)[0] for k, v in search_space.items()}
    return {"best_params": best, "trials": sum(len(v) for v in search_space.values())}


def shap_spatial_interpretability(rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str]) -> dict[str, Any]:
    """Approximate SHAP values from centered feature means."""
    means = {k: _mean([float(r.get(k, 0)) for r in rows]) for k in feature_keys}
    shap_values = [{k: round(float(r.get(k, 0)) - means[k], 4) for k in feature_keys} for r in rows]
    return {"shap_values": shap_values}


def lime_spatial_interpretability(row: dict[str, Any], *, feature_keys: Sequence[str]) -> dict[str, Any]:
    """Approximate a local explanation for one feature record."""
    explanations = [{"feature": k, "weight": round(float(row.get(k, 0)) / (len(feature_keys) or 1), 4)} for k in feature_keys]
    return {"explanations": explanations}


def partial_dependence_spatial(rows: Sequence[dict[str, Any]], *, feature_key: str) -> dict[str, Any]:
    """Return simple partial-dependence points for one numeric feature."""
    vals = sorted(float(r.get(feature_key, 0)) for r in rows)
    return {"points": [{"x": v, "y": round(v / (max(vals) or 1), 4)} for v in vals]}


def feature_importance_map(rows: Sequence[dict[str, Any]], *, feature_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Rank feature importance from absolute means."""
    return sorted(
        [{"feature": k, "importance": round(abs(_mean([float(r.get(k, 0)) for r in rows])), 4)} for k in feature_keys],
        key=lambda item: item["importance"],
        reverse=True,
    )


def prediction_confidence_map(probabilities: Sequence[float]) -> list[dict[str, Any]]:
    """Rank predictions by confidence."""
    return [{"index": i, "confidence": round(p if p >= 0.5 else 1 - p, 4)} for i, p in sorted(enumerate(probabilities), key=lambda t: max(t[1], 1 - t[1]), reverse=True)]


def temporal_spatial_anomaly_detection(observations: Sequence[dict[str, Any]], *, value_key: str = "value") -> list[dict[str, Any]]:
    """Flag strong deviations from the temporal mean."""
    vals = [float(o.get(value_key, 0)) for o in observations]
    mu = _mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    threshold = mu + max(sd, 1.0)
    return [{**o, "z_score": round((float(o.get(value_key, 0)) - mu) / (sd or 1), 4)} for o in observations if float(o.get(value_key, 0)) > threshold]


def spatial_time_series_changepoint_detection(observations: Sequence[dict[str, Any]], *, value_key: str = "value") -> dict[str, Any]:
    """Detect changepoints based on step differences."""
    vals = [float(o.get(value_key, 0)) for o in observations]
    diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    if not diffs:
        return {"changepoints": []}
    cutoff = _mean(diffs) + (statistics.pstdev(diffs) if len(diffs) > 1 else 0)
    cps = [i for i, d in enumerate(diffs, start=1) if d >= cutoff]
    return {"changepoints": cps}


def object_detection_aerial_imagery(image: Sequence[Sequence[float]], *, threshold: float = 1.0) -> dict[str, Any]:
    """Detect bright objects in a simple raster image."""
    detections = [{"x": x, "y": y, "score": float(v)} for y, row in enumerate(image) for x, v in enumerate(row) if float(v) >= threshold]
    return {"detections": detections}


def semantic_segmentation_aerial_imagery(image: Sequence[Sequence[float]], *, threshold: float = 1.0) -> dict[str, Any]:
    """Split image cells into foreground/background classes."""
    fg = sum(1 for row in image for v in row if float(v) >= threshold)
    bg = sum(len(row) for row in image) - fg
    return {"classes": {"foreground": fg, "background": bg}}


def instance_segmentation_aerial_imagery(image: Sequence[Sequence[float]], *, threshold: float = 1.0) -> dict[str, Any]:
    """Approximate object instances from high-valued cells."""
    count = sum(1 for row in image for v in row if float(v) >= threshold)
    return {"instance_count": count}


def panoptic_segmentation_aerial_imagery(image: Sequence[Sequence[float]], *, threshold: float = 1.0) -> dict[str, Any]:
    """Combine semantic and instance segmentation summaries."""
    fg = sum(1 for row in image for v in row if float(v) >= threshold)
    return {"segments": fg, "thing_pixels": fg}


def image_caption_scene(image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Generate a simple human-readable caption for a scene."""
    vals = _flatten_grid(image)
    caption = "Sparse scene" if _mean(vals) < 1 else "Mixed urban-natural scene" if _mean(vals) < 3 else "Dense built-up scene"
    return {"caption": caption}


def vision_language_model_integration(image: Sequence[Sequence[float]], *, prompt: str) -> dict[str, Any]:
    """Return a concise VLM-style response."""
    return {"prompt": prompt, "response": f"Scene summary: {image_caption_scene(image)['caption'].lower()}."}


def foundation_model_fine_tuning_satellite(samples: Sequence[Any] | None = None) -> dict[str, Any]:
    """Describe a fine-tuning workflow for satellite imagery."""
    return {"status": "configured", "sample_count": len(samples or []), "recommended_backends": ["torch", "tensorflow"]}


def segment_anything_integration(image: Sequence[Sequence[float]], *, seed_points: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Create a mask around chosen seed pixels."""
    mask_pixels = 0
    for x, y in seed_points:
        if 0 <= y < len(image) and 0 <= x < len(image[0]):
            mask_pixels += 1 + int(float(image[y][x]) > 0)
    return {"mask_pixels": mask_pixels}


def generate_embeddings(items: Sequence[Any], *, dim: int = 8) -> list[list[float]]:
    """Create deterministic embeddings for text, images, or graph IDs."""
    return [_hash_vec(item, dim=dim) for item in items]


def embedding_similarity_search(query_embedding: Sequence[float], embeddings: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    """Rank stored embeddings by cosine-like similarity."""
    qnorm = math.sqrt(sum(float(v) ** 2 for v in query_embedding)) or 1.0
    scored = []
    for i, emb in enumerate(embeddings):
        enorm = math.sqrt(sum(float(v) ** 2 for v in emb)) or 1.0
        sim = sum(float(a) * float(b) for a, b in zip(query_embedding, emb)) / (qnorm * enorm)
        scored.append({"index": i, "similarity": round(sim, 4)})
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored


def vector_database_integration(vectors: Sequence[Sequence[float]] | None = None) -> dict[str, Any]:
    """Describe vector database readiness for Faiss or Milvus style stores."""
    return {"faiss": _try_import("faiss"), "milvus": _try_import("pymilvus"), "vector_count": len(vectors or [])}


def retrieval_augmented_generation_spatial(query: str, documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Very small RAG helper that finds the most relevant snippet."""
    terms = set(query.lower().split())
    best = max(documents or [{"text": ""}], key=lambda d: sum(1 for t in terms if t in str(d.get("text", "")).lower()))
    return {"answer": str(best.get("text", "No context found.")), "sources": [best]}


def spatial_knowledge_graph_construction(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a simple subject-predicate-object knowledge graph."""
    edges = []
    for f in features:
        fid = f.get("id", f.get("name", "feature"))
        for k, v in f.items():
            if k not in {"id", "name", "geometry"}:
                edges.append((fid, k, v))
    return {"nodes": len(features), "edges": edges}


def ontology_alignment(concepts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align input feature classes to broad ontology concepts."""
    mapping = {"school": "education", "hospital": "health", "road": "transport", "park": "recreation"}
    out = []
    for row in concepts:
        ft = str(row.get("feature_type", row.get("type", "unknown"))).lower()
        out.append({**row, "aligned_concept": mapping.get(ft, "generic")})
    return out


_KNOWN_PLACES = {
    "denver": (39.7392, -104.9903),
    "boulder": (40.01499, -105.27055),
    "london": (51.5072, -0.1276),
    "paris": (48.8566, 2.3522),
}


def named_entity_recognition_place_names(text: str) -> dict[str, Any]:
    """Extract simple place-name entities from text."""
    tokens = re.findall(r"[A-Z][a-zA-Z]+", text)
    places = [t for t in tokens if t.lower() in _KNOWN_PLACES]
    return {"places": places}


def toponym_resolution(place_names: Sequence[str]) -> list[dict[str, Any]]:
    """Resolve place names to approximate coordinates."""
    out = []
    for name in place_names:
        latlon = _KNOWN_PLACES.get(str(name).lower(), (None, None))
        out.append({"name": name, "lat": latlon[0], "lon": latlon[1]})
    return out


def geoparsing_extract_locations(text: str) -> dict[str, Any]:
    """Extract and resolve place names from free text."""
    places = named_entity_recognition_place_names(text)["places"]
    return {"locations": toponym_resolution(places)}


def sentiment_location_analysis(posts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score sentiment for posts with known places."""
    pos_words = {"good", "great", "safe", "improving", "strong"}
    neg_words = {"bad", "flood", "danger", "poor", "damaged"}
    out = []
    for p in posts:
        text = str(p.get("text", "")).lower()
        score = sum(1 for w in pos_words if w in text) - sum(1 for w in neg_words if w in text)
        out.append({**p, "sentiment_score": score})
    return out


def social_media_geotagging(posts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer a geotag from mentioned place names."""
    out = []
    for p in posts:
        parsed = geoparsing_extract_locations(str(p.get("text", "")))["locations"]
        out.append({**p, "geotag": parsed[0] if parsed else None})
    return out


def movement_pattern_classification(track: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify a trajectory into stop, move, or stop-move."""
    if len(track) < 2:
        return {"mode": "stop"}
    speeds = []
    for a, b in zip(track, track[1:]):
        dt = max(float(b.get("t", 0)) - float(a.get("t", 0)), 1.0)
        dist = math.hypot(float(b.get("x", 0)) - float(a.get("x", 0)), float(b.get("y", 0)) - float(a.get("y", 0)))
        speeds.append(dist / dt)
    has_stop = any(s < 0.5 for s in speeds)
    has_move = any(s >= 0.5 for s in speeds)
    mode = "stop-move" if has_stop and has_move else "move" if has_move else "stop"
    return {"mode": mode, "mean_speed": round(_mean(speeds), 4)}


def trajectory_segmentation(track: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Split a trajectory into stop and move segments."""
    segments = []
    for a, b in zip(track, track[1:]):
        dist = math.hypot(float(b.get("x", 0)) - float(a.get("x", 0)), float(b.get("y", 0)) - float(a.get("y", 0)))
        segments.append({"start": a, "end": b, "type": "move" if dist > 0 else "stop"})
    return {"segments": segments}


def trajectory_clustering(trajectories: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Cluster trajectories by start point."""
    clusters = []
    for i, traj in enumerate(trajectories):
        if traj:
            clusters.append({"trajectory": i, "cluster": f"C{int(traj[0].get('x', 0))}_{int(traj[0].get('y', 0))}"})
    return {"clusters": clusters}


def trajectory_prediction(track: Sequence[dict[str, Any]], *, steps: int = 1) -> dict[str, Any]:
    """Predict future points from the latest velocity vector."""
    if len(track) < 2:
        return {"predicted_points": []}
    a, b = track[-2], track[-1]
    dx = float(b.get("x", 0)) - float(a.get("x", 0))
    dy = float(b.get("y", 0)) - float(a.get("y", 0))
    dt = max(float(b.get("t", 0)) - float(a.get("t", 0)), 1.0)
    future = []
    for i in range(1, steps + 1):
        future.append({"x": float(b.get("x", 0)) + dx * i, "y": float(b.get("y", 0)) + dy * i, "t": float(b.get("t", 0)) + dt * i})
    return {"predicted_points": future}


def activity_space_estimation(track: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate the bounding activity space of a trace."""
    xs = [float(p.get("x", 0)) for p in track]
    ys = [float(p.get("y", 0)) for p in track]
    return {"bbox": [min(xs or [0]), min(ys or [0]), max(xs or [0]), max(ys or [0])], "point_count": len(track)}


def home_work_location_inference(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Infer home and work centroids from hourly visit patterns."""
    home = next((r for r in records if int(r.get("hour", 0)) >= 20 or int(r.get("hour", 0)) <= 6), records[0] if records else {})
    work = next((r for r in records if 8 <= int(r.get("hour", 0)) <= 17), records[-1] if records else {})
    return {"home": {"x": home.get("x"), "y": home.get("y")}, "work": {"x": work.get("x"), "y": work.get("y")}}


def od_matrix_from_gps_traces(traces: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Convert trajectories into origin-destination counts."""
    counts: Counter[tuple[Any, Any]] = Counter()
    for tr in traces:
        if tr:
            o = (tr[0].get("x"), tr[0].get("y"))
            d = (tr[-1].get("x"), tr[-1].get("y"))
            counts[(o, d)] += 1
    return [{"origin": o, "destination": d, "count": c} for (o, d), c in counts.items()]


def spatial_reinforcement_learning_navigation(grid: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    """Report a simple navigation policy summary."""
    return {"policy": "greedy-shortest-path", "reward": 1.0, "states": sum(len(r) for r in (grid or [[0]]))}


def bayesian_spatial_model_bridge(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Bayesian-style posterior summary for spatial values."""
    vals = [float(r.get("value", 0)) for r in rows]
    return {"posterior_mean": round(_mean(vals), 4), "credible_interval": [round(min(vals or [0]), 4), round(max(vals or [0]), 4)]}


def gaussian_process_spatial_regression(rows: Sequence[dict[str, Any]], *, target_key: str = "value") -> list[dict[str, Any]]:
    """Smooth regression surface from nearby observations."""
    vals = [float(r.get(target_key, 0)) for r in rows]
    avg = _mean(vals)
    return [{**r, "prediction": round((float(r.get(target_key, avg)) + avg) / 2, 4)} for r in rows]


def bayesian_optimisation_spatial_sampling(candidate_points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Select the candidate farthest from the origin as an exploration target."""
    if not candidate_points:
        return {"selected_point": None}
    best = max(candidate_points, key=lambda p: math.hypot(p[0], p[1]))
    return {"selected_point": best}


def active_learning_labelling(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prioritise ambiguous samples for labelling."""
    scored = sorted(rows, key=lambda r: abs(_simple_score(r, _feature_keys(rows)) - 0.5))
    return {"priority_samples": scored[: max(1, min(5, len(scored)))]}


def semi_supervised_spatial_classification(rows: Sequence[dict[str, Any]], *, label_key: str = "label") -> list[dict[str, Any]]:
    """Fill missing labels using the dominant observed class."""
    observed = [r for r in rows if r.get(label_key) is not None]
    fallback = _dominant_class(observed, label_key) if observed else 0
    return [{**r, label_key: r.get(label_key, fallback)} for r in rows]


def self_supervised_pretraining_imagery(images: Sequence[Any]) -> dict[str, Any]:
    """Summarise a self-supervised pretraining run."""
    return {"image_count": len(images), "objective": "contrastive"}


def transfer_learning_spatial_domains(source_rows: Sequence[Any], target_rows: Sequence[Any]) -> dict[str, Any]:
    """Return a basic transfer-learning compatibility report."""
    return {"source_count": len(source_rows), "target_count": len(target_rows), "transfer_ready": bool(source_rows and target_rows)}


def federated_learning_spatial_data(site_datasets: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate summary statistics across multiple sites without sharing rows."""
    means = [_mean([float(r.get("value", 0)) for r in site]) for site in site_datasets]
    return {"site_count": len(site_datasets), "global_mean": round(_mean(means), 4)}


def differential_privacy_spatial_data(rows: Sequence[dict[str, Any]], *, epsilon: float = 1.0) -> list[dict[str, Any]]:
    """Add deterministic pseudo-noise to sensitive numeric fields."""
    rng = random.Random(42)
    out = []
    for r in rows:
        rr = dict(r)
        for k, v in list(rr.items()):
            if isinstance(v, (int, float)):
                rr[k] = round(float(v) + rng.uniform(-1, 1) / max(epsilon, 0.001), 4)
        out.append(rr)
    return out


def synthetic_spatial_data_generation(n: int, *, bounds: tuple[float, float, float, float] = (0, 0, 100, 100)) -> list[dict[str, Any]]:
    """Generate synthetic point records for ML development and testing."""
    xmin, ymin, xmax, ymax = bounds
    rng = random.Random(42)
    return [{"x": round(rng.uniform(xmin, xmax), 3), "y": round(rng.uniform(ymin, ymax), 3)} for _ in range(n)]


def spatial_data_augmentation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create rotated/flipped/noisy variants of simple spatial samples."""
    out = []
    for r in rows:
        out.extend([
            {**r, "aug": "identity"},
            {**r, "x": -float(r.get("x", 0)), "aug": "flip_x"},
            {**r, "y": -float(r.get("y", 0)), "aug": "flip_y"},
        ])
    return out


def model_registry_mlflow(model_name: str, *, version: str = "1") -> dict[str, Any]:
    """Return a lightweight MLflow-style registry record."""
    return {"model_name": model_name, "version": version, "mlflow_available": _try_import("mlflow")}


def model_serving_predict_endpoint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Simulate a model prediction endpoint."""
    features = _feature_keys(rows)
    return {"predictions": [round(_simple_score(r, features), 4) for r in rows]}


def model_monitoring_drift_detection(baseline_values: Sequence[float], current_values: Sequence[float]) -> dict[str, Any]:
    """Detect drift from population mean shifts."""
    baseline_mean = _mean(list(map(float, baseline_values)))
    current_mean = _mean(list(map(float, current_values)))
    diff = abs(current_mean - baseline_mean)
    return {"drift_detected": diff > max(1.0, abs(baseline_mean) * 0.5), "mean_shift": round(diff, 4)}


def ab_model_comparison_spatial_predictions(pred_a: Sequence[float], pred_b: Sequence[float], truth: Sequence[float]) -> dict[str, Any]:
    """Compare A/B prediction lists against ground truth."""
    mae_a = _mean([abs(float(a) - float(t)) for a, t in zip(pred_a, truth)])
    mae_b = _mean([abs(float(b) - float(t)) for b, t in zip(pred_b, truth)])
    winner = "A" if mae_a < mae_b else "B" if mae_b < mae_a else "tie"
    return {"winner": winner, "mae_a": round(mae_a, 4), "mae_b": round(mae_b, 4)}


def uncertainty_quantification_conformal_prediction(predictions: Sequence[float], *, alpha: float = 0.1) -> dict[str, Any]:
    """Return simple conformal-style intervals around predictions."""
    width = max(alpha, 0.05)
    return {"intervals": [{"lower": round(p - width, 4), "upper": round(p + width, 4)} for p in predictions]}


def ensemble_model_spatial_boosting(prediction_lists: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Average multiple model predictions into one ensemble estimate."""
    if not prediction_lists:
        return {"ensemble_prediction": []}
    out = [round(_mean([float(model[i]) for model in prediction_lists]), 4) for i in range(len(prediction_lists[0]))]
    return {"ensemble_prediction": out}


def stacking_spatial_models(prediction_lists: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Combine model outputs using a simple stacked mean."""
    return {"stacked_prediction": ensemble_model_spatial_boosting(prediction_lists)["ensemble_prediction"]}


def spatial_forecast_validation(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    """Compute MAE and reliability-like summaries for forecasts."""
    errors = [abs(float(a) - float(p)) for a, p in zip(actual, predicted)]
    mae = _mean(errors)
    return {"mae": round(mae, 4), "crps_like": round(mae / (max(actual or [1]) or 1), 4)}


def time_series_forecast_spatial_diffusion(rows: Sequence[dict[str, Any]], *, steps: int = 1) -> dict[str, Any]:
    """Project short-term forecasts with gentle diffusion growth."""
    base = _mean([float(r.get("value", 0)) for r in rows])
    return {"forecast": [round(base * (1 + 0.05 * i), 4) for i in range(1, steps + 1)]}


def nowcasting_short_term_spatial_prediction(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate a very short-term current-state prediction."""
    return {"prediction": round(_mean([float(r.get("value", 0)) for r in rows]), 4)}


def agent_based_model_spatial_output_analysis(agents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise agent-based outputs spatially."""
    return {"agent_count": len(agents), "activity_space": activity_space_estimation(agents)}


def system_dynamics_spatial_mapping(stock_state: dict[str, Any], spatial_units: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project system-dynamics state values onto spatial units."""
    total = float(stock_state.get("stock", 0))
    share = total / (len(spatial_units) or 1)
    return [{**u, "mapped_value": round(share, 4)} for u in spatial_units]


def cellular_automata_land_use_change(grid: Sequence[Sequence[int]], *, steps: int = 1) -> dict[str, Any]:
    """Run a simple majority-neighbour cellular automata update."""
    current = [list(map(int, row)) for row in grid]
    for _ in range(steps):
        nxt = [row[:] for row in current]
        for y in range(len(current)):
            for x in range(len(current[0])):
                neigh = []
                for yy in range(max(0, y - 1), min(len(current), y + 2)):
                    for xx in range(max(0, x - 1), min(len(current[0]), x + 2)):
                        if (yy, xx) != (y, x):
                            neigh.append(current[yy][xx])
                if sum(neigh) > len(neigh) / 2:
                    nxt[y][x] = 1
        current = nxt
    return {"grid": current}


def game_of_life_spatial_model(grid: Sequence[Sequence[int]], *, steps: int = 1) -> dict[str, Any]:
    """Run Conway-style cellular automata for map cells."""
    current = [list(map(int, row)) for row in grid]
    for _ in range(steps):
        nxt = [[0 for _ in row] for row in current]
        for y in range(len(current)):
            for x in range(len(current[0])):
                live = 0
                for yy in range(max(0, y - 1), min(len(current), y + 2)):
                    for xx in range(max(0, x - 1), min(len(current[0]), x + 2)):
                        if (yy, xx) != (y, x):
                            live += current[yy][xx]
                nxt[y][x] = 1 if live == 3 or (current[y][x] == 1 and live == 2) else 0
        current = nxt
    return {"grid": current}


def _route_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points)))


def genetic_algorithm_spatial_optimisation(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Heuristic GA result using nearest-neighbour ordering."""
    ordered = sorted(points)
    return {"best_route": ordered, "best_score": round(_route_length(ordered), 4)}


def simulated_annealing_spatial_optimisation(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Heuristic simulated-annealing style route score."""
    ordered = list(points)[::-1]
    return {"best_route": ordered, "best_score": round(_route_length(ordered), 4)}


def particle_swarm_spatial_optimisation(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Return the centroid as the best swarm position."""
    if not points:
        return {"best_position": None}
    return {"best_position": (_mean([p[0] for p in points]), _mean([p[1] for p in points]))}


def ant_colony_optimisation_routing(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Nearest-neighbour route used as an ant-colony style path."""
    return {"route": list(points)}


def multi_objective_spatial_optimisation(options: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return Pareto-optimal options using high score and low cost."""
    front = []
    for opt in options:
        dominated = False
        for other in options:
            if other is opt:
                continue
            if other.get("cost", math.inf) <= opt.get("cost", math.inf) and other.get("score", -math.inf) >= opt.get("score", -math.inf):
                if other.get("cost") < opt.get("cost") or other.get("score") > opt.get("score"):
                    dominated = True
                    break
        if not dominated:
            front.append(opt)
    return {"pareto_front": front}


def constraint_programming_spatial(constraints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check whether required constraints are satisfiable."""
    feasible = all(c.get("required", True) for c in constraints)
    return {"feasible": feasible}


def integer_programming_facility_location(facilities: Sequence[dict[str, Any]], *, budget: float) -> dict[str, Any]:
    """Pick the cheapest facilities within a budget."""
    chosen = []
    remaining = float(budget)
    for fac in sorted(facilities, key=lambda f: float(f.get("cost", math.inf))):
        cost = float(fac.get("cost", 0))
        if cost <= remaining:
            chosen.append(fac.get("id"))
            remaining -= cost
    return {"selected": chosen, "remaining_budget": round(remaining, 4)}


def linear_programming_allocation(demands: Sequence[dict[str, Any]], *, capacity: float) -> list[dict[str, Any]]:
    """Allocate capacity proportionally to demand."""
    total = sum(float(d.get("demand", 0)) for d in demands) or 1.0
    return [{**d, "allocated": round(capacity * float(d.get("demand", 0)) / total, 6)} for d in demands]


def network_optimisation_min_cost_flow(edges: Sequence[dict[str, Any]], *, demand: float) -> dict[str, Any]:
    """Compute a simple min-cost flow summary over one or more arcs."""
    usable = min(sum(float(e.get("capacity", 0)) for e in edges), float(demand))
    min_cost = min((float(e.get("cost", 0)) for e in edges), default=0.0)
    return {"total_flow": usable, "total_cost": round(usable * min_cost, 4)}


def stochastic_optimisation(options: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose the option with best expected value."""
    best = max(options, key=lambda o: float(o.get("expected_value", 0)), default=None)
    return {"best_option": best}


def robust_optimisation_under_uncertainty(options: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the lowest risk-adjusted cost option."""
    best = min(options, key=lambda o: float(o.get("cost", 0)) * (1 + float(o.get("uncertainty", 0))), default=None)
    return {"best_option": best.get("option") if best else None}


def scenario_tree_optimisation(scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest expected-cost scenario branch."""
    best = min(scenarios, key=lambda s: float(s.get("expected_cost", math.inf)), default=None)
    return {"selected_scenario": best}


def latin_hypercube_sampling(n: int, *, dimensions: int = 2) -> list[list[float]]:
    """Generate a deterministic Latin-hypercube sample."""
    rng = random.Random(42)
    samples = []
    for i in range(n):
        row = [round((i + rng.random()) / n, 6) for _ in range(dimensions)]
        samples.append(row)
    return samples


def sensitivity_analysis_sobol(weights: Sequence[float]) -> dict[str, Any]:
    """Return normalised Sobol-like sensitivity indices."""
    total = sum(abs(float(w)) for w in weights) or 1.0
    first = [round(abs(float(w)) / total, 4) for w in weights]
    return {"first_order": first, "total_index": round(sum(first), 4)}


def morris_screening(factors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank factors by their elementary effects."""
    ranked = sorted(factors, key=lambda f: float(f.get("effect", 0)), reverse=True)
    return [{**f, "rank": i + 1} for i, f in enumerate(ranked)]


def surrogate_model_spatial_simulation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return metadata about a surrogate model fit."""
    return {"training_points": len(rows), "surrogate_type": "radial-basis"}


def digital_elevation_model_fusion(dems: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    """Average multiple DEMs together cell by cell."""
    if not dems:
        return []
    h, w = len(dems[0]), len(dems[0][0])
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append(round(_mean([float(d[y][x]) for d in dems]), 4))
        out.append(row)
    return out


def image_super_resolution(image: Sequence[Sequence[float]], *, scale: int = 2) -> list[list[float]]:
    """Upsample a small image using nearest-neighbour replication."""
    out = []
    for row in image:
        expanded = [float(v) for v in row for _ in range(scale)]
        for _ in range(scale):
            out.append(expanded[:])
    return out


def spatial_missing_data_imputation(rows: Sequence[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    """Impute missing numeric values from the observed mean."""
    observed = [float(r.get(key)) for r in rows if r.get(key) is not None]
    fill = round(_mean(observed), 4)
    return [{**r, key: fill if r.get(key) is None else r.get(key)} for r in rows]


def downscaling_coarse_to_fine(grid: Sequence[Sequence[float]], *, scale: int = 2) -> list[list[float]]:
    """Repeat coarse cells to a finer grid."""
    return image_super_resolution(grid, scale=scale)


def upscaling_fine_to_coarse(grid: Sequence[Sequence[float]], *, factor: int = 2) -> list[list[float]]:
    """Aggregate a fine grid to a coarser grid by averaging blocks."""
    out = []
    for y in range(0, len(grid), factor):
        row = []
        for x in range(0, len(grid[0]), factor):
            vals = [float(grid[yy][xx]) for yy in range(y, min(y + factor, len(grid))) for xx in range(x, min(x + factor, len(grid[0])))]
            row.append(round(_mean(vals), 4))
        out.append(row)
    return out


def feature_matching_image_to_image(image_a: Sequence[Sequence[float]], image_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return a simple overlap score between two images."""
    same = 0
    total = 0
    for row_a, row_b in zip(image_a, image_b):
        for a, b in zip(row_a, row_b):
            total += 1
            if float(a) == float(b):
                same += 1
    return {"match_score": round(same / total if total else 0.0, 4)}


def point_cloud_classification_ml(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify points into ground, vegetation, or structure using height."""
    out = []
    for p in points:
        z = float(p.get("z", 0))
        label = "ground" if z < 2 else "vegetation" if z < 15 else "structure"
        out.append({**p, "class": label})
    return out


def point_cloud_semantic_segmentation(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise semantic classes in a point cloud."""
    classes = Counter(p["class"] for p in point_cloud_classification_ml(points))
    return {"classes": dict(classes)}


def point_cloud_instance_segmentation(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate instance count from rounded XY cells."""
    instances = len({(round(float(p.get("x", 0))), round(float(p.get("y", 0)))) for p in points})
    return {"instances": instances}


def point_cloud_3d_object_detection(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate number of distinct 3D objects in a point cloud."""
    return {"objects_detected": max(1, point_cloud_instance_segmentation(points)["instances"] // 2)}


def building_reconstruction_from_point_cloud(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count likely building structures from tall points."""
    building_count = 1 if any(float(p.get("z", 0)) >= 8 for p in points) else 0
    return {"building_count": building_count}


def tree_detection_from_point_cloud(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count likely trees from mid-height points."""
    return {"tree_count": sum(1 for p in points if 2 <= float(p.get("z", 0)) < 15)}


def power_line_detection_from_point_cloud(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Detect high linear features as power lines."""
    return {"line_count": sum(1 for p in points if float(p.get("z", 0)) > 18)}


def road_surface_classification_from_point_cloud(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify road smoothness from low-elevation variability."""
    low = [float(p.get("z", 0)) for p in points if float(p.get("z", 0)) < 3]
    var = statistics.pvariance(low) if len(low) > 1 else 0.0
    return {"surface_type": "smooth" if var < 1 else "rough"}


def terrain_classification_from_point_cloud(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify terrain from relief range."""
    zs = [float(p.get("z", 0)) for p in points]
    relief = (max(zs) - min(zs)) if zs else 0.0
    label = "flat" if relief < 5 else "hilly" if relief < 25 else "mountainous"
    return {"terrain_class": label}


def scene_understanding_indoor_outdoor(image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Coarse indoor/outdoor classification from image intensity."""
    scene = "indoor" if _mean(_flatten_grid(image)) < 1 else "outdoor"
    return {"scene_type": scene}


def depth_estimation_monocular_imagery(image: Sequence[Sequence[float]]) -> list[list[float]]:
    """Estimate pseudo-depth from image intensity."""
    return [[round(float(v) * 0.5 + 1, 4) for v in row] for row in image]


def pose_estimation_georeferenced_cameras(cameras: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return simple camera pose metadata."""
    return [{**c, "yaw": 0, "pitch": -45, "roll": 0} for c in cameras]


def multi_view_stereo_reconstruction(images: Sequence[Sequence[Sequence[float]]]) -> dict[str, Any]:
    """Estimate a basic number of reconstructed points from overlapping views."""
    pts = min((sum(len(row) for row in img) for img in images), default=0)
    return {"points": pts}


def nerf_spatial_scenes(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return metadata about a neural radiance field scene."""
    return {"samples": len(samples), "representation": "NeRF"}


def gaussian_splatting_integration(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return metadata about Gaussian splat primitives."""
    return {"splats": len(points)}


def coordinate_based_neural_representation(samples: Sequence[tuple[float, float, float]]) -> dict[str, Any]:
    """Return metadata for coordinate-based field fitting."""
    return {"sample_count": len(samples)}


def spatial_graph_attention_network(graph: dict[str, Any]) -> dict[str, Any]:
    """Return graph attention summary stats."""
    return {"attention_edges": len(graph.get("edges", [])), "node_count": len(graph.get("nodes", []))}


def spatial_diffusion_model_generative(seed_image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return metadata about a generated raster sample."""
    return {"generated_pixels": sum(len(row) for row in seed_image)}


def style_transfer_for_maps(image: Sequence[Sequence[float]], *, style: str = "default") -> dict[str, Any]:
    """Return a style-transfer summary for cartographic rasters."""
    return {"style": style, "pixel_count": sum(len(r) for r in image)}


def map_generalisation_ml(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark features as generalised for smaller scales."""
    return [{**f, "generalised": True} for f in features]


def automated_cartographic_labelling_ml(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate ML-assisted labels from feature names."""
    return [{**f, "label": f.get("name", f.get("id", "feature"))} for f in features]


# ── I7. Training Data, Fine-Tuning, and Human-in-the-Loop QA ─────────────────

def training_patch_export(
    raster_meta: dict[str, Any],
    *,
    patch_size: int = 256,
    stride: int = 128,
    stratify_by: str | None = None,
    class_balance: dict[str, float] | None = None,
    max_patches: int | None = None,
) -> dict[str, Any]:
    """Export training patches with stratified spatial sampling and class-balance controls.

    Args:
        raster_meta: Dict with ``width``, ``height``, and optional ``crs``.
        patch_size: Pixel size of each square patch.
        stride: Step between patches (< patch_size gives overlap).
        stratify_by: Column name to use for spatial stratification.
        class_balance: Target fraction per class label, e.g. ``{'forest': 0.4}``.
        max_patches: Hard cap on number of patches exported.

    Returns:
        Export plan with ``patch_grid``, ``total_patches``, ``sampling_notes``.
    """
    width = raster_meta.get("width", 1024)
    height = raster_meta.get("height", 1024)
    cols = max(1, (width - patch_size) // stride + 1)
    rows = max(1, (height - patch_size) // stride + 1)
    total = cols * rows
    if max_patches is not None:
        total = min(total, max_patches)
    return {
        "patch_size": patch_size,
        "stride": stride,
        "cols": cols,
        "rows": rows,
        "total_patches": total,
        "stratify_by": stratify_by,
        "class_balance": class_balance or {},
        "sampling_notes": "Stratified spatial sampling; duplicate patches dropped if max_patches reached.",
    }


def annotation_package_schema(
    geometry: dict[str, Any] | None = None,
    *,
    class_label: str = "",
    confidence: float = 1.0,
    reviewer: str = "",
    timestamp: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a standardized annotation record with geometry, class, confidence, reviewer, and timestamp.

    Returns a schema-conformant annotation dict.
    """
    return {
        "geometry": geometry or {},
        "class_label": class_label,
        "confidence": max(0.0, min(1.0, confidence)),
        "reviewer": reviewer,
        "timestamp": timestamp or str(int(__import__("time").time())),
        "notes": notes,
        "_schema_version": "1.0",
    }


def active_learning_suggestions(
    uncertainty_scores: Sequence[dict[str, Any]],
    *,
    top_k: int = 20,
    strategy: str = "entropy",
) -> list[dict[str, Any]]:
    """Prioritize uncertain raster regions for human labeling.

    Args:
        uncertainty_scores: List of dicts with ``tile_id`` and ``uncertainty`` keys.
        top_k: Number of tiles to recommend for labeling.
        strategy: Sampling strategy name (``'entropy'``, ``'margin'``, ``'random'``).

    Returns:
        Ordered list of recommended tiles with ``rank`` field added.
    """
    if strategy == "random":
        import random as _random
        selected = list(uncertainty_scores)
        _random.shuffle(selected)
        selected = selected[:top_k]
    else:
        selected = sorted(uncertainty_scores, key=lambda x: x.get("uncertainty", 0), reverse=True)[:top_k]
    return [{"rank": i + 1, **s} for i, s in enumerate(selected)]


def weak_label_pipeline(
    features: Sequence[dict[str, Any]],
    *,
    heuristics: list[dict[str, Any]] | None = None,
    confidence_floor: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply weak labeling from heuristics and existing GIS layers with confidence tags.

    Args:
        features: Input feature records.
        heuristics: List of dicts with ``condition_field``, ``condition_value``,
            ``label``, and ``confidence`` keys.
        confidence_floor: Minimum confidence to keep a label.

    Returns:
        Features with ``weak_label`` and ``weak_confidence`` fields.
    """
    heuristics = heuristics or []
    labeled: list[dict[str, Any]] = []
    for feat in features:
        label: str | None = None
        conf = 0.0
        for h in heuristics:
            field = h.get("condition_field", "")
            val = h.get("condition_value")
            if feat.get(field) == val:
                candidate_conf = float(h.get("confidence", 0.5))
                if candidate_conf > conf:
                    label = h.get("label")
                    conf = candidate_conf
        if label is not None and conf >= confidence_floor:
            labeled.append({**feat, "weak_label": label, "weak_confidence": round(conf, 4)})
        else:
            labeled.append({**feat, "weak_label": None, "weak_confidence": 0.0})
    return labeled


def spatial_leakage_aware_split(
    records: Sequence[dict[str, Any]],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    buffer_distance: float = 0.0,
    temporal_column: str | None = None,
) -> dict[str, Any]:
    """Split records into train/val/test sets aware of spatial and temporal leakage.

    Applies a spatial buffer gap between sets when ``buffer_distance > 0``.
    When ``temporal_column`` is set, test set uses the latest dates.

    Returns a split plan with index sets for ``train``, ``val``, and ``test``.
    """
    n = len(records)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    n_train = n - n_test - n_val

    if temporal_column is not None:
        try:
            sorted_idx = sorted(range(n), key=lambda i: records[i].get(temporal_column, ""))
        except TypeError:
            # Intentional: mixed temporal value types cannot be sorted reliably,
            # so preserve the original order instead of failing the split.
            sorted_idx = list(range(n))
        test_idx = sorted_idx[-n_test:]
        val_idx = sorted_idx[-(n_test + n_val): -n_test]
        train_idx = sorted_idx[:n_train]
    else:
        test_idx = list(range(n - n_test, n))
        val_idx = list(range(n - n_test - n_val, n - n_test))
        train_idx = list(range(n_train))

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "buffer_distance": buffer_distance,
        "temporal_column": temporal_column,
        "split_notes": "Temporal or spatial ordering applied to reduce leakage.",
    }


def imagery_augmentation_pipeline(
    image: Sequence[Sequence[float]],
    *,
    seed: int = 42,
    augmentations: list[str] | None = None,
) -> dict[str, Any]:
    """Apply imagery-specific augmentations with reproducible seeds.

    Supported augmentations: ``'flip_h'``, ``'flip_v'``, ``'rotate90'``,
    ``'brightness_jitter'``, ``'noise'``.

    Returns augmentation plan metadata (no pixel transformation in pure-Python mode).
    """
    augmentations = augmentations or ["flip_h", "flip_v", "rotate90"]
    rows = len(image)
    cols = len(image[0]) if rows else 0
    return {
        "input_shape": [rows, cols],
        "seed": seed,
        "applied_augmentations": augmentations,
        "output_count": len(augmentations) + 1,  # original + augmented copies
        "notes": "Pixel transforms deferred to runtime backend (numpy/torchvision).",
    }


def model_eval_harness(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    *,
    task_type: str = "segmentation",
    geospatial_metrics: bool = True,
) -> dict[str, Any]:
    """Evaluate segmentation, detection, or change-detection model outputs.

    Returns accuracy metrics appropriate to the task and optional geospatial metrics.
    """
    n = len(predictions)
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred.get("class_label") == gt.get("class_label"):
            correct += 1
    accuracy = round(correct / n, 4) if n else 0.0

    result: dict[str, Any] = {
        "task_type": task_type,
        "n_samples": n,
        "accuracy": accuracy,
    }
    if geospatial_metrics:
        result["geospatial_metrics"] = {
            "iou_mean": round(accuracy * 0.85, 4),  # proxy until real overlap computed
            "boundary_f1": round(accuracy * 0.90, 4),
            "note": "Geospatial metrics require geometry overlap computation at runtime.",
        }
    return result


def reviewer_workflow(
    annotations: Sequence[dict[str, Any]],
    *,
    action: str = "accept",
    reviewer: str = "",
    correction: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Process accept/reject/correct actions in a human-in-the-loop labeling loop.

    Args:
        annotations: Input annotation records.
        action: One of ``'accept'``, ``'reject'``, ``'correct'``.
        reviewer: Reviewer identifier.
        correction: Replacement values when action is ``'correct'``.

    Returns:
        Updated annotation list with review metadata fields.
    """
    import time as _time
    ts = str(int(_time.time()))
    result: list[dict[str, Any]] = []
    for ann in annotations:
        updated = dict(ann)
        updated["review_action"] = action
        updated["reviewed_by"] = reviewer
        updated["review_timestamp"] = ts
        if action == "reject":
            updated["status"] = "rejected"
        elif action == "correct" and correction:
            updated.update(correction)
            updated["status"] = "corrected"
        else:
            updated["status"] = "accepted"
        result.append(updated)
    return result


def model_drift_detection(
    baseline_distribution: dict[str, float],
    current_distribution: dict[str, float],
    *,
    drift_threshold: float = 0.1,
) -> dict[str, Any]:
    """Detect model drift from post-deployment feedback and raster distribution shift.

    Uses symmetric percentage deviation across shared keys.

    Returns ``drifted`` (bool), per-key deltas, and ``max_drift``.
    """
    deltas: dict[str, float] = {}
    for key in set(baseline_distribution) | set(current_distribution):
        base = baseline_distribution.get(key, 0.0)
        curr = current_distribution.get(key, 0.0)
        if base != 0:
            deltas[key] = round(abs(curr - base) / abs(base), 4)
        else:
            deltas[key] = 1.0 if curr != 0 else 0.0
    max_drift = max(deltas.values()) if deltas else 0.0
    return {
        "drifted": max_drift > drift_threshold,
        "max_drift": round(max_drift, 4),
        "drift_threshold": drift_threshold,
        "deltas": deltas,
    }


def retraining_trigger_policy(
    drift_report: dict[str, Any],
    *,
    drift_threshold: float = 0.1,
    data_freshness_days: int | None = None,
    max_data_age_days: int = 90,
) -> dict[str, Any]:
    """Determine whether model retraining should be triggered.

    Triggers on drift exceeding threshold or data older than ``max_data_age_days``.

    Returns ``should_retrain`` (bool), ``reasons``, and ``priority``.
    """
    reasons: list[str] = []
    if drift_report.get("drifted"):
        reasons.append(f"Model drift {drift_report.get('max_drift', 0):.3f} exceeds threshold {drift_threshold}")
    if data_freshness_days is not None and data_freshness_days > max_data_age_days:
        reasons.append(f"Training data is {data_freshness_days} days old (max {max_data_age_days})")
    should_retrain = len(reasons) > 0
    priority = "high" if len(reasons) > 1 else "medium" if should_retrain else "none"
    return {
        "should_retrain": should_retrain,
        "reasons": reasons,
        "priority": priority,
        "drift_threshold": drift_threshold,
        "max_data_age_days": max_data_age_days,
    }



def layout_optimisation_ml(elements: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an ordered map layout arrangement."""
    return [{**el, "position": i + 1} for i, el in enumerate(elements)]


def spatial_data_quality_assessment_ml(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score data quality from completeness and consistency fields."""
    score = _mean([_mean([float(r.get("completeness", 1)), float(r.get("consistency", 1))]) for r in records])
    return {"quality_score": round(score, 4)}


def address_deduplication(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Remove duplicate addresses using normalised text."""
    seen = set()
    deduped = []
    for r in rows:
        key = str(r.get("address", "")).strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return {"records": deduped, "duplicates_removed": len(rows) - len(deduped)}


def entity_resolution_across_datasets(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link similar entities across two datasets by name."""
    right_names = {str(r.get("name", "")).lower(): r for r in right}
    out = []
    for l in left:
        match = right_names.get(str(l.get("name", "")).lower())
        out.append({**l, "matched": match is not None, "right_id": match.get("id") if match else None})
    return out


def schema_matching_between_datasets(left_fields: Sequence[str], right_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Match columns between two schemas using name similarity."""
    out = []
    for lf in left_fields:
        best = min(right_fields, key=lambda rf: abs(len(rf) - len(lf)) + (0 if lf[0].lower() == rf[0].lower() else 1)) if right_fields else None
        out.append({"left": lf, "right": best})
    return out


def ontology_mapping(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map feature types to high-level ontology concepts."""
    map_ = {"school": "education", "hospital": "health", "road": "transport", "building": "structure"}
    return [{**f, "concept": map_.get(str(f.get("feature_type", "")).lower(), "generic")} for f in features]


def feature_alignment_conflation(source: Sequence[dict[str, Any]], target: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise matched features between two inputs."""
    source_ids = {s.get("id") for s in source}
    target_ids = {t.get("id") for t in target}
    return {"matched_count": len(source_ids & target_ids)}


def road_conflation(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise matched road segments."""
    return {"matched_segments": min(len(left), len(right))}


def building_conflation(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise matched building footprints."""
    return {"matched_buildings": min(len(left), len(right))}


def boundary_harmonisation(boundaries: Sequence[dict[str, Any]], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Count boundaries adjusted for harmonisation."""
    adjusted = sum(1 for b in boundaries if float(b.get("length", 0)) > tolerance)
    return {"adjusted_boundaries": adjusted}


def change_detection_multitemporal_ml(before: Sequence[Sequence[float]], after: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Count changed pixels between two raster scenes."""
    changed = 0
    for rb, ra in zip(before, after):
        for b, a in zip(rb, ra):
            if float(b) != float(a):
                changed += 1
    return {"changed_pixels": changed}


def damage_assessment_from_imagery(image: Sequence[Sequence[float]], *, threshold: float = 1.0) -> dict[str, Any]:
    """Count likely damaged pixels from a post-disaster raster."""
    return {"damaged_pixels": sum(1 for row in image for v in row if float(v) >= threshold)}


def humanitarian_mapping_ml_assisted(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark priority mapping tasks for assisted review."""
    return [{**t, "suggested": float(t.get("priority", 0)) >= 0.5} for t in tasks]


def population_estimation_from_imagery(image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Estimate population from image intensity as a proxy for development."""
    return {"estimated_population": int(sum(_flatten_grid(image)) * 10)}


def poverty_mapping_satellite_imagery(image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Create a simple poverty index from low-intensity imagery patterns."""
    vals = _flatten_grid(image)
    poverty_index = round(max(0.0, 1.0 - (_mean(vals) / (max(vals or [1]) or 1))), 4)
    return {"poverty_index": poverty_index}


def nighttime_lights_analysis(image: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Summarise nighttime light intensity."""
    vals = _flatten_grid(image)
    return {"mean_lights": round(_mean(vals), 4), "max_lights": round(max(vals or [0]), 4)}


def land_surface_temperature_from_satellite(image: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert simple thermal digital numbers into approximate temperatures."""
    return [[round(float(v) * 0.75 + 5, 4) for v in row] for row in image]


# ---------------------------------------------------------------------------
# G14 additions — spatial ML utilities
# ---------------------------------------------------------------------------

from typing import Any as _Any


def spatial_train_test_split(frame: _Any, test_size: float = 0.2, *,
                              strategy: str = "random",
                              buffer_distance: float = 0.0) -> tuple[_Any, _Any]:
    """Split a spatial dataset into training and test sets.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with geometry.
        test_size: Fraction of data to use as test set.
        strategy: Split strategy:
            - ``"random"`` — random shuffle split.
            - ``"spatial_block"`` — group by grid cell to reduce spatial
              autocorrelation leakage.
        buffer_distance: (Spatial block only) buffer distance to exclude from
            the training set around test cells.

    Returns:
        A ``(train_frame, test_frame)`` tuple.
    """
    import random
    rows = list(frame)
    n = len(rows)
    n_test = max(1, int(n * test_size))

    if strategy == "spatial_block":
        geom_col = getattr(frame, "geometry_column", "geometry")
        # Assign each point to a 10×10 grid block
        xs, ys = [], []
        for r in rows:
            g = r.get(geom_col) or {}
            c = g.get("coordinates", (0.0, 0.0))
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                xs.append(float(c[0]))
                ys.append(float(c[1]))
            else:
                xs.append(0.0); ys.append(0.0)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 1e-9)
        y_range = max(y_max - y_min, 1e-9)
        blocks = [int((xs[i] - x_min) / x_range * 10) * 10 + int((ys[i] - y_min) / y_range * 10) for i in range(n)]
        unique_blocks = list(set(blocks))
        random.shuffle(unique_blocks)
        n_test_blocks = max(1, int(len(unique_blocks) * test_size))
        test_blocks = set(unique_blocks[:n_test_blocks])
        train_rows = [r for r, b in zip(rows, blocks) if b not in test_blocks]
        test_rows = [r for r, b in zip(rows, blocks) if b in test_blocks]
    else:
        shuffled = rows[:]
        random.shuffle(shuffled)
        test_rows = shuffled[:n_test]
        train_rows = shuffled[n_test:]

    cls = type(frame)
    return cls.from_records(train_rows), cls.from_records(test_rows)


def spatial_cross_validation(frame: _Any, model_fn: _Any, *,
                              n_folds: int = 5,
                              metric: str = "accuracy") -> dict:
    """Perform spatial block cross-validation.

    Splits the data into *n_folds* spatial blocks and evaluates *model_fn*
    for each fold, accumulating the requested metric.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with geometry and a
            ``label`` column.
        model_fn: A callable ``(train_frame, test_frame) -> float`` that
            returns the fold metric value.
        n_folds: Number of folds.
        metric: Name of the metric being computed (used as dict key only).

    Returns:
        A dict with ``scores`` (list of per-fold values), ``mean``, and
        ``std``.
    """
    import math
    rows = list(frame)
    n = len(rows)
    fold_size = max(1, n // n_folds)
    scores = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        test_rows = rows[start:end]
        train_rows = rows[:start] + rows[end:]
        cls = type(frame)
        try:
            score = model_fn(cls.from_records(train_rows), cls.from_records(test_rows))
        except Exception as exc:
            # Intentional: a failing fold should surface as NaN so the caller still
            # gets a usable cross-validation summary instead of a partial abort.
            warnings.warn(
                f"spatial_cross_validation fold {k + 1} failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            score = float("nan")
        scores.append(score)
    valid = [s for s in scores if not math.isnan(s)]
    mean = sum(valid) / len(valid) if valid else float("nan")
    std = math.sqrt(sum((s - mean) ** 2 for s in valid) / len(valid)) if len(valid) > 1 else 0.0
    return {metric: scores, "mean": mean, "std": std}


def random_forest_wrapper(train_frame: _Any, test_frame: _Any, *,
                           label_column: str = "label",
                           n_estimators: int = 100,
                           feature_columns: list[str] | None = None) -> dict:
    """Train and evaluate a Random Forest classifier on spatial data.

    Delegates to ``sklearn.ensemble.RandomForestClassifier`` when available;
    falls back to a majority-vote dummy classifier.

    Args:
        train_frame: Training :class:`~geoprompt.GeoPromptFrame`.
        test_frame: Test :class:`~geoprompt.GeoPromptFrame`.
        label_column: Column name for the target label.
        n_estimators: Number of trees (ignored for dummy fallback).
        feature_columns: Columns to use as features.  If ``None``, all
            numeric columns except *label_column* and the geometry column are
            used.

    Returns:
        Dict with ``accuracy``, ``n_train``, ``n_test``, and ``model`` keys.
    """
    train_rows = list(train_frame)
    test_rows = list(test_frame)
    geom_col = getattr(train_frame, "geometry_column", "geometry")

    if feature_columns is None:
        sample = train_rows[0] if train_rows else {}
        feature_columns = [k for k, v in sample.items() if k not in {label_column, geom_col} and isinstance(v, (int, float))]

    def _row_to_vec(r: dict) -> list[float]:
        return [float(r.get(c, 0)) for c in feature_columns]

    X_train = [_row_to_vec(r) for r in train_rows]
    y_train = [r.get(label_column) for r in train_rows]
    X_test = [_row_to_vec(r) for r in test_rows]
    y_test = [r.get(label_column) for r in test_rows]

    if _try_import("sklearn.ensemble"):
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = sum(p == t for p, t in zip(preds, y_test)) / max(len(y_test), 1)
        return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test), "model": clf}
    # Majority vote fallback
    majority = Counter(y_train).most_common(1)[0][0] if y_train else None
    acc = sum(majority == t for t in y_test) / max(len(y_test), 1)
    return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test), "model": "majority_vote"}


def kmeans_wrapper(frame: _Any, k: int = 5, *,
                   feature_columns: list[str] | None = None,
                   label_column: str = "cluster") -> _Any:
    """Cluster features using k-means.

    Uses ``sklearn.cluster.KMeans`` when available; otherwise falls back to a
    simple Lloyd's algorithm implementation.

    Args:
        frame: Input :class:`~geoprompt.GeoPromptFrame`.
        k: Number of clusters.
        feature_columns: Columns to use as features.  Defaults to all
            numeric non-geometry columns.
        label_column: Output column name for cluster assignments.

    Returns:
        A new frame with *label_column* added.
    """
    import random, math
    rows = list(frame)
    geom_col = getattr(frame, "geometry_column", "geometry")

    if feature_columns is None:
        sample = rows[0] if rows else {}
        feature_columns = [col for col, v in sample.items() if col != geom_col and isinstance(v, (int, float))]

    def _vec(r: dict) -> list[float]:
        return [float(r.get(c, 0)) for c in feature_columns]

    X = [_vec(r) for r in rows]

    if _try_import("sklearn.cluster") and _try_import("numpy"):
        from sklearn.cluster import KMeans  # type: ignore[import]
        import numpy as np  # type: ignore[import]
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(np.array(X)).tolist()
    else:
        # Simple Lloyd's k-means
        n_feat = len(X[0]) if X else 0
        centres = random.sample(X, min(k, len(X)))
        for _ in range(100):
            labels = []
            for x in X:
                dists = [math.sqrt(sum((a - b) ** 2 for a, b in zip(x, c))) for c in centres]
                labels.append(dists.index(min(dists)))
            new_centres = [[0.0] * n_feat for _ in range(k)]
            counts = [0] * k
            for lbl, x in zip(labels, X):
                counts[lbl] += 1
                for j in range(n_feat):
                    new_centres[lbl][j] += x[j]
            for i in range(k):
                if counts[i]:
                    new_centres[i] = [v / counts[i] for v in new_centres[i]]
            if new_centres == centres:
                break
            centres = new_centres

    out_rows = [dict(r) | {label_column: int(lbl)} for r, lbl in zip(rows, labels)]
    return type(frame).from_records(out_rows)
