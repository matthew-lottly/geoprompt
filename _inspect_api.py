"""Quick API inspection script."""
import geoprompt as gp
import inspect

to_check = [
    ("domain.py", gp.electric_load_flow),
    ("domain.py", gp.water_distribution_solve),
    ("domain.py", gp.drone_flight_path),
    ("ml.py", gp.random_forest_spatial_prediction),
    ("ml.py", gp.neural_network_integration),
    ("ml.py", gp.segment_anything_integration),
    ("imagery.py", gp.deep_learning_object_detection),
    ("imagery.py", gp.pansharpen_raster),
    ("visualization.py", gp.interactive_web_map_mapbox_gl_js),
    ("performance.py", gp.gpu_accelerated_distance_matrix),
    ("standards.py", gp.ogc_api_features_implementation),
    ("frame.py - real", gp.GeoPromptFrame),
    ("geometry.py - real", gp.geometry_centroid),
    ("equations.py - real", gp.haversine_distance),
    ("stats.py - real", gp.morans_i),
]

for src_name, fn in to_check:
    src = inspect.getsource(fn)
    lines = src.strip().split("\n")
    # Find first non-def, non-docstring, non-comment body line
    in_docstring = False
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            continue
        if '"""' in stripped:
            in_docstring = not in_docstring
            if stripped.count('"""') == 2:
                in_docstring = False
            continue
        if in_docstring:
            continue
        if stripped and not stripped.startswith('#'):
            body_lines.append(line)
        if len(body_lines) >= 4:
            break
    body = "\n".join(body_lines) if body_lines else "(empty)"
    print(f"=== {src_name}: {getattr(fn, '__name__', type(fn).__name__)} ({len(lines)} lines) ===")
    print(body)
    print()
