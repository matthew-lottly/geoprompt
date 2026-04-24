# Simulation and Deprecation Labels

Auto-generated symbol labels for simulation-only API surfaces.

- Symbols labeled: 19

| Module | Symbol | Label | Deprecation | Guidance |
| --- | --- | --- | --- | --- |
| ml | gradient_boosted_spatial_prediction | simulation-only | active | Use sklearn.ensemble.GradientBoostingClassifier or xgboost for real gradient boosting. |
| ml | svm_spatial_classification | simulation-only | active | Use sklearn.svm.SVC for a real support vector machine classifier. |
| ml | neural_network_integration | simulation-only | active | Use PyTorch (torch.nn) or TensorFlow for real neural network integration. |
| ml | graph_neural_network_prediction | simulation-only | active | Use PyG (torch_geometric) or DGL for a real Graph Neural Network. |
| ml | convolutional_neural_network_on_rasters | simulation-only | active | Use PyTorch torchvision or TensorFlow Keras for a real CNN on rasters. |
| ml | recurrent_neural_network_spatial_time_series | simulation-only | active | Use PyTorch LSTM/GRU or TensorFlow Keras for a real RNN time-series model. |
| ml | transformer_model_spatial_sequences | simulation-only | active | Use PyTorch transformers or Hugging Face for a real attention/transformer model. |
| performance | gpu_accelerated_distance_matrix | simulation-only | active | Use CuPy + cuSpatial or RAPIDS for real GPU-accelerated spatial operations. |
| performance | gpu_accelerated_raster_algebra | simulation-only | active | Use CuPy for real GPU raster algebra. |
| performance | scale_analysis | simulation-only | active | Replace with a workload-specific scaling profiler and benchmark pipeline. |
| performance | distributed_spatial_join | simulation-only | active | Use Dask-GeoDataFrame for real distributed spatial joins. |
| standards | ogc_api_features_implementation | simulation-only | active | Implement a real FastAPI OGC API - Features endpoint using the pygeoapi or OWSLib library. |
| standards | ogc_api_processes_implementation | simulation-only | active | Implement a real FastAPI OGC API - Processes endpoint. |
| standards | ogc_api_records_implementation | simulation-only | active | Implement a real OGC API - Records endpoint backed by a spatial catalogue. |
| standards | ogc_api_tiles_implementation | simulation-only | active | Implement real tile generation using mapbox/tippecanoe or Rio-COG. |
| standards | ogc_api_maps_implementation | simulation-only | active | Implement a real WMS map rendering endpoint via GDAL or MapServer. |
| standards | ogc_wfs_client | simulation-only | active | Use OWSLib or httpx to implement a real WFS GetCapabilities/GetFeature client. |
| standards | ogc_wms_client | simulation-only | active | Use OWSLib or httpx to implement a real WMS GetMap client. |
| standards | wms_capabilities_document | simulation-only | active | Use OWSLib or httpx to implement a real WMS capabilities document parser. |
