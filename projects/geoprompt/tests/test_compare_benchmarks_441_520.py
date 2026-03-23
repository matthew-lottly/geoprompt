import geoprompt.compare as compare


def test_comparison_report_registers_441_520_benchmarks() -> None:
    original_benchmark = compare._benchmark
    compare._benchmark = lambda operation, func, repeats=20: ({"operation": operation, "repeats": 1}, None)
    try:
        report = compare.build_comparison_report()
    finally:
        compare._benchmark = original_benchmark

    benchmark_ops = {
        str(benchmark["operation"])
        for dataset in report["datasets"]
        for benchmark in dataset["benchmarks"]
    }

    assert "sample.geoprompt.spatial_elastic_net" in benchmark_ops
    assert "sample.geoprompt.spatial_dbscan_clustering" in benchmark_ops
    assert "sample.geoprompt.spatial_hdbscan" in benchmark_ops
    assert "sample.geoprompt.spatial_optimal_transport" in benchmark_ops
    assert "sample.geoprompt.spatial_conformal_predictor" in benchmark_ops