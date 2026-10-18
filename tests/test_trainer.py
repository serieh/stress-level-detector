"""Unit tests for training and benchmarking pipeline."""

from stress_level_detector.trainer import train_and_benchmark


def test_train_and_benchmark_models():
    trained_models, benchmarks, best_name = train_and_benchmark(save_artifacts=False)

    expected_models = [
        "Support Vector Classifier",
        "Random Forest",
        "Decision Tree",
        "K-Nearest Neighbors (k=3)",
    ]
    for model_name in expected_models:
        assert model_name in benchmarks
        bench = benchmarks[model_name]
        assert bench.accuracy >= 0.85
        assert bench.macro_f1 >= 0.85

    assert best_name in expected_models
