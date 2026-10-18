"""Model training, evaluation comparison, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from stress_level_detector.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_NAMES,
    MODELS_DIR,
    STRESS_LABELS,
    TARGET_COLUMN,
)


@dataclass
class ModelBenchmark:
    """Benchmark results for a candidate classifier."""

    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: str


def train_and_benchmark(
    data_path: str | Path = DEFAULT_DATA_PATH,
    test_size: float = 0.3,
    random_state: int = 42,
    save_artifacts: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, ModelBenchmark], str]:
    """Train multiple classifiers on the stress levels dataset and compare performance."""
    df = pd.read_csv(data_path)

    x = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    candidates: dict[str, Any] = {
        "Support Vector Classifier": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3),
    }

    trained_models: dict[str, Any] = {}
    benchmarks: dict[str, ModelBenchmark] = {}
    best_name = ""
    best_acc = -1.0

    target_names = [f"Class {k} ({STRESS_LABELS[k]})" for k in sorted(STRESS_LABELS.keys())]

    for name, model in candidates.items():
        # Scale features for distance-sensitive models (SVC, KNN); trees also work on scaled
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )

        trained_models[name] = model
        benchmarks[name] = ModelBenchmark(
            name=name,
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            report=report,
        )

        if acc > best_acc:
            best_acc = acc
            best_name = name

    if save_artifacts:
        dest_dir = Path(output_dir) if output_dir else MODELS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_models[best_name], dest_dir / DEFAULT_MODEL_PATH.name)
        joblib.dump(scaler, dest_dir / DEFAULT_SCALER_PATH.name)

    return trained_models, benchmarks, best_name
