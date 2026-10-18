"""Inference engine for stress level prediction from lifestyle and physiological factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stress_level_detector.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_NAMES,
    FEATURE_RANGES,
    STRESS_LABELS,
)


@dataclass
class StressRecord:
    """Individual assessment metrics spanning psychological, physical, and environmental factors."""

    anxiety_level: float
    self_esteem: float
    mental_health_history: int
    depression: float
    headache: float
    blood_pressure: float
    sleep_quality: float
    breathing_problem: float
    noise_level: float
    living_conditions: float
    safety: float
    basic_needs: float
    academic_performance: float
    study_load: float
    teacher_student_relationship: float
    future_career_concerns: float
    social_support: float
    peer_pressure: float
    extracurricular_activities: float
    bullying: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Validate feature values within plausible clinical and survey domain ranges."""
        for name, val in self.to_dict().items():
            if name in FEATURE_RANGES:
                min_v, max_v = FEATURE_RANGES[name]
                if not (min_v <= val <= max_v):
                    raise ValueError(
                        f"Feature '{name}' out of bounds: expected [{min_v}, {max_v}], got {val}"
                    )


@dataclass
class PredictionResult:
    """Output prediction containing numeric class, semantic label, and confidence."""

    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StressPredictor:
    """Loads trained classification model and scaler for inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.scaler_path.exists():
            bundle_name = "stress_detector_models_v1.0.0.tar.gz"
            release_bundle = self.model_path.parent.parent / "releases" / "v1.0.0" / bundle_name
            if release_bundle.exists():
                import tarfile
                with tarfile.open(release_bundle, "r:gz") as tar:
                    tar.extractall(path=self.model_path.parent)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Download the release bundle from GitHub Releases, extract the archive "
                "into models/, or run 'stress-cli train' to train locally."
            )
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at {self.scaler_path}. "
                "Download the release bundle from GitHub Releases, extract the archive "
                "into models/, or run 'stress-cli train' to train locally."
            )

        self.model = joblib.load(self.model_path)
        self.scaler: StandardScaler = joblib.load(self.scaler_path)


    def predict(self, record: StressRecord | dict[str, Any]) -> PredictionResult:
        """Predict stress level for an individual record."""
        if isinstance(record, StressRecord):
            record.validate()
            rec_dict = record.to_dict()
        else:
            rec_dict = record

        row = [float(rec_dict[col]) for col in FEATURE_NAMES]
        x_raw = pd.DataFrame([row], columns=FEATURE_NAMES)
        x_scaled = self.scaler.transform(x_raw)

        pred_class = int(self.model.predict(x_scaled)[0])
        label = STRESS_LABELS.get(pred_class, f"Class {pred_class}")

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_scaled)[0]
            confidence = float(np.max(probs))
            prob_dict = {
                f"{k} ({STRESS_LABELS.get(k, 'Unknown')})": float(probs[idx])
                for idx, k in enumerate(sorted(STRESS_LABELS.keys()))
                if idx < len(probs)
            }
        else:
            confidence = 1.0
            prob_dict = {label: 1.0}

        return PredictionResult(
            class_id=pred_class,
            label=label,
            confidence=confidence,
            probabilities=prob_dict,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict stress levels across a batch DataFrame."""
        x_raw = df[FEATURE_NAMES]
        x_scaled = self.scaler.transform(x_raw)
        preds = self.model.predict(x_scaled)

        out_df = df.copy()
        out_df["predicted_stress_level"] = preds
        out_df["stress_label"] = [STRESS_LABELS.get(int(p), f"Class {p}") for p in preds]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_scaled)
            out_df["confidence"] = np.max(probs, axis=1)

        return out_df
