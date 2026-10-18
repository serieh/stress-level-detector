"""Unit tests for prediction engine and validation."""

import pandas as pd
import pytest

from stress_level_detector.config import FEATURE_NAMES
from stress_level_detector.predict import StressPredictor, StressRecord


@pytest.fixture
def predictor() -> StressPredictor:
    return StressPredictor()


def test_stress_record_validation():
    # Valid record
    rec = StressRecord(
        anxiety_level=10.0,
        self_esteem=15.0,
        mental_health_history=1,
        depression=10.0,
        headache=2.0,
        blood_pressure=2.0,
        sleep_quality=3.0,
        breathing_problem=1.0,
        noise_level=2.0,
        living_conditions=3.0,
        safety=3.0,
        basic_needs=3.0,
        academic_performance=3.0,
        study_load=3.0,
        teacher_student_relationship=3.0,
        future_career_concerns=3.0,
        social_support=3.0,
        peer_pressure=2.0,
        extracurricular_activities=2.0,
        bullying=1.0,
    )
    rec.validate()

    # Out of range anxiety_level (max is 21)
    with pytest.raises(ValueError, match="out of bounds"):
        invalid_rec = StressRecord(**{**rec.to_dict(), "anxiety_level": 50.0})
        invalid_rec.validate()


def test_predict_single_record(predictor: StressPredictor):
    rec = StressRecord(
        anxiety_level=2.0,
        self_esteem=28.0,
        mental_health_history=0,
        depression=1.0,
        headache=0.0,
        blood_pressure=1.0,
        sleep_quality=5.0,
        breathing_problem=0.0,
        noise_level=1.0,
        living_conditions=5.0,
        safety=5.0,
        basic_needs=5.0,
        academic_performance=5.0,
        study_load=2.0,
        teacher_student_relationship=5.0,
        future_career_concerns=1.0,
        social_support=5.0,
        peer_pressure=1.0,
        extracurricular_activities=2.0,
        bullying=0.0,
    )
    res = predictor.predict(rec)
    assert res.class_id in [0, 1, 2]
    assert res.label in ["Low", "Moderate", "High"]
    assert 0.0 <= res.confidence <= 1.0
    assert sum(res.probabilities.values()) == pytest.approx(1.0, abs=1e-2)


def test_predict_batch(predictor: StressPredictor):
    sample_rows = [
        {feat: 2.0 for feat in FEATURE_NAMES},
        {feat: 4.0 for feat in FEATURE_NAMES},
    ]
    df = pd.DataFrame(sample_rows)
    out = predictor.predict_batch(df)
    assert "predicted_stress_level" in out.columns
    assert "stress_label" in out.columns
    assert len(out) == 2
