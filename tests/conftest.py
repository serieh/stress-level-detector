"""Pytest session fixtures for stress-level-detector."""

import pytest

from stress_level_detector.config import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH
from stress_level_detector.trainer import train_and_benchmark


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts_exist():
    """Ensure trained model and scaler artifacts exist before any tests execute."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_SCALER_PATH.exists():
        train_and_benchmark(save_artifacts=True)
