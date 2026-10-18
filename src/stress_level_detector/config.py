"""Configuration constants, feature definitions, and default paths."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_DATA_PATH = DATA_DIR / "StressLevelDataset.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "stress_classifier.joblib"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler.joblib"

TARGET_COLUMN = "stress_level"

FEATURE_NAMES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]

STRESS_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

# Domain ranges for validation
FEATURE_RANGES = {
    "anxiety_level": (0, 21),
    "self_esteem": (0, 30),
    "mental_health_history": (0, 1),
    "depression": (0, 27),
    "headache": (0, 5),
    "blood_pressure": (1, 3),
    "sleep_quality": (0, 5),
    "breathing_problem": (0, 5),
    "noise_level": (0, 5),
    "living_conditions": (0, 5),
    "safety": (0, 5),
    "basic_needs": (0, 5),
    "academic_performance": (0, 5),
    "study_load": (0, 5),
    "teacher_student_relationship": (0, 5),
    "future_career_concerns": (0, 5),
    "social_support": (0, 5),
    "peer_pressure": (0, 5),
    "extracurricular_activities": (0, 5),
    "bullying": (0, 5),
}
