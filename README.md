# Stress Level Detector

Machine learning classification pipeline analyzing psychological, physiological, environmental, and academic factors to predict human stress levels.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)
![Model](https://img.shields.io/badge/Model-SVC%20(91.5%25%20Acc)-brightgreen.svg)

---

## Overview

Human stress manifests through a complex interplay of psychological states, physical health indicators, social support networks, and environmental conditions. Stress Level Detector is an end-to-end machine learning system that evaluates 20 multi-domain indicators to classify an individual's stress state into three defined tiers:

- **Class 0: Low Stress**
- **Class 1: Moderate Stress**
- **Class 2: High Stress**

The system benchmarks multiple classification architectures (Support Vector Machines, Random Forests, Decision Trees, and K-Nearest Neighbors) against a 1,100-sample balanced dataset, deploying a standardized, scaled Support Vector Classifier (RBF kernel) as the primary prediction engine.

---

## Key Features

- **Multi-Domain Feature Set**: Incorporates 20 distinct factors across mental health, physiological symptoms, living conditions, and social pressures.
- **Balanced Multiclass Classification**: Evaluated on an evenly balanced distribution across all three stress categories.
- **Model Comparison Suite**: Comprehensive training pipeline benchmarking SVC, Random Forest, Decision Tree, and KNN.
- **CLI Assessment Suite**: Interactive prediction, CSV batch processing, model retraining, and pre-configured scenario evaluations.
- **Deterministic Inference Pipeline**: Fully serialized `StandardScaler` and model weights for reproducible predictions.

---

## Benchmark Results

Models were evaluated on a 30% holdout test partition (330 samples) with random state 42:

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Evaluation Summary |
|---|---|---|---|---|
| **Support Vector Classifier (RBF)** | **91.52%** | **0.92** | **0.92** | **Top performer; optimal boundary separation** |
| **Random Forest (100 trees)** | 89.70% | 0.90 | 0.90 | High precision on moderate and high stress |
| **Decision Tree Classifier** | 88.79% | 0.89 | 0.89 | Fast, interpretable tree baseline |
| **K-Nearest Neighbors (k=3)** | 88.79% | 0.89 | 0.89 | Reliable local neighborhood matching |

### Primary Model Classification Report (SVC)

```text
                    precision    recall  f1-score   support

    Class 0 (Low)        0.95      0.88      0.92       113
Class 1 (Moderate)       0.90      0.93      0.92       107
   Class 2 (High)        0.89      0.93      0.91       110

         accuracy                            0.92       330
        macro avg        0.92      0.92      0.92       330
     weighted avg        0.92      0.92      0.92       330
```

---

## Repository Structure

```text
stress-level-detector/
├── data/
│   └── StressLevelDataset.csv   # Complete 1,100-sample dataset
├── docs_archive/                # Preserved original reports and slides (untracked)
├── models/                      # Local model destination directory (untracked in git)
├── releases/                    # Local release packages for GitHub Releases (untracked in git)


├── notebooks/
│   └── StressLevels.ipynb       # Exploratory analysis and training notebook
├── src/
│   └── stress_level_detector/
│       ├── __init__.py          # Public package API
│       ├── cli.py               # Command-line interface
│       ├── config.py            # Feature definitions, mappings, and ranges
│       ├── predict.py           # Inference engine and input validation
│       └── trainer.py           # Model training and benchmark pipeline
├── tests/
│   ├── test_cli.py              # CLI integration tests
│   ├── test_predict.py          # Prediction and validation tests
│   └── test_trainer.py          # Training and benchmark tests
├── .gitignore                   # Standard repository ignores (includes docs_archive/)
├── .python-version              # Pinned Python version (3.13)
├── LICENSE                      # MIT License
├── pyproject.toml               # Package metadata and dependencies
└── uv.lock                      # Resolved dependency lockfile
```

---

## Installation

This project requires Python 3.10+ and uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync Dependencies

```bash
git clone <repository-url>
cd stress-level-detector
uv sync
```

---

## Usage

### 1. Run Predefined Validation Scenarios (Demo)

Evaluate three canonical lifestyle and stress profiles:

```bash
uv run stress-level-detector demo
```

Sample output:

```text
============================================================================
  Stress Level Detector - Validation Scenarios
============================================================================

Scenario: Profile A: Healthy Balanced Student
Predicted Level: Class 0 (LOW) | Confidence: 94.2%
Probabilities:   0 (Low): 94.2% | 1 (Moderate): 4.1% | 2 (High): 1.7%

Scenario: Profile B: Moderate Exam Pressure
Predicted Level: Class 1 (MODERATE) | Confidence: 88.6%
Probabilities:   0 (Low): 5.1% | 1 (Moderate): 88.6% | 2 (High): 6.3%

Scenario: Profile C: Severe High-Stress Overload
Predicted Level: Class 2 (HIGH) | Confidence: 96.1%
Probabilities:   0 (Low): 1.2% | 1 (Moderate): 2.7% | 2 (High): 96.1%
============================================================================
```

### 2. Retrain and Benchmark Models

Retrain all candidate algorithms and output performance comparisons:

```bash
uv run stress-level-detector train
```

### 3. Batch Scoring (CSV)

Run inference on an unlabelled CSV dataset:

```bash
uv run stress-level-detector batch --input data/StressLevelDataset.csv --output scored_results.csv
```

---

## Pre-Trained Model Releases

To keep the repository lightweight and follow ML best practices, binary model weights (`.joblib`) are distributed via **GitHub Releases** rather than tracked in the git tree.

### Downloading & Unpacking Model Artifacts

Model artifacts are packaged in `releases/v1.0.0/stress_detector_models_v1.0.0.tar.gz`. When running predictions, `StressPredictor` automatically extracts the archive into `models/` if available.

To unpack manually:

```bash
tar -xzvf releases/v1.0.0/stress_detector_models_v1.0.0.tar.gz -C models/
```

To verify archive checksum:

```bash
cd releases/v1.0.0 && sha256sum -c SHA256SUMS
```

### Publishing to GitHub Releases

To attach the model archive to a GitHub Release:

```bash
gh release create v1.0.0 releases/v1.0.0/stress_detector_models_v1.0.0.tar.gz \
    --title "v1.0.0 - Production Release" \
    --notes "Pre-trained Random Forest classifier and StandardScaler for lifestyle stress detection."
```

---

## Running Tests


Run the test suite and code linters:

```bash
uv run pytest
uv run ruff check .
```

---

## Feature Schema

The dataset evaluates 20 input attributes grouped across four domains:

| Category | Feature Name | Domain Range | Description |
|---|---|---|---|
| **Psychological** | `anxiety_level` | 0 – 21 | Self-reported anxiety index |
| | `self_esteem` | 0 – 30 | Self-worth and self-esteem score |
| | `mental_health_history` | 0 or 1 | Prior clinical mental health history |
| | `depression` | 0 – 27 | Depression symptom score |
| **Physiological** | `headache` | 0 – 5 | Frequency/severity of headaches |
| | `blood_pressure` | 1 – 3 | Blood pressure tier (1: Normal, 2: Elevated, 3: High) |
| | `sleep_quality` | 0 – 5 | Self-assessed sleep rating |
| | `breathing_problem` | 0 – 5 | Frequency of respiratory difficulty |
| **Environmental** | `noise_level` | 0 – 5 | Surrounding auditory noise level |
| | `living_conditions` | 0 – 5 | Quality of residential conditions |
| | `safety` | 0 – 5 | Perceived personal safety |
| | `basic_needs` | 0 – 5 | Access to essential daily necessities |
| **Academic / Social** | `academic_performance` | 0 – 5 | Current academic performance level |
| | `study_load` | 0 – 5 | Volume and intensity of study workload |
| | `teacher_student_relationship` | 0 – 5 | Quality of faculty mentorship and rapport |
| | `future_career_concerns` | 0 – 5 | Worry regarding future career and employment |
| | `social_support` | 0 – 5 | Strength of personal peer/family support |
| | `peer_pressure` | 0 – 5 | Level of peer pressure experienced |
| | `extracurricular_activities` | 0 – 5 | Engagement in outside activities |
| | `bullying` | 0 – 5 | Frequency of bullying experiences |

---

## Limitations

- **Advisory Scope**: This model is designed for pattern recognition and exploratory wellness monitoring. It does not provide medical or clinical psychiatric diagnoses.
- **Self-Reported Nature**: Survey responses are subjective and subject to self-reporting variability.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Ahmad Serieh**  
Email: [ahmadserieh.official@gmail.com](mailto:ahmadserieh.official@gmail.com)
