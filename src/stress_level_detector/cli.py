"""CLI command-line interface for Stress Level Detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stress_level_detector.config import DEFAULT_DATA_PATH, FEATURE_NAMES
from stress_level_detector.predict import StressPredictor, StressRecord
from stress_level_detector.trainer import train_and_benchmark


def cmd_train(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    print(f"Loading dataset from: {data_path}")
    trained_models, benchmarks, best_name = train_and_benchmark(
        data_path=data_path, save_artifacts=True
    )

    print("\n" + "=" * 70)
    print(f"{'Classifier Model':<30} | {'Accuracy':>10} | {'Macro F1':>10} | {'Weighted F1':>10}")
    print("-" * 70)
    for name, bench in benchmarks.items():
        mark = " *" if name == best_name else "  "
        print(
            f"{name + mark:<30} | {bench.accuracy:10.2%} | {bench.macro_f1:10.2%} | "
            f"{bench.weighted_f1:10.2%}"
        )
    print("=" * 70)
    print(f"(* Top performing model: {best_name} saved to models/)\n")
    print(benchmarks[best_name].report)
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    predictor = StressPredictor()

    scenarios = [
        (
            "Profile A: Healthy Balanced Student",
            StressRecord(
                anxiety_level=3.0,
                self_esteem=28.0,
                mental_health_history=0,
                depression=2.0,
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
            ),
        ),
        (
            "Profile B: Moderate Exam Pressure",
            StressRecord(
                anxiety_level=12.0,
                self_esteem=16.0,
                mental_health_history=0,
                depression=10.0,
                headache=2.0,
                blood_pressure=2.0,
                sleep_quality=2.0,
                breathing_problem=1.0,
                noise_level=3.0,
                living_conditions=3.0,
                safety=3.0,
                basic_needs=3.0,
                academic_performance=3.0,
                study_load=4.0,
                teacher_student_relationship=3.0,
                future_career_concerns=3.0,
                social_support=2.0,
                peer_pressure=3.0,
                extracurricular_activities=3.0,
                bullying=1.0,
            ),
        ),
        (
            "Profile C: Severe High-Stress Overload",
            StressRecord(
                anxiety_level=19.0,
                self_esteem=5.0,
                mental_health_history=1,
                depression=22.0,
                headache=5.0,
                blood_pressure=3.0,
                sleep_quality=0.0,
                breathing_problem=4.0,
                noise_level=5.0,
                living_conditions=1.0,
                safety=1.0,
                basic_needs=1.0,
                academic_performance=1.0,
                study_load=5.0,
                teacher_student_relationship=0.0,
                future_career_concerns=5.0,
                social_support=0.0,
                peer_pressure=5.0,
                extracurricular_activities=5.0,
                bullying=5.0,
            ),
        ),
    ]

    print("\n" + "=" * 76)
    print("  Stress Level Detector - Validation Scenarios")
    print("=" * 76)

    for name, record in scenarios:
        res = predictor.predict(record)
        print(f"\nScenario: {name}")
        status = f"Class {res.class_id} ({res.label.upper()}) | Confidence: {res.confidence:.1%}"
        print(f"Predicted Level: {status}")
        probs_str = " | ".join(f"{k}: {v:.1%}" for k, v in res.probabilities.items())
        print(f"Probabilities:   {probs_str}")

    print("\n" + "=" * 76 + "\n")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} not found.")
        return 1

    predictor = StressPredictor()
    df = pd.read_csv(input_path)
    output_df = predictor.predict_batch(df)

    out_name = f"{input_path.stem}_scored.csv"
    out_path = Path(args.output) if args.output else input_path.with_name(out_name)
    output_df.to_csv(out_path, index=False)
    print(f"Scored {len(output_df)} records. Output written to: {out_path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = StressPredictor()
    rec_dict = {feat: getattr(args, feat) for feat in FEATURE_NAMES}
    res = predictor.predict(rec_dict)

    print("\n" + "=" * 50)
    print(" Stress Level Detector - Assessment Result")
    print("=" * 50)
    print(f" Predicted Stress Level:  Class {res.class_id} ({res.label})")
    print(f" Confidence:              {res.confidence:.2%}")
    print("-" * 50)
    print(" Probability Distribution:")
    for label, prob in res.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"   {label:<18} {prob:6.2%} | {bar}")
    print("=" * 50 + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stress-level-detector",
        description="Stress Level Detector: Multi-factor stress classification engine.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Run validation scenario demo")
    p_demo.set_defaults(func=cmd_demo)

    # Train
    p_train = subparsers.add_parser("train", help="Train and benchmark candidate models")
    p_train.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Dataset CSV")
    p_train.set_defaults(func=cmd_train)

    # Batch
    p_batch = subparsers.add_parser("batch", help="Batch predict CSV file")
    p_batch.add_argument("--input", "-i", type=str, required=True, help="Input CSV")
    p_batch.add_argument("--output", "-o", type=str, help="Output CSV")
    p_batch.set_defaults(func=cmd_batch)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict stress level from individual features")
    for feat in FEATURE_NAMES:
        p_pred.add_argument(f"--{feat.replace('_', '-')}", dest=feat, type=float, required=True)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    if not args.command:
        return cmd_demo(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
