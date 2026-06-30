"""Make predictions with the saved best model from a numeric CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict outcomes for new rows.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "results" / "best_model.joblib")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input_csv)
    expected = bundle["feature_names"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    probabilities = bundle["model"].predict_proba(frame[expected])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    mapping = bundle["metadata"]["class_mapping"]
    result = frame.copy()
    result["predicted_class"] = [mapping[str(value)] for value in predictions]
    result["positive_class_probability"] = probabilities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
