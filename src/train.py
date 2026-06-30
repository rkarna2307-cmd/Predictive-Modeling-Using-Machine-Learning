"""Train and evaluate supervised classification models.

The default case study uses scikit-learn's breast cancer dataset. A numeric,
binary-classification CSV can be supplied with --data-path and --target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare supervised-learning classifiers on binary outcomes."
    )
    parser.add_argument("--data-path", type=Path, help="Optional input CSV file.")
    parser.add_argument("--target", help="Target column when --data-path is used.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for models, metrics, and plots.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(
    data_path: Path | None, target_column: str | None
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if data_path is None:
        dataset = load_breast_cancer(as_frame=True)
        X = dataset.data.copy()
        # Make the clinically important malignant outcome the positive class.
        y = (dataset.target == 0).astype(int).rename("malignant")
        metadata = {
            "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
            "target": "malignant",
            "class_mapping": {"0": "benign", "1": "malignant"},
            "source": "scikit-learn built-in dataset",
        }
        return X, y, metadata

    if not target_column:
        raise ValueError("--target is required when --data-path is supplied.")
    frame = pd.read_csv(data_path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    if frame[target_column].isna().any():
        raise ValueError("The target column contains missing values.")

    X = frame.drop(columns=[target_column])
    non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError("All predictor columns must be numeric: " + ", ".join(non_numeric))
    if X.empty:
        raise ValueError("The dataset must contain at least one predictor column.")

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(frame[target_column])
    if len(encoder.classes_) != 2:
        raise ValueError("This workflow requires exactly two target classes.")
    y = pd.Series(encoded, index=frame.index, name=target_column)
    class_mapping = {str(i): str(label) for i, label in enumerate(encoder.classes_)}
    metadata = {
        "dataset": data_path.name,
        "target": target_column,
        "class_mapping": class_mapping,
        "source": str(data_path.resolve()),
    }
    return X, y, metadata


def build_models(feature_names: list[str], random_state: int) -> dict[str, Pipeline]:
    scaled_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), feature_names)],
        verbose_feature_names_out=False,
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), feature_names)],
        verbose_feature_names_out=False,
    )
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("model", LogisticRegression(
                max_iter=2_000, solver="liblinear", random_state=random_state
            )),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=3, random_state=random_state
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }


def evaluate_models(
    models: dict[str, Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series, output_dir: Path,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, Pipeline] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })
        curves[name] = roc_curve(y_test, probabilities)[:2]
        fitted[name] = model

        matrix = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set(title=f"{name} - Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        fig.tight_layout()
        filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values(
        ["roc_auc", "f1_score", "accuracy"], ascending=False
    ).reset_index(drop=True)
    return metrics, fitted, curves


def save_roc_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    auc_lookup = metrics.set_index("model")["roc_auc"].to_dict()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (false_positive_rate, true_positive_rate) in curves.items():
        ax.plot(false_positive_rate, true_positive_rate,
                label=f"{name} (AUC = {auc_lookup[name]:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random chance")
    ax.set(title="Receiver Operating Characteristic Curves",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def save_feature_importance(
    best_model: Pipeline, best_name: str, feature_names: list[str], output_dir: Path
) -> bool:
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return False
    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#2878B5")
    ax.set(title=f"Top Feature Importance - {best_name}", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
    return True


def write_findings(
    metrics: pd.DataFrame, metadata: dict[str, Any], train_size: int,
    test_size: int, output_dir: Path, has_importance: bool,
) -> None:
    best = metrics.iloc[0]
    accuracy_leader = metrics.loc[metrics["accuracy"].idxmax()]
    f1_leader = metrics.loc[metrics["f1_score"].idxmax()]
    table = metrics.to_markdown(index=False, floatfmt=".4f")
    class_mapping = metadata["class_mapping"]
    findings = f"""# Results and Findings

## Experiment Summary

- **Dataset:** {metadata['dataset']}
- **Prediction target:** {metadata['target']}
- **Class mapping:** 0 = {class_mapping['0']}, 1 = {class_mapping['1']}
- **Training samples:** {train_size}
- **Testing samples:** {test_size}
- **Split:** stratified {100 * (1 - metadata['test_size']):.0f}/{100 * metadata['test_size']:.0f} train/test holdout with random state {metadata['random_state']}

## Model Performance

{table}

## Main Finding

**{best['model']}** ranked first using ROC-AUC as the primary selection metric,
with ROC-AUC **{best['roc_auc']:.4f}**, accuracy **{best['accuracy']:.4f}**, and
F1-score **{best['f1_score']:.4f}** on the unseen test set.

The result is metric-dependent: **{accuracy_leader['model']}** achieved the
highest accuracy (**{accuracy_leader['accuracy']:.4f}**), while
**{f1_leader['model']}** achieved the highest F1-score
(**{f1_leader['f1_score']:.4f}**). This is why all metrics are retained rather
than labeling one model universally superior.

ROC-AUC was chosen as the primary metric because it measures how well a model
ranks the two classes across all classification thresholds. Accuracy, precision,
recall, and F1-score are reported so that the result is not judged by one number.

## Interpretation

- Logistic Regression provides a strong, interpretable linear baseline.
- The Decision Tree captures nonlinear rules but can overfit small datasets.
- The Random Forest reduces single-tree variance by averaging many trees.
- The confusion matrices show the exact false-positive and false-negative counts.
- The ROC chart compares discrimination over all probability thresholds.
{('- `feature_importance.png` shows which measurements most influenced the selected model.' if has_importance else '')}

## Limitations

- This is one stratified holdout split; repeated cross-validation would give a
  more stable estimate for a formal study.
- The dataset is relatively small and comes from a specific diagnostic setting.
- Predictive importance is association, not proof that a feature causes an outcome.
- A medical model would require external validation, calibration analysis, bias
  assessment, and clinical review before any real-world use.

## Reproduce the Result

From the project root, install the dependencies and run:

```powershell
python src/train.py
```

The script regenerates every file in `artifacts/`, including this report.
"""
    (output_dir / "FINDINGS.md").write_text(findings, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.50.")

    X, y, metadata = load_dataset(args.data_path, args.target)
    if y.value_counts().min() < 2:
        raise ValueError("Each class must contain at least two samples.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    models = build_models(X.columns.tolist(), args.random_state)
    class_names = [metadata["class_mapping"]["0"], metadata["class_mapping"]["1"]]
    metrics, fitted, curves = evaluate_models(
        models, X_train, X_test, y_train, y_test, args.output_dir, class_names
    )
    save_roc_plot(curves, metrics, args.output_dir)

    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    has_importance = save_feature_importance(
        best_model, best_name, X.columns.tolist(), args.output_dir
    )
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    metadata.update({
        "rows": len(X),
        "features": len(X.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "best_model": best_name,
        "selection_metric": "roc_auc",
    })
    # Target-free rows provide a ready-to-use example for the prediction script.
    X_test.head(5).to_csv(args.output_dir / "sample_input.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"model": best_model, "feature_names": X.columns.tolist(), "metadata": metadata},
        args.output_dir / "best_model.joblib",
    )
    write_findings(metrics, metadata, len(X_train), len(X_test),
                   args.output_dir, has_importance)

    print("\nModel comparison (test set):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model: {best_name}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
