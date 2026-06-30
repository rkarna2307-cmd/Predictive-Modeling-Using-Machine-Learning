# Predictive Modeling Using Machine Learning

An end-to-end supervised-learning internship project that predicts a binary
outcome, compares three classification algorithms, and evaluates their behavior
with numerical metrics, confusion matrices, ROC curves, and feature importance.

> A ready-to-submit copy is available in [`submission/`](submission/).

## Project Objective

Build and compare machine-learning models that can predict whether a breast
tumor is malignant or benign from diagnostic measurements. The workflow is also
reusable with another numeric binary-classification CSV.

## Key Features

- Logistic Regression, Decision Tree, and Random Forest models
- reproducible stratified train/test split
- median imputation and feature scaling where appropriate
- accuracy, precision, recall, F1-score, and ROC-AUC evaluation
- confusion matrix for every model and a combined ROC curve
- automatic best-model selection and model serialization
- feature-importance chart and CSV
- automatically generated results and findings report
- optional custom CSV training and batch prediction

## Project Structure

```text
.
|-- artifacts/                 # Generated results, plots, and trained model
|-- data/
|   `-- README.md              # Custom-data instructions
|-- docs/
|   `-- METHODOLOGY.md         # Experiment design and metric definitions
|-- src/
|   |-- predict.py             # Batch predictions with the saved model
|   `-- train.py               # Training and evaluation pipeline
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Quick Start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
```

After the run, open [`artifacts/FINDINGS.md`](artifacts/FINDINGS.md) for the
measured results and interpretation.

## Generated Outputs

| File | Purpose |
|---|---|
| `FINDINGS.md` | Reproducible result report and limitations |
| `model_metrics.csv` | Side-by-side test metrics |
| `roc_curves.png` | ROC curve comparison with AUC values |
| `*_confusion_matrix.png` | Actual versus predicted outcomes |
| `feature_importance.csv/.png` | Ranked influence of input measurements |
| `best_model.joblib` | Best fitted pipeline for later predictions |
| `run_metadata.json` | Dataset, split, label, and selection details |
| `sample_input.csv` | Five target-free rows for testing inference |

## Train on Your Own CSV

The file must have numeric feature columns and a target with exactly two classes.

```powershell
python src/train.py --data-path data/your_data.csv --target outcome
```

See [`data/README.md`](data/README.md) for assumptions and label behavior.

## Make Predictions

Prepare a CSV containing the same feature columns used during training, then run:

```powershell
python src/predict.py data/new_samples.csv
```

For a quick test after training, use the generated sample:

```powershell
python src/predict.py artifacts/sample_input.csv
```

Predictions are written to `artifacts/predictions.csv`. The probability column
is the probability of class `1`, whose meaning is recorded in
`artifacts/run_metadata.json`.

## Reproducibility

The split and all stochastic algorithms use random state `42`. Dependencies are
pinned in `requirements.txt`. Re-running `python src/train.py` recreates the
evaluation artifacts.

## Responsible Use

This project is educational. It is not a medical device and must not be used for
diagnosis. Real deployment would require external validation, probability
calibration, subgroup fairness analysis, privacy controls, monitoring, and
domain-expert approval.
