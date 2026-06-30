# Model Evaluation Results

## Experiment Details

| Item | Value |
|---|---|
| Dataset | Wisconsin Diagnostic Breast Cancer |
| Total observations | 569 |
| Input features | 30 numeric measurements |
| Target | Benign (0) or Malignant (1) |
| Training observations | 455 |
| Testing observations | 114 |
| Data split | Stratified 80% training / 20% testing |
| Random state | 42 |
| Primary selection metric | ROC-AUC |

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9649 | 0.9750 | 0.9286 | 0.9512 | **0.9970** |
| Logistic Regression | **0.9737** | **0.9756** | **0.9524** | **0.9639** | 0.9960 |
| Decision Tree | 0.9035 | 0.9429 | 0.7857 | 0.8571 | 0.9471 |

## Best Model

Random Forest was saved as the final model because it produced the highest
ROC-AUC: **0.9970**. ROC-AUC measures the model's ability to rank malignant cases
above benign cases across all possible probability thresholds.

Logistic Regression performed slightly better at the default 0.5 threshold,
with **97.37% accuracy**, **95.24% recall**, and an **F1-score of 0.9639**. This
shows that the definition of “best” depends on the evaluation objective.

## Result Files

The `results` directory contains:

- `model_metrics.csv` — exact metric values
- `roc_curves.png` — ROC comparison for all models
- three confusion-matrix images
- `feature_importance.png` and `feature_importance.csv`
- `best_model.joblib` — serialized Random Forest pipeline
- `run_metadata.json` — dataset and experiment configuration
- `sample_input.csv` and `predictions.csv` — verified inference example

