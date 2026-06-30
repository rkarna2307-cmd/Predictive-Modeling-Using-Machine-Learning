# Results and Findings

## Experiment Summary

- **Dataset:** Wisconsin Diagnostic Breast Cancer (scikit-learn)
- **Prediction target:** malignant
- **Class mapping:** 0 = benign, 1 = malignant
- **Training samples:** 455
- **Testing samples:** 114
- **Split:** stratified 80/20 train/test holdout with random state 42

## Model Performance

| model               |   accuracy |   precision |   recall |   f1_score |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Random Forest       |     0.9649 |      0.9750 |   0.9286 |     0.9512 |    0.9970 |
| Logistic Regression |     0.9737 |      0.9756 |   0.9524 |     0.9639 |    0.9960 |
| Decision Tree       |     0.9035 |      0.9429 |   0.7857 |     0.8571 |    0.9471 |

## Main Finding

**Random Forest** ranked first using ROC-AUC as the primary selection metric,
with ROC-AUC **0.9970**, accuracy **0.9649**, and
F1-score **0.9512** on the unseen test set.

The result is metric-dependent: **Logistic Regression** achieved the
highest accuracy (**0.9737**), while
**Logistic Regression** achieved the highest F1-score
(**0.9639**). This is why all metrics are retained rather
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
- `feature_importance.png` shows which measurements most influenced the selected model.

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
