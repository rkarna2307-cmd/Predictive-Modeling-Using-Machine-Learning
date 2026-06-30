# Internship Assignment Submission

## Predictive Modeling Using Machine Learning

This submission contains a complete supervised-learning classification project.
It compares Logistic Regression, Decision Tree, and Random Forest models for
predicting whether a breast tumor is malignant or benign.

## Files to Review

- [`RESULTS.md`](RESULTS.md) — experiment setup and measured model performance
- [`FINDINGS.md`](FINDINGS.md) — interpretation, conclusions, and limitations
- [`METHODOLOGY.md`](METHODOLOGY.md) — modeling and evaluation methodology
- `source_code/train.py` — training and evaluation pipeline
- `source_code/predict.py` — batch inference using the saved model
- `results/` — charts, metrics, metadata, predictions, and trained model

## Run the Project

From inside this `submission` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python source_code/train.py
```

To test the saved model:

```powershell
python source_code/predict.py results/sample_input.csv
```

## Final Outcome

Random Forest achieved the highest ROC-AUC of **0.9970**. Logistic Regression
achieved the highest test accuracy of **97.37%** and F1-score of **0.9639**.
These results demonstrate successful application and evaluation of supervised
machine-learning models.
