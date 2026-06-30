# Methodology

## Problem Definition

This is a supervised binary-classification problem. Each row contains diagnostic
measurements, and the target says whether the observed tumor is malignant (1) or
benign (0). Malignant is intentionally the positive class so recall directly
measures how many malignant cases the model detects.

## Dataset

The default Wisconsin Diagnostic Breast Cancer dataset contains 569 observations
and 30 numeric predictors computed from digitized images of fine-needle aspirate
samples. It ships with scikit-learn, which makes the experiment offline and
reproducible.

## Experimental Design

1. Split the data into 80% training and 20% testing partitions.
2. Stratify the split to preserve the class ratio.
3. Fit preprocessing only on training data inside each model pipeline. This
   prevents information leakage from the test set.
4. Replace missing predictor values with training-column medians.
5. Standardize features for Logistic Regression. Tree models do not require
   scaling.
6. Train Logistic Regression, a depth-limited Decision Tree, and a 300-tree
   Random Forest.
7. Evaluate once on the untouched test partition.
8. Rank models by ROC-AUC, with F1-score and accuracy as tie-breakers.

## Why These Models?

- **Logistic Regression** is a transparent linear classification baseline. Plain
  Linear Regression is not used because its unbounded predictions are unsuitable
  as binary-class probabilities.
- **Decision Tree** represents nonlinear decision rules and interactions.
- **Random Forest** combines many randomized trees to improve stability and
  generalization.

## Evaluation Metrics

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** fraction of predicted malignant cases that are malignant.
- **Recall:** fraction of actual malignant cases detected by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** probability that a randomly chosen positive case receives a higher
  score than a randomly chosen negative case.
- **Confusion matrix:** counts of correct and incorrect predictions by class.

No single metric is sufficient. In a screening-style setting, false negatives
are particularly important, so recall and the confusion matrix should be read
alongside the model-ranking metric.

## Limitations and Next Steps

The holdout test is easy to explain but depends on one split. Stronger follow-up
work would add repeated stratified cross-validation, hyperparameter tuning nested
inside training folds, probability calibration, confidence intervals, threshold
selection based on error costs, and evaluation on data from another institution.

