# Findings and Conclusion

## Main Findings

1. All three supervised-learning algorithms learned useful patterns from the
   diagnostic measurements and performed better than random classification.
2. Random Forest provided the strongest overall class-ranking ability, reaching
   a ROC-AUC of **0.9970**.
3. Logistic Regression achieved the best predictions at the default threshold,
   with **97.37% accuracy** and **95.24% recall** for malignant cases.
4. The Decision Tree produced the lowest recall (**78.57%**), meaning it missed
   more malignant cases than the other models.
5. The strong Logistic Regression result suggests the standardized predictors
   contain a highly separable linear signal. The Random Forest still gained a
   small ROC-AUC advantage by representing nonlinear relationships.
6. The most influential Random Forest features included worst area, worst
   perimeter, worst concave points, mean concave points, and worst radius.

## Interpretation

The Random Forest is appropriate when ranking quality across thresholds is the
main goal. Logistic Regression may be preferred when simplicity, explainability,
and performance at the default threshold matter most. The confusion matrices
and ROC curves should therefore be considered alongside the summary scores.

## Skills Gained

This project demonstrates experience with:

- preparing data for supervised machine learning;
- creating leakage-safe preprocessing pipelines;
- training linear and tree-based classifiers;
- using a stratified training and testing split;
- comparing accuracy, precision, recall, F1-score, and ROC-AUC;
- visualizing confusion matrices, ROC curves, and feature importance;
- saving a trained pipeline and using it for new predictions;
- documenting results, limitations, and reproducibility.

## Limitations

- The results use one train/test split and may vary with another sample.
- The dataset is relatively small and represents a specific clinical context.
- Feature importance shows predictive association, not causation.
- Real medical use would require external validation, calibration, fairness
  assessment, privacy safeguards, and approval from qualified professionals.

## Conclusion

The assignment objective was achieved successfully. Multiple supervised-learning
models were trained and tested, their predictive performance was quantified and
visualized, and the selected model was saved for later use. Random Forest ranked
first by ROC-AUC, while Logistic Regression delivered the strongest default-
threshold accuracy and F1-score.

