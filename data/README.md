# Data

By default, the project uses scikit-learn's built-in Wisconsin Diagnostic
Breast Cancer dataset. It is loaded directly by `src/train.py`, so no download
or data file is required.

To use your own dataset, place a CSV file in this directory and run:

```powershell
python src/train.py --data-path data/your_data.csv --target outcome
```

Custom-data requirements:

- one header row;
- one binary target column (exactly two outcome classes);
- numeric predictor columns;
- at least two rows from each class;
- missing predictor values are allowed and are median-imputed.

The class that sorts second alphabetically is treated as the positive class for
custom data. The generated `artifacts/run_metadata.json` records the mapping.

