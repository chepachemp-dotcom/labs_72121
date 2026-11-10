# Lab 4: Data pipeline and quality

## 1. Goal

Build a reproducible, leakage-free data pipeline for the UCI Adult dataset (OpenML `adult`, version 2)
and check its quality. The task is to predict whether income is >50K from 14 mixed numerical and
categorical features. Success is measured by F1 and accuracy on a held-out test set. The constraints are:
no leakage from test to train, and a pipeline that anyone can rerun with one command and get the same
processed data.

## 2. Method

- **Ingestion:** `fetch_openml` loads the raw data without modification: 48842 rows, 15 columns. A raw copy is saved to `data/adult_raw.csv`.
- **Validation:** types, ranges (age 17–90, hours 1–99, capital-gain 0–99999, capital-loss 0–4356), `?` as an allowed missing value, and allowed labels. Violations are reported and no rows are dropped.
- **Split:** 70/30 stratified (seed 42) before any cleaning: train 34189, test 14653.
- **Cleaning (fitted on train only):** mode imputation for categorical columns, median imputation for numeric columns, clipping to the train 1st/99th percentiles. 24 exact duplicates were removed from train only, and test was left untouched. The fitted transformer is saved to `results/preprocessor.joblib`.
- **Derived features:** `capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. Both use only values from the same row, so they cannot leak.
- **Baseline model:** LogisticRegression (standard scaling plus one-hot encoding), evaluated with and without the derived features.
- **Label quality:** a duplicate-conflict audit on the full data (identical features with different labels), plus a random sample of 50 rows checked against the data dictionary.
- **Reproducibility:** SHA256 of the processed train and test CSVs, stored with the config and package versions in `results/run_manifest.yaml`. Everything runs with `python run_pipeline.py`.

## 3. Results

| Item | Result |
|---|---|
| Rows / columns | 48842 / 15 |
| Validation violations | 0 (no rows dropped) |
| Train / test size | 34189 / 14653 |
| Exact duplicates removed (train only) | 24 |
| Duplicate-conflict groups (identical features, different labels) | 5 groups (10 rows) |
| 50-row sample flagged for impossible values | 0 |

| Model features | Accuracy | F1 |
|---|---|---|
| Base features | 0.8540 | 0.6601 |
| Base + derived features | 0.8578 | 0.6696 |

Processed-data hashes (SHA256):
- train: `71b349a557facae96c7b16701ce9ae45a8b00bd80b1188c484cf656ecdeae642`
- test: `ba6a19169035989a50c07d3729e82f30ef74adfe8d761c096ab9a7834197be24`
- combined: `f74ee1c542b1d80bb163d9cef983a1410eaf843dad64cba980e395e634c76b17`

Missing values (`?`) per column: see `results/validation_report.md`. Data card: `data_card.md`.

## 4. Conclusions

1. The raw Adult data is structurally clean: there were no type, range or label violations, and the only quality issues were missing values marked `?` and a few duplicates. Real quality problems here are subtle (missingness, label noise), not impossible values.
2. The derived features gave a small gain (accuracy +0.004, F1 +0.010) on a single split. This is a modest improvement, and without repeated splits I can't say it is statistically significant.
3. Doing the split first and fitting every statistic on train only keeps the test set clean, and fixing the seed and hashing the output makes the run verifiable. The 5 conflicting duplicate groups show a small amount of label noise, which caps how well any model can do.