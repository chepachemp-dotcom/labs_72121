# Problem definition

**Task.** Predict whether a person's annual income exceeds $50K from 1994 US Census records
(UCI Adult, OpenML `adult` v2). This is a binary classification problem with 48,842 rows.

**Inputs.** The 14 features: age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
Six are numeric and eight are categorical, and some categorical fields contain missing values
(`?`). Two derived features are added: `capital_net` and `hours_per_age`.

**Output.** `income` with classes `<=50K` and `>50K` (positive class `>50K`, about 24% of rows).

**Success metrics.** F1 on the positive class and accuracy, both on a held-out 30% test set.
Because of the class imbalance, F1 is the primary metric, since a majority-class predictor
already reaches about 76% accuracy. A LogisticRegression baseline provides the reference
scores, and the effect of the derived features is measured against it.

**Constraints.**
1. *No leakage.* The train/test split (70/30, stratified) happens first. Imputation values,
   clipping bounds, and scaling are fitted on train only and applied to test. Duplicate removal
   touches train only. Derived features use only same-row values.
2. *Reproducibility.* All randomness uses a fixed seed from `config.yaml`. The processed data is
   SHA256-hashed, the config and package versions are stored in a run manifest, and the whole
   pipeline runs with one command: `python run_pipeline.py`.
3. *Transparency.* Schema, range, missing-value, duplicate, and label-quality checks are
   reported. Violations are logged and rows are never silently dropped.

**Risks.** Sensitive attributes (sex, race, native-country) and historical bias; see `data_card.md`.