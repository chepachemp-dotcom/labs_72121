# Data card: UCI Adult (OpenML `adult`, version 2)

## Source
UCI Adult / Census Income dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census
database, obtained via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
Hosted by the UCI Machine Learning Repository under CC BY 4.0. Check the OpenML page for the
current terms before redistributing.

## Size
- Raw: 48842 rows x 15 columns (14 features + `income`).
- Train/test split: 70/30 stratified, seed 42.
- Train after exact-duplicate removal: 34165 rows (removed 24).
- Test (untouched): 14653 rows.

## Processing
Fitted on train only: mode imputation (categorical), median imputation (numeric), clipping to the
train 1st/99th percentiles. Derived features: `capital_net`, `hours_per_age`.
Fitted transformer: `results/preprocessor.joblib`.

## Integrity
- SHA256 processed train: `71b349a557facae96c7b16701ce9ae45a8b00bd80b1188c484cf656ecdeae642`
- SHA256 processed test: `ba6a19169035989a50c07d3729e82f30ef74adfe8d761c096ab9a7834197be24`
- Combined hash: `f74ee1c542b1d80bb163d9cef983a1410eaf843dad64cba980e395e634c76b17`

## Quality findings
- Validation violations: 0 (see `results/validation_report.md`).
- Missing values are marked `?` / NaN in workclass, occupation, native-country.
- Duplicate-conflict audit: 5 feature groups with
  conflicting labels (10 rows).
- 50-row sample auto-flagged: 0 (manual review in
  `results/label_audit_sample50.csv`).

## Baseline (LogisticRegression)
| features | accuracy | F1 |
|---|---|---|
| base | 0.8540 | 0.6601 |
| base + derived | 0.8578 | 0.6696 |

## Known biases
- Data is from 1994 and reflects US labor-market and pay structure of that time.
- Strong class imbalance (about 24% earn >50K); accuracy alone is misleading, hence F1.
- Sex and race are imbalanced and income labels differ systematically across groups, so models can
  reproduce historical discrimination. Report subgroup metrics before any real use.
- `native-country` is dominated by the United States; other countries have very few rows.
- Missingness is not random (concentrated in workclass/occupation).
- `capital-gain` is heavily skewed with a sentinel-like cap at 99999.

## Privacy concerns
Records are anonymized (no names or IDs), but combinations of age, sex, race, country, and
occupation are quasi-identifiers that could enable re-identification when linked with other
data. `fnlwgt` is a census sampling weight, not a person feature. Do not attempt re-identification.
