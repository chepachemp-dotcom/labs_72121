#!/usr/bin/env python

import hashlib
import json
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lab4_preprocessing import DERIVED_COLS, Cleaner

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
OUT = HERE / CFG["paths"]["results_dir"]
OUT.mkdir(exist_ok=True)
TARGET = CFG["target"]
SEED = CFG["seed"]
NUM_COLS = [c for c, t in CFG["schema"]["types"].items() if t == "int"]
CAT_COLS = [c for c, t in CFG["schema"]["types"].items() if t == "category" and c != TARGET]
FEATURES = NUM_COLS + CAT_COLS

EDU_MAP = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "pipeline.log", mode="w")])
log = logging.getLogger("lab4")


# ---------------------------------------------------------------- 2. ingestion
def ingest():
    ds = CFG["dataset"]
    df = fetch_openml(name=ds["name"], version=ds["version"], as_frame=True).frame
    df = df.rename(columns={"class": TARGET})  # only a rename; values untouched
    log.info("Ingested raw data: %d rows, %d columns", *df.shape)
    exp = CFG["expected"]
    if df.shape != (exp["rows"], exp["cols"]):
        log.warning("Shape differs from expected (%d, %d)", exp["rows"], exp["cols"])
    raw_path = (HERE / CFG["paths"]["raw_csv"]).resolve()
    raw_path.parent.mkdir(exist_ok=True)
    df.to_csv(raw_path, index=False)
    return df


# ---------------------------------------------------------------- 3. validation
def validate(df):
    sc, rows = CFG["schema"], []

    def add(check, col, n, detail=""):
        rows.append({"check": check, "column": col, "violations": int(n), "detail": detail})

    exp = CFG["expected"]
    add("row_count", "-", df.shape[0] != exp["rows"], f"got {df.shape[0]}, expected {exp['rows']}")
    add("col_count", "-", df.shape[1] != exp["cols"], f"got {df.shape[1]}, expected {exp['cols']}")
    missing_cols = [c for c in sc["types"] if c not in df.columns]
    add("columns_present", "-", len(missing_cols), ",".join(missing_cols))

    for col, t in sc["types"].items():
        if col not in df.columns:
            continue
        s = df[col]
        n_na = s.isna().sum() + (s.astype(object) == sc["missing_token"]).sum()
        if t == "int":
            if pd.api.types.is_numeric_dtype(s):
                n_bad = int((s.dropna() % 1 != 0).sum())
            else:
                n_bad = len(s)
            add("type_int", col, n_bad, "non-integer values" if n_bad else "ok")
            add("null_in_numeric", col, s.isna().sum(), "")
        else:
            n_bad = len(s) if pd.api.types.is_numeric_dtype(s) else 0
            add("type_category", col, n_bad, f"{s.nunique(dropna=True)} distinct values")
            add("missing_info('?' or NaN; allowed)", col, 0, f"{int(n_na)} missing ({n_na / len(s):.1%})")

    for col, (lo, hi) in sc["ranges"].items():
        s = df[col]
        add("range", col, ((s < lo) | (s > hi)).sum(), f"allowed [{lo}, {hi}], observed [{s.min()}, {s.max()}]")

    bad_lab = ~df[TARGET].isin(sc["allowed_income"])
    add("allowed_labels", TARGET, bad_lab.sum(), f"allowed {sc['allowed_income']}")
    add("exact_duplicate_rows (info, not dropped here)", "-", 0, f"{int(df.duplicated().sum())} duplicates")

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "validation_report.csv", index=False)
    total = int(rep["violations"].sum())
    md = ["# Validation report", "", f"Rows: {df.shape[0]}, columns: {df.shape[1]}. "
          f"Total violations: **{total}**. No rows were dropped.", "",
          "| check | column | violations | detail |", "|---|---|---|---|"]
    md += [f"| {r.check} | {r.column} | {r.violations} | {r.detail} |" for r in rep.itertuples()]
    (OUT / "validation_report.md").write_text("\n".join(md))
    log.info("Validation done: %d violations (report in results/validation_report.md)", total)
    return total


# ---------------------------------------------------------------- 5. model eval
def fit_eval(add_derived, Xtr, ytr, Xte, yte):
    cleaner = Cleaner(NUM_COLS, CAT_COLS, tuple(CFG["clip_quantiles"]), add_derived,
                      CFG["schema"]["missing_token"]).fit(Xtr)
    Ctr, Cte = cleaner.transform(Xtr), cleaner.transform(Xte)
    num = NUM_COLS + (DERIVED_COLS if add_derived else [])
    enc = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    model = Pipeline([("enc", enc), ("lr", LogisticRegression(max_iter=2000))]).fit(Ctr, ytr)
    pred = model.predict(Cte)
    return cleaner, Ctr, Cte, {"accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred)}


# ---------------------------------------------------------------- 6. label quality
def label_audit(df):
    feats = df[FEATURES].astype(object).fillna("?")
    keyed = feats.assign(_y=df[TARGET].astype(str))
    g = keyed.groupby(FEATURES, dropna=False)["_y"].agg(["nunique", "size"])
    conflict = g[g["nunique"] > 1]
    res = {"conflicting_feature_groups": int(len(conflict)),
           "rows_in_conflicting_groups": int(conflict["size"].sum())}

    sample = df.sample(50, random_state=SEED).copy()
    flags = []
    for _, r in sample.iterrows():
        f = []
        for col, (lo, hi) in CFG["schema"]["ranges"].items():
            if not lo <= r[col] <= hi:
                f.append(f"{col} out of range")
        if EDU_MAP.get(r["education"]) != r["education-num"]:
            f.append("education vs education-num mismatch")
        if r[TARGET] not in CFG["schema"]["allowed_income"]:
            f.append("invalid label")
        flags.append("; ".join(f))
    sample["auto_flag"] = flags
    sample["manual_ok (fill in)"] = ""
    sample.to_csv(OUT / "label_audit_sample50.csv", index=False)
    res["sample50_auto_flagged"] = int((sample["auto_flag"] != "").sum())
    json.dump(res, open(OUT / "label_audit.json", "w"), indent=2)
    log.info("Label audit: %s", res)
    return res


# ---------------------------------------------------------------- 7. hashing + data card
def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_data_card(info):
    m = info["metrics"]
    card = f"""# Data card: UCI Adult (OpenML `adult`, version 2)

## Source
UCI Adult / Census Income dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census
database, obtained via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
Hosted by the UCI Machine Learning Repository under CC BY 4.0. Check the OpenML page for the
current terms before redistributing.

## Size
- Raw: {info['n_rows']} rows x {info['n_cols']} columns (14 features + `income`).
- Train/test split: 70/30 stratified, seed {SEED}.
- Train after exact-duplicate removal: {info['n_train']} rows (removed {info['n_dup_removed']}).
- Test (untouched): {info['n_test']} rows.

## Processing
Fitted on train only: mode imputation (categorical), median imputation (numeric), clipping to the
train 1st/99th percentiles. Derived features: `capital_net`, `hours_per_age`.
Fitted transformer: `results/preprocessor.joblib`.

## Integrity
- SHA256 processed train: `{info['hash_train']}`
- SHA256 processed test: `{info['hash_test']}`
- Combined hash: `{info['hash_all']}`

## Quality findings
- Validation violations: {info['violations']} (see `results/validation_report.md`).
- Missing values are marked `?` / NaN in workclass, occupation, native-country.
- Duplicate-conflict audit: {info['audit']['conflicting_feature_groups']} feature groups with
  conflicting labels ({info['audit']['rows_in_conflicting_groups']} rows).
- 50-row sample auto-flagged: {info['audit']['sample50_auto_flagged']} (manual review in
  `results/label_audit_sample50.csv`).

## Baseline (LogisticRegression)
| features | accuracy | F1 |
|---|---|---|
| base | {m['without_derived']['accuracy']:.4f} | {m['without_derived']['f1']:.4f} |
| base + derived | {m['with_derived']['accuracy']:.4f} | {m['with_derived']['f1']:.4f} |

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
"""
    (HERE / "data_card.md").write_text(card)


def main():
    df = ingest()
    n_viol = validate(df)
    audit = label_audit(df)

    X, y = df[FEATURES], (df[TARGET] == CFG["positive_label"]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=CFG["test_size"], stratify=y, random_state=SEED)
    log.info("Split: train=%d test=%d", len(Xtr), len(Xte))

    dup = pd.concat([Xtr, ytr.rename(TARGET)], axis=1).duplicated()  # train only
    n_dup = int(dup.sum())
    Xtr, ytr = Xtr[~dup], ytr[~dup]
    log.info("Dropped %d exact duplicate rows from train (test untouched)", n_dup)

    _, _, _, m_base = fit_eval(False, Xtr, ytr, Xte, yte)
    cleaner, Ctr, Cte, m_der = fit_eval(True, Xtr, ytr, Xte, yte)
    log.info("Without derived: %s", m_base)
    log.info("With derived:    %s", m_der)
    metrics = {"without_derived": m_base, "with_derived": m_der}
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)

    joblib.dump(cleaner, OUT / "preprocessor.joblib")

    ptr, pte = OUT / "processed_train.csv", OUT / "processed_test.csv"
    Ctr.assign(**{TARGET: ytr.values}).to_csv(ptr, index=False, float_format="%.6f")
    Cte.assign(**{TARGET: yte.values}).to_csv(pte, index=False, float_format="%.6f")
    h_tr, h_te = sha256_file(ptr), sha256_file(pte)
    h_all = hashlib.sha256((h_tr + h_te).encode()).hexdigest()
    log.info("SHA256 processed train=%s test=%s combined=%s", h_tr, h_te, h_all)

    manifest = {"config": CFG, "hashes": {"train": h_tr, "test": h_te, "combined": h_all},
                "metrics": metrics, "label_audit": audit, "validation_violations": n_viol,
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__}}
    (OUT / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    write_data_card({"n_rows": df.shape[0], "n_cols": df.shape[1], "n_train": len(Xtr), "n_test": len(Xte),
                     "n_dup_removed": n_dup, "hash_train": h_tr, "hash_test": h_te, "hash_all": h_all,
                     "violations": n_viol, "audit": audit, "metrics": metrics})
    log.info("Done. Outputs in %s", OUT)


if __name__ == "__main__":
    main()