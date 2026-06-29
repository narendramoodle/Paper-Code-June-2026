"""
data_prep.py
============
Data loading, cleaning, and harmonization for the open-data multi-disease
prediction study.

Datasets (all public, no credentialing required):
  - Heart disease : UCI / Kaggle combined heart dataset (~1,190 rows, 11 features)
                    https://archive.ics.uci.edu/dataset/45/heart+disease
                    (combined version widely mirrored on Kaggle as "heart.csv")
  - Diabetes      : Pima Indians Diabetes (768 rows, 8 features)
                    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  - Chronic Kidney: UCI Chronic Kidney Disease (400 rows, 24 features)
                    https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

DESIGN NOTES (these map directly to reviewer concern #1 — reproducibility):
  * Each dataset is loaded with an EXPLICIT, documented schema (see SCHEMAS).
  * Inclusion/exclusion rules are applied in code and logged.
  * Missing-data handling is explicit and per-column-type.
  * Each row is given a stable patient_id so splitting is patient-level.
  * A machine-readable feature dictionary is emitted to outputs/feature_dictionary.csv

This script is written to run on Google Colab / Kaggle. Where a dataset is not
present locally it prints the exact download instruction rather than failing
silently.

Author: (your name)
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ----------------------------------------------------------------------------
# 0. Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT, "data")
OUT_DIR = os.path.join(PROJECT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Explicit dataset schemas  (documentation + validation in one place)
# ----------------------------------------------------------------------------
@dataclass
class DatasetSchema:
    name: str
    filename: str            # expected local CSV filename in data/
    download_hint: str       # exact instruction if file is missing
    target_col: str          # raw target column name
    positive_rule: str       # human-readable description of "positive" label
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    # sentinel values that actually mean "missing" in these datasets
    missing_sentinels: dict = field(default_factory=dict)


SCHEMAS = {
    "heart": DatasetSchema(
        name="heart",
        filename="heart.csv",
        download_hint=(
            "Download the combined heart dataset (1190 rows, 11 features + target). "
            "Common source: Kaggle 'fedesoriano/heart-failure-prediction' (heart.csv) "
            "or UCI Heart Disease. Place it at data/heart.csv"
        ),
        target_col="HeartDisease",          # 1 = disease, 0 = normal
        positive_rule="HeartDisease == 1",
        numeric_cols=["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
        categorical_cols=["Sex", "ChestPainType", "FastingBS", "RestingECG",
                          "ExerciseAngina", "ST_Slope"],
        # In this dataset Cholesterol == 0 and RestingBP == 0 are physiologically
        # impossible and are used as missing-value sentinels.
        missing_sentinels={"Cholesterol": 0, "RestingBP": 0},
    ),
    "diabetes": DatasetSchema(
        name="diabetes",
        filename="diabetes.csv",
        download_hint=(
            "Download Pima Indians Diabetes (768 rows). Kaggle: "
            "'uciml/pima-indians-diabetes-database' (diabetes.csv). "
            "Place it at data/diabetes.csv"
        ),
        target_col="Outcome",               # 1 = diabetes, 0 = not
        positive_rule="Outcome == 1",
        numeric_cols=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        categorical_cols=[],
        # 0 is biologically impossible for these and denotes missing in Pima.
        missing_sentinels={"Glucose": 0, "BloodPressure": 0, "SkinThickness": 0,
                           "Insulin": 0, "BMI": 0},
    ),
    "kidney": DatasetSchema(
        name="kidney",
        filename="kidney_disease.csv",
        download_hint=(
            "Download UCI Chronic Kidney Disease (400 rows). Kaggle: "
            "'mansoordaku/ckdisease' (kidney_disease.csv). "
            "Place it at data/kidney_disease.csv"
        ),
        target_col="classification",        # 'ckd' / 'notckd'
        positive_rule="classification == 'ckd'",
        numeric_cols=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
                      "pot", "hemo", "pcv", "wc", "rc"],
        categorical_cols=["rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
                          "appet", "pe", "ane"],
        missing_sentinels={},               # uses '?' and blanks, handled below
    ),
}


# ----------------------------------------------------------------------------
# 2. Loading with explicit inclusion/exclusion logging
# ----------------------------------------------------------------------------
def _load_raw(schema: DatasetSchema) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, schema.filename)
    if not os.path.exists(path):
        print(f"[MISSING] {schema.name}: file not found at {path}")
        print(f"          -> {schema.download_hint}\n")
        return None
    df = pd.read_csv(path)
    print(f"[LOAD] {schema.name}: {len(df)} raw rows, {df.shape[1]} columns")
    return df


def _clean_common(df: pd.DataFrame, schema: DatasetSchema, log: dict) -> pd.DataFrame:
    """Apply sentinel->NaN, type coercion, and record what happened."""
    df = df.copy()

    # CKD dataset uses '?' and stray whitespace/tab characters
    if schema.name == "kidney":
        df = df.replace({"?": np.nan, "\t?": np.nan, "": np.nan})
        # several numeric columns are read as object due to stray characters
        for c in schema.numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                    errors="coerce")
        # normalize categorical text (strip, lowercase, fix known typos)
        for c in schema.categorical_cols:
            if c in df.columns:
                df[c] = (df[c].astype(str).str.strip().str.lower()
                         .replace({"nan": np.nan, "ckd\t": "ckd",
                                   "\tno": "no", "\tyes": "yes", " yes": "yes"}))
        df[schema.target_col] = (df[schema.target_col].astype(str)
                                 .str.strip().str.lower()
                                 .replace({"ckd\t": "ckd"}))

    # sentinel values -> NaN (Pima / heart)
    for col, sentinel in schema.missing_sentinels.items():
        if col in df.columns:
            n_before = (df[col] == sentinel).sum()
            df[col] = df[col].replace(sentinel, np.nan)
            log.setdefault("sentinel_to_nan", {})[col] = int(n_before)

    return df


def _build_label(df: pd.DataFrame, schema: DatasetSchema) -> pd.Series:
    if schema.name == "heart":
        y = (df["HeartDisease"] == 1).astype(int)
    elif schema.name == "diabetes":
        y = (df["Outcome"] == 1).astype(int)
    elif schema.name == "kidney":
        y = (df["classification"] == "ckd").astype(int)
    else:
        raise ValueError(schema.name)
    return y.rename("label")


def load_dataset(key: str) -> dict | None:
    """
    Returns a dict with keys:
       name, X (DataFrame of features), y (Series 0/1),
       numeric_cols, categorical_cols, log (inclusion/exclusion record)
    or None if the raw file is absent.
    """
    schema = SCHEMAS[key]
    raw = _load_raw(schema)
    if raw is None:
        return None

    log = {"dataset": key, "raw_rows": int(len(raw))}
    df = _clean_common(raw, schema, log)

    # ---- inclusion/exclusion criteria (explicit, logged) ----
    # Rule 1: a usable row must have a non-null target.
    before = len(df)
    df = df[df[schema.target_col].notna()].copy()
    log["dropped_missing_target"] = int(before - len(df))

    # Rule 2: drop rows with > 50% of feature columns missing (too sparse
    # to populate the feature vector). Threshold is a documented choice.
    feat_cols = [c for c in (schema.numeric_cols + schema.categorical_cols)
                 if c in df.columns]
    frac_missing = df[feat_cols].isna().mean(axis=1)
    before = len(df)
    df = df[frac_missing <= 0.50].copy()
    log["missing_threshold"] = 0.50
    log["dropped_too_sparse"] = int(before - len(df))

    y = _build_label(df, schema)
    X = df[feat_cols].copy()

    # stable patient_id: hash of row content so splitting is deterministic &
    # patient-level (each row here is one patient; no repeated visits in these sets)
    df_reset = df.reset_index(drop=True)
    row_strings = df_reset.apply(
        lambda row: "|".join(str(v) for v in row.tolist()), axis=1)
    pid = row_strings.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X.insert(0, "patient_id", pid.values)

    log["final_rows"] = int(len(X))
    log["positive_rate"] = float(y.mean())
    log["n_numeric"] = len([c for c in schema.numeric_cols if c in X.columns])
    log["n_categorical"] = len([c for c in schema.categorical_cols if c in X.columns])

    print(f"[CLEAN] {key}: {log['final_rows']} usable rows "
          f"(pos rate {log['positive_rate']:.3f}); "
          f"dropped {log['dropped_missing_target']} (no target), "
          f"{log['dropped_too_sparse']} (too sparse)")

    return {
        "name": key,
        "X": X,
        "y": y,
        "numeric_cols": [c for c in schema.numeric_cols if c in X.columns],
        "categorical_cols": [c for c in schema.categorical_cols if c in X.columns],
        "log": log,
    }


# ----------------------------------------------------------------------------
# 3. Feature dictionary export (reviewer concern #1: feature dictionary)
# ----------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    # heart
    "Age": "Age in years",
    "Sex": "Sex (M/F)",
    "ChestPainType": "Chest pain type (TA, ATA, NAP, ASY)",
    "RestingBP": "Resting blood pressure (mm Hg)",
    "Cholesterol": "Serum cholesterol (mg/dl); 0 treated as missing",
    "FastingBS": "Fasting blood sugar > 120 mg/dl (1/0)",
    "RestingECG": "Resting electrocardiogram results (Normal, ST, LVH)",
    "MaxHR": "Maximum heart rate achieved",
    "ExerciseAngina": "Exercise-induced angina (Y/N)",
    "Oldpeak": "ST depression induced by exercise",
    "ST_Slope": "Slope of peak exercise ST segment (Up, Flat, Down)",
    # diabetes (Pima)
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose concentration (2-h OGTT); 0 = missing",
    "BloodPressure": "Diastolic blood pressure (mm Hg); 0 = missing",
    "SkinThickness": "Triceps skinfold thickness (mm); 0 = missing",
    "Insulin": "2-h serum insulin (mu U/ml); 0 = missing",
    "BMI": "Body mass index (kg/m^2); 0 = missing",
    "DiabetesPedigreeFunction": "Diabetes pedigree function (family history score)",
    # kidney (CKD)
    "age": "Age in years", "bp": "Blood pressure (mm Hg)",
    "sg": "Specific gravity of urine", "al": "Albumin level",
    "su": "Sugar level", "rbc": "Red blood cells (normal/abnormal)",
    "pc": "Pus cell (normal/abnormal)", "pcc": "Pus cell clumps (present/notpresent)",
    "ba": "Bacteria (present/notpresent)", "bgr": "Blood glucose random (mg/dl)",
    "bu": "Blood urea (mg/dl)", "sc": "Serum creatinine (mg/dl)",
    "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)", "hemo": "Hemoglobin (g/dl)",
    "pcv": "Packed cell volume", "wc": "White blood cell count (cells/cmm)",
    "rc": "Red blood cell count (millions/cmm)", "htn": "Hypertension (yes/no)",
    "dm": "Diabetes mellitus (yes/no)", "cad": "Coronary artery disease (yes/no)",
    "appet": "Appetite (good/poor)", "pe": "Pedal edema (yes/no)",
    "ane": "Anemia (yes/no)",
}


def export_feature_dictionary(datasets: dict):
    rows = []
    for key, d in datasets.items():
        if d is None:
            continue
        for c in d["numeric_cols"]:
            rows.append({"dataset": key, "feature": c, "type": "numeric",
                         "description": FEATURE_DESCRIPTIONS.get(c, "")})
        for c in d["categorical_cols"]:
            rows.append({"dataset": key, "feature": c, "type": "categorical",
                         "description": FEATURE_DESCRIPTIONS.get(c, "")})
    fd = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "feature_dictionary.csv")
    fd.to_csv(path, index=False)
    print(f"[EXPORT] feature dictionary -> {path} ({len(fd)} features)")
    return fd


def export_cohort_log(datasets: dict):
    logs = {k: (d["log"] if d else {"status": "missing file"})
            for k, d in datasets.items()}
    path = os.path.join(OUT_DIR, "cohort_log.json")
    with open(path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"[EXPORT] cohort construction log -> {path}")
    return logs


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Multi-disease open-data preparation")
    print("=" * 70)
    datasets = {k: load_dataset(k) for k in SCHEMAS}

    present = {k: v for k, v in datasets.items() if v is not None}
    if not present:
        print("\nNo datasets found locally. Download the CSVs into data/ using the")
        print("hints above, then re-run. (UCI/Kaggle files are small and open.)")
        return

    export_feature_dictionary(datasets)
    export_cohort_log(datasets)

    print("\nSummary of usable cohorts:")
    for k, d in present.items():
        print(f"  {k:10s}  n={d['log']['final_rows']:5d}  "
              f"pos_rate={d['log']['positive_rate']:.3f}  "
              f"features={d['log']['n_numeric']}num+{d['log']['n_categorical']}cat")


if __name__ == "__main__":
    main()
