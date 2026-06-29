# Multi-Disease Prediction with Explainable Ensembles — Real Experiment Code

This repository runs a **real, reproducible** disease-prediction study on **open
clinical datasets** (no credentialing needed) and produces every result the
journal reviewers asked for. Nothing here is synthetic: you download real data,
run the code, and get real numbers to put in the paper.

## Scope (honest)

This is a **disease-prediction + explainability + calibration + fairness** study
on three open datasets:

| Disease | Dataset | Rows | Source |
|---|---|---|---|
| Heart disease | UCI / Kaggle combined heart | ~918 | `fedesoriano/heart-failure-prediction` |
| Diabetes | Pima Indians Diabetes | 768 | `uciml/pima-indians-diabetes-database` |
| Chronic kidney disease | UCI CKD | 400 | `mansoordaku/ckdisease` |

The **doctor/medicine recommendation** and **knowledge-graph safety** components
from the original manuscript are **not** included here, because they require
prescription/admission data (e.g., MIMIC-IV, which needs CITI training +
PhysioNet credentialing). Those are documented as future work. Don't claim them
in a paper built only on this code.

## What it produces (maps to reviewer concerns)

| Reviewer concern | Module | Output |
|---|---|---|
| #1 Reproducible dataset construction | `data_prep.py` | `feature_dictionary.csv`, `cohort_log.json` (inclusion/exclusion, missing-data rules) |
| Leakage control | `preprocessing.py` | patient-level CV, fold-only imputation/scaling |
| #4 CNN outdated → modern baselines | `models.py` | TabNet + FT-Transformer baselines |
| Calibration | `calibration.py` | Brier, ECE, MCE, reliability diagrams |
| #6 Quantified explainability | `explainability.py` | SHAP vs LIME fidelity & stability table |
| Fairness | `fairness.py` | performance by age & sex, fairness gaps |
| Error analysis | `error_analysis.py` | per-disease FP/FN, failure cases |
| External-ish validation | `run_all.py` | held-out test partition + (optional) cross-dataset |

## How to run

**Easiest:** open `run_experiments_colab.ipynb` in Google Colab and follow the
seven steps. You only need to supply the three CSVs (free, links above).

**Locally:**
```bash
pip install -r requirements.txt
# put heart.csv, diabetes.csv, kidney_disease.csv in data/
python src/data_prep.py        # inspect cohorts, emit feature dictionary
python src/run_all.py --tabnet # full pipeline (omit --tabnet to skip it)
```

Results are written to `outputs/` as CSVs and PNGs.

## Models

- Classical: Logistic Regression, Random Forest, XGBoost
- Neural: MLP, **FT-Transformer** (from scratch), **TabNet** (via `pytorch-tabnet`)
- **Stacking ensemble** with out-of-fold meta-features (leakage-safe)

## Reproducibility

- All randomness is seeded (`SEED = 42`).
- Splits are patient-level and stratified; assertions guard against patient
  leakage between train/val/test.
- Preprocessing is fit on training folds only.
- `cohort_log.json` records exactly how many rows were dropped and why.

## After you have real numbers

The numbers in `outputs/results_*.csv` are what go into the manuscript, replacing
every illustrative placeholder. At that point the results section describes your
own experiments — which is both the honest version and the one that survives peer
review.

## Important

Run this on the real downloaded data. The author is responsible for verifying
every number before submission and for following the target journal's policy on
AI-assisted tools and on data/code availability (consider releasing this repo as
the paper's code-availability statement).
