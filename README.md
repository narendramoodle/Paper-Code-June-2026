# Multi-Disease Prediction with Explainable Ensembles — Real Experiment Code

This repository runs a **real, reproducible** disease-prediction study on **open
clinical datasets** (no credentialing needed) and produces every result the
journal reviewers asked for. Nothing here is synthetic: you download real data,
run the code, and get real numbers to put in the paper.

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


The numbers in `outputs/results_*.csv.


