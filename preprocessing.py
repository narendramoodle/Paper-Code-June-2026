"""
preprocessing.py
================
Leakage-safe preprocessing and patient-level cross-validation splitting.

Directly answers the reviewer's leakage-control requirement:
  * All splits are PATIENT-LEVEL (group-aware) using patient_id.
  * Imputation statistics, scalers, and encoders are fit on the TRAINING FOLD
    ONLY and then applied to validation/test. Nothing from held-out data
    touches preprocessing.
  * A single held-out test partition is created first; cross-validation runs
    inside the remaining development partition.

Typical usage (per dataset):
    from preprocessing import make_holdout_split, kfold_iter, Preprocessor
    dev, test = make_holdout_split(X, y, test_frac=0.15)
    for tr, va in kfold_iter(dev["X"], dev["y"], n_splits=5):
        pp = Preprocessor(numeric_cols, categorical_cols).fit(tr["X"])
        Xtr = pp.transform(tr["X"]); Xva = pp.transform(va["X"])
        ...
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

RANDOM_SEED = 42


# ----------------------------------------------------------------------------
# Held-out test split (patient-level, stratified)
# ----------------------------------------------------------------------------
def make_holdout_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.15,
                       seed: int = RANDOM_SEED) -> tuple[dict, dict]:
    """
    Split into development and held-out test partitions at the patient level.
    Each patient_id lands entirely in one partition.
    """
    groups = X["patient_id"].values
    # number of folds whose 1/k ~ test_frac, e.g. test_frac 0.15 -> ~7 folds
    n_splits = max(2, round(1.0 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(X, y, groups))

    # guarantee no patient overlap (defensive check)
    assert set(groups[dev_idx]).isdisjoint(set(groups[test_idx])), \
        "Patient leakage between dev and test!"

    dev = {"X": X.iloc[dev_idx].reset_index(drop=True),
           "y": y.iloc[dev_idx].reset_index(drop=True)}
    test = {"X": X.iloc[test_idx].reset_index(drop=True),
            "y": y.iloc[test_idx].reset_index(drop=True)}
    return dev, test


def kfold_iter(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
               seed: int = RANDOM_SEED):
    """Yield (train_dict, val_dict) for patient-level stratified k-fold."""
    groups = X["patient_id"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in sgkf.split(X, y, groups):
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx])), \
            "Patient leakage between train and val!"
        tr = {"X": X.iloc[tr_idx].reset_index(drop=True),
              "y": y.iloc[tr_idx].reset_index(drop=True)}
        va = {"X": X.iloc[va_idx].reset_index(drop=True),
              "y": y.iloc[va_idx].reset_index(drop=True)}
        yield tr, va


# ----------------------------------------------------------------------------
# Preprocessor: fit on training fold only
# ----------------------------------------------------------------------------
@dataclass
class Preprocessor:
    numeric_cols: list
    categorical_cols: list
    # learned state (set during fit)
    medians_: dict = field(default_factory=dict)
    modes_: dict = field(default_factory=dict)
    means_: dict = field(default_factory=dict)
    stds_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    feature_names_: list = field(default_factory=list)
    fitted_: bool = False

    def fit(self, X: pd.DataFrame):
        # numeric: median for imputation, mean/std for standardization
        for c in self.numeric_cols:
            col = pd.to_numeric(X[c], errors="coerce")
            self.medians_[c] = float(col.median())
            filled = col.fillna(self.medians_[c])
            self.means_[c] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.stds_[c] = std if std > 1e-8 else 1.0
        # categorical: mode for imputation, observed categories for one-hot
        for c in self.categorical_cols:
            col = X[c].astype("object")
            mode = col.mode(dropna=True)
            self.modes_[c] = mode.iloc[0] if len(mode) else "missing"
            cats = sorted(col.fillna(self.modes_[c]).astype(str).unique().tolist())
            self.categories_[c] = cats
        # build output feature-name list
        names = list(self.numeric_cols)
        for c in self.categorical_cols:
            names += [f"{c}={lvl}" for lvl in self.categories_[c]]
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "Call fit() on training fold before transform()."
        out = {}
        # numeric: impute with TRAIN median, standardize with TRAIN mean/std
        for c in self.numeric_cols:
            col = pd.to_numeric(X[c], errors="coerce").fillna(self.medians_[c])
            out[c] = (col - self.means_[c]) / self.stds_[c]
        # categorical: impute with TRAIN mode, one-hot against TRAIN categories
        for c in self.categorical_cols:
            col = X[c].astype("object").fillna(self.modes_[c]).astype(str)
            for lvl in self.categories_[c]:
                out[f"{c}={lvl}"] = (col == lvl).astype(float)
        return pd.DataFrame(out, index=X.index)[self.feature_names_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# quick self-test ------------------------------------------------------------
if __name__ == "__main__":
    # tiny synthetic frame to confirm shapes & no-leakage assertions
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "chol": rng.integers(0, 300, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
    })
    X.loc[rng.choice(n, 5, replace=False), "chol"] = np.nan
    y = pd.Series(rng.integers(0, 2, n), name="label")

    dev, test = make_holdout_split(X, y, test_frac=0.20)
    print("dev", len(dev["X"]), "test", len(test["X"]))
    for i, (tr, va) in enumerate(kfold_iter(dev["X"], dev["y"], n_splits=3)):
        pp = Preprocessor(["age", "chol"], ["sex"]).fit(tr["X"])
        Xtr, Xva = pp.transform(tr["X"]), pp.transform(va["X"])
        print(f"fold {i}: train {Xtr.shape} val {Xva.shape} feats {pp.feature_names_}")
    print("preprocessing self-test OK")
