"""
fairness.py
===========
Subgroup performance analysis by age and sex — answers the reviewer's fairness
requirement for healthcare AI.

Reports, per subgroup:
  * n, positive rate
  * accuracy, AUC (when both classes present), F1
  * selection rate (predicted-positive rate)
  * TPR / FPR  (for equal-opportunity / equalized-odds inspection)

Also reports common fairness gaps across subgroups:
  * accuracy gap, TPR gap, FPR gap, selection-rate gap (max - min)
  * demographic-parity difference, equal-opportunity difference

Subgroup membership is taken from the ORIGINAL (pre-encoding) frame so the
analysis is on real, human-readable attributes (e.g., Sex = M/F, age bands).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def _age_band(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(age):
        return "unknown"
    if age < 40: return "<40"
    if age < 55: return "40-54"
    if age < 70: return "55-69"
    return "70+"


def _group_metrics(y_true, y_pred, p):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); p = np.asarray(p)
    n = len(y_true)
    out = {"n": int(n), "pos_rate": float(np.mean(y_true)) if n else np.nan,
           "selection_rate": float(np.mean(y_pred)) if n else np.nan}
    out["accuracy"] = float(accuracy_score(y_true, y_pred)) if n else np.nan
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0)) if n else np.nan
    out["auc"] = (float(roc_auc_score(y_true, p))
                  if n and len(np.unique(y_true)) == 2 else np.nan)
    # TPR / FPR
    pos = y_true == 1; neg = y_true == 0
    out["tpr"] = float(np.mean(y_pred[pos] == 1)) if pos.sum() else np.nan
    out["fpr"] = float(np.mean(y_pred[neg] == 1)) if neg.sum() else np.nan
    return out


def subgroup_report(frame_raw: pd.DataFrame, y_true, p, threshold=0.5,
                    sex_col_candidates=("Sex", "sex"),
                    age_col_candidates=("Age", "age")):
    """
    frame_raw : original feature frame (pre-encoding) aligned row-wise with y_true/p
    Returns a dict of DataFrames: {'sex': df, 'age': df} plus a 'gaps' dict.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    y_pred = (p >= threshold).astype(int)

    reports = {}
    gaps = {}

    # --- sex ---
    sex_col = next((c for c in sex_col_candidates if c in frame_raw.columns), None)
    if sex_col is not None:
        rows = []
        for g, idx in frame_raw.groupby(frame_raw[sex_col].astype(str)).groups.items():
            ii = frame_raw.index.get_indexer(idx)
            m = _group_metrics(y_true[ii], y_pred[ii], p[ii]); m["group"] = g
            rows.append(m)
        df = pd.DataFrame(rows).set_index("group")
        reports["sex"] = df
        gaps["sex"] = _gaps(df)

    # --- age bands ---
    age_col = next((c for c in age_col_candidates if c in frame_raw.columns), None)
    if age_col is not None:
        bands = frame_raw[age_col].map(_age_band)
        rows = []
        for g, idx in frame_raw.groupby(bands).groups.items():
            ii = frame_raw.index.get_indexer(idx)
            m = _group_metrics(y_true[ii], y_pred[ii], p[ii]); m["group"] = g
            rows.append(m)
        order = {"<40": 0, "40-54": 1, "55-69": 2, "70+": 3, "unknown": 9}
        df = pd.DataFrame(rows).set_index("group")
        df = df.reindex(sorted(df.index, key=lambda x: order.get(x, 99)))
        reports["age"] = df
        gaps["age"] = _gaps(df)

    return {"reports": reports, "gaps": gaps}


def _gaps(df: pd.DataFrame) -> dict:
    def spread(col):
        v = df[col].dropna()
        return float(v.max() - v.min()) if len(v) >= 2 else np.nan
    return {
        "accuracy_gap": spread("accuracy"),
        "auc_gap": spread("auc"),
        "tpr_gap_equal_opportunity": spread("tpr"),
        "fpr_gap": spread("fpr"),
        "selection_rate_gap_demographic_parity": spread("selection_rate"),
    }


def format_fairness_tables(result: dict) -> str:
    lines = []
    for key, df in result["reports"].items():
        lines.append(f"\n=== Subgroup performance by {key} ===")
        show = df[["n", "pos_rate", "accuracy", "auc", "f1",
                   "selection_rate", "tpr", "fpr"]].round(3)
        lines.append(show.to_string())
        g = result["gaps"][key]
        lines.append(f"Gaps ({key}): " +
                     ", ".join(f"{k}={'' if v!=v else round(v,3)}" for k, v in g.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    raw = pd.DataFrame({
        "Age": rng.integers(25, 85, n),
        "Sex": rng.choice(["M", "F"], n),
    })
    # induce a mild subgroup difference for the test
    base = 0.3 + 0.2 * (raw["Sex"] == "M") + 0.003 * (raw["Age"] - 50)
    y = (rng.uniform(size=n) < base.clip(0, 1)).astype(int)
    p = (base + rng.normal(0, 0.15, n)).clip(0, 1)
    res = subgroup_report(raw, y, p)
    print(format_fairness_tables(res))
    print("\nfairness self-test OK")
