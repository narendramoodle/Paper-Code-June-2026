"""
error_analysis.py
=================
Per-disease error analysis — answers the reviewer's error-analysis requirement.

For each dataset/disease it reports:
  * confusion matrix (TN, FP, FN, TP)
  * false-positive rate, false-negative rate
  * the specific misclassified instances (failure cases) with their predicted
    probability and original feature values, so the discussion can describe
    *what kind* of patients the model gets wrong.

Outputs are returned as DataFrames and can be written to CSV for the appendix.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def error_summary(y_true, p, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "n": int(n), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "accuracy": float((tp + tn) / n) if n else np.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
    }


def failure_cases(frame_raw: pd.DataFrame, y_true, p, threshold=0.5,
                  max_cases=20) -> pd.DataFrame:
    """
    Return the misclassified rows with their true label, predicted prob, error
    type (FP/FN), and original features. Sorted by how confident-and-wrong they
    are (largest |p - y| first), since those are the most informative failures.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    y_pred = (p >= threshold).astype(int)
    wrong = np.where(y_pred != y_true)[0]
    if len(wrong) == 0:
        return pd.DataFrame()
    conf_wrong = np.abs(p[wrong] - y_true[wrong])
    order = wrong[np.argsort(-conf_wrong)][:max_cases]

    rows = frame_raw.iloc[order].copy().reset_index(drop=True)
    rows.insert(0, "error_type",
                np.where(y_pred[order] > y_true[order], "FP", "FN"))
    rows.insert(1, "true_label", y_true[order])
    rows.insert(2, "pred_prob", np.round(p[order], 3))
    return rows


def per_dataset_error_report(results_by_dataset: dict, threshold=0.5) -> pd.DataFrame:
    """
    results_by_dataset: {name: {'y_true':..., 'p':..., 'frame_raw':df}}
    Returns a tidy summary table across datasets.
    """
    rows = []
    for name, r in results_by_dataset.items():
        s = error_summary(r["y_true"], r["p"], threshold)
        s["dataset"] = name
        rows.append(s)
    cols = ["dataset", "n", "TN", "FP", "FN", "TP", "false_positive_rate",
            "false_negative_rate", "precision", "recall", "accuracy"]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    raw = pd.DataFrame({"Age": rng.integers(25, 85, n),
                        "Chol": rng.integers(120, 320, n),
                        "Sex": rng.choice(["M", "F"], n)})
    y = rng.integers(0, 2, n)
    # predictions correlated with y but noisy
    p = (0.5 * y + 0.5 * rng.uniform(size=n)).clip(0, 1)
    print(error_summary(y, p))
    fc = failure_cases(raw, y, p, max_cases=5)
    print("\nTop failure cases:")
    print(fc.to_string(index=False))
    summary = per_dataset_error_report({
        "heart": {"y_true": y, "p": p, "frame_raw": raw},
        "diabetes": {"y_true": y[:150], "p": p[:150], "frame_raw": raw.iloc[:150]},
    })
    print("\nPer-dataset summary:")
    print(summary.round(3).to_string(index=False))
    print("\nerror_analysis self-test OK")
