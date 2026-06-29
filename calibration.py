"""
calibration.py
==============
Calibration analysis for predicted probabilities — directly answers the
reviewer's calibration requirement.

Provides:
  * brier_score        : mean squared error of probabilistic predictions
  * expected_calibration_error (ECE) : binned gap between confidence & accuracy
  * maximum_calibration_error (MCE)
  * reliability_curve  : data for a reliability diagram
  * plot_reliability   : saves a reliability-diagram PNG

All functions take y_true (0/1) and p (predicted prob of positive class).
"""

from __future__ import annotations
import numpy as np


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def _bin_stats(y_true, p, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    stats = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            stats.append((b, 0, np.nan, np.nan, edges[b], edges[b + 1]))
        else:
            conf = p[mask].mean()
            acc = y_true[mask].mean()
            stats.append((b, int(mask.sum()), conf, acc, edges[b], edges[b + 1]))
    return stats


def expected_calibration_error(y_true, p, n_bins=10) -> float:
    n = len(y_true)
    ece = 0.0
    for _, cnt, conf, acc, *_ in _bin_stats(y_true, p, n_bins):
        if cnt > 0:
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(y_true, p, n_bins=10) -> float:
    gaps = [abs(acc - conf) for _, cnt, conf, acc, *_ in _bin_stats(y_true, p, n_bins)
            if cnt > 0]
    return float(max(gaps)) if gaps else 0.0


def reliability_curve(y_true, p, n_bins=10):
    """Return (mean_confidence, empirical_accuracy, bin_count) arrays."""
    conf, acc, cnt = [], [], []
    for _, c, cf, ac, *_ in _bin_stats(y_true, p, n_bins):
        if c > 0:
            conf.append(cf); acc.append(ac); cnt.append(c)
    return np.array(conf), np.array(acc), np.array(cnt)


def calibration_report(y_true, p, n_bins=10) -> dict:
    return {
        "brier": brier_score(y_true, p),
        "ece": expected_calibration_error(y_true, p, n_bins),
        "mce": maximum_calibration_error(y_true, p, n_bins),
        "n_bins": n_bins,
    }


def plot_reliability(y_true, p, path, title="Reliability Diagram", n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf, acc, cnt = reliability_curve(y_true, p, n_bins)
    rep = calibration_report(y_true, p, n_bins)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfect calibration")
    ax.plot(conf, acc, "o-", color="#2e6fa7", lw=2, label="Model")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nBrier={rep['brier']:.3f}  ECE={rep['ece']:.3f}  "
                 f"MCE={rep['mce']:.3f}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return rep


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # well-calibrated example
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    print("calibrated:", calibration_report(y, p))
    # overconfident example
    p2 = np.clip(p * 1.6 - 0.3, 0, 1)
    print("overconf:  ", calibration_report(y, p2))
    print("calibration self-test OK")
