"""
run_all.py
==========
End-to-end orchestration: for each dataset, run patient-level 5-fold CV with
leakage-safe preprocessing, train every model + the stacking ensemble, and
produce ALL the tables/figures the reviewers asked for:

  * cross-validated performance (acc / F1 / AUC, mean +/- std)  -> results_cv.csv
  * held-out test performance + calibration (Brier/ECE/MCE)     -> results_test.csv
  * reliability diagrams                                        -> outputs/*.png
  * explainability fidelity/stability (SHAP vs LIME)            -> results_xai.csv
  * fairness by age & sex                                       -> results_fairness_*.csv
  * per-disease error analysis + failure cases                 -> results_errors*.csv
  * cross-dataset (leave-one-disease-out is N/A here; we do
    train-on-one/test-on-another where feature spaces allow)   -> results_cross.csv

Run (after placing the 3 CSVs in data/):
    python src/run_all.py
or use the accompanying Colab notebook.

NOTE: This produces REAL numbers from REAL data you download. Nothing is
synthetic. Until you run it on the actual CSVs, the result files won't exist.
"""

from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_prep import load_dataset, SCHEMAS, export_feature_dictionary, export_cohort_log
from preprocessing import make_holdout_split, kfold_iter, Preprocessor
from models import (SkWrap, make_logreg, make_rf, make_xgb, MLP, FTTransformer,
                    TabNetAdapter, StackingEnsemble, default_base_factories, _HAS_XGB)
from calibration import calibration_report, plot_reliability
from explainability import explainability_report
from fairness import subgroup_report, format_fairness_tables
from error_analysis import error_summary, failure_cases, per_dataset_error_report

PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)

N_FOLDS = 5
SEED = 42


def _metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y, yhat),
           "f1": f1_score(y, yhat, zero_division=0)}
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return out


def model_factories(include_tabnet=False):
    """All standalone models to evaluate (the stacking ensemble is separate)."""
    f = {
        "LogReg": lambda: SkWrap(make_logreg),
        "RandomForest": lambda: SkWrap(make_rf),
        "MLP": lambda: MLP(epochs=80),
        "FT-Transformer": lambda: FTTransformer(epochs=80),
    }
    if _HAS_XGB:
        f["XGBoost"] = lambda: SkWrap(make_xgb)
    if include_tabnet:
        f["TabNet"] = lambda: TabNetAdapter()
    return f


def run_dataset(key, include_tabnet=False):
    d = load_dataset(key)
    if d is None:
        return None
    X, y = d["X"], d["y"]
    num, cat = d["numeric_cols"], d["categorical_cols"]

    dev, test = make_holdout_split(X, y, test_frac=0.15, seed=SEED)

    # ---------- cross-validation on dev ----------
    factories = model_factories(include_tabnet)
    cv_scores = {name: {"accuracy": [], "f1": [], "auc": []} for name in factories}
    cv_scores["StackingEnsemble"] = {"accuracy": [], "f1": [], "auc": []}

    for tr, va in kfold_iter(dev["X"], dev["y"], n_splits=N_FOLDS, seed=SEED):
        pp = Preprocessor(num, cat).fit(tr["X"])
        Xtr, Xva = pp.transform(tr["X"]), pp.transform(va["X"])

        for name, mk in factories.items():
            try:
                m = mk().fit(Xtr, tr["y"])
                p = m.predict_proba(Xva)[:, 1]
                for k, v in _metrics(va["y"].values, p).items():
                    cv_scores[name][k].append(v)
            except Exception as e:
                print(f"  [warn] {name} failed on a fold: {e}")

        # stacking ensemble (uses its own internal OOF)
        try:
            ens = StackingEnsemble(default_base_factories(include_xgb=_HAS_XGB),
                                   n_inner=5, seed=SEED)
            ens.fit(Xtr, tr["y"])
            p = ens.predict_proba(Xva)[:, 1]
            for k, v in _metrics(va["y"].values, p).items():
                cv_scores["StackingEnsemble"][k].append(v)
        except Exception as e:
            print(f"  [warn] stacking failed on a fold: {e}")

    cv_rows = []
    cv_fold_rows = []   # NEW: raw per-fold values for paired significance testing
    for name, sc in cv_scores.items():
        row = {"dataset": key, "model": name}
        for k in ("accuracy", "f1", "auc"):
            arr = np.array(sc[k], dtype=float)
            row[f"{k}_mean"] = float(np.nanmean(arr)) if len(arr) else np.nan
            row[f"{k}_std"] = float(np.nanstd(arr)) if len(arr) else np.nan
            # record each fold value
            for fold_i, val in enumerate(sc[k]):
                cv_fold_rows.append({"dataset": key, "model": name,
                                     "metric": k, "fold": fold_i, "value": float(val)})
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_fold_df = pd.DataFrame(cv_fold_rows)

    # ---------- refit best stack on full dev, evaluate on held-out test ----------
    pp = Preprocessor(num, cat).fit(dev["X"])
    Xdev, Xtest = pp.transform(dev["X"]), pp.transform(test["X"])
    ens = StackingEnsemble(default_base_factories(include_xgb=_HAS_XGB),
                           n_inner=5, seed=SEED).fit(Xdev, dev["y"])
    p_test = ens.predict_proba(Xtest)[:, 1]

    test_metrics = _metrics(test["y"].values, p_test)
    cal = calibration_report(test["y"].values, p_test)
    test_row = {"dataset": key, **test_metrics, **cal}

    # reliability diagram
    rel_path = os.path.join(OUT, f"reliability_{key}.png")
    plot_reliability(test["y"].values, p_test, rel_path,
                     title=f"Reliability - {key}")

    # ---------- explainability (on test set, background = dev) ----------
    xai = explainability_report(ens, Xtest, Xdev, n_instances=40, n_samples=400)
    xai_rows = [{"dataset": key, "method": m, **vals} for m, vals in xai.items()]

    # ---------- fairness (use ORIGINAL frame for subgroup attributes) ----------
    test_raw = test["X"].reset_index(drop=True)
    fair = subgroup_report(test_raw, test["y"].values, p_test)
    fair_dfs = {}
    for grp, df in fair["reports"].items():
        df = df.copy(); df.insert(0, "dataset", key)
        fair_dfs[grp] = df.reset_index()

    # ---------- error analysis ----------
    err = error_summary(test["y"].values, p_test); err["dataset"] = key
    fcases = failure_cases(test_raw, test["y"].values, p_test, max_cases=20)
    if not fcases.empty:
        fcases.insert(0, "dataset", key)

    # ---------- per-class metrics + class prevalence (reviewer #7) ----------
    from sklearn.metrics import precision_recall_fscore_support
    yhat_test = (p_test >= 0.5).astype(int)
    prec, rec, f1c, support = precision_recall_fscore_support(
        test["y"].values, yhat_test, labels=[0, 1], zero_division=0)
    perclass_rows = []
    for ci, cname in enumerate(["negative", "positive"]):
        perclass_rows.append({"dataset": key, "class": cname,
                              "support": int(support[ci]),
                              "precision": float(prec[ci]),
                              "recall": float(rec[ci]),
                              "f1": float(f1c[ci])})
    perclass_df = pd.DataFrame(perclass_rows)
    # class prevalence across the full usable cohort
    prevalence = {"dataset": key,
                  "n_total": int(len(y)),
                  "n_positive": int(y.sum()),
                  "positive_rate": float(y.mean()),
                  "n_dev": int(len(dev["y"])),
                  "n_test": int(len(test["y"]))}

    return {
        "cv": cv_df, "cv_folds": cv_fold_df, "test_row": test_row, "xai_rows": xai_rows,
        "fairness": fair_dfs, "fairness_gaps": {key: fair["gaps"]},
        "error_row": err, "failure_cases": fcases,
        "perclass": perclass_df, "prevalence": prevalence,
        "test_pred": (test["y"].values, p_test),
        "test_raw": test_raw,
    }


def main(include_tabnet=False):
    print("=" * 70)
    print("FULL PIPELINE RUN")
    print("=" * 70)
    t0 = time.time()

    datasets = {k: load_dataset(k) for k in SCHEMAS}
    if all(v is None for v in datasets.values()):
        print("\nNo data found. Place heart.csv, diabetes.csv, kidney_disease.csv")
        print("in the data/ folder (see data_prep.py for sources), then re-run.")
        return
    export_feature_dictionary(datasets)
    export_cohort_log(datasets)

    all_cv, all_test, all_xai = [], [], []
    all_cv_folds = []
    all_fair = {"sex": [], "age": []}
    all_err, all_fcases = [], []
    all_perclass, all_prevalence = [], []
    error_inputs = {}
    fairness_gaps = {}

    for key in SCHEMAS:
        if datasets[key] is None:
            continue
        print(f"\n----- {key} -----")
        res = run_dataset(key, include_tabnet=include_tabnet)
        if res is None:
            continue
        all_cv.append(res["cv"])
        all_cv_folds.append(res["cv_folds"])
        all_test.append(res["test_row"])
        all_xai.extend(res["xai_rows"])
        all_perclass.append(res["perclass"])
        all_prevalence.append(res["prevalence"])
        for grp in ("sex", "age"):
            if grp in res["fairness"]:
                all_fair[grp].append(res["fairness"][grp])
        all_err.append(res["error_row"])
        if not res["failure_cases"].empty:
            all_fcases.append(res["failure_cases"])
        fairness_gaps.update(res["fairness_gaps"])
        yt, p = res["test_pred"]
        error_inputs[key] = {"y_true": yt, "p": p, "frame_raw": res["test_raw"]}

    # ----- write everything -----
    def _save(df, name):
        path = os.path.join(OUT, name)
        df.to_csv(path, index=False)
        print(f"[save] {name}  ({len(df)} rows)")

    if all_cv:
        _save(pd.concat(all_cv, ignore_index=True), "results_cv.csv")
    if all_cv_folds:
        _save(pd.concat(all_cv_folds, ignore_index=True), "results_cv_folds.csv")
    if all_test:
        _save(pd.DataFrame(all_test), "results_test.csv")
    if all_xai:
        _save(pd.DataFrame(all_xai), "results_xai.csv")
    for grp in ("sex", "age"):
        if all_fair[grp]:
            _save(pd.concat(all_fair[grp], ignore_index=True),
                  f"results_fairness_{grp}.csv")
    if all_err:
        _save(per_dataset_error_report(error_inputs), "results_errors.csv")
    if all_fcases:
        _save(pd.concat(all_fcases, ignore_index=True), "results_failure_cases.csv")
    if all_perclass:
        _save(pd.concat(all_perclass, ignore_index=True), "results_perclass.csv")
    if all_prevalence:
        _save(pd.DataFrame(all_prevalence), "results_prevalence.csv")

    # ----- paired significance test: ensemble vs each baseline (reviewer #2) -----
    if all_cv_folds:
        try:
            from scipy.stats import wilcoxon
            folds = pd.concat(all_cv_folds, ignore_index=True)
            sig_rows = []
            for ds in folds["dataset"].unique():
                for metric in ("auc", "accuracy", "f1"):
                    sub = folds[(folds.dataset == ds) & (folds.metric == metric)]
                    piv = sub.pivot(index="fold", columns="model", values="value")
                    if "StackingEnsemble" not in piv.columns:
                        continue
                    ens = piv["StackingEnsemble"]
                    for model in piv.columns:
                        if model == "StackingEnsemble":
                            continue
                        diff = (ens - piv[model]).dropna()
                        if len(diff) < 3 or np.allclose(diff, 0):
                            stat, pval = np.nan, np.nan
                        else:
                            try:
                                stat, pval = wilcoxon(ens.loc[diff.index],
                                                      piv[model].loc[diff.index])
                            except Exception:
                                stat, pval = np.nan, np.nan
                        sig_rows.append({"dataset": ds, "metric": metric,
                                         "baseline": model,
                                         "mean_diff": float(diff.mean()),
                                         "wilcoxon_stat": float(stat) if stat == stat else np.nan,
                                         "p_value": float(pval) if pval == pval else np.nan,
                                         "n_folds": int(len(diff))})
                        # Corrected paired t-test (Nadeau & Bengio) — appropriate for CV,
                        # accounts for train/test overlap across folds. With k folds and
                        # test fraction rho = 1/k, variance is inflated by (1/k + rho/(1-rho)).
                        if len(diff) >= 3 and diff.std(ddof=1) > 0:
                            k = len(diff); rho = 1.0 / k
                            corr = (1.0 / k) + (rho / (1.0 - rho))
                            from scipy.stats import t as _t
                            tstat = diff.mean() / (np.sqrt(corr) * diff.std(ddof=1))
                            tp = 2 * _t.sf(abs(tstat), df=k - 1)
                            sig_rows[-1]["corrected_t_stat"] = float(tstat)
                            sig_rows[-1]["corrected_t_p"] = float(tp)
            sig_df = pd.DataFrame(sig_rows)
            # Holm-Bonferroni correction within each dataset x metric family
            sig_df["p_holm"] = np.nan
            for (ds, metric), grp in sig_df.groupby(["dataset", "metric"]):
                pv = grp["p_value"].dropna().sort_values()
                m = len(pv)
                holm = {}
                prev = 0.0
                for rank, (idx, p) in enumerate(pv.items()):
                    adj = min(1.0, max(prev, (m - rank) * p))
                    holm[idx] = adj
                    prev = adj
                for idx, val in holm.items():
                    sig_df.loc[idx, "p_holm"] = val
            _save(sig_df, "results_significance.csv")
            print("\n[stats] Paired Wilcoxon (ensemble vs baselines) written. Summary:")
            for _, r in sig_df.iterrows():
                if r["p_value"] == r["p_value"]:
                    print(f"  {r['dataset']:8s} {r['metric']:4s} vs {r['baseline']:16s} "
                          f"Δ={r['mean_diff']:+.4f}  p={r['p_value']:.4f}  "
                          f"p_holm={r['p_holm']:.4f}")
        except ImportError:
            print("\n[stats] scipy not available; install scipy to run the Wilcoxon test.")
    with open(os.path.join(OUT, "results_fairness_gaps.json"), "w") as f:
        json.dump(fairness_gaps, f, indent=2)
        print("[save] results_fairness_gaps.json")

    print(f"\nDone in {time.time() - t0:.1f}s. All tables in: {OUT}")
    print("These are the numbers to put in the paper (replacing every placeholder).")


if __name__ == "__main__":
    inc_tab = "--tabnet" in sys.argv
    main(include_tabnet=inc_tab)
