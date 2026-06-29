# =====================================================================
# STATS RE-RUN CELL  —  run this in Colab after src/ and data/ are set up
# (heart.csv, diabetes.csv, kidney_disease.csv in data/; the 8 .py files in src/)
#
# It runs the full pipeline (now with per-fold scores, significance tests,
# class prevalence, and per-class metrics) and then PRINTS, in a copy-paste
# friendly form, exactly the numbers you need to fill into the paper:
#   - Table 1   : class prevalence (positive rate) per dataset
#   - Section 5.1 : paired significance test (ensemble vs each baseline)
#   - Table 7   : per-class precision / recall / F1  (sanity check vs paper)
# =====================================================================

import sys, os
sys.path.insert(0, os.path.abspath('src'))

# make sure scipy is present for the significance tests
try:
    import scipy  # noqa
except ImportError:
    !pip -q install scipy
    import scipy  # noqa

import importlib
import run_all
importlib.reload(run_all)

# --- run the full pipeline (set include_tabnet=False to skip the slow TabNet) ---
run_all.main(include_tabnet=True)

# ---------------------------------------------------------------------
# Now print the exact values to paste into the manuscript
# ---------------------------------------------------------------------
import pandas as pd
OUT = 'outputs'

print("\n" + "=" * 70)
print("VALUES TO FILL INTO THE PAPER")
print("=" * 70)

# ---- Table 1: class prevalence ----
prev_path = os.path.join(OUT, 'results_prevalence.csv')
if os.path.exists(prev_path):
    prev = pd.read_csv(prev_path)
    print("\n--- Table 1: positive rate (class prevalence) ---")
    for _, r in prev.iterrows():
        print(f"  {r['dataset']:10s}  positive_rate = {r['positive_rate']:.3f}  "
              f"(usable n={int(r['n_total'])}, dev={int(r['n_dev'])}, test={int(r['n_test'])})")

# ---- Section 5.1: significance tests ----
sig_path = os.path.join(OUT, 'results_significance.csv')
if os.path.exists(sig_path):
    sig = pd.read_csv(sig_path)
    auc = sig[sig.metric == 'auc'].copy()
    print("\n--- Section 5.1: ensemble vs baselines (AUC, per-fold paired tests) ---")
    print("    (Wilcoxon floor with 5 folds is p=0.0625; rely on corrected t-test)")
    for ds in auc['dataset'].unique():
        print(f"  [{ds}]")
        sub = auc[auc.dataset == ds].sort_values('mean_diff', ascending=False)
        for _, r in sub.iterrows():
            ct = r.get('corrected_t_p', float('nan'))
            ph = r.get('p_holm', float('nan'))
            print(f"    vs {r['baseline']:16s}  ΔAUC={r['mean_diff']:+.4f}  "
                  f"Wilcoxon p={r['p_value']:.4f} (Holm {ph:.4f})  "
                  f"corrected-t p={ct:.4f}")

# ---- Table 7 sanity check: per-class metrics ----
pc_path = os.path.join(OUT, 'results_perclass.csv')
if os.path.exists(pc_path):
    pc = pd.read_csv(pc_path)
    print("\n--- Table 7: per-class metrics (verify these match the paper) ---")
    print(pc.round(3).to_string(index=False))

print("\n" + "=" * 70)
print("Paste the prevalence values into Table 1, and the significance")
print("p-values into the Section 5.1 paragraph and a supplementary table.")
print("If any number here differs from the paper, the RUN wins — update the paper.")
print("=" * 70)
