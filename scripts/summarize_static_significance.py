#!/usr/bin/env python3
"""Statistical treatment of the static table (Table IV) — A2 of the plan.

Paired Wilcoxon signed-rank tests over the 40 paired per-run ARIs
(DelauClusterAuto vs each oracle-tuned baseline), Holm-corrected across the
comparisons, plus a seeded bootstrap 95% CI of the mean paired difference.
Reported honestly whichever way they land: non-significance against the
DBSCAN grid optimum SUPPORTS the calibrated "competitive" claim; it is not
evidence of equivalence, and the paper must not claim equivalence.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINES = ["DBSCAN", "HDBSCAN", "OPTICS", "Spectral"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv",
                    default="results/static_5seed_summary/baseline_comparison_summary.csv")
    ap.add_argument("--out-dir", default="results/static_significance")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    df = df[df.status == "ok"]
    piv = df.pivot_table(index=["dataset", "seed"], columns="algorithm",
                         values="ari")
    auto = piv["DelauClusterAuto"]
    rng = np.random.default_rng(args.seed)

    rows = []
    for b in BASELINES:
        pair = piv[[b]].join(auto.rename("auto")).dropna()
        d = (pair["auto"] - pair[b]).to_numpy()
        n = len(d)
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                           method="auto")
        boots = np.array([rng.choice(d, size=n, replace=True).mean()
                          for _ in range(args.bootstrap)])
        rows.append({"baseline": b, "n_pairs": n,
                     "mean_diff_auto_minus_baseline": d.mean(),
                     "median_diff": float(np.median(d)),
                     "wilcoxon_stat": stat, "p_raw": p,
                     "ci95_lo": float(np.quantile(boots, 0.025)),
                     "ci95_hi": float(np.quantile(boots, 0.975))})
    res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(res)
    # Holm step-down on ascending p: adjust by (m - rank), enforce monotone
    # non-decreasing adjusted p, cap at 1.
    res["p_holm"] = np.minimum(
        1.0, np.maximum.accumulate(res.p_raw.to_numpy() * (m - np.arange(m))))
    res["significant_at_0.05"] = res.p_holm < 0.05
    res.to_csv(out / "static_significance.csv", index=False)
    print(res.to_string(index=False))

    # --- Grouping-aware reanalysis (external review, Issue 9): seeds within a
    # dataset family are not independent. (a) aggregate seeds -> paired
    # Wilcoxon across the n=8 families; (b) hierarchical bootstrap CI
    # (resample families with replacement, then seeds within each).
    fam = piv.groupby(level="dataset").mean()
    rows2 = []
    for b in BASELINES:
        d8 = (fam["DelauClusterAuto"] - fam[b]).dropna().to_numpy()
        stat8, p8 = wilcoxon(d8, zero_method="wilcox", alternative="two-sided",
                             method="exact")
        per_seed = piv["DelauClusterAuto"] - piv[b]  # indexed (dataset, seed)
        by_fam = {f: g.to_numpy() for f, g in per_seed.groupby(level="dataset")}
        fams = list(by_fam)
        boots = np.empty(args.bootstrap)
        for i in range(args.bootstrap):
            fs = rng.choice(len(fams), size=len(fams), replace=True)
            vals = [rng.choice(by_fam[fams[j]], size=len(by_fam[fams[j]]),
                               replace=True).mean() for j in fs]
            boots[i] = float(np.mean(vals))
        rows2.append({"baseline": b, "n_families": len(d8),
                      "mean_diff_family_level": float(d8.mean()),
                      "wilcoxon_stat_n8": stat8, "p_raw_n8": p8,
                      "hier_ci95_lo": float(np.quantile(boots, 0.025)),
                      "hier_ci95_hi": float(np.quantile(boots, 0.975))})
    res2 = pd.DataFrame(rows2).sort_values("p_raw_n8").reset_index(drop=True)
    m2 = len(res2)
    res2["p_holm_n8"] = np.minimum(
        1.0, np.maximum.accumulate(res2.p_raw_n8.to_numpy() * (m2 - np.arange(m2))))
    res2.to_csv(out / "static_significance_family_level.csv", index=False)
    fam_diffs = pd.DataFrame({b: fam["DelauClusterAuto"] - fam[b]
                              for b in BASELINES}).round(4)
    fam_diffs.to_csv(out / "per_family_differences.csv")
    print("\n--- family-level (n=8) + hierarchical bootstrap ---")
    print(res2.to_string(index=False))
    print(f"wrote {out}/static_significance_family_level.csv and per_family_differences.csv")


if __name__ == "__main__":
    main()
