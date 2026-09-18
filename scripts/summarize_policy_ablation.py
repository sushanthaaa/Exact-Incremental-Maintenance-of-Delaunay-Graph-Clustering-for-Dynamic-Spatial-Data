#!/usr/bin/env python3
"""Policy-family / selector ablation — A1 of the full-paper plan.

Aggregates the ALREADY-LOGGED per-mode artifacts of the canonical static study
(results/static_5seed_summary/runs/<ds>/n_*/seed_*/DelauClusterAuto/<mode>/):
per-mode ARI/NMI vs the ground-truth labels in the run's points.csv, plus the
fixed-policy DelauCluster row and the Auto row from the summary CSV, and the
histogram of which mode the label-free objective selected. NO new fits: this
reports evidence that already exists.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MODES = ["raw_length", "local_scale", "knn_median", "supported_merge",
         "saddle_cut", "manifold_filter"]


def labels_from_clusters_csv(p):
    df = pd.read_csv(p)
    lab = df.cluster_id.to_numpy().copy()
    lab[df.is_noise.to_numpy().astype(bool)] = -1
    return df, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/static_5seed_summary/runs")
    ap.add_argument("--summary-csv",
                    default="results/static_5seed_summary/baseline_comparison_summary.csv")
    ap.add_argument("--out-dir", default="results/ablation_policy")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_dir in sorted(runs.iterdir()):
        if not ds_dir.is_dir():
            continue
        for seed_dir in sorted(ds_dir.glob("n_*/seed_*")):
            seed = int(re.search(r"seed_(\d+)", seed_dir.name).group(1))
            pts = pd.read_csv(seed_dir / "points.csv", header=None,
                              names=["x", "y", "label"])
            true = pts.label.to_numpy()
            for mode in MODES:
                cpath = seed_dir / "DelauClusterAuto" / mode / "clusters.csv"
                if not cpath.exists():
                    rows.append({"dataset": ds_dir.name, "seed": seed,
                                 "mode": mode, "ari": np.nan, "nmi": np.nan,
                                 "status": "missing"})
                    continue
                cdf, lab = labels_from_clusters_csv(cpath)
                if len(cdf) != len(pts) or not (
                        np.allclose(cdf.x.to_numpy(), pts.x.to_numpy()) and
                        np.allclose(cdf.y.to_numpy(), pts.y.to_numpy())):
                    raise SystemExit(f"point mismatch: {cpath}")
                rows.append({"dataset": ds_dir.name, "seed": seed, "mode": mode,
                             "ari": adjusted_rand_score(true, lab),
                             "nmi": normalized_mutual_info_score(true, lab),
                             "status": "ok"})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "policy_ablation_per_run.csv", index=False)

    ok = per_run[per_run.status == "ok"]
    per_mode = ok.groupby("mode").agg(
        runs=("ari", "size"), ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean")).reindex(MODES)
    per_mode.to_csv(out / "policy_ablation_per_mode.csv")

    per_ds = ok.pivot_table(index="dataset", columns="mode", values="ari",
                            aggfunc="mean").reindex(columns=MODES)
    per_ds.to_csv(out / "policy_ablation_per_dataset.csv")

    summ = pd.read_csv(args.summary_csv)
    auto = summ[(summ.algorithm == "DelauClusterAuto") & (summ.status == "ok")]
    fixed = summ[(summ.algorithm == "DelauCluster") & (summ.status == "ok")]
    sel = auto.params.str.extract(r"selected_mode=([a-z_]+)")[0]
    sel_hist = sel.value_counts().reindex(MODES).fillna(0).astype(int)
    sel_hist.to_csv(out / "selected_mode_histogram.csv", header=["count"])

    with open(out / "policy_ablation_report.md", "w") as f:
        f.write("# Policy-family ablation (from canonical static_5seed_summary)\n\n")
        f.write(f"Auto (label-free selector): ARI {auto.ari.mean():.3f} "
                f"± {auto.ari.std():.3f} (n={len(auto)})\n\n")
        f.write(f"Fixed policy (local_scale): ARI {fixed.ari.mean():.3f} "
                f"± {fixed.ari.std():.3f} (n={len(fixed)})\n\n")
        f.write("## Per-mode means over the same 40 runs\n\n")
        f.write(per_mode.to_string() + "\n\n")
        f.write("## Per-dataset per-mode ARI\n\n")
        f.write(per_ds.round(3).to_string() + "\n\n")
        f.write("## Selected-mode histogram (Auto)\n\n")
        f.write(sel_hist.to_string() + "\n")
    print(per_mode)
    print(f"\nAuto {auto.ari.mean():.3f} vs fixed {fixed.ari.mean():.3f}; "
          f"selector picks: {dict(sel_hist)}")
    print(f"wrote {out}/policy_ablation_report.md")


if __name__ == "__main__":
    main()
