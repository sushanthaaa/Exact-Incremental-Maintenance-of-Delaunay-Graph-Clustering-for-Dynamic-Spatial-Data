#!/usr/bin/env python3
"""#4: exact incremental-DBSCAN as a strong dynamic head-to-head baseline.

The fast incremental-DBSCAN path (grow-only union-find) is cheap but inexact under
deletion. The *exact* incremental DBSCAN (Ester et al. 1998) deletes exactly by
rebuilding connectivity for the affected cluster, so it matches a from-scratch
DBSCAN (ARI 1.0). This script measures what that exactness costs on delete/move
streams, against DelauCluster's guarded exact update, in two regimes:

  * many small clusters (the synthetic benchmark datasets) -- the affected
    cluster is small, so exact deletion is cheap;
  * one large cluster (a single dense blob) -- the affected cluster is the whole
    dataset, so exact deletion is O(n) and blows up past a full recompute.

This is the "no cheap exact deletion" hardness (cf. Gan & Tao 2017) made concrete:
exact DBSCAN deletion is O(affected cluster), data-dependent and unbounded, while
DelauCluster's guarded update is uniform, parameter-free, and certified. All three
methods here are *exact*; the comparison is cost (and, in the table, the axes
DelauCluster wins: parameter-freedom and a per-update certificate).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import fit_base_threshold, make_stream, write_points  # noqa: E402
from generate_datasets import build  # noqa: E402
from run_incremental_dbscan_comparison import resize_points  # noqa: E402


def idb_delmove(idbscan_exe, base, stream, out, eps, min_pts, exact, every):
    cmd = [str(idbscan_exe), str(base), str(stream), str(out),
           "--eps", f"{eps:.6f}", "--min-pts", str(min_pts), "--every", str(every)]
    if exact:
        cmd.append("--exact")
    subprocess.run(cmd, check=True, capture_output=True)
    d = pd.read_csv(out)
    dm = d[d.operation.isin(["delete", "move"])]
    ari = d[d.incr_ari > -2].incr_ari
    return (float((dm.incr_ns / 1e6).mean()),
            float(np.percentile(dm.incr_ns / 1e6, 95)),
            float(ari.min()) if len(ari) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--ops", type=int, default=120)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--big-eps", type=float, default=0.15)
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/exact_dbscan_deletion_cost")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="exactdel_"))
    every = max(args.ops // 2, 1)  # confirm exactness a couple of times, cheaply

    rows = []
    # --- many-small-clusters regime: the synthetic benchmark datasets ---
    for ds in datasets:
        fast_mean, exact_mean, exact_p95, delau_mean, exact_minari = [], [], [], [], []
        for seed in seeds:
            x, y = build(ds, seed)
            x, y = resize_points(x, y, args.n, seed + 2000)
            rd = tmp / f"{ds}_{seed}"
            rd.mkdir(parents=True, exist_ok=True)
            base, stream = rd / "base.csv", rd / "stream.csv"
            write_points(base, x, y)
            stream.write_text("\n".join(
                make_stream(x, y, args.ops, seed + 1000, 0.30, 0.45, 0.04)) + "\n")
            eps = float(estimate_eps(x, args.min_pts))
            fm, _, _ = idb_delmove(args.idbscan_exe, base, stream, rd / "fast.csv",
                                   eps, args.min_pts, False, every)
            em, ep, ea = idb_delmove(args.idbscan_exe, base, stream, rd / "exact.csv",
                                     eps, args.min_pts, True, every)
            thr = fit_base_threshold(Path(args.cluster_exe), base, rd / "bs")
            subprocess.run([args.cluster_exe, "dynamic", str(base), str(stream),
                            str(rd / "delau"), "--threshold", f"{thr:.17g}",
                            "--local-bucket-refresh"], check=True, capture_output=True)
            dl = pd.read_csv(rd / "delau" / "dynamic_log.csv")
            dl = dl[dl.operation.isin(["delete", "move"])]
            fast_mean.append(fm); exact_mean.append(em); exact_p95.append(ep)
            exact_minari.append(ea); delau_mean.append(float((dl.time_ns / 1e6).mean()))
        rows.append({
            "regime": "small-clusters", "dataset": ds, "n": args.n,
            "fast_del_ms": np.mean(fast_mean),
            "exact_del_ms": np.mean(exact_mean), "exact_del_p95_ms": np.mean(exact_p95),
            "exact_min_ari": np.min(exact_minari),
            "delau_del_ms": np.mean(delau_mean),
        })
        print(f"[small] {ds}: exact={np.mean(exact_mean):.2f}ms "
              f"fast={np.mean(fast_mean):.3f}ms delau={np.mean(delau_mean):.2f}ms "
              f"exact_minARI={np.min(exact_minari):.4f}")

    # --- one-large-cluster regime: a single dense blob ---
    for seed in seeds:
        rng = np.random.default_rng(seed)
        P = np.c_[rng.normal(0, 1.0, size=args.n), rng.normal(0, 1.0, size=args.n)]
        lab = np.zeros(args.n, dtype=int)
        rd = tmp / f"blob_{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        base, stream = rd / "base.csv", rd / "stream.csv"
        write_points(base, P, lab)
        stream.write_text("\n".join(
            make_stream(P, lab, max(args.ops // 2, 40), seed + 1000, 0.30, 0.45, 0.03)) + "\n")
        em, ep, ea = idb_delmove(args.idbscan_exe, base, stream, rd / "exact.csv",
                                 args.big_eps, args.min_pts, True, every)
        fm, _, _ = idb_delmove(args.idbscan_exe, base, stream, rd / "fast.csv",
                               args.big_eps, args.min_pts, False, every)
        rows.append({"regime": "one-large-cluster", "dataset": "dense_blob", "n": args.n,
                     "fast_del_ms": fm, "exact_del_ms": em, "exact_del_p95_ms": ep,
                     "exact_min_ari": ea, "delau_del_ms": float("nan")})
        print(f"[large] dense_blob seed={seed}: exact={em:.1f}ms p95={ep:.1f}ms "
              f"fast={fm:.3f}ms exact_minARI={ea:.4f}")

    df = pd.DataFrame(rows)
    csv_path = out / "exact_dbscan_deletion_cost.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
