#!/usr/bin/env python3
"""A7: same-language incremental-DBSCAN vs DelauCluster local-update latency.

Runs an established incremental clustering baseline (incremental DBSCAN, Ester et
al. 1998, grid + union-find over cores) and DelauCluster on IDENTICAL insert-only
streams (matching the real-spatial insert-stream protocol). Reports per-update
latency for three update strategies on the same data:

  * DBSCAN full recompute   -- re-run DBSCAN from scratch each insert
  * DBSCAN incremental       -- local union-find update (field competitor)
  * DelauCluster local       -- incremental Delaunay-graph maintenance

This is a latency-only, field-relative comparison: the methods produce different
clusterings, so we compare update cost, not quality.
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


def resize_points(x, y, target, seed):
    if target <= 0 or target == len(x):
        return x.copy(), y.copy()
    rng = np.random.default_rng(seed)
    if target < len(x):
        idx = rng.choice(len(x), size=target, replace=False)
        return x[idx].copy(), y[idx].copy()
    extra = target - len(x)
    idx = rng.integers(0, len(x), size=extra)
    jitter = np.maximum(x.std(axis=0) * 0.01, 1e-6)
    ex = x[idx] + rng.normal(0.0, jitter, size=(extra, x.shape[1]))
    return np.vstack([x, ex]), np.r_[y, y[idx]]


def pctl(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--sizes", default="50000")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--ops", type=int, default=100)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/incremental_dbscan_comparison")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="a7_"))

    rows = []
    for ds in datasets:
        for n in sizes:
            full_ms, incr_ms, delau_ms = [], [], []
            idb_mismatch = 0
            for seed in seeds:
                x, y = build(ds, seed)
                x, y = resize_points(x, y, n, seed + 2000)
                rd = tmp / f"{ds}_{n}_{seed}"
                rd.mkdir(parents=True, exist_ok=True)
                base = rd / "base.csv"
                stream = rd / "stream.csv"
                write_points(base, x, y)
                lines = make_stream(x, y, args.ops, seed + 1000, 1.0, 0.0, 0.035)
                stream.write_text("\n".join(lines) + "\n")

                # DelauCluster local update path (frozen base-fit threshold).
                thr = fit_base_threshold(Path(args.cluster_exe), base, rd / "base_static")
                dmode = rd / "delau"
                subprocess.run(
                    [args.cluster_exe, "dynamic", str(base), str(stream), str(dmode),
                     "--threshold", f"{thr:.17g}", "--local-bucket-refresh"],
                    check=True, capture_output=True,
                )
                dlog = pd.read_csv(dmode / "dynamic_log.csv")
                dlog = dlog[dlog.operation == "insert"]
                delau_ms.extend((dlog.time_ns / 1e6).tolist())

                # Incremental DBSCAN on the identical stream.
                eps = float(estimate_eps(x, args.min_pts))
                io = rd / "idb.csv"
                subprocess.run(
                    [args.idbscan_exe, str(base), str(stream), str(io),
                     "--eps", f"{eps:.6f}", "--min-pts", str(args.min_pts)],
                    check=True, capture_output=True,
                )
                ilog = pd.read_csv(io)
                full_ms.extend((ilog.full_ns / 1e6).tolist())
                incr_ms.extend((ilog.incr_ns / 1e6).tolist())

            rows.append({
                "dataset": ds, "base_points": n, "ops": len(delau_ms),
                "dbscan_full_mean_ms": float(np.mean(full_ms)),
                "dbscan_full_p95_ms": pctl(full_ms, 95),
                "dbscan_incr_mean_ms": float(np.mean(incr_ms)),
                "dbscan_incr_p95_ms": pctl(incr_ms, 95),
                "delau_local_mean_ms": float(np.mean(delau_ms)),
                "delau_local_p95_ms": pctl(delau_ms, 95),
                "dbscan_incr_speedup_vs_full": float(np.mean(full_ms) / np.mean(incr_ms)),
                "delau_speedup_vs_dbscan_full": float(np.mean(full_ms) / np.mean(delau_ms)),
            })
            print(f"[done] {ds} n={n}")

    df = pd.DataFrame(rows)
    csv_path = out / "incremental_dbscan_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
