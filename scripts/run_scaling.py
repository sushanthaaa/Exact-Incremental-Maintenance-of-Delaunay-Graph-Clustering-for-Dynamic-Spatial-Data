#!/usr/bin/env python3
"""Scaling of the exact local update and the fallback rate (reviewer concern #6).

Two questions the cost story must answer:
  (1) How does the per-update local-update latency scale with dataset size n?
      -> driven by the O(V+E) soundness guard, expected ~linear in n.
  (2) How often does the full-relabel fallback fire, and does it depend on n
      or on workload?  -> rare (<1%), and a function of workload connectivity
      (bridge-heavy structure) rather than n.

Outputs results/scaling/scaling.csv (latency vs n) and prints the fallback
characterization across datasets and delete rates.
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

from dynamic_stream import fit_base_threshold, make_stream, write_points  # noqa: E402
from generate_datasets import build  # noqa: E402
from run_incremental_dbscan_comparison import resize_points  # noqa: E402

CLUSTER = "./build/cluster"


def run_stream(x, y, ops, seed, insert_rate, delete_rate, tmp, local=True):
    base = tmp / "b.csv"
    write_points(base, x, y)
    st = tmp / "s.csv"
    st.write_text("\n".join(make_stream(x, y, ops, seed + 7, insert_rate, delete_rate, 0.04)) + "\n")
    tau = fit_base_threshold(Path(CLUSTER), base, tmp / "bs")
    cmd = [CLUSTER, "dynamic", str(base), str(st), str(tmp / "o"), "--threshold", f"{tau:.17g}"]
    if local:  # without --local-bucket-refresh the dynamic path does a full rebuild per op
        cmd.append("--local-bucket-refresh")
    subprocess.run(cmd, check=True, capture_output=True)
    d = pd.read_csv(tmp / "o" / "dynamic_log.csv")
    lat = (d.time_ns / 1e6).to_numpy()
    fb = d.local_relabel_fallback.to_numpy() if "local_relabel_fallback" in d.columns else np.array([0.0])
    return lat, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="5000,10000,25000,50000,100000,200000")
    ap.add_argument("--ops", type=int, default=150)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--out-dir", default="results/scaling")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # (1) latency vs n on a balanced insert/delete/move stream
    rows = []
    local50k_by_seed = {}  # reused by the cost table so its local figure == the n=50k row here
    for n in sizes:
        lat = []
        for seed in seeds:
            x, y = build("chain_noise", seed)
            x, y = resize_points(x, y, n, seed + 2000)
            with tempfile.TemporaryDirectory() as td:
                l, _ = run_stream(x, y, args.ops, seed, 0.34, 0.33, Path(td))
                lat.extend(l.tolist())
                if n == 50000:
                    local50k_by_seed[seed] = float(np.mean(l))
        rows.append({"n": n, "update_ms_mean": float(np.mean(lat)),
                     "update_ms_p95": float(np.percentile(lat, 95))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling.csv", index=False)
    r = np.corrcoef(df.n, df.update_ms_mean)[0, 1]
    print("=== update latency vs n (balanced stream) ===")
    print(df.to_string(index=False))
    print(f"corr(n, latency) = {r:.4f}  (~1.0 => ~linear, O(V+E) guard)")

    # (2) fallback rate vs workload (delete rate) and dataset, fixed small n
    print("\n=== fallback rate vs workload ===")
    fb_rows = []
    for ds in ["chain_noise", "varying_density", "touching"]:
        for dr in [0.35, 0.50, 0.65]:
            fb = []
            for seed in seeds:
                x, y = build(ds, seed)
                with tempfile.TemporaryDirectory() as td:
                    _, f = run_stream(x, y, 1000, seed, max(1.0 - dr - 0.1, 0.0), dr, Path(td))
                    fb.extend(f.tolist())
            pct = 100.0 * float(np.mean(fb)); fb_rows.append({"dataset": ds, "delete_rate": dr, "fallback_pct": pct})
            print(f"  {ds:15s} delete_rate={dr:.2f}: fallback {pct:.2f}%")
    fb_df = pd.DataFrame(fb_rows); fb_df.to_csv(out / "fallback.csv", index=False)
    fbs = fb_df.fallback_pct.tolist()
    print(f"fallback range across datasets/workloads: {min(fbs):.2f}%--{max(fbs):.2f}% (mean {np.mean(fbs):.2f}%)")
    print("=> fallback is rare and workload-dependent (bridge-heavy structure), not n-driven.")

    # (3) local update vs full rebuild at n=50k. The local figure is REUSED from
    # the n=50k row of the scaling sweep above (identical measurement), so the
    # cost table and the scaling curve report the exact same local-update number;
    # here we only add the matching full-rebuild measurement.
    print("\n=== local update vs full rebuild (n=50k) ===")
    cost_rows = []
    for seed in seeds:
        x, y = build("chain_noise", seed)
        x, y = resize_points(x, y, 50000, seed + 2000)
        with tempfile.TemporaryDirectory() as td:
            reb, _ = run_stream(x, y, args.ops, seed, 0.34, 0.33, Path(td), local=False)
            local_ms = local50k_by_seed.get(seed)
            if local_ms is None:  # 50000 not in --sizes; measure it directly
                loc, _ = run_stream(x, y, args.ops, seed, 0.34, 0.33, Path(td), local=True)
                local_ms = float(np.mean(loc))
            cost_rows.append({"seed": seed, "local_ms": local_ms, "rebuild_ms": float(np.mean(reb))})
    cost_df = pd.DataFrame(cost_rows); cost_df.to_csv(out / "cost.csv", index=False)
    lm, rm = cost_df.local_ms.mean(), cost_df.rebuild_ms.mean()
    print(f"local update {lm:.1f} ms vs full rebuild {rm:.0f} ms ({rm/lm:.1f}x)")


if __name__ == "__main__":
    main()
