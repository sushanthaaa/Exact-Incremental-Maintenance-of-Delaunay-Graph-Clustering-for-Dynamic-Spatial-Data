#!/usr/bin/env python3
"""Experiment 2/3 (application): exact maintenance under REAL data corrections.

Real spatial catalogs are continually revised: reports are withdrawn/de-duplicated,
geocodes are corrected, earthquake picks are relocated or removed. These are
arbitrary delete + move operations -- exactly the regime the fast incremental-DBSCAN
union-find path cannot maintain exactly.

We replay a correction stream over real projected catalogs (NYC motor-vehicle
collisions; USGS earthquakes) in temporal order:
  * insert  = a new incoming event (next record in the file),
  * delete  = a withdrawn / duplicate / erroneous report,
  * move    = a geocode / relocation correction (small positional fix).
For each method we measure per-step faithfulness (ARI vs a from-scratch run of the
same method on the currently-active points). DelauCluster's verifier confirms 0
mismatches (exact); the fast incremental-DBSCAN baseline drifts.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import write_points  # noqa: E402
from run_dynamic_faithfulness import run_baseline, run_delaucluster_verified  # noqa: E402


def load_xy(path: Path) -> np.ndarray:
    raw = pd.read_csv(path)
    cols = [c for c in raw.columns if str(c).lower() in ("x", "y")]
    if len(cols) >= 2:
        arr = raw[["x", "y"]].to_numpy(dtype=float)
    else:
        arr = pd.read_csv(path, header=None).iloc[:, :2].to_numpy(dtype=float)
    return arr[np.isfinite(arr).all(axis=1)]


def make_correction_stream(base_pts, pool_pts, nops, seed,
                           insert_rate, delete_rate, move_frac_of_std):
    rng = np.random.default_rng(seed)
    active = [tuple(p) for p in base_pts]
    pool = [tuple(p) for p in pool_pts]
    pi = 0
    jit = float(np.std(base_pts, axis=0).mean()) * move_frac_of_std
    lines = []
    for _ in range(nops):
        u = rng.random()
        if (u < insert_rate or len(active) < 10) and pi < len(pool):
            p = pool[pi]; pi += 1
            lines.append(f"insert,{p[0]:.6f},{p[1]:.6f},0")
            active.append(p)
        elif u < insert_rate + delete_rate:
            j = rng.integers(0, len(active))
            p = active.pop(j)  # withdrawn / duplicate / erroneous report
            lines.append(f"delete,{p[0]:.6f},{p[1]:.6f}")
        else:
            j = rng.integers(0, len(active))
            ox, oy = active[j]               # geocode / relocation correction
            nx = ox + float(rng.normal(0, jit))
            ny = oy + float(rng.normal(0, jit))
            active[j] = (nx, ny)
            lines.append(f"move,{ox:.6f},{oy:.6f},{nx:.6f},{ny:.6f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=(
        "nyc=data/real_spatial_large/nyc_collisions_2012_2026_large.csv,"
        "usgs_western=data/real_spatial_large/usgs_western_us_earthquakes_1980_2026_large.csv"))
    ap.add_argument("--n-base", type=int, default=3000)
    ap.add_argument("--ops", type=int, default=600)
    ap.add_argument("--insert-rate", type=float, default=0.50)
    ap.add_argument("--delete-rate", type=float, default=0.35)
    ap.add_argument("--move-frac-of-std", type=float, default=0.10)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/realdata_corrections")
    args = ap.parse_args()

    specs = []
    for tok in args.datasets.split(","):
        tok = tok.strip()
        if "=" in tok:
            name, path = tok.split("=", 1)
            specs.append((name.strip(), ROOT / path.strip()))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="realcorr_"))

    agg, rows = {}, []
    for name, path in specs:
        pts = load_xy(path)
        n_base = min(args.n_base, len(pts) // 2)
        base_pts = pts[:n_base]
        pool_pts = pts[n_base:]
        per_seed = {}
        total_mism = total_updates = ok_count = 0
        for seed in seeds:
            rd = tmp / f"{name}_s{seed}"
            rd.mkdir(parents=True, exist_ok=True)
            base, stream = rd / "base.csv", rd / "stream.csv"
            write_points(base, base_pts, np.zeros(len(base_pts), dtype=int))
            stream.write_text(make_correction_stream(
                base_pts, pool_pts, args.ops, seed,
                args.insert_rate, args.delete_rate, args.move_frac_of_std))
            eps = float(estimate_eps(base_pts, args.min_pts))
            bdf = run_baseline(args.idbscan_exe, base, stream, eps, args.min_pts, args.every, rd)
            per_seed[seed] = bdf.set_index("step")["incr_ari"]
            ok, mism, n = run_delaucluster_verified(args.cluster_exe, base, stream, rd)
            ok_count += int(ok); total_mism += max(mism, 0); total_updates += n
        mat = pd.DataFrame(per_seed)
        agg[name] = mat
        finals = mat.iloc[-1].values
        rows.append({
            "dataset": name, "n_base": n_base, "ops": args.ops, "seeds": len(seeds),
            "insert_rate": args.insert_rate, "delete_rate": args.delete_rate,
            "baseline_final_ari_mean": float(np.mean(finals)),
            "baseline_final_ari_std": float(np.std(finals)),
            "baseline_min_ari_mean": float(mat.min().mean()),
            "delaucluster_runs_ok": ok_count,
            "delaucluster_total_updates": total_updates,
            "delaucluster_total_mismatches": total_mism,
            "delaucluster_ari": 1.0 if (ok_count == len(seeds) and total_mism == 0) else float("nan"),
        })
        print(f"[{name}] base={n_base} baseline final ARI {np.mean(finals):.3f}+/-{np.std(finals):.3f} "
              f"| DelauCluster {ok_count}/{len(seeds)} exact, {total_mism} mismatches / {total_updates} updates")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "realdata_corrections_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        colors = plt.cm.plasma(np.linspace(0.15, 0.7, len(agg)))
        for (name, mat), col in zip(agg.items(), colors):
            steps = mat.index.values
            m, s = mat.mean(axis=1).values, mat.std(axis=1).values
            ax.plot(steps, m, lw=1.5, color=col, label=f"Incr. DBSCAN ({name})")
            ax.fill_between(steps, m - s, m + s, color=col, alpha=0.18)
        ax.axhline(1.0, color="crimson", lw=2.2, ls="--", label="DelauCluster (verified exact)")
        ax.set_xlabel("real correction-stream operations (insert/delete/move)")
        ax.set_ylabel("ARI vs from-scratch (same method)")
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title("Exact maintenance under real catalog corrections", fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / "figures" / f"realdata_corrections.{ext}", dpi=160)
        plt.close(fig)
        print(f"wrote {out/'figures'/'realdata_corrections.pdf'}")
    except Exception as e:
        print(f"figure skipped: {e}")

    print("\n=== REAL-DATA CORRECTION SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
