#!/usr/bin/env python3
"""Guarded vs --no-guard per-update latency — the B3 guard-share measurement.

TIMING HARNESS: quotable only when run in the dedicated timing session,
sequentially and alone, single-threaded, Release, on the paper's machine.
Mirrors the scaling harness's n=50k setting (chain_noise-style streams,
insert/delete/move mix 0.34/0.33/0.33) so the guard share applies to the
paper's headline local-update figure. The --no-guard arm's labels may drift
(that is the point of the drift ablation); here we only time the updates.
"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--ops", type=int, default=150)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--out-dir", default="results/guard_latency")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            x, y = build(ds, seed)
            x, y = resize_points(x, y, args.n, seed + 2000)
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                base = tmp / "b.csv"
                write_points(base, x, y)
                st = tmp / "s.csv"
                st.write_text("\n".join(make_stream(x, y, args.ops, seed + 7,
                                                    0.34, 0.33, 0.04)) + "\n")
                tau = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
                ARMS = (("guarded", ["--local-bucket-refresh"]),
                        ("always_full_relabel", ["--local-bucket-refresh",
                                                 "--always-full-relabel"]),
                        ("no_guard", ["--local-bucket-refresh", "--no-guard"]),
                        ("full_rebuild", []))  # no local flags -> rebuild per op
                for arm, flags in ARMS:
                    rd = tmp / arm
                    res = subprocess.run(["/usr/bin/time", "-l", args.cluster_exe,
                                          "dynamic", str(base), str(st), str(rd),
                                          "--threshold", f"{tau:.17g}", *flags],
                                         check=True, capture_output=True, text=True)
                    rss = -1
                    for ln in res.stderr.splitlines():
                        if "maximum resident set size" in ln:
                            rss = int(ln.strip().split()[0])
                    d = pd.read_csv(rd / "dynamic_log.csv")
                    lat = (d.time_ns / 1e6).to_numpy()
                    per_op = d.groupby("operation").time_ns.mean() / 1e6
                    rows.append({"dataset": ds, "seed": seed, "arm": arm,
                                 "n": args.n, "ops": len(d),
                                 "mean_ms": float(np.mean(lat)),
                                 "p95_ms": float(np.percentile(lat, 95)),
                                 "insert_ms": float(per_op.get("insert", float("nan"))),
                                 "delete_ms": float(per_op.get("delete", float("nan"))),
                                 "move_ms": float(per_op.get("move", float("nan"))),
                                 "peak_rss_bytes": rss,
                                 "fallback_pct": 100.0 * d.local_relabel_fallback.mean()})
                    r = rows[-1]
                    print(f"[{ds} s{seed} {arm}] mean={r['mean_ms']:.2f}ms "
                          f"p95={r['p95_ms']:.2f}ms rss={rss/1e6:.0f}MB")
    df = pd.DataFrame(rows)
    df.to_csv(out / "guard_latency.csv", index=False)
    g = df[df.arm == "guarded"].mean_ms.mean()
    ng = df[df.arm == "no_guard"].mean_ms.mean()
    share = 100.0 * (g - ng) / g
    with open(out / "guard_share.txt", "w") as f:
        f.write(f"guarded_mean_ms={g:.4f}\nno_guard_mean_ms={ng:.4f}\n"
                f"guard_share_pct={share:.2f}\n")
    print(f"guarded {g:.2f} ms vs no-guard {ng:.2f} ms -> guard share {share:.1f}%")


if __name__ == "__main__":
    main()
