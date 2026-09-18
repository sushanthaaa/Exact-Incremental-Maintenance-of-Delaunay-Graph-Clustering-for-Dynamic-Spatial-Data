#!/usr/bin/env python3
"""Relabel-shortcut ladder ablation — B2 of the full-paper plan.

Replays identical delete/move-heavy streams under each shortcut configuration
(all shortcuts on; each --no-* alone; all off), with --verify-dynamic ON in
every arm: the shortcuts are cost optimizations, so every arm must stay exact
(0 mismatches; nonzero exit would mean a real bug). Reports fallback rates and
the TRUE work-scope columns (edge_refresh_scope_vertices,
relabel_scope_vertices, relabel_search_vertices) — never the heuristic
affected_edges estimate. Latency columns are quotable ONLY from the timing-day
rerun of this script (machine alone).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_stream import fit_base_threshold, write_points  # noqa: E402
import run_dynamic_faithfulness as rdf  # noqa: E402

ARMS = {
    "all_shortcuts": [],
    "no_component_witness": ["--no-component-witness"],
    "no_connectivity_summary": ["--no-connectivity-summary"],
    "no_small_side_split": ["--no-small-side-split"],
    "all_off": ["--no-component-witness", "--no-connectivity-summary",
                "--no-small-side-split"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ops", type=int, default=1000)
    ap.add_argument("--insert-rate", type=float, default=0.25)
    ap.add_argument("--delete-rate", type=float, default=0.50)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--no-verify", action="store_true",
                    help="timing-day mode: skip the verifier so latencies are clean")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--out-dir", default="results/ladder_ablation")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            x, y = rdf.build(ds, seed)
            x, y = rdf.resize_points(x, y, args.n, seed + 2000)
            tmp = Path(tempfile.mkdtemp(prefix=f"ladder_{ds}_{seed}_"))
            base = tmp / "base.csv"
            write_points(base, x, y)
            stream = tmp / "stream.csv"
            stream.write_text(rdf.make_dm_stream(x, args.ops, seed + 1000,
                                                 args.insert_rate, args.delete_rate))
            thr = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
            for arm, flags in ARMS.items():
                rd = tmp / arm
                cmd = [args.cluster_exe, "dynamic", str(base), str(stream), str(rd),
                       "--threshold", f"{thr:.17g}", "--local-bucket-refresh", *flags]
                if not args.no_verify:
                    cmd.append("--verify-dynamic")
                res = subprocess.run(cmd, capture_output=True, text=True)
                dl = pd.read_csv(rd / "dynamic_log.csv")
                rows.append({
                    "dataset": ds, "seed": seed, "arm": arm, "ops": len(dl),
                    "exit_code": res.returncode,
                    "mismatches": int(dl.verification_mismatches.sum()) if "verification_mismatches" in dl and not args.no_verify else -1,
                    "fallback_pct": 100.0 * dl.local_relabel_fallback.mean(),
                    "edge_refresh_fallback_pct": 100.0 * dl.local_edge_refresh_fallback.mean(),
                    "mean_edge_refresh_scope_vertices": dl.edge_refresh_scope_vertices.mean(),
                    "mean_relabel_scope_vertices": dl.relabel_scope_vertices.mean(),
                    "mean_relabel_search_vertices": dl.relabel_search_vertices.mean(),
                    "mean_update_ms": dl.time_ns.mean() / 1e6,
                    "p95_update_ms": dl.time_ns.quantile(0.95) / 1e6,
                })
                r = rows[-1]
                print(f"[{ds} s{seed} {arm}] exit={r['exit_code']} mism={r['mismatches']} "
                      f"fallback={r['fallback_pct']:.2f}% scope={r['mean_relabel_search_vertices']:.1f}")
                if res.returncode != 0:
                    raise SystemExit(f"ladder arm FAILED (should be exact!): {ds} s{seed} {arm}\n{res.stderr[-1000:]}")
    pd.DataFrame(rows).to_csv(out / "ladder_ablation.csv", index=False)
    print(f"wrote {out}/ladder_ablation.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
