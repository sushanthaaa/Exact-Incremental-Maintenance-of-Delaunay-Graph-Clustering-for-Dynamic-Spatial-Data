#!/usr/bin/env python3
"""Guard-off ("heuristic-only") ablation — B1c of the full-paper plan.

Replays delete/move-heavy streams twice with identical inputs:
  guarded : --local-bucket-refresh                  (Theorem-1 exact)
  noguard : --local-bucket-refresh --no-guard       (heuristic-only)
and measures, at prefix checkpoints, the ARI of the heuristic-only maintained
labels against the guarded (exact) labels on the same prefix — i.e., the drift
the guard prevents. Also records the first step at which the heuristic-only
labeling diverges from a from-scratch fit (via a --verify-dynamic replay that
aborts at the first mismatch; the log row count is the divergence step).

ARI convention: labels are cluster_id with all noise pooled as one class (-1),
the same convention as the incremental-DBSCAN baseline's internal ARI.

Latency columns in the logs are NOT quotable from this harness; the timing-day
rerun (same script, machine alone) is the only quotable source.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_stream import fit_base_threshold, write_points  # noqa: E402
import run_dynamic_faithfulness as rdf  # noqa: E402
import run_decluster_experiment as rde  # noqa: E402


def replay(exe, base, stream, thr, out_dir, extra=(), verify=False):
    cmd = [str(exe), "dynamic", str(base), str(stream), str(out_dir),
           "--threshold", f"{thr:.17g}", "--local-bucket-refresh", *extra]
    if verify:
        cmd.append("--verify-dynamic")
    return subprocess.run(cmd, capture_output=True, text=True)


def labels_of(out_dir):
    df = pd.read_csv(Path(out_dir) / "clusters.csv")
    lab = df.cluster_id.to_numpy().copy()
    lab[df.is_noise.to_numpy().astype(bool)] = -1
    return df.x.to_numpy(), df.y.to_numpy(), lab


def checkpoint_ari(exe, base, stream_lines, thr, k, tmp):
    """ARI(noguard vs guarded) on the first k stream ops."""
    ps = tmp / f"prefix_{k}.csv"
    ps.write_text("\n".join(stream_lines[:k]) + "\n")
    ga = tmp / f"g_{k}"
    na = tmp / f"n_{k}"
    rg = replay(exe, base, ps, thr, ga)
    rn = replay(exe, base, ps, thr, na, extra=("--no-guard",))
    if rg.returncode != 0 or rn.returncode != 0:
        raise RuntimeError(f"prefix replay failed at k={k}: "
                           f"g={rg.returncode} n={rn.returncode}\n{rg.stderr[-500:]}{rn.stderr[-500:]}")
    gx, gy, gl = labels_of(ga)
    nx, ny, nl = labels_of(na)
    if len(gl) != len(nl) or not (np.allclose(gx, nx) and np.allclose(gy, ny)):
        raise RuntimeError(f"active sets differ between arms at k={k}")
    return adjusted_rand_score(gl, nl)


def first_divergence(exe, base, stream, thr, tmp):
    """Step of first mismatch vs a from-scratch fit, via verifier abort."""
    vd = tmp / "noguard_verify"
    res = replay(exe, base, stream, thr, vd, extra=("--no-guard",), verify=True)
    log = vd / "dynamic_log.csv"
    nrows = len(pd.read_csv(log)) if log.exists() else 0
    if res.returncode == 0:
        return -1, nrows  # never diverged from the from-scratch fit
    return nrows, nrows  # aborted at op #nrows


def run_one(exe, name, base, stream_lines, thr, checkpoints, out, rows, curves):
    tmp = Path(tempfile.mkdtemp(prefix=f"guardabl_{name}_"))
    stream = tmp / "stream.csv"
    stream.write_text("\n".join(stream_lines) + "\n")
    nops = len(stream_lines)
    ks = sorted({max(1, round(nops * i / checkpoints)) for i in range(1, checkpoints + 1)})
    aris = []
    for k in ks:
        a = checkpoint_ari(exe, base, stream_lines, thr, k, tmp)
        aris.append(a)
        curves.append({"run": name, "step": k, "noguard_vs_exact_ari": a})
    div_step, _ = first_divergence(exe, base, stream, thr, tmp)
    glog = pd.read_csv(tmp / f"g_{ks[-1]}" / "dynamic_log.csv")
    rows.append({
        "run": name, "ops": nops, "threshold": thr,
        "final_ari": aris[-1], "min_ari": min(aris),
        "first_divergence_step": div_step,
        "guarded_fallbacks": int(glog.local_relabel_fallback.sum()),
    })
    print(f"[{name}] ops={nops} final_ari={aris[-1]:.4f} min_ari={min(aris):.4f} "
          f"first_div={div_step} guarded_fallbacks={int(glog.local_relabel_fallback.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ops", type=int, default=1000)
    ap.add_argument("--insert-rate", type=float, default=0.25)
    ap.add_argument("--delete-rate", type=float, default=0.50)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--checkpoints", type=int, default=10)
    ap.add_argument("--decluster", action="store_true",
                    help="also run the real SoCal declustering stream (cached; no refetch)")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--out-dir", default="results/guard_ablation")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            x, y = rdf.build(ds, seed)
            x, y = rdf.resize_points(x, y, args.n, seed + 2000)
            tmp = Path(tempfile.mkdtemp(prefix="guardabl_base_"))
            base = tmp / "base.csv"
            write_points(base, x, y)
            lines = rdf.make_dm_stream(x, args.ops, seed + 1000,
                                       args.insert_rate, args.delete_rate).strip().split("\n")
            thr = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
            run_one(args.cluster_exe, f"{ds}_s{seed}", base, lines, thr,
                    args.checkpoints, out, rows, curves)

    if args.decluster:
        class A:  # cached fetch, defaults of run_decluster_experiment
            refetch = False
            starttime = "2012-01-01"; endtime = "2020-01-01"
            lat0, lat1, lon0, lon1 = 32.0, 37.0, -121.0, -114.0
            minmag = 2.5
        df = rde.fetch(A())
        aft = rde.gardner_knopoff(df)
        P = rde.project_km(df)
        nb, n = 300, len(df)
        tmp = Path(tempfile.mkdtemp(prefix="guardabl_dec_"))
        base = tmp / "base.csv"
        write_points(base, P[:nb], np.zeros(nb, dtype=int))
        lines = [f"insert,{P[k][0]:.6f},{P[k][1]:.6f},0" for k in range(nb, n)]
        lines += [f"delete,{P[k][0]:.6f},{P[k][1]:.6f}" for k in range(n) if aft[k]]
        thr = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
        run_one(args.cluster_exe, "socal_decluster", base, lines, thr,
                args.checkpoints, out, rows, curves)

    pd.DataFrame(rows).to_csv(out / "guard_ablation_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "guard_ablation_curves.csv", index=False)
    print(f"wrote {out}/guard_ablation_summary.csv ({len(rows)} runs)")


if __name__ == "__main__":
    main()
