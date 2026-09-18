#!/usr/bin/env python3
"""Spatial-isolation statistic for the ComCat revision withdrawals — A10.

The paper says the union-find baseline is accurate-by-chance on the real
revision stream because the withdrawn quarry-blast/explosion events are
"spatially isolated and hence benign". This script measures that: replaying
the cached revision stream (NO refetch), at each withdrawal step it fits the
active set from scratch at the frozen base threshold and records
  (a) the withdrawn event's retained-edge degree at deletion time, and
  (b) whether deleting it changes the partition of the REMAINING points
      (component split), by comparing the before/after partitions restricted
      to the surviving points.
"benign" = deletion does not re-partition the survivors.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import fit_base_threshold, write_points  # noqa: E402
import run_comcat_revision_experiment as rcr  # noqa: E402


def dbscan_split(pts, keep, j, eps, min_pts):
    """Does deleting point j split its DBSCAN cluster (the baseline's model)?
    Same eps/min_pts as the experiment's incremental-DBSCAN baseline."""
    lb = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(pts)
    if lb[j] == -1:
        return False
    members = (lb == lb[j])
    members[j] = False
    la = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(pts[keep])
    after_ids = {int(v) for v in la[members[keep]] if v != -1}
    return len(after_ids) > 1


def static_fit(exe, pts, thr, out_dir):
    tmp = Path(out_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    pcsv = tmp / "p.csv"
    write_points(pcsv, pts, np.zeros(len(pts), dtype=int))
    res = subprocess.run([str(exe), "static", str(pcsv), str(tmp),
                          "--threshold", f"{thr:.17g}"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-500:])
    cl = pd.read_csv(tmp / "clusters.csv")
    ed = pd.read_csv(tmp / "edges.csv")
    return cl, ed


def partition_signature(cl, keep_mask):
    lab = cl.cluster_id.to_numpy().copy()
    lab[cl.is_noise.to_numpy().astype(bool)] = -1
    kept = lab[keep_mask]
    # canonical: group indices by label, noise pooled
    groups = {}
    for i, l in enumerate(kept):
        groups.setdefault(l, []).append(i)
    return sorted(tuple(sorted(v)) for v in groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=1500)
    ap.add_argument("--ops", type=int, default=2500)
    ap.add_argument("--review-lag", type=int, default=25)
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--out-dir", default="results/revision_isolation")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    class A:
        refetch = False
        starttime = "2018-01-01"; endtime = "2024-01-01"
        lat0, lat1, lon0, lon1 = 32.0, 37.0, -122.0, -114.0
        minmag = 2.0
    df = rcr.load_or_fetch(A())
    df = rcr.project_km(df)
    base_pts, stream_text, n_del = rcr.build_real_stream(
        df, args.n_base, args.ops, args.review_lag)
    lines = stream_text.strip().split("\n")
    print(f"stream: {len(lines)} ops, {n_del} withdrawals, base {len(base_pts)}")

    tmp = Path(tempfile.mkdtemp(prefix="revisol_"))
    bcsv = tmp / "base.csv"
    write_points(bcsv, base_pts, np.zeros(len(base_pts), dtype=int))
    thr = fit_base_threshold(Path(args.cluster_exe), bcsv, tmp / "bs")
    eps = float(estimate_eps(base_pts, 4))  # exactly as the experiment derives it
    print(f"frozen threshold: {thr:.6g}   baseline eps: {eps:.6g} (min_pts=4)")

    active = [tuple(p) for p in base_pts]
    rows = []
    k = 0
    for ln in lines:
        parts = ln.split(",")
        if parts[0] == "insert":
            active.append((float(parts[1]), float(parts[2])))
            continue
        assert parts[0] == "delete"
        k += 1
        x, y = float(parts[1]), float(parts[2])
        pts = np.array(active)
        # nearest active point = the withdrawn event (same rule as the binary)
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        j = int(np.argmin(d2))
        cl, ed = static_fit(args.cluster_exe, pts, thr, tmp / f"w{k}_before")
        ret = ed[ed.retained == 1] if "retained" in ed.columns else ed
        deg = int(((ret.u == j) | (ret.v == j)).sum()) if {"u", "v"} <= set(ret.columns) else -1
        was_noise = bool(cl.iloc[j].is_noise)
        keep = np.ones(len(pts), dtype=bool)
        keep[j] = False
        sig_before = partition_signature(cl, keep)
        cl2, _ = static_fit(args.cluster_exe, pts[keep], thr, tmp / f"w{k}_after")
        sig_after = partition_signature(cl2, np.ones(len(cl2), dtype=bool))
        benign = sig_before == sig_after
        # magnitude of the survivor-partition change
        from sklearn.metrics import adjusted_rand_score
        lab_b = cl.cluster_id.to_numpy().copy()
        lab_b[cl.is_noise.to_numpy().astype(bool)] = -1
        lab_a = cl2.cluster_id.to_numpy().copy()
        lab_a[cl2.is_noise.to_numpy().astype(bool)] = -1
        ari_surv = adjusted_rand_score(lab_b[keep], lab_a)
        # SPLIT event: does the deleted point's non-noise component fall apart
        # into >1 non-noise clusters among the survivors? (the event a
        # grow-only union-find structurally cannot follow)
        split_event = False
        if not was_noise:
            cid = int(cl.iloc[j].cluster_id)
            members = (lab_b == cid)
            members[j] = False
            after_ids = {int(v) for v in lab_a[members[keep]] if v != -1}
            split_event = len(after_ids) > 1
        rows.append({"withdrawal": k, "x": x, "y": y, "n_active_before": len(pts),
                     "retained_degree": deg, "was_noise": was_noise,
                     "partition_unchanged": benign,
                     "survivor_ari": ari_surv, "split_event": split_event,
                     "dbscan_split_event": dbscan_split(pts, keep, j, eps, 4)})
        active.pop(j)

    res = pd.DataFrame(rows)
    res.to_csv(out / "revision_isolation.csv", index=False)
    iso = int((res.retained_degree == 0).sum())
    ben = int(res.partition_unchanged.sum())
    noi = int(res.was_noise.sum())
    spl = int(res.split_event.sum())
    dspl = int(res.dbscan_split_event.sum())
    mari = float(res.survivor_ari.mean())
    wari = float(res.survivor_ari.min())
    print(f"withdrawals={len(res)}  retained-isolated(deg 0)={iso}  "
          f"noise-at-deletion={noi}  partition-unchanged={ben}  "
          f"Delaunay-SPLIT-events={spl}  DBSCAN-split-events={dspl}  "
          f"survivor-ARI mean={mari:.6f} min={wari:.6f}")
    with open(out / "revision_isolation_summary.txt", "w") as f:
        f.write(f"withdrawals={len(res)}\nretained_isolated_deg0={iso}\n"
                f"noise_at_deletion={noi}\npartition_unchanged={ben}\n"
                f"delaunay_split_events={spl}\ndbscan_split_events={dspl}\n"
                f"survivor_ari_mean={mari:.6f}\nsurvivor_ari_min={wari:.6f}\n"
                f"frozen_threshold={thr:.17g}\nbaseline_eps={eps:.17g}\n")
    print(f"wrote {out}/revision_isolation.csv")


if __name__ == "__main__":
    main()
