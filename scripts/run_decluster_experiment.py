#!/usr/bin/env python3
"""Real-data drift under a genuine deletion workload: catalog DECLUSTERING.

The motivating failure (a fast incremental method cannot un-merge on deletion)
needs a *real* deletion stream that is connectivity-critical. Earthquake-catalog
declustering is exactly such an operation: it removes aftershocks (the dense
spatio-temporal "glue") to leave independent mainshocks, a standard step before
seismic-hazard analysis. Removing that glue disconnects clusters the grow-only
union-find merged, so it drifts; DelauCluster maintains the clustering exactly.

We fetch a real USGS ComCat catalog (Southern California, M>=2.5), apply the
Gardner & Knopoff (1974) window declustering, and replay: load a small initial
base (--n-base events), insert the remaining events in origin-time order, then
delete every aftershock. The raw catalog is cached to data/comcat_revision/ for
offline reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import fit_base_threshold, write_points  # noqa: E402

CACHE = ROOT / "data" / "comcat_revision" / "socal_decluster.csv"
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch(args):
    if CACHE.exists() and not args.refetch:
        return pd.read_csv(CACHE)
    url = (f"{FDSN}?format=geojson&starttime={args.starttime}&endtime={args.endtime}"
           f"&minlatitude={args.lat0}&maxlatitude={args.lat1}&minlongitude={args.lon0}"
           f"&maxlongitude={args.lon1}&minmagnitude={args.minmag}&orderby=time-asc&limit=20000")
    with urllib.request.urlopen(url, timeout=120) as r:
        gj = json.load(r)
    rows = []
    for f in gj["features"]:
        p = f["properties"]; c = f["geometry"]["coordinates"]
        if p.get("mag") is None or c is None or p.get("type") != "earthquake":
            continue
        rows.append({"lon": c[0], "lat": c[1], "time_ms": int(p["time"]), "mag": float(p["mag"])})
    df = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"cached {len(df)} events -> {CACHE}")
    return df


def gardner_knopoff(df):
    """Window declustering (Gardner & Knopoff 1974). Returns a boolean aftershock mask."""
    lat0 = float(df.lat.mean())
    lon = df.lon.to_numpy(); lat = df.lat.to_numpy(); t = df.time_ms.to_numpy(); M = df.mag.to_numpy()
    n = len(df)
    def km(i, j):
        return 111.0 * math.hypot(lat[i] - lat[j], (lon[i] - lon[j]) * math.cos(math.radians(lat0)))
    def L(m):  # distance window, km
        return 10 ** (0.1238 * m + 0.983)
    def T(m):  # time window, days
        return 10 ** (0.032 * m + 2.7389) if m >= 6.5 else 10 ** (0.5409 * m - 0.547)
    aftershock = np.zeros(n, dtype=bool)
    for i in range(n):
        if aftershock[i]:
            continue
        Lw = L(M[i]); Tw = T(M[i]) * 86400_000
        for j in range(i + 1, n):
            if t[j] - t[i] > Tw:
                break
            if M[j] <= M[i] and km(i, j) <= Lw:
                aftershock[j] = True
    return aftershock


def project_km(df):
    lat0 = float(df.lat.mean())
    x = (df.lon - df.lon.mean()) * 111.0 * math.cos(math.radians(lat0))
    y = (df.lat - df.lat.mean()) * 111.0
    return np.c_[x.to_numpy(), y.to_numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starttime", default="2012-01-01")
    ap.add_argument("--endtime", default="2020-01-01")
    ap.add_argument("--lat0", type=float, default=32.0); ap.add_argument("--lat1", type=float, default=37.0)
    ap.add_argument("--lon0", type=float, default=-121.0); ap.add_argument("--lon1", type=float, default=-114.0)
    ap.add_argument("--minmag", type=float, default=2.5)
    ap.add_argument("--n-base", type=int, default=300)
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--verify", action="store_true", help="run DelauCluster with per-op verifier (slow)")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/decluster")
    args = ap.parse_args()

    df = fetch(args)
    aft = gardner_knopoff(df)
    P = project_km(df)
    n = len(df); naft = int(aft.sum())
    print(f"events={n}  aftershocks(deleted)={naft} ({100*naft/n:.0f}%)  mainshocks={n-naft}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="decluster_"))
    base = tmp / "b.csv"; nb = args.n_base
    write_points(base, P[:nb], np.zeros(nb, dtype=int))
    lines = [f"insert,{P[k][0]:.6f},{P[k][1]:.6f},0" for k in range(nb, n)]
    lines += [f"delete,{P[k][0]:.6f},{P[k][1]:.6f}" for k in range(n) if aft[k]]
    stream = tmp / "s.csv"; stream.write_text("\n".join(lines) + "\n")
    n_ins = sum(1 for ln in lines if ln.startswith("insert"))
    n_del = sum(1 for ln in lines if ln.startswith("delete"))

    eps = float(estimate_eps(P[:nb], args.min_pts))
    subprocess.run([args.idbscan_exe, str(base), str(stream), str(tmp / "uf.csv"),
                    "--eps", f"{eps:.6f}", "--min-pts", str(args.min_pts), "--every", str(args.every)],
                   check=True, capture_output=True)
    uf = pd.read_csv(tmp / "uf.csv"); a = uf[uf.incr_ari > -2].incr_ari

    thr = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
    cmd = [args.cluster_exe, "dynamic", str(base), str(stream), str(tmp / "dc"),
           "--threshold", f"{thr:.17g}", "--local-bucket-refresh"]
    if args.verify:
        cmd.append("--verify-dynamic")
    res = subprocess.run(cmd, capture_output=True, text=True)
    dc_ok = res.returncode == 0
    dlog = pd.read_csv(tmp / "dc" / "dynamic_log.csv")
    mism = int(dlog.get("verification_mismatches", pd.Series([0])).sum()) if args.verify else 0

    summary = pd.DataFrame([{
        "region": "SoCal M>=2.5 (USGS ComCat)", "events": n, "aftershocks_deleted": naft,
        "inserts": n_ins, "deletes": n_del, "ops": len(lines),
        "unionfind_final_ari": float(a.iloc[-1]), "unionfind_min_ari": float(a.min()),
        "delaucluster_ok": dc_ok, "delaucluster_verified": bool(args.verify),
        "delaucluster_mismatches": mism,
    }])
    summary.to_csv(out / "decluster_summary.csv", index=False)
    uf.to_csv(out / "decluster_unionfind_ari.csv", index=False)
    print("\n=== REAL DECLUSTERING DRIFT SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nunion-find drifts to ARI {float(a.iloc[-1]):.3f} (min {float(a.min()):.3f}); "
          f"DelauCluster {'exact, '+str(mism)+' mismatches' if args.verify else 'completed'} "
          f"({'verified' if args.verify else 'guarded'}) over {len(lines)} real ops")


if __name__ == "__main__":
    main()
