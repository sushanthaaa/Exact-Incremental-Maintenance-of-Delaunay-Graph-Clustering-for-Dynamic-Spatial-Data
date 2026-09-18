#!/usr/bin/env python3
"""#5: exact maintenance under a GENUINELY REAL USGS ComCat revision stream.

Unlike a modeled correction stream, this builds the insert/delete operations from
the actual USGS ComCat catalog:

  * INSERT -- every catalogued event, in real origin-time order;
  * DELETE -- real reclassification withdrawals: events catalogued as quarry
    blasts / explosions / other non-earthquakes are withdrawn from the earthquake
    hotspot map after review (here, after a fixed review lag).

So both operation types come from the real catalog, not a synthetic model. (Public
ComCat does not expose per-event relocation histories, so arbitrary *moves* remain
covered by the synthetic and NYC studies; this stream is real insert+delete.)

The raw events are fetched once from the FDSN event web service and cached to
data/comcat_revision/ so the experiment is reproducible offline. We then measure
per-step faithfulness (ARI vs a from-scratch run of the same method on the active
points): DelauCluster's verifier confirms 0 mismatches; the fast incremental
union-find baseline is reported for reference.

The output figure is produced to IEEE (ICTAI 2026 / IEEE Xplore) graphics specs
via the shared helpers imported from run_dynamic_faithfulness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import write_points  # noqa: E402
from run_dynamic_faithfulness import (  # noqa: E402
    run_baseline,
    run_delaucluster_verified,
    set_ieee_style,
    save_ieee,
    ieee_top_legend,
    IEEE_FIG_W,
    IEEE_FIG_H,
    IEEE_EXACT_COLOR,
    IEEE_EXACT_DASH,
)
from run_realdata_corrections import load_xy, make_correction_stream  # noqa: E402

NYC_PATH = ROOT / "data" / "real_spatial_large" / "nyc_collisions_2012_2026_large.csv"

CACHE = ROOT / "data" / "comcat_revision" / "socal_events.csv"
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_events(starttime, endtime, bbox, minmag):
    lat0, lat1, lo0, lo1 = bbox
    url = (f"{FDSN}?format=geojson&starttime={starttime}&endtime={endtime}"
           f"&minlatitude={lat0}&maxlatitude={lat1}&minlongitude={lo0}"
           f"&maxlongitude={lo1}&minmagnitude={minmag}&orderby=time-asc&limit=20000")
    with urllib.request.urlopen(url, timeout=120) as r:
        gj = json.load(r)
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        c = f["geometry"]["coordinates"]  # [lon, lat, depth]
        if p.get("time") is None or c is None:
            continue
        rows.append({"lon": c[0], "lat": c[1], "time_ms": int(p["time"]),
                     "type": p.get("type", "earthquake")})
    df = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    return df


def load_or_fetch(args):
    if CACHE.exists() and not args.refetch:
        return pd.read_csv(CACHE)
    print(f"fetching ComCat events -> {CACHE} ...")
    df = fetch_events(args.starttime, args.endtime,
                      (args.lat0, args.lat1, args.lon0, args.lon1), args.minmag)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"  cached {len(df)} events ({(df.type!='earthquake').sum()} non-earthquakes)")
    return df


def project_km(df):
    """Equirectangular projection to local kilometers about the region center."""
    lat0 = float(df.lat.mean())
    x = (df.lon - df.lon.mean()) * 111.0 * math.cos(math.radians(lat0))
    y = (df.lat - df.lat.mean()) * 111.0
    out = df.copy()
    out["x"] = x.to_numpy()
    out["y"] = y.to_numpy()
    return out


def build_real_stream(df, n_base, n_ops, review_lag):
    """Real insert+delete stream: inserts in time order; non-earthquakes withdrawn
    after a fixed review lag (real reclassification). Returns (base_pts, lines)."""
    df = df.reset_index(drop=True)
    base = df.iloc[:n_base]
    base = base[base.type == "earthquake"]  # base map = earthquakes present at start
    base_pts = base[["x", "y"]].to_numpy(float)
    rest = df.iloc[n_base:].reset_index(drop=True)
    lines, pending = [], {}  # step_to_fire -> list of (x,y)
    step = 0
    n_del = 0
    for _, e in rest.iterrows():
        if len(lines) >= n_ops:
            break
        # fire any scheduled reclassification withdrawals
        for x, y in pending.pop(step, []):
            if len(lines) >= n_ops:
                break
            lines.append(f"delete,{x:.6f},{y:.6f}")
            n_del += 1
        if len(lines) >= n_ops:
            break
        lines.append(f"insert,{e.x:.6f},{e.y:.6f},0")
        if e.type != "earthquake":
            pending.setdefault(step + review_lag, []).append((e.x, e.y))
        step += 1
    return base_pts, "\n".join(lines) + "\n", n_del


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starttime", default="2018-01-01")
    ap.add_argument("--endtime", default="2024-01-01")
    ap.add_argument("--lat0", type=float, default=32.0)
    ap.add_argument("--lat1", type=float, default=37.0)
    ap.add_argument("--lon0", type=float, default=-122.0)
    ap.add_argument("--lon1", type=float, default=-114.0)
    ap.add_argument("--minmag", type=float, default=2.0)
    ap.add_argument("--n-base", type=int, default=1500)
    ap.add_argument("--ops", type=int, default=2500)
    ap.add_argument("--review-lag", type=int, default=25)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--nyc-seeds", default="42,43,44")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/comcat_revision")
    args = ap.parse_args()

    df = project_km(load_or_fetch(args))
    # De-duplicate exact-coincident events (the general-position precondition; also
    # a real correction type). ComCat rounds lon/lat, so a small fraction project
    # to identical coordinates; keep the earliest by time.
    before = len(df)
    df = df.sort_values("time_ms").drop_duplicates(subset=["x", "y"], keep="first").reset_index(drop=True)
    n_noneq = int((df.type != "earthquake").sum())
    print(f"events={len(df)} (de-duplicated {before - len(df)})  "
          f"non-earthquakes(real withdrawals)={n_noneq}")

    base_pts, stream_text, n_del = build_real_stream(df, args.n_base, args.ops, args.review_lag)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="comcat_"))
    base, stream = tmp / "base.csv", tmp / "stream.csv"
    write_points(base, base_pts, np.zeros(len(base_pts), dtype=int))
    stream.write_text(stream_text)
    n_ins = stream_text.count("insert,")
    print(f"base={len(base_pts)} earthquakes | stream: {n_ins} real inserts, "
          f"{n_del} real reclassification deletes")

    eps = float(estimate_eps(base_pts, args.min_pts))
    usgs_bdf = run_baseline(args.idbscan_exe, base, stream, eps, args.min_pts, args.every, tmp)
    ok, mism, n = run_delaucluster_verified(args.cluster_exe, base, stream, tmp)

    rows = [{
        "dataset": "USGS earthquakes (real revision)", "stream_kind": "real (ComCat)",
        "n_base": len(base_pts), "inserts": n_ins, "deletes": n_del,
        "baseline_final_ari": float(usgs_bdf.iloc[-1]["incr_ari"]),
        "baseline_min_ari": float(usgs_bdf["incr_ari"].min()),
        "delaucluster_updates": n, "delaucluster_mismatches": max(mism, 0),
        "delaucluster_ari": 1.0 if (ok and mism == 0) else float("nan"),
    }]
    print(f"\n[USGS real] DelauCluster {'exact' if ok and mism==0 else 'FAILED'}, "
          f"{max(mism,0)} mismatches / {n} real updates; baseline final ARI "
          f"{float(usgs_bdf.iloc[-1]['incr_ari']):.4f}")

    # NYC reference: real coordinates with a MODELED correction stream over several
    # seeds (the drift case; public NYC Open Data exposes no per-record revision
    # log). We report mean +/- std and a mean-std band, matching the synthetic study.
    nyc_mat = None
    nyc_seeds = [int(s) for s in args.nyc_seeds.split(",") if s.strip()]
    if NYC_PATH.exists():
        npts = load_xy(NYC_PATH)
        nb = min(3000, len(npts) // 2)
        nbase_pts, npool = npts[:nb], npts[nb:]
        per_seed, ntot_mism, ntot_upd, nok_count = {}, 0, 0, 0
        for sd in nyc_seeds:
            nrd = tmp / f"nyc_s{sd}"
            nrd.mkdir(parents=True, exist_ok=True)
            nbase, nstream = nrd / "base.csv", nrd / "stream.csv"
            write_points(nbase, nbase_pts, np.zeros(len(nbase_pts), dtype=int))
            nstream.write_text(make_correction_stream(nbase_pts, npool, 600, sd, 0.50, 0.35, 0.10))
            neps = float(estimate_eps(nbase_pts, args.min_pts))
            b = run_baseline(args.idbscan_exe, nbase, nstream, neps, args.min_pts, args.every, nrd)
            per_seed[sd] = b.set_index("step")["incr_ari"]
            nok, nmism, nn = run_delaucluster_verified(args.cluster_exe, nbase, nstream, nrd)
            nok_count += int(nok); ntot_mism += max(nmism, 0); ntot_upd += nn
        nyc_mat = pd.DataFrame(per_seed)
        finals = nyc_mat.iloc[-1].values
        rows.append({
            "dataset": "NYC collisions (modeled corr.)", "stream_kind": "modeled",
            "n_base": nb, "inserts": "~", "deletes": "~",
            "baseline_final_ari": float(np.mean(finals)),
            "baseline_min_ari": float(nyc_mat.min().mean()),
            "delaucluster_updates": ntot_upd, "delaucluster_mismatches": ntot_mism,
            "delaucluster_ari": 1.0 if (nok_count == len(nyc_seeds) and ntot_mism == 0) else float("nan"),
        })
        print(f"[NYC modeled] DelauCluster {nok_count}/{len(nyc_seeds)} exact, "
              f"{ntot_mism} mismatches / {ntot_upd} updates; baseline final ARI "
              f"{np.mean(finals):.3f}+/-{np.std(finals):.3f} (min {nyc_mat.min().mean():.3f})")

    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "comcat_revision_summary.csv", index=False)
    print("\n=== REAL-DATA FAITHFULNESS SUMMARY ===")
    print(summary.to_string(index=False))

    # Save the per-step curves so the figure is reproducible without re-running.
    usgs_bdf.to_csv(out / "usgs_revision_ari.csv", index=False)
    if nyc_mat is not None:
        nyc_mat.to_csv(out / "nyc_modeled_ari.csv")

    # Figure (paper fig:real): real-catalog faithfulness on a common x (fraction of
    # the stream, since the three real streams differ in length). Three incremental-
    # DBSCAN baselines vs a from-scratch run of the same method, and DelauCluster
    # pinned at 1.0 (verified exact). The genuine declustering stream is the headline
    # real drift; its curve is loaded from run_decluster_experiment.py's output.
    #
    # IEEE single-column figure: vector PDF, true 8 pt text, each series given a
    # distinct colour + line style + marker so it reads in grey-scale / for CVD.
    # The legend is a compact, frameless 2-column block ABOVE the axes; labels are
    # kept short to fit the column, with the full detail (the 0.65 drift, the
    # benign case, "incremental DBSCAN") carried by the LaTeX \caption.
    decluster_csv = ROOT / "results" / "decluster" / "decluster_unionfind_ari.csv"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        set_ieee_style()
        fig, ax = plt.subplots(figsize=(IEEE_FIG_W, IEEE_FIG_H),
                               constrained_layout=True)

        rs = usgs_bdf.step.to_numpy(float)
        mev = max(1, len(rs) // 6)
        ax.plot(rs / rs.max(), usgs_bdf.incr_ari, color="#E69F00", ls="-",
                marker="o", markevery=mev, ms=3.0, lw=1.3,
                label="USGS revision")

        if nyc_mat is not None:
            steps = nyc_mat.index.values.astype(float)
            m, s = nyc_mat.mean(axis=1).values, nyc_mat.std(axis=1).values
            xf = steps / steps.max()
            mev_n = max(1, len(xf) // 6)
            ax.plot(xf, m, color="#0072B2", ls="--", marker="s", markevery=mev_n,
                    ms=3.0, lw=1.3, label="NYC (modeled)")
            ax.fill_between(xf, m - s, m + s, color="#0072B2", alpha=0.15, linewidth=0)

        if decluster_csv.exists():
            dec = pd.read_csv(decluster_csv)
            dec = dec[dec.incr_ari > -2]
            ds = dec.step.to_numpy(float)
            mev_d = max(1, len(ds) // 6)
            ax.plot(ds / ds.max(), dec.incr_ari, color="#009E73", ls="-.",
                    marker="^", markevery=mev_d, ms=3.2, lw=1.6,
                    label="USGS declustering")
        else:
            print(f"declustering curve not found ({decluster_csv}); "
                  f"run run_decluster_experiment.py to include it")

        ax.axhline(1.0, color=IEEE_EXACT_COLOR, lw=1.6, ls=IEEE_EXACT_DASH,
                   label="DelauCluster (exact)")
        ax.set_ylim(0.5, 1.02)
        ax.set_xlabel("fraction of correction stream (operations)")
        ax.set_ylabel("ARI vs from-scratch (same method)")
        ax.grid(True, which="major", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        ieee_top_legend(fig, ax)  # compact, frameless 2-col legend above the axes (caption is the title)
        save_ieee(fig, out / "figures" / "faithfulness_real")
        plt.close(fig)
        print(f"wrote {out / 'figures' / 'faithfulness_real.pdf'}")
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()