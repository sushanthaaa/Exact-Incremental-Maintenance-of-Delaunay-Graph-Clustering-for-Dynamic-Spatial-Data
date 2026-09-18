#!/usr/bin/env python3
"""Genuine-move stream from real AIS vessel traces (LA/Long Beach) — Tier C.

The paper's other real streams exercise genuine inserts and deletes; AIS
vessel traces supply the missing op: GENUINE moves. From the cached 24-hour
LA/LB AIS snapshot (data/ais_lalb/, see its PROVENANCE.txt) we build a
fully-dynamic stream in strict report order:

  * decimation: per vessel, a report is kept only if >= --min-dt seconds AND
    >= --min-disp km from that vessel's last kept report (standard AIS
    thinning; anchored vessels contribute ~no ops);
  * base set: each vessel first reporting within the first --base-window
    seconds, at its first kept position;
  * insert: a vessel's first kept report after the base window;
  * move:   every subsequent kept report (old exact position -> new);
  * delete: a vessel is withdrawn --gone-after seconds after its last RAW
    report (true disappearance from the receiver log, so an anchored vessel
    that keeps transmitting stays active even though decimation keeps none
    of its reports), scheduled in stream time;
  * an insert/move landing exactly on another active vessel's coordinates is
    skipped (general-position de-duplication, counted).

Coordinates are projected to local km (equirectangular about the basin mean).
DelauCluster replays under --verify-dynamic (every update checked against a
from-scratch fit at the frozen base threshold); the grow-only union-find
incremental-DBSCAN baseline replays the identical stream for drift. This is a
correctness experiment: latencies here are NOT quotable.
"""
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_baselines import estimate_eps  # noqa: E402
from dynamic_stream import fit_base_threshold, write_points  # noqa: E402

CACHE = ROOT / "data" / "ais_lalb" / "ais_lalb_big.csv"


def build_stream(df, min_dt, min_disp, base_window, gone_after):
    lat0 = float(df.lat.mean())
    kmx = 111.0 * math.cos(math.radians(lat0))
    lon0 = float(df.lon.mean())
    df = df.sort_values(["t", "mmsi"], kind="mergesort").reset_index(drop=True)
    t0 = int(df.t.min())
    t1 = int(df.t.max())

    def proj(lat, lon):
        return round((lon - lon0) * kmx, 6), round((lat - lat0) * 111.0, 6)

    # pass 1: decimate per vessel
    kept = []  # (t, mmsi, x, y)
    last = {}
    for r in df.itertuples():
        x, y = proj(r.lat, r.lon)
        p = last.get(r.mmsi)
        if p is None:
            last[r.mmsi] = (r.t, x, y)
            kept.append((r.t, r.mmsi, x, y))
            continue
        if r.t - p[0] >= min_dt and math.hypot(x - p[1], y - p[2]) >= min_disp:
            last[r.mmsi] = (r.t, x, y)
            kept.append((r.t, r.mmsi, x, y))

    # schedule deletes gone_after seconds after each vessel's last RAW report
    # (true disappearance; decimation must not fake departures of anchored,
    # still-transmitting vessels)
    last_seen = df.groupby("mmsi").t.max().to_dict()
    events = [(t, 0, m, x, y) for (t, m, x, y) in kept]  # kind 0 = report
    for m, tl in last_seen.items():
        if tl + gone_after <= t1:
            events.append((int(tl) + gone_after, 1, m, 0.0, 0.0))  # kind 1 = delete
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    base_pts, base_ids = [], set()
    lines = []
    pos = {}        # mmsi -> current (x, y)
    occupied = {}   # (x, y) -> mmsi
    skipped = 0
    n_ins = n_mov = n_del = 0
    for t, kind, m, x, y in events:
        if kind == 1:  # departure
            if m in pos:
                ox, oy = pos.pop(m)
                del occupied[(ox, oy)]
                lines.append(f"delete,{ox:.6f},{oy:.6f}")
                n_del += 1
            continue
        if m not in pos:
            if (x, y) in occupied:
                skipped += 1
                continue
            if t <= t0 + base_window and m not in base_ids:
                base_ids.add(m)
                base_pts.append((x, y))
                pos[m] = (x, y)
                occupied[(x, y)] = m
            else:
                lines.append(f"insert,{x:.6f},{y:.6f},0")
                n_ins += 1
                pos[m] = (x, y)
                occupied[(x, y)] = m
        else:
            ox, oy = pos[m]
            if (x, y) == (ox, oy):
                continue
            if (x, y) in occupied:
                skipped += 1
                continue
            lines.append(f"move,{ox:.6f},{oy:.6f},{x:.6f},{y:.6f}")
            n_mov += 1
            del occupied[(ox, oy)]
            pos[m] = (x, y)
            occupied[(x, y)] = m
    stats = {"vessels": df.mmsi.nunique(), "reports": len(df), "kept_reports": len(kept),
             "base": len(base_pts), "inserts": n_ins, "moves": n_mov,
             "deletes": n_del, "ops": len(lines), "skipped_coincident": skipped,
             "final_active": len(pos)}
    return np.array(base_pts), "\n".join(lines) + "\n", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dt", type=int, default=300)
    ap.add_argument("--min-disp", type=float, default=0.25)
    ap.add_argument("--base-window", type=int, default=900)
    ap.add_argument("--gone-after", type=int, default=3600)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/ais_moves")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"missing cache {CACHE}; see data/ais_lalb/PROVENANCE.txt (never refetch)")
    df = pd.read_csv(CACHE, header=None, names=["mmsi", "t", "lat", "lon", "status"])
    base_pts, stream_text, st = build_stream(df, args.min_dt, args.min_disp,
                                             args.base_window, args.gone_after)
    print("stream:", st)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ais_"))
    base = tmp / "base.csv"
    write_points(base, base_pts, np.zeros(len(base_pts), dtype=int))
    stream = tmp / "stream.csv"
    stream.write_text(stream_text)

    # grow-only union-find baseline (drift)
    eps = float(estimate_eps(base_pts, args.min_pts))
    subprocess.run([args.idbscan_exe, str(base), str(stream), str(tmp / "uf.csv"),
                    "--eps", f"{eps:.6f}", "--min-pts", str(args.min_pts),
                    "--every", str(args.every)], check=True, capture_output=True)
    uf = pd.read_csv(tmp / "uf.csv")
    a = uf[uf.incr_ari > -2].incr_ari

    # DelauCluster, verified
    thr = fit_base_threshold(Path(args.cluster_exe), base, tmp / "bs")
    res = subprocess.run([args.cluster_exe, "dynamic", str(base), str(stream),
                          str(tmp / "dc"), "--threshold", f"{thr:.17g}",
                          "--local-bucket-refresh", "--verify-dynamic"],
                         capture_output=True, text=True)
    dc_ok = res.returncode == 0
    dlog = pd.read_csv(tmp / "dc" / "dynamic_log.csv")
    checked = int(dlog.verification_checked.sum()) if "verification_checked" in dlog else len(dlog)
    mism = int(dlog.verification_mismatches.sum())
    fallb = int(dlog.local_relabel_fallback.sum())

    summary = pd.DataFrame([{**st, "eps": eps, "frozen_threshold": thr,
                             "unionfind_final_ari": float(a.iloc[-1]),
                             "unionfind_min_ari": float(a.min()),
                             "delaucluster_ok": dc_ok,
                             "delaucluster_log_rows": len(dlog),
                             "delaucluster_verified": checked,
                             "delaucluster_mismatches": mism,
                             "delaucluster_fallbacks": fallb}])
    summary.to_csv(out / "ais_moves_summary.csv", index=False)
    uf.to_csv(out / "ais_unionfind_ari.csv", index=False)
    print("\n=== AIS GENUINE-MOVE STREAM SUMMARY ===")
    print(summary.T.to_string())
    if not dc_ok:
        print("\nDELAUCLUSTER REPLAY FAILED (verifier abort?):")
        print(res.stderr[-1500:])
        raise SystemExit(1)
    print(f"\nunion-find final ARI {float(a.iloc[-1]):.3f} (min {float(a.min()):.3f}); "
          f"DelauCluster exact: {checked} verified, {mism} mismatches, {fallb} fallbacks "
          f"over {st['ops']} real ops")


if __name__ == "__main__":
    main()
