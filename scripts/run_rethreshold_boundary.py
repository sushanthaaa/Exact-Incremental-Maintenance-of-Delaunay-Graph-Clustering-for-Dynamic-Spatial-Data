#!/usr/bin/env python3
"""When is re-thresholding warranted? -- the frozen-tau boundary.

DelauCluster freezes the score threshold tau for a dynamic session and maintains
*that* rule exactly. A fair question is how long the frozen rule stays meaningful
as the point distribution drifts, i.e. when re-deriving tau (a global O(n) re-fit)
becomes advisable. We measure it directly: at stream checkpoints we compare the
frozen-tau partition (static fit at the base tau0) against a re-derived-tau fresh
fit on the same active points (ARI), and track how far tau0 has drifted.

  * Real USGS ComCat revision stream -- does the frozen rule stay valid?
  * Synthetic density-shift stream    -- where does it break (re-fit advisable)?
"""

from __future__ import annotations

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

from dynamic_stream import fit_base_threshold, write_points  # noqa: E402
from generate_datasets import build  # noqa: E402

CLUSTER = "./build/cluster"


def static_fit(csv_path, out_dir, tau=None):
    """Run cluster static; return (labels, tau). With tau set, the fit is frozen."""
    cmd = [CLUSTER, "static", str(csv_path), str(out_dir)]
    if tau is not None:
        cmd += ["--threshold", f"{tau:.17g}"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    t = None
    for line in res.stdout.splitlines():
        if "threshold:" in line:
            t = float(line.split(":")[1])
    lab = pd.read_csv(out_dir / "clusters.csv")["cluster_id"].to_numpy(int)
    return lab, t


def ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return 1.0
    from collections import Counter
    nij = Counter(zip(a.tolist(), b.tolist()))
    ai = Counter(a.tolist()); bj = Counter(b.tolist())
    c2 = lambda m: m * (m - 1) / 2.0
    s_ij = sum(c2(v) for v in nij.values())
    s_a = sum(c2(v) for v in ai.values()); s_b = sum(c2(v) for v in bj.values())
    cn = c2(n)
    if cn == 0:
        return 1.0
    exp = s_a * s_b / cn
    mx = 0.5 * (s_a + s_b)
    return 1.0 if mx - exp == 0 else (s_ij - exp) / (mx - exp)


def replay_active(base_pts, stream_lines, upto):
    """Active points after applying the first `upto` stream ops (nearest semantics)."""
    active = [tuple(p) for p in base_pts]
    apts = np.array(active, float)
    for line in stream_lines[:upto]:
        f = line.split(",")
        if f[0] == "insert":
            active.append((float(f[1]), float(f[2])))
        elif f[0] == "delete":
            q = np.array([float(f[1]), float(f[2])])
            arr = np.array(active)
            j = int(np.argmin(((arr - q) ** 2).sum(1)))
            active.pop(j)
        elif f[0] == "move":
            q = np.array([float(f[1]), float(f[2])])
            arr = np.array(active)
            j = int(np.argmin(((arr - q) ** 2).sum(1)))
            active[j] = (float(f[3]), float(f[4]))
    return np.array(active, float)


def boundary_curve(name, base_pts, stream_lines, tmp, checkpoints=(0.2, 0.4, 0.6, 0.8, 1.0)):
    base = tmp / f"{name}_base.csv"
    write_points(base, base_pts, np.zeros(len(base_pts), dtype=int))
    tau0 = fit_base_threshold(Path(CLUSTER), base, tmp / f"{name}_bs")
    rows = []
    for frac in checkpoints:
        k = int(frac * len(stream_lines))
        A = replay_active(base_pts, stream_lines, k)
        acsv = tmp / f"{name}_A_{k}.csv"
        write_points(acsv, A, np.zeros(len(A), dtype=int))
        lab_frozen, _ = static_fit(acsv, tmp / f"{name}_fz_{k}", tau=tau0)
        lab_fresh, tau_t = static_fit(acsv, tmp / f"{name}_fr_{k}", tau=None)
        rows.append({"stream": name, "frac": frac, "ops": k, "n_active": len(A),
                     "tau0": tau0, "tau_refit": tau_t,
                     "tau_drift_pct": 100.0 * abs(tau_t - tau0) / tau0,
                     "ari_frozen_vs_refit": ari(lab_frozen, lab_fresh)})
    return rows


def synthetic_drift_stream(seed, n_base, n_ins):
    """Base = blobs at one scale; stream inserts a progressively denser region so
    the global score distribution (and re-derived tau) drifts."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-3, 0], [3, 0], [0, 3]], float)
    base = np.vstack([c + rng.normal(0, 0.25, size=(n_base // 3, 2)) for c in centers])
    lines = []
    for i in range(n_ins):
        # dense new cluster at a 5x smaller scale, growing over the stream
        p = np.array([0.0, -3.0]) + rng.normal(0, 0.05, size=2)
        lines.append(f"insert,{p[0]:.6f},{p[1]:.6f},0")
    return base, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/rethreshold_boundary")
    ap.add_argument("--minmag", type=float, default=1.5)
    ap.add_argument("--starttime", default="2021-01-01")
    ap.add_argument("--endtime", default="2024-01-01")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="reth_"))
    rows = []

    # --- Real USGS ComCat revision stream (reuse the cached data + builder) ---
    from run_comcat_revision_experiment import load_or_fetch, project_km, build_real_stream
    rargs = argparse.Namespace(starttime=args.starttime, endtime=args.endtime,
                               lat0=32.0, lat1=37.0, lon0=-122.0, lon1=-114.0,
                               minmag=args.minmag, refetch=False)
    df = project_km(load_or_fetch(rargs))
    df = df.sort_values("time_ms").drop_duplicates(subset=["x", "y"], keep="first").reset_index(drop=True)
    base_pts, stream_text, _ = build_real_stream(df, 1500, 2500, 25)
    usgs_lines = [ln for ln in stream_text.splitlines() if ln.strip()]
    rows += boundary_curve("USGS-real", base_pts, usgs_lines, tmp)

    # --- Synthetic density-shift stream (where re-fit becomes advisable) ---
    sb, sl = synthetic_drift_stream(7, 1500, 2000)
    rows += boundary_curve("synthetic-drift", sb, sl, tmp)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / "rethreshold_boundary.csv", index=False)
    print(df_out.to_string(index=False))
    print(f"\nwrote {out/'rethreshold_boundary.csv'}")
    for name in df_out.stream.unique():
        sub = df_out[df_out.stream == name]
        fin = sub.iloc[-1]
        print(f"[{name}] final: ARI(frozen,refit)={fin.ari_frozen_vs_refit:.3f}  "
              f"tau {fin.tau0:.4f}->{fin.tau_refit:.4f} ({fin.tau_drift_pct:.1f}% drift)  "
              f"min ARI over stream={sub.ari_frozen_vs_refit.min():.3f}")


if __name__ == "__main__":
    main()
