#!/usr/bin/env python3
"""Spatial before/after of a real declustering split (paper Fig. 1).

Makes the abstract "delete a bridge point and a cluster splits" concrete on a
GENUINE USGS ComCat sub-sequence. We take one real DelauCluster component from
the Southern-California catalog (a dense cluster held together by aftershock
"glue"), strip the Gardner-Knopoff (1974) aftershocks, and re-cluster the
surviving mainshocks at the SAME frozen threshold:

  (a) before declustering : all events form ONE Delaunay cluster (the dense
      aftershocks bridge two seismicity patches);
  (b) DelauCluster, exact  : removing the aftershock glue disconnects the
      component, so the mainshocks split into TWO clusters (a from-scratch fit
      at the frozen tau == what the exact maintainer produces);
  (c) grow-only baseline   : a union-find incremental method never un-merges,
      so the survivors keep their single pre-deletion id and stay merged (wrong).

The cluster is a real connected component of the catalog (cropped for legibility
to that one component, as seismic sequences are routinely analysed regionally);
the threshold is DelauCluster's own auto-selected tau on the sub-sequence and the
split is robust to it (the same two clusters appear for tau, 1.12 tau, 1.25 tau).

Figures follow the shared IEEE styling block in run_dynamic_faithfulness.py
(TrueType-embedded fonts, CVD-safe colour + redundant marker shapes, vector PDF
plus 600-dpi PNG), built at the exact paper column width.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dynamic_stream import write_points  # noqa: E402
from run_dynamic_faithfulness import (  # noqa: E402
    IEEE_FIG_W,
    save_ieee,
    set_ieee_style,
)

CACHE = ROOT / "data" / "comcat_revision" / "socal_decluster.csv"

# Okabe-Ito CVD-safe colours; each clustering state ALSO gets a distinct marker
# so the figure survives a grey-scale print and colour-vision deficiency.
C_MERGED = "#444444"   # one cluster (before, and the grow-only baseline after)
C_AFTER1 = "#0072B2"   # exact split, cluster 1
C_AFTER2 = "#D55E00"   # exact split, cluster 2
C_GLUE = "#9C9C9C"     # Gardner-Knopoff aftershocks (the bridge that is deleted)
M_MERGED, M_AFTER1, M_AFTER2, M_GLUE = "o", "o", "^", "x"


def gardner_knopoff(df: pd.DataFrame) -> np.ndarray:
    """Window declustering (Gardner & Knopoff 1974); boolean aftershock mask."""
    lat0 = float(df.lat.mean())
    lon = df.lon.to_numpy(); lat = df.lat.to_numpy()
    t = df.time_ms.to_numpy(); M = df.mag.to_numpy(); n = len(df)
    def km(i, j):
        return 111.0 * math.hypot(lat[i] - lat[j], (lon[i] - lon[j]) * math.cos(math.radians(lat0)))
    def L(m):
        return 10 ** (0.1238 * m + 0.983)
    def T(m):
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


def project_km(df: pd.DataFrame) -> np.ndarray:
    lat0 = float(df.lat.mean())
    x = (df.lon - df.lon.mean()) * 111.0 * math.cos(math.radians(lat0))
    y = (df.lat - df.lat.mean()) * 111.0
    return np.c_[x.to_numpy(), y.to_numpy()]


def static_labels(cluster_exe: str, P: np.ndarray, thr: float, tmp: Path, name: str) -> np.ndarray:
    csv = tmp / f"{name}.csv"
    write_points(csv, P, np.zeros(len(P), dtype=int))
    out = tmp / name
    subprocess.run([cluster_exe, "static", str(csv), str(out), "--threshold", f"{thr:.17g}"],
                   check=True, capture_output=True)
    return pd.read_csv(out / "clusters.csv")["cluster_id"].to_numpy(int)


def fit_threshold(cluster_exe: str, P: np.ndarray, tmp: Path, name: str) -> float:
    csv = tmp / f"{name}.csv"
    write_points(csv, P, np.zeros(len(P), dtype=int))
    subprocess.run([cluster_exe, "static", str(csv), str(tmp / f"{name}_fit")],
                   check=True, capture_output=True)
    md = pd.read_csv(tmp / f"{name}_fit" / "metadata.csv", header=None, names=["k", "v"])
    return float(md.loc[md.k == "threshold", "v"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    # default window = one real SoCal component that splits cleanly on declustering
    ap.add_argument("--x0", type=float, default=145.0)
    ap.add_argument("--x1", type=float, default=255.0)
    ap.add_argument("--y0", type=float, default=-280.0)
    ap.add_argument("--y1", type=float, default=-170.0)
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--out-dir", default="results/decluster")
    args = ap.parse_args()

    df = pd.read_csv(CACHE)
    aft = gardner_knopoff(df)
    P = project_km(df)

    tmp = Path(tempfile.mkdtemp(prefix="decluster_spatial_"))
    sel = (P[:, 0] >= args.x0) & (P[:, 0] <= args.x1) & (P[:, 1] >= args.y0) & (P[:, 1] <= args.y1)
    Pw, aw = P[sel], aft[sel]
    thr = fit_threshold(args.cluster_exe, Pw, tmp, "window")
    before = static_labels(args.cluster_exe, Pw, thr, tmp, "before")

    # the dominant component (by surviving-mainshock count) is the cluster we track
    dom = Counter(before[~aw][before[~aw] >= 0]).most_common(1)[0][0]
    inC = before == dom
    Pc, awc = Pw[inC], aw[inC]
    pm = Pc[~awc]                                   # surviving mainshocks
    after = static_labels(args.cluster_exe, pm, thr, tmp, "after")
    order = [c for c, _ in Counter(after[after >= 0]).most_common()]

    print(f"window x[{args.x0:.0f},{args.x1:.0f}] y[{args.y0:.0f},{args.y1:.0f}]  frozen tau={thr:.4f}")
    print(f"tracked component: {inC.sum()} events = {len(pm)} mainshocks + {int(awc.sum())} GK aftershocks")
    print(f"after declustering -> {len(order)} clusters {[int((after==c).sum()) for c in order]}, "
          f"noise {int((after<0).sum())}")

    # ----- IEEE figure: 3 panels at the exact paper column width --------------
    set_ieee_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axs = plt.subplots(1, 3, figsize=(IEEE_FIG_W, 1.7), sharex=True, sharey=True,
                            constrained_layout=True)

    def style(ax, title):
        ax.set_title(title, fontsize=8, pad=2)
        # square panel box (not equal data aspect): the data y-range is ~2x the
        # x-range, so a square box spreads the points out and keeps them legible.
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=6, length=2)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("x (km)", fontsize=7, labelpad=1)

    # (a) before: one cluster + aftershock glue (the bridge that gets deleted)
    axs[0].scatter(Pc[awc, 0], Pc[awc, 1], s=12, c=C_GLUE, marker=M_GLUE, linewidths=0.7)
    axs[0].scatter(pm[:, 0], pm[:, 1], s=9, c=C_MERGED, marker=M_MERGED, linewidths=0)
    style(axs[0], "(a) before")
    axs[0].set_ylabel("y (km)", fontsize=7, labelpad=1)

    # (b) after, DelauCluster exact: the split
    for c, col, mk in zip(order, (C_AFTER1, C_AFTER2), (M_AFTER1, M_AFTER2)):
        m = after == c
        axs[1].scatter(pm[m, 0], pm[m, 1], s=9, c=col, marker=mk, linewidths=0)
    style(axs[1], "(b) exact")

    # (c) after, grow-only baseline: stays merged
    axs[2].scatter(pm[:, 0], pm[:, 1], s=9, c=C_MERGED, marker=M_MERGED, linewidths=0)
    style(axs[2], "(c) grow-only")

    handles = [
        Line2D([], [], color=C_MERGED, marker=M_MERGED, ls="none", ms=3, label="one cluster"),
        Line2D([], [], color=C_GLUE, marker=M_GLUE, ls="none", ms=3, label="aftershock glue"),
        Line2D([], [], color=C_AFTER1, marker=M_AFTER1, ls="none", ms=3, label="cluster 1"),
        Line2D([], [], color=C_AFTER2, marker=M_AFTER2, ls="none", ms=3, label="cluster 2"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               fontsize=6.5, handlelength=1.2, handletextpad=0.3, columnspacing=0.9)

    outdir = Path(args.out_dir) / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    save_ieee(fig, outdir / "spatial_split")
    print(f"wrote {outdir/'spatial_split.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
