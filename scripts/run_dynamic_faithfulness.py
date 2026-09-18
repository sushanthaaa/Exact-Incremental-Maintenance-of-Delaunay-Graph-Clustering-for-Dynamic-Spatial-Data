#!/usr/bin/env python3
"""Experiment 1 (core): incremental faithfulness under arbitrary delete/move.

On a delete/move-heavy stream we measure, per step, how faithfully each method's
incrementally-maintained clustering matches a FROM-SCRATCH run of the SAME method
on the currently-active points (ARI):

  * Incremental DBSCAN (fast grid + union-find path): grow-only, so it cannot
    un-merge / demote cores on delete -- its ARI DRIFTS below 1.0 as deletes/moves
    accumulate. (Insert-only streams give ARI 1.0, so this is not a strawman.)
  * DelauCluster: exact fully-dynamic maintenance -- its built-in 3-oracle verifier
    reports 0 mismatches vs from-scratch on every step, i.e. ARI == 1.0.

Both methods process the IDENTICAL stream. Outputs:
  * faithfulness_summary.csv + figures/dynamic_faithfulness.* : per-step ARI,
    mean +/- std across seeds, baseline drifting vs DelauCluster flat at 1.0.
  * drift_vs_rate.csv + figures/drift_vs_rate.* : final ARI as a function of the
    delete/move (correction) rate -- shows drift grows with correction intensity.

Figures are produced to IEEE (ICTAI 2026 / IEEE Xplore) graphics specifications;
see the IEEE-styling block below for the exact compliance choices.
"""

from __future__ import annotations

import argparse
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
from generate_datasets import build  # noqa: E402


# ============================================================================
# IEEE-compliant figure styling (shared by the figure scripts in this project)
# ----------------------------------------------------------------------------
# Why these values: ICTAI 2026 mandates the IEEEtran[conference] template and
# publishes to IEEE Xplore, so figures must meet IEEE's graphics rules --
# vector line art, >=600 dpi if rasterised, 8-10 pt text, >=0.5 pt lines,
# EMBEDDED (non-Type-3) fonts, and colour + shape redundancy so the figure is
# still readable in grey-scale and for colour-vision-deficient (CVD) readers.
#
# Both paper figures are included at width=0.98\columnwidth, so we BUILD them at
# that exact physical width. The LaTeX scale factor is then ~1.0, which means
# the matplotlib point sizes below render as TRUE points in the final PDF
# (8 pt stays 8 pt) -- no oversize-then-shrink guesswork, the failure mode of
# the "make it 10in @ 20pt and let LaTeX shrink it" approach.
# ============================================================================
IEEE_COLUMN_WIDTH_IN = 3.5                   # IEEEtran[conference] \columnwidth (~3.5 in)
IEEE_FIG_W = 0.98 * IEEE_COLUMN_WIDTH_IN     # 3.43 in == \includegraphics width in the paper
IEEE_FIG_H = 2.6                             # in (a touch taller to seat the compact top legend)

# Okabe-Ito colour-blind-safe palette; each series ALSO gets a distinct line
# style and marker so every figure survives a grey-scale printout and CVD.
IEEE_SERIES = [
    {"color": "#0072B2", "linestyle": "-",               "marker": "o"},  # blue
    {"color": "#D55E00", "linestyle": "--",              "marker": "s"},  # vermillion
    {"color": "#009E73", "linestyle": "-.",              "marker": "^"},  # green
    {"color": "#CC79A7", "linestyle": ":",               "marker": "D"},  # purple
    {"color": "#E69F00", "linestyle": (0, (3, 1, 1, 1)), "marker": "v"},  # orange
]
IEEE_EXACT_COLOR = "#D62728"        # reference line for the exact method (DelauCluster)
IEEE_EXACT_DASH = (0, (4, 2))       # its unique dash pattern, distinct from the series


def set_ieee_style():
    """Apply IEEE-compliant matplotlib rcParams. Call once before plotting.

    Compliance highlights:
      * pdf/ps.fonttype = 42 -> embed TrueType, NEVER Type 3. IEEE PDF eXpress
        rejects Type 3 fonts (and they silently mangle maths/axis labels).
      * serif family (Times-like) + STIX maths to match the IEEEtran body text;
        falls back to DejaVu Serif if Times New Roman is not installed (still a
        TrueType embed, so still compliant -- just not exactly Times).
      * 8 pt base / 7 pt ticks -> true IEEE 8-10 pt figure text at column width.
      * every line / spine / tick >= 0.5 pt -> no hairlines that drop out in print.

    To use Arial/Helvetica instead (also IEEE-accepted), change three lines:
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # --- font embedding: TrueType only, no Type 3 (hard IEEE requirement) ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,           # use mathtext; no external LaTeX needed
        "mathtext.fontset": "stix",     # Times-like maths to match IEEEtran body
        # --- fonts: serif (Times-like) to match the IEEE conference body text ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L",
                       "Nimbus Roman", "DejaVu Serif"],
        # --- true point sizes (rendered 1:1 because we build at column width) ---
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # --- line / spine / tick widths >= 0.5 pt (no hairlines) ---
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        # --- clean, exactly-sized output ---
        "legend.frameon": False,
        "legend.handlelength": 2.4,     # enough to show each dashed/dotted style
        "savefig.bbox": "standard",     # keep the EXACT figure size (do NOT trim)
        "savefig.pad_inches": 0.01,
        "savefig.dpi": 600,             # 600 dpi for any rasterised content / PNG
    })


def save_ieee(fig, stem):
    """Save the IEEE way: vector PDF (include/submit this) + 600-dpi PNG preview.

    IEEE accepts PDF/EPS/PS/PNG/TIFF (NOT SVG) and wants vector line art (or
    >=600 dpi rasters). A vector PDF satisfies that at any zoom and is exactly
    what \\includegraphics{...pdf} loads. After compiling, verify embedding with:
        pdffonts faithfulness_synthetic.pdf   # no font should be 'Type 3'
    """
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"))             # vector -> goes into the paper
    fig.savefig(stem.with_suffix(".png"), dpi=600)    # 600-dpi raster preview/slides


def ieee_top_legend(fig, ax, ncol=2, fontsize=7):
    """Compact, frameless legend in a block ABOVE the axes (the BeautifulFigures
    house style). Uses loc='outside upper center', so constrained_layout reserves
    the strip of space and the legend is never clipped.

    Why a 2-column block and not a single row: with 4 descriptive entries in a
    3.5 in column a single row overflows the figure width (measured ~110-140%);
    a 2-column block fits (~70%) and stays readable at 7 pt. Labels are kept
    short on purpose -- the method context (incremental DBSCAN vs DelauCluster)
    lives in the LaTeX \\caption, exactly as in the reference figure that uses
    bare class names (Setosa/Virginica) rather than restating the model.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol,
               frameon=False, fontsize=fontsize, handlelength=1.8,
               handletextpad=0.5, columnspacing=1.1)


def resize_points(x, y, target, seed):
    if target <= 0 or target == len(x):
        return x.copy(), y.copy()
    rng = np.random.default_rng(seed)
    if target < len(x):
        idx = rng.choice(len(x), size=target, replace=False)
        return x[idx].copy(), y[idx].copy()
    extra = target - len(x)
    idx = rng.integers(0, len(x), size=extra)
    jitter = np.maximum(x.std(axis=0) * 0.01, 1e-6)
    ex = x[idx] + rng.normal(0.0, jitter, size=(extra, x.shape[1]))
    return np.vstack([x, ex]), np.r_[y, y[idx]]


def make_dm_stream(x, nops, seed, insert_rate, delete_rate):
    rng = np.random.default_rng(seed)
    active = [tuple(p) for p in x]
    span = x.std(axis=0)
    lines = []
    for _ in range(nops):
        u = rng.random()
        if u < insert_rate or len(active) < 10:
            j = rng.integers(0, len(active))
            p = np.array(active[j]) + rng.normal(0, span * 0.25, size=2)
            lines.append(f"insert,{p[0]:.6f},{p[1]:.6f},0")
            active.append((float(p[0]), float(p[1])))
        elif u < insert_rate + delete_rate:
            j = rng.integers(0, len(active))
            p = active.pop(j)
            lines.append(f"delete,{p[0]:.6f},{p[1]:.6f}")
        else:
            j = rng.integers(0, len(active))
            ox, oy = active[j]
            nx = ox + float(rng.normal(0, span[0] * 0.5))
            ny = oy + float(rng.normal(0, span[1] * 0.5))
            active[j] = (nx, ny)
            lines.append(f"move,{ox:.6f},{oy:.6f},{nx:.6f},{ny:.6f}")
    return "\n".join(lines) + "\n"


def run_baseline(idbscan_exe, base, stream, eps, min_pts, every, rd):
    bout = rd / "idb.csv"
    subprocess.run([idbscan_exe, str(base), str(stream), str(bout),
                    "--eps", f"{eps:.6f}", "--min-pts", str(min_pts),
                    "--every", str(every)], check=True, capture_output=True)
    bdf = pd.read_csv(bout)
    return bdf[bdf.incr_ari > -2].copy()


def run_delaucluster_verified(cluster_exe, base, stream, rd):
    thr = fit_base_threshold(Path(cluster_exe), base, rd / "base_static")
    dout = rd / "delau"
    res = subprocess.run([cluster_exe, "dynamic", str(base), str(stream), str(dout),
                          "--threshold", f"{thr:.17g}", "--local-bucket-refresh",
                          "--verify-dynamic"], capture_output=True, text=True)
    ok = res.returncode == 0
    mism, n = -1, 0
    if (dout / "dynamic_log.csv").exists():
        dl = pd.read_csv(dout / "dynamic_log.csv")
        n = len(dl)
        if "verification_mismatches" in dl:
            mism = int(dl["verification_mismatches"].sum())
    return ok, mism, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="chain_noise,varying_density,touching")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ops", type=int, default=1000)
    ap.add_argument("--insert-rate", type=float, default=0.25)
    ap.add_argument("--delete-rate", type=float, default=0.50)
    ap.add_argument("--rate-sweep", default="0.2,0.35,0.5")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--cluster-exe", default="./build/cluster")
    ap.add_argument("--idbscan-exe", default="./build/incremental_dbscan_stream")
    ap.add_argument("--out-dir", default="results/dynamic_faithfulness")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="faith_"))

    # ---- Main multi-seed faithfulness run (headline rate) ----
    agg = {}   # dataset -> DataFrame indexed by step, columns = seeds
    rows = []
    for ds in datasets:
        per_seed = {}
        total_mism, total_updates, ok_count = 0, 0, 0
        for seed in seeds:
            x, y = build(ds, seed)
            x, y = resize_points(x, y, args.n, seed + 2000)
            rd = tmp / f"{ds}_s{seed}"
            rd.mkdir(parents=True, exist_ok=True)
            base, stream = rd / "base.csv", rd / "stream.csv"
            write_points(base, x, y)
            stream.write_text(make_dm_stream(x, args.ops, seed + 1000,
                                             args.insert_rate, args.delete_rate))
            eps = float(estimate_eps(x, args.min_pts))
            bdf = run_baseline(args.idbscan_exe, base, stream, eps, args.min_pts, args.every, rd)
            per_seed[seed] = bdf.set_index("step")["incr_ari"]
            ok, mism, n = run_delaucluster_verified(args.cluster_exe, base, stream, rd)
            ok_count += int(ok); total_mism += max(mism, 0); total_updates += n
        mat = pd.DataFrame(per_seed)  # index=step, cols=seeds
        agg[ds] = mat
        finals = mat.iloc[-1].values
        rows.append({
            "dataset": ds, "n_base": args.n, "ops": args.ops, "seeds": len(seeds),
            "insert_rate": args.insert_rate, "delete_rate": args.delete_rate,
            "baseline_final_ari_mean": float(np.mean(finals)),
            "baseline_final_ari_std": float(np.std(finals)),
            "baseline_min_ari_mean": float(mat.min().mean()),
            "delaucluster_verified_runs_ok": ok_count,
            "delaucluster_total_updates": total_updates,
            "delaucluster_total_mismatches": total_mism,
            "delaucluster_ari": 1.0 if (ok_count == len(seeds) and total_mism == 0) else float("nan"),
        })
        print(f"[{ds}] baseline final ARI {np.mean(finals):.3f}+/-{np.std(finals):.3f} "
              f"| DelauCluster {ok_count}/{len(seeds)} runs exact, {total_mism} mismatches / {total_updates} updates")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "faithfulness_summary.csv", index=False)

    # ---- Drift vs correction-rate sweep (baseline only; DelauCluster stays 1.0) ----
    sweep_rates = [float(r) for r in args.rate_sweep.split(",") if r.strip()]
    sweep_rows = []
    for ds in datasets:
        for d_rate in sweep_rates:
            i_rate = 0.5 * (1.0 - d_rate)  # insert==move; rest deletes
            finals = []
            for seed in seeds:
                x, y = build(ds, seed)
                x, y = resize_points(x, y, args.n, seed + 2000)
                rd = tmp / f"sweep_{ds}_{int(d_rate*100)}_s{seed}"
                rd.mkdir(parents=True, exist_ok=True)
                base, stream = rd / "base.csv", rd / "stream.csv"
                write_points(base, x, y)
                stream.write_text(make_dm_stream(x, args.ops, seed + 1000, i_rate, d_rate))
                eps = float(estimate_eps(x, args.min_pts))
                bdf = run_baseline(args.idbscan_exe, base, stream, eps, args.min_pts, args.every, rd)
                finals.append(float(bdf.iloc[-1]["incr_ari"]))
            sweep_rows.append({"dataset": ds, "delete_rate": d_rate,
                               "baseline_final_ari_mean": float(np.mean(finals)),
                               "baseline_final_ari_std": float(np.std(finals))})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "drift_vs_rate.csv", index=False)

    # ---- Figures (IEEE-compliant: vector PDF, true 8 pt text, single column) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        set_ieee_style()

        # Fig. 1 (paper fig:faith). Saved under the paper's figure name so it
        # copies straight into paper/figures/ with no manual rename.
        fig, ax = plt.subplots(figsize=(IEEE_FIG_W, IEEE_FIG_H),
                               constrained_layout=True)
        for (ds, mat), st in zip(agg.items(), IEEE_SERIES):
            steps = mat.index.values
            m, s = mat.mean(axis=1).values, mat.std(axis=1).values
            mev = max(1, len(steps) // 6)  # ~6 markers/curve: grey-scale-distinct, uncluttered
            ax.plot(steps, m, color=st["color"], ls=st["linestyle"],
                    marker=st["marker"], markevery=mev, ms=3.0, lw=1.3,
                    label=ds)  # bare dataset name; "Incr. DBSCAN" context is in the caption
            ax.fill_between(steps, m - s, m + s, color=st["color"], alpha=0.15,
                            linewidth=0)
        ax.axhline(1.0, color=IEEE_EXACT_COLOR, lw=1.6, ls=IEEE_EXACT_DASH,
                   label="DelauCluster (exact)")
        ax.set_xlabel("stream operations processed (insert/delete/move)")
        ax.set_ylabel("ARI vs from-scratch (same method)")
        lo = min(float((mat.mean(axis=1) - mat.std(axis=1)).min())
                 for mat in agg.values())
        ax.set_ylim(max(0.0, lo - 0.05), 1.02)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        ieee_top_legend(fig, ax)  # compact, frameless 2-col legend above the axes (caption is the title)
        save_ieee(fig, out / "figures" / "faithfulness_synthetic")
        plt.close(fig)

        # Auxiliary: drift vs correction rate. Not referenced in the current
        # paper; kept for the supplement, in the same single-column IEEE style.
        fig2, ax2 = plt.subplots(figsize=(IEEE_FIG_W, IEEE_FIG_H),
                                 constrained_layout=True)
        for ds, st in zip(datasets, IEEE_SERIES):
            sub = sweep[sweep.dataset == ds].sort_values("delete_rate")
            ax2.errorbar(sub.delete_rate, sub.baseline_final_ari_mean,
                         yerr=sub.baseline_final_ari_std,
                         color=st["color"], ls=st["linestyle"], marker=st["marker"],
                         ms=3.5, lw=1.3, capsize=2, capthick=0.6, elinewidth=0.6,
                         label=ds)
        ax2.axhline(1.0, color=IEEE_EXACT_COLOR, lw=1.6, ls=IEEE_EXACT_DASH,
                    label="DelauCluster")
        ax2.set_xlabel("delete/correction rate")
        ax2.set_ylabel("final ARI vs from-scratch")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, which="major", linewidth=0.4, alpha=0.25)
        ax2.set_axisbelow(True)
        ieee_top_legend(fig2, ax2)  # compact, frameless 2-col legend above the axes
        save_ieee(fig2, out / "figures" / "drift_vs_rate")
        plt.close(fig2)
        print(f"wrote figures to {out/'figures'}")
    except Exception as e:
        print(f"figure skipped: {e}")

    print("\n=== MAIN SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== DRIFT VS RATE ===")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()