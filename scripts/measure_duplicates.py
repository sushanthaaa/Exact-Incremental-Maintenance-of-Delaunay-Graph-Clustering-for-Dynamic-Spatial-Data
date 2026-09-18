#!/usr/bin/env python3
"""Coordinate-duplicate census across every real dataset/stream — R3/E4.

Measures (never assumes): exact coordinate-duplicate rows in each real
dataset as used, plus coincident-coordinate collisions at the 1e-6 stream
precision for the ComCat-derived streams. Backs the paper's
duplicate-handling paragraph (external review, Issue 5).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_comcat_revision_experiment as rcr  # noqa: E402
import run_decluster_experiment as rde  # noqa: E402


def main():
    out = Path(sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv \
        else ROOT / "results" / "duplicates"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(name, df, xc, yc, note):
        n = len(df)
        d = int(df.duplicated(subset=[xc, yc]).sum())
        rows.append({"dataset": name, "rows": n, "exact_coord_duplicate_rows": d,
                     "pct": round(100 * d / n, 3), "handling": note})

    nyc = pd.read_csv(ROOT / "data/real_spatial_large/nyc_collisions_2012_2026_large.csv",
                      header=None, names=["x", "y", "l"])
    add("NYC collisions (normalized)", nyc, "x", "y",
        "coordinate-deduplicated at normalization (generate_datasets.py drop_duplicates)")
    usw = pd.read_csv(ROOT / "data/real_spatial_large/usgs_western_us_earthquakes_1980_2026_large.csv",
                      header=None, names=["x", "y", "l"])
    add("USGS western US (normalized)", usw, "x", "y", "same normalization dedup")
    ev = pd.read_csv(ROOT / "data/comcat_revision/socal_events.csv")
    add("ComCat revision cache", ev, "lat", "lon",
        "no dedup; coincident second insert rejected as no-op by the binary")
    dc = pd.read_csv(ROOT / "data/comcat_revision/socal_decluster.csv")
    add("ComCat decluster cache", dc, "lat", "lon", "same as revision cache")
    ais = pd.read_csv(ROOT / "data/ais_lalb/ais_lalb_big.csv",
                      header=None, names=["m", "t", "lat", "lon", "s"])
    add("AIS raw reports", ais, "lat", "lon",
        "same-vessel repeats dropped by decimation; cross-vessel exact "
        "collisions blocked by the stream builder's occupied-coordinate guard "
        "(0 skipped in the built stream)")

    # stream-precision collisions (1e-6 km formatting) for the ComCat streams
    class A:
        refetch = False
        starttime = "2012-01-01"; endtime = "2020-01-01"
        lat0, lat1, lon0, lon1 = 32.0, 37.0, -121.0, -114.0; minmag = 2.5
    P = rde.project_km(rde.fetch(A()))
    c = [f"{p[0]:.6f},{p[1]:.6f}" for p in P]
    rows.append({"dataset": "decluster stream (projected, 1e-6)", "rows": len(c),
                 "exact_coord_duplicate_rows": len(c) - len(set(c)),
                 "pct": round(100 * (len(c) - len(set(c))) / len(c), 3),
                 "handling": "coincident insert -> no-op; delete -> nearest active"})

    class B:
        refetch = False
        starttime = "2018-01-01"; endtime = "2024-01-01"
        lat0, lat1, lon0, lon1 = 32.0, 37.0, -122.0, -114.0; minmag = 2.0
    df2 = rcr.project_km(rcr.load_or_fetch(B()))
    c2 = [f"{x:.6f},{y:.6f}" for x, y in zip(df2.x, df2.y)]
    rows.append({"dataset": "revision catalog (projected, 1e-6)", "rows": len(c2),
                 "exact_coord_duplicate_rows": len(c2) - len(set(c2)),
                 "pct": round(100 * (len(c2) - len(set(c2))) / len(c2), 3),
                 "handling": "same"})

    res = pd.DataFrame(rows)
    res.to_csv(out / "duplicate_census.csv", index=False)
    print(res.to_string(index=False))
    print(f"wrote {out}/duplicate_census.csv")


if __name__ == "__main__":
    main()
