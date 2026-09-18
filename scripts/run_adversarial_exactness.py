#!/usr/bin/env python3
"""Targeted adversarial exactness stress tests for the guarded dynamic update.

The bulk faithfulness runs replay "natural" streams. This script instead builds
configurations chosen to break the local relabel and its soundness guard, and
verifies each update against a from-scratch fit (`--verify-dynamic`, three
oracles: retained edges, spatial-index registry, point partition). The C++ tool
aborts with a non-zero exit on the first mismatch, so a case passes only if the
run completes AND every mismatch column is zero.

Cases (paper Sec. IV-A):
  A  high-degree hub deletion      - delete a vertex of very high Delaunay degree
  B  bridge-point deletion         - break the only chain joining two blobs (split)
  C  move-induced merge then split - bridge two blobs with a move, then retract
  D  ceil(log2 n) noise boundary   - cross n=16<->17 so the noise cutoff shifts and
                                     a size-4 component flips cluster<->noise globally
  E  near-cocircular stress        - inserts/moves on a near-common circle
  F  threshold-boundary edges      - moves that flip edges across the frozen tau
  G  adversarial mixed ordering    - bridge-prone layout, heavy random ins/del/move

Writes results/adversarial_exactness/adversarial_exactness.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CLUSTER = "./build/cluster"


def write_points(path: Path, pts: np.ndarray) -> None:
    with open(path, "w") as fh:
        for x, y in pts:
            fh.write(f"{x:.10f},{y:.10f}\n")


def blob(cx, cy, n, s, rng):
    return np.column_stack([rng.normal(cx, s, n), rng.normal(cy, s, n)])


def run_case(name, base, ops, out_dir, cluster_exe):
    """Run one adversarial stream under the verifier; return a summary row."""
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    bpath, spath = case_dir / "base.csv", case_dir / "stream.csv"
    write_points(bpath, base)
    spath.write_text("\n".join(ops) + "\n")

    proc = subprocess.run(
        [cluster_exe, "dynamic", str(bpath), str(spath), str(case_dir / "out"),
         "--local-bucket-refresh", "--verify-dynamic"],
        capture_output=True, text=True,
    )
    log = case_dir / "out" / "dynamic_log.csv"
    vm = erm = brm = fb = 0
    n_ops = 0
    if log.exists():
        d = pd.read_csv(log)
        n_ops = len(d)
        vm = int(d.verification_mismatches.sum())
        erm = int(d.edge_refresh_mismatches.sum())
        brm = int(d.bucket_refresh_mismatches.sum())
        fb = int(d.local_relabel_fallback.sum())
    passed = proc.returncode == 0 and vm == 0 and erm == 0 and brm == 0
    print(f"  {name:28s} ops={n_ops:4d} fallback={fb:3d} "
          f"mismatch(part/edge/bucket)={vm}/{erm}/{brm} exit={proc.returncode} "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed and proc.stderr.strip():
        print("     stderr:", proc.stderr.strip()[:300])
    return {"case": name, "verified_updates": n_ops, "fallbacks": fb,
            "partition_mismatches": vm, "edge_mismatches": erm,
            "bucket_mismatches": brm, "exit_code": proc.returncode,
            "passed": int(passed)}


def build_cases(rng):
    cases = []

    # A. high-degree hub: one vertex adjacent to every point on a ring.
    ang = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    hub = np.vstack([[0, 0]] + [[math.cos(a), math.sin(a)] for a in ang])
    base = np.vstack([hub, blob(8, 8, 18, 0.25, rng)])
    ops = ["delete,0,0"] + [f"delete,{math.cos(a):.6f},{math.sin(a):.6f}" for a in ang[:6]]
    cases.append(("A_high_degree_hub_delete", base, ops))

    # B. bridge-point deletion: chain joining two blobs, broken middle-out.
    bridge = np.array([[x, 0.0] for x in np.linspace(0.5, 3.5, 9)])
    base = np.vstack([blob(0, 0, 16, 0.18, rng), bridge, blob(4, 0, 16, 0.18, rng)])
    ops = [f"delete,{x:.4f},0" for x in [2.0, 1.5, 2.5, 1.0, 3.0, 0.5, 3.5]]
    cases.append(("B_bridge_delete_split", base, ops))

    # C. move that merges two clusters, then retracts (split).
    base = np.vstack([blob(0, 0, 16, 0.18, rng), blob(3, 0, 16, 0.18, rng),
                      np.array([[0.2, 0.0]])])
    ops = [f"move,{a:.3f},0,{b:.3f},0"
           for a, b in [(0.2, 1.5), (1.5, 2.8), (2.8, 1.5), (1.5, 0.2)]]
    cases.append(("C_move_merge_then_split", base, ops))

    # D. cross the ceil(log2 n) noise boundary (n=16<->17): a size-4 component
    #    must flip cluster<->noise globally, far from the edited region.
    tetrad = np.array([[10, 10], [10.2, 10], [10, 10.2], [10.2, 10.2]])
    base = np.vstack([blob(0, 0, 12, 0.25, rng), tetrad])  # 16 active points
    ops = []
    for k in range(6):
        ops.append(f"insert,{20 + k * 0.5:.3f},0.000")   # 16 -> 17, cutoff 4 -> 5
        ops.append(f"delete,{20 + k * 0.5:.3f},0.000")   # 17 -> 16, cutoff 5 -> 4
    cases.append(("D_log2n_noise_boundary", base, ops))

    # E. near-cocircular stress (general position held, predicates strained).
    R, ang = 3.0, np.linspace(0, 2 * math.pi, 20, endpoint=False)
    ring = np.array([[R * math.cos(a) + rng.normal(0, 1e-4),
                      R * math.sin(a) + rng.normal(0, 1e-4)] for a in ang])
    base = np.vstack([ring, blob(0, 0, 10, 0.2, rng)])
    ops = [f"insert,{R * math.cos(a) + 1e-3:.6f},{R * math.sin(a):.6f}"
           for a in np.linspace(0.1, 2.0, 6)]
    ops += [f"move,{R * math.cos(a):.6f},{R * math.sin(a):.6f},"
            f"{(R + 0.05) * math.cos(a):.6f},{(R + 0.05) * math.sin(a):.6f}" for a in ang[:5]]
    cases.append(("E_near_cocircular_stress", base, ops))

    # F. edges sitting near the frozen threshold; nudge them across it.
    xs = np.concatenate([np.linspace(0, 1, 8), np.linspace(1.4, 2.4, 8), [3.2, 3.35, 3.5]])
    base = np.array([[x, rng.normal(0, 0.05)] for x in xs])
    ops = [f"move,{x:.4f},0,{x + dx:.4f},{dy:.4f}" for x, dx, dy in
           [(1.0, 0.2, 0.0), (1.4, -0.2, 0.0), (3.2, -0.3, 0.0),
            (3.5, 0.3, 0.0), (1.2, 0.1, 0.05), (2.4, 0.3, 0.0)]]
    cases.append(("F_threshold_boundary_edges", base, ops))

    # G. adversarial mixed ordering on a bridge-prone four-blob layout.
    r2 = np.random.default_rng(123)
    pts = np.vstack([blob(cx, cy, 14, 0.2, r2) for cx, cy in [(0, 0), (3, 0), (0, 3), (3, 3)]])
    base, active, ops = pts.copy(), list(map(tuple, pts)), []
    for _ in range(120):
        r = r2.random()
        if r < 0.34 and len(active) > 8:
            x, y = active.pop(r2.integers(len(active)))
            ops.append(f"delete,{x:.5f},{y:.5f}")
        elif r < 0.67:
            cx, cy = [(1.5, 0), (0, 1.5), (1.5, 1.5), (3, 1.5)][r2.integers(4)]
            x, y = cx + r2.normal(0, 0.3), cy + r2.normal(0, 0.3)
            active.append((x, y))
            ops.append(f"insert,{x:.5f},{y:.5f}")
        elif active:
            x, y = active[r2.integers(len(active))]
            ops.append(f"move,{x:.5f},{y:.5f},{x + r2.normal(0, 0.4):.5f},{y + r2.normal(0, 0.4):.5f}")
    cases.append(("G_adversarial_mixed_order", base, ops))

    # H. EXACTLY-cocircular configurations (outside the theorem's general-
    # position assumption; run to DOCUMENT behavior at the boundary, not to
    # corroborate the theorem). All coordinates are exactly representable
    # doubles and exactly cocircular: (+-10,0),(0,+-10) and the Pythagorean
    # points (+-6,+-8),(+-8,+-6) lie on the radius-10 circle; the axis-aligned
    # square (18,0),(20,0),(18,2),(20,2) is a second cocircular 4-tuple.
    r3 = np.random.default_rng(31)
    anchor = blob(30.0, 30.0, 20, 0.5, r3)
    circle4 = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -10.0]])
    square4 = np.array([[18.0, 0.0], [20.0, 0.0], [18.0, 2.0], [20.0, 2.0]])
    base = np.vstack([anchor, circle4, square4])
    ops = [
        "insert,6,8",        # 5 cocircular
        "insert,-6,8",       # 6 cocircular
        "insert,8,-6",       # 7 cocircular
        "delete,10,0",       # remove an original cocircular vertex
        "insert,10,0",       # re-add it
        "move,8,-6,-8,-6",   # exact move ALONG the circle
        "delete,0,10",
        "move,6,8,6.5,8.25", # off the circle
        "insert,0,10",
        "delete,-6,8",
        "insert,19,1",       # center of the square's circumcircle
        "delete,20,2",       # remove a cocircular square vertex
    ]
    cases.append(("H_exactly_cocircular", base, ops))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/adversarial_exactness")
    ap.add_argument("--cluster-exe", default=CLUSTER)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=== adversarial exactness (each update verified vs a from-scratch fit) ===")
    rows = [run_case(name, base, ops, out, args.cluster_exe)
            for name, base, ops in build_cases(rng)]

    csv_path = out / "adversarial_exactness.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = sum(r["verified_updates"] for r in rows)
    mism = sum(r["partition_mismatches"] + r["edge_mismatches"] + r["bucket_mismatches"]
               for r in rows)
    all_pass = all(r["passed"] for r in rows)
    print(f"\ntotal verified updates = {total}; total mismatches = {mism}; "
          f"all cases passed = {all_pass}")
    print(f"wrote {csv_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
