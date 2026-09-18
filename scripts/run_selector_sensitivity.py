#!/usr/bin/env python3
"""One-at-a-time sensitivity of the local-scale selector's eight base weights.

Replay full-precision candidate statistics; keep the candidate grid, stability
adjustment, eligibility rules and tie handling fixed. Recompute each selected
partition with the unmodified C++ engine at an explicit threshold. This changes
no production defaults and is not a held-out quality benchmark or timing study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from dynamic_stream import write_points
from generate_datasets import SYNTHETIC_DATASETS, build

TERMS = ("coverage", "entropy", "largest", "giant_excess", "noise_excess",
         "log_clusters", "fragmentation", "fewer_than_two")


def contributions(row):
    n, k = float(row.active_points), float(row.num_clusters)
    noise = float(row.noise_count) / n
    largest = float(row.largest_cluster) / n
    return np.array([.80 * (1-noise), .35 * row.component_entropy,
                     -.45 * largest**2, -1.50 * max(0., largest-.65)**2,
                     -.75 * max(0., noise-.30)**2, -.08*np.log2(k+1),
                     -.15*k/np.sqrt(n+k), -1.50 if k < 2 else 0.])


def select(trace, base):
    nontrivial = trace.nontrivial.to_numpy().astype(bool)
    prefer_nontrivial = bool(nontrivial.any())
    pool = nontrivial if prefer_nontrivial else np.ones(len(trace), dtype=bool)
    best_base = max(base[pool])
    best = None
    for i in range(len(trace)):
        if not pool[i] or base[i] + 1e-9 < best_base:
            continue
        objective = base[i] + trace.stability_adjustment.iloc[i]
        if best is None:
            best = i
            continue
        previous = base[best] + trace.stability_adjustment.iloc[best]
        tie_preferred = (trace.threshold.iloc[i] < trace.threshold.iloc[best]
                         if prefer_nontrivial else
                         trace.threshold.iloc[i] > trace.threshold.iloc[best])
        if objective > previous or (abs(objective-previous) <= 1e-9 and tie_preferred):
            best = i
    if best is None:
        raise AssertionError("No eligible threshold")
    return best


def read_labels(folder):
    # Static export iterates active records in input order (no update/reordering).
    frame = pd.read_csv(folder / "clusters.csv", float_precision="round_trip")
    return np.where(frame.is_noise.astype(bool), -1, frame.cluster_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-exe", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    exe = args.cluster_exe.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    variants = [("default", None, 1.)] + [
        (f"{term}_{factor:.1f}", index, factor)
        for index, term in enumerate(TERMS) for factor in (.8, 1.2)]
    manifest = {"command_executable": str(exe), "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "platform": platform.platform(), "seeds": args.seeds,
                "datasets": SYNTHETIC_DATASETS, "variants": variants,
                "scope": "Eight base weights individually +/-20%; all other constants fixed; no tuning by labels."}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for dataset in SYNTHETIC_DATASETS:
        for seed in map(int, args.seeds.split(",")):
            case = args.out_dir / f"{dataset}_{seed}"
            case.mkdir()
            x, truth = build(dataset, seed)
            base_csv = case / "input.csv"
            write_points(base_csv, x, truth)
            default_dir = case / "default"
            subprocess.run([str(exe), "static", str(base_csv), str(default_dir)], check=True, capture_output=True)
            trace = pd.read_csv(default_dir / "threshold_trace.csv", float_precision="round_trip")
            terms = np.stack([contributions(row) for row in trace.itertuples()])
            base = trace.base_objective.to_numpy()
            np.testing.assert_allclose(terms.sum(axis=1), base, rtol=0, atol=2e-14)
            original = np.flatnonzero(trace.selected.to_numpy())
            assert len(original) == 1 and select(trace, base) == original[0], "Default selector replay mismatch"
            reference = read_labels(default_dir)
            assert len(reference) == len(truth)
            exported = pd.read_csv(default_dir / "clusters.csv", float_precision="round_trip")
            np.testing.assert_allclose(exported[["x", "y"]].to_numpy(), x, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(exported.label.to_numpy(), truth)
            labels_by_index = {int(original[0]): reference}
            for name, term, factor in variants:
                score = base.copy() if term is None else base + (factor-1) * terms[:, term]
                index = select(trace, score)
                tau = float(trace.threshold.iloc[index])
                if index not in labels_by_index:
                    target = case / f"candidate_{index}"
                    subprocess.run([str(exe), "static", str(base_csv), str(target), "--threshold", format(tau, ".17g")], check=True, capture_output=True)
                    labels_by_index[index] = read_labels(target)
                    candidate_points = pd.read_csv(target / "clusters.csv", float_precision="round_trip")
                    np.testing.assert_array_equal(candidate_points[["x", "y", "label"]].to_numpy(),
                                                  exported[["x", "y", "label"]].to_numpy())
                labels = labels_by_index[index]
                rows.append({"dataset": dataset, "seed": seed, "variant": name,
                             "threshold": tau, "selected_index": index,
                             "default_index": int(original[0]), "changed_threshold": index != original[0],
                             "ari_truth": adjusted_rand_score(truth, labels),
                             "ari_default": adjusted_rand_score(reference, labels),
                             "clusters": int(trace.num_clusters.iloc[index]),
                             "noise_count": int(trace.noise_count.iloc[index])})
            pd.DataFrame(rows).to_csv(args.out_dir / "runs.csv", index=False)
            print(f"completed {dataset} seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("variant", sort=False).agg(
        runs=("ari_truth", "size"), mean_ari_truth=("ari_truth", "mean"),
        mean_ari_default=("ari_default", "mean"), min_ari_default=("ari_default", "min"),
        changed_thresholds=("changed_threshold", "sum"))
    summary.to_csv(args.out_dir / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
