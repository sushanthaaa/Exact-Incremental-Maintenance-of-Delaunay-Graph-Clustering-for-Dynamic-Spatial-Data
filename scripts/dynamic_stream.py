#!/usr/bin/env python3
"""Generate synthetic dynamic streams and summarize DelauCluster latency."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from generate_datasets import build


def metadata_value(path: Path, key: str) -> float:
    metadata = pd.read_csv(path, header=None, names=["key", "value"])
    matches = metadata.loc[metadata["key"] == key, "value"]
    if matches.empty:
        raise ValueError(f"missing metadata key {key}: {path}")
    return float(matches.iloc[0])


def fit_base_threshold(cluster_exe: Path, base_csv: Path, out_dir: Path) -> float:
    subprocess.run([str(cluster_exe), "static", str(base_csv), str(out_dir)], check=True)
    return metadata_value(out_dir / "metadata.csv", "threshold")


def write_points(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y.astype(int)}).to_csv(
        path, index=False, header=False
    )


def make_stream(
    x: np.ndarray,
    y: np.ndarray,
    ops: int,
    seed: int,
    insert_rate: float,
    delete_rate: float,
    step_scale: float,
) -> list[str]:
    rng = np.random.default_rng(seed)
    active_x = x.copy()
    active_y = y.copy()
    lo = np.percentile(active_x, 5, axis=0)
    hi = np.percentile(active_x, 95, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    rows: list[str] = ["operation,x,y,label"]

    for _ in range(ops):
        r = rng.random()
        if r < insert_rate:
            center_idx = int(rng.integers(0, len(active_x)))
            new_xy = active_x[center_idx] + rng.normal(0.0, step_scale, size=2)
            new_xy = np.minimum(np.maximum(new_xy, lo), hi)
            label = int(active_y[center_idx]) if len(active_y) else -1
            rows.append(f"insert,{new_xy[0]:.17g},{new_xy[1]:.17g},{label}")
            active_x = np.vstack([active_x, new_xy])
            active_y = np.r_[active_y, label]
        elif r < insert_rate + delete_rate and len(active_x) > 10:
            idx = int(rng.integers(0, len(active_x)))
            xy = active_x[idx]
            rows.append(f"delete,{xy[0]:.17g},{xy[1]:.17g},-1")
            active_x = np.delete(active_x, idx, axis=0)
            active_y = np.delete(active_y, idx)
        else:
            idx = int(rng.integers(0, len(active_x)))
            old_xy = active_x[idx]
            new_xy = old_xy + rng.normal(0.0, step_scale, size=2)
            new_xy = np.minimum(np.maximum(new_xy, lo - 0.05 * span), hi + 0.05 * span)
            rows.append(
                "move,"
                f"{old_xy[0]:.17g},{old_xy[1]:.17g},"
                f"{new_xy[0]:.17g},{new_xy[1]:.17g}"
            )
            active_x[idx] = new_xy

    return rows


def summarize_log(mode: str, dataset: str, base_points: int, log_path: Path) -> dict:
    log = pd.read_csv(log_path)
    runtime_ms = log["time_ns"] / 1_000_000.0
    return {
        "mode": mode,
        "dataset": dataset,
        "base_points": base_points,
        "ops": len(log),
        "mean_ms": runtime_ms.mean(),
        "p50_ms": runtime_ms.quantile(0.50),
        "p95_ms": runtime_ms.quantile(0.95),
        "local_bucket_rate": 1.0 - log["bucket_refresh_global"].mean()
        if "bucket_refresh_global" in log
        else 0.0,
        "mean_affected_cells": log["affected_cells"].mean(),
        "mean_edge_scope_vertices": log.get(
            "edge_refresh_scope_vertices", pd.Series([np.nan])
        ).mean(),
        "mean_relabel_scope_vertices": log.get(
            "relabel_scope_vertices", pd.Series([np.nan])
        ).mean(),
        "mean_relabel_search_vertices": log.get(
            "relabel_search_vertices", pd.Series([np.nan])
        ).mean(),
        "mean_retained_edges_removed": log.get(
            "retained_edges_removed", pd.Series([np.nan])
        ).mean(),
        "mean_retained_edges_added": log.get(
            "retained_edges_added", pd.Series([np.nan])
        ).mean(),
        "small_side_split_rate": log.get(
            "small_side_split_used", pd.Series([0])
        ).mean(),
        "point_delete_split_rate": log.get(
            "point_delete_split_used", pd.Series([0])
        ).mean(),
        "verified_ops": log.get("verification_checked", pd.Series([0])).sum(),
        "edge_refresh_mismatches": log.get(
            "edge_refresh_mismatches", pd.Series([0])
        ).sum(),
        "bucket_refresh_mismatches": log.get(
            "bucket_refresh_mismatches", pd.Series([0])
        ).sum(),
        "verification_mismatches": log.get("verification_mismatches", pd.Series([0])).sum(),
    }


def run_mode(
    mode: str,
    cluster_exe: Path,
    base_csv: Path,
    stream_csv: Path,
    out_dir: Path,
    verify: bool,
    threshold: float | None = None,
) -> dict:
    mode_dir = out_dir / mode
    cmd = [str(cluster_exe), "dynamic", str(base_csv), str(stream_csv), str(mode_dir)]
    if threshold is not None:
        cmd.extend(["--threshold", f"{threshold:.17g}"])
    if mode == "local":
        cmd.append("--local-bucket-refresh")
    if verify:
        cmd.append("--verify-dynamic")
    subprocess.run(cmd, check=True)
    base_points = len(pd.read_csv(base_csv, header=None))
    row = summarize_log(mode, out_dir.name, base_points, mode_dir / "dynamic_log.csv")
    row["base_threshold"] = threshold if threshold is not None else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chain_noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ops", type=int, default=250)
    parser.add_argument("--base-limit", type=int, default=1200)
    parser.add_argument("--insert-rate", type=float, default=0.20)
    parser.add_argument("--delete-rate", type=float, default=0.20)
    parser.add_argument("--step-scale", type=float, default=0.035)
    parser.add_argument("--out-dir", default="results/dynamic_latency")
    parser.add_argument("--cluster-exe", default="./build/cluster")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    x, y = build(args.dataset, args.seed)
    if args.base_limit > 0:
        x = x[: args.base_limit]
        y = y[: args.base_limit]

    out_dir = Path(args.out_dir) / args.dataset
    base_csv = out_dir / "base.csv"
    stream_csv = out_dir / "stream.csv"
    write_points(base_csv, x, y)
    stream_csv.write_text(
        "\n".join(
            make_stream(
                x,
                y,
                args.ops,
                args.seed + 1,
                args.insert_rate,
                args.delete_rate,
                args.step_scale,
            )
        )
        + "\n"
    )

    cluster_exe = Path(args.cluster_exe)
    threshold = fit_base_threshold(cluster_exe, base_csv, out_dir / "base_static")
    summaries = [
        run_mode(
            "full", cluster_exe, base_csv, stream_csv, out_dir, args.verify, threshold
        ),
        run_mode(
            "local", cluster_exe, base_csv, stream_csv, out_dir, args.verify, threshold
        ),
    ]
    summary_path = out_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
