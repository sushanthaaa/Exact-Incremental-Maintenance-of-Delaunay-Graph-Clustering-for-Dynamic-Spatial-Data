#!/usr/bin/env python3
"""Generate paper-ready external clustering baseline comparison tables."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path

_default_loky_cores = max(1, min(8, (os.cpu_count() or 2) - 1))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_default_loky_cores))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="The default value of `copy` will change.*")
warnings.filterwarnings("ignore", message="Graph is not fully connected.*")

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)

try:
    from sklearn.cluster import HDBSCAN
except Exception:  # pragma: no cover - optional sklearn feature/version
    HDBSCAN = None

from benchmark_baselines import SCHEMA, estimate_eps, score
from generate_datasets import DATASETS, build


ALGORITHM_ORDER = [
    "DelauCluster",
    "DelauClusterAuto",
    "KMeans",
    "MiniBatchKMeans",
    "DBSCAN",
    "OPTICS",
    "HDBSCAN",
    "BIRCH",
    "Agglomerative",
    "Spectral",
]
EXTRA_COLUMNS = [
    "threshold_mode",
    "seed",
    "base_points",
    "params",
    "selection_basis",
    "fits_evaluated",
    "tuning_runtime_ms",
    "status",
    "error",
    "model_dir",
]
SUMMARY_COLUMNS = SCHEMA + EXTRA_COLUMNS


def parse_csv_list(value: str) -> list[str]:
    if value == "all":
        return list(DATASETS)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def format_float(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "--"
    return f"{float(numeric):.{digits}f}"


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def resize_points(
    x: np.ndarray, y: np.ndarray, target_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if target_size <= 0 or target_size == len(x):
        return x.copy(), y.copy()
    rng = np.random.default_rng(seed)
    if target_size < len(x):
        indices = rng.choice(len(x), size=target_size, replace=False)
        return x[indices].copy(), y[indices].copy()
    extra = target_size - len(x)
    indices = rng.integers(0, len(x), size=extra)
    jitter_scale = np.maximum(x.std(axis=0) * 0.01, 1e-6)
    extra_x = x[indices] + rng.normal(0.0, jitter_scale, size=(extra, x.shape[1]))
    extra_y = y[indices]
    return np.vstack([x, extra_x]), np.r_[y, extra_y]


def write_points(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y.astype(int)}).to_csv(
        path, index=False, header=False
    )


def true_cluster_count(y: np.ndarray | None) -> int:
    if y is None:
        return 3
    labels = set(int(item) for item in y if int(item) >= 0)
    return max(len(labels), 2)


def min_samples_candidates(n: int) -> list[int]:
    base = max(5, int(math.ceil(math.log2(max(n, 2)))))
    values = {5, base, max(5, base * 2), max(5, int(round(math.sqrt(n))))}
    return sorted(value for value in values if value < n)


def selection_basis(y_true: np.ndarray | None) -> str:
    return "oracle_external_ari_nmi" if y_true is not None else "internal_silhouette"


def selection_key(row: dict) -> tuple[float, float, float, float]:
    if row.get("selection_basis") == "oracle_external_ari_nmi":
        return (
            float(pd.to_numeric(pd.Series([row.get("ari")]), errors="coerce").fillna(-1).iloc[0]),
            float(pd.to_numeric(pd.Series([row.get("nmi")]), errors="coerce").fillna(-1).iloc[0]),
            float(pd.to_numeric(pd.Series([row.get("fmi")]), errors="coerce").fillna(-1).iloc[0]),
            -float(pd.to_numeric(pd.Series([row.get("runtime_ms")]), errors="coerce").fillna(1e18).iloc[0]),
        )
    return (
        float(pd.to_numeric(pd.Series([row.get("silhouette")]), errors="coerce").fillna(-1).iloc[0]),
        -float(pd.to_numeric(pd.Series([row.get("davies_bouldin")]), errors="coerce").fillna(1e18).iloc[0]),
        float(pd.to_numeric(pd.Series([row.get("calinski_harabasz")]), errors="coerce").fillna(-1).iloc[0]),
        -float(pd.to_numeric(pd.Series([row.get("runtime_ms")]), errors="coerce").fillna(1e18).iloc[0]),
    )


def complete_row(row: dict) -> dict:
    full = dict.fromkeys(SUMMARY_COLUMNS, np.nan)
    full.update(row)
    return full


def score_labels(
    dataset: str,
    algorithm: str,
    x: np.ndarray,
    y_true: np.ndarray | None,
    labels: np.ndarray,
    runtime_ms: float,
    *,
    seed: int,
    base_points: int,
    params: str,
    basis: str,
    fits_evaluated: int = 1,
    tuning_runtime_ms: float | None = None,
    threshold_mode: str = "",
    model_dir: str = "",
) -> dict:
    row = score(dataset, algorithm, x, y_true, labels, runtime_ms)
    row.update(
        threshold_mode=threshold_mode,
        seed=seed,
        base_points=base_points,
        params=params,
        selection_basis=basis,
        fits_evaluated=fits_evaluated,
        tuning_runtime_ms=runtime_ms if tuning_runtime_ms is None else tuning_runtime_ms,
        status="ok",
        error="",
        model_dir=model_dir,
    )
    return complete_row(row)


def error_row(
    dataset: str,
    algorithm: str,
    seed: int,
    base_points: int,
    message: str,
    *,
    status: str = "error",
) -> dict:
    return complete_row(
        {
            "dataset": dataset,
            "algorithm": algorithm,
            "seed": seed,
            "base_points": base_points,
            "status": status,
            "error": message,
            "fits_evaluated": 0,
            "tuning_runtime_ms": 0.0,
        }
    )


def delaucluster_command(
    csv_path: Path,
    model_dir: Path,
    mode: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        str(args.cluster_exe),
        "static",
        str(csv_path),
        str(model_dir),
        "--threshold-mode",
        mode,
        "--stability-tie-band",
        f"{args.stability_tie_band:.17g}",
    ]
    if mode == "bucket_percentile":
        cmd.extend(
            [
                "--bucket-percentile",
                f"{args.bucket_percentile:.17g}",
                "--bucket-min-scores",
                str(args.bucket_min_scores),
                "--bucket-max-radius",
                str(args.bucket_max_radius),
            ]
        )
    if mode == "knn_median" and args.knn_scale_k > 0:
        cmd.extend(["--knn-scale-k", str(args.knn_scale_k)])
    if mode == "supported_merge":
        cmd.extend(
            [
                "--supported-merge-factor",
                f"{args.supported_merge_factor:.17g}",
                "--supported-merge-min-edges",
                str(args.supported_merge_min_edges),
            ]
        )
    if mode == "saddle_cut":
        cmd.extend(
            [
                "--saddle-cut-relax-factor",
                f"{args.saddle_cut_relax_factor:.17g}",
                "--saddle-cut-cross-factor",
                f"{args.saddle_cut_cross_factor:.17g}",
                "--saddle-cut-min-clusters",
                str(args.saddle_cut_min_clusters),
                "--saddle-cut-max-cross-retained",
                f"{args.saddle_cut_max_cross_retained:.17g}",
            ]
        )
    if mode == "manifold_filter":
        cmd.extend(
            [
                "--manifold-core-factor",
                f"{args.manifold_core_factor:.17g}",
                "--manifold-bridge-factor",
                f"{args.manifold_bridge_factor:.17g}",
                "--manifold-alignment-min",
                f"{args.manifold_alignment_min:.17g}",
                "--manifold-anisotropy-min",
                f"{args.manifold_anisotropy_min:.17g}",
                "--manifold-dataset-anisotropy-min",
                f"{args.manifold_dataset_anisotropy_min:.17g}",
                "--manifold-min-clusters",
                str(args.manifold_min_clusters),
            ]
        )
    return cmd


def selected_threshold_objective(model_dir: Path) -> float:
    metadata_path = model_dir / "metadata.csv"
    if metadata_path.exists():
        try:
            meta = pd.read_csv(
                metadata_path, header=None, names=["key", "value"], dtype=str
            )
            meta_map = dict(zip(meta["key"], meta["value"]))
            if (
                meta_map.get("threshold_mode") == "saddle_cut"
                and meta_map.get("saddle_cut_activated") != "1"
            ):
                return -float("inf")
            if (
                meta_map.get("threshold_mode") == "manifold_filter"
                and meta_map.get("manifold_filter_activated") != "1"
            ):
                return -float("inf")
        except Exception:
            pass
    trace_path = model_dir / "threshold_trace.csv"
    if not trace_path.exists():
        return -float("inf")
    trace = pd.read_csv(trace_path)
    if trace.empty or "objective" not in trace.columns:
        return -float("inf")
    selected = trace.loc[trace.get("selected", 0).astype(int) == 1]
    if selected.empty:
        selected = trace
    values = pd.to_numeric(selected["objective"], errors="coerce").dropna()
    if values.empty:
        return -float("inf")
    return float(values.max())


def run_delaucluster(
    dataset: str,
    csv_path: Path,
    run_dir: Path,
    x: np.ndarray,
    y_true: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    model_dir = run_dir / "DelauCluster"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    cmd = delaucluster_command(csv_path, model_dir, args.delau_mode, args)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    clusters = pd.read_csv(model_dir / "clusters.csv")
    labels = clusters["cluster_id"].to_numpy(int)
    if "is_noise" in clusters.columns:
        labels = labels.copy()
        labels[clusters["is_noise"].to_numpy(int) == 1] = -1
    return score_labels(
        dataset,
        "DelauCluster",
        x,
        y_true,
        labels,
        runtime_ms,
        seed=seed,
        base_points=len(x),
        params=f"threshold_mode={args.delau_mode}",
        basis="unsupervised_fixed_policy",
        threshold_mode=args.delau_mode,
        model_dir=str(model_dir),
    )


def run_delaucluster_auto(
    dataset: str,
    csv_path: Path,
    run_dir: Path,
    x: np.ndarray,
    y_true: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    modes = []
    for mode in parse_csv_list(args.delau_auto_modes):
        if mode not in modes:
            modes.append(mode)
    if not modes:
        return error_row(
            dataset,
            "DelauClusterAuto",
            seed,
            len(x),
            "no DelauCluster auto modes configured",
        )

    auto_dir = run_dir / "DelauClusterAuto"
    if auto_dir.exists():
        shutil.rmtree(auto_dir)
    rows: list[dict] = []
    total_runtime_ms = 0.0
    for mode in modes:
        model_dir = auto_dir / mode
        cmd = delaucluster_command(csv_path, model_dir, mode, args)
        try:
            start = time.perf_counter()
            subprocess.run(cmd, check=True)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            total_runtime_ms += runtime_ms
            clusters = pd.read_csv(model_dir / "clusters.csv")
            labels = clusters["cluster_id"].to_numpy(int)
            if "is_noise" in clusters.columns:
                labels = labels.copy()
                labels[clusters["is_noise"].to_numpy(int) == 1] = -1
            row = score_labels(
                dataset,
                "DelauClusterAuto",
                x,
                y_true,
                labels,
                runtime_ms,
                seed=seed,
                base_points=len(x),
                params=f"candidate_mode={mode}",
                basis="unsupervised_mode_objective",
                threshold_mode=mode,
                model_dir=str(model_dir),
            )
            row["_mode_objective"] = selected_threshold_objective(model_dir)
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive subprocess path
            rows.append(
                error_row(
                    dataset,
                    "DelauClusterAuto",
                    seed,
                    len(x),
                    f"{mode}: {exc}",
                )
            )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        failed = "; ".join(str(row.get("error", "")) for row in rows)
        return error_row(
            dataset,
            "DelauClusterAuto",
            seed,
            len(x),
            failed or "all DelauCluster auto modes failed",
        )

    best = max(
        ok_rows,
        key=lambda row: (
            float(row.get("_mode_objective", -float("inf"))),
            -float(
                pd.to_numeric(
                    pd.Series([row.get("runtime_ms")]), errors="coerce"
                )
                .fillna(1e18)
                .iloc[0]
            ),
        ),
    )
    selected_mode = str(best.get("threshold_mode", ""))
    selected_score = float(best.get("_mode_objective", -float("inf")))
    best.pop("_mode_objective", None)
    best.update(
        runtime_ms=total_runtime_ms,
        tuning_runtime_ms=total_runtime_ms,
        fits_evaluated=len(ok_rows),
        params=(
            f"selected_mode={selected_mode};modes={','.join(modes)};"
            f"mode_objective={selected_score:.6g}"
        ),
        model_dir=str(auto_dir / selected_mode),
    )
    return complete_row(best)


def candidate_models(
    algorithm: str,
    x: np.ndarray,
    y_true: np.ndarray | None,
    args: argparse.Namespace,
) -> list[tuple[str, object]]:
    n = len(x)
    k = true_cluster_count(y_true)
    mins = min_samples_candidates(n)
    if algorithm == "KMeans":
        return [("k=oracle", KMeans(n_clusters=k, n_init=20, random_state=args.random_state))]
    if algorithm == "MiniBatchKMeans":
        return [
            (
                "k=oracle",
                MiniBatchKMeans(
                    n_clusters=k,
                    n_init=20,
                    random_state=args.random_state,
                    batch_size=min(1024, max(100, n)),
                ),
            )
        ]
    if algorithm == "DBSCAN":
        models = []
        quantiles = [float(item) for item in parse_csv_list(args.dbscan_eps_quantiles)]
        for min_samples in mins:
            eps_base = estimate_eps(x, min_samples)
            for quantile in quantiles:
                nn = estimate_eps_at_quantile(x, min_samples, quantile)
                eps = nn if np.isfinite(nn) and nn > 0 else eps_base
                models.append(
                    (
                        f"eps_q={quantile:g},min_samples={min_samples}",
                        DBSCAN(eps=eps, min_samples=min_samples),
                    )
                )
        return models
    if algorithm == "OPTICS":
        models = []
        for min_samples in mins:
            for xi in [0.03, 0.05, 0.10]:
                models.append(
                    (
                        f"xi={xi:g},min_samples={min_samples}",
                        OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_samples),
                    )
                )
        return models
    if algorithm == "HDBSCAN":
        if HDBSCAN is None:
            return []
        models = []
        for min_cluster_size in mins:
            models.append(
                (
                    f"min_cluster_size={min_cluster_size}",
                    HDBSCAN(min_cluster_size=min_cluster_size),
                )
            )
        return models
    if algorithm == "BIRCH":
        return [
            (
                f"threshold={threshold:g},k=oracle",
                Birch(threshold=threshold, n_clusters=k),
            )
            for threshold in [0.25, 0.35, 0.50, 0.75, 1.00]
        ]
    if algorithm == "Agglomerative":
        if n > args.max_agglomerative_n:
            return []
        return [("k=oracle", AgglomerativeClustering(n_clusters=k))]
    if algorithm == "Spectral":
        if n > args.max_spectral_n:
            return []
        n_neighbors = min(10, max(1, n - 1))
        return [
            (
                f"k=oracle,n_neighbors={n_neighbors}",
                SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    random_state=args.random_state,
                ),
            )
        ]
    return []


def estimate_eps_at_quantile(x: np.ndarray, min_samples: int, quantile: float) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min_samples).fit(x)
    dists, _ = nn.kneighbors(x)
    kth = np.sort(dists[:, -1])
    return float(np.percentile(kth, quantile))


def run_baseline_algorithm(
    dataset: str,
    algorithm: str,
    x: np.ndarray,
    y_true: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    basis = selection_basis(y_true)
    models = candidate_models(algorithm, x, y_true, args)
    if not models:
        return error_row(
            dataset,
            algorithm,
            seed,
            len(x),
            "algorithm unavailable or skipped by size limit",
            status="excluded",
        )

    candidates = []
    total_runtime = 0.0
    for params, model in models:
        try:
            start = time.perf_counter()
            labels = model.fit_predict(x)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            total_runtime += runtime_ms
            candidates.append(
                score_labels(
                    dataset,
                    algorithm,
                    x,
                    y_true,
                    labels,
                    runtime_ms,
                    seed=seed,
                    base_points=len(x),
                    params=params,
                    basis=basis,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive for sklearn variants
            total_runtime += 0.0
            candidates.append(
                error_row(dataset, algorithm, seed, len(x), f"{type(exc).__name__}: {exc}")
            )

    ok = [row for row in candidates if row.get("status") == "ok"]
    if not ok:
        failed = candidates[0] if candidates else {}
        return error_row(
            dataset,
            algorithm,
            seed,
            len(x),
            str(failed.get("error", "all candidate fits failed")),
        )
    best = max(ok, key=selection_key)
    best["fits_evaluated"] = len(models)
    best["tuning_runtime_ms"] = total_runtime
    return complete_row(best)


def run_case(
    dataset: str,
    seed: int,
    target_size: int,
    algorithms: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    x, y = build(dataset, seed)
    x, y = resize_points(x, y, target_size, seed + args.resize_seed_offset)
    run_dir = out_dir / "runs" / dataset / f"n_{len(x)}" / f"seed_{seed}"
    csv_path = run_dir / "points.csv"
    write_points(csv_path, x, y)

    rows: list[dict] = []
    for algorithm in algorithms:
        if algorithm == "DelauCluster":
            rows.append(run_delaucluster(dataset, csv_path, run_dir, x, y, args, seed))
        elif algorithm == "DelauClusterAuto":
            rows.append(
                run_delaucluster_auto(dataset, csv_path, run_dir, x, y, args, seed)
            )
        else:
            rows.append(run_baseline_algorithm(dataset, algorithm, x, y, args, seed))
    return rows


def sample_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def build_dataset_table(summary: pd.DataFrame, algorithms: list[str]) -> pd.DataFrame:
    ok = summary.loc[summary["status"] == "ok"].copy()
    rows = []
    for (dataset, base_points, algorithm), group in ok.groupby(
        ["dataset", "base_points", "algorithm"], sort=False
    ):
        rows.append(
            {
                "dataset": dataset,
                "base_points": int(base_points),
                "algorithm": algorithm,
                "seeds": group["seed"].nunique(),
                "ari": group["ari"].mean(),
                "ari_std": sample_std(group["ari"]),
                "nmi": group["nmi"].mean(),
                "nmi_std": sample_std(group["nmi"]),
                "ami": group["ami"].mean(),
                "fmi": group["fmi"].mean(),
                "fmi_std": sample_std(group["fmi"]),
                "silhouette": group["silhouette"].mean(),
                "davies_bouldin": group["davies_bouldin"].mean(),
                "calinski_harabasz": group["calinski_harabasz"].mean(),
                "runtime_ms": group["runtime_ms"].mean(),
                "runtime_ms_std": sample_std(group["runtime_ms"]),
                "tuning_runtime_ms": group["tuning_runtime_ms"].mean(),
                "num_clusters": group["num_clusters"].mean(),
                "noise_pct": group["noise_pct"].mean(),
                "mean_fits": group["fits_evaluated"].mean(),
                "selection_basis": ",".join(sorted(set(group["selection_basis"].dropna().astype(str)))),
            }
        )
    table = pd.DataFrame(rows)
    order = {algorithm: idx for idx, algorithm in enumerate(algorithms)}
    if not table.empty:
        table["algorithm_order"] = table["algorithm"].map(order).fillna(999)
        table = table.sort_values(["dataset", "base_points", "algorithm_order"]).drop(
            columns=["algorithm_order"]
        )
    return table


def build_main_table(
    dataset_table: pd.DataFrame, algorithms: list[str], summary: pd.DataFrame
) -> pd.DataFrame:
    if dataset_table.empty:
        return pd.DataFrame()
    ok = summary.loc[summary["status"] == "ok"].copy()
    winners: Counter[str] = Counter()
    for _, group in dataset_table.groupby(["dataset", "base_points"], sort=True):
        values = pd.to_numeric(group["ari"], errors="coerce")
        if values.dropna().empty:
            continue
        best = values.max()
        for algorithm in group.loc[(values - best).abs() <= 1e-12, "algorithm"]:
            winners[str(algorithm)] += 1

    rows = []
    for algorithm, group in dataset_table.groupby("algorithm", sort=False):
        raw = ok.loc[ok["algorithm"] == algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "datasets": group["dataset"].nunique(),
                "runs": raw[["dataset", "base_points", "seed"]]
                .drop_duplicates()
                .shape[0],
                "mean_ari": raw["ari"].mean(),
                "std_ari": sample_std(raw["ari"]),
                "mean_nmi": raw["nmi"].mean(),
                "std_nmi": sample_std(raw["nmi"]),
                "mean_ami": group["ami"].mean(),
                "mean_fmi": raw["fmi"].mean(),
                "std_fmi": sample_std(raw["fmi"]),
                "mean_silhouette": group["silhouette"].mean(),
                "mean_runtime_ms": raw["runtime_ms"].mean(),
                "std_runtime_ms": sample_std(raw["runtime_ms"]),
                "mean_tuning_runtime_ms": group["tuning_runtime_ms"].mean(),
                "mean_noise_pct": group["noise_pct"].mean(),
                "ari_wins": winners.get(str(algorithm), 0),
            }
        )
    table = pd.DataFrame(rows)
    order = {algorithm: idx for idx, algorithm in enumerate(algorithms)}
    table["algorithm_order"] = table["algorithm"].map(order).fillna(999)
    return table.sort_values("algorithm_order").drop(columns=["algorithm_order"])


def write_markdown_table(path: Path, table: pd.DataFrame, *, compact: bool) -> None:
    if compact:
        header = "| Algorithm | Datasets | Runs | ARI | ARI SD | NMI | NMI SD | FMI | Runtime ms | Runtime SD | Noise % | ARI wins |"
        lines = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in table.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["algorithm"]),
                        str(int(row["datasets"])),
                        str(int(row["runs"])),
                        format_float(row["mean_ari"]),
                        format_float(row["std_ari"]),
                        format_float(row["mean_nmi"]),
                        format_float(row["std_nmi"]),
                        format_float(row["mean_fmi"]),
                        format_float(row["mean_runtime_ms"], 1),
                        format_float(row["std_runtime_ms"], 1),
                        format_float(row["mean_noise_pct"], 1),
                        str(int(row["ari_wins"])),
                    ]
                )
                + " |"
            )
    else:
        header = "| Dataset | n | Algorithm | Seeds | ARI | ARI SD | NMI | NMI SD | FMI | Runtime ms | Runtime SD | Clusters | Noise % |"
        lines = [header, "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in table.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        str(int(row["base_points"])),
                        str(row["algorithm"]),
                        str(int(row["seeds"])),
                        format_float(row["ari"]),
                        format_float(row["ari_std"]),
                        format_float(row["nmi"]),
                        format_float(row["nmi_std"]),
                        format_float(row["fmi"]),
                        format_float(row["runtime_ms"], 1),
                        format_float(row["runtime_ms_std"], 1),
                        format_float(row["num_clusters"], 1),
                        format_float(row["noise_pct"], 1),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(path: Path, table: pd.DataFrame, *, compact: bool) -> None:
    if compact:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{External clustering baseline comparison averaged across labeled synthetic spatial datasets. Values are mean $\pm$ sample standard deviation over dataset--seed runs. Density baselines are oracle-tuned using labels; DelauCluster variants use label-free policies.}",
            r"\label{tab:external-baseline-comparison}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Algorithm & Runs & ARI & NMI & FMI & Runtime & Noise & ARI wins \\",
            r"\midrule",
        ]
        for _, row in table.iterrows():
            lines.append(
                " & ".join(
                    [
                        latex_escape(row["algorithm"]),
                        str(int(row["runs"])),
                        rf"{format_float(row['mean_ari'])} $\pm$ {format_float(row['std_ari'])}",
                        rf"{format_float(row['mean_nmi'])} $\pm$ {format_float(row['std_nmi'])}",
                        rf"{format_float(row['mean_fmi'])} $\pm$ {format_float(row['std_fmi'])}",
                        rf"{format_float(row['mean_runtime_ms'], 1)} $\pm$ {format_float(row['std_runtime_ms'], 1)}",
                        format_float(row["mean_noise_pct"], 1),
                        str(int(row["ari_wins"])),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    else:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Per-dataset external clustering baseline comparison.}",
            r"\label{tab:external-baseline-appendix}",
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            r"Dataset & $n$ & Algorithm & Seeds & ARI & NMI & FMI & Runtime & Noise \\",
            r"\midrule",
        ]
        for _, row in table.iterrows():
            lines.append(
                " & ".join(
                    [
                        latex_escape(row["dataset"]),
                        str(int(row["base_points"])),
                        latex_escape(row["algorithm"]),
                        str(int(row["seeds"])),
                        rf"{format_float(row['ari'])} $\pm$ {format_float(row['ari_std'])}",
                        rf"{format_float(row['nmi'])} $\pm$ {format_float(row['nmi_std'])}",
                        rf"{format_float(row['fmi'])} $\pm$ {format_float(row['fmi_std'])}",
                        rf"{format_float(row['runtime_ms'], 1)} $\pm$ {format_float(row['runtime_ms_std'], 1)}",
                        format_float(row["noise_pct"], 1),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def draft_section(main_table: pd.DataFrame, dataset_table: pd.DataFrame) -> tuple[str, str]:
    if main_table.empty:
        tex = "\n".join(
            [
                r"\subsection{External Baseline Comparison}",
                r"\label{sec:external-baseline-comparison}",
                "",
                "External baseline tables have not been generated yet.",
                "",
            ]
        )
        return tex, "## External Baseline Comparison\n\nExternal baseline tables have not been generated yet.\n"

    delau = main_table.loc[main_table["algorithm"] == "DelauCluster"]
    best = main_table.sort_values(["mean_ari", "mean_nmi"], ascending=False).iloc[0]
    delau_text = "DelauCluster is not present in the generated table."
    if not delau.empty:
        row = delau.iloc[0]
        rank = int(
            (main_table["mean_ari"] > row["mean_ari"]).sum()
            + 1
        )
        delau_text = (
            f"DelauCluster obtains mean ARI/NMI {format_float(row['mean_ari'])}/"
            f"{format_float(row['mean_nmi'])} and ranks {rank} of "
            f"{len(main_table)} by mean ARI in this sweep."
        )
    auto = main_table.loc[main_table["algorithm"] == "DelauClusterAuto"]
    auto_text = ""
    if not auto.empty:
        row = auto.iloc[0]
        rank = int((main_table["mean_ari"] > row["mean_ari"]).sum() + 1)
        auto_text = (
            f" The label-free DelauClusterAuto selector obtains mean ARI/NMI "
            f"{format_float(row['mean_ari'])}/{format_float(row['mean_nmi'])} "
            f"and ranks {rank} of {len(main_table)}."
        )
    winner_text = (
        f"The strongest average ARI in the current table is {latex_escape(best['algorithm'])} "
        f"with ARI/NMI {format_float(best['mean_ari'])}/{format_float(best['mean_nmi'])}."
    )
    cases = dataset_table["dataset"].nunique() if not dataset_table.empty else 0
    tex = "\n".join(
        [
            r"\subsection{External Baseline Comparison}",
            r"\label{sec:external-baseline-comparison}",
            "",
            "We compare fixed and label-free adaptive DelauCluster policies "
            "against common clustering baselines on the labeled synthetic "
            "spatial benchmark. KMeans, "
            "MiniBatchKMeans, BIRCH, agglomerative clustering, and spectral "
            "clustering use the oracle number of non-noise classes when labels "
            "exist. DBSCAN, OPTICS, and HDBSCAN are selected from parameter grids "
            "using external ARI/NMI on the same labels. This gives the density "
            "baselines an oracle-assisted advantage; DelauCluster variants use "
            "unsupervised threshold or mode-selection policies and do not fit to "
            "labels.",
            "",
            f"Table~\\ref{{tab:external-baseline-comparison}} summarizes {cases} "
            f"dataset-size cases. {delau_text}{auto_text} {winner_text} The per-dataset "
            "appendix table should be used to identify where DelauCluster is "
            "competitive, where density baselines are stronger, and which cases "
            "remain open research weaknesses rather than hidden failures.",
            "",
            r"\paragraph{Recommended captions.}",
            r"\textbf{Table~\ref{tab:external-baseline-comparison}.} External baseline comparison averaged across labeled synthetic spatial datasets. Density baselines are oracle-tuned by labels, while DelauCluster variants use label-free policies.",
            "",
            r"\textbf{Table~\ref{tab:external-baseline-appendix}.} Per-dataset external baseline comparison with runtime, tuning cost, noise percentage, and selected parameter policy.",
            "",
        ]
    )
    md = "\n".join(
        [
            "## External Baseline Comparison",
            "",
            "Baselines are compared on labeled synthetic spatial datasets. Density baselines are oracle-tuned by labels; DelauCluster variants use label-free policies.",
            "",
            f"{delau_text}{auto_text} {winner_text}",
            "",
        ]
    )
    return tex, md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--base-limits", default="0")
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHM_ORDER),
    )
    parser.add_argument("--out-dir", default="results/run_baseline_comparison")
    parser.add_argument("--cluster-exe", default="./build/cluster")
    parser.add_argument("--delau-mode", default="local_scale")
    parser.add_argument(
        "--delau-auto-modes",
        default="raw_length,local_scale,knn_median,supported_merge,saddle_cut,manifold_filter",
        help="Comma-separated DelauCluster modes for DelauClusterAuto.",
    )
    parser.add_argument("--stability-tie-band", type=float, default=1e-9)
    parser.add_argument("--bucket-percentile", type=float, default=0.75)
    parser.add_argument("--bucket-min-scores", type=int, default=0)
    parser.add_argument("--bucket-max-radius", type=int, default=0)
    parser.add_argument("--knn-scale-k", type=int, default=0)
    parser.add_argument("--supported-merge-factor", type=float, default=1.20)
    parser.add_argument("--supported-merge-min-edges", type=int, default=8)
    parser.add_argument("--saddle-cut-relax-factor", type=float, default=1.05)
    parser.add_argument("--saddle-cut-cross-factor", type=float, default=0.40)
    parser.add_argument("--saddle-cut-min-clusters", type=int, default=12)
    parser.add_argument("--saddle-cut-max-cross-retained", type=float, default=0.05)
    parser.add_argument("--manifold-core-factor", type=float, default=0.85)
    parser.add_argument("--manifold-bridge-factor", type=float, default=2.20)
    parser.add_argument("--manifold-alignment-min", type=float, default=0.85)
    parser.add_argument("--manifold-anisotropy-min", type=float, default=0.35)
    parser.add_argument("--manifold-dataset-anisotropy-min", type=float, default=0.42)
    parser.add_argument("--manifold-min-clusters", type=int, default=6)
    parser.add_argument("--dbscan-eps-quantiles", default="60,70,80,85,90,95")
    parser.add_argument("--max-agglomerative-n", type=int, default=5000)
    parser.add_argument("--max-spectral-n", type=int, default=2500)
    parser.add_argument("--resize-seed-offset", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.manifold_core_factor = max(0.0, args.manifold_core_factor)
    args.manifold_bridge_factor = max(
        args.manifold_core_factor, args.manifold_bridge_factor
    )
    args.manifold_alignment_min = min(1.0, max(0.0, args.manifold_alignment_min))
    args.manifold_anisotropy_min = min(1.0, max(0.0, args.manifold_anisotropy_min))
    args.manifold_dataset_anisotropy_min = max(
        0.0, args.manifold_dataset_anisotropy_min
    )
    args.manifold_min_clusters = max(2, args.manifold_min_clusters)

    datasets = parse_csv_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    base_limits = parse_int_list(args.base_limits)
    algorithms = parse_csv_list(args.algorithms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            for base_limit in base_limits:
                print(f"benchmarking {dataset} n={base_limit or 'native'} seed={seed}")
                rows.extend(run_case(dataset, seed, base_limit, algorithms, args, out_dir))

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "baseline_comparison_summary.csv", index=False)
    dataset_table = build_dataset_table(summary, algorithms)
    main_table = build_main_table(dataset_table, algorithms, summary)

    dataset_table.to_csv(out_dir / "baseline_comparison_appendix_table.csv", index=False)
    main_table.to_csv(out_dir / "baseline_comparison_main_table.csv", index=False)
    write_markdown_table(out_dir / "baseline_comparison_main_table.md", main_table, compact=True)
    write_latex_table(out_dir / "baseline_comparison_main_table.tex", main_table, compact=True)
    write_markdown_table(out_dir / "baseline_comparison_appendix_table.md", dataset_table, compact=False)
    write_latex_table(out_dir / "baseline_comparison_appendix_table.tex", dataset_table, compact=False)
    section_tex, section_md = draft_section(main_table, dataset_table)
    (out_dir / "baseline_comparison_section.tex").write_text(section_tex)
    (out_dir / "baseline_comparison_section.md").write_text(section_md)
    excluded = summary.loc[summary["status"] != "ok"].copy()
    excluded.to_csv(out_dir / "baseline_comparison_exclusions.csv", index=False)

    print(f"wrote {out_dir / 'baseline_comparison_main_table.csv'}")


if __name__ == "__main__":
    main()
