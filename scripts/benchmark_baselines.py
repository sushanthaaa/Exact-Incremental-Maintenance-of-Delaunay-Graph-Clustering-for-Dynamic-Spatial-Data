#!/usr/bin/env python3
"""Benchmark DelauCluster against common clustering baselines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

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
except Exception:  # pragma: no cover - optional dependency/version feature
    HDBSCAN = None
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


SCHEMA = [
    "dataset",
    "algorithm",
    "ari",
    "nmi",
    "ami",
    "fmi",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "runtime_ms",
    "memory_mb",
    "num_clusters",
    "noise_pct",
]


def load_xy_label(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path, header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[0, 1])
    x = numeric.iloc[:, :2].to_numpy(float)
    if numeric.shape[1] >= 3 and numeric.iloc[:, 2].notna().any():
        y = numeric.iloc[:, 2].fillna(-1).to_numpy(int)
    else:
        y = None
    return x, y


def estimate_eps(x: np.ndarray, min_samples: int) -> float:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(x)
    dists, _ = nn.kneighbors(x)
    kth = np.sort(dists[:, -1])
    return float(np.percentile(kth, 90))


def score(dataset: str, algorithm: str, x: np.ndarray, y_true: np.ndarray | None,
          labels: np.ndarray, runtime_ms: float) -> dict:
    valid = labels >= 0
    num_clusters = len(set(labels[valid]))
    noise_pct = 100.0 * float(np.mean(~valid)) if len(labels) else 0.0
    row = dict.fromkeys(SCHEMA, np.nan)
    row.update(
        dataset=dataset,
        algorithm=algorithm,
        runtime_ms=runtime_ms,
        memory_mb=np.nan,
        num_clusters=num_clusters,
        noise_pct=noise_pct,
    )
    if y_true is not None:
        row["ari"] = adjusted_rand_score(y_true, labels)
        row["nmi"] = normalized_mutual_info_score(y_true, labels)
        row["ami"] = adjusted_mutual_info_score(y_true, labels)
        row["fmi"] = fowlkes_mallows_score(y_true, labels)
    if valid.sum() > num_clusters and num_clusters > 1:
        row["silhouette"] = silhouette_score(x[valid], labels[valid])
        row["davies_bouldin"] = davies_bouldin_score(x[valid], labels[valid])
        row["calinski_harabasz"] = calinski_harabasz_score(x[valid], labels[valid])
    return row


def run_delaucluster(dataset: str, csv_path: Path, out_dir: Path, x: np.ndarray,
                     y_true: np.ndarray | None, exe: Path) -> dict:
    run_dir = out_dir / "delaucluster"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    start = time.perf_counter()
    subprocess.run([str(exe), "static", str(csv_path), str(run_dir)], check=True)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    labels = pd.read_csv(run_dir / "clusters.csv")["cluster_id"].to_numpy(int)
    return score(dataset, "DelauCluster", x, y_true, labels, runtime_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("input_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--cluster-exe", default="build/cluster")
    args = parser.parse_args()

    dataset = args.dataset
    csv_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y_true = load_xy_label(csv_path)
    k = len(set(y_true[y_true >= 0])) if y_true is not None else 3
    min_samples = max(5, int(np.ceil(np.log2(len(x)))))
    eps = estimate_eps(x, min_samples)

    rows = [run_delaucluster(dataset, csv_path, out_dir, x, y_true, Path(args.cluster_exe))]

    candidates = [
        ("KMeans", KMeans(n_clusters=k, n_init=20, random_state=42)),
        ("MiniBatchKMeans", MiniBatchKMeans(n_clusters=k, n_init=20, random_state=42)),
        ("DBSCAN", DBSCAN(eps=eps, min_samples=min_samples)),
        ("OPTICS", OPTICS(min_samples=min_samples, xi=0.05)),
        ("BIRCH", Birch(n_clusters=k)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=k)),
    ]
    if HDBSCAN is not None:
        candidates.append(("HDBSCAN", HDBSCAN(min_cluster_size=min_samples)))
    if len(x) <= 5000:
        candidates.append(
            ("Spectral", SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                            random_state=42))
        )

    for name, model in candidates:
        start = time.perf_counter()
        labels = model.fit_predict(x)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        rows.append(score(dataset, name, x, y_true, labels, runtime_ms))

    df = pd.DataFrame(rows, columns=SCHEMA)
    df.to_csv(out_dir / "benchmark_metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
