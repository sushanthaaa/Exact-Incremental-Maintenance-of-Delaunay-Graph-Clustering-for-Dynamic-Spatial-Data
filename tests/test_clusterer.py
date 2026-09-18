#!/usr/bin/env python3
"""Tests for the DelauCluster command-line implementation and reframed experiments."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = Path(os.environ.get("CLUSTER_EXE", ROOT / "build" / "cluster"))
IDBSCAN = Path(os.environ.get("IDBSCAN_EXE", ROOT / "build" / "incremental_dbscan_stream"))
GENERATE_DATASETS = ROOT / "scripts" / "generate_datasets.py"
BASELINE_COMPARISON = ROOT / "scripts" / "run_baseline_comparison.py"
FAITHFULNESS = ROOT / "scripts" / "run_dynamic_faithfulness.py"


class DelauClusterCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CLUSTER.exists():
            raise unittest.SkipTest(f"{CLUSTER} not built")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="delaucluster_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_two_blobs(self, path: Path):
        rng = np.random.default_rng(42)
        x0 = rng.normal([-2, 0], 0.1, size=(40, 2))
        x1 = rng.normal([2, 0], 0.1, size=(40, 2))
        x = np.vstack([x0, x1])
        y = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
        pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y}).to_csv(
            path, index=False, header=False
        )

    def write_bridge_noise(self, path: Path):
        rng = np.random.default_rng(7)
        x0 = rng.normal([-2, 0], 0.11, size=(90, 2))
        x1 = rng.normal([2, 0], 0.11, size=(90, 2))
        bridge_x = np.linspace(-1.35, 1.35, 16)
        bridge_y = rng.normal(0, 0.015, size=bridge_x.shape[0])
        bridge = np.column_stack([bridge_x, bridge_y])
        x = np.vstack([x0, x1, bridge])
        y = np.r_[
            np.zeros(len(x0), dtype=int),
            np.ones(len(x1), dtype=int),
            np.full(len(bridge), 2, dtype=int),
        ]
        pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y}).to_csv(
            path, index=False, header=False
        )

    def write_varying_density(self, path: Path):
        rng = np.random.default_rng(42)
        x0 = rng.normal([-2.0, 0.0], [0.18, 0.18], size=(500, 2))
        x1 = rng.normal([1.2, 0.0], [0.65, 0.65], size=(900, 2))
        x2 = rng.normal([0.0, 2.0], [0.32, 0.75], size=(500, 2))
        x = np.vstack([x0, x1, x2])
        std = x.std(axis=0)
        std[std < 1e-12] = 1.0
        x = (x - x.mean(axis=0)) / std
        y = np.r_[
            np.zeros(len(x0), dtype=int),
            np.ones(len(x1), dtype=int),
            np.full(len(x2), 2, dtype=int),
        ]
        pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y}).to_csv(
            path, index=False, header=False
        )

    # ----- Core static behavior -----
    def test_static_outputs_valid_clusters(self):
        csv_path = self.tmp / "blobs.csv"
        out_dir = self.tmp / "out"
        self.write_two_blobs(csv_path)
        subprocess.run([str(CLUSTER), "static", str(csv_path), str(out_dir)], check=True)

        clusters = pd.read_csv(out_dir / "clusters.csv")
        edges = pd.read_csv(out_dir / "edges.csv")
        bucket_points = pd.read_csv(out_dir / "bucket_points.csv")
        bucket_edges = pd.read_csv(out_dir / "bucket_edges.csv")
        trace = pd.read_csv(out_dir / "threshold_trace.csv")
        meta = pd.read_csv(out_dir / "metadata.csv", header=None)

        self.assertEqual(len(clusters), 80)
        self.assertIn("cluster_id", clusters.columns)
        self.assertIn("is_noise", clusters.columns)
        self.assertGreaterEqual(clusters.loc[clusters.is_noise == 0, "cluster_id"].nunique(), 1)
        self.assertIn("normalized_score", edges.columns)
        self.assertIn("effective_threshold", edges.columns)
        self.assertIn("threshold_mode", set(meta.iloc[:, 0]))
        self.assertEqual(len(bucket_points), 80)
        self.assertIn("edge_id", bucket_edges.columns)
        self.assertIn("objective", trace.columns)
        self.assertEqual(int(trace["selected"].sum()), 1)

    def test_default_pruning_keeps_sparse_bridge_from_collapsing_clusters(self):
        csv_path = self.tmp / "bridge.csv"
        out_dir = self.tmp / "bridge_out"
        self.write_bridge_noise(csv_path)
        subprocess.run([str(CLUSTER), "static", str(csv_path), str(out_dir)], check=True)
        clusters = pd.read_csv(out_dir / "clusters.csv")
        non_noise = clusters.loc[clusters.is_noise == 0, "cluster_id"]
        self.assertGreaterEqual(non_noise.nunique(), 2)

    def test_threshold_modes_write_valid_static_outputs(self):
        csv_path = self.tmp / "blobs.csv"
        self.write_two_blobs(csv_path)
        for mode in [
            "raw_length", "local_scale", "bucket_percentile", "knn_median",
            "supported_merge", "saddle_cut", "manifold_filter",
        ]:
            out_dir = self.tmp / f"threshold_{mode}"
            subprocess.run(
                [str(CLUSTER), "static", str(csv_path), str(out_dir),
                 "--threshold-mode", mode, "--stability-tie-band", "0.005"],
                check=True,
            )
            clusters = pd.read_csv(out_dir / "clusters.csv")
            trace = pd.read_csv(out_dir / "threshold_trace.csv")
            meta = pd.read_csv(out_dir / "metadata.csv", header=None)
            self.assertEqual(len(clusters), 80)
            self.assertIn(mode, set(meta.loc[meta.iloc[:, 0] == "threshold_mode", 1]))
            self.assertEqual(int(trace["selected"].sum()), 1)

    def test_knn_threshold_mode_is_reproducible(self):
        csv_path = self.tmp / "varying_density.csv"
        self.write_varying_density(csv_path)
        outputs = []
        for idx in range(2):
            out_dir = self.tmp / f"knn_repro_{idx}"
            subprocess.run(
                [str(CLUSTER), "static", str(csv_path), str(out_dir),
                 "--threshold-mode", "knn_median", "--stability-tie-band", "0.001"],
                check=True,
            )
            meta = pd.read_csv(out_dir / "metadata.csv", header=None, names=["key", "value"])
            outputs.append(dict(zip(meta["key"], meta["value"])))
        self.assertEqual(outputs[0]["threshold"], outputs[1]["threshold"])
        self.assertEqual(outputs[0]["num_clusters"], outputs[1]["num_clusters"])
        self.assertEqual(outputs[0]["noise_count"], outputs[1]["noise_count"])

    def test_bucket_percentile_adaptive_cap_differs_from_local_scale(self):
        csv_path = self.tmp / "bridge.csv"
        self.write_bridge_noise(csv_path)
        outputs = {}
        for mode in ["local_scale", "bucket_percentile"]:
            out_dir = self.tmp / f"adaptive_{mode}"
            subprocess.run(
                [str(CLUSTER), "static", str(csv_path), str(out_dir), "--threshold-mode", mode],
                check=True,
            )
            meta = pd.read_csv(out_dir / "metadata.csv", header=None, names=["key", "value"])
            outputs[mode] = dict(zip(meta["key"], meta["value"]))
        self.assertLessEqual(
            int(float(outputs["bucket_percentile"]["retained_edge_count"])),
            int(float(outputs["local_scale"]["retained_edge_count"])),
        )

    # ----- assign -----
    def test_assign_uses_saved_model(self):
        csv_path = self.tmp / "blobs.csv"
        out_dir = self.tmp / "out"
        query_path = self.tmp / "query.csv"
        assign_path = self.tmp / "assign.csv"
        self.write_two_blobs(csv_path)
        query_path.write_text("-2,0\n2,0\n")
        subprocess.run([str(CLUSTER), "static", str(csv_path), str(out_dir)], check=True)
        subprocess.run(
            [str(CLUSTER), "assign", str(out_dir), str(query_path), str(assign_path)],
            check=True,
        )
        assigned = pd.read_csv(assign_path)
        self.assertEqual(len(assigned), 2)
        self.assertIn("cluster_id", assigned.columns)

    def test_assign_preserves_saved_threshold(self):
        csv_path = self.tmp / "blobs.csv"
        out_dir = self.tmp / "strict_out"
        query_path = self.tmp / "query.csv"
        assign_path = self.tmp / "assign.csv"
        self.write_two_blobs(csv_path)
        query_path.write_text("-2,0\n2,0\n")
        subprocess.run(
            [str(CLUSTER), "static", str(csv_path), str(out_dir), "--threshold", "0.01"],
            check=True,
        )
        subprocess.run(
            [str(CLUSTER), "assign", str(out_dir), str(query_path), str(assign_path)],
            check=True,
        )
        assigned = pd.read_csv(assign_path)
        self.assertTrue((assigned.is_noise == 1).all())

    # ----- Exact fully-dynamic maintenance (reframed core) -----
    def test_dynamic_stream_writes_honest_log(self):
        base_path = self.tmp / "base.csv"
        stream_path = self.tmp / "stream.csv"
        out_dir = self.tmp / "dyn"
        self.write_two_blobs(base_path)
        stream_path.write_text("insert,0,0,2\nmove,0,0,0,0.05\ndelete,0,0.05\n")
        subprocess.run(
            [str(CLUSTER), "dynamic", str(base_path), str(stream_path), str(out_dir),
             "--local-edge-refresh", "--local-bucket-refresh", "--verify-dynamic"],
            check=True,
        )
        log = pd.read_csv(out_dir / "dynamic_log.csv")
        self.assertEqual(log.operation.tolist(), ["insert", "move", "delete"])
        self.assertTrue((log.time_ns > 0).all())
        self.assertTrue((log.incremental_geometry_used == 1).all())
        self.assertTrue((log.local_edge_refresh_used == 1).all())
        self.assertTrue((log.local_bucket_refresh_used == 1).all())
        self.assertTrue((log.local_relabel_used == 1).all())
        self.assertTrue((log.verification_checked == 1).all())
        self.assertTrue((log.verification_mismatches == 0).all())
        self.assertIn("local_relabel_fallback", log.columns)

    def test_dynamic_fuzz_stream_verifier_never_passes_wrong_result(self):
        """Randomised insert/delete/move streams verify exactly or are safely caught.

        With ``--verify-dynamic`` the binary recomputes after every update via three
        oracles. The exact-fully-dynamic guard (sound connected-components check plus
        full-relabel fallback) should keep the partition equal to a from-scratch fit,
        so a completed run reports zero mismatches; any divergence must be caught
        (non-zero exit with a recorded mismatch), never silently returned.
        """
        base_path = self.tmp / "base.csv"
        self.write_two_blobs(base_path)
        rng = np.random.default_rng(20260618)
        active = [(-2.0, 0.0), (2.0, 0.0)]
        ops: list[str] = []
        for _ in range(80):
            u = rng.random()
            if u < 0.7 or len(active) < 6:
                ax = float(rng.uniform(-3.5, 3.5))
                ay = float(rng.uniform(-1.5, 1.5))
                ops.append(f"insert,{ax:.5f},{ay:.5f},{int(rng.integers(0, 3))}")
                active.append((ax, ay))
            elif u < 0.85:
                px, py = active[int(rng.integers(0, len(active)))]
                ops.append(
                    f"move,{px:.5f},{py:.5f},"
                    f"{px + float(rng.normal(0, 0.15)):.5f},"
                    f"{py + float(rng.normal(0, 0.15)):.5f}"
                )
            else:
                px, py = active[int(rng.integers(0, len(active)))]
                ops.append(f"delete,{px:.5f},{py:.5f}")
        stream_path = self.tmp / "fuzz_stream.csv"
        stream_path.write_text("\n".join(ops) + "\n")
        out_dir = self.tmp / "fuzz"
        result = subprocess.run(
            [str(CLUSTER), "dynamic", str(base_path), str(stream_path), str(out_dir),
             "--local-edge-refresh", "--local-bucket-refresh", "--verify-dynamic"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log = pd.read_csv(out_dir / "dynamic_log.csv")
            self.assertTrue((log.verification_mismatches == 0).all())
            self.assertTrue((log.edge_refresh_mismatches == 0).all())
            self.assertTrue((log.bucket_refresh_mismatches == 0).all())
        else:
            combined = (result.stderr + result.stdout).lower()
            log_path = out_dir / "dynamic_log.csv"
            caught = "verif" in combined
            if log_path.exists():
                last = pd.read_csv(log_path).iloc[-1]
                caught = caught or bool(
                    last.get("verification_mismatches", 0)
                    or last.get("edge_refresh_mismatches", 0)
                    or last.get("bucket_refresh_mismatches", 0)
                )
            self.assertTrue(caught, msg=f"non-zero exit not from the verifier: {combined[:400]}")

    def test_dynamic_coincident_points_stay_exact_without_crashing(self):
        """Coincident-coordinate inserts/moves must not crash or break exactness.

        ``dt_.insert`` returns the *existing* vertex when a point coincides
        exactly with one; previously the dynamic path overwrote that vertex's id,
        aliasing two ids onto a single CGAL vertex. A later delete then freed the
        shared vertex and left a dangling handle, segfaulting the next local
        refresh -- reachable on real catalogs, which contain duplicate
        coordinates. The path now rejects a coincident op as a no-op, preserving
        the distinct-coordinate (general-position) invariant so the verifier still
        certifies every update. Without the fix this stream crashes on the
        post-delete insert; with it the run completes with zero mismatches.
        """
        base_path = self.tmp / "base.csv"
        self.write_two_blobs(base_path)
        pts = pd.read_csv(base_path, header=None)
        x0, y0 = float(pts.iloc[10, 0]), float(pts.iloc[10, 1])
        x1, y1 = float(pts.iloc[20, 0]), float(pts.iloc[20, 1])
        x2, y2 = float(pts.iloc[30, 0]), float(pts.iloc[30, 1])
        stream_path = self.tmp / "coincident_stream.csv"
        stream_path.write_text(
            f"insert,{x0!r},{y0!r},0\n"          # exact duplicate -> insert rejected
            f"delete,{x0!r},{y0!r}\n"            # remove that base location
            f"insert,{x0 + 1e-3!r},{y0!r},0\n"   # nearby insert forces a local refresh
            f"move,{x2!r},{y2!r},{x1!r},{y1!r}\n"  # move onto another point -> move rejected
        )
        out_dir = self.tmp / "coincident"
        result = subprocess.run(
            [str(CLUSTER), "dynamic", str(base_path), str(stream_path), str(out_dir),
             "--local-edge-refresh", "--local-bucket-refresh", "--verify-dynamic"],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            msg="coincident-point stream crashed or failed verification: "
                f"{(result.stderr + result.stdout)[:400]}",
        )
        log = pd.read_csv(out_dir / "dynamic_log.csv")
        self.assertEqual(len(log), 4)
        self.assertTrue((log.verification_mismatches == 0).all())
        self.assertTrue((log.edge_refresh_mismatches == 0).all())
        self.assertTrue((log.bucket_refresh_mismatches == 0).all())

    # ----- Incremental-DBSCAN baseline + faithfulness pipeline (reframed) -----
    def test_incremental_dbscan_stream_smoke(self):
        if not IDBSCAN.exists():
            self.skipTest(f"{IDBSCAN} not built")
        base = self.tmp / "base.csv"
        self.write_two_blobs(base)
        stream = self.tmp / "stream.csv"
        stream.write_text("insert,0.0,0.0,0\ndelete,-2.0,0.0\nmove,2.0,0.0,2.1,0.1\n")
        out = self.tmp / "idb.csv"
        subprocess.run(
            [str(IDBSCAN), str(base), str(stream), str(out),
             "--eps", "0.5", "--min-pts", "4", "--every", "1"],
            check=True,
        )
        df = pd.read_csv(out)
        self.assertEqual(df.operation.tolist(), ["insert", "delete", "move"])
        self.assertIn("incr_ari", df.columns)
        self.assertIn("n_active", df.columns)
        self.assertTrue((df.incr_ari <= 1.0 + 1e-9).all())

    def test_dynamic_faithfulness_smoke(self):
        if not IDBSCAN.exists():
            self.skipTest(f"{IDBSCAN} not built")
        out_dir = self.tmp / "faith"
        subprocess.run(
            ["python3", str(FAITHFULNESS), "--datasets", "chain_noise",
             "--n", "300", "--ops", "40", "--seeds", "42",
             "--rate-sweep", "0.5", "--every", "10",
             "--out-dir", str(out_dir), "--cluster-exe", str(CLUSTER),
             "--idbscan-exe", str(IDBSCAN)],
            check=True, cwd=ROOT,
        )
        summary = pd.read_csv(out_dir / "faithfulness_summary.csv")
        self.assertIn("baseline_final_ari_mean", summary.columns)
        # DelauCluster maintains the clustering exactly under insert/delete/move.
        self.assertEqual(int(summary["delaucluster_total_mismatches"].sum()), 0)

    # ----- Static sanity-check tool + real-data normalization -----
    def test_run_baseline_comparison_writes_tables(self):
        out_dir = self.tmp / "run_baseline_comparison"
        subprocess.run(
            ["python3", str(BASELINE_COMPARISON), "--datasets", "chain_noise",
             "--seeds", "42", "--base-limits", "80",
             "--algorithms", "DelauCluster,DelauClusterAuto,KMeans,DBSCAN,BIRCH",
             "--out-dir", str(out_dir), "--cluster-exe", str(CLUSTER)],
            check=True, cwd=ROOT,
        )
        main = pd.read_csv(out_dir / "baseline_comparison_main_table.csv")
        summary = pd.read_csv(out_dir / "baseline_comparison_summary.csv")
        self.assertIn("DelauClusterAuto", set(main["algorithm"]))
        self.assertTrue((summary["status"] == "ok").all())

    def test_generate_real_spatial_local_csv_normalizes_coordinates(self):
        raw_path = self.tmp / "raw_real.csv"
        raw_path.write_text(
            "longitude,latitude,category\n"
            "-122.42,37.77,A\n-122.41,37.78,B\n-122.40,37.79,A\n0,0,C\nnot-a-number,37.8,D\n"
        )
        out_dir = self.tmp / "real_spatial"
        subprocess.run(
            ["python3", str(GENERATE_DATASETS), "--source", "real", "--datasets", "",
             "--local-csv", f"toy={raw_path},longitude,latitude,category",
             "--out-dir", str(out_dir), "--include-labels"],
            check=True, cwd=ROOT,
        )
        normalized = pd.read_csv(out_dir / "toy.csv")
        self.assertEqual(list(normalized.columns), ["x", "y", "label"])
        self.assertEqual(len(normalized), 4)
        self.assertTrue(np.isfinite(normalized[["x", "y"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
