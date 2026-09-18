# Experiment scripts

Run from the repository root with the Python environment active and C++ tools built. Most experiment runners support `--help`; helper modules are imported by those runners. Complete commands are in [the reproduction guide](../docs/reproducibility.md).

| Group | Files |
|---|---|
| Input construction and shared metrics | `generate_datasets.py`, `dynamic_stream.py`, `benchmark_baselines.py` |
| Snapshot integrity and restoration | `restore_data.py` |
| Synthetic and real-catalog updates | `run_dynamic_faithfulness.py`, `run_realdata_corrections.py` |
| Cached ComCat workloads | `run_comcat_revision_experiment.py`, `run_decluster_experiment.py`, `run_decluster_spatial.py`, `run_revision_isolation.py` |
| Observed AIS movement workload | `run_ais_moves.py` |
| Boundary and correctness cases | `run_rethreshold_boundary.py`, `run_adversarial_exactness.py`, `measure_duplicates.py` |
| Cost and guard/locality ablations | `run_scaling.py`, `run_guard_latency.py`, `run_guard_ablation.py`, `run_ladder_ablation.py` |
| DBSCAN timing studies | `run_incremental_dbscan_comparison.py`, `run_exact_dbscan_deletion_cost.py` |
| Static policy and quality evaluation | `run_baseline_comparison.py`, `summarize_policy_ablation.py`, `summarize_static_significance.py` |
| Fixed-weight sensitivity | `run_selector_sensitivity.py` |

The experiment scripts are retained from the research working tree without behavioral changes for this release. Some historical comments use shorthand such as “real corrections”: the input catalogs are real, but the generated delete/move corrections and review-lag policy are modeled. The authoritative interpretation and cache caveats are in the reproduction and data documentation.
