# Research evidence and provenance

This repository was prepared from the working tree of the earlier DelauCluster research repository. `artifacts/source_manifest.json` records its Git HEAD and the SHA-256 hashes of imported files. Some research changes were uncommitted, so that HEAD alone does not identify the released working tree. The current repository starts a separate, curated history.

`artifacts/paper/` contains selected historical CSVs, available provenance notes, and full-precision selector traces. These files are preserved as research records; they are not replaced by release verification runs. Historical commit identifiers refer to the earlier research repository, not this new repository. Historical notes may name old manuscript paths or table numbers; use the experiment names below as the stable mapping.

| Experiment / paper result | Directory under `artifacts/paper/` | Script in `scripts/` |
|---|---|---|
| Synthetic fixed-rule faithfulness | `dynamic_faithfulness` | `run_dynamic_faithfulness.py` |
| NYC and western-US modeled corrections | `realdata_corrections` | `run_realdata_corrections.py` |
| ComCat delayed withdrawal | `comcat_revision` | `run_comcat_revision_experiment.py` |
| Earthquake declustering | `decluster` | `run_decluster_experiment.py`, `run_decluster_spatial.py` |
| AIS observed vessel movements | `ais_moves_20260711_p3b` | `run_ais_moves.py` |
| Eight adversarial cases / 179 updates | `adversarial_exactness_20260716_R3` | `run_adversarial_exactness.py` |
| Size sweep, full rebuild comparison, fallback counts | `scaling_20260716_T2` | `run_scaling.py` |
| DBSCAN runtime comparison | `incremental_dbscan_comparison_20260716_T2` | `run_incremental_dbscan_comparison.py` |
| Exact DBSCAN deletion cost | `exact_dbscan_deletion_cost_20260716_T2` | `run_exact_dbscan_deletion_cost.py` |
| Guard cost and unconditional relabel comparison | `guard_latency_20260716_T2` | `run_guard_latency.py` |
| Locality ladder: correctness / timing | `ladder_ablation_20260710_p3`, `ladder_latency_20260716_T2` | `run_ladder_ablation.py` |
| Guard-disabled disagreement | `guard_ablation_20260710_p3` | `run_guard_ablation.py` |
| Static policy ablation | `ablation_policy_20260710_p3` | `summarize_policy_ablation.py` |
| Static ARI/NMI | `static_5seed_summary` | `run_baseline_comparison.py` |
| Family-level significance | `static_significance_20260716_R3` | `summarize_static_significance.py` |
| Frozen threshold versus reselected threshold | `rethreshold_boundary_20260710_p3` | `run_rethreshold_boundary.py` |
| Revision isolation | `revision_isolation_20260710_p3` | `run_revision_isolation.py` |
| Duplicate census | `duplicates_20260716_R3` | `measure_duplicates.py` |
| Camera-ready selector weight sensitivity | `selector_sensitivity` | `run_selector_sensitivity.py` |

The main dynamic total is 36,755 verified updates with zero mismatches; the adversarial 179 updates are additional. The timing directories with suffix `20260716_T2` supply the final historical timings. Earlier timing vintages are deliberately omitted to avoid presenting conflicting measurements as equally current.

The sensitivity archive includes `runs.csv`, `summary.csv`, each generated input, and default full-precision threshold traces. `native_checks.json` is an additional historical 40-case cross-check record; the temporary modified C++ build used for that cross-check is not included. The public replay harness is included and uses the unchanged production executable at explicit thresholds. The worst observed default-versus-perturbed ARI (about 0.22 for one case) must not be hidden by the stable aggregate mean.

## What is and is not archived

This is a compact research artifact, not every scratch file from the development directory. It includes the engine, experiment harnesses, tests, exact cached paper inputs, and selected numerical outputs. Large intermediate baseline directories, temporary binaries, duplicate result vintages, internal review drafts, and submission documents are excluded. The scripts regenerate intermediate experiment outputs under `results/`.

[Data provenance](../data/README.md) documents input origins, the historical row-count metadata discrepancy, and the missing original AIS acquisition recipe. The archived data bytes are verifiable despite those provenance limits. [Release verification](release-verification.md) separately records fresh checks and their scope.
