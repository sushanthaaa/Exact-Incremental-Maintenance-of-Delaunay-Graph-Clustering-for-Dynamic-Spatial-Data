# Reproducing the experiments

Run commands from the repository root after following the build instructions in the README. Install the Python requirements in a virtual environment and keep that environment active. The tested Python package versions for this release are recorded in `environment/tested-requirements.txt`; these describe release verification, not necessarily every historical experiment.

## Quick verification

```bash
bash run
python scripts/restore_data.py
python scripts/run_adversarial_exactness.py
```

The first command compiles the C++ tools and runs the smoke experiment and tests. The adversarial script checks eight cases comprising 179 updates, including an exactly cocircular case outside the theorem's general-position assumption. Inspect its summary CSV and exit status. A successful test of these finite cases is not a proof for arbitrary floating-point inputs.

## Main dynamic correctness workloads

These commands use the original harnesses and their paper configurations. `results/` is ignored and separated from the immutable historical CSVs in `artifacts/paper/`. Do not infer correctness from process completion alone: inspect the verification counters in the generated summary files.

```bash
python scripts/run_dynamic_faithfulness.py --datasets chain_noise,varying_density,touching --n 2000 --ops 1000 --seeds 42,43,44,45,46 --out-dir results/dynamic_faithfulness
python scripts/run_realdata_corrections.py --out-dir results/realdata_corrections
python scripts/run_decluster_experiment.py --verify --out-dir results/decluster
python scripts/run_comcat_revision_experiment.py --out-dir results/comcat_revision
python scripts/run_ais_moves.py --out-dir results/ais_moves
```

Run declustering before the ComCat command to make its curve available to the combined figure. These workloads account for 15,000 + 3,600 + 9,903 + 2,500 + 5,752 = 36,755 archived verified updates; extra deletion-rate sweeps are separate. AIS includes 5,569 movements, 96 insertions, and 87 deletions. The downloaded event catalog is real; randomly generated catalog corrections and the 25-event delayed-withdrawal policy are modeled, not a historical edit log.

## Threshold, guard, and locality studies

```bash
python scripts/run_rethreshold_boundary.py
python scripts/run_revision_isolation.py
python scripts/run_guard_ablation.py --decluster
python scripts/run_ladder_ablation.py
python scripts/run_selector_sensitivity.py --cluster-exe ./build/cluster --out-dir results/selector_sensitivity
python scripts/measure_duplicates.py
```

The selector-sensitivity output directory must not already exist. The script tests eight base-objective weights one at a time at 0.8 and 1.2 times their defaults, holding other constants fixed, over eight synthetic families and five seeds. It first asserts that full-precision trace replay reproduces the default selector, then evaluates the selected thresholds with the unchanged C++ engine. This is a bounded sensitivity study, not a full search of all constants or a held-out tuning exercise.

The frozen-threshold study compares two clustering rules on the same current points, not predicted labels against external truth. Its cache reuse behavior is documented in [data/README.md](../data/README.md). `--no-guard` runs in the ablation intentionally allow incorrect maintenance; never use that flag to support an exactness claim.

## Static comparison and policy summary

```bash
python scripts/run_baseline_comparison.py --datasets all --seeds 42,43,44,45,46 --base-limits 0 --out-dir results/static_5seed_summary
python scripts/summarize_policy_ablation.py
python scripts/summarize_static_significance.py
```

`all` refers to eight synthetic dataset families. The Auto policy comparison is separate from the local-scale maintenance experiments. ARI and NMI in the static comparison use the synthetic labels. Family-level inference treats eight families as the units of comparison rather than treating all 40 seeds as independent datasets. The full static run generates intermediate outputs needed by `summarize_policy_ablation.py`; these are not all checked into Git. Fresh runs can take substantially longer than the smoke check.

## Timing experiments

Run on an otherwise idle machine, one experiment at a time:

```bash
python scripts/run_scaling.py
python scripts/run_incremental_dbscan_comparison.py
python scripts/run_exact_dbscan_deletion_cost.py
python scripts/run_guard_latency.py
python scripts/run_ladder_ablation.py --no-verify --out-dir results/ladder_latency
```

The paper's historical T2 timings used an Apple M3, 8 GB RAM, macOS 26.5.1, CGAL 6.1.1, a Release build, and single-threaded experiments. Do not expect identical timings across machines or compare a verifier-enabled run against verifier-disabled reported latencies. The scaling script includes a 50,000-point cost comparison in addition to its size sweep. The guard-on/off difference estimates the effect of disabling the guard in that workload; it is not an independent proof of an asymptotic speedup.

## Optional container

```bash
docker build -f environment/Dockerfile -t delaucluster .
docker run --rm delaucluster
```

The recipe builds the tools and the default container command runs the smoke driver and tests. It provides an Ubuntu environment, not an emulation of the historical macOS timing machine. See the release verification record for whether this recipe was executed locally.

## Comparing to archived evidence

The [provenance map](provenance.md) identifies the original CSV for each experiment. Compare deterministic counts, partitions, and correctness counters before comparing timings. Dependency versions can change data generators and metrics slightly; investigate differences rather than overwriting historical evidence. Preserve command lines, package versions, input hashes, and random seeds for new experiments.
