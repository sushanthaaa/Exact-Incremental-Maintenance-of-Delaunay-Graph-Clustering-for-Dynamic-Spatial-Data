# DelauCluster

Research code and reproducibility materials for **Exact Incremental Maintenance of Delaunay-Graph Clustering for Dynamic Spatial Data**, accepted at IEEE ICTAI 2026 (paper 222).

DelauCluster maintains a locally normalized Delaunay-component clustering of **2D points** under insertion, deletion, and movement. It selects a threshold at initialization, freezes that threshold for the session, updates the triangulation and affected edges, and checks the resulting labels with a global connected-component guard.

**Exactness means agreement with recomputation at the same frozen threshold.** It does not mean agreement with a newly selected threshold, or with ground-truth categories. The implementation uses fixed selector constants. The global guard prevents a sublinear worst-case update-time claim. See [method and scope](docs/architecture.md).

## Quick start

Requires a C++17 compiler, CMake 3.16+, CGAL, and Python 3.10+ for experiments. Python 3.12 is used in CI. Install native dependencies with either:

```bash
# macOS (Homebrew)
brew install cmake cgal

# Ubuntu / Debian
sudo apt-get install build-essential cmake libcgal-dev python3-venv
```

Then, from this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run
```

`bash run` builds both C++ executables, runs a 60-update verified smoke experiment, and runs the test suite. Generated files go under ignored `build/` and `results/` directories. For C++-only use:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

## Use the clustering tool

Point input is a **headerless** CSV: `x,y` or `x,y,label`. Optional labels are for evaluation. Coordinates must use a consistent planar coordinate system; the tool does not infer a geographic projection.

```bash
./build/cluster static points.csv results/static
./build/cluster dynamic base.csv stream.csv results/dynamic --local-bucket-refresh --verify-dynamic
```

Dynamic stream rows are `insert,x,y[,label]`, `delete,x,y`, or `move,old_x,old_y,new_x,new_y`. With no explicit threshold, the dynamic command selects one from the base points and freezes it. `--local-bucket-refresh` also enables incremental geometry, local edge refresh, and local relabeling. `--verify-dynamic` additionally runs the expensive reference verifier after every update; omit it when measuring production update latency. The global guard remains enabled. `--no-guard` is an ablation option and can produce incorrect partitions.

The second executable, `incremental_dbscan_stream`, provides both a grow-only union-find comparator and an exact DBSCAN mode. These implement a different clustering rule; cross-method runtime comparisons must account for that distinction.

## Reproduce the research

```bash
python scripts/restore_data.py
python scripts/run_adversarial_exactness.py
```

Five compressed, checksummed input snapshots are included. Restoration is offline and refuses to overwrite changed local inputs. The [reproduction guide](docs/reproducibility.md) gives the experiment commands, dependencies, and evaluation definitions. The [evidence map](docs/provenance.md) connects archived measurements to their scripts.

The paper reports zero verifier mismatches over **36,755** updates, plus a separate **179-update** adversarial suite. Those are archived research results, not the size of the quick-start test. The [release verification record](docs/release-verification.md) states exactly which checks were rerun for this release. Performance measurements depend on hardware, software versions, and workload; the historical timings are preserved separately from new runs.

## Repository layout

| Path | Contents |
|---|---|
| `include/`, `src/` | C++ clustering engine and CLI tools |
| `scripts/` | Dataset generators, experiment runners, and result summaries |
| `tests/` | CLI regression tests and snapshot restoration tests |
| `data/` | Compressed paper inputs, checksums, and source information |
| `artifacts/paper/` | Selected historical result CSVs and selector traces |
| `docs/` | Method scope, reproduction instructions, and provenance |
| `environment/` | Container recipe and tested Python dependency versions |
| `.github/workflows/` | Build and verification workflow |

The repository contains the research pipeline and evidence. Submission receipts, correspondence, review drafts, local build products, and the publisher-certified manuscript PDF are not part of this software release.

## Citation and license

Paper authors: Sushanth Purushothama, Gene Eu Jan, Chaomin Luo, Bor-Shing Lin, and Pooja R. Use [CITATION.cff](CITATION.cff) for the software citation. The paper was accepted at ICTAI 2026; publication identifiers will be added when available.

The software retains its existing **GPL-3.0-or-later** license; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Third-party data retain their source terms and attribution, documented in [data/README.md](data/README.md). For changes and bug reports, see [CONTRIBUTING.md](CONTRIBUTING.md).
