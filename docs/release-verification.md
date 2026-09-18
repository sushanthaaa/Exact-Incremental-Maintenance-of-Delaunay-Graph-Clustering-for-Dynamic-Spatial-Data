# Release verification — 2026-09-18

These are fresh checks in the clean release checkout, separate from the archived paper results. The original C++ engine and original experiment scripts were copied unchanged. The smoke driver now discovers all tests; four tests cover the new input-restoration helper.

| Check | Observed result |
|---|---|
| macOS Release build | Passed; Apple Clang 21.0.0, CGAL 6.1.1, CMake 4.3.2 |
| Unit/regression suite | 18 passed (14 existing CLI/experiment tests plus 4 restoration tests) |
| Verified synthetic smoke | 60 updates, zero mismatches |
| Adversarial suite | Eight cases, 179 updates, zero mismatches |
| Input archives | Five archive and decompressed SHA-256 checks passed; restoration verified |
| Full selector sensitivity | 40 cases × 17 variants = 680 rows; default replay assertions passed |
| Sensitivity versus historical results | Both runs and summary CSVs match with relative/absolute tolerance 1e-12 |
| AIS replay | 5,752 verified updates, zero mismatches and zero fallbacks; summary matches historical results at tolerance 1e-12 |
| Ubuntu 24.04 container (ARM64) | Built with GCC 13.3.0 / CGAL 5.6; all 18 tests, 60-update smoke, snapshot integrity, and 179-update adversarial suite passed |
| Python syntax | All scripts and tests compiled |
| Imported file integrity | Working-tree and staged Git bytes match the import manifest |
| Release file audit | No generated builds, temporary runs, submission PDFs, detected credentials, or personal absolute machine paths in staged files |

Machine-readable details and fresh summary CSVs are in [artifacts/release-checks/2026-09-18](../artifacts/release-checks/2026-09-18). Host Python package versions are recorded in [environment/tested-requirements.txt](../environment/tested-requirements.txt); container versions are in the verification JSON. The container recipe installs current compatible dependencies and is not a fully locked environment.

The complete historical 36,755-update study and the published timing suite were not rerun for this release. Historical evidence remains in `artifacts/paper/`. These checks establish the observed build and reproduction results above; they do not prove every scientific claim or guarantee behavior on every platform. The GitHub workflow repeats the quick checks on Ubuntu x86-64 when enabled.
