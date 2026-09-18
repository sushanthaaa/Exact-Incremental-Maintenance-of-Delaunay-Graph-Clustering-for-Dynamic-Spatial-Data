# Contributing

Open an issue with a minimal input and update stream, the command, expected and observed behavior, and compiler/CGAL/Python versions. Remove private information from example data and logs.

For changes, create a branch and keep the implementation, tests, and documentation in the same pull request. Run `bash run` and `python scripts/run_adversarial_exactness.py` before requesting review. An algorithm change should include a regression case that would fail without the fix. Explain any change to the clustering rule, selector defaults, numerical handling, or timing protocol.

Do not overwrite `artifacts/paper/` with a new run. Historical results remain evidence for the published experiments. Write fresh results under `results/`, and describe any deliberate update in a separate record with input hashes and environment details. Keep generated binaries, environments, and temporary datasets out of Git.

The source files use the existing GPL-3.0-or-later license. Contributions must be compatible with that license. Third-party data are governed separately by their source terms.
