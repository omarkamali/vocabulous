## [0.1.1] - 2025-11-05

### Added
- Benchmarks: `benchmarks/benchmark_vocabulous.py` with `benchmark_output.txt` with results.
- Multiple scoring backends surfaced and documented: default `apply`, plus `vectorized`, `numba`, `sparse`.
- Trusted publishing via GitHub Actions with OIDC (`.github/workflows/publish.yml`).
- Upgraded publish script to a more robust Python implementation (`scripts/publish.py`).
- Increased testing coverage from Python 3.8 to 3.14.
