## [0.1.2] - 2025-12-07

### Added
- Multiprocessing hooks for training via `clean_workers` and `token_workers`, plus benchmarking harness `benchmarks/train_parallel_compare.py` with deterministic synthetic dataset.
- Markdown benchmark report documenting sequential vs parallel results (`benchmarks/parallel_training_report.md`).
- README updates describing parallel training options, reproduction commands, and 3.26× speedup highlight.

## [0.1.1] - 2025-11-05

### Added
- Benchmarks: `benchmarks/benchmark_vocabulous.py` with `benchmark_output.txt` with results.
- Multiple scoring backends surfaced and documented: default `apply`, plus `vectorized`, `numba`, `sparse`.
- Trusted publishing via GitHub Actions with OIDC (`.github/workflows/publish.yml`).
- Upgraded publish script to a more robust Python implementation (`scripts/publish.py`).
- Increased testing coverage from Python 3.8 to 3.14.
