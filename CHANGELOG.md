## [0.1.3] - 2025-12-12

### Added
- **`num_proc` parameter** on `train()`: a single knob that drives all parallel stages (sentence expansion, cleaning, tokenization). Individual stage workers can still be overridden.
- Chunked + parallel sentence expansion with configurable `_sentence_chunk_size`, `sentence_workers`, tqdm progress, and `benchmarks/run_sentence_expansion.py` runner.
- README + `parallel_training_report.md` updates documenting the sentence-expansion benchmark methodology (5M rows, chunk size 10k, workers 1–16) and reproduction commands.

### Changed
- Default benchmarks now cover cleaning/tokenization and sentence expansion stages; report clarifies how to tune chunk size for better scaling.
- README examples updated to use `num_proc` for simpler parallel training.

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
