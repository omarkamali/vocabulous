# Parallel Training Benchmark Report

This document summarizes the comparison between sequential and parallel training modes in Vocabulous using the new `clean_workers` and `token_workers` options introduced in version 0.1.2.

## Dataset Setup

- **Rows:** 200,000 samples (balanced across `en`, `es`, `fr`)
- **Vocabulary:** 10,000 randomly generated alphanumeric tokens (length 3	3	10)
- **Sentence length:** 20 words per sentence sampled from the vocabulary
- **Noise:** 5% chance of randomly dropping a character from a token (typo simulation)
- **Train/Eval split:** identical dataframes (mirrors `train_parallel_compare.py` harness)
- **Training config:** `cycles=1`, `base_confidence=0.2`, `confidence_margin=0.2`

Generation and benchmarking are automated via [`benchmarks/train_parallel_compare.py`](./train_parallel_compare.py).

## Environment

- macOS (pyenv CPython 3.12.3 via `uv run`)
- Vocabulous 0.1.2-dev
- Command: `uv run python benchmarks/train_parallel_compare.py --rows 200000 --clean-workers 4 --token-workers 4`

## Results

| Mode        | Clean Workers | Token Workers | Wall Time (s) | Speedup vs Sequential |
|-------------|---------------|----------------|---------------|-----------------------|
| Sequential  | 1             | 1              | 393.199       | 1.00	             |
| Parallel    | 4             | 4              | 120.446       | 3.26	             |

Additional observations:

- Parallel run spawned `spawn` multiprocessing pools for cleaning and sentence tokenization, dramatically reducing pre-scoring time.
- Dictionary contents and evaluation predictions matched exactly between both runs (validated in harness).
- After deduplication, the dataset remained at 200k sentences (thanks to random vocabulary), so worker pools had meaningful work to perform.

## How to Reproduce

```bash
uv run python benchmarks/train_parallel_compare.py \
  --rows 200000 \
  --clean-workers 4 \
  --token-workers 4
```

Adjust `--rows`, `--clean-workers`, and `--token-workers` to explore scaling characteristics. The script exits with a non-zero code if dictionaries or predictions diverge between sequential and parallel runs.
