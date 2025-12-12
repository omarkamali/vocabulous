#!/usr/bin/env python
"""Benchmark for parallel scoring in Vocabulous.

This script benchmarks the scoring stage of the pipeline with and without
parallelization, measuring performance and validating equivalence.

FINDINGS:
---------
Scoring does NOT benefit from multiprocessing because:
1. The operation is very fast (~170k rows/s sequential)
2. Process spawn overhead + serializing word_lang_freq dict dominates
3. Parallel scoring is ~2x SLOWER than sequential

Recommendation: Keep scoring sequential (default). The parallel code path
exists for API consistency but should not be used in practice.

Usage:
    uv run python benchmarks/bench_scoring.py
    # Or with custom parameters:
    uv run python benchmarks/bench_scoring.py --rows 100000 --workers 1,4
"""

import argparse
import random
import string
import time
import sys
import os

# Ensure we can import from parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def synth_dataset(n_rows: int, langs: list[str] = None, words_per_sentence: int = 10):
    """Generate synthetic dataset with random words."""
    if langs is None:
        langs = ["en", "fr", "es", "de"]
    
    rows = []
    for i in range(n_rows):
        lang = langs[i % len(langs)]
        # Generate random words
        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(words_per_sentence)
        ]
        text = " ".join(words)
        rows.append({"text": text, "lang": lang})
    return rows


def time_fn(fn, *args, **kwargs):
    """Time a function call and return elapsed seconds."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def bench_scoring(n_rows: int = 200_000, worker_options: tuple = (1, 2, 4, 8)):
    """Benchmark scoring with different worker counts."""
    import pandas as pd
    from vocabulous import Vocabulous
    
    print(f"\n{'='*60}")
    print(f"Scoring Benchmark: {n_rows:,} rows")
    print(f"{'='*60}")
    
    # Generate dataset
    print("Generating synthetic dataset...")
    data = synth_dataset(n_rows, langs=["en", "fr", "es", "de"])
    df = pd.DataFrame(data)
    
    # Train a model first (single-threaded, small subset for speed)
    print("Training model on subset...")
    model = Vocabulous()
    train_subset = data[:min(50_000, n_rows)]
    model, _ = model.train(train_subset, cycles=1)
    print(f"Dictionary size: {len(model.word_lang_freq):,} words")
    
    # Pre-clean the data so we only benchmark scoring (use parallel cleaning)
    print("Pre-cleaning data...")
    df["text"] = model._clean_series(df["text"], workers=4)
    df = df[df["text"] != ""].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df):,}")
    
    # Benchmark scoring with different worker counts
    results = {}
    baseline_scores = None
    
    for workers in worker_options:
        print(f"\nScoring with workers={workers}...")
        elapsed, scored_df = time_fn(model._score, df.copy(), already_clean=True, workers=workers)
        
        rows_per_sec = len(df) / elapsed
        results[workers] = {
            "time": elapsed,
            "rows_per_sec": rows_per_sec,
        }
        
        # Store baseline for equivalence check
        if workers == 1:
            baseline_scores = scored_df["scores"].tolist()
        else:
            # Equivalence check
            parallel_scores = scored_df["scores"].tolist()
            if baseline_scores is not None:
                matches = sum(1 for a, b in zip(baseline_scores, parallel_scores) if a == b)
                pct = 100.0 * matches / len(baseline_scores)
                results[workers]["equivalence"] = pct
                if pct < 100.0:
                    print(f"  WARNING: {100-pct:.2f}% scores differ from baseline!")
        
        print(f"  Time: {elapsed:.2f}s | {rows_per_sec:,.0f} rows/s")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Rows/s':<15} {'Speedup':<10} {'Equiv %':<10}")
    print("-" * 60)
    
    baseline_time = results[1]["time"]
    for workers, r in results.items():
        speedup = baseline_time / r["time"]
        equiv = r.get("equivalence", 100.0)
        print(f"{workers:<10} {r['time']:<12.2f} {r['rows_per_sec']:<15,.0f} {speedup:<10.2f}x {equiv:<10.1f}")
    
    return results


def bench_equivalence(n_rows: int = 10_000):
    """Detailed equivalence test between sequential and parallel scoring."""
    import pandas as pd
    from vocabulous import Vocabulous
    
    print(f"\n{'='*60}")
    print(f"Equivalence Test: {n_rows:,} rows")
    print(f"{'='*60}")
    
    # Generate dataset
    data = synth_dataset(n_rows, langs=["en", "fr", "es"])
    df = pd.DataFrame(data)
    
    # Train model
    model = Vocabulous()
    model, _ = model.train(data[:5000], cycles=1)
    
    # Pre-clean
    df["text"] = df["text"].apply(model._clean_text)
    df = df[df["text"] != ""].reset_index(drop=True)
    
    # Score sequentially
    print("Scoring sequentially (workers=1)...")
    seq_df = model._score(df.copy(), already_clean=True, workers=1)
    
    # Score in parallel
    print("Scoring in parallel (workers=4)...")
    par_df = model._score(df.copy(), already_clean=True, workers=4)
    
    # Compare
    seq_scores = seq_df["scores"].tolist()
    par_scores = par_df["scores"].tolist()
    
    mismatches = []
    for i, (s, p) in enumerate(zip(seq_scores, par_scores)):
        if s != p:
            mismatches.append((i, s, p))
    
    if mismatches:
        print(f"\n❌ FAILED: {len(mismatches)} mismatches found!")
        for i, s, p in mismatches[:5]:
            print(f"  Row {i}: seq={s}, par={p}")
        return False
    else:
        print(f"\n✅ PASSED: All {len(seq_scores):,} scores match exactly!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel scoring")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows")
    parser.add_argument("--workers", type=str, default="1,2,4,8", help="Comma-separated worker counts")
    parser.add_argument("--equiv-only", action="store_true", help="Run only equivalence test")
    args = parser.parse_args()
    
    worker_options = tuple(int(w) for w in args.workers.split(","))
    
    if args.equiv_only:
        success = bench_equivalence(n_rows=10_000)
        sys.exit(0 if success else 1)
    
    # Run equivalence test first
    print("Running equivalence test...")
    if not bench_equivalence(n_rows=10_000):
        print("Equivalence test failed! Aborting benchmark.")
        sys.exit(1)
    
    # Run performance benchmark
    bench_scoring(n_rows=args.rows, worker_options=worker_options)


if __name__ == "__main__":
    # Required for multiprocessing spawn on macOS
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
