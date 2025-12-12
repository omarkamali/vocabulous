#!/usr/bin/env python
"""Benchmark for integrated parallel dictionary accumulation.

Tests the actual _accumulate_dict method in Vocabulous.

Usage:
    uv run python benchmarks/bench_dict_accumulation_integrated.py
"""

import random
import string
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def synth_dataset(n_rows: int, langs: list[str] = None, words_per_sentence: int = 10):
    if langs is None:
        langs = ["en", "fr", "es", "de"]
    rows = []
    for i in range(n_rows):
        lang = langs[i % len(langs)]
        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(words_per_sentence)
        ]
        text = " ".join(words)
        rows.append({"text": text, "lang": lang})
    return rows


def time_fn(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def main():
    import pandas as pd
    from vocabulous import Vocabulous
    from vocabulous.vocabulous import _sentence_record
    
    n_rows = 200_000
    print(f"\n{'='*60}")
    print(f"Integrated Dictionary Accumulation Benchmark: {n_rows:,} rows")
    print(f"{'='*60}")
    
    # Generate dataset
    print("Generating synthetic dataset...")
    data = synth_dataset(n_rows, langs=["en", "fr", "es", "de"])
    df = pd.DataFrame(data)
    
    # Pre-process sentences
    print("Pre-processing sentences...")
    processed = df.apply(
        lambda row: _sentence_record(row["lang"], row["text"]), axis=1
    )
    processed_df = pd.DataFrame(processed.tolist())
    print(f"Processed {len(processed_df):,} sentences")
    
    # Sequential
    print("\nBenchmarking sequential accumulation...")
    model_seq = Vocabulous()
    elapsed_seq, _ = time_fn(model_seq._accumulate_dict, processed_df, workers=1)
    dict_size_seq = len(model_seq.word_lang_freq)
    print(f"Sequential: {elapsed_seq:.2f}s | {n_rows/elapsed_seq:,.0f} rows/s | {dict_size_seq:,} words")
    
    # Parallel (4 workers)
    print("\nBenchmarking parallel accumulation (workers=4)...")
    model_par = Vocabulous()
    elapsed_par, _ = time_fn(model_par._accumulate_dict, processed_df, workers=4)
    dict_size_par = len(model_par.word_lang_freq)
    print(f"Parallel-4: {elapsed_par:.2f}s | {n_rows/elapsed_par:,.0f} rows/s | {dict_size_par:,} words")
    
    # Equivalence check
    if model_seq.word_lang_freq == model_par.word_lang_freq and model_seq.languages == model_par.languages:
        print("\n✅ EQUIVALENCE: Dictionaries match exactly!")
    else:
        print("\n❌ MISMATCH: Dictionaries differ!")
        return
    
    # Summary
    speedup = elapsed_seq / elapsed_par
    print(f"\n{'='*60}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
