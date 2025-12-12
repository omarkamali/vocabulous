#!/usr/bin/env python
"""Benchmark for dictionary accumulation in Vocabulous.

This script benchmarks the dictionary building stage of the pipeline,
which iterates over processed sentences and accumulates word-language counts.

Usage:
    uv run python benchmarks/bench_dict_accumulation.py
    uv run python benchmarks/bench_dict_accumulation.py --rows 500000 --workers 1,2,4,8
"""

import argparse
import random
import string
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def synth_dataset(n_rows: int, langs: list[str] = None, words_per_sentence: int = 10):
    """Generate synthetic dataset with random words."""
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


def _accumulate_chunk(payload):
    """Worker function for parallel accumulation."""
    langs_list, words_list = payload
    partial_freq = {}
    partial_langs = set()
    for lang, words in zip(langs_list, words_list):
        partial_langs.add(lang)
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = {}
            if lang not in partial_freq[word]:
                partial_freq[word][lang] = 0
            partial_freq[word][lang] += 1
    return partial_freq, partial_langs


def time_fn(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def bench_dict_accumulation(n_rows: int = 200_000, worker_options: tuple = (1, 2, 4, 8)):
    """Benchmark dictionary accumulation with different worker counts."""
    import pandas as pd
    from tqdm import tqdm
    from vocabulous import Vocabulous
    from vocabulous.vocabulous import _sentence_record, _split_indices
    import multiprocessing as mp
    
    print(f"\n{'='*60}")
    print(f"Dictionary Accumulation Benchmark: {n_rows:,} rows")
    print(f"{'='*60}")
    
    # Generate dataset
    print("Generating synthetic dataset...")
    data = synth_dataset(n_rows, langs=["en", "fr", "es", "de"])
    df = pd.DataFrame(data)
    
    # Pre-process sentences (simulate what _process_sentences returns)
    print("Pre-processing sentences...")
    processed = df.apply(
        lambda row: _sentence_record(row["lang"], row["text"]), axis=1
    )
    processed_df = pd.DataFrame(processed.tolist())
    print(f"Processed {len(processed_df):,} sentences")
    
    # Sequential accumulation (baseline)
    def accumulate_sequential(processed_df):
        word_lang_freq = {}
        languages = set()
        for _, result in processed_df.iterrows():
            lang = result["lang"]
            words = result["words"]
            languages.add(lang)
            for word in words:
                if word not in word_lang_freq:
                    word_lang_freq[word] = {}
                if lang not in word_lang_freq[word]:
                    word_lang_freq[word][lang] = 0
                word_lang_freq[word][lang] += 1
        return word_lang_freq, languages
    
    print("\nBenchmarking sequential accumulation...")
    elapsed, (baseline_dict, baseline_langs) = time_fn(accumulate_sequential, processed_df)
    print(f"Sequential: {elapsed:.2f}s | {n_rows/elapsed:,.0f} rows/s | {len(baseline_dict):,} words")
    
    # Now let's see if we can parallelize this
    # The challenge: we need to merge partial dicts from workers
    
    def merge_dicts(partials):
        """Merge partial word_lang_freq dicts."""
        merged = {}
        merged_langs = set()
        for partial_freq, partial_langs in partials:
            merged_langs.update(partial_langs)
            for word, lang_counts in partial_freq.items():
                if word not in merged:
                    merged[word] = {}
                for lang, count in lang_counts.items():
                    merged[word][lang] = merged[word].get(lang, 0) + count
        return merged, merged_langs
    
    def accumulate_parallel(processed_df, workers):
        splits = _split_indices(len(processed_df), workers)
        langs = processed_df["lang"].tolist()
        words = processed_df["words"].tolist()
        
        ctx = mp.get_context("spawn")
        tasks = [
            ([langs[i] for i in split], [words[i] for i in split])
            for split in splits
        ]
        
        with ctx.Pool(processes=len(tasks)) as pool:
            partials = pool.map(_accumulate_chunk, tasks)
        
        return merge_dicts(partials)
    
    results = {"sequential": {"time": elapsed, "dict_size": len(baseline_dict)}}
    
    for workers in worker_options:
        if workers == 1:
            continue
        print(f"\nBenchmarking parallel accumulation (workers={workers})...")
        elapsed, (parallel_dict, parallel_langs) = time_fn(accumulate_parallel, processed_df, workers)
        
        # Equivalence check
        if parallel_dict == baseline_dict and parallel_langs == baseline_langs:
            equiv = "✅ MATCH"
        else:
            equiv = "❌ MISMATCH"
        
        results[workers] = {"time": elapsed, "dict_size": len(parallel_dict), "equiv": equiv}
        print(f"Parallel ({workers}): {elapsed:.2f}s | {n_rows/elapsed:,.0f} rows/s | {equiv}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    baseline_time = results["sequential"]["time"]
    print(f"{'Mode':<15} {'Time (s)':<12} {'Rows/s':<15} {'Speedup':<10}")
    print("-" * 55)
    print(f"{'sequential':<15} {baseline_time:<12.2f} {n_rows/baseline_time:<15,.0f} {'1.00x':<10}")
    for workers, r in results.items():
        if workers == "sequential":
            continue
        speedup = baseline_time / r["time"]
        print(f"{'parallel-'+str(workers):<15} {r['time']:<12.2f} {n_rows/r['time']:<15,.0f} {speedup:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dictionary accumulation")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows")
    parser.add_argument("--workers", type=str, default="1,2,4,8", help="Comma-separated worker counts")
    args = parser.parse_args()
    
    worker_options = tuple(int(w) for w in args.workers.split(","))
    bench_dict_accumulation(n_rows=args.rows, worker_options=worker_options)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
