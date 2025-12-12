import time
import random
import pandas as pd
from vocabulous import Vocabulous


def synth_dataset(n_rows=20000, sentence_len=12, dup_ratio=0.2):
    """Create synthetic multilingual dataset with control over length and duplication.

    Args:
        n_rows: number of rows
        sentence_len: approx token length per sentence (by repeating a base template)
        dup_ratio: fraction of rows duplicated from previously seen rows (to exercise token cache)
    """
    import random
    base = {
        "en": "this is an example sentence in english",
        "es": "esta es una oracion de ejemplo en espanol",
        "fr": "ceci est une phrase d exemple en francais",
    }
    # Build sentences to reach target length by repeating tokens
    def make_text(lang):
        toks = base[lang].split()
        rep = max(1, sentence_len // max(1, len(toks)))
        return (" ".join(toks)) * rep

    langs = list(base.keys())
    texts = {l: make_text(l) for l in langs}
    data = []
    for i in range(n_rows):
        lang = random.choice(langs)
        if dup_ratio > 0 and i > 0 and random.random() < dup_ratio:
            # duplicate a previous row
            data.append(data[random.randrange(0, len(data))])
        else:
            data.append({"lang": lang, "text": texts[lang]})
    return pd.DataFrame(data)


def build_model(vocab_per_lang=50, langs=("en", "es", "fr")):
    """Build a synthetic dictionary of configurable size."""
    model = Vocabulous()
    for lang in langs:
        for i in range(vocab_per_lang):
            w = f"{lang}_w{i}"
            model.word_lang_freq.setdefault(w, {}).setdefault(lang, 0)
            model.word_lang_freq[w][lang] += 1
        model.languages.add(lang)
    return model


def time_fn(fn, *args, repeat=3, **kwargs):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_modes(n_rows_list=(5000, 20000, 50000)):
    print("\n== Benchmark: apply vs vectorized ==")
    for n in n_rows_list:
        df = synth_dataset(n_rows=n)
        model = build_model()

        # Force apply
        model._scoring_choice = "apply"
        ta = time_fn(model._score, df.copy(), already_clean=False)

        # Force vectorized
        model2 = build_model()
        model2._scoring_choice = "vectorized"
        tv = time_fn(model2._score, df.copy(), already_clean=False)

        print(f"n={n:6d} | apply: {ta:.3f}s ({n/ta:.1f} rows/s) | vectorized: {tv:.3f}s ({n/tv:.1f} rows/s)")


def bench_duplication(dup_values=(0.0, 0.5, 0.9), n_rows=20000):
    print("\n== Benchmark: duplication ratio (token cache effect) ==")
    for dup in dup_values:
        df = synth_dataset(n_rows=n_rows, dup_ratio=dup)
        model = build_model()
        model._scoring_choice = "apply"
        ta = time_fn(model._score, df.copy(), already_clean=False)
        print(f"dup={dup:.1f} | apply: {ta:.3f}s ({n_rows/ta:.1f} rows/s)")


def bench_sentence_length(lengths=(5, 20, 50), n_rows=20000):
    print("\n== Benchmark: sentence length ==")
    for L in lengths:
        df = synth_dataset(n_rows=n_rows, sentence_len=L)
        model = build_model()
        # Compare both
        m1 = build_model(); m1._scoring_choice = "apply"
        ta = time_fn(m1._score, df.copy(), already_clean=False)
        m2 = build_model(); m2._scoring_choice = "vectorized"; m2._vectorized_batch_size = 20000
        tv = time_fn(m2._score, df.copy(), already_clean=False)
        tn = float('inf')
        ts = float('inf')
        try:
            m3 = build_model(); m3._scoring_choice = "numba"
            tn = time_fn(m3._score, df.copy(), already_clean=False)
        except Exception:
            pass
        try:
            m4 = build_model(); m4._scoring_choice = "sparse"; m4._vectorized_batch_size = 20000
            ts = time_fn(m4._score, df.copy(), already_clean=False)
        except Exception:
            pass
        extra = (
            (f" | numba: {tn:.3f}s ({n_rows/tn:.1f} rows/s)" if tn != float('inf') else "") +
            (f" | sparse: {ts:.3f}s ({n_rows/ts:.1f} rows/s)" if ts != float('inf') else "")
        )
        print(f"len={L:2d} | apply: {ta:.3f}s ({n_rows/ta:.1f} rows/s) | vectorized: {tv:.3f}s ({n_rows/tv:.1f} rows/s){extra}")


def bench_dict_size(sizes=(50, 500, 5000), n_rows=20000):
    print("\n== Benchmark: dictionary size ==")
    df = synth_dataset(n_rows=n_rows)
    for sz in sizes:
        m1 = build_model(vocab_per_lang=sz); m1._scoring_choice = "apply"
        ta = time_fn(m1._score, df.copy(), already_clean=False)
        m2 = build_model(vocab_per_lang=sz); m2._scoring_choice = "vectorized"; m2._vectorized_batch_size = 20000
        tv = time_fn(m2._score, df.copy(), already_clean=False)
        tn = float('inf')
        ts = float('inf')
        try:
            m3 = build_model(vocab_per_lang=sz); m3._scoring_choice = "numba"
            tn = time_fn(m3._score, df.copy(), already_clean=False)
        except Exception:
            pass
        try:
            m4 = build_model(vocab_per_lang=sz); m4._scoring_choice = "sparse"; m4._vectorized_batch_size = 20000
            ts = time_fn(m4._score, df.copy(), already_clean=False)
        except Exception:
            pass
        extra = (
            (f" | numba: {tn:.3f}s ({n_rows/tn:.1f} rows/s)" if tn != float('inf') else "") +
            (f" | sparse: {ts:.3f}s ({n_rows/ts:.1f} rows/s)" if ts != float('inf') else "")
        )
        print(f"dict={sz:5d} | apply: {ta:.3f}s ({n_rows/ta:.1f} rows/s) | vectorized: {tv:.3f}s ({n_rows/tv:.1f} rows/s){extra}")


def bench_large_n(n=100000):
    print("\n== Benchmark: large-n (vectorized batched) ==")
    df = synth_dataset(n_rows=n, sentence_len=20, dup_ratio=0.3)
    model = build_model(vocab_per_lang=500)
    # Force vectorized and ensure batching
    model._scoring_choice = "vectorized"
    model._vectorized_batch_size = 20000
    tv = time_fn(model._score, df.copy(), already_clean=False)
    print(f"n={n} vectorized-batched: {tv:.3f}s ({n/tv:.1f} rows/s)")


def dataset_from_dict(model: Vocabulous, n_rows=20000, sentence_len=20):
    """Create a dataset whose tokens all exist in the dictionary (high match density)."""
    import random
    langs = list(model.languages)
    words_by_lang = {l: [w for w, langs_map in model.word_lang_freq.items() if l in langs_map] for l in langs}
    data = []
    for _ in range(n_rows):
        l = random.choice(langs)
        toks = [random.choice(words_by_lang[l]) for _ in range(sentence_len)]
        data.append({"lang": l, "text": " ".join(toks)})
    return pd.DataFrame(data)


def bench_high_match(n_rows=20000, sentence_len=20, vocab_sizes=(50, 500, 5000)):
    print("\n== Benchmark: high match density (dataset built from dict) ==")
    for sz in vocab_sizes:
        model = build_model(vocab_per_lang=sz)
        df = dataset_from_dict(model, n_rows=n_rows, sentence_len=sentence_len)
        m1 = build_model(vocab_per_lang=sz); m1._scoring_choice = "apply"
        ta = time_fn(m1._score, df.copy(), already_clean=False)
        m2 = build_model(vocab_per_lang=sz); m2._scoring_choice = "vectorized"; m2._vectorized_batch_size = 20000
        tv = time_fn(m2._score, df.copy(), already_clean=False)
        tn = float('inf')
        ts = float('inf')
        try:
            m3 = build_model(vocab_per_lang=sz); m3._scoring_choice = "numba"
            tn = time_fn(m3._score, df.copy(), already_clean=False)
        except Exception:
            pass
        try:
            m4 = build_model(vocab_per_lang=sz); m4._scoring_choice = "sparse"; m4._vectorized_batch_size = 20000
            ts = time_fn(m4._score, df.copy(), already_clean=False)
        except Exception:
            pass
        extra = (
            (f" | numba: {tn:.3f}s ({n_rows/tn:.1f} rows/s)" if tn != float('inf') else "") +
            (f" | sparse: {ts:.3f}s ({n_rows/ts:.1f} rows/s)" if ts != float('inf') else "")
        )
        print(f"high-match dict={sz:5d} | apply: {ta:.3f}s ({n_rows/ta:.1f} rows/s) | vectorized: {tv:.3f}s ({n_rows/tv:.1f} rows/s){extra}")


def bench_large_n_compare(n=200000, sentence_len=20, dict_size=500):
    print("\n== Benchmark: large-n compare (apply vs vectorized batched) ==")
    model_seed = build_model(vocab_per_lang=dict_size)
    df = dataset_from_dict(model_seed, n_rows=n, sentence_len=sentence_len)
    m_apply = build_model(vocab_per_lang=dict_size); m_apply._scoring_choice = "apply"
    ta = time_fn(m_apply._score, df.copy(), already_clean=False)
    m_vec = build_model(vocab_per_lang=dict_size); m_vec._scoring_choice = "vectorized"; m_vec._vectorized_batch_size = 25000
    tv = time_fn(m_vec._score, df.copy(), already_clean=False)
    tn = float('inf')
    ts = float('inf')
    try:
        m_numba = build_model(vocab_per_lang=dict_size); m_numba._scoring_choice = "numba"
        tn = time_fn(m_numba._score, df.copy(), already_clean=False)
    except Exception:
        pass
    try:
        m_sparse = build_model(vocab_per_lang=dict_size); m_sparse._scoring_choice = "sparse"; m_sparse._vectorized_batch_size = 25000
        ts = time_fn(m_sparse._score, df.copy(), already_clean=False)
    except Exception:
        pass
    extra = (
        (f" | numba: {tn:.3f}s ({n/tn:.1f} rows/s)" if tn != float('inf') else "") +
        (f" | sparse: {ts:.3f}s ({n/ts:.1f} rows/s)" if ts != float('inf') else "")
    )
    print(
        f"n={n} dict={dict_size} len={sentence_len} | apply: {ta:.3f}s ({n/ta:.1f} rows/s) | vectorized-batched: {tv:.3f}s ({n/tv:.1f} rows/s){extra}"
    )


def bench_parallel_clean_token(n_rows=50000, worker_options=(1, 4, 8)):
    print("\n== Benchmark: cleaning/tokenization parallelism ==")
    df = synth_dataset(n_rows=n_rows)
    model = Vocabulous()
    series = df["text"]

    def run_clean(w):
        model._clean_series(series, workers=w)

    def run_token(w):
        model._process_sentences(df, workers=w)

    for workers in worker_options:
        tc = time_fn(run_clean, workers)
        tt = time_fn(run_token, workers)
        print(
            f"workers={workers:2d} | clean: {tc:.3f}s ({n_rows/tc:.1f} rows/s) | token: {tt:.3f}s ({n_rows/tt:.1f} rows/s)"
        )


def bench_train_workers(n_rows=20000, worker_pairs=((1, 1), (2, 2), (4, 4))):
    print("\n== Benchmark: train() with worker settings ==")
    df = synth_dataset(n_rows=n_rows)
    for clean_workers, token_workers in worker_pairs:
        model = Vocabulous()
        t = time_fn(
            model.train,
            df.copy(),
            df.copy(),
            1,
            0.2,
            0.2,
            "text",
            "lang",
            clean_workers,
            token_workers,
        )
        print(
            f"clean={clean_workers:2d} token={token_workers:2d} | train: {t:.3f}s ({n_rows/t:.1f} sentences/s)"
        )


def bench_sentence_expansion(
    n_rows=200000, worker_options=(1, 2, 4, 8), chunk_size=100_000
):
    print("\n== Benchmark: sentence expansion parallelism ==")
    df = synth_dataset(n_rows=n_rows)
    model = Vocabulous()
    model._sentence_chunk_size = chunk_size

    def run(w):
        model._expand_to_sentence_level(df, workers=w)

    for workers in worker_options:
        t = time_fn(run, workers)
        print(f"workers={workers:2d} | expand: {t:.3f}s ({n_rows/t:.1f} rows/s)")


def main():
    # Single-size baseline
    df = synth_dataset(n_rows=20000)
    model = Vocabulous()

    # Benchmark cleaning + scoring (auto mode)
    t_score = time_fn(model._score, df.copy(), already_clean=False)
    print(f"Auto Score (clean+score) on 20k rows: {t_score:.3f}s  -> {20000/t_score:.1f} rows/s")

    # Benchmark cleaning only
    t_clean = time_fn(lambda d: d["text"].apply(model._clean_text), df.copy())
    print(f"Clean only on 20k rows: {t_clean:.3f}s  -> {20000/t_clean:.1f} rows/s")

    # Benchmark cycle clean end-to-end
    t_cycle_clean = time_fn(model._cycle_clean, model._score(df.copy()), 0.1, 0.1)
    print(f"Cycle clean on 20k rows (pre-scored): {t_cycle_clean:.3f}s")

    # Modes and sizes comparison
    bench_modes()
    bench_duplication()
    bench_sentence_length()
    bench_dict_size()
    bench_large_n(n=100000)
    bench_high_match(n_rows=20000, sentence_len=20, vocab_sizes=(50,500,5000))
    bench_large_n_compare(n=200000, sentence_len=20, dict_size=1000)
    bench_parallel_clean_token()
    bench_train_workers()
    bench_sentence_expansion()


if __name__ == "__main__":
    main()
