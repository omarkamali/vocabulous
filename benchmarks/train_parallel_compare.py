import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import string

from vocabulous import Vocabulous


@dataclass
class TrainResult:
    model: Vocabulous
    report: dict
    duration: float


def synth_dataset(
    n_rows: int,
    *,
    words_per_sentence: int = 20,
    vocab_size: int = 10_000,
    typo_probability: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multilingual dataset with random alphanumeric words.

    Each sentence is composed of random tokens (3-10 chars) to avoid heavy
    deduplication and exercise cleaning/tokenization paths.
    """

    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits
    langs = ["en", "es", "fr"]

    def random_word() -> str:
        length = rng.randint(3, 10)
        return "".join(rng.choice(alphabet) for _ in range(length))

    vocabulary = [random_word() for _ in range(vocab_size)]

    def maybe_typo(word: str) -> str:
        if len(word) <= 3 or rng.random() > typo_probability:
            return word
        drop_index = rng.randrange(len(word))
        return word[:drop_index] + word[drop_index + 1 :]

    def make_sentence() -> str:
        return " ".join(
            maybe_typo(rng.choice(vocabulary)) for _ in range(words_per_sentence)
        )

    rows = []
    for _ in range(n_rows):
        lang = rng.choice(langs)
        rows.append({"lang": lang, "text": make_sentence()})

    return pd.DataFrame(rows)


def train_once(
    df: pd.DataFrame,
    *,
    clean_workers: Optional[int] = None,
    token_workers: Optional[int] = None,
    cycles: int = 1,
    base_confidence: float = 0.2,
    confidence_margin: float = 0.2,
) -> TrainResult:
    model = Vocabulous()
    start = time.perf_counter()
    trained_model, report = model.train(
        df.copy(),
        df.copy(),
        cycles=cycles,
        base_confidence=base_confidence,
        confidence_margin=confidence_margin,
        clean_workers=clean_workers,
        token_workers=token_workers,
    )
    duration = time.perf_counter() - start
    return TrainResult(trained_model, report, duration)


def compare_models(seq: Vocabulous, par: Vocabulous) -> Tuple[bool, str]:
    if seq.languages != par.languages:
        return False, "Language sets differ"
    if seq.word_lang_freq != par.word_lang_freq:
        return False, "word_lang_freq differs"
    return True, "Models have identical languages and dictionaries"


def validate_scores(seq: Vocabulous, par: Vocabulous, sample_df: pd.DataFrame) -> Tuple[bool, str]:
    seq_scores = seq._score(sample_df.copy())
    par_scores = par._score(sample_df.copy())
    seq_pred = seq_scores["predicted_lang"].tolist()
    par_pred = par_scores["predicted_lang"].tolist()
    if seq_pred != par_pred:
        return False, "Predicted languages differ on evaluation sample"
    return True, "Predicted languages match for evaluation sample"


def main():
    parser = argparse.ArgumentParser(description="Compare sequential vs parallel training.")
    parser.add_argument("--rows", type=int, default=10000, help="Number of training rows")
    parser.add_argument("--clean-workers", type=int, default=4, help="Parallel clean workers")
    parser.add_argument("--token-workers", type=int, default=4, help="Parallel token workers")
    args = parser.parse_args()

    df = synth_dataset(args.rows)
    eval_sample = df.sample(n=min(2000, len(df)), random_state=123)

    seq = train_once(df, clean_workers=1, token_workers=1)
    par = train_once(
        df,
        clean_workers=args.clean_workers,
        token_workers=args.token_workers,
    )

    equal_dicts, dict_msg = compare_models(seq.model, par.model)
    equal_scores, score_msg = validate_scores(seq.model, par.model, eval_sample)

    print("=== Training Benchmark ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Sequential: {seq.duration:.3f}s")
    print(f"Parallel  : {par.duration:.3f}s (clean={args.clean_workers}, token={args.token_workers})")
    speedup = seq.duration / par.duration if par.duration > 0 else float("inf")
    print(f"Speedup   : {speedup:.2f}x")
    print("=== Correctness Checks ===")
    print(f"Dictionary equality: {equal_dicts} ({dict_msg})")
    print(f"Prediction equality: {equal_scores} ({score_msg})")

    if not (equal_dicts and equal_scores):
        raise SystemExit("Parallel training diverged from sequential baseline")


if __name__ == "__main__":
    main()
