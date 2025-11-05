import swifter
import nltk
from tqdm import tqdm
import pandas as pd
import logging
import json
import re
import logging
import numpy as np
import pandas as pd
import hashlib
from functools import lru_cache
from unscript import (
    get_dominant_script,
    unscript,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logging.info("Downloading nltk punkt tokenizer...")
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logging.info("Downloading nltk punkt_tab tokenizer...")
    nltk.download("punkt_tab")


@lru_cache(maxsize=200_000)
def _clean_text_cached(default_script, text):
    """Module-level LRU cached cleaner. Keyed by (default_script, text).

    Mirrors the original `_clean_text` behavior, including script detection,
    unscript options, Arabic lowercase handling, and numbers-only filtering.
    """
    if not text or not str(text).strip():
        logging.debug(f"_clean_text_cached: empty/whitespace input: '{text}'")
        return ""

    # Determine script
    if default_script:
        script_to_use = default_script
        logging.debug(f"_clean_text_cached: Using default script: {script_to_use}")
    else:
        logging.debug("_clean_text_cached: Detecting dominant script...")
        script_to_use = get_dominant_script(text, min_percentage=20.0)
        if script_to_use is None:
            script_to_use = "Latn"
            logging.debug(
                "_clean_text_cached: No dominant script detected, defaulting to Latin."
            )
        else:
            logging.debug(
                f"_clean_text_cached: Detected dominant script: {script_to_use}"
            )

    # Apply unscript
    logging.debug(
        f"_clean_text_cached: Applying unscript with script '{script_to_use}'"
    )
    cleaned = unscript(
        script_to_use,
        text,
        {"spaces": True, "numbers": True, "punctuation": False, "symbols": False},
        lowercase=(script_to_use != "Arab"),
    )
    logging.debug(f"_clean_text_cached: Cleaned preview: '{cleaned[:50]}...'")

    # Numbers-only becomes empty
    if cleaned and re.match(r"^\d+$", cleaned.strip()):
        logging.debug(
            "_clean_text_cached: Cleaned text is only numbers -> empty string"
        )
        return ""

    logging.debug(f"_clean_text_cached: Final cleaned length: {len(cleaned)}")
    return cleaned


class Vocabulous:
    def __init__(self, store_training_data=False, default_script=None):
        """Initialize Vocabulous model for dictionary building and language detection.

        Args:
            store_training_data (bool): Whether to store training data internally
            default_script (str, optional): Default script to use for text cleaning
                instead of auto-detection (e.g., 'Latn', 'Arab', 'Hans')
        """
        self.word_lang_freq = {}  # {word: {lang: frequency}}
        self.store_training_data = store_training_data
        self.training_data = [] if store_training_data else None
        self.languages = set()
        self.default_script = default_script
        # Cached DF representation of dictionary for vectorized scoring
        self._dict_df = None  # pd.DataFrame with columns: word, lang
        self._dict_dirty = True
        # Scoring mode: default to 'apply'; 'auto' will calibrate among modes
        self._scoring_mode = "apply"
        self._scoring_choice = None
        # Default batch size for vectorized scoring
        self._vectorized_batch_size = 50000
        # Numba-related cached structures
        self._numba_available = self._check_numba()
        self._numba_cached = False
        self._w2id = None  # dict word->int id
        self._id2lang = None  # list of lang names by id
        self._lang2id = None  # dict lang->id
        self._tok_ptr = None  # np.ndarray ptr
        self._tok_idx = None  # np.ndarray idx
        # Sparse-sharded scorer structures
        self._sparse_shards = None  # list of shards with CSR-like ptr/idx and word maps
        self._sparse_shard_count = 256
        self._sparse_workers = 0  # 0 means no parallelism; >0 uses ThreadPool
        self._sparse_store = None  # path to store/load shards
        self._sparse_backend = "memory"  # or "memmap"
        # Aggregated stats (computed on demand)
        self._lang_token_totals = None  # {lang: total_tokens}
        # Aggregated stats (computed on demand)
        self._lang_token_totals = None  # {lang: total_tokens}

    def _split_into_sentences(self, text):
        """Split text into sentences using NLTK sentence tokenizer.

        Args:
            text (str): Input text to split

        Returns:
            List[str]: List of sentences
        """
        if not text or not text.strip():
            return []

        try:
            sentences = nltk.sent_tokenize(text)
            # Do not filter by word count; keep short sentences for downstream logic/tests
            return sentences
        except Exception as e:
            logging.warning(f"Error tokenizing sentences: {e}")
            return [text]  # Fallback to original text

    def _expand_to_sentence_level(self, df, text_column="text", lang_column="lang"):
        """Expand dataframe from sample level to sentence level.

        Args:
            df (pd.DataFrame): Input dataframe with samples
            text_column (str): Name of text column
            lang_column (str): Name of language column

        Returns:
            pd.DataFrame: Expanded dataframe with one row per sentence
        """
        logging.info("Expanding data from sample level to sentence level...")

        # Vectorized expansion using pandas explode to avoid Python-level row loops
        tmp_col = "__sentences__"
        df = df.copy()
        df[tmp_col] = df[text_column].apply(self._split_into_sentences)
        expanded_df = df.explode(tmp_col, ignore_index=True)
        # Move exploded sentences back into the original text column
        expanded_df[text_column] = expanded_df[tmp_col]
        expanded_df = expanded_df.drop(columns=[tmp_col])
        # Drop any rows where sentence splitting yielded None/empty
        expanded_df = expanded_df[expanded_df[text_column].notna()]
        expanded_df = expanded_df[expanded_df[text_column] != ""]
        logging.info(f"Expanded from {len(df)} samples to {len(expanded_df)} sentences")

        return expanded_df

    def train(
        self,
        train_df,
        eval_df=None,
        cycles=2,
        base_confidence=0.5,
        confidence_margin=0.5,
        text_column="text",
        lang_column="lang",
    ):
        """Build language dictionaries from training data.

        Args:
            train_df (List[Dict]): Training data with text and language labels
            eval_df (List[Dict]): Evaluation data
            cycles (int): Number of dictionary refinement cycles
            base_confidence (float): Minimum confidence threshold for word-language associations
            confidence_margin (float): Minimum margin between top two language scores (0-1)

        Returns:
            Tuple[Vocabulous, Dict]: Updated model and training report
        """
        logging.info("Starting Vocabulous training process...")

        # Normalize inputs: support dict form {lang: [sentences]} or DataFrame with columns
        if eval_df is None:
            eval_df = train_df

        def _to_rows(obj, text_col, lang_col):
            # Accept dict {lang: [texts]} or list[dict] or DataFrame
            if isinstance(obj, dict):
                rows = []
                for l, texts in obj.items():
                    for t in texts:
                        rows.append({text_col: t, lang_col: l})
                return pd.DataFrame(rows)
            elif isinstance(obj, pd.DataFrame):
                return obj
            else:
                # Try to construct DataFrame directly; caller may pass list[dict]
                return pd.DataFrame(obj)

        logging.info("Converting training and evaluation data to normalized DataFrames...")
        train_df = _to_rows(train_df, text_column, lang_column)
        eval_df = _to_rows(eval_df, text_column, lang_column)

        # Handle empty data gracefully
        if len(train_df) == 0 or len(eval_df) == 0:
            logging.warning("Training or evaluation data is empty. Skipping training.")
            return self, {
                "cycles": 0,
                "cycle_reports": [],
                "dictionary_size": 0,
                "train_data": train_df,
            }

        # Check for required columns
        required_columns = [text_column, lang_column]
        if not all(col in train_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in train_df.columns]
            raise ValueError(
                f"Training DataFrame is missing required columns: {missing}"
            )
        if not all(col in eval_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in eval_df.columns]
            raise ValueError(
                f"Evaluation DataFrame is missing required columns: {missing}"
            )

        # Expand to sentence level
        logging.info("Expanding training data to sentence level...")
        train_df = self._expand_to_sentence_level(train_df, text_column, lang_column)
        logging.info("Expanding evaluation data to sentence level...")
        eval_df = self._expand_to_sentence_level(eval_df, text_column, lang_column)

        # Clean text before training
        logging.info("Cleaning text in training data...")
        train_df[text_column] = train_df[text_column].swifter.apply(
            lambda x: self._clean_text(x)
        )
        logging.info("Cleaning text in evaluation data...")
        eval_df[text_column] = eval_df[text_column].swifter.apply(
            lambda x: self._clean_text(x)
        )

        # Remove empty strings after cleaning
        logging.info("Removing empty strings from data after cleaning...")
        train_df = train_df[train_df[text_column] != ""]
        eval_df = eval_df[eval_df[text_column] != ""]
        logging.info(f"Training data size after cleaning: {len(train_df)} sentences")
        logging.info(f"Evaluation data size after cleaning: {len(eval_df)} sentences")

        cycle_reports = []
        prev_samples = len(train_df)

        for cycle in range(cycles):
            logging.info(f"Starting training cycle {cycle+1}/{cycles}")
            self.word_lang_freq = {}

            # Only clean data until n-1 cycle
            if cycle < cycles - 1:
                train_df, report = self._train_cycle(
                    train_df, eval_df, base_confidence, confidence_margin
                )
            else:
                # Skip cleaning on final cycle
                train_df, report = self._train_cycle(
                    train_df,
                    eval_df,
                    base_confidence,
                    confidence_margin,
                    skip_cleaning=True,
                )

            cycle_reports.append(report)

            # Check if we should stop early (only for non-final cycles)
            if cycle < cycles - 1 and len(train_df) == prev_samples:
                logging.info("No sentences removed in this cycle - stopping early")
                break

            prev_samples = len(train_df)

            # Log metrics after each cycle
            logging.info("=" * 100)
            logging.info(f"Cycle {cycle+1} metrics:")
            logging.info(f"  F1 Score: {report['f1']:.4f}")
            logging.info(f"  Accuracy: {report['accuracy']:.4f}")
            logging.info(f"  Precision: {report['precision']:.4f}")
            logging.info(f"  Recall: {report['recall']:.4f}")
            logging.info(f"  Confusion Score: {report['confusion']:.4f}")
            logging.info(f"  Confidence Margin: {report['confidence_margin']:.4f}")
            logging.info(
                f"  Sentences removed: {report['removed_samples']}/{report['total_samples']}"
            )
            logging.info(f"  Dictionary size: {len(self.word_lang_freq)}")

        return self, {
            "cycles": len(cycle_reports),
            "cycle_reports": cycle_reports,
            "dictionary_size": len(self.word_lang_freq),
            "train_data": train_df,
        }

    def _train_cycle(
        self,
        train_df,
        eval_df,
        base_confidence=0.5,
        confidence_margin=0.5,
        skip_cleaning=False,
    ):
        # first let's deduplicate the training data at sentence level
        logging.info(f"Deduplicating training sentences for current cycle...")
        train_df = self._deduplicate(train_df)
        original_samples = len(train_df)
        logging.info(
            f"Starting cycle with {original_samples} sentences after deduplication"
        )

        # Reset word_lang_freq for this cycle
        self.word_lang_freq = {}
        self.languages = set()

        # Process sentences using swifter but in a way that avoids race conditions
        def process_sentence(row):
            lang = row["lang"]
            # Use fast Unicode-aware regex tokenization and unique words per sentence
            tokens = re.findall(r"\w+", row["text"], flags=re.UNICODE)
            words = set(tokens)
            return {"lang": lang, "words": words}

        # First collect all words and languages
        logging.info("Processing sentences...")
        processed = train_df.swifter.apply(process_sentence, axis=1)
        # Ensure we have a DataFrame with ['lang', 'words'] columns
        if isinstance(processed, pd.Series):
            processed = pd.DataFrame(processed.tolist())

        # Then update the dictionaries safely
        logging.info("Building language dictionaries...")
        # processed is a DataFrame with columns ['lang', 'words']
        for _, result in tqdm(processed.iterrows(), total=len(processed)):
            lang = result["lang"]
            words = result["words"]
            self.languages.add(lang)

            for word in words:
                if word not in self.word_lang_freq:
                    self.word_lang_freq[word] = {}
                if lang not in self.word_lang_freq[word]:
                    self.word_lang_freq[word][lang] = 0
                self.word_lang_freq[word][lang] += 1

        # now we have a dictionary of unique words for each language, let's produce a report
        logging.info("Generating cycle report...")
        report = self._report_cycle(eval_df)

        # now let's use it to filter the training data where the ground truth lang conflicts with the language classification using the dictionaries
        logging.info("Scoring training sentences...")
        # Training data was cleaned earlier in train(), so skip re-cleaning
        train_df = self._score(train_df, already_clean=True)

        # Skip cleaning if this is the final cycle
        if not skip_cleaning:
            # now we have a dataframe with scores for each sentence, and we need to clean it by removing:
            # - sentences where the top matching language is different from the ground truth
            # - sentences where the scores are too close to each other (low confidence margin)
            # - sentences where the scores are too low (below a base confidence threshold)
            logging.info("Cleaning training sentences...")
            cleaned_train_df = self._cycle_clean(
                train_df, base_confidence, confidence_margin
            )
            report["removed_samples"] = original_samples - len(cleaned_train_df)
            report["total_samples"] = original_samples
            return cleaned_train_df, report
        else:
            report["removed_samples"] = 0
            report["total_samples"] = original_samples
            return train_df, report

    def _clean_text(self, text):
        """Clean text using Unscript with LRU caching keyed by (default_script, text)."""
        return _clean_text_cached(self.default_script, text)

    def _report_cycle(self, eval_df):
        """Produce a report for the cycle"""
        # Score the evaluation data
        scored_df = self._score(eval_df)
        # Use precomputed predicted if present (from _score), otherwise derive it
        if "predicted_lang" not in scored_df.columns or scored_df["predicted_lang"].isna().any():
            def get_max_lang(scores):
                if not scores:
                    return None
                return max(scores.items(), key=lambda item: item[1])[0]
            scored_df["predicted_lang"] = scored_df["scores"].apply(get_max_lang)

        # Calculate confusion matrix
        confusion_matrix = []
        langs = list(self.languages)
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                confusion_score = self._calculate_confusion(scored_df, lang1, lang2)
                if confusion_score > 0:  # Only include non-zero confusion scores
                    confusion_matrix.append(
                        {"langs": [lang1, lang2], "score": confusion_score}
                    )

        # Calculate overall confusion score (average of all pairwise confusions)
        confusion = (
            sum(item["score"] for item in confusion_matrix) / len(confusion_matrix)
            if confusion_matrix
            else 0
        )

        # Vectorized classification metrics via crosstab
        df_valid = scored_df[scored_df["predicted_lang"].notna()].copy()
        if not df_valid.empty:
            cm = pd.crosstab(df_valid["lang"], df_valid["predicted_lang"], dropna=False)
            # Ensure all languages appear as both index and columns
            langs = sorted(list(self.languages))
            cm = cm.reindex(index=langs, columns=langs, fill_value=0)

            # Overall accuracy
            correct = cm.values.diagonal().sum() if hasattr(cm.values, 'diagonal') else cm.to_numpy().diagonal().sum()
            total = cm.values.sum()
            accuracy = (correct / total) if total > 0 else 0.0

            # Per-language precision/recall
            tp = cm.values.diagonal()
            pred_totals = cm.sum(axis=0).values  # column sums
            true_totals = cm.sum(axis=1).values  # row sums

            with pd.option_context("mode.use_inf_as_na", True):
                precision_per_lang = [ (tp_i / p_i) if p_i > 0 else 0.0 for tp_i, p_i in zip(tp, pred_totals) ]
                recall_per_lang = [ (tp_i / t_i) if t_i > 0 else 0.0 for tp_i, t_i in zip(tp, true_totals) ]
                f1_per_lang = []
                for pr, rc in zip(precision_per_lang, recall_per_lang):
                    f1_per_lang.append((2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0)

            metrics = {
                "precision": sum(precision_per_lang) / len(langs) if langs else 0.0,
                "recall": sum(recall_per_lang) / len(langs) if langs else 0.0,
                "f1": sum(f1_per_lang) / len(langs) if langs else 0.0,
            }
        else:
            accuracy = 0.0
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        def _get_top_score_diff(scores):
            if not scores:  # Handle empty scores
                return 1
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) < 2:
                return 1  # If only one language, it's infinitely spiky
            return sorted_scores[0] - sorted_scores[1]

        # Vectorized confidence margin if precomputed top/second exist
        if {"top_lang", "top_score", "second_score"}.issubset(scored_df.columns):
            diffs = (scored_df["top_score"] - scored_df["second_score"].fillna(0.0))
            diffs = diffs.where(scored_df["top_lang"].notna(), 1.0)
            confidence_margin = float(diffs.mean()) if not diffs.empty else 0.0
        else:
            confidence_margin = scored_df["scores"].apply(_get_top_score_diff).mean()

        return {
            "f1": metrics["f1"],
            "accuracy": accuracy,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "confusion": confusion,
            "confidence_margin": confidence_margin,  # average difference between the top two scores
            "confusion_matrix": confusion_matrix,
            "total_samples": 0,  # Will be filled in _train_cycle
            "removed_samples": 0,  # Will be filled in _train_cycle
        }

    def _calculate_confusion(self, df, lang1, lang2):
        """Calculate confusion score between two languages based on predictions.

        A high confusion score means the model often confuses these languages with each other.
        Score is calculated as the proportion of misclassifications between these two languages.
        """
        # Filter relevant rows
        relevant = df["lang"].isin([lang1, lang2])
        if not relevant.any():
            return 0.0

        rel = df.loc[relevant]
        mask_confuse = ((rel["lang"] == lang1) & (rel["predicted_lang"] == lang2)) | (
            (rel["lang"] == lang2) & (rel["predicted_lang"] == lang1)
        )
        confusions = mask_confuse.sum()
        return float(confusions) / float(len(rel))

    def _cycle_clean(self, df, base_confidence, confidence_margin):
        """Clean the dataframe by removing sentences where the top matching language is different from the ground truth,
        and sentences where the scores are too close to each other (low confidence margin)
        """
        # Fast path: if 'top_lang' and 'top_score' exist, use them
        if "top_lang" in df.columns and "top_score" in df.columns:
            mask_lang = (df["top_lang"] == df["lang"]) & (df["top_score"] >= base_confidence)
            filtered = df[mask_lang]
            # Margin: prefer precomputed 'second_score' if present
            if "second_score" in filtered.columns:
                margin = filtered["top_score"].fillna(0) - filtered["second_score"].fillna(0)
                filtered = filtered[margin > confidence_margin]
            else:
                # Fallback to computing margin per row only for filtered subset
                def _get_top_score_diff(scores):
                    if not scores:
                        return 0
                    sorted_scores = sorted(scores.values(), reverse=True)
                    if len(sorted_scores) < 2:
                        return float("inf")
                    return sorted_scores[0] - sorted_scores[1]

                filtered = filtered[filtered["scores"].apply(_get_top_score_diff) > confidence_margin]
            return filtered

        # Fallback path: original per-row logic
        def check_top_lang(row):
            scores = row["scores"]
            if not scores:
                return False
            max_score = max(scores.values())
            return (
                max_score == scores.get(row["lang"], 0) and max_score >= base_confidence
            )

        df = df[df.apply(check_top_lang, axis=1)]

        def _get_top_score_diff(scores):
            if not scores:
                return 0
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) < 2:
                return float("inf")
            return sorted_scores[0] - sorted_scores[1]

        df = df[df["scores"].apply(_get_top_score_diff) > confidence_margin]
        return df

    def _deduplicate(self, df):
        """Deduplicate the training data by removing duplicate sentences."""
        logging.info("Deduplicating training data...")
        deduplicated_df = df.drop_duplicates(subset="text")
        logging.info(
            f"Deduplication complete. Original: {len(df)} samples, Deduplicated: {len(deduplicated_df)} samples"
        )
        return deduplicated_df

    def _score(self, df, already_clean=False):
        """Score the provided dataframe by matching words in sentences with language dictionaries
        and then calculate the confidence scores. Adds precomputed top-2 columns.
        """
        if not already_clean:
            logging.info("Cleaning text before scoring...")
            df = df.copy()
            df["text"] = df["text"].swifter.apply(lambda x: self._clean_text(x))
            df = df[df["text"] != ""]

        df = df.copy()
        # Apply-only scoring path
        logging.info("Scoring sentences (apply swifter)...")
        df["scores"] = df["text"].swifter.apply(self._score_sentence)

        # Precompute top-1 and top-2 info to speed up downstream steps
        def _top2(scores):
            if not scores:
                return None, 0.0, 0.0
            items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(items) == 1:
                return items[0][0], items[0][1], 0.0
            return items[0][0], items[0][1], items[1][1]

        tops = df["scores"].apply(_top2)
        if len(tops) == 0:
            # Ensure expected columns exist for downstream consumers
            df["top_lang"] = pd.Series(dtype="object")
            df["top_score"] = pd.Series(dtype="float")
            df["second_score"] = pd.Series(dtype="float")
            df["predicted_lang"] = pd.Series(dtype="object")
            return df
        df["top_lang"], df["top_score"], df["second_score"] = zip(*tops)
        df["predicted_lang"] = df["top_lang"]

        return df

    def _ensure_dict_df(self):
        """Build or reuse a cached DataFrame mapping words to languages."""
        if (self._dict_df is not None) and (not self._dict_dirty):
            return self._dict_df
        # Build from self.word_lang_freq
        rows = []
        for w, lang_freqs in self.word_lang_freq.items():
            for lang in lang_freqs.keys():
                if lang in self.languages:
                    rows.append((w, lang))
        if rows:
            self._dict_df = pd.DataFrame(rows, columns=["word", "lang"])
        else:
            self._dict_df = pd.DataFrame(columns=["word", "lang"])
        # Optimize dtypes to category to speed up join/groupby and reduce memory
        if not self._dict_df.empty:
            self._dict_df["word"] = self._dict_df["word"].astype("category")
            # Restrict langs to known set ordering for stable behavior
            lang_cats = pd.CategoricalDtype(categories=sorted(self.languages))
            self._dict_df["lang"] = self._dict_df["lang"].astype(lang_cats)
        else:
            # Ensure columns have category dtype even when empty
            self._dict_df = self._dict_df.astype({
                "word": "category",
                "lang": pd.CategoricalDtype(categories=sorted(self.languages)),
            })
        self._dict_dirty = False
        return self._dict_df

    def _score_vectorized(self, texts: pd.Series) -> pd.Series:
        """Vectorized scoring for a Series of sentences.
        Returns a Series of dicts: {lang: score} per row.
        """
        if texts.empty:
            return pd.Series([], dtype="object")

        # Tokenize all texts -> explode tokens with row ids
        # Use Unicode-aware \w+ regex to match word tokens
        tokens_list = texts.str.findall(r"\w+", flags=re.UNICODE)
        # Create a DataFrame with row_id to preserve mapping
        row_ids = texts.index
        token_df = (
            pd.DataFrame({"row_id": row_ids, "tokens": tokens_list})
            .explode("tokens")
            .dropna()
        )
        if token_df.empty:
            return pd.Series([{ } for _ in range(len(texts))], index=texts.index, dtype="object")

        # Total tokens per row for normalization
        totals = token_df.groupby("row_id").size().rename("total")

        # Join with dictionary (word->lang) to count known words per lang per row
        dict_df = self._ensure_dict_df()
        if dict_df.empty:
            # No known words; all scores empty
            return pd.Series([{ } for _ in range(len(texts))], index=texts.index, dtype="object")

        merged = token_df.merge(dict_df, left_on="tokens", right_on="word", how="left")
        merged = merged.dropna(subset=["lang"])  # keep only known words
        if merged.empty:
            return pd.Series([{ } for _ in range(len(texts))], index=texts.index, dtype="object")

        counts = (
            merged.groupby(["row_id", "lang"], observed=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        counts = counts.join(totals, on="row_id")
        counts["score"] = counts["cnt"] / counts["total"].astype(float)

        # Pivot to dense matrix (rows x langs), then convert per-row dicts
        piv = counts.pivot_table(
            index="row_id",
            columns="lang",
            values="score",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        )
        # Ensure all expected row_ids present
        piv = piv.reindex(texts.index, fill_value=0.0)
        # Convert each row to a compact dict of non-zero scores
        records = piv.to_dict(orient="index")
        scores_series = pd.Series({rid: {k: v for k, v in rec.items() if v > 0.0} for rid, rec in records.items()})
        return scores_series

    def _check_numba(self) -> bool:
        try:
            import numba  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_numba_structs(self):
        """Prepare CSR-like mapping from token->lang ids for Numba path."""
        if not self._dict_dirty and self._numba_cached:
            return
        # Build language ids
        langs = sorted(list(self.languages))
        self._lang2id = {l: i for i, l in enumerate(langs)}
        self._id2lang = langs
        # Build word ids and token->langs list
        words = list(self.word_lang_freq.keys())
        self._w2id = {w: i for i, w in enumerate(words)}
        ptr = [0]
        idx = []
        for w in words:
            langs_map = self.word_lang_freq.get(w, {})
            for l in langs_map.keys():
                if l in self._lang2id:
                    idx.append(self._lang2id[l])
            ptr.append(len(idx))
        self._tok_ptr = np.asarray(ptr, dtype=np.int32)
        self._tok_idx = np.asarray(idx, dtype=np.int32)
        self._numba_cached = True
        self._dict_dirty = False

    def _score_numba(self, texts: pd.Series) -> pd.Series:
        """Numba-accelerated scoring using integer-encoded vocab and CSR mapping."""
        if not self._numba_available:
            # Fallback
            return self._score_apply(texts)
        self._ensure_numba_structs()
        # Tokenize and encode to ids
        row_ids = []
        tok_ids = []
        totals = {}
        for rid, t in texts.items():
            toks = _tokenize_cached(t)
            count = 0
            for tok in toks:
                wid = self._w2id.get(tok)
                if wid is not None:
                    row_ids.append(rid)
                    tok_ids.append(wid)
                count += 1
            totals[rid] = max(1, count)  # avoid div-by-zero
        if not tok_ids:
            # No matches at all -> empty dicts
            return pd.Series([{ } for _ in range(len(texts))], index=texts.index, dtype="object")

        row_ids_np = np.asarray(row_ids, dtype=np.int32)
        tok_ids_np = np.asarray(tok_ids, dtype=np.int32)
        # Map row ids to compact [0..R) indices
        unique_rows = {rid: i for i, rid in enumerate(texts.index.tolist())}
        row_idx_np = np.asarray([unique_rows[r] for r in row_ids], dtype=np.int32)
        n_rows = len(texts)
        n_langs = len(self._id2lang)
        counts = np.zeros((n_rows, n_langs), dtype=np.int32)

        # Numba kernel
        from numba import njit

        @njit(cache=True)
        def accumulate(row_idx, tok_ids, ptr, idx, out):
            for i in range(tok_ids.shape[0]):
                r = row_idx[i]
                w = tok_ids[i]
                start = ptr[w]
                end = ptr[w + 1]
                for k in range(start, end):
                    l = idx[k]
                    out[r, l] += 1

        accumulate(row_idx_np, tok_ids_np, self._tok_ptr, self._tok_idx, counts)

        # Normalize and convert to per-row dicts
        result = {}
        for rid in texts.index:
            rpos = unique_rows[rid]
            total = float(totals.get(rid, 1))
            row_counts = counts[rpos]
            rec = {}
            for lid, c in enumerate(row_counts):
                if c > 0:
                    rec[self._id2lang[lid]] = c / total
            result[rid] = rec
        return pd.Series(result)

    # Public configuration setters
    def set_scoring_mode(self, mode: str):
        """Set scoring mode ('apply', 'vectorized', 'numba', 'sparse', or 'auto')."""
        self._scoring_mode = mode
        self._scoring_choice = None if mode == "auto" else mode

    def set_vectorized_batch_size(self, size: int):
        self._vectorized_batch_size = int(size)

    def set_sparse_shard_count(self, count: int):
        self._sparse_shard_count = int(count)
        self._sparse_shards = None  # force rebuild

    def set_sparse_workers(self, workers: int):
        self._sparse_workers = int(workers)

    def set_sparse_store(self, path: str, backend: str = "memmap"):
        """Configure on-disk shard storage. backend: 'memmap' or 'memory'."""
        self._sparse_store = path
        self._sparse_backend = backend
        self._sparse_shards = None  # force reload/build

    def _ensure_sparse_shards(self):
        """Partition vocabulary into shards and build CSR-like token->lang maps per shard."""
        if self._sparse_shards is not None and not self._dict_dirty:
            return
        langs = sorted(list(self.languages))
        lang2id = {l: i for i, l in enumerate(langs)}
        shard_count = max(1, int(self._sparse_shard_count))
        shards = [None] * shard_count
        store_base = self._sparse_store
        use_memmap = (self._sparse_backend == "memmap" and store_base)
        import os, json
        if use_memmap:
            os.makedirs(store_base, exist_ok=True)
        # Try to load existing memmap shards if available
        loaded = False
        if use_memmap:
            try:
                for si in range(shard_count):
                    sdir = os.path.join(store_base, f"shard_{si}")
                    if not os.path.isdir(sdir):
                        loaded = False
                        break
                    with open(os.path.join(sdir, "w2id.json"), "r", encoding="utf-8") as f:
                        w2id = json.load(f)
                    ptr = np.load(os.path.join(sdir, "ptr.npy"), mmap_mode="r")
                    idx = np.load(os.path.join(sdir, "idx.npy"), mmap_mode="r")
                    shards[si] = {"w2id": w2id, "ptr": ptr, "idx": idx}
                else:
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            # Build from in-memory dictionary
            tmp = [
                {"w2id": {}, "ptr": [0], "idx": []}
                for _ in range(shard_count)
            ]
            def _stable_shard(s: str) -> int:
                h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
                return int.from_bytes(h, 'little') % shard_count
            for w, m in self.word_lang_freq.items():
                s = _stable_shard(w)
                shard = tmp[s]
                shard["w2id"][w] = len(shard["w2id"])
                for l in m.keys():
                    if l in lang2id:
                        shard["idx"].append(lang2id[l])
                shard["ptr"].append(len(shard["idx"]))
            # finalize arrays and optionally persist
            for si, sh in enumerate(tmp):
                ptr_arr = np.asarray(sh["ptr"], dtype=np.int32)
                idx_arr = np.asarray(sh["idx"], dtype=np.int32)
                if use_memmap:
                    sdir = os.path.join(store_base, f"shard_{si}")
                    os.makedirs(sdir, exist_ok=True)
                    # save
                    np.save(os.path.join(sdir, "ptr.npy"), ptr_arr)
                    np.save(os.path.join(sdir, "idx.npy"), idx_arr)
                    with open(os.path.join(sdir, "w2id.json"), "w", encoding="utf-8") as f:
                        json.dump(sh["w2id"], f)
                    # reopen as memmap read-only
                    ptr = np.load(os.path.join(sdir, "ptr.npy"), mmap_mode="r")
                    idx = np.load(os.path.join(sdir, "idx.npy"), mmap_mode="r")
                    shards[si] = {"w2id": sh["w2id"], "ptr": ptr, "idx": idx}
                else:
                    shards[si] = {"w2id": sh["w2id"], "ptr": ptr_arr, "idx": idx_arr}
        self._sparse_shards = {"langs": langs, "lang2id": lang2id, "shards": shards}
        self._dict_dirty = False

    def _score_sparse(self, texts: pd.Series) -> pd.Series:
        """Sparse sharded scorer using NumPy vectorized accumulation.
        Returns Series[row_id -> {lang: score}].
        """
        self._ensure_sparse_shards()
        langs = self._sparse_shards["langs"]
        lang2id = self._sparse_shards["lang2id"]
        shards = self._sparse_shards["shards"]

        idx_list = texts.index.tolist()
        n_rows = len(idx_list)
        n_langs = len(langs)
        rowpos = {rid: i for i, rid in enumerate(idx_list)}
        counts = np.zeros((n_rows, n_langs), dtype=np.int32)
        totals = np.ones(n_rows, dtype=np.int32)

        # Collect per-shard token occurrences (row_pos, word_id)
        per_shard_rows = [[] for _ in shards]
        per_shard_wids = [[] for _ in shards]

        for rid, text in texts.items():
            r = rowpos[rid]
            toks = _tokenize_cached(text)
            totals[r] = max(1, len(toks))
            for tok in toks:
                s = (hash(tok) & 0xFFFFFFFF) % len(shards)
                w2id = shards[s]["w2id"]
                wid = w2id.get(tok)
                if wid is not None:
                    per_shard_rows[s].append(r)
                    per_shard_wids[s].append(wid)

        def _prep_accum(s):
            sh = shards[s]
            ptr = sh["ptr"]
            idx = sh["idx"]
            rows_list = per_shard_rows[s]
            wids_list = per_shard_wids[s]
            if not wids_list:
                return None
            rows = np.asarray(rows_list, dtype=np.int32)
            wids = np.asarray(wids_list, dtype=np.int32)
            lengths = ptr[wids + 1] - ptr[wids]
            starts = ptr[wids]
            rep_rows = np.repeat(rows, lengths)
            take_positions = np.concatenate([
                np.arange(starts[i], starts[i] + lengths[i], dtype=np.int32)
                for i in range(len(wids))
            ])
            lang_ids = idx[take_positions]
            return rep_rows, lang_ids

        shard_indices = list(range(len(shards)))
        work = []
        if self._sparse_workers and self._sparse_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self._sparse_workers) as ex:
                for res in ex.map(_prep_accum, shard_indices):
                    if res is not None:
                        work.append(res)
        else:
            for s in shard_indices:
                res = _prep_accum(s)
                if res is not None:
                    work.append(res)
        # Accumulate on main thread to avoid races
        for rep_rows, lang_ids in work:
            np.add.at(counts, (rep_rows, lang_ids), 1)

        # Build result dicts
        result = {}
        for i, rid in enumerate(idx_list):
            c = counts[i]
            tot = float(totals[i])
            rec = {langs[j]: (c[j] / tot) for j in range(n_langs) if c[j] > 0}
            result[rid] = rec
        return pd.Series(result)

    def _score_apply(self, texts: pd.Series) -> pd.Series:
        """Apply-based scoring wrapper for calibration and fallback."""
        return texts.swifter.apply(self._score_sentence)

    def _score_sentence(self, sentence):
        """Score a sentence using the dictionary of unique words for each language.
        Each sentence gets a score from 0 to 1, which is known_words / total tokens.
        """
        scores = {}

        # Fast Unicode-aware tokenization with LRU caching
        tokens = _tokenize_cached(sentence)
        if not tokens:
            return {}

        for word in tokens:
            for lang in self.word_lang_freq.get(word, {}).keys():
                if lang in self.languages:
                    scores.setdefault(lang, 0)
                    scores[lang] += 1

        # Normalize by total tokens
        total = len(tokens)
        for lang in list(scores.keys()):
            scores[lang] /= total

        return scores

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "word_lang_freq": self.word_lang_freq,
                    "languages": list(self.languages),
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
            model = cls()
            model.word_lang_freq = data["word_lang_freq"]
            model.languages = set(data["languages"])
            return model

    def integrate(self, dataset, eval_df):
        """To be implemented"""
        pass

    # ===== Normalized statistics and algebraic utilities =====
    def recompute_lang_token_totals(self):
        """Recompute per-language total token counts from word_lang_freq and cache them."""
        totals = {}
        for w, m in self.word_lang_freq.items():
            for l, c in m.items():
                totals[l] = totals.get(l, 0) + int(c)
        self._lang_token_totals = totals
        self.languages.update(totals.keys())
        return totals

    def get_language_word_probs(self, alpha: float = 0.0):
        """Return lang -> {word: P(word|lang)} using MLE with optional Laplace smoothing.

        If alpha==0, P(word|lang) = count(word,lang)/sum_w count(w,lang).
        If alpha>0, apply Laplace smoothing with vocabulary size per language.
        """
        if self._lang_token_totals is None:
            self.recompute_lang_token_totals()
        totals = self._lang_token_totals
        per_lang_vocab = {l: 0 for l in self.languages}
        for w, m in self.word_lang_freq.items():
            for l in m.keys():
                per_lang_vocab[l] += 1
        probs = {l: {} for l in self.languages}
        for w, m in self.word_lang_freq.items():
            for l in self.languages:
                c = float(m.get(l, 0))
                if alpha == 0.0:
                    denom = float(totals.get(l, 0))
                    if c > 0 and denom > 0:
                        probs[l][w] = c / denom
                else:
                    denom = float(totals.get(l, 0)) + alpha * per_lang_vocab.get(l, 0)
                    if denom > 0:
                        probs[l][w] = (c + alpha) / denom
        return probs

    def merge(self, other: "Vocabulous") -> "Vocabulous":
        """Return a new Vocabulous with counts equal to the sum of the two models.

        Ensures: train(A+B) == merge(train(A), train(B)) for additive training.
        """
        merged = Vocabulous()
        merged.languages = set(self.languages) | set(other.languages)
        out = {}
        for src in (self.word_lang_freq, other.word_lang_freq):
            for w, m in src.items():
                dst = out.setdefault(w, {})
                for l, c in m.items():
                    dst[l] = int(dst.get(l, 0)) + int(c)
        merged.word_lang_freq = out
        merged._dict_dirty = True
        merged._lang_token_totals = None
        return merged

    def partial_train(
        self,
        train_df,
        eval_df=None,
        cycles=1,
        base_confidence=0.5,
        confidence_margin=0.5,
        text_column="text",
        lang_column="lang",
    ):
        """Incrementally train with new data by training a temporary model and merging.

        This avoids resetting current counts within self and ensures:
        train(A).partial_train(B) == train(A+B).
        """
        tmp = Vocabulous()
        tmp, _ = tmp.train(
            train_df,
            eval_df if eval_df is not None else train_df,
            cycles=cycles,
            base_confidence=base_confidence,
            confidence_margin=confidence_margin,
            text_column=text_column,
            lang_column=lang_column,
        )
        merged = self.merge(tmp)
        # In-place update
        self.word_lang_freq = merged.word_lang_freq
        self.languages = merged.languages
        self._dict_dirty = True
        self._lang_token_totals = None
        return self

    def language_overlap(self, metric: str = "cosine"):
        """Compute pairwise language similarity based on word distributions.

        metric: 'cosine' or 'jaccard'. Returns DataFrame [langs x langs].
        """
        import numpy as _np
        import pandas as _pd
        langs = sorted(list(self.languages))
        if not langs:
            return _pd.DataFrame()
        vocab = list(self.word_lang_freq.keys())
        V = len(vocab)
        L = len(langs)
        if V == 0:
            return _pd.DataFrame(_np.eye(L), index=langs, columns=langs)
        widx = {w: i for i, w in enumerate(vocab)}
        if metric == "cosine":
            probs = self.get_language_word_probs(alpha=0.0)
            M = _np.zeros((L, V), dtype=_np.float64)
            for li, l in enumerate(langs):
                for w, p in probs.get(l, {}).items():
                    M[li, widx[w]] = p
            norms = _np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            M /= norms
            sim = M @ M.T
            return _pd.DataFrame(sim, index=langs, columns=langs)
        elif metric == "jaccard":
            B = _np.zeros((L, V), dtype=_np.uint8)
            for w, m in self.word_lang_freq.items():
                j = widx[w]
                for li, l in enumerate(langs):
                    if m.get(l, 0):
                        B[li, j] = 1
            inter = B @ B.T
            rs = B.sum(axis=1)
            union = rs[:, None] + rs[None, :] - inter
            union[union == 0] = 1
            J = inter / union
            return _pd.DataFrame(J, index=langs, columns=langs)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def plot_language_overlap(self, metric: str = "cosine"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        mat = self.language_overlap(metric=metric)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(mat, ax=ax, cmap="viridis")
        ax.set_title(f"Language overlap ({metric})")
        return fig

    def clean(self, dataset):
        """Score dataset and filter to keep only confident predictions that match ground truth.
        Operates at sentence level for better granularity.

        Args:
            dataset: DataFrame with 'text' and 'lang' columns

        Returns:
            DataFrame containing only sentences where:
            1. Top scoring language has significantly higher score than second best
            2. Top scoring language matches the ground truth language label
        """
        # Expand to sentence level first
        logging.info("Expanding dataset to sentence level for cleaning...")
        sentence_df = self._expand_to_sentence_level(dataset)

        # Score the dataset at sentence level
        scored_df = self._score(sentence_df)

        # If no sentences remain after cleaning/scoring, return empty with expected columns
        if scored_df.empty:
            for col in [
                "top_lang",
                "top_score",
                "second_lang",
                "second_score",
            ]:
                scored_df[col] = pd.Series(dtype="object")
            return scored_df

        # Get top 2 scores for each row
        def get_top_2(scores):
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) == 0:
                return None, None, None, None
            elif len(sorted_scores) == 1:
                return (
                    sorted_scores[0][0],
                    sorted_scores[0][1],
                    None,
                    0.0,
                )
            else:
                return (
                    sorted_scores[0][0],
                    sorted_scores[0][1],
                    sorted_scores[1][0],
                    sorted_scores[1][1],
                )

        (
            scored_df["top_lang"],
            scored_df["top_score"],
            scored_df["second_lang"],
            scored_df["second_score"],
        ) = zip(*scored_df["scores"].swifter.apply(get_top_2))

        # Filter to keep only confident predictions matching ground truth
        # Handle cases where second_score might be None or 0.0
        second_score_safe = scored_df["second_score"].fillna(0.0)
        confident_df = scored_df[
            (scored_df["top_score"] > second_score_safe)  # Top score higher than second
            & (scored_df["top_lang"] == scored_df["lang"])  # Matches ground truth
            & (scored_df["top_lang"].notna())  # Has a valid top language
        ]

        return confident_df


@lru_cache(maxsize=200_000)
def _tokenize_cached(text: str) -> tuple:
    if not text:
        return tuple()
    return tuple(re.findall(r"\w+", text, flags=re.UNICODE))
