import swifter
import nltk
from tqdm import tqdm
import pandas as pd
import logging
import json
import re
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

    def train(
        self, train_df, eval_df, cycles=2, base_confidence=0.5, confidence_margin=0.5
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
        # Convert to pandas DataFrame if not already
        if not isinstance(train_df, pd.DataFrame):
            train_df = pd.DataFrame(train_df)
        if not isinstance(eval_df, pd.DataFrame):
            eval_df = pd.DataFrame(eval_df)

        # Handle empty data gracefully
        if len(train_df) == 0 or len(eval_df) == 0:
            return self, {
                "cycles": 0,
                "cycle_reports": [],
                "dictionary_size": 0,
                "train_data": train_df,
            }

        # Clean text before training
        train_df["text"] = train_df["text"].apply(lambda x: self._clean_text(x))
        eval_df["text"] = eval_df["text"].apply(lambda x: self._clean_text(x))

        # Remove empty strings after cleaning
        train_df = train_df[train_df["text"] != ""]
        eval_df = eval_df[eval_df["text"] != ""]

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
                logging.info("No samples removed in this cycle - stopping early")
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
                f"  Samples removed: {report['removed_samples']}/{report['total_samples']}"
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
        # first let's deduplicate the training data
        train_df = self._deduplicate(train_df)
        original_samples = len(train_df)
        logging.info(
            f"Starting cycle with {original_samples} samples after deduplication"
        )

        # Reset word_lang_freq for this cycle
        self.word_lang_freq = {}

        # Process sentences using swifter but in a way that avoids race conditions
        def process_sentence(row):
            lang = row["lang"]
            words = set(
                word for word in nltk.word_tokenize(row["text"]) if word.isalnum()
            )
            return {"lang": lang, "words": words}

        # First collect all words and languages
        logging.info("Processing sentences...")
        processed = train_df.swifter.apply(process_sentence, axis=1)

        # Then update the dictionaries safely
        logging.info("Building language dictionaries...")
        for result in tqdm(processed):
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
        logging.info("Scoring training data...")
        train_df = self._score(train_df)

        # Skip cleaning if this is the final cycle
        if not skip_cleaning:
            # now we have a dataframe with scores for each sentence, and we need to clean it by removing:
            # - sentences where the top matching language is different from the ground truth
            # - sentences where the scores are too close to each other (low confidence margin)
            # - sentences where the scores are too low (below a base confidence threshold)
            logging.info("Cleaning training data...")
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
        """Clean text using Unscript library for proper multilingual handling.

        This replaces the previous hardcoded Arabic/Latin handling with
        proper script detection and cleaning.
        """
        if not text or not text.strip():
            return ""

        # Use explicitly set script or detect the dominant script
        if self.default_script:
            script_to_use = self.default_script
        else:
            script_to_use = get_dominant_script(text, min_percentage=20.0)
            # If no dominant script detected, default to Latin
            if script_to_use is None:
                script_to_use = "Latn"

        # Always use unscript for consistent multilingual text processing
        cleaned = unscript(
            script_to_use,
            text,
            {"spaces": True, "numbers": True, "punctuation": False, "symbols": False},
            lowercase=(script_to_use != "Arab"),  # Keep original case for Arabic
        )

        # Check if resulting text is only numbers
        if cleaned and re.match(r"^\d+$", cleaned.strip()):
            return ""

        return cleaned

    def _report_cycle(self, eval_df):
        """Produce a report for the cycle"""
        # Score the evaluation data
        scored_df = self._score(eval_df)

        # Get predicted languages (highest score for each row)
        # Handle empty scores by defaulting to None
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

        # Calculate classification metrics
        true_labels = scored_df["lang"]
        predicted_labels = scored_df["predicted_lang"]

        # Only count non-None predictions
        valid_predictions = [
            (t, p) for t, p in zip(true_labels, predicted_labels) if p is not None
        ]
        if valid_predictions:
            true_labels, predicted_labels = zip(*valid_predictions)
            correct_predictions = sum(t == p for t, p in valid_predictions)
            total_predictions = len(valid_predictions)
        else:
            correct_predictions = 0
            total_predictions = 0

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        # Calculate precision, recall, and F1 for each language and average them
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        for lang in self.languages:
            true_pos = sum(
                (t == lang and p == lang) for t, p in zip(true_labels, predicted_labels)
            )
            false_pos = sum(
                (t != lang and p == lang) for t, p in zip(true_labels, predicted_labels)
            )
            false_neg = sum(
                (t == lang and p != lang) for t, p in zip(true_labels, predicted_labels)
            )

            precision = (
                true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            )
            recall = (
                true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics["precision"] += precision
            metrics["recall"] += recall
            metrics["f1"] += f1

        num_languages = len(self.languages)
        metrics = {k: v / num_languages for k, v in metrics.items()}

        def _get_top_score_diff(scores):
            if not scores:  # Handle empty scores
                return 1
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) < 2:
                return 1  # If only one language, it's infinitely spiky
            return sorted_scores[0] - sorted_scores[1]

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
        # Filter for cases where true label is either lang1 or lang2
        relevant_df = df[df["lang"].isin([lang1, lang2])]

        if len(relevant_df) == 0:
            return 0.0

        # Count cases where true is lang1 and predicted is lang2 or vice versa
        confusions = sum(
            (
                (row["lang"] == lang1 and row["predicted_lang"] == lang2)
                or (row["lang"] == lang2 and row["predicted_lang"] == lang1)
            )
            for _, row in relevant_df.iterrows()
        )

        return confusions / len(relevant_df)

    def _cycle_clean(self, df, base_confidence, confidence_margin):
        """Clean the dataframe by removing sentences where the top matching language is different from the ground truth,
        and sentences where the scores are too close to each other (low confidence margin)
        """

        # remove sentences where the top matching language is different from the ground truth
        def check_top_lang(row):
            scores = row["scores"]
            if not scores:
                return False
            max_score = max(scores.values())
            return (
                max_score == scores.get(row["lang"], 0) and max_score >= base_confidence
            )

        df = df[df.apply(check_top_lang, axis=1)]

        # remove sentences where the scores are too close to each other (low confidence margin)
        def _get_top_score_diff(scores):
            if not scores:  # Handle empty scores
                return 0
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) < 2:
                return float("inf")  # If only one language, difference is infinite
            return sorted_scores[0] - sorted_scores[1]

        df = df[df["scores"].apply(_get_top_score_diff) > confidence_margin]

        return df

    def _deduplicate(self, df):
        """Deduplicate the training data by removing duplicate sentences."""
        return df.drop_duplicates(subset="text")

    def _score(self, df):
        """Score a dataframe using the dictionary of unique words for each language.
        Each sentence gets a score from 0 to 1, which is known_words / total words in the sentence
        """

        # Clean text before scoring
        df["text"] = df["text"].apply(lambda x: self._clean_text(x))
        df = df[df["text"] != ""]  # Remove empty strings after cleaning

        df["scores"] = df["text"].swifter.apply(self._score_sentence)
        return df

    def _score_sentence(self, sentence):
        """Score a sentence using the dictionary of unique words for each language.
        Each sentence gets a score from 0 to 1, which is known_words / total words in the sentence
        """

        scores = {}

        for word in sentence.split():
            # if len(self.word_lang_freq.get(word, {}).keys()) > 1:
            #     continue  # skip words that are ambiguous
            for lang in self.word_lang_freq.get(word, {}).keys():
                if lang in self.languages:
                    scores.setdefault(lang, 0)
                    scores[lang] += 1

        # Divide the score by total of words in the sentence
        for lang in scores.keys():
            scores[lang] /= len(sentence.split())

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

    def clean(self, dataset):
        """Score dataset and filter to keep only confident predictions that match ground truth.

        Args:
            dataset: DataFrame with 'text' and 'lang' columns

        Returns:
            DataFrame containing only rows where:
            1. Top scoring language has significantly higher score than second best
            2. Top scoring language matches the ground truth language label
        """
        # Score the dataset
        scored_df = self._score(dataset)

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
