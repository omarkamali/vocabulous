import pytest
import pandas as pd
import json
import os
import tempfile
from vocabulous import Vocabulous


class TestVocabulousInit:
    """Test Vocabulous initialization."""

    def test_init_default(self):
        """Test default initialization."""
        model = Vocabulous()
        assert model.word_lang_freq == {}
        assert model.store_training_data is False
        assert model.training_data is None
        assert model.languages == set()

    def test_init_with_training_data_storage(self):
        """Test initialization with training data storage enabled."""
        model = Vocabulous(store_training_data=True)
        assert model.store_training_data is True
        assert model.training_data == []


class TestTextCleaning:
    """Test text cleaning functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello, world!"
        cleaned = self.model._clean_text(text)
        assert cleaned == "hello world"

    def test_clean_text_arabic(self):
        """Test Arabic text cleaning (should preserve case)."""
        text = "مرحبا بالعالم"
        cleaned = self.model._clean_text(text)
        assert cleaned == "مرحبا بالعالم"

    def test_clean_text_mixed_punctuation(self):
        """Test cleaning text with various punctuation."""
        text = "Hello... world!!! How are you???"
        cleaned = self.model._clean_text(text)
        assert cleaned == "hello world how are you"

    def test_clean_text_repeating_letters(self):
        """Test collapsing repeating letters."""
        text = "hellooooo worldddd"
        cleaned = self.model._clean_text(text)
        assert (
            cleaned == "helloo worldd"
        )  # Unscript collapses 3+ repetitions to 2 to preserve real words like "cool"

    def test_clean_text_numbers_only(self):
        """Test that number-only text returns empty string."""
        text = "12345"
        cleaned = self.model._clean_text(text)
        assert cleaned == ""

    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        text = ""
        cleaned = self.model._clean_text(text)
        assert cleaned == ""

    def test_clean_text_whitespace_only(self):
        """Test cleaning whitespace-only text."""
        text = "   \n\t   "
        cleaned = self.model._clean_text(text)
        assert cleaned == ""


class TestScoringFunctionality:
    """Test sentence scoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()
        # Set up a simple dictionary for testing
        self.model.word_lang_freq = {
            "hello": {"en": 5, "de": 1},
            "world": {"en": 4},
            "bonjour": {"fr": 5},
            "monde": {"fr": 3},
            "hallo": {"de": 6},
            "welt": {"de": 4},
        }
        self.model.languages = {"en", "fr", "de"}

    def test_score_sentence_english(self):
        """Test scoring an English sentence."""
        scores = self.model._score_sentence("hello world")
        assert "en" in scores
        assert scores["en"] == 1.0  # 2 words known / 2 total words

    def test_score_sentence_french(self):
        """Test scoring a French sentence."""
        scores = self.model._score_sentence("bonjour monde")
        assert "fr" in scores
        assert scores["fr"] == 1.0  # 2 words known / 2 total words

    def test_score_sentence_mixed(self):
        """Test scoring a sentence with mixed language words."""
        scores = self.model._score_sentence("hello monde")
        assert "en" in scores
        assert "fr" in scores
        assert scores["en"] == 0.5  # 1 word known / 2 total words
        assert scores["fr"] == 0.5  # 1 word known / 2 total words

    def test_score_sentence_unknown_words(self):
        """Test scoring a sentence with unknown words."""
        scores = self.model._score_sentence("unknown words here")
        assert scores == {}

    def test_score_sentence_empty(self):
        """Test scoring an empty sentence."""
        scores = self.model._score_sentence("")
        assert scores == {}


class TestVectorizedEquivalence:
    """Ensure vectorized scoring matches per-row scoring results."""

    def setup_method(self):
        self.model = Vocabulous()
        self.model.word_lang_freq = {
            "hello": {"en": 5, "de": 1},
            "world": {"en": 4},
            "bonjour": {"fr": 5},
            "monde": {"fr": 3},
            "hallo": {"de": 6},
            "welt": {"de": 4},
        }
        self.model.languages = {"en", "fr", "de"}

    def test_vectorized_scores_match_row_apply(self):
        df = pd.DataFrame(
            [
                {"text": "hello world", "lang": "en"},
                {"text": "bonjour monde", "lang": "fr"},
                {"text": "hallo welt", "lang": "de"},
                {"text": "hello bonjour", "lang": "en"},
                {"text": "unknown tokens here", "lang": "en"},
                {"text": "", "lang": "en"},
            ]
        )

        # Clean through model to emulate pipeline
        df["text"] = df["text"].apply(self.model._clean_text)
        vec_scores = self.model._score_vectorized(df["text"]).tolist()
        row_scores = df["text"].apply(self.model._score_sentence).tolist()
        assert vec_scores == row_scores

    def test_score_dataframe_vectorized_path(self):
        df = pd.DataFrame(
            [
                {"text": "Hello world", "lang": "en"},
                {"text": "Bonjour monde", "lang": "fr"},
                {"text": "Hallo Welt", "lang": "de"},
            ]
        )
        scored = self.model._score(df)
        # Should have precomputed top fields and scores dicts
        assert "scores" in scored.columns
        assert "top_lang" in scored.columns
        assert "top_score" in scored.columns


class TestTrainingData:
    """Test training data processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()
        self.sample_train_data = [
            {"text": "Hello world", "lang": "en"},
            {"text": "Bonjour monde", "lang": "fr"},
            {"text": "Hallo Welt", "lang": "de"},
            {"text": "Hello everyone", "lang": "en"},
            {"text": "Bonjour tout le monde", "lang": "fr"},
        ]
        self.sample_eval_data = [
            {"text": "Hello there", "lang": "en"},
            {"text": "Bonjour amis", "lang": "fr"},
            {"text": "Hallo Freunde", "lang": "de"},
        ]

    def test_deduplicate(self):
        """Test data deduplication."""
        data_with_dups = [
            {"text": "Hello world", "lang": "en"},
            {"text": "Hello world", "lang": "en"},  # Duplicate
            {"text": "Bonjour monde", "lang": "fr"},
        ]
        df = pd.DataFrame(data_with_dups)
        deduplicated = self.model._deduplicate(df)
        assert len(deduplicated) == 2
        assert "Hello world" in deduplicated["text"].values

    def test_score_dataframe(self):
        """Test scoring a dataframe."""
        self.model.word_lang_freq = {
            "hello": {"en": 5},
            "world": {"en": 4},
            "bonjour": {"fr": 5},
        }
        self.model.languages = {"en", "fr"}

        df = pd.DataFrame(
            [{"text": "Hello world", "lang": "en"}, {"text": "Bonjour", "lang": "fr"}]
        )

        scored_df = self.model._score(df)
        assert "scores" in scored_df.columns
        assert len(scored_df) == 2


class TestTrainingProcess:
    """Test the training process."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()
        self.train_data = [
            {"text": "Hello world how are you", "lang": "en"},
            {"text": "Good morning everyone", "lang": "en"},
            {"text": "Bonjour tout le monde", "lang": "fr"},
            {"text": "Comment allez vous", "lang": "fr"},
            {"text": "Hallo wie geht es dir", "lang": "de"},
            {"text": "Guten Morgen alle", "lang": "de"},
        ]
        self.eval_data = [
            {"text": "Hello there", "lang": "en"},
            {"text": "Bonjour amis", "lang": "fr"},
            {"text": "Hallo Freunde", "lang": "de"},
        ]

    def test_train_basic(self):
        """Test basic training functionality."""
        model, report = self.model.train(
            self.train_data,
            self.eval_data,
            cycles=1,
            base_confidence=0.1,
            confidence_margin=0.1,
        )

        assert isinstance(model, Vocabulous)
        assert "cycles" in report
        assert "dictionary_size" in report
        assert len(model.word_lang_freq) > 0
        assert len(model.languages) == 3  # en, fr, de

    def test_train_multiple_cycles(self):
        """Test training with multiple cycles."""
        model, report = self.model.train(
            self.train_data,
            self.eval_data,
            cycles=2,
            base_confidence=0.1,
            confidence_margin=0.1,
        )

        assert report["cycles"] <= 2  # May stop early
        assert "cycle_reports" in report
        assert len(report["cycle_reports"]) >= 1


class TestModelPersistence:
    """Test model saving and loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()
        self.model.word_lang_freq = {
            "hello": {"en": 5},
            "world": {"en": 4},
            "bonjour": {"fr": 5},
        }
        self.model.languages = {"en", "fr"}

    def test_save_and_load(self):
        """Test saving and loading a model."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save the model
            self.model.save(temp_path)

            # Verify file exists and has content
            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)
                assert "word_lang_freq" in data
                assert "languages" in data

            # Load the model
            loaded_model = Vocabulous.load(temp_path)

            # Verify loaded model matches original
            assert loaded_model.word_lang_freq == self.model.word_lang_freq
            assert loaded_model.languages == self.model.languages

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataCleaning:
    """Test dataset cleaning functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()
        self.model.word_lang_freq = {
            "hello": {"en": 10},
            "world": {"en": 8},
            "good": {"en": 6},
            "bonjour": {"fr": 10},
            "monde": {"fr": 8},
            "salut": {"fr": 6},
        }
        self.model.languages = {"en", "fr"}

    def test_clean_dataset(self):
        """Test cleaning a dataset with confident predictions."""
        dataset = pd.DataFrame(
            [
                {"text": "hello world", "lang": "en"},  # Should be confident English
                {"text": "bonjour monde", "lang": "fr"},  # Should be confident French
                {"text": "hello bonjour", "lang": "en"},  # Mixed - might be filtered
                {
                    "text": "unknown text",
                    "lang": "en",
                },  # Unknown words - should be filtered
            ]
        )

        cleaned_df = self.model.clean(dataset)

        # Should have at least the confident predictions
        assert len(cleaned_df) >= 2
        assert "top_lang" in cleaned_df.columns
        assert "top_score" in cleaned_df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = Vocabulous()

    def test_empty_training_data(self):
        """Test training with empty data."""
        empty_train = []
        empty_eval = []

        model, report = self.model.train(empty_train, empty_eval, cycles=1)
        assert len(model.word_lang_freq) == 0
        assert len(model.languages) == 0

    def test_single_language_training(self):
        """Test training with only one language."""
        single_lang_data = [
            {"text": "Hello world", "lang": "en"},
            {"text": "Good morning", "lang": "en"},
        ]

        model, report = self.model.train(single_lang_data, single_lang_data, cycles=1)
        assert "en" in model.languages
        assert len(model.languages) == 1

    def test_train_with_pandas_dataframe(self):
        """Test training with pandas DataFrame input."""
        train_df = pd.DataFrame(
            [
                {"text": "Hello world", "lang": "en"},
                {"text": "Bonjour monde", "lang": "fr"},
            ]
        )
        eval_df = pd.DataFrame(
            [
                {"text": "Hello there", "lang": "en"},
                {"text": "Bonjour amis", "lang": "fr"},
            ]
        )

        model, report = self.model.train(train_df, eval_df, cycles=1)
        assert len(model.languages) == 2

    def test_text_with_only_punctuation(self):
        """Test cleaning text with only punctuation."""
        text = "!@#$%^&*()"
        cleaned = self.model._clean_text(text)
        assert cleaned == ""

    def test_confusion_calculation_empty(self):
        """Test confusion calculation with empty data."""
        empty_df = pd.DataFrame(columns=["lang", "predicted_lang"])
        confusion = self.model._calculate_confusion(empty_df, "en", "fr")
        assert confusion == 0.0


if __name__ == "__main__":
    pytest.main([__file__])


class TestSentenceExpansion:
    """Test expansion to sentence level using vectorized explode path."""

    def setup_method(self):
        self.model = Vocabulous()

    def test_expand_to_sentence_level(self):
        df = pd.DataFrame(
            [
                {"text": "Hello world. How are you?", "lang": "en"},
                {"text": "Bonjour le monde! Ça va bien.", "lang": "fr"},
            ]
        )
        expanded = self.model._expand_to_sentence_level(df, text_column="text", lang_column="lang")

        # Expect at least 2 sentences from first row and 2 from second (NLTK splits on . ? !)
        assert len(expanded) >= 4
        # Ensure each expanded row retains language and has non-empty sentence
        assert expanded["lang"].isin(["en", "fr"]).all()
        assert (expanded["text"].str.len() > 0).all()


class TestTokenizationBehavior:
    """Test tokenization behavior for punctuation and unicode words."""

    def setup_method(self):
        self.model = Vocabulous()
        self.model.word_lang_freq = {
            "hello": {"en": 5},
            "world": {"en": 4},
            "مرحبا": {"ar": 3},
        }
        self.model.languages = {"en", "ar"}

    def test_score_sentence_ignores_punctuation(self):
        scores_plain = self.model._score_sentence("hello world")
        scores_punct = self.model._score_sentence("hello, world!!!")
        assert scores_plain == scores_punct

    def test_score_sentence_unicode_tokens(self):
        scores = self.model._score_sentence("مرحبا world")
        assert "ar" in scores
        assert "en" in scores
        # 2 tokens, both known: expect 0.5 for each
        assert scores["ar"] == 0.5
        assert scores["en"] == 0.5


class TestScoreAlreadyClean:
    """Test the ability to skip cleaning inside _score when input is pre-cleaned."""

    def setup_method(self):
        self.model = Vocabulous()
        self.model.word_lang_freq = {
            "hello": {"en": 10},
            "world": {"en": 8},
        }
        self.model.languages = {"en"}

    def test_score_dataframe_already_clean_matches(self):
        raw_df = pd.DataFrame([
            {"text": "Hello, world!!!", "lang": "en"},
            {"text": "hello world", "lang": "en"},
        ])

        # Clean the text externally as train() would
        cleaned_df = raw_df.copy()
        cleaned_df["text"] = cleaned_df["text"].apply(lambda t: self.model._clean_text(t))

        # Score with cleaning inside _score
        scored_with_clean = self.model._score(raw_df.copy())
        # Score without cleaning as input is already cleaned
        scored_already_clean = self.model._score(cleaned_df.copy(), already_clean=True)

        # The scores should match row-wise
        assert scored_with_clean["scores"].tolist() == scored_already_clean["scores"].tolist()


class TestCleanLRUBehavior:
    """Exercise cleaning with many duplicates to ensure deterministic outputs at scale.
    This indirectly validates the viability of caching without relying on mocks.
    """

    def setup_method(self):
        self.model = Vocabulous()

    def test_clean_text_idempotent_many_duplicates(self):
        base = "Hello, world!!!"
        df = pd.DataFrame({"text": [base] * 5000 + ["  \t  "] * 100 + ["12345"] * 100})
        cleaned = df["text"].apply(lambda t: self.model._clean_text(t))
        # All 'Hello, world!!!' should normalize to 'hello world'
        assert (cleaned.iloc[:5000] == "hello world").all()
        # Whitespace-only should become empty
        assert (cleaned.iloc[5000:5100] == "").all()
        # Numbers-only should become empty
        assert (cleaned.iloc[5100:] == "").all()


class TestConfusionAndTop2:
    """Tests for confusion calculation and precomputed top-2 usage in cleaning."""

    def setup_method(self):
        self.model = Vocabulous()
        self.model.languages = {"en", "fr"}

    def test_calculate_confusion_pair(self):
        # Create a small scored df with true and predicted
        df = pd.DataFrame(
            {
                "lang": ["en", "en", "fr", "fr", "en"],
                "predicted_lang": ["en", "fr", "fr", "en", "fr"],
            }
        )
        # For pair (en, fr): relevant rows = all (since only en/fr present)
        # Confusions are en->fr and fr->en: here 3/5 (rows 2,3,4 0-based indexing)
        confusion = self.model._calculate_confusion(df, "en", "fr")
        assert pytest.approx(confusion, rel=1e-9) == 3 / 5

    def test_cycle_clean_uses_precomputed_top2(self):
        # Construct a df with precomputed top/second scores
        df = pd.DataFrame(
            [
                {"text": "hello world", "lang": "en", "scores": {"en": 1.0}},
                {
                    "text": "bonjour monde",
                    "lang": "fr",
                    "scores": {"fr": 0.6, "en": 0.55},
                },
                {
                    "text": "hello bonjour",
                    "lang": "en",
                    "scores": {"en": 0.51, "fr": 0.5},
                },
            ]
        )
        # Precompute
        def get_top_2(scores):
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) == 0:
                return None, None
            elif len(sorted_scores) == 1:
                return sorted_scores[0][1], 0.0
            else:
                return sorted_scores[0][1], sorted_scores[1][1]

        df["top_score"], df["second_score"] = zip(*df["scores"].apply(get_top_2))

        cleaned = self.model._cycle_clean(df, base_confidence=0.5, confidence_margin=0.05)
        # First row should pass (1.0 >= 0.5 and matches en; infinite margin or 1.0-0)
        # Second row should fail (fr top but margin 0.6-0.55=0.05 equals threshold? strict > used)
        # Third row: passes base_confidence and margin 0.01, but matches en (top) — margin < 0.05 so filtered out
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["text"] == "hello world"
