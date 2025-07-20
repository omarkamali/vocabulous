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
