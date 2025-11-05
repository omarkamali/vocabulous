# Vocabulous

A bootstrapping language detection system that builds high-quality dictionaries from noisy training data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/vocabulous.svg)](https://badge.fury.io/py/vocabulous)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vocabulous Tests](https://github.com/omarkamali/vocabulous/actions/workflows/pytest.yml/badge.svg)](https://github.com/omarkamali/vocabulous/actions/workflows/pytest.yml)


## Overview

Vocabulous addresses a common challenge in NLP: building reliable language detection systems when you only have noisy, potentially mislabeled training data. Traditional approaches either require clean, manually curated datasets or sophisticated neural networks. Vocabulous takes a different approach by using iterative dictionary building and progressive data cleaning to bootstrap accurate language detection from imperfect data.

### Key Features

- **Bootstrapping from Noisy Data**: Starts with potentially mislabeled training data and iteratively improves
- **Dictionary-Based Detection**: Uses word frequency dictionaries for fast, interpretable language detection  
- **Progressive Data Cleaning**: Removes ambiguous and mislabeled samples across training cycles
- **Multi-Script Support**: Handles both Latin and Arabic scripts with appropriate text normalization
- **Configurable Training**: Adjustable confidence thresholds and confidence margin parameters for different scenarios
- **Model Persistence**: Save and load trained models for reuse

## Installation

```bash
uv pip install vocabulous
```

### Development Installation

```bash
git clone https://github.com/omarkamali/vocabulous.git
cd vocabulous
uv pip install -e ".[dev]"
```

## Quick Start

```python
from vocabulous import Vocabulous

# Initialize model
model = Vocabulous()

# Prepare training data (list of dicts with 'text' and 'lang' keys)
train_data = [
    {'text': 'Hello world', 'lang': 'en'},
    {'text': 'Bonjour le monde', 'lang': 'fr'},
    {'text': 'Hola mundo', 'lang': 'es'},
    # ... more training examples
]

# Evaluation data for monitoring training progress
eval_data = [
    {'text': 'Good morning', 'lang': 'en'},
    {'text': 'Bon matin', 'lang': 'fr'},
    {'text': 'Buenos días', 'lang': 'es'},
]

# Train the model
model, report = model.train(
    train_data=train_data,
    eval_data=eval_data,
    cycles=3,
    base_confidence=0.5,
    confidence_margin=0.3
)

# Use for language detection
scores = model._score_sentence("Hello there")
print(scores)  # {'en': 1.0}

# Save the model
model.save('my_model.json')

# Load later
loaded_model = Vocabulous.load('my_model.json')
```

## Methodology

### The Bootstrapping Approach

Vocabulous implements a novel bootstrapping methodology for language detection:

1. **Initial Dictionary Building**: Creates word-language frequency dictionaries from all training data
2. **Scoring & Evaluation**: Scores evaluation data to measure current model performance
3. **Data Cleaning**: Removes training samples that contradict the current dictionaries
4. **Iteration**: Repeats the process with cleaned data to progressively improve quality

### Why This Works

The approach is based on several key insights:

- **Majority Signal**: Even noisy datasets typically contain more correct than incorrect labels
- **Word Uniqueness**: Many words are language-specific and provide strong signals
- **Progressive Refinement**: Each iteration removes the most problematic samples first
- **Convergence**: The process naturally converges when no more samples can be confidently removed

### Training Parameters

- **`cycles`**: Number of training iterations (default: 2)
- **`base_confidence`**: Minimum score threshold for keeping samples (0-1)
- **`confidence_margin`**: Minimum difference between top two language scores (0-1)

Higher values make the filtering more aggressive, while lower values are more permissive.

## Use Cases

### 1. Bootstrapping Language Detection

**Scenario**: You have a large dataset of multilingual text with potentially noisy language labels.

```python
# Start with noisy data
noisy_data = [
    {'text': 'Hello world', 'lang': 'en'},
    {'text': 'Bonjour', 'lang': 'en'},  # Mislabeled!
    {'text': 'Hello', 'lang': 'fr'},    # Mislabeled!
    {'text': 'Comment ça va?', 'lang': 'fr'},
    # ... thousands more with ~10% label noise
]

model = Vocabulous()
model, report = model.train(noisy_data, eval_data, cycles=3)

# The model learns to ignore mislabeled samples
print(f"Dictionary size: {len(model.word_lang_freq)}")
print(f"Final accuracy: {report['cycle_reports'][-1]['accuracy']:.3f}")
```

### 2. Data Cleaning Pipeline

**Scenario**: Clean a noisy multilingual dataset before using it for other NLP tasks.

```python
# Train model on subset of data
model, _ = model.train(sample_data, eval_data)

# Clean the full dataset
cleaned_dataset = model.clean(full_noisy_dataset)

# Now use cleaned_dataset for training other models
print(f"Kept {len(cleaned_dataset)}/{len(full_noisy_dataset)} samples")
```

### 3. Incremental Learning

**Scenario**: Continuously improve language detection as new data becomes available.

```python
# Initial training
model, _ = model.train(initial_data, eval_data)

# Later, integrate new data
model, updated_report = model.train(
    new_data + initial_data,  # Combine old and new
    eval_data,
    cycles=2
)
```

### 4. Cross-Domain Adaptation

**Scenario**: Adapt a model trained on one domain (e.g., news) to another (e.g., social media).

```python
# Train on news data
news_model, _ = news_model.train(news_data, news_eval)

# Adapt to social media by combining datasets
adapted_model = Vocabulous()
adapted_model, _ = adapted_model.train(
    social_media_data + news_data,
    social_media_eval,
    cycles=3,
    base_confidence=0.3  # Lower threshold for noisy social media text
)
```

## Advanced Usage

### Custom Text Preprocessing

```python
# Subclass to customize text cleaning
class CustomVocabulous(Vocabulous):
    def _clean_text(self, text):
        # Add custom preprocessing
        text = super()._clean_text(text)
        # Your custom logic here
        return text
```

### Training Monitoring

```python
model, report = model.train(train_data, eval_data, cycles=5)

# Analyze training progress
for i, cycle_report in enumerate(report['cycle_reports']):
    print(f"Cycle {i+1}:")
    print(f"  Accuracy: {cycle_report['accuracy']:.3f}")
    print(f"  F1 Score: {cycle_report['f1']:.3f}")
    print(f"  Samples removed: {cycle_report['removed_samples']}")
    print(f"  Confidence Margin: {cycle_report['confidence_margin']:.3f}")
```

### Scoring backends

- **Default**: swifter-accelerated Pandas apply via `model._score(...)`.
- **Alternatives (experimental)**: `vectorized`, `numba`, `sparse` backends exist and are used in benchmarks.

Planned API to switch backends:

```python
model.set_scoring_mode("vectorized")  # or: "apply", "numba", "sparse", "auto"
scored = model._score(df)
```

Until switching is wired end-to-end, call experimental methods directly:

```python
# Vectorized scoring for a text Series -> Series[dict]
scores_vec = model._score_vectorized(df["text"])  # experimental API
df = df.copy()
df["scores"] = scores_vec

# Numba-backed scoring (if numba installed) -> Series[dict]
scores_numba = model._score_numba(df["text"])  # experimental API

# Note: _score(...) remains the default swifter-apply path as of 0.1.2
```

### Confidence Scoring

```python
# Get detailed scores for a sentence
scores = model._score_sentence("Hello world")
# Returns: {'en': 0.75, 'fr': 0.25}

# For datasets
scored_df = model._score(test_data)
print(scored_df[['text', 'scores', 'lang']])
```

## Performance Tips

### Memory Optimization

```python
# For large datasets, disable training data storage
model = Vocabulous(store_training_data=False)
```

### Speed Optimization

```python
# Use fewer cycles for faster training
model, _ = model.train(data, eval_data, cycles=1)

# Lower confidence margin for less aggressive filtering
model, _ = model.train(data, eval_data, confidence_margin=0.1)
```

### Quality Optimization

```python
# More cycles for higher quality
model, _ = model.train(data, eval_data, cycles=5)

# Higher confidence threshold for cleaner dictionaries
model, _ = model.train(data, eval_data, base_confidence=0.7)
```

## Evaluation Metrics
 
## Benchmarks

- **Full results**: See [benchmark.md](./benchmarks/benchmark.md) for the complete output and methodology.
- **Highlights**:
  - 20k rows clean+score: ~454k rows/s
  - Apply vs Vectorized: ~450–500k rows/s on 5k–50k rows
  - Longer sentences reduce throughput (len=50 ~190k rows/s)
  - Dictionary size (50→5000 per language): near ~450–480k rows/s
  - Large-n vectorized batched (100k): ~403k rows/s
  - Large-n compare (200k, dict=1000, len=20): ~224k–225k rows/s across modes

Run locally with uv:

```bash
uv run python benchmarks/benchmark_vocabulous.py | tee benchmarks/benchmark_output.txt
```

## Classification Performance

Vocabulous provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-language and macro-averaged metrics
- **Confusion Score**: Measures how often languages are confused with each other
- **Confidence Margin**: Average difference between top two language scores (higher = more confident)

## Limitations

1. **Vocabulary-Based**: Works best with languages that have distinct vocabularies
2. **Training Data Size**: Requires sufficient training data for each language
3. **Script Mixing**: May struggle with code-switched text within sentences
4. **Short Text**: Performance degrades on very short texts (1-2 words)

## API Reference

### Core Classes

#### `Vocabulous(store_training_data=False)`

Main class for language detection and training.

**Parameters:**
- `store_training_data` (bool): Whether to store training data internally

#### Methods

##### `train(train_df, eval_df, cycles=2, base_confidence=0.5, confidence_margin=0.5)`

Train the model on provided data.

**Parameters:**
- `train_df`: Training data (list of dicts or DataFrame)
- `eval_df`: Evaluation data (list of dicts or DataFrame)
- `cycles` (int): Number of training cycles
- `base_confidence` (float): Minimum confidence threshold
- `confidence_margin` (float): Minimum score difference threshold

**Returns:**
- `(model, report)`: Updated model and training report

##### `clean(dataset)`

Clean a dataset by filtering confident predictions.

**Parameters:**
- `dataset`: DataFrame with 'text' and 'lang' columns

**Returns:**
- DataFrame with confident predictions only

##### `save(path)` / `load(path)`

Save/load model to/from JSON file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/omarkamali/vocabulous.git
cd vocabulous
pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is supported by Omneity Labs, a research lab focused on building NLP and generative AI models for low-resource languages and techniques for cultural alignment.

## Contributors

[Omar Kamali](https://omarkama.li)

## Citation

If you use Vocabulous in your research, please cite:

```bibtex
@software{vocabulous2025,
  title={Vocabulous: Bootstrapping Language Detection from Noisy \& Ambiguous Data},
  author={Omar Kamali},
  year={2025},
  url={https://github.com/omarkamali/vocabulous},
  note={Project developed under Omneity Labs}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/omarkamali/vocabulous/issues)
- **Discussions**: [GitHub Discussions](https://github.com/omarkamali/vocabulous/discussions)
- **Documentation**: [GitHub README](https://github.com/omarkamali/vocabulous#readme) 