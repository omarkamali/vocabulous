"""
Vocabulous: A bootstrapping language detection system that builds dictionaries from noisy training data.

This package provides tools for:
- Building language-specific dictionaries from training data
- Iterative training with noise reduction
- Language detection using dictionary-based scoring
- Dataset cleaning and confidence filtering

Example:
    >>> from vocabulous import Vocabulous
    >>> model = Vocabulous()
    >>> model, report = model.train(train_data, eval_data)
    >>> scores = model._score_sentence("Hello world")
"""

from .vocabulous import Vocabulous

__version__ = "0.1.0"
__author__ = "Vocabulous Contributors"
__all__ = ["Vocabulous"]
