"""Single-word morphology; importing this package does not load a dictionary."""

from .analyzer import Analysis, Morpheme, TurkishMorphAnalyzer

__all__ = ["Analysis", "Morpheme", "TurkishMorphAnalyzer"]
