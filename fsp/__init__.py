"""
Found-Speech-Pipeline (FSP) - A modular pipeline for speech processing.

This package provides:
- fsp.core: Core processing modules (text, audio, alignment, rover)
- fsp.utils: Utility modules (paths, language, models)
- fsp.pipeline: Pipeline orchestration class
"""

from fsp.pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Pipeline"]
