"""
Found-Speech-Pipeline (FSP) - A modular pipeline for speech processing.

This package provides:
- fsp.core: Core processing modules (text, audio, alignment, rover)
- fsp.utils: Utility modules (paths, language, models)
- fsp.pipeline: Pipeline orchestration class
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["Pipeline"]


def __getattr__(name: str) -> Any:
    if name == "Pipeline":
        from fsp.pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
