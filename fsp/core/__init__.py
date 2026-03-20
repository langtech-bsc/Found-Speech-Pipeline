"""
Core processing modules for the FSP pipeline.

- text: Text normalization, cleaning, and splitting
- audio: Audio normalization and duration filtering
- alignment: Forced alignment and ASR transcription
- rover: ROVER merge and scoring
- segmenter: Audio segmentation
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # text
    "clean_text",
    "expand_text",
    "replace_chars",
    "numbers_to_chars",
    "remove_chars",
    "split_text",
    # audio
    "normalize_audio",
    "select_tsv",
    "filter_and_cleanup",
    # rover
    "majority_vote",
    "centroid",
    "process_file",
    "cer",
    "wer",
    "RoverConfig",
    # alignment
    "transcribe",
    "build_manifest",
    "generate_final_data",
    "MODELS_BY_LANG",
    "CTC_MODELS",
    # segmenter
    "Segmenter",
]


def __getattr__(name: str) -> Any:
    if name in {"CTC_MODELS", "MODELS_BY_LANG", "build_manifest", "generate_final_data", "transcribe"}:
        from fsp.core import alignment as _alignment

        return getattr(_alignment, name)
    if name in {"filter_and_cleanup", "normalize_audio", "select_tsv"}:
        from fsp.core import audio as _audio

        return getattr(_audio, name)
    if name in {"RoverConfig", "centroid", "cer", "majority_vote", "process_file", "wer"}:
        from fsp.core import rover as _rover

        return getattr(_rover, name)
    if name == "Segmenter":
        from fsp.core.segmenter import Segmenter

        return Segmenter
    if name in {"clean_text", "expand_text", "numbers_to_chars", "remove_chars", "replace_chars", "split_text"}:
        from fsp.core import text as _text

        return getattr(_text, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
