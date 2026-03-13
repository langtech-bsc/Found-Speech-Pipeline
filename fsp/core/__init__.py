"""
Core processing modules for the FSP pipeline.

- text: Text normalization, cleaning, and splitting
- audio: Audio normalization and duration filtering
- alignment: Forced alignment and ASR transcription
- rover: ROVER merge and scoring
- segmenter: Audio segmentation
"""

from fsp.core.alignment import (
    CTC_MODELS,
    MODELS_BY_LANG,
    build_manifest,
    generate_final_data,
    transcribe,
)
from fsp.core.audio import filter_and_cleanup, normalize_audio, select_tsv
from fsp.core.rover import RoverConfig, centroid, cer, majority_vote, process_file, wer
from fsp.core.segmenter import Segmenter
from fsp.core.text import (
    clean_text,
    expand_text,
    numbers_to_chars,
    remove_chars,
    replace_chars,
    split_text,
)

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
