"""
Utility modules for the FSP pipeline.

- paths: Common path constants
- language: Language detection and dictionaries
- models: Model loading/unloading utilities
"""

from fsp.utils.language import choose_language
from fsp.utils.models import download_hf_model, download_nemo_ctc, load_model, unload_model
from fsp.utils.paths import (
    INGESTION_DIR,
    NORM_DIR,
    OUT_DIR,
    ROOT,
    ROVER_DIR,
    SCRIPTS_DIR,
    STEPS_DIR,
)

__all__ = [
    # paths
    "ROOT",
    "SCRIPTS_DIR",
    "STEPS_DIR",
    "INGESTION_DIR",
    "NORM_DIR",
    "OUT_DIR",
    "ROVER_DIR",
    # language
    "choose_language",
    # models
    "load_model",
    "unload_model",
    "download_nemo_ctc",
    "download_hf_model",
]
