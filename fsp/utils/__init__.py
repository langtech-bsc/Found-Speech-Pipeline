"""
Utility modules for the FSP pipeline.

- paths: Common path constants
- language: Language detection and dictionaries
- models: Model loading/unloading utilities
"""

from __future__ import annotations

from typing import Any

from fsp.utils.language import choose_language
from fsp.utils.paths import (
    GL_EXTRA_ASR_IMAGE,
    GL_EXTRA_ASR_IMAGE_NAME,
    IMAGES_DIR,
    IMAGES_DIR_ENV_VAR,
    HF_MODEL_DIR,
    HF_MODEL_DIR_ENV_VAR,
    INGESTION_DIR,
    LID_MODEL_PATH,
    LID_MODEL_PATH_ENV_VAR,
    MODEL_DIR_ENV_VAR,
    MODELS_DIR,
    NEMO_MODEL_DIR,
    NEMO_MODEL_DIR_ENV_VAR,
    NORM_DIR,
    OUT_DIR,
    ROOT,
    ROVER_DIR,
    SCRIPTS_DIR,
    STEPS_DIR,
    resolve_gl_extra_asr_image,
    resolve_images_dir,
    resolve_hf_model_dir,
    resolve_lid_model_path,
    resolve_model_dir,
    resolve_model_paths,
    resolve_nemo_model_dir,
)

__all__ = [
    # paths
    "ROOT",
    "MODEL_DIR_ENV_VAR",
    "LID_MODEL_PATH_ENV_VAR",
    "NEMO_MODEL_DIR_ENV_VAR",
    "HF_MODEL_DIR_ENV_VAR",
    "IMAGES_DIR_ENV_VAR",
    "GL_EXTRA_ASR_IMAGE_NAME",
    "MODELS_DIR",
    "IMAGES_DIR",
    "LID_MODEL_PATH",
    "NEMO_MODEL_DIR",
    "HF_MODEL_DIR",
    "GL_EXTRA_ASR_IMAGE",
    "resolve_model_dir",
    "resolve_lid_model_path",
    "resolve_nemo_model_dir",
    "resolve_images_dir",
    "resolve_gl_extra_asr_image",
    "resolve_hf_model_dir",
    "resolve_model_paths",
    "SCRIPTS_DIR",
    "STEPS_DIR",
    "INGESTION_DIR",
    "NORM_DIR",
    "OUT_DIR",
    "ROVER_DIR",
    # language
    "choose_language",
    # models
    "configure_model_environment",
    "load_model",
    "unload_model",
    "download_nemo_ctc",
    "download_hf_model",
]


def __getattr__(name: str) -> Any:
    if name in {
        "configure_model_environment",
        "load_model",
        "unload_model",
        "download_nemo_ctc",
        "download_hf_model",
    }:
        from fsp.utils import models as _models

        return getattr(_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
