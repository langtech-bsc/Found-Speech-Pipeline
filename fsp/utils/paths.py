"""
Common path constants and path resolvers for the FSP pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root (parent of fsp/ directory)
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_ROOT = ROOT / "utils" / "models"

MODELS_ROOT_ENV_VAR = "MODELS_ROOT"
MODEL_DIR_ENV_VAR = "MODEL_DIR"
LID_MODEL_PATH_ENV_VAR = "LID_MODEL_PATH"
NEMO_MODEL_DIR_ENV_VAR = "NEMO_MODEL_DIR"
HF_MODEL_DIR_ENV_VAR = "HF_MODEL_DIR"
IMAGES_DIR_ENV_VAR = "FSP_IMAGES_DIR"
GL_EXTRA_ASR_IMAGE_NAME = "fsp-gl-extra-asr.sif"


@dataclass(frozen=True)
class ModelPaths:
    lid_model_path: Path
    nemo_model_dir: Path
    hf_model_dir: Path


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """
    Resolve the shared ASR model root directory.
    """
    if model_dir is None:
        model_dir = os.getenv(MODELS_ROOT_ENV_VAR) or os.getenv(MODEL_DIR_ENV_VAR)

    if model_dir is None:
        return DEFAULT_MODELS_ROOT

    return Path(model_dir).expanduser()


def resolve_lid_model_path(lid_model_path: str | Path | None = None) -> Path:
    """
    Resolve the FastText language-ID model file path.
    """
    if lid_model_path is not None:
        return Path(lid_model_path).expanduser()

    if env_path := os.getenv(LID_MODEL_PATH_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir() / "fasttext" / "lid.176.bin"


def resolve_nemo_model_dir(nemo_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing local NeMo model folders.
    """
    if nemo_model_dir is not None:
        return Path(nemo_model_dir).expanduser()

    if env_path := os.getenv(NEMO_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir()


def resolve_hf_model_dir(hf_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing local HuggingFace model folders.
    """
    if hf_model_dir is not None:
        return Path(hf_model_dir).expanduser()

    if env_path := os.getenv(HF_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir()


def resolve_images_dir(images_dir: str | Path | None = None) -> Path:
    """
    Resolve the shared directory that stores Singularity/Apptainer images.
    """
    if images_dir is not None:
        return Path(images_dir).expanduser()

    if env_path := os.getenv(IMAGES_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    return Path("/gpfs/projects/bsc88/singularity-images/")


def resolve_gl_extra_asr_image(images_dir: str | Path | None = None) -> Path:
    """
    Resolve the GL extra ASR sidecar image path.
    """
    image_dir = resolve_images_dir(images_dir)
    candidates = [
        image_dir / GL_EXTRA_ASR_IMAGE_NAME,
        image_dir / "gl-extra-asr.sif",
        ROOT / GL_EXTRA_ASR_IMAGE_NAME,
        ROOT / "gl-extra-asr.sif",
    ]
    return _first_existing_path(candidates) or candidates[0]


def resolve_model_paths(
    lid_model_path: str | Path | None = None,
    nemo_model_dir: str | Path | None = None,
    hf_model_dir: str | Path | None = None,
) -> ModelPaths:
    """Resolve all runtime model paths together."""
    return ModelPaths(
        lid_model_path=resolve_lid_model_path(lid_model_path),
        nemo_model_dir=resolve_nemo_model_dir(nemo_model_dir),
        hf_model_dir=resolve_hf_model_dir(hf_model_dir),
    )


MODELS_DIR = resolve_model_dir()
LID_MODEL_PATH = resolve_lid_model_path()
NEMO_MODEL_DIR = resolve_nemo_model_dir()
HF_MODEL_DIR = resolve_hf_model_dir()
IMAGES_DIR = resolve_images_dir()
GL_EXTRA_ASR_IMAGE = resolve_gl_extra_asr_image()

# Directory paths
SCRIPTS_DIR = ROOT / "scripts"
STEPS_DIR = ROOT / "steps"
INGESTION_DIR = ROOT / "ingestion"
NORM_DIR = ROOT / "inputs" / "normalized"
OUT_DIR = ROOT / "inputs" / "normalized"
ROVER_DIR = ROOT / "merged"
MANIFEST_DIR = ROOT / "inputs" / "manifest"
ALIGN_DIR = ROOT / "inputs" / "wordlevel_alignment"
OUTPUT_SEGMENT_DIR = ROOT / "inputs" / "output_segment"
LOG_DIR = ROOT / "inputs"

# External commands
FFMPEG_CMD = "ffmpeg"

__all__ = [
    "ModelPaths",
    "ROOT",
    "DEFAULT_MODELS_ROOT",
    "MODELS_ROOT_ENV_VAR",
    "MODEL_DIR_ENV_VAR",
    "LID_MODEL_PATH_ENV_VAR",
    "NEMO_MODEL_DIR_ENV_VAR",
    "HF_MODEL_DIR_ENV_VAR",
    "IMAGES_DIR_ENV_VAR",
    "GL_EXTRA_ASR_IMAGE_NAME",
    "MODELS_DIR",
    "LID_MODEL_PATH",
    "NEMO_MODEL_DIR",
    "HF_MODEL_DIR",
    "IMAGES_DIR",
    "GL_EXTRA_ASR_IMAGE",
    "resolve_model_dir",
    "resolve_lid_model_path",
    "resolve_nemo_model_dir",
    "resolve_hf_model_dir",
    "resolve_images_dir",
    "resolve_gl_extra_asr_image",
    "resolve_model_paths",
    "SCRIPTS_DIR",
    "STEPS_DIR",
    "INGESTION_DIR",
    "NORM_DIR",
    "OUT_DIR",
    "ROVER_DIR",
    "MANIFEST_DIR",
    "ALIGN_DIR",
    "OUTPUT_SEGMENT_DIR",
    "LOG_DIR",
    "FFMPEG_CMD",
]
