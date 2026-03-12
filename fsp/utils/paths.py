"""
Common path constants and path resolvers for the FSP pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root (parent of fsp/ directory)
ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR_ENV_VAR = "MODEL_DIR"
LID_MODEL_PATH_ENV_VAR = "LID_MODEL_PATH"
NEMO_MODEL_DIR_ENV_VAR = "NEMO_MODEL_DIR"
HF_MODEL_DIR_ENV_VAR = "HF_MODEL_DIR"


@dataclass(frozen=True)
class ModelPaths:
    lid_model_path: Path
    nemo_model_dir: Path
    hf_model_dir: Path


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """
    Resolve the legacy shared model root directory.

    Resolution order:
    1. Explicit function argument
    2. MODEL_DIR environment variable
    3. Repository default: ROOT / "utils" / "models"
    """
    if model_dir is None:
        model_dir = os.getenv(MODEL_DIR_ENV_VAR)

    if model_dir is None:
        return ROOT / "utils" / "models"

    return Path(model_dir).expanduser()


def resolve_lid_model_path(lid_model_path: str | Path | None = None) -> Path:
    """
    Resolve the FastText language-ID model file path.

    Resolution order:
    1. Explicit function argument
    2. LID_MODEL_PATH environment variable
    3. Legacy MODEL_DIR / "lid.176.bin"
    4. Repository default
    """
    if lid_model_path is not None:
        return Path(lid_model_path).expanduser()

    if env_path := os.getenv(LID_MODEL_PATH_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir() / "lid.176.bin"


def resolve_nemo_model_dir(nemo_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing local NeMo checkpoints.

    Resolution order:
    1. Explicit function argument
    2. NEMO_MODEL_DIR environment variable
    3. Legacy MODEL_DIR / "nemo"
    4. Repository default
    """
    if nemo_model_dir is not None:
        return Path(nemo_model_dir).expanduser()

    if env_path := os.getenv(NEMO_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir() / "nemo"


def resolve_hf_model_dir(hf_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing the HuggingFace model cache root.

    Resolution order:
    1. Explicit function argument
    2. HF_MODEL_DIR environment variable
    3. Legacy MODEL_DIR / "huggingface"
    4. Repository default
    """
    if hf_model_dir is not None:
        return Path(hf_model_dir).expanduser()

    if env_path := os.getenv(HF_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    return resolve_model_dir() / "huggingface"


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
    "MODEL_DIR_ENV_VAR",
    "LID_MODEL_PATH_ENV_VAR",
    "NEMO_MODEL_DIR_ENV_VAR",
    "HF_MODEL_DIR_ENV_VAR",
    "MODELS_DIR",
    "LID_MODEL_PATH",
    "NEMO_MODEL_DIR",
    "HF_MODEL_DIR",
    "resolve_model_dir",
    "resolve_lid_model_path",
    "resolve_nemo_model_dir",
    "resolve_hf_model_dir",
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
