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


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    3. Legacy MODEL_DIR / "fasttext" / "lid.176.bin"
    4. Legacy MODEL_DIR / "lid.176.bin"
    5. Repository default
    """
    if lid_model_path is not None:
        return Path(lid_model_path).expanduser()

    if env_path := os.getenv(LID_MODEL_PATH_ENV_VAR):
        return Path(env_path).expanduser()

    model_dir = resolve_model_dir()
    candidates = [
        model_dir / "fasttext" / "lid.176.bin",
        model_dir / "lid.176.bin",
    ]
    return _first_existing_path(candidates) or candidates[0]


def resolve_nemo_model_dir(nemo_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing local NeMo checkpoints.

    Resolution order:
    1. Explicit function argument
    2. NEMO_MODEL_DIR environment variable
    3. Legacy MODEL_DIR / "nemo"
    4. Legacy MODEL_DIR itself when it already contains NeMo checkpoints
    5. Repository default
    """
    if nemo_model_dir is not None:
        return Path(nemo_model_dir).expanduser()

    if env_path := os.getenv(NEMO_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    model_dir = resolve_model_dir()
    candidates = [
        model_dir / "nemo",
        model_dir,
    ]
    return _first_existing_path(candidates) or candidates[0]


def resolve_hf_model_dir(hf_model_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing the HuggingFace model cache root.

    Resolution order:
    1. Explicit function argument
    2. HF_MODEL_DIR environment variable
    3. Legacy MODEL_DIR / "huggingface"
    4. Legacy MODEL_DIR itself when it already contains HF model directories
    5. Repository default
    """
    if hf_model_dir is not None:
        return Path(hf_model_dir).expanduser()

    if env_path := os.getenv(HF_MODEL_DIR_ENV_VAR):
        return Path(env_path).expanduser()

    model_dir = resolve_model_dir()
    candidates = [
        model_dir / "huggingface",
        model_dir,
    ]
    return _first_existing_path(candidates) or candidates[0]


def resolve_model_reference(
    repo: str | Path,
    kind: str,
    nemo_model_dir: str | Path | None = None,
    hf_model_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a model reference to a local path when possible.

    The reference may already be an absolute/local path, or a short model name
    that should be looked up under the configured NeMo or HF roots.
    """
    candidate = Path(repo).expanduser()
    if candidate.exists():
        return candidate

    direct_name = candidate.name
    nemo_root = resolve_nemo_model_dir(nemo_model_dir)
    hf_root = resolve_hf_model_dir(hf_model_dir)
    search_roots = [hf_root, nemo_root]
    if kind in {"rnnt", "ctc"}:
        search_roots = [nemo_root, hf_root]

    for root in search_roots:
        options = [
            root / candidate,
            root / direct_name,
            (root / candidate).with_suffix(".nemo"),
            (root / direct_name).with_suffix(".nemo"),
        ]
        resolved = _first_existing_path(options)
        if resolved is not None:
            return resolved

    return candidate


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
    "resolve_model_reference",
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
