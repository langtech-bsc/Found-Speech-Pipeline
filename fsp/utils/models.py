"""
Model loading and unloading utilities.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Union

import torch
from loguru import logger

from fsp.utils.paths import (
    HF_MODEL_DIR_ENV_VAR,
    LID_MODEL_PATH_ENV_VAR,
    NEMO_MODEL_DIR_ENV_VAR,
    ModelPaths,
    resolve_model_paths,
)


def configure_model_environment(
    lid_model_path: Union[str, Path, None] = None,
    nemo_model_dir: Union[str, Path, None] = None,
    hf_model_dir: Union[str, Path, None] = None,
) -> ModelPaths:
    """
    Configure runtime model paths and cache locations.
    """
    model_paths = resolve_model_paths(
        lid_model_path=lid_model_path,
        nemo_model_dir=nemo_model_dir,
        hf_model_dir=hf_model_dir,
    )
    hf_home = model_paths.hf_model_dir
    hf_hub_cache = hf_home / "hub"
    nemo_cache_dir = model_paths.nemo_model_dir

    os.environ[LID_MODEL_PATH_ENV_VAR] = str(model_paths.lid_model_path)
    os.environ[NEMO_MODEL_DIR_ENV_VAR] = str(model_paths.nemo_model_dir)
    os.environ[HF_MODEL_DIR_ENV_VAR] = str(model_paths.hf_model_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_hub_cache)
    os.environ["NEMO_CACHE_DIR"] = str(nemo_cache_dir)

    return model_paths


def load_model(
    kind: str,
    repo: str,
    device: str,
    nemo_model_dir: Union[str, Path, None] = None,
    hf_model_dir: Union[str, Path, None] = None,
) -> Any:
    """
    Load one ASR model (no global cache so RSS stays small).

    Args:
        kind: Model type ('pipe', 'rnnt', or 'multi')
        repo: Model repository/name
        device: Device to load model on ('cuda' or 'cpu')
        nemo_model_dir: Directory containing local NeMo checkpoints
        hf_model_dir: Directory containing the HuggingFace cache root

    Returns:
        Loaded model object
    """
    configure_model_environment(nemo_model_dir=nemo_model_dir, hf_model_dir=hf_model_dir)

    if kind == "pipe":
        from transformers import pipeline as hf_pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return hf_pipeline(
            "automatic-speech-recognition",
            model=repo,
            device=-1 if device == "cpu" else 0,
            torch_dtype=dtype,
        )
    if kind == "rnnt":
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

        return EncDecRNNTBPEModel.from_pretrained(repo, map_location=device).to(device).eval()
    if kind == "multi":
        from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

        m = EncDecMultiTaskModel.from_pretrained(repo, map_location=device).to(device).eval()
        m.cfg.prompt_format = m.prompt_format = "canary"
        m.cfg.decoding.beam.beam_size = 1
        m.change_decoding_strategy(m.cfg.decoding)
        return m
    raise ValueError(f"Unknown model kind: {kind}")


def unload_model(model: Any) -> None:
    """
    Free CPU/GPU RAM once a model is no longer needed.

    Args:
        model: Model to unload
    """
    try:
        del model
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def download_nemo_ctc(model_name: str, out_dir: Path) -> None:
    """
    Download a NeMo CTC model from NGC and save the .nemo file locally.

    Args:
        model_name: Name of the NeMo model
        out_dir: Output directory for the .nemo file
    """
    out_path = out_dir / f"{model_name}.nemo"
    if out_path.exists():
        logger.info(f"  Already exists: {out_path}")
        return

    import nemo.collections.asr as nemo_asr

    logger.info(f"  Downloading NeMo model: {model_name} ...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)

    # Save to our local directory
    model.save_to(str(out_path))
    logger.info(f"  Saved to: {out_path}")

    # Clean up RAM
    unload_model(model)


def download_hf_model(repo_id: str, hf_home: Path) -> None:
    """
    Download a HuggingFace model snapshot.

    Args:
        repo_id: HuggingFace repository ID
        hf_home: HuggingFace home directory
    """
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")

    from huggingface_hub import snapshot_download

    cache_name = repo_id.replace("/", "--")
    model_dir = hf_home / "hub" / f"models--{cache_name}"
    if model_dir.exists():
        logger.info(f"  Already cached: {repo_id}")
        return

    logger.info(f"  Downloading HuggingFace model: {repo_id} ...")
    snapshot_download(repo_id)
    logger.info(f"  Cached: {repo_id}")


__all__ = [
    "configure_model_environment",
    "load_model",
    "unload_model",
    "download_nemo_ctc",
    "download_hf_model",
]
