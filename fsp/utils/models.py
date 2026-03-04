"""
Model loading and unloading utilities.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Union

import torch


def load_model(kind: str, repo: str, device: str) -> Any:
    """
    Load one ASR model (no global cache so RSS stays small).

    Args:
        kind: Model type ('pipe', 'rnnt', or 'multi')
        repo: Model repository/name
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model object
    """
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
        print(f"  ✓ Already exists: {out_path}")
        return

    import nemo.collections.asr as nemo_asr

    print(f"  ↓ Downloading NeMo model: {model_name} ...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)

    # Save to our local directory
    model.save_to(str(out_path))
    print(f"  ✓ Saved to: {out_path}")

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

    from huggingface_hub import snapshot_download

    cache_name = repo_id.replace("/", "--")
    model_dir = hf_home / "hub" / f"models--{cache_name}"
    if model_dir.exists():
        print(f"  ✓ Already cached: {repo_id}")
        return

    print(f"  ↓ Downloading HuggingFace model: {repo_id} ...")
    snapshot_download(repo_id)
    print(f"  ✓ Cached: {repo_id}")


__all__ = ["load_model", "unload_model", "download_nemo_ctc", "download_hf_model"]
