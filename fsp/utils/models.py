"""
Model loading and unloading utilities.
"""

from __future__ import annotations

import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Union

import torch
from loguru import logger

from fsp.core.text import clean_apostrophes
from fsp.utils.paths import (
    HF_MODEL_DIR_ENV_VAR,
    LID_MODEL_PATH_ENV_VAR,
    MODEL_DIR_ENV_VAR,
    MODELS_ROOT_ENV_VAR,
    NEMO_MODEL_DIR_ENV_VAR,
    ModelPaths,
    resolve_hf_model_dir,
    resolve_model_dir,
    resolve_nemo_model_dir,
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
    model_root = resolve_model_dir()
    hf_home = model_paths.hf_model_dir
    hf_hub_cache = hf_home / "hub"
    nemo_cache_dir = model_paths.nemo_model_dir

    os.environ[MODELS_ROOT_ENV_VAR] = str(model_root)
    os.environ[LID_MODEL_PATH_ENV_VAR] = str(model_paths.lid_model_path)
    os.environ[MODEL_DIR_ENV_VAR] = str(model_root)
    os.environ[NEMO_MODEL_DIR_ENV_VAR] = str(model_paths.nemo_model_dir)
    os.environ[HF_MODEL_DIR_ENV_VAR] = str(model_paths.hf_model_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["NEMO_CACHE_DIR"] = str(nemo_cache_dir)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return model_paths


@contextmanager
def suppress_nemo_restore_warnings():
    """
    Temporarily lower NeMo logging noise during model restore.
    """
    try:
        from nemo.utils import logging as nemo_logging
    except Exception:
        yield
        return

    original = nemo_logging.get_verbosity()
    try:
        nemo_logging.set_verbosity(nemo_logging.ERROR)
        yield
    finally:
        nemo_logging.set_verbosity(original)


def find_local_nemo_checkpoint(
    model_name: str,
    nemo_model_dir: Union[str, Path, None] = None,
) -> Path | None:
    """Resolve a local NeMo checkpoint without remote fallback."""
    nemo_root = resolve_nemo_model_dir(nemo_model_dir)
    nested_checkpoint = nemo_root / model_name / f"{model_name}.nemo"
    direct_checkpoint = nemo_root / f"{model_name}.nemo"

    if nested_checkpoint.is_file():
        return nested_checkpoint
    if direct_checkpoint.is_file():
        return direct_checkpoint

    return None


def load_model(
    kind: str,
    model_name: str,
    device: str,
    nemo_model_dir: Union[str, Path, None] = None,
    hf_model_dir: Union[str, Path, None] = None,
) -> Any:
    """
    Load one ASR model (no global cache so RSS stays small).

    Args:
        kind: Model type ('pipe', 'rnnt', 'ctc', or 'multi')
        model_name: Explicit local model folder name
        device: Device to load model on ('cuda' or 'cpu')
        nemo_model_dir: Directory containing local NeMo checkpoints
        hf_model_dir: Directory containing local HuggingFace model folders

    Returns:
        Loaded model object
    """
    configure_model_environment(nemo_model_dir=nemo_model_dir, hf_model_dir=hf_model_dir)

    if kind == "pipe":
        from transformers import pipeline as hf_pipeline
        from transformers.utils import logging as transformers_logging

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        local_model_dir = resolve_hf_model_dir(hf_model_dir) / model_name
        if not local_model_dir.is_dir():
            raise FileNotFoundError(f"HF model directory not found: {local_model_dir}")
        previous_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            return hf_pipeline(
                "automatic-speech-recognition",
                model=str(local_model_dir),
                tokenizer=str(local_model_dir),
                feature_extractor=str(local_model_dir),
                device=-1 if device == "cpu" else 0,
                torch_dtype=dtype,
            )
        finally:
            transformers_logging.set_verbosity(previous_verbosity)
    if kind == "rnnt":
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

        restore_path = find_local_nemo_checkpoint(model_name, nemo_model_dir=nemo_model_dir)
        if restore_path is None:
            raise FileNotFoundError(
                f"NeMo checkpoint not found: {resolve_nemo_model_dir(nemo_model_dir) / model_name / f'{model_name}.nemo'}"
            )
        logger.info("Loading NeMo RNNT model '{}' from {} on requested device {}", model_name, restore_path, device)
        with suppress_nemo_restore_warnings():
            model = EncDecRNNTBPEModel.restore_from(str(restore_path), map_location=device).to(device).eval()
        log_loaded_model_device(model_name, model, requested_device=device)
        return model
    if kind == "ctc":
        import nemo.collections.asr as nemo_asr

        restore_path = find_local_nemo_checkpoint(model_name, nemo_model_dir=nemo_model_dir)
        if restore_path is None:
            raise FileNotFoundError(
                f"NeMo checkpoint not found: {resolve_nemo_model_dir(nemo_model_dir) / model_name / f'{model_name}.nemo'}"
            )
        logger.info("Loading NeMo CTC model '{}' from {} on requested device {}", model_name, restore_path, device)
        with suppress_nemo_restore_warnings():
            model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(restore_path), map_location=device).to(device).eval()
        log_loaded_model_device(model_name, model, requested_device=device)
        return model
    if kind == "multi":
        from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

        restore_path = find_local_nemo_checkpoint(model_name, nemo_model_dir=nemo_model_dir)
        if restore_path is None:
            raise FileNotFoundError(
                f"NeMo checkpoint not found: {resolve_nemo_model_dir(nemo_model_dir) / model_name / f'{model_name}.nemo'}"
            )
        logger.info("Loading NeMo multitask model '{}' from {} on requested device {}", model_name, restore_path, device)
        with suppress_nemo_restore_warnings():
            model = EncDecMultiTaskModel.restore_from(str(restore_path), map_location=device).to(device).eval()
        model.cfg.prompt_format = model.prompt_format = "canary"
        model.cfg.decoding.beam.beam_size = 1
        model.change_decoding_strategy(model.cfg.decoding)
        log_loaded_model_device(model_name, model, requested_device=device)
        return model
    raise ValueError(f"Unknown model kind: {kind}")


def get_model_device(model: Any) -> str:
    """
    Best-effort device inspection for loaded ASR models.
    """
    torch_model = getattr(model, "model", model)
    if hasattr(torch_model, "device"):
        return str(torch_model.device)
    if hasattr(torch_model, "parameters"):
        try:
            return str(next(torch_model.parameters()).device)
        except (StopIteration, TypeError):
            pass
    return "unknown"


def _clean_singleton_json_array(txt: str) -> str:
    """
    Some NeMo checkpoints return a one-item JSON array encoded as a string.
    Detect and unwrap that safely.
    """
    s = txt.strip()
    if not (s.startswith('["') or s.startswith("['")):
        return s
    try:
        parsed = json.loads(s.replace("'", '"'))
        if isinstance(parsed, list) and len(parsed) == 1:
            return str(parsed[0]).strip()
    except Exception:
        pass
    return s[2:-2].strip()


def _extract_transcription_text(out: Any) -> str:
    if isinstance(out, dict):
        txt = out.get("text", "")
    elif isinstance(out, list) and out and isinstance(out[0], dict):
        txt = " ".join(d.get("text", "") for d in out)
    else:
        if out is None:
            txt = ""
        else:
            txt = getattr(out, "text", out)
            if txt is None:
                txt = ""
            elif not isinstance(txt, str):
                txt = str(txt)
    return clean_apostrophes(_clean_singleton_json_array(txt))


def transcribe_model(model: Any, kind: str, audio: str, lang: str = "ca") -> str:
    """Run one audio file through a loaded ASR model and normalize the output."""
    if kind == "pipe":
        out = model(audio, generate_kwargs={"task": "transcribe", "language": lang})
        return _extract_transcription_text(out)

    out = model.transcribe([audio], batch_size=1, verbose=False)
    if kind == "rnnt" and isinstance(out, tuple):
        out = out[0]
    if len(out) != 1:
        raise ValueError(f"Expected 1 transcript from {kind} model, got {len(out)}")
    return _extract_transcription_text(out[0])


def transcribe_model_batch(
    model: Any,
    kind: str,
    audio_paths: List[str],
    lang: str = "ca",
    batch_size: int = 8,
) -> List[str]:
    """Run a batch of audio files through a loaded ASR model and normalize the outputs."""
    if kind == "pipe":
        outputs = model(
            audio_paths,
            batch_size=batch_size,
            generate_kwargs={"task": "transcribe", "language": lang},
        )
        return [_extract_transcription_text(out) for out in outputs]

    outputs = model.transcribe(audio_paths, batch_size=batch_size, verbose=False)
    if kind == "rnnt" and isinstance(outputs, tuple):
        outputs = outputs[0]
    if len(outputs) != len(audio_paths):
        raise ValueError(f"Expected {len(audio_paths)} transcripts from {kind} model, got {len(outputs)}")
    return [_extract_transcription_text(out) for out in outputs]


def log_loaded_model_device(model_name: str, model: Any, requested_device: str) -> None:
    """
    Emit a consistent post-load device log for every ASR model.
    """
    actual_device = get_model_device(model)
    logger.info(
        "Model '{}' ready. requested_device={} actual_device={} cuda_available={}",
        model_name,
        requested_device,
        actual_device,
        torch.cuda.is_available(),
    )


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
    model.save_to(str(out_path))
    logger.info(f"  Saved to: {out_path}")
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
    os.environ.pop("TRANSFORMERS_CACHE", None)

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
    "find_local_nemo_checkpoint",
    "suppress_nemo_restore_warnings",
    "transcribe_model",
    "transcribe_model_batch",
]
