"""
Forced alignment and ASR transcription functions.

This module contains alignment processing functions migrated from:
- steps/generate_final_data.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from loguru import logger

from fsp.core.text import clean_text
from fsp.utils.language import choose_language
from fsp.utils.models import (
    configure_model_environment,
    find_local_nemo_checkpoint,
    load_model,
    log_loaded_model_device,
    unload_model,
)
from fsp.utils.paths import ALIGN_DIR, MANIFEST_DIR, NORM_DIR, OUTPUT_SEGMENT_DIR, ROOT

if TYPE_CHECKING:
    import fasttext


# Model catalogue
MODELS_BY_LANG: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ca": (
        ("whisper-large-v3-ca-3catparla", "pipe"),
        ("Whisper-bsc-large-v3-cat", "pipe"),
        ("whisper-large-v3-ca-punctuated-3370h", "pipe"),
        ("stt_ca-es_conformer_transducer_large", "rnnt"),
    ),
    "es": (
        ("parakeet-rnnt-1.1b_cv17_es_ep18_1270h", "rnnt"),
        ("stt_es_conformer_transducer_large", "rnnt"),
        ("whisper-large-v3", "pipe"),
    ),
}

CTC_MODELS = {
    "ca": "stt_ca_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
}

LEGACY_KEYS = {"pred_text", "cer_score"}


def _clean_alignment_text(text: str, lang: str) -> str:
    """Normalize text for alignment while preserving segment separators."""
    parts = [part.strip() for part in text.split("|")]
    cleaned_parts = [clean_text(part, lang, False, False) for part in parts if part.strip()]
    return "|".join(part for part in cleaned_parts if part)


def _clean_singleton_json_array(txt: str) -> str:
    """
    Some NeMo RNN-T checkpoints return their transcript wrapped like
    '[\"text\"]' (or "['text']") – i.e. a JSON list encoded as a string.
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


def transcribe(model: Any, kind: str, audio: str) -> str:
    """
    Run model on audio and normalize the string it returns.

    Args:
        model: Loaded ASR model
        kind: Model type ('pipe', 'rnnt', 'multi')
        audio: Path to audio file

    Returns:
        Transcribed text
    """
    if kind == "pipe":
        out = model(audio)
        if isinstance(out, dict):
            txt = out.get("text", "")
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            txt = " ".join(d.get("text", "") for d in out)
        else:
            txt = str(out)
    else:  # rnnt / multi
        out = model.transcribe([audio], batch_size=1)[0]
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))

    return _clean_singleton_json_array(txt)


def hhmmss_to_sec(t: str | int | float) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return float(t)


def build_manifest(meta_path: Path, lid_model: "fasttext.FastText._FastText") -> Path:
    """
    Build a NeMo manifest from metadata.

    Args:
        meta_path: Path to metadata JSON file
        lid_model: FastText language identification model

    Returns:
        Path to the generated manifest file
    """
    input_id = meta_path.stem.replace("_metadata", "")
    mdir = MANIFEST_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    manifest_fp = mdir / f"{input_id}_manifest.json"

    entries: List[Dict[str, str | float]] = []
    blocks = json.loads(meta_path.read_text(encoding="utf-8"))

    for blk in blocks:
        wav = Path(blk["audio_filepath"]).expanduser()
        if not wav.is_file():
            continue
        if hhmmss_to_sec(blk["start"]) >= hhmmss_to_sec(blk["end"]):
            continue

        if isinstance(blk["normalized_text"], str):
            src_norm = blk["normalized_text"]
        else:
            src_norm = " ".join(w for _sp, w in blk["normalized_text"])
        if not src_norm.strip():
            continue

        if isinstance(blk["original_text"], str):
            src_org = blk["original_text"]
        else:
            src_org = " ".join(w for _sp, w in blk["original_text"])
        if not src_org.strip():
            continue

        lang, conf = choose_language(src_norm, lid_model)
        cleaned_normalized = _clean_alignment_text(src_norm, lang)

        entries.append(
            {
                "audio_filepath": str(wav.resolve()),
                "text": cleaned_normalized,
                "original_text": src_org,
                "language": f"{lang}__{conf:.2f}",
            }
        )

    if not entries:
        raise RuntimeError(f"No valid blocks in {input_id}")

    with manifest_fp.open("w", encoding="utf-8") as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")

    return manifest_fp


def run_forced_alignment(
    manifest_fp: Path,
    input_id: str,
    model_name: str,
    local_nemo_path: Path | None = None,
    align_device: str = "cpu",
) -> Path:
    """
    Run NeMo forced aligner.

    Args:
        manifest_fp: Path to manifest file
        input_id: Input identifier
        model_name: Name of the NeMo model
        local_nemo_path: Path to local .nemo file (optional)
        align_device: Device for NeMo alignment transcription and Viterbi

    Returns:
        Path to alignment output directory
    """
    input_align_dir = ALIGN_DIR / input_id
    input_align_dir.mkdir(parents=True, exist_ok=True)

    if local_nemo_path and local_nemo_path.is_file():
        aligner_model_arg = f"model_path={local_nemo_path}"
    else:
        aligner_model_arg = f"pretrained_name={model_name}"

    logger.info(
        "Running NeMo forced alignment for '{}' with transcribe_device={} viterbi_device={}",
        input_id,
        align_device,
        align_device,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py"),
            aligner_model_arg,
            f"manifest_filepath={manifest_fp}",
            f"output_dir={input_align_dir}",
            "align_using_pred_text=false",
            f"transcribe_device={align_device}",
            f"viterbi_device={align_device}",
            "additional_segment_grouping_separator=|",
            "hydra.run.dir=.",
        ],
        check=True,
    )

    return input_align_dir


def generate_final_data(
    input_id: str,
    lang: str = "ca",
    output_name: str | None = None,
    device: str = "auto",
    lid_model_path: str | Path | None = None,
    nemo_model_dir: str | Path | None = None,
    hf_model_dir: str | Path | None = None,
) -> Path:
    """
    Main function to generate word-level aligned JSON with ASR enrichment.

    Args:
        input_id: Input identifier
        lang: Primary language ('ca' or 'es')
        output_name: Custom output JSON name
        device: Device for ASR ('auto', 'cuda', 'cpu')
        lid_model_path: Path to the FastText language-ID model file
        nemo_model_dir: Directory containing local NeMo checkpoints
        hf_model_dir: Directory containing the HuggingFace cache root

    Returns:
        Path to the output JSON file
    """
    model_paths = configure_model_environment(
        lid_model_path=lid_model_path,
        nemo_model_dir=nemo_model_dir,
        hf_model_dir=hf_model_dir,
    )

    import fasttext
    import nemo.collections.asr as nemo_asr

    from fsp.core.segmenter import Segmenter

    start = time.perf_counter()
    output_name = output_name or f"final_output_{input_id}.json"

    resolved_lid_model = model_paths.lid_model_path
    if not resolved_lid_model.is_file():
        raise FileNotFoundError(f"language-ID model not found: {resolved_lid_model}")
    logger.info(f"Loading language-ID model: {resolved_lid_model}")
    lid_model = fasttext.load_model(str(resolved_lid_model))

    resolved_device = (
        "cpu"
        if device == "cpu"
        else (
            "cuda"
            if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
            else "cpu"
        )
    )

    # Load CTC model
    primary_model_name = CTC_MODELS[lang]
    local_nemo = find_local_nemo_checkpoint(
        primary_model_name,
        nemo_model_dir=model_paths.nemo_model_dir,
    )

    if local_nemo is not None and local_nemo.is_file():
        logger.info(
            "Loading primary NeMo CTC model '{}' from {} on requested device {}",
            primary_model_name,
            local_nemo,
            resolved_device,
        )
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            str(local_nemo),
            map_location=resolved_device,
        ).to(resolved_device).eval()
        log_loaded_model_device(primary_model_name, primary_asr, requested_device=resolved_device)
    else:
        raise FileNotFoundError(
            f"Primary NeMo model not found locally under {model_paths.nemo_model_dir}: {primary_model_name}"
        )

    ca_asr = primary_asr if lang == "ca" else None
    es_asr = primary_asr if lang == "es" else None

    # Paths
    norm_root = NORM_DIR / input_id
    meta_path = norm_root / f"{input_id}_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    # 1. Forced alignment
    manifest_fp = build_manifest(meta_path, lid_model)
    input_align_dir = run_forced_alignment(
        manifest_fp,
        input_id,
        primary_model_name,
        local_nemo_path=local_nemo,
        align_device=resolved_device,
    )

    # 2. Segment & language-tag
    out_seg_dir = OUTPUT_SEGMENT_DIR / input_id
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {"ca": [], "es": []}
    combined: Dict[str, Any] = {}

    ctm_dir = input_align_dir / "ctm" / "segments"
    for ctm_file in ctm_dir.rglob("*.ctm"):
        block_id = ctm_file.stem
        wav_src = norm_root / f"{block_id}.wav"
        if not wav_src.is_file():
            continue

        seg = Segmenter(str(ctm_file), str(wav_src), str(out_seg_dir), lid_model, ca_asr, es_asr)
        results = seg.segment_audio()

        for r in results:
            for k in LEGACY_KEYS:
                r.pop(k, None)

            detected_lang, conf = choose_language(r["normalized_text"], lid_model)
            if detected_lang not in ("ca", "es"):
                continue
            r["language"] = detected_lang
            r["language_confidence"] = round(conf, 2)
            buckets[detected_lang].append(r)

        combined[block_id] = {
            "input_id": input_id,
            "results": results,
        }

    # 3. ASR per-language, one model at a time
    for seg_lang, segs in buckets.items():
        logger.info(f"\nProcessing language: {seg_lang}, number of segments: {len(segs)}")
        if not segs:
            continue
        for model_dir_name, kind in MODELS_BY_LANG[seg_lang]:
            model = None
            try:
                logging.info("loading %s", model_dir_name)
                model = load_model(
                    kind,
                    model_dir_name,
                    resolved_device,
                    nemo_model_dir=model_paths.nemo_model_dir,
                    hf_model_dir=model_paths.hf_model_dir,
                )
                for r in segs:
                    key = f"pred_text_{model_dir_name}"
                    norm_key = f"norm_text_{model_dir_name}"
                    transcription = ""
                    if r.get(key):
                        continue
                    try:
                        transcription = transcribe(model, kind, r["segment_path"])
                        r[key] = transcription
                    except Exception as err:
                        logging.warning("%s on %s: %s", model_dir_name, r["segment_path"], err)
                        r[key] = ""
                    try:
                        r[norm_key] = clean_text(transcription, seg_lang, False, False)
                    except Exception as err:
                        logging.warning("[norm_script:] %s on %s: %s", model_dir_name, r["segment_path"], err)
                        r[norm_key] = ""
            except Exception as err:
                logging.warning("Could not load model %s: %s - skipping", model_dir_name, err)
                logger.warning(f"Skipping model {model_dir_name}: {err}")
            finally:
                unload_model(model)

    # 4. Write JSON
    final_fp = OUTPUT_SEGMENT_DIR / output_name
    final_fp.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"JSON written to {final_fp} ({time.perf_counter() - start:.1f}s)")

    return final_fp


__all__ = [
    "MODELS_BY_LANG",
    "CTC_MODELS",
    "transcribe",
    "hhmmss_to_sec",
    "build_manifest",
    "run_forced_alignment",
    "generate_final_data",
]
