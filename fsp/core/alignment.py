"""
Forced alignment and ASR transcription functions.

This module contains alignment processing functions migrated from:
- steps/generate_final_data.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import torch
from loguru import logger

from fsp.core.text import clean_text
from fsp.utils.language import choose_language, choose_language_from_predictions, predict_languages
from fsp.utils.models import (
    configure_model_environment,
    find_local_nemo_checkpoint,
    load_model,
    suppress_nemo_restore_warnings,
    transcribe_model,
    transcribe_model_batch,
    unload_model,
)
from fsp.utils.paths import ALIGN_DIR, LOG_DIR, MANIFEST_DIR, NORM_DIR, OUTPUT_SEGMENT_DIR, ROOT

if TYPE_CHECKING:
    import fasttext


MODELS_BY_LANG: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "ca": (
        ("whisper_ca_3catparla", "pipe", "whisper-large-v3-ca-3catparla"),
        ("whisper_bsc_cat", "pipe", "whisper-bsc-large-v3-cat"),
        ("whisper_ca_punct_3370h", "pipe", "whisper-large-v3-ca-punctuated-3370h"),
        ("stt_ca_es_conformer_transducer_large", "rnnt", "stt_ca-es_conformer_transducer_large"),
    ),
    "es": (
        ("parakeet_rnnt_es", "rnnt", "parakeet-rnnt-1.1b_cv17_es_ep18_1270h"),
        ("stt_es_conformer_transducer_large", "rnnt", "stt_es_conformer_transducer_large"),
        ("whisper_large_v3", "pipe", "whisper-large-v3"),
    ),
    "eu": (
        ("stt_eu_conformer_transducer_large", "rnnt", "stt_eu_conformer_transducer_large"),
        ("stt_eu_conformer_ctc_large", "ctc", "stt_eu_conformer_ctc_large"),
        ("whisper_tiny_eu", "pipe", "whisper-tiny-eu"),
        ("whisper_small_eu", "pipe", "whisper-small-eu"),
        ("whisper_base_eu", "pipe", "whisper-base-eu"),
        ("whisper_medium_eu", "pipe", "whisper-medium-eu"),
        ("whisper_large_eu", "pipe", "whisper-large-eu"),
        ("whisper_large_v2_eu", "pipe", "whisper-large-v2-eu"),
        ("whisper_large_v3_eu", "pipe", "whisper-large-v3-eu"),
    ),
    "gl": (
        ("stt_gl_conformer_ctc_large", "ctc", "stt_gl_conformer_ctc_large"),
        ("whisper_large_v3_gl", "pipe", "whisper-large-v3-gl"),
    ),
}

CTC_MODELS = {
    "ca": "stt_ca_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
    "eu": "stt_eu_conformer_ctc_large",
    "gl": "stt_gl_conformer_ctc_large",
}

LEGACY_KEYS = {"pred_text", "cer_score"}
MIN_SEGMENT_LANGUAGE_CONFIDENCE = 0.35
DEFAULT_ASR_BATCH_SIZE = 8


def _format_language_predictions(predictions: List[Tuple[str, float]]) -> str:
    """Render FastText predictions for debug logging."""
    return ", ".join(f"{lang}={conf:.2f}" for lang, conf in predictions)


def _write_drop_log(drop_log_path: Path, dropped_segments: List[Dict[str, Any]]) -> None:
    """Persist dropped-segment records for the current run."""
    drop_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dropped_segments": dropped_segments,
    }
    drop_log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _clean_alignment_text(text: str, lang: str) -> str:
    """Normalize text for alignment while preserving segment separators."""
    parts = [part.strip() for part in text.split("|")]
    cleaned_parts = [clean_text(part, lang, False, False) for part in parts if part.strip()]
    return "|".join(part for part in cleaned_parts if part)


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _normalize_transcription(result: Dict[str, Any], norm_key: str, transcription: str, seg_lang: str, model_name: str) -> None:
    try:
        result[norm_key] = clean_text(transcription, seg_lang, False, False)
    except Exception as err:
        logging.warning("[norm_script:] %s on %s: %s", model_name, result["segment_path"], err)
        result[norm_key] = ""


def _populate_model_predictions(
    segs: List[Dict[str, Any]],
    model: Any,
    name: str,
    kind: str,
    model_name: str,
    seg_lang: str,
    batch_size: int = DEFAULT_ASR_BATCH_SIZE,
) -> None:
    pending: List[Dict[str, Any]] = []

    for result in segs:
        key = f"pred_text_{name}"
        norm_key = f"norm_text_{name}"
        transcription = result.get(key, "") or ""
        if transcription:
            if norm_key not in result:
                _normalize_transcription(result, norm_key, transcription, seg_lang, model_name)
            continue
        pending.append(result)

    for batch in _chunked(pending, batch_size):
        audio_paths = [str(result["segment_path"]) for result in batch]
        try:
            transcriptions = transcribe_model_batch(model, kind, audio_paths, lang=seg_lang, batch_size=batch_size)
            logger.info(
                "Batch ASR succeeded model={} kind={} batch_size={} inputs={} outputs={}",
                model_name,
                kind,
                batch_size,
                len(audio_paths),
                len(transcriptions),
            )
            for result, transcription in zip(batch, transcriptions, strict=True):
                key = f"pred_text_{name}"
                norm_key = f"norm_text_{name}"
                result[key] = transcription
                _normalize_transcription(result, norm_key, transcription, seg_lang, model_name)
        except Exception as err:
            logging.warning("%s batch on %s items failed: %s", model_name, len(batch), err)
            for result in batch:
                key = f"pred_text_{name}"
                norm_key = f"norm_text_{name}"
                try:
                    transcription = transcribe_model(model, kind, str(result["segment_path"]), lang=seg_lang)
                    result[key] = transcription
                except Exception as item_err:
                    logging.warning("%s on %s: %s", model_name, result["segment_path"], item_err)
                    result[key] = ""
                    transcription = ""
                _normalize_transcription(result, norm_key, transcription, seg_lang, model_name)


def hhmmss_to_sec(t: str | int | float) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return float(t)


def build_manifest(
    meta_path: Path,
    lid_model: "fasttext.FastText._FastText",
    pri_lang: str | None = None,
) -> Path:
    """Build a NeMo manifest from metadata."""
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

        detected_lang, conf = choose_language(src_norm, lid_model, pri_lang=pri_lang)
        cleaned_normalized = _clean_alignment_text(src_norm, detected_lang)

        entries.append(
            {
                "audio_filepath": str(wav.resolve()),
                "text": cleaned_normalized,
                "original_text": src_org,
                "language": f"{detected_lang}__{conf:.2f}",
            }
        )

    if not entries:
        raise RuntimeError(f"No valid blocks in {input_id}")

    with manifest_fp.open("w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    return manifest_fp


def run_forced_alignment(
    manifest_fp: Path,
    input_id: str,
    aligner_model_arg: str,
    align_device: str = "cpu",
) -> Path:
    """Run NeMo forced aligner."""
    input_align_dir = ALIGN_DIR / input_id
    input_align_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running NeMo forced alignment for '{}' with transcribe_device={} viterbi_device={}",
        input_id,
        align_device,
        align_device,
    )
    env = os.environ.copy()
    env["NEMO_SUPPRESS_SETUP_WARNINGS"] = "1"
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
            "hydra.job.chdir=False",
            "hydra.run.dir=.",
        ],
        env=env,
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
    min_language_confidence: float = MIN_SEGMENT_LANGUAGE_CONFIDENCE,
    asr_batch_size: int = DEFAULT_ASR_BATCH_SIZE,
) -> Path:
    """Generate word-level aligned JSON with ASR enrichment."""
    if asr_batch_size < 1:
        raise ValueError("asr_batch_size must be >= 1")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drop_log_path = LOG_DIR / "dropped_segments" / f"{input_id}_{lang}_{timestamp}.json"
    dropped_segments: List[Dict[str, Any]] = []

    resolved_lid_model = model_paths.lid_model_path
    if not resolved_lid_model.is_file():
        raise FileNotFoundError(f"language-ID model not found: {resolved_lid_model}")
    logger.info(f"Loading language-ID model: {resolved_lid_model}")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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

    primary_model_name = CTC_MODELS[lang]
    local_nemo = find_local_nemo_checkpoint(
        primary_model_name,
        nemo_model_dir=model_paths.nemo_model_dir,
    )

    if local_nemo is not None and local_nemo.is_file():
        logger.info(f"Loading local NeMo model: {local_nemo}")
        with suppress_nemo_restore_warnings():
            primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                str(local_nemo),
                map_location=resolved_device,
            )
        aligner_model_arg = f"model_path={local_nemo}"
    else:
        raise FileNotFoundError(
            f"Primary NeMo model not found locally under {model_paths.nemo_model_dir}: {primary_model_name}"
        )

    ca_asr = primary_asr if lang == "ca" else None
    es_asr = primary_asr if lang == "es" else None

    norm_root = NORM_DIR / input_id
    meta_path = norm_root / f"{input_id}_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    manifest_fp = build_manifest(meta_path, lid_model, pri_lang=lang)
    align_device = resolved_device if resolved_device == "cuda" else "cpu"
    input_align_dir = run_forced_alignment(
        manifest_fp,
        input_id,
        aligner_model_arg,
        align_device=align_device,
    )

    out_seg_dir = OUTPUT_SEGMENT_DIR / input_id
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {"ca": [], "es": [], "eu": [], "gl": []}
    combined: Dict[str, Any] = {}

    ctm_dir = input_align_dir / "ctm" / "segments"
    for ctm_file in ctm_dir.rglob("*.ctm"):
        block_id = ctm_file.stem
        wav_src = norm_root / f"{block_id}.wav"
        if not wav_src.is_file():
            dropped_segments.append(
                {
                    "stage": "alignment",
                    "block_id": block_id,
                    "reason": "missing_source_audio",
                    "expected_language": lang,
                    "source_audio_path": str(wav_src),
                }
            )
            continue

        seg = Segmenter(
            str(ctm_file),
            str(wav_src),
            str(out_seg_dir),
            lid_model,
            ca_asr,
            es_asr,
            drop_callback=dropped_segments.append,
        )
        results = seg.segment_audio()
        kept_results: List[Dict[str, Any]] = []

        for result in results:
            for key in LEGACY_KEYS:
                result.pop(key, None)

            predictions = predict_languages(result["normalized_text"], lid_model, k=3)
            detected_lang, conf = choose_language_from_predictions(predictions, pri_lang=lang)
            prediction_summary = _format_language_predictions(predictions)
            if detected_lang not in ("ca", "es", "eu", "gl"):
                dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "unsupported_language",
                        "expected_language": lang,
                        "detected_language": detected_lang,
                        "confidence": round(conf, 4),
                        "fasttext_predictions": predictions,
                    }
                )
                logger.warning(
                    "Dropping segment {} {:.2f}-{:.2f}: unsupported language={} expected={} fasttext=[{}] | text={!r}",
                    block_id,
                    result["start"],
                    result["end"],
                    detected_lang,
                    lang,
                    prediction_summary,
                    result["normalized_text"],
                )
                continue
            if detected_lang != lang:
                dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "language_mismatch",
                        "expected_language": lang,
                        "detected_language": detected_lang,
                        "confidence": round(conf, 4),
                        "fasttext_predictions": predictions,
                    }
                )
                logger.warning(
                    "Dropping segment {} {:.2f}-{:.2f}: language mismatch detected={} conf={:.2f} expected={} fasttext=[{}] | text={!r}",
                    block_id,
                    result["start"],
                    result["end"],
                    detected_lang,
                    conf,
                    lang,
                    prediction_summary,
                    result["normalized_text"],
                )
                continue
            if conf < min_language_confidence:
                dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "language_confidence_below_threshold",
                        "expected_language": lang,
                        "detected_language": detected_lang,
                        "confidence": round(conf, 4),
                        "minimum_confidence": min_language_confidence,
                        "fasttext_predictions": predictions,
                    }
                )
                logger.warning(
                    "Dropping segment {} {:.2f}-{:.2f}: low language confidence detected={} conf={:.2f} threshold={:.2f} expected={} fasttext=[{}] | text={!r}",
                    block_id,
                    result["start"],
                    result["end"],
                    detected_lang,
                    conf,
                    min_language_confidence,
                    lang,
                    prediction_summary,
                    result["normalized_text"],
                )
                continue
            result["language"] = detected_lang
            result["language_confidence"] = round(conf, 2)
            if detected_lang in ("ca", "es") and "pred_text" in result:
                result["pred_text_segmenter"] = result.pop("pred_text")
                result.pop("cer_score", None)
            else:
                result.pop("pred_text", None)
                result.pop("cer_score", None)
            buckets[detected_lang].append(result)
            kept_results.append(result)

        combined[block_id] = {
            "input_id": input_id,
            "results": kept_results,
        }

    for seg_lang, segs in buckets.items():
        logger.info(f"\nProcessing language: {seg_lang}, number of segments: {len(segs)}")
        if not segs:
            continue
        for name, kind, model_name in MODELS_BY_LANG[seg_lang]:
            model = None
            try:
                logging.info("loading %s", model_name)
                model = load_model(
                    kind,
                    model_name,
                    resolved_device,
                    nemo_model_dir=model_paths.nemo_model_dir,
                    hf_model_dir=model_paths.hf_model_dir,
                )
                logger.info(f"Model {name} loaded on device {resolved_device}")
                _populate_model_predictions(
                    segs,
                    model,
                    name,
                    kind,
                    model_name,
                    seg_lang,
                    batch_size=asr_batch_size,
                )
            except Exception as err:
                logging.warning("Could not load model %s (%s): %s - skipping", name, model_name, err)
                logger.warning(f"Skipping model {name}: {err}")
            finally:
                unload_model(model)

    final_fp = OUTPUT_SEGMENT_DIR / output_name
    final_fp.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_drop_log(drop_log_path, dropped_segments)
    logger.info("Dropped-segment log written to {}", drop_log_path)
    logger.info(f"JSON written to {final_fp} ({time.perf_counter() - start:.1f}s)")

    return final_fp


__all__ = [
    "MODELS_BY_LANG",
    "CTC_MODELS",
    "hhmmss_to_sec",
    "build_manifest",
    "run_forced_alignment",
    "generate_final_data",
]
