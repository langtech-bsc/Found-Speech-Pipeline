"""
Forced alignment and ASR transcription functions.

This module contains alignment processing functions migrated from:
- steps/generate_final_data.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple

import torch
from loguru import logger

from fsp.core.text import clean_text
from fsp.utils.logging import sanitize_captured_output, write_captured_output
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
MIN_SEGMENT_LANGUAGE_CONFIDENCE = 0.30
DEFAULT_ASR_BATCH_SIZE = 8
SUPPORTED_SEGMENT_LANGS = ("ca", "es", "eu", "gl")


@dataclass
class BatchItemState:
    input_id: str
    lang: str
    output_name: str
    final_fp: Path
    drop_log_path: Path
    run_log_dir: Path | None
    norm_root: Path
    out_seg_dir: Path
    combined: Dict[str, Any] = field(default_factory=dict)
    dropped_segments: List[Dict[str, Any]] = field(default_factory=list)
    buckets: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {seg_lang: [] for seg_lang in SUPPORTED_SEGMENT_LANGS}
    )
    started_at: float = field(default_factory=time.perf_counter)


def _format_language_predictions(predictions: List[Tuple[str, float]]) -> str:
    """Render FastText predictions for debug logging."""
    return ", ".join(f"{lang}={conf:.2f}" for lang, conf in predictions)


def _write_json_atomic(path: Path, payload: Any, *, ensure_ascii: bool, add_trailing_newline: bool = False) -> None:
    """Write JSON atomically so partial progress remains readable if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=ensure_ascii)
    if add_trailing_newline:
        text += "\n"
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _write_drop_log(drop_log_path: Path, dropped_segments: List[Dict[str, Any]]) -> None:
    """Persist dropped-segment records for the current run."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dropped_segments": dropped_segments,
    }
    _write_json_atomic(drop_log_path, payload, ensure_ascii=True, add_trailing_newline=True)


def _resolve_device(device: str) -> str:
    return (
        "cpu"
        if device == "cpu"
        else ("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu")
    )


def _build_batch_item_state(
    input_id: str,
    lang: str,
    *,
    output_name: str | None = None,
    run_log_dir: Path | None = None,
) -> BatchItemState:
    output_name = output_name or f"final_output_{input_id}.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BatchItemState(
        input_id=input_id,
        lang=lang,
        output_name=output_name,
        final_fp=OUTPUT_SEGMENT_DIR / output_name,
        drop_log_path=LOG_DIR / "dropped_segments" / f"{input_id}_{lang}_{timestamp}.json",
        run_log_dir=run_log_dir,
        norm_root=NORM_DIR / input_id,
        out_seg_dir=OUTPUT_SEGMENT_DIR / input_id,
    )


def _persist_batch_item(state: BatchItemState, *, write_drop_log: bool = False) -> None:
    _write_json_atomic(state.final_fp, state.combined, ensure_ascii=False)
    logger.info("JSON written to {} elapsed_s={:.1f}", state.final_fp, time.perf_counter() - state.started_at)
    if write_drop_log:
        _write_drop_log(state.drop_log_path, state.dropped_segments)
        dropped_by_reason = dict(sorted(Counter(item.get("reason", "unknown") for item in state.dropped_segments).items()))
        logger.info(
            "Dropped-segment log written to {} dropped_segments={} dropped_reasons={}",
            state.drop_log_path,
            len(state.dropped_segments),
            dropped_by_reason,
        )


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
        logger.warning("[norm_script] model={} segment_path={} error={}", model_name, result["segment_path"], err)
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
            logger.warning("Batch ASR failed model={} batch_items={} error={}", model_name, len(batch), err)
            for result in batch:
                key = f"pred_text_{name}"
                norm_key = f"norm_text_{name}"
                try:
                    transcription = transcribe_model(model, kind, str(result["segment_path"]), lang=seg_lang)
                    result[key] = transcription
                except Exception as item_err:
                    logger.warning(
                        "Single-item ASR failed model={} segment_path={} error={}",
                        model_name,
                        result["segment_path"],
                        item_err,
                    )
                    result[key] = ""
                    transcription = ""
                _normalize_transcription(result, norm_key, transcription, seg_lang, model_name)


def _load_lid_model(lid_model_path: Path) -> "fasttext.FastText._FastText":
    import fasttext

    if not lid_model_path.is_file():
        raise FileNotFoundError(f"language-ID model not found: {lid_model_path}")
    logger.info("Loading language-ID model: {}", lid_model_path)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fasttext.load_model(str(lid_model_path))


def _load_primary_ctc_model(lang: str, resolved_device: str, nemo_model_dir: Path) -> tuple[Any, str]:
    import nemo.collections.asr as nemo_asr

    primary_model_name = CTC_MODELS[lang]
    local_nemo = find_local_nemo_checkpoint(primary_model_name, nemo_model_dir=nemo_model_dir)
    if local_nemo is None or not local_nemo.is_file():
        raise FileNotFoundError(
            f"Primary NeMo model not found locally under {nemo_model_dir}: {primary_model_name}"
        )
    logger.info("Loading local NeMo model: {}", local_nemo)
    with suppress_nemo_restore_warnings():
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            str(local_nemo),
            map_location=resolved_device,
        )
    return primary_asr, f"model_path={local_nemo}"


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
    run_log_dir: Path | None = None,
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
    completed = subprocess.run(
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
        capture_output=True,
        text=True,
    )
    if run_log_dir is not None:
        write_captured_output(
            run_log_dir / "forced_alignment.log",
            (
                ("stdout", completed.stdout),
                ("stderr", completed.stderr),
            ),
        )
    if completed.returncode != 0:
        stderr = sanitize_captured_output(completed.stderr).strip()
        if stderr:
            logger.error("Forced alignment stderr:\n{}", stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    logger.info("Forced alignment completed input_id={} log_file={}", input_id, run_log_dir / "forced_alignment.log" if run_log_dir else "<not_saved>")

    return input_align_dir


def _segment_and_bucket_input(
    state: BatchItemState,
    *,
    lid_model: "fasttext.FastText._FastText",
    primary_asr: Any,
    aligner_model_arg: str,
    align_device: str,
    min_language_confidence: float,
) -> None:
    from fsp.core.segmenter import Segmenter

    ca_asr = primary_asr if state.lang == "ca" else None
    es_asr = primary_asr if state.lang == "es" else None

    meta_path = state.norm_root / f"{state.input_id}_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    manifest_fp = build_manifest(meta_path, lid_model, pri_lang=state.lang)
    input_align_dir = run_forced_alignment(
        manifest_fp,
        state.input_id,
        aligner_model_arg,
        align_device=align_device,
        run_log_dir=state.run_log_dir,
    )

    state.out_seg_dir.mkdir(parents=True, exist_ok=True)

    ctm_dir = input_align_dir / "ctm" / "segments"
    for ctm_file in ctm_dir.rglob("*.ctm"):
        block_id = ctm_file.stem
        wav_src = state.norm_root / f"{block_id}.wav"
        if not wav_src.is_file():
            state.dropped_segments.append(
                {
                    "stage": "alignment",
                    "block_id": block_id,
                    "reason": "missing_source_audio",
                    "expected_language": state.lang,
                    "source_audio_path": str(wav_src),
                }
            )
            continue

        seg = Segmenter(
            str(ctm_file),
            str(wav_src),
            str(state.out_seg_dir),
            lid_model,
            ca_asr,
            es_asr,
            drop_callback=state.dropped_segments.append,
        )
        results = seg.segment_audio()
        kept_results: List[Dict[str, Any]] = []

        for result in results:
            for key in LEGACY_KEYS:
                result.pop(key, None)

            predictions = predict_languages(result["normalized_text"], lid_model, k=3)
            detected_lang, conf = choose_language_from_predictions(predictions, pri_lang=state.lang)
            prediction_summary = _format_language_predictions(predictions)
            if detected_lang not in SUPPORTED_SEGMENT_LANGS:
                state.dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "unsupported_language",
                        "expected_language": state.lang,
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
                    state.lang,
                    prediction_summary,
                    result["normalized_text"],
                )
                continue
            if detected_lang != state.lang:
                state.dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "language_mismatch",
                        "expected_language": state.lang,
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
                    state.lang,
                    prediction_summary,
                    result["normalized_text"],
                )
                continue
            if conf < min_language_confidence:
                state.dropped_segments.append(
                    {
                        "stage": "alignment",
                        "block_id": block_id,
                        "start": round(result["start"], 2),
                        "end": round(result["end"], 2),
                        "text": result["normalized_text"],
                        "reason": "language_confidence_below_threshold",
                        "expected_language": state.lang,
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
                    state.lang,
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
            state.buckets[detected_lang].append(result)
            kept_results.append(result)

        state.combined[block_id] = {
            "input_id": state.input_id,
            "results": kept_results,
        }


def _run_model_sweeps(
    states: Mapping[str, BatchItemState],
    *,
    resolved_device: str,
    nemo_model_dir: Path,
    hf_model_dir: Path,
    asr_batch_size: int,
) -> None:
    states_by_lang: Dict[str, List[BatchItemState]] = {
        seg_lang: [state for state in states.values() if state.buckets[seg_lang]]
        for seg_lang in SUPPORTED_SEGMENT_LANGS
    }

    for seg_lang, lang_states in states_by_lang.items():
        if not lang_states:
            logger.info("Skipping ASR sweeps for language={} segments=0", seg_lang)
            continue

        segs = [result for state in lang_states for result in state.buckets[seg_lang]]
        logger.info(
            "ASR sweep phase started language={} audios={} segments={}",
            seg_lang,
            len(lang_states),
            len(segs),
        )

        for name, kind, model_name in MODELS_BY_LANG[seg_lang]:
            model = None
            try:
                logger.info(
                    "Loading ASR model for batch sweep name={} type={} language={} audios={} segments={}",
                    model_name,
                    kind,
                    seg_lang,
                    len(lang_states),
                    len(segs),
                )
                model = load_model(
                    kind,
                    model_name,
                    resolved_device,
                    nemo_model_dir=nemo_model_dir,
                    hf_model_dir=hf_model_dir,
                )
                _populate_model_predictions(
                    segs,
                    model,
                    name,
                    kind,
                    model_name,
                    seg_lang,
                    batch_size=asr_batch_size,
                )
                for state in lang_states:
                    _persist_batch_item(state)
                logger.info(
                    "Completed ASR model sweep model={} language={} audios={} segments={}",
                    model_name,
                    seg_lang,
                    len(lang_states),
                    len(segs),
                )
            except Exception as err:
                logger.warning("Skipping model {} model_name={} error={}", name, model_name, err)
            finally:
                unload_model(model)


def generate_final_data_batch(
    input_ids: List[str],
    *,
    lang: str = "ca",
    device: str = "auto",
    lid_model_path: str | Path | None = None,
    nemo_model_dir: str | Path | None = None,
    hf_model_dir: str | Path | None = None,
    min_language_confidence: float = MIN_SEGMENT_LANGUAGE_CONFIDENCE,
    asr_batch_size: int = DEFAULT_ASR_BATCH_SIZE,
    output_names: Mapping[str, str] | None = None,
    run_log_dirs: Mapping[str, Path | None] | None = None,
) -> Dict[str, Path]:
    if asr_batch_size < 1:
        raise ValueError("asr_batch_size must be >= 1")

    model_paths = configure_model_environment(
        lid_model_path=lid_model_path,
        nemo_model_dir=nemo_model_dir,
        hf_model_dir=hf_model_dir,
    )
    resolved_device = _resolve_device(device)
    align_device = resolved_device if resolved_device == "cuda" else "cpu"

    states: Dict[str, BatchItemState] = {}
    for input_id in input_ids:
        states[input_id] = _build_batch_item_state(
            input_id,
            lang,
            output_name=output_names[input_id] if output_names else None,
            run_log_dir=run_log_dirs[input_id] if run_log_dirs and input_id in run_log_dirs else None,
        )

    lid_model = None
    primary_asr = None
    try:
        lid_model = _load_lid_model(model_paths.lid_model_path)
        primary_asr, aligner_model_arg = _load_primary_ctc_model(lang, resolved_device, model_paths.nemo_model_dir)

        failed_ids: List[str] = []
        for index, input_id in enumerate(input_ids, 1):
            state = states[input_id]
            logger.info(
                "Step 3/7 generate_final_data started input_id={} item={}/{} logs={}",
                input_id,
                index,
                len(input_ids),
                state.run_log_dir,
            )
            try:
                _segment_and_bucket_input(
                    state,
                    lid_model=lid_model,
                    primary_asr=primary_asr,
                    aligner_model_arg=aligner_model_arg,
                    align_device=align_device,
                    min_language_confidence=min_language_confidence,
                )
                _persist_batch_item(state, write_drop_log=True)
                logger.info(
                    "Step 3/7 segmentation completed input_id={} kept_segments={} dropped_segments={}",
                    input_id,
                    sum(len(payload["results"]) for payload in state.combined.values()),
                    len(state.dropped_segments),
                )
            except Exception as err:
                logger.error("Step 3/7 generate_final_data failed input_id={} error={}", input_id, err)
                failed_ids.append(input_id)

        unload_model(primary_asr)
        primary_asr = None
        logger.info("Primary CTC model unloaded after alignment/segmentation phase")

        for input_id in failed_ids:
            states.pop(input_id, None)
        if not states:
            return {}

        _run_model_sweeps(
            states,
            resolved_device=resolved_device,
            nemo_model_dir=model_paths.nemo_model_dir,
            hf_model_dir=model_paths.hf_model_dir,
            asr_batch_size=asr_batch_size,
        )
    finally:
        unload_model(primary_asr)
        if lid_model is not None:
            del lid_model

    return {input_id: state.final_fp for input_id, state in states.items()}


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
    run_log_dir: Path | None = None,
) -> Path:
    """Generate word-level aligned JSON with ASR enrichment."""
    return generate_final_data_batch(
        [input_id],
        lang=lang,
        device=device,
        lid_model_path=lid_model_path,
        nemo_model_dir=nemo_model_dir,
        hf_model_dir=hf_model_dir,
        min_language_confidence=min_language_confidence,
        asr_batch_size=asr_batch_size,
        output_names={input_id: output_name} if output_name else None,
        run_log_dirs={input_id: run_log_dir},
    )[input_id]


__all__ = [
    "MODELS_BY_LANG",
    "CTC_MODELS",
    "hhmmss_to_sec",
    "build_manifest",
    "run_forced_alignment",
    "generate_final_data_batch",
    "generate_final_data",
]
