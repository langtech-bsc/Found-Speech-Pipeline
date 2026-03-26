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

from fsp.core.text import clean_apostrophes, clean_text
from fsp.utils.language import choose_language
from fsp.utils.models import (
    configure_model_environment,
    find_local_nemo_checkpoint,
    load_model,
    unload_model,
)
from fsp.utils.paths import ALIGN_DIR, MANIFEST_DIR, NORM_DIR, OUTPUT_SEGMENT_DIR, ROOT

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
        ("whisper_large_v3_fallback", "pipe", "whisper-large-v3"),
    ),
    "gl": (
        ("stt_gl_conformer_ctc_large", "ctc", "stt_gl_conformer_ctc_large"),
        ("whisper_large_v3_gl", "pipe", "whisper-large-v3-gl"),
        ("whisper_large_v3_fallback", "pipe", "whisper-large-v3"),
    ),
}

CTC_MODELS = {
    "ca": "stt_ca_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
    "eu": "stt_eu_conformer_ctc_large",
    "gl": "stt_gl_conformer_ctc_large",
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
    '["text"]' (or "['text']") – i.e. a JSON list encoded as a string.
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



def transcribe(model: Any, kind: str, audio: str, lang: str = "ca") -> str:
    """Run model on audio and normalize the string it returns."""
    if kind == "pipe":
        out = model(audio, generate_kwargs={"task": "transcribe", "language": lang})
        if isinstance(out, dict):
            txt = out.get("text", "")
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            txt = " ".join(d.get("text", "") for d in out)
        else:
            txt = str(out)
    else:
        out = model.transcribe([audio], batch_size=1)[0]
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))

    return clean_apostrophes(_clean_singleton_json_array(txt))


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
) -> Path:
    """Run NeMo forced aligner."""
    input_align_dir = ALIGN_DIR / input_id
    input_align_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py"),
            aligner_model_arg,
            f"manifest_filepath={manifest_fp}",
            f"output_dir={input_align_dir}",
            "align_using_pred_text=false",
            "transcribe_device=cpu",
            "viterbi_device=cpu",
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
    """Generate word-level aligned JSON with ASR enrichment."""
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

    primary_model_name = CTC_MODELS[lang]
    local_nemo = find_local_nemo_checkpoint(
        primary_model_name,
        nemo_model_dir=model_paths.nemo_model_dir,
    )

    if local_nemo is not None and local_nemo.is_file():
        logger.info(f"Loading local NeMo model: {local_nemo}")
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
    input_align_dir = run_forced_alignment(manifest_fp, input_id, aligner_model_arg)

    out_seg_dir = OUTPUT_SEGMENT_DIR / input_id
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {"ca": [], "es": [], "eu": [], "gl": []}
    combined: Dict[str, Any] = {}

    ctm_dir = input_align_dir / "ctm" / "segments"
    for ctm_file in ctm_dir.rglob("*.ctm"):
        block_id = ctm_file.stem
        wav_src = norm_root / f"{block_id}.wav"
        if not wav_src.is_file():
            continue

        seg = Segmenter(str(ctm_file), str(wav_src), str(out_seg_dir), lid_model, ca_asr, es_asr)
        results = seg.segment_audio()
        kept_results: List[Dict[str, Any]] = []

        for result in results:
            for key in LEGACY_KEYS:
                result.pop(key, None)

            detected_lang, conf = choose_language(result["normalized_text"], lid_model, pri_lang=lang)
            if detected_lang not in ("ca", "es", "eu", "gl"):
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
                for result in segs:
                    key = f"pred_text_{name}"
                    norm_key = f"norm_text_{name}"
                    transcription = result.get(key, "") or ""
                    if transcription:
                        if norm_key not in result:
                            try:
                                result[norm_key] = clean_text(transcription, seg_lang, False, False)
                            except Exception as err:
                                logging.warning("[norm_script:] %s on %s: %s", model_name, result["segment_path"], err)
                                result[norm_key] = ""
                        continue
                    try:
                        transcription = transcribe(model, kind, result["segment_path"], lang=seg_lang)
                        result[key] = transcription
                    except Exception as err:
                        logging.warning("%s on %s: %s", model_name, result["segment_path"], err)
                        result[key] = ""
                    try:
                        result[norm_key] = clean_text(transcription, seg_lang, False, False)
                    except Exception as err:
                        logging.warning("[norm_script:] %s on %s: %s", model_name, result["segment_path"], err)
                        result[norm_key] = ""
            except Exception as err:
                logging.warning("Could not load model %s (%s): %s - skipping", name, model_name, err)
                logger.warning(f"Skipping model {name}: {err}")
            finally:
                unload_model(model)

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
