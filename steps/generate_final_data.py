#!/usr/bin/env python3
"""
generate_final_data.py
======================

Forced alignment + ASR enrichment for **one** audio-transcript pair

• Segments are language-detected first.
• For each language we load *one* model at a time → transcribe → unload.
"""

from __future__ import annotations

# standard library
import argparse
import gc
import json
import logging
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Suppress the irritating expected warning from using HuggingFace pipelines sequentially
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

# third-party libs
import fasttext
import nemo.collections.asr as nemo_asr
import torch
from num2words import num2words
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from pyctcdecode import build_ctcdecoder
from transformers import pipeline as hf_pipeline
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_paths import configure_model_env, resolve_fasttext_model
from scripts.clean_and_expand import clean_text

# ── Resolve model locations for cluster and repo-local runs ─────────────
MODELS_ROOT = str(configure_model_env())

# project local
from segment import Segmenter

# logging
LOG_DIR = Path("inputs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "output.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate word-level aligned JSON (one input-id)")
    p.add_argument("--input-id", required=True, help="YouTube video-id to process")
    p.add_argument("--lang", choices=("ca", "es", "eu", "gl"), default="ca", help="Primary language (only its CTC model is loaded)")
    p.add_argument("--output", metavar="NAME.json", help="Custom JSON name (default: final_output_<input-id>.json)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Run ASR on cuda / cpu (default auto)")
    return p.parse_args()


# Only parse CLI args when run as a script, not when imported
ARGS = None
if __name__ == "__main__" or "pytest" not in sys.modules:
    try:
        ARGS = parse_args()
        if not ARGS.output:
            ARGS.output = f"final_output_{ARGS.input_id}.json"
    except SystemExit:
        # Allow import without CLI args (e.g. for testing)
        pass


# model catalogue — all paths are local directories in GPFS
MODELS_BY_LANG: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "ca": (
        ("whisper_ca_3catparla", "pipe", f"{MODELS_ROOT}/whisper-large-v3-ca-3catparla"),
        ("whisper_bsc_cat", "pipe", f"{MODELS_ROOT}/Whisper-bsc-large-v3-cat"),
        ("whisper_ca_punct_3370h", "pipe", f"{MODELS_ROOT}/whisper-large-v3-ca-punctuated-3370h"),
        ("stt_ca_es_conformer_transducer_large", "rnnt", f"{MODELS_ROOT}/stt_ca-es_conformer_transducer_large"),
    ),
    "es": (
        ("parakeet_rnnt_es", "rnnt", f"{MODELS_ROOT}/parakeet-rnnt-1.1b_cv17_es_ep18_1270h"),
        ("stt_es_conformer_transducer_large", "rnnt", f"{MODELS_ROOT}/stt_es_conformer_transducer_large"),
        ("whisper_large_v3", "pipe", f"{MODELS_ROOT}/whisper-large-v3"),
    ),
    "eu": (
        ("stt_eu_conformer_transducer_large", "rnnt", f"{MODELS_ROOT}/stt_eu_conformer_transducer_large"),
        ("stt_eu_conformer_ctc_large", "ctc", f"{MODELS_ROOT}/stt_eu_conformer_ctc_large"),
        ("whisper_tiny_eu", "pipe", f"{MODELS_ROOT}/whisper-tiny-eu"),
        ("whisper_small_eu", "pipe", f"{MODELS_ROOT}/whisper-small-eu"),
        ("whisper_base_eu", "pipe", f"{MODELS_ROOT}/whisper-base-eu"),
        ("whisper_medium_eu", "pipe", f"{MODELS_ROOT}/whisper-medium-eu"),
        ("whisper_large_eu", "pipe", f"{MODELS_ROOT}/whisper-large-eu"),
        ("whisper_large_v2_eu", "pipe", f"{MODELS_ROOT}/whisper-large-v2-eu"),
        ("whisper_large_v3_eu", "pipe", f"{MODELS_ROOT}/whisper-large-v3-eu"),
        # Last-resort fallback (multilingual)
        ("whisper_large_v3_fallback", "pipe", f"{MODELS_ROOT}/whisper-large-v3"),
    ),
    "gl": (
        ("stt_gl_conformer_ctc_large", "ctc", f"{MODELS_ROOT}/stt_gl_conformer_ctc_large"),
        ("whisper_large_v3_gl", "pipe", f"{MODELS_ROOT}/whisper-large-v3-gl"),
        # Last-resort fallback (multilingual)
        ("whisper_large_v3_fallback", "pipe", f"{MODELS_ROOT}/whisper-large-v3"),
    ),
}

# NFA (NeMo Forced Aligner) models by language — all local
NFA_MODELS_BY_LANG: Dict[str, Tuple[str, str]] = {
    "ca": ("local", f"{MODELS_ROOT}/stt_ca_conformer_ctc_large"),
    "es": ("local", f"{MODELS_ROOT}/stt_es_conformer_ctc_large"),
    "eu": ("local", f"{MODELS_ROOT}/stt_eu_conformer_ctc_large"),
    "gl": ("local", f"{MODELS_ROOT}/stt_gl_conformer_ctc_large"),
}


def get_nfa_model_arg(lang: str) -> str:
    """Get the NFA model argument for the given language."""
    if lang not in NFA_MODELS_BY_LANG:
        # Default to Catalan for unknown languages
        lang = "ca"
    _model_type, model_id = NFA_MODELS_BY_LANG[lang]
    nemo = _find_nemo_file(model_id)
    if nemo:
        return f"model_path={nemo}"
    # Directory without .nemo — try as pretrained name (from offline cache)
    return f"pretrained_name={Path(model_id).name}"


LEGACY_KEYS = {"pred_text", "cer_score"}


def _find_nemo_file(path: str) -> str | None:
    """If *path* is (or contains) a .nemo file, return its absolute path."""
    p = Path(path)
    if p.is_file() and p.suffix == ".nemo":
        return str(p)
    if p.is_dir():
        nemo_files = sorted(p.glob("*.nemo"))
        if nemo_files:
            return str(nemo_files[0])
    return None


# helper functions
def _clean_singleton_json_array(txt: str) -> str:
    """
    Some NeMo RNN-T checkpoints return their transcript wrapped like
    '[\"text\"]' (or "['text']") – i.e. a JSON list encoded as a string.
    Detect and unwrap that safely.
    """
    s = txt.strip()
    if not (s.startswith('[\"') or s.startswith("['")):
        return s

    try:
        parsed = json.loads(s.replace("'", '"'))
        if isinstance(parsed, list) and len(parsed) == 1:
            return str(parsed[0]).strip()
    except Exception:
        pass

    return s[2:-2].strip()


def load_model(kind: str, repo: str, device: str):
    """Load one ASR model from a local path (no downloading)."""
    if kind == "pipe":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return hf_pipeline(
            "automatic-speech-recognition",
            model=repo,
            device=-1 if device == "cpu" else 0,
            torch_dtype=dtype,
        )
    if kind == "rnnt":
        nemo = _find_nemo_file(repo)
        if nemo:
            return EncDecRNNTBPEModel.restore_from(nemo, map_location=device).to(device).eval()
        # local HF-format dir or pretrained name (offline cache)
        return EncDecRNNTBPEModel.from_pretrained(repo, map_location=device).to(device).eval()
    if kind == "ctc":
        nemo = _find_nemo_file(repo)
        if nemo:
            return nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo, map_location=device).to(device).eval()
        return nemo_asr.models.EncDecCTCModelBPE.from_pretrained(repo, map_location=device).to(device).eval()
    if kind == "multi":
        m = EncDecMultiTaskModel.from_pretrained(repo, map_location=device).to(device).eval()
        m.cfg.prompt_format = m.prompt_format = "canary"
        m.cfg.decoding.beam.beam_size = 1
        m.change_decoding_strategy(m.cfg.decoding)
        return m
    raise ValueError(f"Unknown model kind: {kind}")


def unload_model(model) -> None:
    """Free CPU/GPU RAM once a model is no longer needed."""
    try:
        del model
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def transcribe(model, kind: str, audio: str, lang: str = "ca") -> str:
    """Run *model* on *audio* and normalise the string it returns."""
    if kind == "pipe":
        gen_kwargs = {"task": "transcribe", "language": lang}
        out = model(audio, generate_kwargs=gen_kwargs)
        if isinstance(out, dict):
            txt = out.get("text", "")
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            txt = " ".join(d.get("text", "") for d in out)
        else:
            txt = str(out)
    elif kind in ("rnnt", "multi"):
        out = model.transcribe([audio], batch_size=1)[0]
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))
    elif kind == "ctc":
        out = model.transcribe([audio], batch_size=1)[0]
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return _clean_singleton_json_array(txt)


def hhmmss_to_sec(t: str | int | float) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return float(t)


def choose_language(text: str, lid, conf_delta: float = 0.2,
                    pri_lang: str | None = None) -> Tuple[str, float]:
    """FastText-based language choice for ca/es/eu/gl."""
    
    SUPPORTED = {"ca", "es", "eu", "gl"}

    labels3, confs3 = lid.predict(text, k=3)
    langs3 = [x.replace("__label__", "") for x in labels3]
    confs3 = [float(x) for x in confs3]

    l1, c1 = langs3[0], confs3[0]
    l2, c2 = langs3[1], confs3[1]

    # ── Priority: if the expected pipeline language appears in top-3 and
    #    top-1 is an unsupported language with low confidence, prefer pri_lang.
    if (
        pri_lang in SUPPORTED
        and l1 not in SUPPORTED
        and c1 < 0.5 
        and pri_lang in langs3
    ):
        idx = langs3.index(pri_lang)
        return pri_lang, confs3[idx]

    # Basque (Euskara)
    if l1 == "eu":
        return "eu", c1

    if l1 == "es" and l2 == "eu" and (c1 - c2) < conf_delta:
        return "eu", c2

    # Galician
    if l1 == "gl": return "gl", c1
    if l1 == "pt" and l2 == "gl" and (c1 - c2) < conf_delta: return "gl", c2

    # Catalan / Spanish disambiguation
    if l1 == "ca":
        return "ca", c1
    if l1 == "es" and l2 == "ca" and (c1 - c2) < conf_delta:
        return "ca", c2

    catalan_tokens = (" l’", " d’", "ç", " ny", "això", "qüestió")
    if any(tok in text.lower() for tok in catalan_tokens):
        return "ca", c2 if l2 == "ca" else 0.01

    return l1, c1



def build_manifest(meta_path: Path, lid_model) -> Path:
    """Return a NeMo manifest."""
    input_id = meta_path.stem.replace("_metadata", "")
    mdir = Path("inputs/manifest"); mdir.mkdir(parents=True, exist_ok=True)
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
        else:  # list of (span, token)
            src_norm = " ".join(w for _sp, w in blk["normalized_text"])
        if not src_norm.strip():
            continue
        
        if isinstance(blk["original_text"], str):
            src_org = blk["original_text"]
        else:  # list of (span, token)
            src_org = " ".join(w for _sp, w in blk["original_text"])
        if not src_org.strip():
            continue

        lang, conf = choose_language(src_norm, lid_model, pri_lang=ARGS.lang)
        cleaned_normalized = clean_text(src_norm, lang, False, False)

        entries.append({
            "audio_filepath": str(wav.resolve()),
            "text": cleaned_normalized,
            "original_text": src_org,
            "language": f"{lang}__{conf:.2f}",
        })

    if not entries:
        raise RuntimeError(f"No valid blocks in {input_id}")

    with manifest_fp.open("w", encoding="utf-8") as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
    return manifest_fp

def main() -> None:
    print("Starting main", flush=True)
    global ARGS
    ARGS = parse_args()
    if not ARGS.output:
        ARGS.output = f"final_output_{ARGS.session}.json"

    start = time.perf_counter()

    # Static resources
    lid_path = resolve_fasttext_model(Path(MODELS_ROOT))
    if not lid_path.exists():
        sys.exit(f"❌  FastText model not found: {lid_path}")
    lid_model = fasttext.load_model(str(lid_path))

    # Load the CTC model for the requested language using NFA_MODELS_BY_LANG
    _nfa_type, nfa_id = NFA_MODELS_BY_LANG[ARGS.lang]
    nfa_path = Path(nfa_id)

    nemo_file = _find_nemo_file(str(nfa_path))
    if nemo_file:
        print(f"Loading local NeMo model: {nemo_file}")
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo_file)
    elif nfa_path.is_dir():
        print(f"Loading NeMo model from directory: {nfa_path}")
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(nfa_path))
    else:
        sys.exit(f"❌  NFA model not found: {nfa_path}")

    ca_asr = primary_asr if ARGS.lang == "ca" else None
    es_asr = primary_asr if ARGS.lang == "es" else None

    device = (
        "cpu"
        if ARGS.device == "cpu"
        else "cuda"
        if (ARGS.device == "cuda" or (ARGS.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    def transcribe_segmenter_baseline(segment: Dict[str, Any], lang: str) -> None:
        # Preserve the legacy pred_text_segmenter field for ca/es without reintroducing
        # a second source of truth for language assignment.
        if lang == "ca":
            model = ca_asr or es_asr
        elif lang == "es":
            model = es_asr or ca_asr
        else:
            return

        if model is None:
            return

        try:
            segment["pred_text_segmenter"] = transcribe(model, "ctc", segment["segment_path"], lang=lang)
        except Exception as err:  # noqa: BLE001
            logging.warning("segmenter baseline on %s: %s", segment["segment_path"], err)
            segment["pred_text_segmenter"] = ""

    # Paths
    input_id = ARGS.input_id
    norm_root = Path("inputs/normalized") / input_id
    meta_path = norm_root / f"{input_id}_metadata.json"
    if not meta_path.is_file():
        sys.exit(f"❌  metadata not found: {meta_path}")

    # 1 ── Forced alignment
    manifest_fp = build_manifest(meta_path, lid_model)
    align_root = Path("inputs/wordlevel_alignment")
    input_align_dir = align_root / input_id
    input_align_dir.mkdir(parents=True, exist_ok=True)

    # Use get_nfa_model_arg for the aligner
    aligner_model_arg = get_nfa_model_arg(ARGS.lang)

    subprocess.run([
        sys.executable,
        "NeMo/tools/nemo_forced_aligner/align.py",
        aligner_model_arg,
        f"manifest_filepath={manifest_fp}",
        f"output_dir={input_align_dir}",
        "align_using_pred_text=false",
        "transcribe_device=cpu",
        "viterbi_device=cpu",
        "additional_segment_grouping_separator=|",
        "hydra.run.dir=.",
    ], check=True)

    # 2 ── Segment & language-tag
    out_seg_dir = Path("inputs/output_segment") / input_id
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {"ca": [], "es": [], "eu": [], "gl": []}
    combined: Dict[str, Any] = {}

    ctm_dir = input_align_dir / "ctm" / "segments"
    for ctm_file in ctm_dir.rglob("*.ctm"):
        block_id = ctm_file.stem
        wav_src = norm_root / f"{block_id}.wav"
        if not wav_src.is_file():
            continue

        seg = Segmenter(str(ctm_file), str(wav_src), str(out_seg_dir))
        results = seg.segment_audio()
        kept_results: List[Dict[str, Any]] = []

        for r in results:
            lang, conf = choose_language(r["normalized_text"], lid_model, pri_lang=ARGS.lang)
            if lang not in ("ca", "es", "eu", "gl"):
                continue

            r["language"] = lang
            r["language_confidence"] = round(conf, 2)
            transcribe_segmenter_baseline(r, lang)
            kept_results.append(r)
            buckets[lang].append(r)

        combined[block_id] = {
            "input_id": input_id,
            "results": kept_results,
        }
          
    # 3 ── ASR per-language, one model at a time
    for lang, segs in buckets.items():
        print(f"\nProcessing language: {lang}, number of segments: {len(segs)}")
        if not segs:
            continue
        for name, kind, repo in MODELS_BY_LANG.get(lang, ()):
            model = None
            try:
                logging.info("loading %s", repo)
                model = load_model(kind, repo, device)
                print(f"Model {name} loaded on device {device}")
                for r in segs:
                    key = f"pred_text_{name}"
                    norm_key = f"norm_text_{name}"
                    transcription = ""
                    if r.get(key):
                        continue
                    try:
                        transcription = transcribe(model, kind, r["segment_path"], lang=lang)
                        r[key] = transcription
                    except Exception as err:  # noqa: BLE001
                        logging.warning("%s on %s: %s",
                                        repo, r["segment_path"], err)
                        r[key] = ""
                    try:
                        r[norm_key] = clean_text(transcription, lang, False, False)
                    except Exception as err:  # noqa: BLE001
                        logging.warning("[norm_script:] %s on %s: %s",
                                        repo, r["segment_path"], err)
                        r[key] = ""
            except Exception as err:  # noqa: BLE001
                logging.warning("Could not load model %s (%s): %s — skipping",
                                name, repo, err)
                print(f"⚠  Skipping model {name}: {err}")
            finally:
                unload_model(model)

    # 4 ── write JSON
    final_fp = Path("inputs/output_segment") / ARGS.output
    final_fp.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓  JSON written to {final_fp}  ({time.perf_counter()-start:.1f}s)")


if __name__ == "__main__":
    print("Starting script", flush=True)
    try:
        main()
    except Exception as e:  # noqa: BLE001
        logging.error("Fatal error: %s", e)
        sys.exit(1)
