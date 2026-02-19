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
from pathlib import Path
from typing import Any, Dict, List, Tuple

# third-party libs
import fasttext
import nemo.collections.asr as nemo_asr
import torch
from num2words import num2words
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from pyctcdecode import build_ctcdecoder
from transformers import pipeline as hf_pipeline

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
    p.add_argument("--lang", choices=("ca", "es"), default="ca", help="Primary language (only its CTC model is loaded)")
    p.add_argument("--output", metavar="NAME.json", help="Custom JSON name (default: final_output_<input-id>.json)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Run ASR on cuda / cpu (default auto)")
    return p.parse_args()


ARGS = parse_args()
if not ARGS.output:
    ARGS.output = f"final_output_{ARGS.input_id}.json"

# model catalogue
MODELS_BY_LANG: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "ca": (
        ("whisper_ca_3catparla", "pipe", "projecte-aina/whisper-large-v3-ca-3catparla"),
        ("whisper_bsc_cat", "pipe", "langtech-veu/whisper-bsc-large-v3-cat"),
        ("whisper_ca_punct_3370h", "pipe", "langtech-veu/whisper-large-v3-ca-punctuated-3370h"),
        ("stt_ca_es_conformer_transducer_large", "rnnt", "projecte-aina/stt_ca-es_conformer_transducer_large"),
    ),
    "es": (
        ("parakeet_rnnt_es", "rnnt", "projecte-aina/parakeet-rnnt-1.1b_cv17_es_ep18_1270h"),
        ("stt_es_conformer_transducer_large", "rnnt", "nvidia/stt_es_conformer_transducer_large"),
        ("whisper_large_v3", "pipe", "openai/whisper-large-v3"),
    ),
    "eu": (
        ("stt_eu_conformer_transducer_large", "rnnt", "HiTZ/stt_eu_conformer_transducer_large"),
        ("stt_eu_conformer_ctc_large", "ctc", "HiTZ/stt_eu_conformer_ctc_large"),
        ("whisper_tiny_eu", "pipe", "HiTZ/whisper-tiny-eu"),
        ("whisper_small_eu", "pipe", "HiTZ/whisper-small-eu"),
        ("whisper_base_eu", "pipe", "HiTZ/whisper-base-eu"),
        ("whisper_medium_eu", "pipe", "HiTZ/whisper-medium-eu"),
        ("whisper_large_eu", "pipe", "HiTZ/whisper-large-eu"),
        ("whisper_large_v2_eu", "pipe", "HiTZ/whisper-large-v2-eu"),
        ("whisper_large_v3_eu", "pipe", "HiTZ/whisper-large-v3-eu"),
        # Last-resort fallback (multilingual)
        ("whisper_large_v3_fallback", "pipe", "openai/whisper-large-v3"),
    ),
    "gl": (
        ("whisper_large_v3_gl", "pipe", "mozilla-ai/whisper-large-v3-gl"),
        # Last-resort fallback (multilingual)
        ("whisper_large_v3_fallback", "pipe", "openai/whisper-large-v3"),
    ),
}

# NFA (NeMo Forced Aligner) models by language
# Use pretrained_name for HuggingFace models, model_path for local .nemo files
NFA_MODELS_BY_LANG: Dict[str, Tuple[str, str]] = {
    # (type, model_identifier) - type is "pretrained" or "local"
    "ca": ("pretrained", "stt_ca_conformer_ctc_large"),
    "es": ("pretrained", "stt_es_conformer_ctc_large"),
    "eu": ("local", "utils/models/nfa/eu/stt_eu_conformer_ctc_large.nemo"),
    "gl": ("local", "utils/models/nfa/stt_gl_conformer_ctc_large.nemo"),  # Repackaged .nemo file
}


def get_nfa_model_arg(lang: str) -> str:
    """Get the NFA model argument for the given language."""
    if lang not in NFA_MODELS_BY_LANG:
        # Default to Catalan for unknown languages
        lang = "ca"
    model_type, model_id = NFA_MODELS_BY_LANG[lang]
    if model_type == "pretrained":
        return f"pretrained_name={model_id}"
    return f"model_path={model_id}"


LEGACY_KEYS = {"pred_text", "cer_score"}

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
    """Load one ASR model (no global cache so RSS stays small)."""
    if kind == "pipe":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return hf_pipeline(
            "automatic-speech-recognition",
            model=repo,
            device=-1 if device == "cpu" else 0,
            torch_dtype=dtype,
        )
    if kind == "rnnt":
        return EncDecRNNTBPEModel.from_pretrained(repo, map_location=device).to(device).eval()
    if kind == "ctc":
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


def transcribe(model, kind: str, audio: str) -> str:
    """Run *model* on *audio* and normalise the string it returns."""
    if kind == "pipe":
        out = model(audio, generate_kwargs={"task": "transcribe"})
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


def choose_language(text: str, lid, conf_delta: float = 0.2) -> Tuple[str, float]:
    """FastText-based language choice for ca/es/eu/gl."""
    labels, confs = lid.predict(text, k=2)
    l1, c1 = labels[0].replace("__label__", ""), float(confs[0])
    l2, c2 = labels[1].replace("__label__", ""), float(confs[1])

    # Basque (Euskara)
    if l1 == "eu":
        return "eu", c1

    # Galician
    if l1 == "gl": return "gl", c1
    if l1 == "pt" and l2 == "gl"  and (c1 - c2) < conf_delta: return "gl", c2

    # Catalan / Spanish disambiguation
    if l1 == "ca":
        return "ca", c1
    if l1 == "es" and l2 == "ca" and (c1 - c2) < conf_delta:
        return "ca", c2

    catalan_tokens = (" l’", " d’", "ç", " ny", "això", "qüestió")
    if any(tok in text.lower() for tok in catalan_tokens):
        return "ca", c2 if l2 == "ca" else 0.01

    return l1, c1


def numbers_to_words(txt: str, tgt_lang: str) -> str:
    """
    Replace every integer in *txt* with its cardinal representation
    in Catalan (tgt_lang == 'cat') or Spanish (tgt_lang == 'spa') using num2words.
    """
    lang_map = {"cat": "ca", "spa": "es"}
    n2w_lang = lang_map.get(tgt_lang, "en")

    def _replace(match: re.Match[str]) -> str:
        try:
            return num2words(int(match.group()), lang=n2w_lang)
        except Exception:
            return match.group()

    return re.sub(r"\d+", _replace, txt)


def clean_text(text: str, label: str) -> str:
    text = re.sub(r"(\d+)[.,](\d+)", r"\1\2", text)
    text = re.sub(r"\([^)]*\)", "", text).replace("...", ".")
    if label == "__label__ca":
        text = numbers_to_words(text.replace("%", " per cent"), "cat")
    elif label == "__label__es":
        text = numbers_to_words(text.replace("%", " por ciento"), "spa")
    elif label == "__label__eu":
        # Basque: keep text mostly as-is (num2words doesn't support eu)
        text = text.replace("%", " ehuneko")
    elif label == "__label__gl":
        # Galician: close to Spanish for num2words purposes
        text = numbers_to_words(text.replace("%", " por cento"), "spa")

    forbidden = set(",;:?¿«»-¡!@*{}[]=/\\&#…")
    text = "".join(ch if ch not in forbidden else " " for ch in text)
    return " ".join(text.split()).lower()


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

        lang, conf = choose_language(src_norm, lid_model)
        cleaned_normalized = clean_text(src_norm, f"__label__{lang}")

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
    global ARGS
    ARGS = parse_args()
    if not ARGS.output:
        ARGS.output = f"final_output_{ARGS.session}.json"

    start = time.perf_counter()

    # Static resources
    lid_model = fasttext.load_model("utils/models/lid.176.bin")

    # Only load the CTC model for the requested language
    CTC_MODELS = {
        "ca": "stt_ca_conformer_ctc_large",
        "es": "stt_es_conformer_ctc_large",
    }
    primary_model_name = CTC_MODELS[ARGS.lang]

    # Check for pre-downloaded local model first
    local_nemo = Path("utils/models/nemo") / f"{primary_model_name}.nemo"
    if local_nemo.is_file():
        print(f"Loading local NeMo model: {local_nemo}")
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(local_nemo))
    else:
        primary_asr = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(primary_model_name)

    ca_asr = primary_asr if ARGS.lang == "ca" else None
    es_asr = primary_asr if ARGS.lang == "es" else None

    device = (
        "cpu"
        if ARGS.device == "cpu"
        else "cuda"
        if (ARGS.device == "cuda" or (ARGS.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

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

    # Use local model path for the aligner if available
    if local_nemo.is_file():
        aligner_model_arg = f"model_path={local_nemo}"
    else:
        aligner_model_arg = f"pretrained_name={primary_model_name}"

    subprocess.run([
        sys.executable,
        "NeMo/tools/nemo_forced_aligner/align.py",
        aligner_model_arg,
        f"manifest_filepath={manifest_fp}",
        f"output_dir={input_align_dir}",
        "align_using_pred_text=false",
        "transcribe_device=cpu",
        "viterbi_device=cpu",
        "additional_segment_grouping_separator=.",
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

        seg = Segmenter(str(ctm_file), str(wav_src), str(out_seg_dir), lid_model, ca_asr, es_asr)
        results = seg.segment_audio()

        for r in results:
            # Preserve Segmenter baseline hypotheses so downstream always has something to merge.
            if isinstance(r.get("pred_text"), str) and r["pred_text"].strip():
                r["pred_text_segmenter"] = r["pred_text"]
            if isinstance(r.get("pred_text_lm"), str) and r["pred_text_lm"].strip():
                r["pred_text_segmenter_lm"] = r["pred_text_lm"]

            # Drop legacy keys to avoid ambiguity
            for k in LEGACY_KEYS:
                r.pop(k, None)

            lang, conf = choose_language(r["normalized_text"], lid_model)
            if lang not in ("ca", "es", "eu", "gl"): continue
            r["language"] = lang
            r["language_confidence"] = round(conf, 2)
            buckets[lang].append(r)

        combined[block_id] = {
            "input_id": input_id,
            "results": results,
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
                    if r.get(key):
                        continue
                    try:
                        r[key] = transcribe(model, kind, r["segment_path"])
                    except Exception as err:  # noqa: BLE001
                        logging.warning("%s on %s: %s", repo, r["segment_path"], err)
                        r[key] = ""
            except Exception as err:  # noqa: BLE001
                logging.warning("Could not load model %s (%s): %s — skipping",
                                name, repo, err)
                print(f"⚠  Skipping model {name}: {err}")
            finally:
                unload_model(model)

    # 4 ── write JSON
    final_fp = Path("inputs/output_segment") / ARGS.output
    final_fp.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"✓  JSON written to {final_fp}  ({time.perf_counter()-start:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        logging.error("Fatal error: %s", e)
        sys.exit(1)