"""
Punctuation and capitalization restoration utilities.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

from fsp.utils.paths import resolve_model_reference

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

LOG = logging.getLogger(__name__)

MODELS: Dict[str, Any] = {}

MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "ca": ("catalan_capitalization_punctuation_restoration", "token-classification"),
    "es": ("spanish_capitalization_punctuation_restoration", "token-classification"),
    "gl": ("galician_capitalization_punctuation_restoration", "token-classification"),
    "eu": ("cap-punct-eu", "seq2seq"),
}

UMUTEAM_LABELS: Dict[str, Dict[str, str]] = {
    "l": {"punct": "", "case": "lower"},
    "u": {"punct": "", "case": "upper"},
    "?u": {"punct": "?", "case": "upper"},
    "?l": {"punct": "?", "case": "lower"},
    "!u": {"punct": "!", "case": "upper"},
    "!l": {"punct": "!", "case": "lower"},
    ",u": {"punct": ",", "case": "upper"},
    ",l": {"punct": ",", "case": "lower"},
    ".u": {"punct": ".", "case": "upper"},
    ".l": {"punct": ".", "case": "lower"},
    ":u": {"punct": ":", "case": "upper"},
    ":l": {"punct": ":", "case": "lower"},
    "O": {"punct": "", "case": "same"},
}


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_local_model_dir(model_ref: str, hf_model_dir: str | Path | None = None) -> Path:
    candidate = resolve_model_reference(model_ref, "pipe", hf_model_dir=hf_model_dir)
    if _is_model_dir(candidate):
        return candidate
    raise FileNotFoundError(f"Local punctuation model not found for '{model_ref}': {candidate}")


def load_model_cached(lang: str, device: int, hf_model_dir: str | Path | None = None):
    cache_key = f"{lang}:{Path(hf_model_dir).expanduser() if hf_model_dir else ''}:{device}"
    if cache_key in MODELS:
        return MODELS[cache_key]

    if lang not in MODEL_MAP:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(MODEL_MAP.keys())}")

    model_name, model_type = MODEL_MAP[lang]
    model_dir = resolve_local_model_dir(model_name, hf_model_dir=hf_model_dir)
    LOG.info("Loading %s model from local path: %s (%s)", lang, model_dir, model_type)

    if model_type == "token-classification":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="none",
        )
        MODELS[cache_key] = (pipe, "token-classification")
        return MODELS[cache_key]

    if model_type == "seq2seq":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`num_beams` is set to None - defaulting to 1\.",
                category=UserWarning,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        model = model.to("cuda" if device != -1 else "cpu")
        MODELS[cache_key] = ((model, tokenizer), "seq2seq")
        return MODELS[cache_key]

    raise ValueError(f"Unknown model type '{model_type}' for language '{lang}'")


def apply_umuteam_punctuation(text: str, pipe) -> str:
    if not text.strip():
        return ""

    results = pipe(text)
    words: List[Tuple[str, str]] = []
    current_start = None
    current_end = 0
    current_label = "l"
    prev_end = 0

    for res in results:
        label = res.get("entity", "l")
        tok_start = res.get("start", prev_end)
        tok_end = res.get("end", tok_start + 1)

        if tok_start > prev_end and current_start is not None:
            words.append((text[current_start:current_end], current_label))
            current_start = tok_start
            current_label = label
        elif current_start is None:
            current_start = tok_start
            current_label = label

        current_end = tok_end
        prev_end = tok_end

    if current_start is not None:
        words.append((text[current_start:current_end], current_label))

    output_words: List[str] = []
    for word, label in words:
        meta = UMUTEAM_LABELS.get(label, UMUTEAM_LABELS["O"])
        punct = meta["punct"]
        case = meta["case"]

        if case == "upper":
            word = word.capitalize()
        elif case == "lower":
            word = word.lower()

        output_words.append(word + punct)

    return " ".join(output_words)


def apply_seq2seq_punctuation(text: str, model_tuple) -> str:
    model, tokenizer = model_tuple
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`num_beams` is set to None - defaulting to 1\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"You have modified the pretrained model configuration to control generation\..*",
            category=UserWarning,
        )
        generation_config = GenerationConfig.from_dict(model.generation_config.to_dict())
        generation_config.num_beams = 1
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=len(inputs["input_ids"][0]) * 2,
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def punctuate_text(text: str, lang: str, device: int, hf_model_dir: str | Path | None = None) -> str:
    model_obj, kind = load_model_cached(lang, device, hf_model_dir=hf_model_dir)
    if kind == "token-classification":
        return apply_umuteam_punctuation(text, model_obj)
    if kind == "seq2seq":
        return apply_seq2seq_punctuation(text, model_obj)
    return text


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read/parse JSON: {path}") from exc


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_video_data(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    try:
        return next(iter(data.values()))
    except Exception as exc:
        raise ValueError(f"Invalid JSON structure: {path}") from exc


def _get_results(video_data: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    results = video_data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing or invalid 'results' in {path}")
    return results


def _require_language(segment: Dict[str, Any], path: Path) -> str:
    lang = segment.get("language")
    if not lang:
        raise ValueError(f"Missing 'language' in {path}")
    if lang not in MODEL_MAP:
        raise ValueError(f"Unsupported language '{lang}' in {path}")
    return lang


def _extract_input_text(segment: Dict[str, Any]) -> str:
    text = (segment.get("rover_text") or "").strip()
    if text:
        return text
    for key, value in segment.items():
        if key.startswith("pred_text_") and isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def process_file(json_path: Path, device: int, hf_model_dir: str | Path | None = None) -> None:
    LOG.info("Processing %s ...", json_path)
    data = _read_json(json_path)
    video_data = _get_video_data(data, json_path)
    results = _get_results(video_data, json_path)

    for segment in results:
        lang = _require_language(segment, json_path)
        input_text = _extract_input_text(segment)
        if not input_text:
            LOG.warning("No text found in segment, skipping.")
            continue

        segment["text_normalized"] = input_text
        segment["text_punctuated"] = punctuate_text(
            input_text,
            lang,
            device,
            hf_model_dir=hf_model_dir,
        )

    _write_json(json_path, data)
    LOG.info("Updated %s", json_path)


__all__ = [
    "MODEL_MAP",
    "process_file",
    "punctuate_text",
]
