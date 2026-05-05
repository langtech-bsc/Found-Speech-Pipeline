"""
Punctuation and capitalization restoration utilities.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

from fsp.utils.paths import resolve_hf_model_dir

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

LOG = logger.bind(name=__name__)
DEFAULT_PUNCT_BATCH_SIZE = 8

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
    candidate = resolve_hf_model_dir(hf_model_dir) / model_ref
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
    LOG.info("Loading {} model from local path: {} ({})", lang, model_dir, model_type)

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
    return _format_umuteam_results(text, results)


def _format_umuteam_results(text: str, results: List[Dict[str, Any]]) -> str:
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
    if not text.strip():
        return ""

    model, tokenizer = model_tuple
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return _generate_seq2seq_batch(model, tokenizer, inputs)[0]


def _generate_seq2seq_batch(model, tokenizer, inputs: Dict[str, torch.Tensor]) -> List[str]:
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
                max_new_tokens=int(inputs["attention_mask"].sum(dim=1).max().item()) * 2,
            )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def apply_umuteam_punctuation_batch(
    texts: Iterable[str],
    pipe,
    batch_size: int = DEFAULT_PUNCT_BATCH_SIZE,
) -> List[str]:
    text_list = list(texts)
    if not text_list:
        return []

    outputs = pipe(text_list, batch_size=batch_size)
    if len(outputs) != len(text_list):
        raise ValueError(f"Expected {len(text_list)} punctuation outputs, got {len(outputs)}")

    return [
        _format_umuteam_results(text, result) if text.strip() else ""
        for text, result in zip(text_list, outputs, strict=True)
    ]


def apply_seq2seq_punctuation_batch(
    texts: Iterable[str],
    model_tuple,
) -> List[str]:
    text_list = list(texts)
    if not text_list:
        return []

    nonempty_idx = [idx for idx, text in enumerate(text_list) if text.strip()]
    if not nonempty_idx:
        return ["" for _ in text_list]

    model, tokenizer = model_tuple
    nonempty_texts = [text_list[idx] for idx in nonempty_idx]
    inputs = tokenizer(nonempty_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    decoded = _generate_seq2seq_batch(model, tokenizer, inputs)
    if len(decoded) != len(nonempty_texts):
        raise ValueError(f"Expected {len(nonempty_texts)} seq2seq outputs, got {len(decoded)}")

    results = ["" for _ in text_list]
    for idx, text in zip(nonempty_idx, decoded, strict=True):
        results[idx] = text
    return results


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


def punctuate_text_batch(
    texts: List[str],
    lang: str,
    device: int,
    hf_model_dir: str | Path | None = None,
    batch_size: int = DEFAULT_PUNCT_BATCH_SIZE,
) -> List[str]:
    model_obj, kind = load_model_cached(lang, device, hf_model_dir=hf_model_dir)
    if kind == "token-classification":
        return apply_umuteam_punctuation_batch(texts, model_obj, batch_size=batch_size)
    if kind == "seq2seq":
        return apply_seq2seq_punctuation_batch(texts, model_obj)
    return texts


def process_file(
    json_path: Path,
    device: int,
    hf_model_dir: str | Path | None = None,
    batch_size: int = DEFAULT_PUNCT_BATCH_SIZE,
) -> None:
    LOG.info("Processing {} ...", json_path)
    data = _read_json(json_path)
    video_data = _get_video_data(data, json_path)
    results = _get_results(video_data, json_path)

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    grouped_segments: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}

    for segment in results:
        lang = _require_language(segment, json_path)
        input_text = _extract_input_text(segment)
        if not input_text:
            LOG.warning("No text found in segment, skipping.")
            continue

        segment["text_normalized"] = input_text
        grouped_segments.setdefault(lang, []).append((segment, input_text))

    for lang, items in grouped_segments.items():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts = [text for _segment, text in batch]
            try:
                punctuated = punctuate_text_batch(
                    texts,
                    lang,
                    device,
                    hf_model_dir=hf_model_dir,
                    batch_size=batch_size,
                )
                LOG.info(
                    "Batch punctuation succeeded lang={} batch_size={} inputs={} outputs={}",
                    lang,
                    batch_size,
                    len(texts),
                    len(punctuated),
                )
                for (segment, _text), output in zip(batch, punctuated, strict=True):
                    segment["text_punctuated"] = output
            except Exception as err:
                LOG.warning("Punctuation batch failed lang={} size={}: {}", lang, len(texts), err)
                for segment, text in batch:
                    segment["text_punctuated"] = punctuate_text(
                        text,
                        lang,
                        device,
                        hf_model_dir=hf_model_dir,
                    )

    _write_json(json_path, data)
    LOG.info("Updated {}", json_path)


__all__ = [
    "MODEL_MAP",
    "process_file",
    "punctuate_text",
]
