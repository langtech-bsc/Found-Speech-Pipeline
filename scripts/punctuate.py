#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress warning from using HuggingFace pipelines sequentially
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

# Local imports
try:
    from clean_and_expand import clean_text
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from clean_and_expand import clean_text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model_paths import configure_model_env


MODELS_ROOT = str(configure_model_env())


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


MODELS: Dict[str, Any] = {}

MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "ca": ("catalan_capitalization_punctuation_restoration", "token-classification"),
    "es": ("spanish_capitalization_punctuation_restoration", "token-classification"),
    "gl": ("galician_capitalization_punctuation_restoration", "token-classification"),
    "eu": ("cap-punct-eu", "seq2seq"),
}

# Label map for UMUTeam token-classification models (based on config.id2label)
UMUTEAM_LABELS: Dict[str, Dict[str, str]] = {
    "l":  {"punct": "",  "case": "lower"},
    "u":  {"punct": "",  "case": "upper"},
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
    "O":  {"punct": "", "case": "same"},
}


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_local_model_dir(model_ref: str) -> Path:
    candidate = Path(model_ref).expanduser()
    if _is_model_dir(candidate):
        return candidate

    models_root = Path(MODELS_ROOT)
    direct_candidates = [
        models_root / model_ref,
        models_root / Path(model_ref).name,
    ]
    for direct in direct_candidates:
        if _is_model_dir(direct):
            return direct

    raise FileNotFoundError(
        f"Local punctuation model not found for '{model_ref}'. "
        f"Checked only direct model directories under {models_root}."
    )


def load_model_cached(lang: str, device: int):
    if lang in MODELS:
        return MODELS[lang]

    if lang not in MODEL_MAP:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(MODEL_MAP.keys())}")

    model_name, model_type = MODEL_MAP[lang]
    model_dir = resolve_local_model_dir(model_name)
    logging.info("Loading %s model from local path: %s (%s)", lang, model_dir, model_type)

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
        MODELS[lang] = (pipe, "token-classification")
        return MODELS[lang]

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
        MODELS[lang] = ((model, tokenizer), "seq2seq")
        return MODELS[lang]

    raise ValueError(f"Unknown model type '{model_type}' for language '{lang}'")


def apply_umuteam_punctuation(text: str, pipe) -> str:
    """
    Applies punctuation using UMUTeam token classification models.

    Uses per-token offsets to re-join subword tokens into words robustly.
    """
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
            # Gap in offsets => space => word boundary
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
    """Applies punctuation using a Seq2Seq model (HiTZ)."""
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


def punctuate_text(text: str, lang: str, device: int) -> str:
    """Orchestrates punctuation restoration for a given text and language."""
    model_obj, kind = load_model_cached(lang, device)

    if kind == "token-classification":
        return apply_umuteam_punctuation(text, model_obj)

    if kind == "seq2seq":
        return apply_seq2seq_punctuation(text, model_obj)

    # Should never happen
    return text


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse JSON: {path}") from e


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_video_data(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    # Assume one video_id, just take the first
    try:
        return next(iter(data.values()))
    except Exception:
        raise ValueError(f"Invalid JSON structure: {path}")


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

    for k, v in segment.items():
        if k.startswith("pred_text_") and isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def process_file(json_path: Path, device: int) -> None:
    """
    Process a single JSON file (one video per file).
    Requires:
    - "results" list
    - each segment has "language"

    Adds:
    - text_normalized
    - text_punctuated
    """
    logging.info("Processing %s ...", json_path)

    data = _read_json(json_path)
    video_data = _get_video_data(data, json_path)
    results = _get_results(video_data, json_path)

    for segment in results:
        lang = _require_language(segment, json_path)

        input_text = _extract_input_text(segment)
        if not input_text:
            logging.warning("No text found in segment, skipping.")
            continue

        segment["text_normalized"] = input_text
        segment["text_punctuated"] = punctuate_text(input_text, lang, device)

    _write_json(json_path, data)
    logging.info("Updated %s", json_path)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore punctuation and capitalization in ASR JSON output."
    )
    parser.add_argument("input", help="Path to a JSON file")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logging.error("Input file does not exist: %s", in_path)
        return 2
    if not in_path.is_file():
        logging.error("Input is not a file: %s", in_path)
        return 2
    if in_path.suffix.lower() != ".json":
        logging.error("Input must be a .json file: %s", in_path)
        return 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    device_id = 0 if device == "cuda" else -1

    try:
        process_file(in_path, device_id)
    except Exception:
        logging.exception("Failed to process file: %s", in_path)
        return 1

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
