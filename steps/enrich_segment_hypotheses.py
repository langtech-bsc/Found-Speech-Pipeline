#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_paths import configure_model_env
from scripts.clean_and_expand import clean_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hf_enrich")

MODELS_ROOT = configure_model_env()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    repo_name: str
    language: str
    enabled_by_default: bool = True

    @property
    def repo_path(self) -> Path:
        return MODELS_ROOT / self.repo_name


HF_ENRICH_MODELS: dict[str, tuple[ModelSpec, ...]] = {
    "gl": (
        ModelSpec(
            name="whisper_large_v3_turbo_gl_v1_0",
            kind="whisper_seq2seq",
            repo_name="whisper-large-v3-turbo-gl-v1.0",
            language="gl",
        ),
        ModelSpec(
            name="w2v_bert_2_0_gl",
            kind="wav2vec2_bert_ctc",
            repo_name="w2v-bert-2.0-gl",
            language="gl",
            enabled_by_default=False,
        ),
        ModelSpec(
            name="phi_4_multimodal_instruct_gl_v1_0",
            kind="phi4_multimodal_audio",
            repo_name="phi-4-multimodal-instruct-gl-v1.0",
            language="gl",
            enabled_by_default=False,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add modern HF hypotheses to an existing final_output JSON.",
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--langs", nargs="+", default=["gl"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def choose_device(raw: str) -> str:
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_audio_path(raw_path: str, json_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    return (json_path.parent / path).resolve()


def iter_segments(data: dict[str, Any], keep_langs: set[str]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for block in data.values():
        for segment in block.get("results", []):
            if segment.get("language") in keep_langs:
                segments.append(segment)
    return segments


def selected_specs(langs: list[str], model_names: list[str] | None) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for lang in langs:
        specs.extend(HF_ENRICH_MODELS.get(lang, ()))

    deduped: dict[str, ModelSpec] = {spec.name: spec for spec in specs}
    if model_names:
        missing = [name for name in model_names if name not in deduped]
        if missing:
            raise SystemExit(f"Unknown enrichment models: {', '.join(sorted(missing))}")
        return [deduped[name] for name in model_names]
    return [spec for spec in deduped.values() if spec.enabled_by_default]


def unload_model(model: Any) -> None:
    try:
        del model
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def sanitize_local_model_dir(repo_path: Path) -> Path:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in repo_path.name).strip("_")
    alias_root = Path(tempfile.gettempdir()) / "fsp_hf_model_aliases"
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_path = alias_root / safe_name
    if alias_path.exists():
        return alias_path
    alias_path.symlink_to(repo_path.resolve(), target_is_directory=True)
    return alias_path


def load_whisper_seq2seq(repo_path: Path, device: str) -> Any:
    processor = AutoProcessor.from_pretrained(repo_path, local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo_path,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if device == "cuda":
        model = model.to(device)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if device == "cuda" else -1,
    )
    return {"pipe": pipe}


def transcribe_whisper_seq2seq(model: dict[str, Any], audio_path: Path, lang: str) -> str:
    out = model["pipe"](
        str(audio_path),
        generate_kwargs={"task": "transcribe", "language": lang},
    )
    if isinstance(out, dict):
        return str(out.get("text", "")).strip()
    return str(out).strip()


def load_wav2vec2_bert_ctc(repo_path: Path, device: str) -> Any:
    processor = AutoProcessor.from_pretrained(repo_path, local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCTC.from_pretrained(
        repo_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if device == "cuda":
        model = model.to(device)
    model.eval()
    return {"model": model, "processor": processor}


def transcribe_wav2vec2_bert_ctc(model: dict[str, Any], audio_path: Path, _lang: str) -> str:
    audio, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    processor = model["processor"]
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    device = next(model["model"].parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model["model"](**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
    return str(processor.batch_decode(pred_ids)[0]).strip()


def load_phi4_multimodal_audio(repo_path: Path, device: str) -> Any:
    alias_path = sanitize_local_model_dir(repo_path)
    processor = AutoProcessor.from_pretrained(alias_path, trust_remote_code=True, local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        alias_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=dtype,
        local_files_only=True,
    )
    if device == "cuda":
        model = model.to(device)
    model.eval()
    return {"model": model, "processor": processor}


def transcribe_phi4_multimodal_audio(model: dict[str, Any], audio_path: Path, lang: str) -> str:
    audio, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    processor = model["processor"]
    user_msg = {
        "role": "user",
        "content": "<|audio_1|>\nTranscribe the audio clip into Galician text.",
    }
    prompt = processor.tokenizer.apply_chat_template(
        [user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        audios=[(audio, sample_rate)],
        return_tensors="pt",
    )
    device = next(model["model"].parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_ids = inputs.get("input_ids")
    prompt_len = input_ids.shape[-1] if input_ids is not None else 0
    with torch.inference_mode():
        output_ids = model["model"].generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_logits_to_keep=1,
        )
    new_tokens = output_ids[:, prompt_len:] if prompt_len else output_ids
    return str(processor.batch_decode(new_tokens, skip_special_tokens=True)[0]).strip()


LOADERS = {
    "whisper_seq2seq": load_whisper_seq2seq,
    "wav2vec2_bert_ctc": load_wav2vec2_bert_ctc,
    "phi4_multimodal_audio": load_phi4_multimodal_audio,
}

TRANSCRIBERS = {
    "whisper_seq2seq": transcribe_whisper_seq2seq,
    "wav2vec2_bert_ctc": transcribe_wav2vec2_bert_ctc,
    "phi4_multimodal_audio": transcribe_phi4_multimodal_audio,
}


def should_skip_segment(segment: dict[str, Any], spec: ModelSpec, overwrite: bool) -> bool:
    key = f"pred_text_{spec.name}"
    return bool(segment.get(key)) and not overwrite


def enrich_json(
    input_json: Path,
    output_json: Path,
    langs: list[str],
    model_names: list[str] | None,
    device: str,
    overwrite_existing: bool,
    limit: int | None,
) -> None:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    segments = iter_segments(data, set(langs))
    if limit is not None:
        segments = segments[:limit]

    specs = selected_specs(langs, model_names)
    LOG.info("Eligible segments: %s", len(segments))
    LOG.info("Selected models: %s", ", ".join(spec.name for spec in specs) or "<none>")

    for spec in specs:
        repo_path = spec.repo_path
        if not repo_path.exists():
            LOG.warning("Model path not found for %s: %s", spec.name, repo_path)
            continue

        model = None
        try:
            LOG.info("Loading %s from %s", spec.name, repo_path)
            model = LOADERS[spec.kind](repo_path, device)
            transcriber = TRANSCRIBERS[spec.kind]

            for segment in segments:
                if segment.get("language") != spec.language:
                    continue
                if should_skip_segment(segment, spec, overwrite_existing):
                    continue

                audio_path = resolve_audio_path(segment["segment_path"], input_json)
                pred_key = f"pred_text_{spec.name}"
                norm_key = f"norm_text_{spec.name}"
                if not audio_path.exists():
                    LOG.warning("Missing segment audio for %s: %s", spec.name, audio_path)
                    segment[pred_key] = ""
                    segment[norm_key] = ""
                    continue

                try:
                    transcription = transcriber(model, audio_path, spec.language)
                    segment[pred_key] = transcription
                    segment[norm_key] = clean_text(transcription, spec.language, False, False)
                except Exception as err:  # noqa: BLE001
                    LOG.warning("%s failed on %s: %s", spec.name, audio_path, err)
                    segment[pred_key] = ""
                    segment[norm_key] = ""
        except Exception as err:  # noqa: BLE001
            LOG.warning("Could not load %s: %s", spec.name, err)
        finally:
            unload_model(model)

    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Updated JSON written to %s", output_json)


def main() -> None:
    args = parse_args()
    input_json = args.input_json.resolve()
    output_json = (args.output_json or input_json).resolve()
    enrich_json(
        input_json=input_json,
        output_json=output_json,
        langs=args.langs,
        model_names=args.models,
        device=choose_device(args.device),
        overwrite_existing=args.overwrite_existing,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
