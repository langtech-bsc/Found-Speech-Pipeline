#!/usr/bin/env python3
"""
generate_final_data.py
======================
Forced alignment + ASR enrichment for **one** audio-transcript pair.

CLI wrapper for fsp.core.alignment.generate_final_data

- Segments are language-detected first.
- For each language we load one model at a time -> transcribe -> unload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from fsp.utils.paths import (
    HF_MODEL_DIR_ENV_VAR,
    LID_MODEL_PATH_ENV_VAR,
    LOG_DIR,
    NEMO_MODEL_DIR_ENV_VAR,
)

# Setup loguru to also write to file
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "output.log", level="INFO")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Generate word-level aligned JSON (one input-id)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-id", required=True, help="YouTube video-id to process")
    p.add_argument(
        "--lang",
        choices=("ca", "es"),
        default="ca",
        help="Primary language (only its CTC model is loaded)",
    )
    p.add_argument(
        "--output",
        metavar="NAME.json",
        help="Custom JSON name (default: final_output_<input-id>.json)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Run ASR on cuda / cpu",
    )
    p.add_argument(
        "--lid-model-path",
        type=Path,
        help=f"Path to lid.176.bin (default: ${LID_MODEL_PATH_ENV_VAR} or utils/models/lid.176.bin)",
    )
    p.add_argument(
        "--nemo-model-dir",
        type=Path,
        help=f"Directory containing local NeMo checkpoints (default: ${NEMO_MODEL_DIR_ENV_VAR} or utils/models/nemo)",
    )
    p.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR} or utils/models/huggingface)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here to avoid loading heavy deps before arg parsing
    from fsp.core.alignment import generate_final_data

    output_name = args.output or f"final_output_{args.input_id}.json"
    try:
        generate_final_data(
            input_id=args.input_id,
            lang=args.lang,
            output_name=output_name,
            device=args.device,
            lid_model_path=args.lid_model_path,
            nemo_model_dir=args.nemo_model_dir,
            hf_model_dir=args.hf_model_dir,
        )
    except Exception as e:
        logger.error("Fatal error: {}", e)
        raise Exception(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise Exception(e)
