#!/usr/bin/env python3
"""
generate_final_data.py
======================
Forced alignment + ASR enrichment for **one** audio-transcript pair.

CLI wrapper for fsp.core.alignment.generate_final_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fsp.utils.paths import (  # noqa: E402
    HF_MODEL_DIR_ENV_VAR,
    LID_MODEL_PATH_ENV_VAR,
    LOG_DIR,
    NEMO_MODEL_DIR_ENV_VAR,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "output.log", level="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Generate word-level aligned JSON (one input-id)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-id", required=True, help="YouTube video-id to process")
    parser.add_argument(
        "--lang",
        choices=("ca", "es", "eu", "gl"),
        default="ca",
        help="Primary language (only its CTC model is loaded)",
    )
    parser.add_argument(
        "--output",
        metavar="NAME.json",
        help="Custom JSON name (default: final_output_<input-id>.json)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Run ASR on cuda / cpu",
    )
    parser.add_argument(
        "--lid-model-path",
        type=Path,
        help=f"Path to lid.176.bin (default: ${LID_MODEL_PATH_ENV_VAR} or utils/models/lid.176.bin)",
    )
    parser.add_argument(
        "--nemo-model-dir",
        type=Path,
        help=f"Directory containing local NeMo checkpoints (default: ${NEMO_MODEL_DIR_ENV_VAR})",
    )
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal error: {}", exc)
        raise


if __name__ == "__main__":
    main()
