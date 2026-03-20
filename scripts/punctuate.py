#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fsp.core.punctuation import process_file
from fsp.utils.paths import HF_MODEL_DIR_ENV_VAR


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR})",
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
        process_file(in_path, device_id, hf_model_dir=args.hf_model_dir)
    except Exception:
        logging.exception("Failed to process file: %s", in_path)
        return 1

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
