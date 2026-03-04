#!/usr/bin/env python3
"""
generate_final_data.py
======================
Forced alignment + ASR enrichment for **one** audio-transcript pair.

CLI wrapper for fsp.core.alignment.generate_final_data

• Segments are language-detected first.
• For each language we load *one* model at a time → transcribe → unload.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
LOG_DIR = Path("inputs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "output.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate word-level aligned JSON (one input-id)")
    p.add_argument("--input-id", required=True,
                   help="YouTube video-id to process")
    p.add_argument("--lang", choices=("ca", "es"), default="ca",
                   help="Primary language (only its CTC model is loaded)")
    p.add_argument("--output", metavar="NAME.json",
                   help="Custom JSON name (default: final_output_<input-id>.json)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="Run ASR on cuda / cpu (default auto)")
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
        )
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
