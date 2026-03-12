#!/usr/bin/env python3
"""
normalize_audio.py
==================
Normalize audio to 16kHz mono WAV + generate metadata JSON.

CLI wrapper for fsp.core.audio.normalize_audio
"""

from __future__ import annotations

import argparse
import sys

# Import core logic from fsp package
from fsp.core.audio import normalize_audio, select_tsv


def main() -> None:
    p = argparse.ArgumentParser(
        description="Normalize audio to 16kHz mono WAV + generate metadata JSON"
    )
    p.add_argument(
        "--input-id", required=True, help="audio-transcript pair id (WAV + TSV filename stem)"
    )
    args = p.parse_args()

    try:
        normalize_audio(args.input_id)
    except FileNotFoundError as e:
        sys.exit(str(e))
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
