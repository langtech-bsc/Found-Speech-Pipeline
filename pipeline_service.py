#!/usr/bin/env python3
"""
pipeline_service.py – orchestrates the FSP pipeline
(starting from existing WAV + TSV files)

Modes
-----
• Single:  --input-id <id>
• Batch:   (no --input-id) → auto-detect all valid pairs in ingestion/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Import the Pipeline class from fsp package
from fsp.pipeline import Pipeline
from fsp.utils.paths import ROOT, INGESTION_DIR, ALIGN_DIR


def main() -> None:
    ap = argparse.ArgumentParser("Run FSP pipeline from existing WAV+TSV")
    ap.add_argument("--input-id", help="Process a single audio-transcript pair (WAV+TSV filename stem in ingestion)")
    ap.add_argument("--lang", choices=("ca", "es"), default="ca")
    ap.add_argument("--max-duration", type=float, default=30,
                    help="Maximum segment duration in seconds (default: 30)")
    ap.add_argument("--min-duration", type=float, default=2,
                    help="Minimum segment duration in seconds (default: 2)")

    args = ap.parse_args()

    if not INGESTION_DIR.exists():
        sys.exit("❌ ingestion/ directory not found")

    # Create pipeline instance
    pipeline = Pipeline(
        lang=args.lang,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )

    # -------------------------------
    # SINGLE MODE
    # -------------------------------
    if args.input_id:
        print("\n" + "═" * 70)
        print(f"Processing single audio-transcript pair: {args.input_id}")
        print("═" * 70)

        try:
            pipeline.run_all(args.input_id)
        except Exception as e:
            sys.exit(f"❌ {e}")

    # -------------------------------
    # BATCH MODE
    # -------------------------------
    else:
        pipeline.run_batch()

    print("\n✅ Pipeline finished – find JSON in", ALIGN_DIR)


if __name__ == "__main__":
    main()