#!/usr/bin/env python3
"""
pipeline_service.py – orchestrates the FSP pipeline
(starting from existing WAV + TSV files)

Modes
-----
- Single:       --input-id <id>
- File batch:   --input-id-file <path>  (file with one ID per line)
- Full batch:   (no --input-id, no --input-id-file) -> auto-detect all valid pairs in ingestion/
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Import the Pipeline class from fsp package
from fsp.pipeline import Pipeline
from fsp.utils.paths import (HF_MODEL_DIR_ENV_VAR, INGESTION_DIR,
                             LID_MODEL_PATH_ENV_VAR, NEMO_MODEL_DIR_ENV_VAR,
                             OUTPUT_SEGMENT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(
        "Run FSP pipeline from existing WAV+TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input-id",
        help="Process a single audio-transcript pair (WAV+TSV filename stem in ingestion)",
    )
    ap.add_argument(
        "--input-id-file",
        type=Path,
        help="Path to file with one input ID per line (for batch processing a subset)",
    )
    ap.add_argument("--lang", choices=("ca", "es"), default="ca")
    ap.add_argument(
        "--max-duration",
        type=float,
        default=30,
        help="Maximum segment duration in seconds",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=2,
        help="Minimum segment duration in seconds",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Run ASR on cuda / cpu",
    )
    ap.add_argument(
        "--lid-model-path",
        type=Path,
        help=f"Path to lid.176.bin (default: ${LID_MODEL_PATH_ENV_VAR} or utils/models/lid.176.bin)",
    )
    ap.add_argument(
        "--nemo-model-dir",
        type=Path,
        help=f"Directory containing local NeMo checkpoints (default: ${NEMO_MODEL_DIR_ENV_VAR} or utils/models/nemo)",
    )
    ap.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR} or utils/models/huggingface)",
    )

    args = ap.parse_args()

    if args.input_id and args.input_id_file:
        raise ValueError("Cannot use both --input-id and --input-id-file")

    if not INGESTION_DIR.exists():
        raise FileNotFoundError("ingestion/ directory not found")

    # Create pipeline instance
    pipeline = Pipeline(
        lang=args.lang,
        device=args.device,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        lid_model_path=args.lid_model_path,
        nemo_model_dir=args.nemo_model_dir,
        hf_model_dir=args.hf_model_dir,
    )

    # -------------------------------
    # SINGLE MODE
    # -------------------------------
    if args.input_id:
        print("\n" + "=" * 70)
        print(f"Processing single audio-transcript pair: {args.input_id}")
        print("=" * 70)

        output_path = pipeline.run_all(args.input_id)
        print(f"\nPipeline finished. Final JSON file: {output_path}")

    # -------------------------------
    # FILE BATCH MODE (--input-id-file)
    # -------------------------------
    elif args.input_id_file:
        if not args.input_id_file.exists():
            raise FileNotFoundError(f"Input ID file not found: {args.input_id_file}")

        input_ids = [
            line.strip()
            for line in args.input_id_file.read_text().splitlines()
            if line.strip()
        ]
        if not input_ids:
            raise ValueError(f"No IDs found in {args.input_id_file}")

        print(f"\nProcessing {len(input_ids)} IDs from {args.input_id_file}")
        pipeline.run_batch(input_ids=input_ids)
        print(f"\nPipeline finished. Final JSON files are in {OUTPUT_SEGMENT_DIR}")

    # -------------------------------
    # FULL BATCH MODE
    # -------------------------------
    else:
        pipeline.run_batch()
        print(f"\nPipeline finished. Final JSON files are in {OUTPUT_SEGMENT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        raise Exception(e)
