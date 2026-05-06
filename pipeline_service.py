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
import os
from pathlib import Path

from loguru import logger

from fsp.pipeline import Pipeline
from fsp.utils.logging import build_run_label, build_run_log_dir, setup_logging
from fsp.utils.paths import (
    HF_MODEL_DIR_ENV_VAR,
    INGESTION_DIR,
    LID_MODEL_PATH_ENV_VAR,
    NEMO_MODEL_DIR_ENV_VAR,
    ROVER_DIR,
)


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
    ap.add_argument("--lang", choices=("ca", "es", "eu", "gl"), default="ca")
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
        "--asr-batch-size",
        type=int,
        default=8,
        help="Batch size for segment-level ASR inference",
    )
    ap.add_argument(
        "--lid-model-path",
        type=Path,
        help=f"Path to lid.176.bin (default: ${LID_MODEL_PATH_ENV_VAR} or utils/models/lid.176.bin)",
    )
    ap.add_argument(
        "--nemo-model-dir",
        type=Path,
        help=f"Directory containing local NeMo checkpoints (default: ${NEMO_MODEL_DIR_ENV_VAR})",
    )
    ap.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR})",
    )
    ap.add_argument(
        "--skip-gl-extra-asr",
        action="store_true",
        help="Skip the GL-specific Apptainer enrichment step",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"),
        help="Application log level",
    )

    args = ap.parse_args()

    if args.input_id and args.input_id_file:
        raise ValueError("Cannot use both --input-id and --input-id-file")

    if not INGESTION_DIR.exists():
        raise FileNotFoundError("ingestion/ directory not found")

    run_scope = args.input_id or (args.input_id_file.stem if args.input_id_file else "batch")
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    run_label = build_run_label("pipeline", run_scope, args.lang, slurm_job_id)
    run_log_dir = build_run_log_dir(run_label)
    app_log_path = setup_logging(log_level=args.log_level, run_label=run_label, log_dir=run_log_dir)

    pipeline = Pipeline(
        lang=args.lang,
        device=args.device,
        asr_batch_size=args.asr_batch_size,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        lid_model_path=args.lid_model_path,
        nemo_model_dir=args.nemo_model_dir,
        hf_model_dir=args.hf_model_dir,
        enable_gl_extra_asr=not args.skip_gl_extra_asr,
    )
    pipeline.set_run_context(run_label=run_label, run_log_dir=run_log_dir)

    if args.input_id:
        logger.info("Processing input_id={} lang={} app_log={}", args.input_id, args.lang, app_log_path)

        output_path = pipeline.run_all(args.input_id)
        logger.info("Pipeline finished. Final JSON file: {}", output_path)
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

        logger.info(
            "Processing {} IDs from {} app_log={}",
            len(input_ids),
            args.input_id_file,
            app_log_path,
        )
        pipeline.run_batch(input_ids=input_ids)
        logger.info("Pipeline finished. Final JSON files are in {}", ROVER_DIR)
    else:
        pipeline.run_batch()
        logger.info("Pipeline finished. Final JSON files are in {}", ROVER_DIR)


if __name__ == "__main__":
    main()
