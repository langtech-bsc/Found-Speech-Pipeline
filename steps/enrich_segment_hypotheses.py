#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fsp.core.enrichment import choose_device, enrich_json
from fsp.utils.logging import build_run_label, setup_logging
from fsp.utils.paths import HF_MODEL_DIR_ENV_VAR


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
    parser.add_argument("--pipeline-batch-size", type=int, default=8)
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        help=f"Directory containing the HuggingFace cache root (default: ${HF_MODEL_DIR_ENV_VAR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = args.input_json.resolve()
    output_json = (args.output_json or input_json).resolve()
    setup_logging(run_label=build_run_label("gl-enrichment", input_json.stem))
    enrich_json(
        input_json=input_json,
        output_json=output_json,
        langs=args.langs,
        model_names=args.models,
        device=choose_device(args.device),
        overwrite_existing=args.overwrite_existing,
        limit=args.limit,
        hf_model_dir=args.hf_model_dir,
        pipeline_batch_size=args.pipeline_batch_size,
    )


if __name__ == "__main__":
    main()
