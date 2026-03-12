#!/usr/bin/env python3
"""
download_models.py
==================
Download all ASR models needed by the FSP pipeline to utils/models/
so the pipeline can run fully offline.

CLI wrapper for fsp.utils.models functions.

Usage (from project root, inside the Docker container or with the venv active):
    python scripts/download_models.py --lang es      # Spanish only
    python scripts/download_models.py --lang ca      # Catalan only
    python scripts/download_models.py --lang all     # Both languages

Models are saved to:
    utils/models/nemo/          – NeMo .nemo checkpoints (NGC)
    utils/models/huggingface/   – HuggingFace model snapshots
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Import core logic from fsp package
from fsp.utils.models import download_hf_model, download_nemo_ctc
from fsp.utils.paths import MODEL_DIR_ENV_VAR, MODELS_DIR

# Model catalogue
NEMO_CTC_MODELS = {
    "ca": "stt_ca_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
}

HF_MODELS = {
    "ca": [
        "projecte-aina/whisper-large-v3-ca-3catparla",
        "langtech-veu/whisper-bsc-large-v3-cat",
        "langtech-veu/whisper-large-v3-ca-punctuated-3370h",
        "projecte-aina/stt_ca-es_conformer_transducer_large",
    ],
    "es": [
        "projecte-aina/parakeet-rnnt-1.1b_cv17_es_ep18_1270h",
        "nvidia/stt_es_conformer_transducer_large",
        "openai/whisper-large-v3",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ASR models for offline use",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--lang",
        choices=("ca", "es", "all"),
        default="all",
        help="Which language models to download (default: all)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Root output directory (default: ${MODEL_DIR_ENV_VAR} or utils/models)",
    )
    args = ap.parse_args()

    langs = ["ca", "es"] if args.lang == "all" else [args.lang]
    nemo_dir = args.out_dir / "nemo"
    hf_dir = args.out_dir / "huggingface"
    nemo_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model output directory: {args.out_dir.resolve()}")
    print(f"Languages: {', '.join(langs)}\n")

    # 1. NeMo CTC models
    for lang in langs:
        model_name = NEMO_CTC_MODELS[lang]
        print(f"[NeMo CTC] {model_name}")
        download_nemo_ctc(model_name, nemo_dir)

    # 2. HuggingFace models
    for lang in langs:
        for repo_id in HF_MODELS[lang]:
            print(f"[HuggingFace] {repo_id}")
            download_hf_model(repo_id, hf_dir)

    print(f"\nAll models downloaded to {args.out_dir.resolve()}")
    print("Mount this directory into the container at /app/utils/models for offline use.")


if __name__ == "__main__":
    main()
