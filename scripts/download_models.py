#!/usr/bin/env python3
"""
download_models.py
==================
Download all ASR models needed by the FSP pipeline to utils/models/
so the pipeline can run fully offline.

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
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model catalogue  (mirrors MODELS_BY_LANG in generate_final_data.py)
# ---------------------------------------------------------------------------

# NeMo CTC models downloaded from NVIDIA NGC via nemo_asr
NEMO_CTC_MODELS = {
    "ca": "stt_ca_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
}

# Models downloaded via HuggingFace Hub
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


def download_nemo_ctc(model_name: str, out_dir: Path) -> None:
    """Download a NeMo CTC model from NGC and save the .nemo file locally."""
    out_path = out_dir / f"{model_name}.nemo"
    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    import nemo.collections.asr as nemo_asr

    print(f"  ↓ Downloading NeMo model: {model_name} ...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)

    # Save to our local directory
    model.save_to(str(out_path))
    print(f"  ✓ Saved to: {out_path}")

    # Clean up RAM
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download_hf_model(repo_id: str, hf_home: Path) -> None:
    """Download a HuggingFace model snapshot."""
    os.environ["HF_HOME"] = str(hf_home)

    from huggingface_hub import snapshot_download

    cache_name = repo_id.replace("/", "--")
    model_dir = hf_home / "hub" / f"models--{cache_name}"
    if model_dir.exists():
        print(f"  ✓ Already cached: {repo_id}")
        return

    print(f"  ↓ Downloading HuggingFace model: {repo_id} ...")
    snapshot_download(repo_id)
    print(f"  ✓ Cached: {repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ASR models for offline use")
    ap.add_argument(
        "--lang",
        choices=("ca", "es", "all"),
        default="all",
        help="Which language models to download (default: all)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("utils/models"),
        help="Root output directory (default: utils/models)",
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

    print(f"\n✅ All models downloaded to {args.out_dir.resolve()}")
    print("Mount this directory into the container at /app/utils/models for offline use.")


if __name__ == "__main__":
    main()
