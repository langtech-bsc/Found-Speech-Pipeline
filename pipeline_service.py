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
import argparse, subprocess, sys
from pathlib import Path
from typing import List

# project paths
ROOT          = Path(__file__).resolve().parent
SCRIPTS_DIR   = ROOT / "scripts"
INGESTION_DIR = ROOT / "ingestion"
STEPS_DIR     = ROOT / "steps"
ROVER_DIR     = ROOT / "merged"
ROVER_DIR.mkdir(exist_ok=True, parents=True)

PY = sys.executable

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def run(label: str,
        cmd:  List[str | Path],
        cwd:  Path | None = None,
        env:  dict | None = None) -> None:
    txt = " ".join(str(c) for c in cmd)
    print(f"\n► {label}\n  $ {txt}")
    if subprocess.run(cmd, cwd=cwd, env=env).returncode:
        sys.exit(f"✖  {label} failed")

def find_valid_input_ids() -> list[str]:
    """
    Scan ingestion/ and return all valid audio-transcript pair IDs
    (that have BOTH .wav and .tsv files).
    """
    wav_ids = {p.stem for p in INGESTION_DIR.glob("*.wav")}
    tsv_ids = {p.stem for p in INGESTION_DIR.glob("*.tsv")}

    valid_ids = sorted(wav_ids & tsv_ids)

    if not valid_ids:
        print("⚠️  No valid (.wav + .tsv) pairs found in ingestion/")
    else:
        print(f"🔎 Found {len(valid_ids)} valid input pair(s)")

    return valid_ids



# -------------------------------------------------
# Pipeline
# -------------------------------------------------

def process_existing_paired_input(input_id: str, lang: str, max_dur: float = 30) -> None:

    raw_tsv = INGESTION_DIR / f"{input_id}.tsv"
    raw_wav = INGESTION_DIR / f"{input_id}.wav"

    if not raw_tsv.exists() or not raw_wav.exists():
        raise RuntimeError(f"Missing pair for input ID: {input_id}")

    out_json_name = f"final_output_{input_id}.json"
    out_json_path = ROOT / "inputs" / "output_segment" / out_json_name
    
    # 1️⃣ Normalize TSV
    run("Normalise TSV",
        [PY, SCRIPTS_DIR / "normalize_tsv.py", raw_tsv, lang, "|"])

    # 2️⃣ Normalize audio + metadata 
    run("Ingest single",
        [PY, SCRIPTS_DIR / "normalize_audio.py", f"--input-id={input_id}"])

    # 3️⃣ Generate final data (aligner runs on CPU internally, but ASR can use GPU)
    run("Generate final data",
        [PY, STEPS_DIR / "generate_final_data.py",
         f"--input-id={input_id}", f"--lang={lang}", "--output", out_json_name])
    

    # 4️⃣ Duration filter
    run("Duration filter",
        [PY, SCRIPTS_DIR / "duration_filter.py", out_json_path, "--max", str(max_dur)])

    # 5️⃣ ROVER merge
    run("ROVER merge",
        [PY, SCRIPTS_DIR / "rover_merge.py",
         out_json_path, "--csv", "--plot", "--out-dir", ROVER_DIR])

    # 6️⃣ Punctuation & Capitalization restoration
    rover_json = ROVER_DIR / f"final_output_{input_id}.json"
    run("Punctuation & Capitalization",
        [PY, SCRIPTS_DIR / "punctuate.py", rover_json])


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Run FSP pipeline from existing WAV+TSV")
    ap.add_argument("--input-id", help="Process a single audio-transcript pair (WAV+TSV filename stem in ingestion)")
    ap.add_argument("--lang", choices=("ca", "es", "eu", "gl"), default="ca")
    ap.add_argument("--max-duration", type=float, default=30,
                    help="Maximum segment duration in seconds (default: 30)")

    args = ap.parse_args()

    if not INGESTION_DIR.exists():
        sys.exit("❌ ingestion/ directory not found")

    # -------------------------------
    # SINGLE MODE
    # -------------------------------
    if args.input_id:
        print("\n" + "═" * 70)
        print(f"Processing single audio-transcript pair: {args.input_id}")
        print("═" * 70)

        try:
            process_existing_paired_input(args.input_id, args.lang, args.max_duration)
        except Exception as e:
            sys.exit(f"❌ {e}")

    # -------------------------------
    # BATCH MODE
    # -------------------------------
    else:
        input_ids = find_valid_input_ids()

        for i, input_id in enumerate(input_ids, 1):
            print("\n" + "═" * 70)
            print(f"[{i}/{len(input_ids)}] Processing {input_id}")
            print("═" * 70)

            try:
                process_existing_paired_input(input_id, args.lang, args.max_duration)
            except Exception as e:
                print(f"❌ Failed: {input_id} → {e}")
                print("⚠️  Skipping...\n")

    print("\n✅ Pipeline finished – find JSON in",
          ROOT / "inputs" / "wordlevel_alignment")


if __name__ == "__main__":
    main()