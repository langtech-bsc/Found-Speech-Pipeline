#!/usr/bin/env python3
"""
ingest_single.py
================
Ingest *one* WAV+TSV pair into the FSP folder structure.
"""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent      # project root
INGEST_DIR  = ROOT / "ingestion"
NORM_DIR    = ROOT / "inputs" / "normalized"
OUT_DIR = ROOT / "inputs" / "normalized"
FFMPEG_CMD  = "ffmpeg"                                    # on $PATH

def select_tsv(input_id: str) -> Path:
    """Return the normalized TSV for this input_id or exit."""

    mark = NORM_DIR / input_id / f"{input_id}_norm_mark.tsv"
    if mark.is_file():
        return mark

    norm = NORM_DIR / input_id / f"{input_id}_norm.tsv"
    if norm.is_file():
        return norm

    sys.exit(f"❌  No normalized TSV found for '{input_id}' in {NORM_DIR}")


def normalize_audio(input_id: str | None = None) -> None:
    wav_src = INGEST_DIR / f"{input_id}.wav"
    tsv_src = select_tsv(input_id)

    df = pd.read_csv(tsv_src, sep="\t", header=None, dtype=str)
    if df.shape[1] < 3:
        sys.exit("❌ TSV must contain at least 3 columns: wav_path, original_text, normalized_text")
    
    original_text   = df.iloc[0,1].strip()
    normalized_text = df.iloc[0,2].strip()

    # transcode to canonical WAV
    wav_dst = OUT_DIR / input_id / f"{input_id}.wav"
    cmd = [
        FFMPEG_CMD, "-y",
        "-i", str(wav_src),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_dst),
    ]
    subprocess.run(cmd, check=True)

    # duration
    import wave
    with wave.open(str(wav_dst), "rb") as wf:
        dur_s = wf.getnframes() / wf.getframerate()
    h, rem = divmod(int(dur_s), 3600); m, s = divmod(rem, 60)
    end_ts = f"{h:02d}:{m:02d}:{s:02d}"

    # metadata 
    meta_path    = OUT_DIR / input_id / f"{input_id}_metadata.json"

    entry = {
        "audio_filepath":     str(wav_dst.resolve()),
        "normalized_text":    normalized_text,
        "original_text":      original_text,
        "start":              "00:00:00",
        "end":                end_ts,
    }
    meta_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓  Normalized audio and metadata written to → {OUT_DIR.relative_to(ROOT) / input_id}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-id", required=True, help="audio-transcript pair id (WAV + TSV filename stem)")
    args = p.parse_args()
    normalize_audio(args.input_id)

if __name__ == "__main__":
    main()