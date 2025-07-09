#!/usr/bin/env python3
"""
ingest_single.py
================
Ingest *one* WAV+TSV pair into the FSP folder structure.
"""

from __future__ import annotations
import argparse, json, shutil, subprocess, sys, uuid
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent      # project root
INGEST_DIR  = ROOT / "ingestion"
OUT_SEG_DIR = ROOT / "inputs" / "segments"
FFMPEG_CMD  = "ffmpeg"                                    # on $PATH

def select_tsv() -> Path:
    """Return the *best* TSV file in ingestion/ or exit."""
    # 1st choice: the cleaned, dedup-marked file
    best = sorted(INGEST_DIR.glob("*_norm_mark.tsv"))
    if best:
        if len(best) > 1:
            print(f"⚠️  Multiple *_norm_mark.tsv found – using {best[0].name}")
        return best[0]

    # 2nd choice: any other .tsv
    others = sorted(INGEST_DIR.glob("*.tsv"))
    if not others:
        sys.exit("❌  No .tsv file found in ingestion/")
    if len(others) > 1:
        print(f"⚠️  {len(others)} TSVs found – using {others[0].name}")
    return others[0]


def select_wav() -> Path:
    """Return the newest WAV in ingestion/ or exit."""
    wavs = sorted(INGEST_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        sys.exit("❌  No .wav file found in ingestion/")
    if len(wavs) > 1:
        print(f"⚠️  {len(wavs)} WAVs found – using most recent: {wavs[0].name}")
    return wavs[0]

def ingest(session_id: str, speaker: str | None = None) -> None:
    wav_src = select_wav()
    tsv_src = select_tsv()

    try:
        _, transcription = tsv_src.read_text(encoding="utf-8").split("\t", 1)
    except ValueError:
        sys.exit("❌  TSV must contain <anything><TAB><full transcription>")
    transcription = transcription.strip()

    # prepare per-video folders
    seg_dir   = OUT_SEG_DIR / session_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    # purge previous run of THIS video only
    shutil.rmtree(seg_dir, ignore_errors=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # transcode to canonical WAV
    wav_dst = seg_dir / f"{session_id}_{uuid.uuid4().hex[:8]}.wav"
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

    # metadata + NeMo manifest
    manifest_dir = ROOT / "inputs" / "manifest"; manifest_dir.mkdir(parents=True, exist_ok=True)
    meta_path    = seg_dir / f"{session_id}_metadata.json"
    manifest_fp  = manifest_dir / f"{session_id}_manifest.json"

    entry = {
        "audio_filepath":     str(wav_dst.resolve()),
        "text":              [[speaker or "unknown", transcription]],
        "start":              "00:00:00",
        "end":                end_ts,
        "path_segment_audio": str(wav_dst.resolve()),
    }
    meta_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_fp.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")

    print(f"✓  Ingested → {seg_dir.relative_to(ROOT)}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True, help="YouTube video-id")
    p.add_argument("--speaker", help="Speaker name (optional)")
    args = p.parse_args()
    ingest(args.session_id, args.speaker)

if __name__ == "__main__":
    main()