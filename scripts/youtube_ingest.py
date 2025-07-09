#!/usr/bin/env python3
"""
youtube_ingest.py  –  ingest audio + transcript *if* licence is acceptable.

Workflow
--------
1) Inspect the video’s licence (no download needed).
2) If allowed:
       • Download audio as WAV   →  ./ingestion/<video_id>.wav
       • Try captions in ca / es / en, else Whisper fallback
       • Emit TSV row            →  ./ingestion/<video_id>.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import pipeline

try:
    import yt_dlp as youtube_dl
except ImportError:
    print("Please install yt-dlp:  pip install -U yt-dlp")
    sys.exit(1)

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

# Configuration
INGEST_DIR = (Path.cwd() / "ingestion").resolve()
INGEST_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "openai/whisper-large-v3"

# Licence utilities
def detect_license(
    youtube_url: str,
) -> Tuple[str, str]:
    """
    Return (licence_code, video_id).

    *Uses yt-dlp with download=False; costs no bandwidth.*
    Licence codes normalised to match YouTube Data API values:
        • creativeCommon
        • youtube          (a.k.a. “Standard YouTube Licence”)
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # speed-up: don’t fetch every format if we only want meta
        "extract_flat": True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    video_id = info.get("id")
    raw_license = (info.get("license") or "").lower()

    if "creative commons" in raw_license:
        return "creativeCommon", video_id
    # yt-dlp returns "" (empty) for Standard licence; that’s fine.
    return "youtube", video_id


# Audio download
def download_audio_as_wav(youtube_url: str, video_id: str) -> Path:
    """Download only the audio stream and convert it to WAV."""
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": str(INGEST_DIR / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        # We already know the ID → avoid a second metadata fetch
        "forceid": True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    wav_path = INGEST_DIR / f"{video_id}.wav"
    if not wav_path.is_file():
        raise RuntimeError(f"Download failed: {wav_path} not found")
    return wav_path


# Captions / Whisper
def fetch_youtube_captions(video_id: str) -> str:
    """Try official captions (ca → es → en)."""
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    for lang in ("ca", "es", "en"):
        try:
            transcript = transcript_list.find_transcript([lang])
            return " ".join(item["text"] for item in transcript.fetch()).strip()
        except NoTranscriptFound:
            continue
    raise NoTranscriptFound


def transcribe_via_whisper_local(wav_path: Path) -> str:
    """Run an offline Whisper model."""
    print(f"🔈  Transcribing locally with Whisper: {wav_path.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        chunk_length_s=30,
    )
    return whisper_pipe(str(wav_path), batch_size=8)["text"].strip()


# CLI helpers
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube video if its licence is acceptable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("youtube_url", help="Full YouTube URL or video ID")
    parser.add_argument(
        "--reject-license",
        "-R",
        default="",
        metavar="LIST",
        help="Comma-separated list of licence codes to block "
             "(e.g. youtube,creativeCommon). "
             "Case-insensitive. Default: allow all.",
    )
    return parser.parse_args()

# Main entry point
def main() -> None:
    args = parse_args()
    rejected = {lic.lower() for lic in args.reject_license.split(",") if lic.strip()}

    # 0) Licence inspection
    try:
        licence_code, video_id = detect_license(args.youtube_url)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Could not determine licence: {exc}")
        sys.exit(1)

    print(f"🎫  Licence for {video_id}: {licence_code}")

    if licence_code.lower() in rejected:
        print(f"🚫 Ingestion blocked (licence '{licence_code}' "
              f"is in --reject-license list).")
        sys.exit(0)                       # Exit *without* error: intentional skip.

    # 1) Download audio
    try:
        wav_path = download_audio_as_wav(args.youtube_url, video_id)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to download audio: {exc}")
        sys.exit(1)

    # 2) Captions or Whisper
    try:
        transcript_text = fetch_youtube_captions(video_id)
        print("✓ Retrieved official captions")
    except Exception:
        print("⚠️  No captions found → Whisper fallback")
        try:
            transcript_text = transcribe_via_whisper_local(wav_path)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Whisper transcription failed: {exc}")
            sys.exit(1)

    # 3) Write TSV
    tsv_path = INGEST_DIR / f"{video_id}.tsv"
    with tsv_path.open("w", encoding="utf-8") as fp:
        fp.write(f"{wav_path.resolve()}\t{transcript_text}\n")

    print(f"✅ Wrote TSV → {tsv_path}")


if __name__ == "__main__":
    main()