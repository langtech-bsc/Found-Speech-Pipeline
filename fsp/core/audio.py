"""
Audio normalization and duration filtering functions.

This module contains audio processing functions migrated from:
- scripts/normalize_audio.py
- scripts/duration_filter.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from fsp.utils.paths import FFMPEG_CMD, INGESTION_DIR, NORM_DIR, OUT_DIR


def select_tsv(input_id: str) -> Path:
    """
    Return the normalized TSV for this input_id or raise an error.

    Args:
        input_id: The input identifier

    Returns:
        Path to the normalized TSV file

    Raises:
        FileNotFoundError: If no normalized TSV is found
    """
    mark = NORM_DIR / input_id / f"{input_id}_norm_mark.tsv"
    if mark.is_file():
        return mark

    norm = NORM_DIR / input_id / f"{input_id}_norm.tsv"
    if norm.is_file():
        return norm

    raise FileNotFoundError(f"No normalized TSV found for '{input_id}' in {NORM_DIR}")


def normalize_audio(
    input_id: str, ingestion_dir: Optional[Path] = None, output_dir: Optional[Path] = None
) -> Path:
    """
    Normalize audio to 16kHz mono WAV and generate metadata JSON.

    Args:
        input_id: The input identifier
        ingestion_dir: Directory containing source WAV files (default: INGESTION_DIR)
        output_dir: Directory for output files (default: OUT_DIR)

    Returns:
        Path to the output directory
    """
    ingestion_dir = ingestion_dir or INGESTION_DIR
    output_dir = output_dir or OUT_DIR

    wav_src = ingestion_dir / f"{input_id}.wav"
    tsv_src = select_tsv(input_id)

    df = pd.read_csv(tsv_src, sep="\t", header=None, dtype=str)
    if df.shape[1] < 3:
        raise ValueError(
            "TSV must contain at least 3 columns: wav_path, original_text, normalized_text"
        )

    original_text = df.iloc[0, 1].strip()
    normalized_text = df.iloc[0, 2].strip()

    # Transcode to canonical WAV
    wav_dst = output_dir / input_id / f"{input_id}.wav"
    wav_dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_CMD,
        "-y",
        "-i",
        str(wav_src),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(wav_dst),
    ]
    subprocess.run(cmd, check=True)

    # Calculate duration
    with wave.open(str(wav_dst), "rb") as wf:
        dur_s = wf.getnframes() / wf.getframerate()
    h, rem = divmod(int(dur_s), 3600)
    m, s = divmod(rem, 60)
    end_ts = f"{h:02d}:{m:02d}:{s:02d}"

    # Write metadata
    meta_path = output_dir / input_id / f"{input_id}_metadata.json"

    entry = {
        "audio_filepath": str(wav_dst.resolve()),
        "normalized_text": normalized_text,
        "original_text": original_text,
        "start": "00:00:00",
        "end": end_ts,
    }
    meta_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Normalized audio and metadata written to: {output_dir / input_id}")
    return output_dir / input_id


def filter_and_cleanup(json_path: str, min_dur: float = 2, max_dur: float = 30) -> None:
    """
    Filter segments by duration and cleanup files outside the range.

    - Loads the JSON at json_path.
    - For each segment in each top-level entry, computes duration = end - start.
    - If not (duration > min_dur and duration <= max_dur):
        - Deletes the file at segment_path (if it exists).
        - Omits that segment from the filtered JSON.
    - Overwrites the original JSON with the filtered version.

    Args:
        json_path: Path to the JSON file to filter
        min_dur: Minimum duration in seconds (default: 2)
        max_dur: Maximum duration in seconds (default: 30)
    """
    # Load the existing JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Iterate over each top-level item
    for item_key, item_content in data.items():
        if "results" not in item_content:
            continue

        filtered_results = []
        for segment in item_content["results"]:
            start = segment.get("start")
            end = segment.get("end")
            seg_path = segment.get("segment_path")

            # If either start or end is missing, keep the segment by default
            if start is None or end is None:
                filtered_results.append(segment)
                continue

            duration = end - start

            # Remove segments that do NOT satisfy min < duration <= max
            if not (duration > min_dur and duration <= max_dur):
                if seg_path and os.path.isfile(seg_path):
                    try:
                        os.remove(seg_path)
                        logger.info(f"Deleted file: {seg_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete {seg_path}: {e}")
                # Skip adding this segment to filtered_results
            else:
                # Duration is in the allowed range; keep it
                filtered_results.append(segment)

        # Replace the "results" list with the filtered one
        item_content["results"] = filtered_results

    # Overwrite the original JSON file with the filtered data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Filtering complete. Updated JSON written to: {json_path}")


__all__ = [
    "select_tsv",
    "normalize_audio",
    "filter_and_cleanup",
]
