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
from typing import Any, Callable, Mapping, Optional

import pandas as pd
import soundfile as sf
from loguru import logger

from fsp.utils.paths import FFMPEG_CMD, INGESTION_DIR, NORM_DIR, OUT_DIR
from fsp.utils.tsv import read_tsv_with_optional_header


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

    df = read_tsv_with_optional_header(tsv_src)
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
        "-hide_banner",
        "-loglevel",
        "error",
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
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "<empty>"
        raise RuntimeError(f"ffmpeg normalization failed for {input_id}: {stderr}")

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

    logger.info("Normalized audio and metadata written to {}", output_dir / input_id)
    return output_dir / input_id


def build_segment_output_path(output_dir: Path, base_name: str, start: float, end: float) -> Path:
    return output_dir / f"{base_name}_{start}_{end}.wav"


def materialize_segments(
    segments: list[dict[str, Any]],
    segment_sources: Mapping[int, tuple[Path, str]],
    output_dir: Path,
) -> int:
    """
    Materialize logical segments to WAV files only after they survive filtering.

    Args:
        segments: Surviving segment dictionaries
        segment_sources: Mapping of ``id(segment)`` to ``(source_audio_path, base_name)``
        output_dir: Directory where segment WAVs should be written

    Returns:
        Number of clips materialized (or reused if already present)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = 0
    grouped: dict[Path, list[tuple[dict[str, Any], str]]] = {}
    for segment in segments:
        source_info = segment_sources.get(id(segment))
        if source_info is None:
            raise KeyError(f"Missing source info for segment {segment}")
        source_audio_path, base_name = source_info
        grouped.setdefault(source_audio_path, []).append((segment, base_name))

    for source_audio_path, grouped_segments in grouped.items():
        with sf.SoundFile(str(source_audio_path), "r") as src_audio:
            sample_rate = src_audio.samplerate
            subtype = src_audio.subtype or "PCM_16"
            for segment, base_name in grouped_segments:
                start = float(segment["start"])
                end = float(segment["end"])
                out_path = build_segment_output_path(output_dir, base_name, start, end)
                if not out_path.is_file():
                    start_frame = max(0, int(round(start * sample_rate)))
                    end_frame = max(start_frame, int(round(end * sample_rate)))
                    frame_count = end_frame - start_frame
                    src_audio.seek(start_frame)
                    frames = src_audio.read(frame_count, dtype="int16", always_2d=False)
                    if getattr(frames, "shape", (len(frames),))[0] == 0:
                        raise RuntimeError(
                            f"Unable to materialize empty segment from {source_audio_path} "
                            f"for {base_name} {start:.2f}-{end:.2f}"
                        )
                    sf.write(str(out_path), frames, sample_rate, subtype=subtype)
                segment["segment_path"] = str(out_path)
                materialized += 1

    return materialized


def filter_segment_results(
    data: dict[str, Any],
    *,
    min_dur: float,
    max_dur: float,
    cleanup_files: bool = True,
    drop_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, dict[str, int]]:
    """
    Filter a ``final_output``-style payload by duration.

    Args:
        data: Parsed JSON payload with top-level items containing ``results``
        min_dur: Minimum duration in seconds
        max_dur: Maximum duration in seconds
        cleanup_files: Whether to delete referenced ``segment_path`` files
        drop_callback: Optional callback for structured drop records

    Returns:
        ``(dropped_count, dropped_reasons)``
    """
    dropped_count = 0
    dropped_reasons: dict[str, int] = {}
    for item_content in data.values():
        if "results" not in item_content:
            continue

        filtered_results = []
        for segment in item_content["results"]:
            start = segment.get("start")
            end = segment.get("end")
            seg_path = segment.get("segment_path")

            if start is None or end is None:
                filtered_results.append(segment)
                continue

            duration = end - start
            if not (duration > min_dur and duration <= max_dur):
                dropped_count += 1
                reason = "duration_out_of_range"
                dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
                logger.warning(
                    "Dropping segment during duration filter: {:.2f}-{:.2f} dur={:.2f}s outside ({:.2f}, {:.2f}] | path={}",
                    start,
                    end,
                    duration,
                    min_dur,
                    max_dur,
                    seg_path or "<missing>",
                )
                if drop_callback is not None:
                    drop_callback(
                        {
                            "stage": "duration_filter",
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "text": segment.get("normalized_text", ""),
                            "reason": reason,
                            "duration": round(duration, 2),
                            "minimum_duration": min_dur,
                            "maximum_duration": max_dur,
                            "segment_path": seg_path,
                        }
                    )
                if cleanup_files and seg_path and os.path.isfile(seg_path):
                    try:
                        os.remove(seg_path)
                        logger.info(f"Deleted file: {seg_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete {seg_path}: {e}")
            else:
                filtered_results.append(segment)

        item_content["results"] = filtered_results

    return dropped_count, dropped_reasons


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

    dropped_count, dropped_reasons = filter_segment_results(
        data,
        min_dur=min_dur,
        max_dur=max_dur,
        cleanup_files=True,
    )

    # Overwrite the original JSON file with the filtered data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Filtering complete. Updated JSON written to {} dropped_segments={} dropped_reasons={}",
        json_path,
        dropped_count,
        dropped_reasons,
    )


__all__ = [
    "select_tsv",
    "normalize_audio",
    "build_segment_output_path",
    "materialize_segments",
    "filter_segment_results",
    "filter_and_cleanup",
]
