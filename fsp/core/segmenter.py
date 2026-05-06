"""
Audio segmentation class.

This module contains the Segmenter class migrated from:
- steps/segment.py
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import pandas as pd


class Segmenter:
    """Build logical word-level segments from alignment without materializing audio clips."""

    MIN_DUR = 0.25  # seconds – segments shorter than this are skipped

    def __init__(
        self,
        alignment_file: str,
        audio_file: str,
        out_path: str,
        drop_callback: Callable[[Dict[str, Any]], None] | None = None,
    ):
        """
        Initialize the Segmenter.

        Args:
            alignment_file: Path to CTM alignment file
            audio_file: Path to source audio file
            out_path: Output directory for segments
        """
        if not (os.path.isfile(alignment_file) and os.path.isfile(audio_file)):
            raise IOError(
                f"audio file or alignment file doesn't exist:\n  {audio_file}\n  {alignment_file}"
            )

        self.audio_file = audio_file
        self.alignment = pd.read_csv(
            alignment_file, sep=" ", names=["file", "nb", "start", "duration", "text"]
        )
        self.out_path = out_path
        self.drop_callback = drop_callback
        os.makedirs(out_path, exist_ok=True)

    def _record_drop(
        self,
        *,
        base_name: str,
        start: float,
        end: float,
        normalized_text: str,
        reason: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Record a dropped segment if a callback is configured."""
        if self.drop_callback is None:
            return
        payload: Dict[str, Any] = {
            "stage": "segmentation",
            "block_id": base_name,
            "start": round(start, 2),
            "end": round(end, 2),
            "text": normalized_text,
            "reason": reason,
        }
        if details:
            payload.update(details)
        self.drop_callback(payload)

    def segment_audio(self) -> List[Dict[str, Any]]:
        """
        Segment audio based on alignment, returning logical segments only.

        Returns:
            List of segment dictionaries containing timestamps and text.
        """
        results = []
        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]

        for _, row in self.alignment.iterrows():
            start, dur = row["start"], row["duration"]
            end = start + dur
            normalized_text = row["text"].replace("<space>", " ")
            if dur < self.MIN_DUR:
                self._record_drop(
                    base_name=base_name,
                    start=start,
                    end=end,
                    normalized_text=normalized_text,
                    reason="duration_below_minimum",
                    details={
                        "duration": round(dur, 2),
                        "minimum_duration": self.MIN_DUR,
                    },
                )
                continue  # skip ultra-short segments

            result = {
                "start": start,
                "end": end,
                "normalized_text": normalized_text,
                "segment_base_name": base_name,
            }
            results.append(result)

        return results


__all__ = ["Segmenter"]
