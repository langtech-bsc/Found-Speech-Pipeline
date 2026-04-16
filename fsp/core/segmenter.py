"""
Audio segmentation class.

This module contains the Segmenter class migrated from:
- steps/segment.py
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import pandas as pd
import soundfile as sf
from jiwer import cer
from loguru import logger

if TYPE_CHECKING:
    import fasttext


class Segmenter:
    """Cut aligned blocks into word-level segments, run ASR + LM, score CER."""

    MIN_DUR = 0.25  # seconds – segments shorter than this are skipped

    def __init__(
        self,
        alignment_file: str,
        audio_file: str,
        out_path: str,
        lg_model: "fasttext.FastText._FastText",
        ca_model: Any = None,
        es_model: Any = None,
        drop_callback: Callable[[Dict[str, Any]], None] | None = None,
    ):
        """
        Initialize the Segmenter.

        Args:
            alignment_file: Path to CTM alignment file
            audio_file: Path to source audio file
            out_path: Output directory for segments
            lg_model: FastText language identification model
            ca_model: Catalan ASR model
            es_model: Spanish ASR model
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
        self.lg_model = lg_model
        self.ca_model = ca_model
        self.es_model = es_model
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

    def _asr(self, model: Any, wav: str) -> str:
        """Plain CTC decode."""
        return model.transcribe(paths2audio_files=[wav], batch_size=1)[0]

    def _identify_language(self, text: str) -> str:
        """Identify language of text."""
        lang_id, _conf = self.lg_model.predict(text, k=1)
        return lang_id  # e.g. "__label__ca"

    def segment_audio(self) -> List[Dict[str, Any]]:
        """
        Segment audio based on alignment and run ASR.

        Returns:
            List of segment dictionaries containing paths, timestamps,
            language, text, predictions, and CER scores.
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

            wav_path, failure_reason = self._segment_cue(start, end, base_name, dur)
            if wav_path is None:  # ffmpeg produced an empty file; skip it
                self._record_drop(
                    base_name=base_name,
                    start=start,
                    end=end,
                    normalized_text=normalized_text,
                    reason="segment_clip_generation_failed",
                    details={"failure_reason": failure_reason or "unknown clip generation failure"},
                )
                logger.warning(
                    "Skipping segment %s %.2f-%.2f: %s | text=%r",
                    base_name,
                    start,
                    end,
                    failure_reason or "unknown clip generation failure",
                    normalized_text,
                )
                continue

            lang_label = self._identify_language(normalized_text)
            result = {
                "segment_path": wav_path,
                "start": start,
                "end": end,
                "normalized_text": normalized_text,
            }

            if self.ca_model is not None or self.es_model is not None:
                if lang_label == "__label__ca":
                    model = self.ca_model or self.es_model
                    language = "ca"
                else:
                    model = self.es_model or self.ca_model
                    language = "es"

                if model is not None:
                    pred_text = self._asr(model, wav_path)
                    result["language"] = language
                    result["pred_text"] = pred_text
                    result["cer_score"] = cer(normalized_text, pred_text)

            results.append(result)

        return results

    def _segment_cue(
        self,
        start: float,
        end: float,
        base_name: str,
        duration: float,
    ) -> tuple[str | None, str | None]:
        """
        Cut a segment from the audio file using ffmpeg.

        Args:
            start: Start time in seconds
            end: End time in seconds
            base_name: Base name for output file
            duration: Duration in seconds

        Returns:
            Tuple of ``(path, reason)``. When clipping fails, path is None and
            reason contains the specific failure cause.
        """
        file_name = f"{base_name}_{start}_{end}.wav"
        out_file = os.path.join(self.out_path, file_name)

        if os.path.isfile(out_file):
            logger.debug(f"{out_file} already exists – skipping")
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                self.audio_file,
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-ac",
                "1",
                "-ar",
                "16000",
                out_file,
            ]
            return_code = subprocess.call(cmd)
            if return_code != 0:
                return None, f"ffmpeg exited with code {return_code}"

        # Validate the produced file
        if not os.path.isfile(out_file):
            logger.error(f"ffmpeg failed for {out_file}")
            return None, "ffmpeg did not create the output file"
        info = sf.info(out_file)
        if info.frames == 0:
            os.remove(out_file)  # clean up
            return None, "ffmpeg created a zero-frame clip"
        return out_file, None


__all__ = ["Segmenter"]
