from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from fsp.core.alignment import generate_final_data
from fsp.core.audio import filter_and_cleanup
from fsp.core.audio import normalize_audio as _normalize_audio
from fsp.core.rover import RoverConfig, process_file
from fsp.core.text import remove_chars, split_text
from fsp.utils.paths import INGESTION_DIR, NORM_DIR, OUTPUT_SEGMENT_DIR, ROOT, ROVER_DIR


class Pipeline:
    """
    FSP Pipeline orchestrator.

    This class coordinates all pipeline steps using direct Python function calls
    instead of subprocess invocations, enabling better optimization and resource
    management.
    """

    def __init__(
        self,
        lang: str = "ca",
        max_duration: float = 30,
        min_duration: float = 2,
    ):
        """
        Initialize the pipeline.

        Args:
            lang: Primary language ('ca' or 'es')
            max_duration: Maximum segment duration in seconds
            min_duration: Minimum segment duration in seconds
        """
        self.lang = lang
        self.max_duration = max_duration
        self.min_duration = min_duration

    def normalize_tsv(self, input_tsv: Path, lang: str, mark: str = ". ") -> Path:
        """
        Normalize a TSV file (Step 1).

        Args:
            input_tsv: Path to input TSV file
            lang: Language code ('ca' or 'es')
            mark: Sentence separator mark

        Returns:
            Path to the normalized TSV file
        """
        if lang not in ("ca", "es"):
            raise ValueError("lang must be 'ca' or 'es'")
        if not input_tsv.is_file():
            raise FileNotFoundError(f"input file '{input_tsv}' not found")

        # Load TSV (no header)
        df = pd.read_csv(input_tsv, sep="\t", header=None, names=["wav_path", "text"], dtype=str)

        # Normalize text column
        df["normalized_text"] = df["text"].apply(
            lambda t: split_text(remove_chars(t, False, lang), False, mark)
        )
        suffix = "norm_mark"

        # Build output filename in normalized/ directory
        input_path = input_tsv.resolve()
        base = input_path.stem

        out_dir = NORM_DIR / base
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = out_dir / f"{base}_{suffix}.tsv"

        # Write wav_path + original text + normalized_text
        df[["wav_path", "text", "normalized_text"]].to_csv(
            out_name, sep="\t", index=False, header=False, quoting=csv.QUOTE_ALL
        )
        print(f"✔ Normalized TSV written to: {out_name}")
        return out_name

    def normalize_audio(self, input_id: str) -> Path:
        """
        Normalize audio to 16kHz mono WAV + generate metadata (Step 2).

        Args:
            input_id: Audio-transcript pair ID

        Returns:
            Path to the output directory
        """
        return _normalize_audio(input_id)

    def generate_final_data(
        self,
        input_id: str,
        lang: Optional[str] = None,
        output_name: Optional[str] = None,
        device: str = "auto",
    ) -> Path:
        """
        Run forced alignment + ASR enrichment (Step 3).

        Args:
            input_id: Audio-transcript pair ID
            lang: Language code (default: pipeline language)
            output_name: Custom output JSON name
            device: Device for ASR ('auto', 'cuda', 'cpu')

        Returns:
            Path to the output JSON file
        """
        lang = lang or self.lang
        return generate_final_data(
            input_id=input_id,
            lang=lang,
            output_name=output_name,
            device=device,
        )

    def duration_filter(self, json_path: Path) -> None:
        """
        Filter segments by duration (Step 4).

        Args:
            json_path: Path to the JSON file to filter
        """
        filter_and_cleanup(
            str(json_path),
            min_dur=self.min_duration,
            max_dur=self.max_duration,
        )

    def rover_merge(
        self,
        json_path: Path,
        out_dir: Optional[Path] = None,
        csv: bool = True,
        plot: bool = True,
    ) -> None:
        """
        Run ROVER merge and scoring (Step 5).

        Args:
            json_path: Path to the JSON file to process
            out_dir: Output directory (default: ROVER_DIR)
            csv: Whether to output CSV
            plot: Whether to output plot
        """
        out_dir = out_dir or ROVER_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        config = RoverConfig(
            out_dir=out_dir,
            langs=[self.lang, "es"] if self.lang == "ca" else ["es", "ca"],
            csv=csv,
            plot=plot,
        )
        process_file(json_path, config)

    def run_all(self, input_id: str) -> Path:
        """
        Run the complete pipeline for a single input.

        Args:
            input_id: Audio-transcript pair ID

        Returns:
            Path to the final output JSON
        """
        raw_tsv = INGESTION_DIR / f"{input_id}.tsv"
        raw_wav = INGESTION_DIR / f"{input_id}.wav"

        if not raw_tsv.exists() or not raw_wav.exists():
            raise RuntimeError(f"Missing pair for input ID: {input_id}")

        out_json_name = f"final_output_{input_id}.json"
        out_json_path = OUTPUT_SEGMENT_DIR / out_json_name

        # 1. Normalize TSV
        print(f"\n► Normalise TSV")
        self.normalize_tsv(raw_tsv, self.lang)

        # 2. Normalize audio + metadata
        print(f"\n► Ingest single")
        self.normalize_audio(input_id)

        # 3. Generate final data
        print(f"\n► Generate final data")
        self.generate_final_data(input_id, output_name=out_json_name)

        # 4. Duration filter
        print(f"\n► Duration filter")
        self.duration_filter(out_json_path)

        # 5. ROVER merge
        print(f"\n► ROVER merge")
        self.rover_merge(out_json_path)

        return out_json_path

    @staticmethod
    def find_valid_input_ids() -> List[str]:
        """
        Scan ingestion/ and return all valid audio-transcript pair IDs.

        Returns:
            List of valid input IDs (that have BOTH .wav and .tsv files)
        """
        wav_ids = {p.stem for p in INGESTION_DIR.glob("*.wav")}
        tsv_ids = {p.stem for p in INGESTION_DIR.glob("*.tsv")}

        valid_ids = sorted(wav_ids & tsv_ids)

        if not valid_ids:
            print("⚠️  No valid (.wav + .tsv) pairs found in ingestion/")
        else:
            print(f"🔎 Found {len(valid_ids)} valid input pair(s)")

        return valid_ids

    def run_batch(self, input_ids: Optional[List[str]] = None) -> List[Path]:
        """
        Run the pipeline on multiple inputs.

        Args:
            input_ids: List of input IDs (default: auto-detect all valid pairs)

        Returns:
            List of output JSON paths
        """
        if input_ids is None:
            input_ids = self.find_valid_input_ids()

        results = []
        for i, input_id in enumerate(input_ids, 1):
            print("\n" + "═" * 70)
            print(f"[{i}/{len(input_ids)}] Processing {input_id}")
            print("═" * 70)

            try:
                result = self.run_all(input_id)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed: {input_id} → {e}")
                print("⚠️  Skipping...\n")

        return results


__all__ = ["Pipeline"]
