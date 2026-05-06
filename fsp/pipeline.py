from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import List, Optional

import pandas as pd
import torch
from loguru import logger

from fsp.core.alignment import generate_final_data, generate_final_data_batch
from fsp.core.audio import filter_and_cleanup
from fsp.core.audio import normalize_audio as _normalize_audio
from fsp.core.punctuation import process_file as punctuate_file
from fsp.core.rover import RoverConfig, process_file
from fsp.core.text import clean_text, remove_chars, split_text
from fsp.utils.logging import build_run_label, build_run_log_dir, write_captured_output
from fsp.utils.paths import (
    GL_EXTRA_ASR_IMAGE,
    ROOT,
    INGESTION_DIR,
    NORM_DIR,
    OUTPUT_SEGMENT_DIR,
    ROVER_DIR,
    resolve_hf_model_dir,
    resolve_lid_model_path,
    resolve_nemo_model_dir,
)

SINGULARITY_FALLBACK = Path("/apps/GPP/SINGULARITY/3.11.5/bin/singularity")


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
        device: str = "auto",
        asr_batch_size: int = 16,
        max_duration: float = 30,
        min_duration: float = 2,
        lid_model_path: Optional[Path] = None,
        nemo_model_dir: Optional[Path] = None,
        hf_model_dir: Optional[Path] = None,
        enable_gl_extra_asr: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            lang: Primary language ('ca', 'es', 'eu', or 'gl')
            device: Device for ASR ('auto', 'cuda', 'cpu')
            asr_batch_size: Batch size for segment-level ASR inference
            max_duration: Maximum segment duration in seconds
            min_duration: Minimum segment duration in seconds
            lid_model_path: Path to the FastText language-ID model file
            nemo_model_dir: Directory containing local NeMo checkpoints
            hf_model_dir: Directory containing the HuggingFace cache root
            enable_gl_extra_asr: Whether to run the GL-only sidecar enrichment
        """
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        if asr_batch_size < 1:
            raise ValueError("asr_batch_size must be >= 1")

        self.lang = lang
        self.device = device
        self.asr_batch_size = asr_batch_size
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.lid_model_path = resolve_lid_model_path(lid_model_path)
        self.nemo_model_dir = resolve_nemo_model_dir(nemo_model_dir)
        self.hf_model_dir = resolve_hf_model_dir(hf_model_dir)
        self.enable_gl_extra_asr = enable_gl_extra_asr
        self.run_label = "pipeline"
        self.run_log_dir: Optional[Path] = None

    def set_run_context(self, run_label: str, run_log_dir: Optional[Path] = None) -> None:
        self.run_label = run_label
        self.run_log_dir = run_log_dir

    def normalize_tsv(self, input_tsv: Path, lang: str, mark: str = "|") -> Path:
        """
        Normalize a TSV file (Step 1).

        Args:
            input_tsv: Path to input TSV file
            lang: Language code ('ca', 'es', 'eu', or 'gl')
            mark: Sentence separator mark

        Returns:
            Path to the normalized TSV file
        """
        if lang not in ("ca", "es", "eu", "gl"):
            raise ValueError("lang must be 'ca', 'es', 'eu', or 'gl'")
        if not input_tsv.is_file():
            raise FileNotFoundError(f"input file '{input_tsv}' not found")

        # Load TSV (no header)
        df = pd.read_csv(input_tsv, sep="\t", header=None, names=["wav_path", "text"], dtype=str)

        # Normalize text column
        df["normalized_text"] = df["text"].apply(lambda t: self._normalize_row(t, lang, mark))
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
        logger.info(f"Normalized TSV written to: {out_name}")
        return out_name

    @staticmethod
    def _normalize_row(text: str, lang: str, mark: str) -> str:
        text = text.replace("\n", ".").replace(" - ", ".").replace(" · ", ".").replace("|", ".")
        split_str = split_text(remove_chars(text, False, lang), False, mark)
        if not split_str:
            return ""
        if not mark:
            return clean_text(split_str.strip(), lang, False, False)
        return mark.join(
            [clean_text(chunk.strip(), lang, False, False) for chunk in split_str.split(mark) if chunk.strip()]
        )

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
        device: Optional[str] = None,
        asr_batch_size: Optional[int] = None,
        lid_model_path: Optional[Path] = None,
        nemo_model_dir: Optional[Path] = None,
        hf_model_dir: Optional[Path] = None,
        run_log_dir: Optional[Path] = None,
    ) -> Path:
        """
        Run forced alignment + ASR enrichment (Step 3).

        Args:
            input_id: Audio-transcript pair ID
            lang: Language code (default: pipeline language)
            output_name: Custom output JSON name
            device: Device for ASR ('auto', 'cuda', 'cpu')
            asr_batch_size: Batch size for segment-level ASR inference
            lid_model_path: Path to the FastText language-ID model file
            nemo_model_dir: Directory containing local NeMo checkpoints
            hf_model_dir: Directory containing the HuggingFace cache root

        Returns:
            Path to the output JSON file
        """
        lang = lang or self.lang
        device = device or self.device
        asr_batch_size = asr_batch_size or self.asr_batch_size
        return generate_final_data(
            input_id=input_id,
            lang=lang,
            output_name=output_name,
            device=device,
            asr_batch_size=asr_batch_size,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            lid_model_path=lid_model_path or self.lid_model_path,
            nemo_model_dir=nemo_model_dir or self.nemo_model_dir,
            hf_model_dir=hf_model_dir or self.hf_model_dir,
            run_log_dir=run_log_dir,
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

        lang_order = [self.lang] + [lang for lang in ("ca", "es", "eu", "gl") if lang != self.lang]

        config = RoverConfig(
            out_dir=out_dir,
            langs=lang_order,
            csv=csv,
            plot=plot,
        )
        process_file(json_path, config)

    def maybe_run_gl_extra_asr(self, out_json_path: Path, run_log_dir: Optional[Path] = None) -> None:
        if self.lang != "gl" or not self.enable_gl_extra_asr:
            return

        runner = which("apptainer") or which("singularity")
        if not runner and SINGULARITY_FALLBACK.is_file():
            runner = str(SINGULARITY_FALLBACK)
        if not runner:
            raise RuntimeError("Neither 'apptainer' nor 'singularity' is available")
        if not GL_EXTRA_ASR_IMAGE.is_file():
            raise RuntimeError(f"Missing GL extra ASR image: {GL_EXTRA_ASR_IMAGE}")

        model_bind_root = self.hf_model_dir
        if not model_bind_root.is_dir():
            raise RuntimeError(f"HF model directory not found: {model_bind_root}")

        env = os.environ.copy()
        env.setdefault("HF_MODEL_DIR", str(self.hf_model_dir))
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")

        cmd = [
            runner,
            "exec",
            "--nv",
            "--bind",
            f"{ROOT}:{ROOT}",
            "--bind",
            f"{model_bind_root}:{model_bind_root}",
            str(GL_EXTRA_ASR_IMAGE),
            "python",
            str(ROOT / "steps" / "enrich_segment_hypotheses.py"),
            str(out_json_path),
            "--langs",
            "gl",
            "--models",
            "whisper_large_v3_turbo_gl_v1_0",
            "phi_4_multimodal_instruct_gl_v1_0",
            "--device",
            "cuda",
            "--overwrite-existing",
            "--hf-model-dir",
            str(self.hf_model_dir),
        ]
        logger.info("Step 4/7 gl_extra_asr started output={}", out_json_path)
        completed = subprocess.run(
            cmd,
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if run_log_dir is not None:
            write_captured_output(
                run_log_dir / "gl_extra_asr.log",
                (
                    ("stdout", completed.stdout),
                    ("stderr", completed.stderr),
                ),
            )
        if completed.returncode:
            if completed.stderr.strip():
                logger.error("GL extra ASR stderr:\n{}", completed.stderr.strip())
            raise RuntimeError("GL extra ASR enrichment failed")
        logger.info("Step 4/7 gl_extra_asr completed")

    def punctuate(self, json_path: Path, device: str = "auto") -> None:
        chosen_device = device
        if chosen_device == "auto":
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        if chosen_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            chosen_device = "cpu"
        device_id = 0 if chosen_device == "cuda" else -1
        punctuate_file(json_path, device_id, hf_model_dir=self.hf_model_dir)

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
        run_log_dir = self.run_log_dir or build_run_log_dir(build_run_label(self.run_label, input_id, self.lang))

        logger.info("Step 1/7 normalize_tsv started input_id={} lang={}", input_id, self.lang)
        self.normalize_tsv(raw_tsv, self.lang, "|")
        logger.info("Step 1/7 normalize_tsv completed input_id={}", input_id)

        logger.info("Step 2/7 normalize_audio started input_id={}", input_id)
        self.normalize_audio(input_id)
        logger.info("Step 2/7 normalize_audio completed input_id={}", input_id)

        logger.info("Step 3/7 generate_final_data started input_id={} logs={}", input_id, run_log_dir)
        self.generate_final_data(input_id, output_name=out_json_name, run_log_dir=run_log_dir)
        logger.info("Step 3/7 generate_final_data completed input_id={}", input_id)

        self.maybe_run_gl_extra_asr(out_json_path, run_log_dir=run_log_dir)

        logger.info("Step 5/7 duration_filter started input_id={}", input_id)
        self.duration_filter(out_json_path)
        logger.info("Step 5/7 duration_filter completed input_id={}", input_id)

        logger.info("Step 6/7 rover_merge started input_id={}", input_id)
        self.rover_merge(out_json_path)
        logger.info("Step 6/7 rover_merge completed input_id={}", input_id)

        rover_json = ROVER_DIR / out_json_name

        logger.info("Step 7/7 punctuation started input_id={}", input_id)
        self.punctuate(rover_json)
        logger.info("Step 7/7 punctuation completed input_id={} output={}", input_id, rover_json)

        return rover_json

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
            logger.warning("No valid (.wav + .tsv) pairs found in ingestion/")
        else:
            logger.info(f"Found {len(valid_ids)} valid input pair(s)")

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

        prepared_ids: List[str] = []
        output_names: dict[str, str] = {}
        run_log_dirs: dict[str, Path] = {}

        for i, input_id in enumerate(input_ids, 1):
            logger.info("Batch item {}/{} preparation started input_id={}", i, len(input_ids), input_id)
            raw_tsv = INGESTION_DIR / f"{input_id}.tsv"
            raw_wav = INGESTION_DIR / f"{input_id}.wav"
            run_log_dir = self.run_log_dir or build_run_log_dir(build_run_label(self.run_label, input_id, self.lang))
            out_json_name = f"final_output_{input_id}.json"

            try:
                if not raw_tsv.exists() or not raw_wav.exists():
                    raise RuntimeError(f"Missing pair for input ID: {input_id}")

                logger.info("Step 1/7 normalize_tsv started input_id={} lang={}", input_id, self.lang)
                self.normalize_tsv(raw_tsv, self.lang, "|")
                logger.info("Step 1/7 normalize_tsv completed input_id={}", input_id)

                logger.info("Step 2/7 normalize_audio started input_id={}", input_id)
                self.normalize_audio(input_id)
                logger.info("Step 2/7 normalize_audio completed input_id={}", input_id)
            except Exception as err:
                logger.error("Batch item failed input_id={} error={}", input_id, err)
                logger.warning("Skipping input_id={}", input_id)
                continue

            prepared_ids.append(input_id)
            output_names[input_id] = out_json_name
            run_log_dirs[input_id] = run_log_dir

        if not prepared_ids:
            return []

        logger.info("Batch phase started: generate_final_data inputs={}", len(prepared_ids))
        generated_outputs: dict[str, Path] = {}
        try:
            generated_outputs = generate_final_data_batch(
                prepared_ids,
                lang=self.lang,
                device=self.device,
                lid_model_path=self.lid_model_path,
                nemo_model_dir=self.nemo_model_dir,
                hf_model_dir=self.hf_model_dir,
                asr_batch_size=self.asr_batch_size,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                output_names=output_names,
                run_log_dirs=run_log_dirs,
            )
        except Exception as err:
            logger.error("Batch generate_final_data phase failed error={}", err)
            raise

        results = []
        for input_id in prepared_ids:
            out_json_path = generated_outputs.get(input_id)
            if out_json_path is None:
                logger.warning("Skipping input_id={} because generate_final_data produced no output", input_id)
                continue

            logger.info("Step 3/7 generate_final_data completed input_id={}", input_id)
            try:
                self.maybe_run_gl_extra_asr(out_json_path, run_log_dir=run_log_dirs[input_id])

                logger.info("Step 5/7 duration_filter started input_id={}", input_id)
                self.duration_filter(out_json_path)
                logger.info("Step 5/7 duration_filter completed input_id={}", input_id)

                logger.info("Step 6/7 rover_merge started input_id={}", input_id)
                self.rover_merge(out_json_path)
                logger.info("Step 6/7 rover_merge completed input_id={}", input_id)

                rover_json = ROVER_DIR / output_names[input_id]

                logger.info("Step 7/7 punctuation started input_id={}", input_id)
                self.punctuate(rover_json)
                logger.info("Step 7/7 punctuation completed input_id={} output={}", input_id, rover_json)

                results.append(rover_json)
            except Exception as err:
                logger.error("Batch item failed input_id={} error={}", input_id, err)
                logger.warning("Skipping input_id={}", input_id)

        return results


__all__ = ["Pipeline"]
