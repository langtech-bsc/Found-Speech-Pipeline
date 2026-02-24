import os
import pandas as pd
from jiwer import cer
import soundfile as sf
import subprocess


class Segmenter:
    """Cut aligned blocks into word‑level segments, run ASR + LM, score CER."""

    MIN_DUR = 0.25  # seconds – segments shorter than this are skipped

    def __init__(self, alignment_file, audio_file, out_path,
                 lg_model, ca_model, es_model):
        if not (os.path.isfile(alignment_file) and os.path.isfile(audio_file)):
            raise IOError(
                f"audio file or alignment file doesn't exist:\n  {audio_file}\n  {alignment_file}")

        self.audio_file = audio_file
        self.alignment = pd.read_csv(
            alignment_file, sep=" ", names=["file", "nb", "start", "duration", "text"])
        self.out_path = out_path
        self.lg_model = lg_model
        self.ca_model = ca_model
        self.es_model = es_model
        os.makedirs(out_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _asr(self, model, wav):
        """Plain CTC decode."""
        return model.transcribe(paths2audio_files=[wav], batch_size=1)[0]

    def _identify_language(self, text):
        lang_id, _conf = self.lg_model.predict(text, k=1)
        return lang_id[0]  # e.g. "__label__ca"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment_audio(self):
        results = []
        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]

        for _, row in self.alignment.iterrows():
            start, dur = row["start"], row["duration"]
            if dur < self.MIN_DUR:
                continue  # skip ultra‑short segments

            end = start + dur
            normalized_text = row["text"].replace("<space>", " ")

            wav_path = self._segment_cue(start, end, base_name, dur)
            if wav_path is None:  # ffmpeg produced an empty file → skip
                continue

            lang_label = self._identify_language(normalized_text)
            if lang_label == "__label__ca":
                model = self.ca_model or self.es_model
                language = "ca"
            elif lang_label in ("__label__gl", "__label__pt"):
                model = None  # no inline CTC model for Galician
                language = "gl"
            elif lang_label == "__label__eu":
                model = None  # no inline CTC model for Basque
                language = "eu"
            else:
                model = self.es_model or self.ca_model
                language = "es"

            if model is not None:
                pred_text = self._asr(model, wav_path)
            else:
                pred_text = ""  # will be filled by multi-model ASR later

            result = {
                "segment_path": wav_path,
                "start": start,
                "end": end,
                "language": language,
                "normalized_text": normalized_text,
                "pred_text": pred_text,
                "cer_score": cer(normalized_text, pred_text)
            }
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # ffmpeg cutter
    # ------------------------------------------------------------------
    def _segment_cue(self, start, end, base_name, duration):
        file_name = f"{base_name}_{start}_{end}.wav"
        out_file = os.path.join(self.out_path, file_name)

        if os.path.isfile(out_file):
            print(f"{out_file} already exists – skipping")
        else:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", self.audio_file,
                "-ss", str(start), "-t", str(duration),
                "-ac", "1", "-ar", "16000", out_file
            ]
            subprocess.call(cmd)

        # validate the produced file
        if not os.path.isfile(out_file):
            print(f"ffmpeg failed for {out_file}")
            return None
        info = sf.info(out_file)
        if info.frames == 0:
            os.remove(out_file)  # clean up
            return None
        return out_file

