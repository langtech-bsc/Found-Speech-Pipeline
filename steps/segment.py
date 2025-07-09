import os
import pandas as pd
from jiwer import cer
import soundfile as sf
import subprocess


class Segmenter:
    """Cut aligned blocks into word‑level segments, run ASR + LM, score CER."""

    MIN_DUR = 0.25  # seconds – segments shorter than this are skipped

    def __init__(self, alignment_file, audio_file, out_path,
                 lg_model, ca_model, es_model, decoder):
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
        self.decoder = decoder
        os.makedirs(out_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _asr(self, model, wav):
        """Plain CTC decode."""
        return model.transcribe(paths2audio_files=[wav], batch_size=1)[0]

    def _asr_lm(self, model, wav):
        """CTC beam‑search with KenLM."""
        logits = model.transcribe(paths2audio_files=[wav], logprobs=True)[0]
        return self.decoder.decode(logits)

    def _identify_language(self, text):
        lang_id, _conf = self.lg_model.predict(text, k=1)
        return lang_id  # e.g. "__label__ca"

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
            text = row["text"].replace("<space>", " ")

            wav_path = self._segment_cue(start, end, base_name, dur)
            if wav_path is None:  # ffmpeg produced an empty file → skip
                continue

            lang_label = self._identify_language(text)
            if lang_label == "__label__ca":
                pred_text = self._asr(self.ca_model, wav_path)
                pred_lm = self._asr_lm(self.ca_model, wav_path)
                language = "ca"
            else:
                pred_text = self._asr(self.es_model, wav_path)
                # TODO: Spanish LM. Using CA LM as placeholder.
                pred_lm = self._asr_lm(self.ca_model, wav_path)
                language = "es"

            result = {
                "segment_path": wav_path,
                "start": start,
                "end": end,
                "language": language,
                "org_text": text,
                "pred_text": pred_text,
                "pred_text_lm": pred_lm,
                "cer_score": cer(text, pred_text),
                "cer_score_lm": cer(text, pred_lm),
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


# ----------------------------------------------------------------------
# Convenience CLI (rarely used – keeps original behaviour)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from pyctcdecode import build_ctcdecoder
    import fasttext, nemo.collections.asr as nemo_asr

    # dummy demo to show it runs – real pipeline passes models via kwargs
    lg_model = fasttext.load_model("utils/models/lid.176.ftz")
    ca_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_ca_conformer_ctc_large")
    es_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_es_conformer_ctc_large")
    vocab = ca_model.decoder.vocabulary
    decoder = build_ctcdecoder(vocab, kenlm_model_path="utils/models/5GRAM_Parla_Text_Corpus.arpa")

    alignment_directory = "inputs/wordlevel_alignment/"
    for root, _, files in os.walk(alignment_directory):
        if "/ctm/segments" not in root:
            continue
        session_id = root.split("/")[1]
        for file in files:
            if not file.endswith(".ctm"):
                continue
            seg_id = os.path.splitext(file)[0]
            audio_file = f"inputs/segments/{session_id}/{seg_id}.wav"
            out_path = f"inputs/output_segment/{session_id}"
            seg = Segmenter(os.path.join(root, file), audio_file, out_path,
                            lg_model, ca_model, es_model, decoder)
            print(seg.segment_audio())