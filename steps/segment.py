import os
import pandas as pd
import soundfile as sf
import subprocess


class Segmenter:
    """Cut aligned blocks into word-level segments and return metadata only."""

    MIN_DUR = 0.25  # seconds - segments shorter than this are skipped

    def __init__(self, alignment_file, audio_file, out_path):
        if not (os.path.isfile(alignment_file) and os.path.isfile(audio_file)):
            raise IOError(
                f"audio file or alignment file doesn't exist:\n  {audio_file}\n  {alignment_file}")

        self.audio_file = audio_file
        self.alignment = pd.read_csv(
            alignment_file, sep=" ", names=["file", "nb", "start", "duration", "text"])
        self.out_path = out_path
        os.makedirs(out_path, exist_ok=True)

    def segment_audio(self):
        results = []
        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]

        for _, row in self.alignment.iterrows():
            start, dur = row["start"], row["duration"]
            if dur < self.MIN_DUR:
                continue  # skip ultra-short segments

            end = start + dur
            normalized_text = row["text"].replace("<space>", " ")

            wav_path = self._segment_cue(start, end, base_name, dur)
            if wav_path is None:  # ffmpeg produced an empty file - skip
                continue

            results.append({
                "segment_path": wav_path,
                "start": start,
                "end": end,
                "normalized_text": normalized_text,
            })

        return results

    def _segment_cue(self, start, end, base_name, duration):
        file_name = f"{base_name}_{start}_{end}.wav"
        out_file = os.path.join(self.out_path, file_name)

        if os.path.isfile(out_file):
            print(f"{out_file} already exists - skipping")
        else:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", self.audio_file,
                "-ss", str(start), "-t", str(duration),
                "-ac", "1", "-ar", "16000", out_file
            ]
            subprocess.call(cmd)

        if not os.path.isfile(out_file):
            print(f"ffmpeg failed for {out_file}")
            return None
        info = sf.info(out_file)
        if info.frames == 0:
            os.remove(out_file)
            return None
        return out_file
