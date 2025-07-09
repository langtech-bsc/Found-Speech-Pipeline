# Found‑Speech Pipeline

## Introduction

This repository brings together every script required to take a **public YouTube recording** and turn it into a clean, word‑level aligned JSON file plus optional CER/WER statistics.  The design goal is **zero external services**: everything runs locally in a Python virtual‑environment on Linux or on **Windows Subsystem for Linux (WSL)**.

---

## Hardware & OS prerequisites

| Requirement                        | Why it matters                                                                                                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ≥ 24 GB RAM                        | Whisper‐large‑v3 and forced alignment are memory‑hungry.  Inside WSL open **Windows Terminal → Settings → `wsl.conf`** and set e.g.:<br>`[wsl2]\nmemory=24GB` then run `wsl --shutdown`. |
| Ubuntu 22.04 LTS (native or WSL 2) | Verified host for NVIDIA NeMo.                                                                                                                                     |
| FFmpeg on `$PATH`                  | Transcodes audio to 16 kHz mono WAV.                                                                                                                                                    |
| (Optional) CUDA 11+                | Speeds up Whisper and RNNT by \~4×.                                                                                                                                                     |

---

## Installation (native/WSL)

0. **Enlarge WSL memory** (see table above) or make sure your Linux box has ≥ 24 GB free.

```bash
# 1 – create model folder
mkdir -p utils/models

# 2 – copy the pre‑trained models into it. Here's the link of the model for download: https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN
cp /path/to/lid.176.bin utils/models/

# 3 – create a Python 3.11 virtual‑env
python3.11 -m venv venv

# 4 – activate it
source venv/bin/activate

# 5 – up‑to‑date build tooling
pip install -U pip setuptools wheel

# 6 – all Python dependencies
PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt

# 7 – output folders expected by the pipeline
mkdir ingestion merged
```

Everything is now in place.  Test with the **Quick‑start** below.

---

## Quick‑start

Download, process and merge the two most recent videos of the Interior Catalunya channel:

```bash
python pipeline_service.py \
    --channel "https://www.youtube.com/@InteriorCatalunya/videos" \
    --max 2
```

Outputs will appear in

* `inputs/wordlevel_alignment/final_output_<video‑id>.json`
* `merged/final_output_<video‑id>.json` (+ .csv, .png if requested)

---

## Script reference

Below is a **complete list of CLI entry‑points**.  Run any file with `-h/--help` for the authoritative parser.

### 1. `pipeline_service.py`

High‑level orchestrator – call this **in almost all cases**.

| Argument    | Type & default              | Description                                    |
| ----------- | --------------------------- | ---------------------------------------------- |
| `--video`   | str                         | Single YouTube URL or ID.                      |
| `--channel` | str                         | Channel/playlist URL; processes newest videos. |
| `--lang`    | `ca` \| `es` (default `ca`) | Language used by `normalize_tsv.py`.           |
| `--max`     | int                         | Limit when using `--channel`.                  |

Internally it calls the remaining scripts in the order shown below.

---

### 2. `scripts/youtube_ingest.py`

| Argument                 | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| `youtube_url`            | Full YouTube link or bare ID.                                           |
| `--reject-license`, `-R` | Comma‑separated list (`youtube`, `creativeCommon`) to prevent download. |

*Checks the licence → downloads audio as WAV → grabs official captions or falls back to Whisper large.  Produces `<video‑id>.wav` and `<video‑id>.tsv` inside `ingestion/`.  Exits gracefully if the licence is blocked.*

---

### 3. `scripts/normalize_tsv.py`

| Positional          | Description                                                                      |
| ------------------- | -------------------------------------------------------------------------------- |
| `input_tsv`         | A two‑column TSV: `wav_path<TAB>text`.                                           |
| `lang`              | `ca` or `es` – language‐specific normalisation rules.                            |
| `mark` *(optional)* | If provided, deduplicates sentences containing this marker before normalisation. |

Writes `<input>_norm.tsv` (or `_norm_mark.tsv`) next to the source file.

---

### 4. `scripts/ingest_single.py`

| Option                      | Description                               |
| --------------------------- | ----------------------------------------- |
| `--session-id` *(required)* | YouTube video‑id – used for folder names. |
| `--speaker`                 | Speaker label embedded in metadata JSON.  |

Transcodes the newest WAV in `ingestion/` to the canonical 16 kHz mono format and produces

* `inputs/segments/<session>/…wav`
* `inputs/segments/<session>/<session>_metadata.json`
* `inputs/manifest/<session>_manifest.json`

---

### 5. `steps/generate_final_data.py`

| Option      | Default                       | Description                                                                               |
| ----------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `--session` | *(required)*                  | Video‑id to process.                                                                      |
| `--output`  | `final_output_<session>.json` | Custom JSON file name.                                                                    |
| `--device`  | `auto`                        | `cuda`, `cpu` or auto‑detect.  Only affects the ASR stage; forced alignment stays on CPU. |

Stages: forced alignment → language detection & segmentation → per‑language ASR loop → final JSON.

---

### 6. `scripts/duration_filter.py`

| Positional  | Description                      |
| ----------- | -------------------------------- |
| `json_path` | Path to a `final_output_*.json`. |

Removes segment files *and* JSON entries whose duration is **≤ 2 s or > 30 s**.

---

### 7. `scripts/rover_merge.py`

| Option            | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `input_glob`      | Single JSON file or quoted glob pattern.                            |
| `-o`, `--out-dir` | Destination folder (default `../merged`).                           |
| `--fields`        | Space‑separated list of `pred_*` keys to merge.  Default: all.      |
| `--langs`         | Languages to keep (`ca es`).                                        |
| `--norm`          | Lower‑case + strip punctuation prior to scoring.                    |
| `--strategy`      | `centroid` (Levenshtein barycentre) or `vote` (per‑token majority). |
| `--csv`           | Emit a per‑segment CSV alongside the merged JSON.                   |
| `--plot`          | Save a corpus‑level CER bar chart (.png).                           |

Outputs a merged JSON **with a new `rover_text` field per segment** plus optional analytics.

---

## Input/Output directories overview

```
.
├─ ingestion/                 # transient audio + TSV inputs
├─ inputs/
│  ├─ manifest/              # NeMo manifests
│  ├─ segments/              # canonicalised WAV + meta per video
│  ├─ output_segment/        # short sentence‑level clips
│  └─ wordlevel_alignment/   # final JSON & CTM files
├─ merged/                    # ROVER outputs
└─ utils/models/              # lid.176.bin
```

---

## Troubleshooting

* **"ffmpeg: command not found"** – install `ffmpeg` via `apt‑get install ffmpeg`.
* **CUDA out‑of‑memory** – run with `--device cpu` on `generate_final_data.py` or set `CUDA_VISIBLE_DEVICES=` in the environment.
---