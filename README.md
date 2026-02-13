# Found‑Speech Pipeline

## Introduction

The **Found‑Speech Pipeline** processes **audio-transcript pairs** (WAV + TSV) into clean, word‑level aligned JSON files with optional CER/WER analytics.  The design goal is **zero external services**: everything runs locally in a Python virtual‑environment on Linux or on **Windows Subsystem for Linux (WSL)**.



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

Everything is now in place. 

---

## Input Format

Place matching `.wav` and `.tsv` files inside the `ingestion/` directory.

Example:

```
ingestion/
  input-id_01.wav
  input-id_01.tsv
  input-id_02.wav
  input-id_02.tsv
```

Each `.tsv` must contain **exactly one line** in this format:

```
/absolute/path/to/input-id.wav<TAB>Full transcript text
```

Requirements:

* WAV filename and TSV filename must share the same stem.
* The WAV path inside the TSV must be absolute.
* The TSV must contain a single row.

---

## Running the Pipeline

### Process a Single Session

```bash
python pipeline_service.py --input-id input-id_01 --lang [ca,es]
```

### Batch Mode (Process All Valid Pairs)

```bash
python pipeline_service.py --lang [ca,es]
```

Batch mode automatically:

* Scans `ingestion/`
* Detects valid `.wav + .tsv` pairs
* Processes them sequentially
* Skips incomplete pairs

---

## Output Files

Final outputs appear in:

```
merged/
  final_output_<input-id>.json
  final_output_<input-id>.csv   
  final_output_<input-id>.png  

```

---

## Script Reference

Below is a **complete list of CLI entry‑points**.  Run any file with `-h/--help` for the authoritative parser.

### 1. `pipeline_service.py`

High‑level orchestrator – call this **in almost all cases**.

| Argument     | Description                                          |
| ------------ | ---------------------------------------------------- |
| `--input-id` | Process a single audio-transcript pair (optional).   |
| `--lang`     | `ca` or `es` (default: `ca`).                        |

Modes:

* If `--input-id` is provided → single mode
* If omitted → batch mode

Internally it calls the remaining scripts in the order shown below.

---

### 2. `scripts/normalize_tsv.py`

| Positional          | Description                                                                      |
| ------------------- | -------------------------------------------------------------------------------- |
| `input_tsv`         | A two‑column TSV: `wav_path<TAB>text`.                                           |
| `lang`              | `ca` or `es` – language‐specific normalisation rules.                            |
| `mark` *(optional)* | If provided, deduplicates sentences containing this marker before normalisation. |

Writes `<input-id>_norm_mark.tsv` (or `_norm.tsv)`  in `normalized/<input-id>`.

---

### 3. `scripts/normalize_audio.py`

| Option                    | Description              |
| ------------------------- | ------------------------ |
| `--input-id` (required)   | WAV + TSV filename stem  |

Transcodes the WAV in `ingestion/` to the canonical 16 kHz mono format and produces:

* `inputs/normalized/<input-id>/<input-id>.wav`
* `inputs/normalized/<input-id>/<input-id>_metadata.json`

---

### 4. `steps/generate_final_data.py`

| Option                            | Description                             |
| --------------------------------- | --------------------------------------- |
| `--input-id` (required)           | WAV + TSV filename stem                 |
| `--output`                        | Custom JSON filename.                   |
| `--device`   (Default: `auto`)    | `cuda`, `cpu`, or `auto`. (Only affects the ASR stage; forced alignment stays on CPU. )                                                                        |

Stages:

1. NeMo Manifest generation, producing:
* `inputs/manifest/<input-id>_manifest.json`

2. Neural Forced alignment, producing:
* `inputs/wordlevel_alignment/<input-id>/ass/*`
* `inputs/wordlevel_alignment/<input-id>/ctm/*`
* `inputs/wordlevel_alignment/<input-id>/PKuuatqwz00_manifest_with_output_file_paths.json`

2. Language detection & segmentation , producing:
*  `inputs/output_segment/<input-id>/<input-id>_start_end.wav`

3. Per-language ASR

4. Final JSON generation, producing:
*  `inputs/output_segment/final_output_<input-id>.json`

---

### 5. `scripts/duration_filter.py`

| Argument    | Description                      |
| ----------- | -------------------------------- |
| `json_path` | Path to a `final_output_<input-id>.json`. |

Removes segment files *and* JSON entries whose duration is less than minimum duration and higher than maximum durations (Default: **≤ 2 s or > 30 s**).


---

### 6. `scripts/rover_merge.py`

| Option           | Description                                    |
| ---------------- | ---------------------------------------------- |
| `input_glob`     | Single JSON file or quoted glob pattern.       |
| `-o`, `--out-dir`| Destination folder (default: `../merged`)      |
| `--fields`       | Space-separated `pred_*` fields to merge       |
| `--langs`        | Languages to keep (`ca es`)                    |
| `--norm`         | Lower-case + strip punctuation before scoring  |
| `--strategy`     | `centroid` or `vote`.                          |
| `--csv`          | Emit per-segment CSV alongside the merged JSON |
| `--plot`         | Save corpus-level CER bar chart (`.png`)       |

Outputs a merged JSON **with a new `rover_text` field per segment** plus analytics.

---

## Directory Structure

```
.
├─ ingestion/                 # Input WAV + TSV pairs
├─ inputs/
│  ├─ manifest/               # NeMo manifests
│  ├─ normalized/              # Canonical WAV + Normalized TSV + metadata per input-id
│  ├─ output_segment/         # Sentence-level clips + Final JSON
│  └─ wordlevel_alignment/    # ASS + CTM files
├─ merged/                    # ROVER outputs (.csv, .json, .png)
└─ utils/models/              # lid.176.bin
```

---

## Troubleshooting

**"ffmpeg: command not found"**

```
sudo apt install ffmpeg
```

**CUDA out-of-memory**

Run on CPU:

```
CUDA_VISIBLE_DEVICES= python pipeline_service.py --lang ca
```

Or pass `--device cpu` to `generate_final_data.py`.

---

## Notes

* Forced alignment always runs on CPU.
* Batch mode continues even if a input-id fails (incomplete WAV/TSV or processing error).
* Ensure sufficient RAM before processing long recordings.

---

## License

Refer to the repository license file for usage terms.
