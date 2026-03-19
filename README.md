# Found‑Speech Pipeline

## Introduction

The **Found‑Speech Pipeline** processes **audio‑transcript pairs** (WAV + TSV) into clean, word‑level aligned JSON files with CER/WER analytics via ROVER merging of multiple ASR hypotheses. Everything runs locally — no external services required.

Supported languages: **Catalan** (`ca`) and **Spanish** (`es`).

---

## Hardware & OS Prerequisites

| Requirement | Notes |
|---|---|
| ≥ 24 GB RAM (or RAM + swap) | Whisper‑large‑v3 and forced alignment are memory‑hungry |
| Ubuntu 22.04+ (native, WSL 2, or Docker) | Verified host for NVIDIA NeMo |
| FFmpeg on `$PATH` | Transcodes audio to 16 kHz mono WAV (included in Docker image) |
| (Optional) CUDA 11+ | Speeds up Whisper and RNNT by ~4× |

> [!NOTE]
> **WSL users:** Open **Windows Terminal → Settings → `.wslconfig`** and set `memory=24GB`, then run `wsl --shutdown`.

---

## Input Format

Place matching `.wav` and `.tsv` files inside the `ingestion/` directory:

```
ingestion/
  my_recording.wav
  my_recording.tsv
```

Each `.tsv` must contain **exactly one line** in this format:

```
/app/ingestion/my_recording.wav<TAB>Full transcript text here.
```

> [!IMPORTANT]
> - The WAV filename and TSV filename must share the same stem (e.g. `my_recording`).
> - **Docker/Singularity:** use the container path `/app/ingestion/<filename>.wav`.
> - **Native:** use the absolute host path (e.g. `/home/user/Found-Speech-Pipeline/ingestion/my_recording.wav`).
> - The TSV must contain a single row with two tab‑separated columns: WAV path and transcript text.

---

## Quick Start (Docker — recommended)

### Step 1. Build the Docker image

```bash
docker build -t fsp-pipeline -f Dockerfile .
```

### Step 2. Download models (once, on host)

All models are stored **outside** the image in `utils/models/` and mounted at runtime.
This keeps the image lean and lets you run fully offline after downloading.

```bash
# Create model directories
mkdir -p utils/models/nemo utils/models/huggingface

# Download the FastText language-ID model (~126 MB)
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/lid.176.bin

# Download all ASR models — Catalan + Spanish (~15 GB)
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang all
```

To download only one language instead:

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang es    # Spanish only (~8 GB)

docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang ca    # Catalan only (~7 GB)
```

> [!TIP]
> The `--user $(id -u):$(id -g)` flag ensures files created by Docker are owned by **your user**
> instead of root, avoiding permission issues with Singularity and on the host.

### Step 3. Run the pipeline

```bash
# Process a single recording
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --input-id my_recording --lang es

# Batch mode — process all WAV+TSV pairs in ingestion/
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --lang es
```

If your models live outside the repository, pass the explicit runtime paths:
`--lid-model-path /path/to/lid.176.bin`, `--nemo-model-dir /path/to/nemo`, and
`--hf-model-dir /path/to/huggingface`. For `run_singularity.sh`, set
`LID_MODEL_PATH`, `NEMO_MODEL_DIR`, and `HF_MODEL_DIR` before invoking the wrapper.

### Step 4. Run fully offline (optional)

Once all models are downloaded, you can run with no network access:

```bash
docker run --rm --network=none --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --input-id my_recording --lang es
```

---

## Singularity / Apptainer (for HPC)

For HPC clusters or restricted environments where Docker is not available, you can convert the Docker image to a Singularity `.sif` file.

**Prerequisites:**
- Docker image already built (Step 1 above)
- Models already downloaded to `utils/models/` (Step 2 above)
- `singularity` or `apptainer` installed on the target machine

### Build the `.sif` image

```bash
./build_singularity.sh
```

This creates `fsp-pipeline.sif` (~3–4 GB) from the Docker image. Takes ~5–10 minutes.

### Run with Singularity

```bash
# Single recording
./run_singularity.sh --input-id my_recording --lang es

# Batch from file (one ID per line, e.g. from create_splits.sh)
./run_singularity.sh --input-id-file /path/to/ids.txt --lang es

# Batch mode (all pairs in ingestion/)
./run_singularity.sh --lang es
```

The `run_singularity.sh` wrapper automatically binds `ingestion/`, `inputs/`, `merged/`,
`${LID_MODEL_PATH:-utils/models/lid.176.bin}`, `${NEMO_MODEL_DIR:-utils/models/nemo}`,
and `${HF_MODEL_DIR:-utils/models/huggingface}` into the container.

---

## Installation (native / WSL — without Docker)

If you prefer to run without containers:

```bash
# 1 — Create model directories
mkdir -p utils/models/nemo utils/models/huggingface

# 2 — Download the FastText language-ID model (~126 MB)
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/lid.176.bin

# 3 — Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4 — Install dependencies
pip install -U pip setuptools wheel
pip install "Cython>=0.29"
pip install --no-build-isolation youtokentome==1.0.6
PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt

# 5 — Create runtime directories
mkdir -p ingestion merged

# 6 — Download ASR models (both Catalan and Spanish)
python scripts/download_models.py --lang all
```

Then run directly:

```bash
# Single recording
python pipeline_service.py --input-id my_recording --lang es

# Batch from file (one ID per line)
python pipeline_service.py --input-id-file /path/to/ids.txt --lang es

# Batch mode (all pairs in ingestion/)
python pipeline_service.py --lang es
```

---

## Pipeline Stages

The orchestrator (`pipeline_service.py`) runs these steps in order:

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | **Normalize TSV** | `scripts/normalize_tsv.py` | `inputs/normalized/<id>/<id>_norm_mark.tsv` |
| 2 | **Normalize audio** | `scripts/normalize_audio.py` | `inputs/normalized/<id>/<id>.wav` (16 kHz mono) |
| 3 | **Generate final data** | `steps/generate_final_data.py` | Forced alignment → segmentation → per‑language ASR |
| 4 | **Duration filter** | `scripts/duration_filter.py` | Removes segments < 2 s or > 30 s |
| 5 | **ROVER merge** | `scripts/rover_merge.py` | Merged `rover_text` + CER/WER analytics |

---

## ASR Models

### Spanish (`--lang es`)

| Model | Type | Source |
|---|---|---|
| `stt_es_conformer_ctc_large` | NeMo CTC | NVIDIA NGC |
| `parakeet-rnnt-1.1b_cv17_es_ep18_1270h` | NeMo RNNT | HuggingFace (projecte‑aina) |
| `stt_es_conformer_transducer_large` | NeMo RNNT | HuggingFace (nvidia) |
| `whisper-large-v3` | Whisper | HuggingFace (openai) |

### Catalan (`--lang ca`)

| Model | Type | Source |
|---|---|---|
| `stt_ca_conformer_ctc_large` | NeMo CTC | NVIDIA NGC |
| `whisper-large-v3-ca-3catparla` | Whisper | HuggingFace (projecte‑aina) |
| `whisper-bsc-large-v3-cat` | Whisper | HuggingFace (langtech‑veu) |
| `whisper-large-v3-ca-punctuated-3370h` | Whisper | HuggingFace (langtech‑veu) |
| `stt_ca-es_conformer_transducer_large` | NeMo RNNT | HuggingFace (projecte‑aina) |

### Shared (both languages)

| Model | Type | Source |
|---|---|---|
| `lid.176.bin` | FastText language‑ID | [B2Drop](https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN) |

---

## Output Files

After processing, results appear in `merged/`:

```
merged/
  final_output_<input-id>.json   # Word-level aligned JSON with rover_text
  final_output_<input-id>.csv    # Per-segment CER/WER table
  final_output_<input-id>.png    # CER bar chart
```

---

## Directory Structure

```
.
├── ingestion/                  # ← Place input WAV + TSV pairs here
├── inputs/
│   ├── manifest/               # NeMo manifests (auto-generated)
│   ├── normalized/             # Canonical WAV + normalized TSV + metadata
│   ├── output_segment/         # Sentence-level clips + final JSON
│   └── wordlevel_alignment/    # ASS + CTM alignment files
├── merged/                     # ← Final outputs appear here (.json, .csv, .png)
├── utils/models/               # ← Pre-downloaded models (mounted at runtime)
│   ├── lid.176.bin             #   FastText language-ID model
│   ├── nemo/                   #   NeMo .nemo checkpoints
│   └── huggingface/            #   HuggingFace model snapshots
├── scripts/
│   ├── download_models.py      # Download models for offline use
│   ├── normalize_tsv.py        # Text normalization
│   ├── normalize_audio.py      # Audio resampling
│   ├── duration_filter.py      # Remove too-short/long segments
│   └── rover_merge.py          # Multi-hypothesis merging
├── steps/
│   └── generate_final_data.py  # Forced alignment + segmentation + ASR
├── pipeline_service.py         # Main orchestrator
├── Dockerfile                  # Docker image definition
├── build_singularity.sh        # Docker → Singularity converter
└── run_singularity.sh          # Singularity run wrapper
```

---

## Script Reference

Run any script with `-h` / `--help` for full argument documentation.

### `pipeline_service.py`

| Argument | Description |
|---|---|
| `--input-id` | Process a single audio‑transcript pair (optional; omit for batch mode) |
| `--input-id-file` | Path to file with one input ID per line (for batch processing a subset) |
| `--lang` | `ca` or `es` (default: `ca`) |
| `--max-duration` | Maximum segment duration in seconds (default: `30`) |
| `--lid-model-path` | FastText language-ID model file (default: `$LID_MODEL_PATH` or `utils/models/lid.176.bin`) |
| `--nemo-model-dir` | Directory with local NeMo checkpoints (default: `$NEMO_MODEL_DIR` or `utils/models/nemo`) |
| `--hf-model-dir` | HuggingFace cache root (default: `$HF_MODEL_DIR` or `utils/models/huggingface`) |

### `scripts/download_models.py`

| Argument | Description |
|---|---|
| `--lang` | `ca`, `es`, or `all` (default: `all`) |
| `--out-dir` | Root output directory (default: `$MODEL_DIR` or `utils/models`) |

### `steps/generate_final_data.py`

| Argument | Description |
|---|---|
| `--input-id` | WAV + TSV filename stem (required) |
| `--lang` | `ca` or `es` (default: `ca`) |
| `--output` | Custom JSON filename |
| `--device` | `cuda`, `cpu`, or `auto` (default: `auto`) |
| `--lid-model-path` | FastText language-ID model file (default: `$LID_MODEL_PATH` or `utils/models/lid.176.bin`) |
| `--nemo-model-dir` | Directory with local NeMo checkpoints (default: `$NEMO_MODEL_DIR` or `utils/models/nemo`) |
| `--hf-model-dir` | HuggingFace cache root (default: `$HF_MODEL_DIR` or `utils/models/huggingface`) |

### `scripts/rover_merge.py`

| Argument | Description |
|---|---|
| `input_glob` | Single JSON file or quoted glob pattern |
| `-o`, `--out-dir` | Destination folder (default: `merged/`) |
| `--csv` | Emit per‑segment CSV alongside the merged JSON |
| `--plot` | Save corpus‑level CER bar chart (`.png`) |

---

## Troubleshooting

**"ffmpeg: command not found"**

```bash
sudo apt install ffmpeg
```

**Out of memory**

- Add swap space:
  ```bash
  sudo fallocate -l 24G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```
- Or force CPU‑only execution:
  ```bash
  CUDA_VISIBLE_DEVICES= python pipeline_service.py --lang es
  ```

**Docker commands require `sudo`**

```bash
sudo usermod -aG docker $USER
# Log out and back in (or run: newgrp docker)
```

**Model download fails inside Docker**

Pre‑download on the host and mount:
```bash
docker run --rm -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang all
```

**`lid.176.bin cannot be opened for loading!`**

The `lid.176.bin` file must exist on the **host** inside `utils/models/`. When you mount
`utils/models/` into the container, it overwrites the copy that was downloaded during image build:
```bash
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/lid.176.bin
```

**Permission denied on `utils/models/`, `inputs/`, or `merged/`**

If you ran Docker without `--user $(id -u):$(id -g)`, files may be owned by root. Fix retroactively:
```bash
sudo chown -R $(whoami) utils/models/ inputs/ merged/
```
To prevent this, always include `--user $(id -u):$(id -g)` in `docker run` commands.
The `run_singularity.sh` script detects and auto-fixes this.

**Singularity build fails with "no space left on device"**

Apptainer uses `/tmp` for intermediate files. If your `/tmp` partition is small, `build_singularity.sh` 
automatically redirects to `.apptainer_tmp/` in the project directory. If it still fails, 
free up disk space or set `APPTAINER_TMPDIR` to a partition with ≥ 10 GB free.

---

## Notes

- Forced alignment always runs on CPU regardless of `--device` setting.
- Batch mode continues processing even if an individual input fails.
- Ensure ≥ 24 GB effective memory (RAM + swap) before processing long recordings.
- The Docker image does **not** contain models — they are always mounted from the host at runtime.
