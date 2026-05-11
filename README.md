# Found-Speech Pipeline

## Introduction

The **Found-Speech Pipeline** processes **audio-transcript pairs** (`.wav` + `.tsv`) into clean, word-level aligned outputs with CER/WER analytics and final merged segment text.

Supported languages: **Catalan** (`ca`), **Spanish** (`es`), **Basque** (`eu`), and **Galician** (`gl`).

---

## Hardware & OS Prerequisites

| Requirement | Notes |
|---|---|
| >= 24 GB RAM (or RAM + swap) | Whisper-large-v3 and forced alignment are memory-hungry |
| Ubuntu 22.04+ (native, WSL 2, or Docker) | Verified host for NVIDIA NeMo |
| FFmpeg on `$PATH` | Transcodes audio to 16 kHz mono WAV (included in Docker image) |
| (Optional) CUDA 11+ | Speeds up Whisper and RNNT by ~4x |

> [!NOTE]
> **WSL users:** Open **Windows Terminal -> Settings -> `.wslconfig`** and set `memory=24GB`, then run `wsl --shutdown`.

---

## Input Format

Place matching `.wav` and `.tsv` files inside the `ingestion/` directory:

```text
ingestion/
  my_recording.wav
  my_recording.tsv
```

Each `.tsv` must contain **exactly one line** in this format:

```text
/path/to/my_recording.wav<TAB>Full transcript text here.
```

> [!IMPORTANT]
> - The WAV filename and TSV filename must share the same stem, for example `my_recording`.
> - The TSV must contain a single row with two tab-separated columns: WAV path and transcript text.
> - The first TSV column is kept as metadata, but the pipeline loads audio by input ID from `ingestion/<input-id>.wav`.

---

## How The Pipeline Runs

The pipeline supports both single-input runs and batch execution.

For a single input:
- one `input_id` is processed from `ingestion/<input_id>.wav` and `ingestion/<input_id>.tsv`
- intermediate files are written under `inputs/`
- final outputs are written under `merged/`

For larger runs:
- inputs are split into balanced bucket files based on audio file size
- each bucket file contains one `input_id` per line
- one job is launched per bucket
- each job processes all inputs listed in its bucket file

Inside each job, the processing flow is:
1. Prepare the input transcript and audio.
2. Align the transcript against the audio.
3. Split the audio into segments.
4. Generate segment-level ASR hypotheses.
5. Filter unusable segments.
6. Merge hypotheses into final segment text.
7. Restore punctuation and casing.

This means the pipeline now works at two levels:
- full-input processing for each audio-transcript pair
- segment-level processing after alignment

---

## Quick Start (Docker - recommended)

### Step 1. Build the Docker image

```bash
./deployment/build/build_docker.sh
```

### Step 2. Download models (once, on host)

All models are stored **outside** the image in `utils/models/` and mounted at runtime.
This keeps the image lean and lets you run fully offline after downloading.

```bash
# Create model directories
mkdir -p utils/models/fasttext utils/models

# Download the FastText language-ID model (~126 MB)
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/fasttext/lid.176.bin

# Download ASR models for Catalan + Spanish (~15 GB)
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang all
```

This download step only prepares the Catalan/Spanish ASR models handled by
`scripts/download_models.py`. Basque, Galician, punctuation, and the optional
Galician sidecar require extra assets not covered by this command.

To download only one language instead:

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang es

docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang ca
```

> [!TIP]
> The `--user $(id -u):$(id -g)` flag ensures files created by Docker are owned by **your user**
> instead of root, avoiding permission issues later.

### Step 3. Run the pipeline

Direct run modes:

```bash
# Process a single recording
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --input-id my_recording --lang es

# Process a prepared file containing one input_id per line
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --input-id-file /app/inputs/bucket_1.txt --lang es

# Batch mode: process all WAV+TSV pairs in ingestion/
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/ingestion:/app/ingestion \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/merged:/app/merged \
  -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python pipeline_service.py --lang es
```

If your models live outside the repository, pass explicit runtime paths:
- `--lid-model-path /path/to/fasttext/lid.176.bin`
- `--nemo-model-dir /path/to/model-root`
- `--hf-model-dir /path/to/model-root`

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

## Singularity / Apptainer

For environments where Docker is not available, you can convert the Docker image to a Singularity or Apptainer `.sif` file.

**Prerequisites:**
- Docker image already built
- Models already downloaded
- `singularity` or `apptainer` installed on the target machine

### Build the `.sif` image

```bash
./deployment/build/build_singularity.sh
```

This creates the main pipeline `.sif` in the repository root.

### Run with Singularity

```bash
# Single recording
./deployment/run/run_singularity_image.sh --input-id my_recording --lang es

# Batch from file (one input_id per line)
./deployment/run/run_singularity_image.sh --input-id-file /path/to/ids.txt --lang es

# Batch mode (all pairs in ingestion/)
./deployment/run/run_singularity_image.sh --lang es
```

The container runner mounts:
- `ingestion/`
- `inputs/`
- `merged/`
- the shared model root

### Bucket-based batch launching

For larger runs, the normal pattern is:

```bash
./deployment/run/create_splits_run_pipeline.sh -n 10 -o input_ids -l es
```

This command:
- scans the inputs
- creates balanced `bucket_*.txt` files
- writes one `input_id` per line
- launches one job per bucket

### Runtime `.env` configuration

If present, `.env` can be used to define runtime paths such as:

```bash
SIF=/path/to/fsp-multilingual-pipeline.sif
MODELS_ROOT=/path/to/models
LID_MODEL_PATH=/path/to/models/fasttext/lid.176.bin
NEMO_MODEL_DIR=/path/to/models
HF_MODEL_DIR=/path/to/models
FSP_INGESTION_DIR=/path/to/ingestion
FSP_INPUTS_DIR=/path/to/inputs
FSP_MERGED_DIR=/path/to/merged
FSP_CACHE_DIR=/path/to/cache
```

---

## Installation (native / WSL - without Docker)

If you prefer to run without containers:

```bash
# 1 - Create model directories
mkdir -p utils/models/fasttext utils/models

# 2 - Download the FastText language-ID model (~126 MB)
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/fasttext/lid.176.bin

# 3 - Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4 - Install dependencies
pip install -U pip setuptools wheel
pip install "Cython>=0.29"
pip install --no-build-isolation youtokentome==1.0.6
PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt

# 5 - Create runtime directories
mkdir -p ingestion merged

# 6 - Download ASR models (both Catalan and Spanish)
python scripts/download_models.py --lang all
```

Then run directly:

```bash
# Single recording
python pipeline_service.py --input-id my_recording --lang es

# Batch from file (one input_id per line)
python pipeline_service.py --input-id-file /path/to/ids.txt --lang es

# Batch mode (all pairs in ingestion/)
python pipeline_service.py --lang es
```

---

## Pipeline Stages

The pipeline runs these stages in order:

| # | Stage | Output |
|---|---|---|
| 1 | **Normalize TSV** | `inputs/normalized/<id>/<id>_norm_mark.tsv` |
| 2 | **Normalize audio** | `inputs/normalized/<id>/<id>.wav` (16 kHz mono) |
| 3 | **Alignment + segment generation** | `inputs/output_segment/final_output_<id>.json` |
| 4 | **Optional GL extra ASR** | Adds extra Galician hypotheses when `--lang gl` |
| 5 | **Duration filter** | Removes segments outside the allowed duration range |
| 6 | **ROVER merge** | Merged `rover_text` and scoring outputs in `merged/` |
| 7 | **Punctuation** | Adds `text_normalized` and `text_punctuated` to the merged JSON |

### Normalization

Text normalization is language-specific:

- `ca` and `es` use in-process dictionaries plus number expansion
- `eu` uses the external `modulo1y2` normalizer
- `gl` uses the external `cotovia` normalizer

### Language ID

Language ID is part of the filtering logic, not just metadata. Segments can be
dropped when the detected language differs from the selected pipeline language
or when confidence is too low.

### GL sidecar

When running with `--lang gl`, the pipeline can optionally launch a separate
Galician-only enrichment step after the main ASR pass. This adds extra GL
hypotheses before duration filtering and ROVER. You can disable it with
`--skip-gl-extra-asr`.

---

## Output Files

After processing, the pipeline uses four main locations:
- `ingestion/` for raw input pairs
- `inputs/` for intermediate and audit artifacts
- `merged/` for final outputs
- `slurm_output/` for batch-job stdout and stderr

If you want:
- final deliverables, open `merged/`
- intermediate files or debugging artifacts, open `inputs/`
- job logs or failure logs, open `slurm_output/`
- to verify the original provided files, open `ingestion/`

Main final outputs in `merged/`:

```text
merged/
  final_output_<input-id>.json   # Main final structured result
  final_output_<input-id>.csv    # Per-segment CER/WER table
  final_output_<input-id>.png    # CER comparison plot
```

Intermediate and audit outputs under `inputs/`:

```text
inputs/
  normalized/<id>/<id>_norm_mark.tsv      # Normalized transcript TSV
  normalized/<id>/<id>.wav                # Canonical 16 kHz mono WAV
  normalized/<id>/<id>_metadata.json      # Audio/text metadata
  manifest/<id>_manifest.json             # Manifest for alignment/inference
  wordlevel_alignment/<id>/               # Forced-alignment artifacts
  output_segment/<id>/                    # Segment WAV clips
  output_segment/final_output_<id>.json   # Intermediate enriched JSON
  dropped_segments/<id>_<lang>_*.json     # Dropped-segment audit log
  logs/                                   # Per-run pipeline logs
```

Batch-job logs under `slurm_output/`:

```text
slurm_output/
  fsp-pipeline_<jobid>.out
  fsp-pipeline_<jobid>.err
```

---

## Directory Structure

```text
.
├── ingestion/                  # Place input WAV + TSV pairs here
├── inputs/
│   ├── manifest/              # Alignment and inference manifests
│   ├── normalized/            # Canonical WAV + normalized TSV + metadata
│   ├── output_segment/        # Segment clips + intermediate JSON
│   ├── wordlevel_alignment/   # Alignment artifacts
│   ├── dropped_segments/      # Audit logs for filtered segments
│   └── logs/                  # Per-run pipeline logs
├── merged/                    # Final outputs (.json, .csv, .png)
├── slurm_output/              # Batch job stdout/stderr
├── utils/models/              # Shared model root
├── scripts/                   # Helper scripts
├── steps/                     # Standalone pipeline stages
├── deployment/                # Build and runtime wrappers
└── pipeline_service.py        # Main orchestrator
```

---

## Script Reference

Run any script with `-h` / `--help` for full argument documentation.

### `pipeline_service.py`

| Argument | Description |
|---|---|
| `--input-id` | Process a single audio-transcript pair |
| `--input-id-file` | Path to file with one input ID per line |
| `--lang` | `ca`, `es`, `eu`, or `gl` (default: `ca`) |
| `--max-duration` | Maximum segment duration in seconds (default: `30`) |
| `--min-duration` | Minimum segment duration in seconds (default: `2`) |
| `--device` | `auto`, `cuda`, or `cpu` for ASR execution |
| `--lid-model-path` | FastText language-ID model file |
| `--nemo-model-dir` | Directory with local NeMo checkpoints |
| `--hf-model-dir` | HuggingFace cache root |
| `--skip-gl-extra-asr` | Skip the optional Galician sidecar enrichment step |

### `scripts/download_models.py`

| Argument | Description |
|---|---|
| `--lang` | `ca`, `es`, or `all` (default: `all`) |
| `--out-dir` | Root output directory |

### `steps/generate_final_data.py`

| Argument | Description |
|---|---|
| `--input-id` | WAV + TSV filename stem |
| `--lang` | `ca`, `es`, `eu`, or `gl` (default: `ca`) |
| `--output` | Custom JSON filename |
| `--device` | `cuda`, `cpu`, or `auto` (default: `auto`) |
| `--lid-model-path` | FastText language-ID model file |
| `--nemo-model-dir` | Directory with local NeMo checkpoints |
| `--hf-model-dir` | HuggingFace cache root |

### `scripts/rover_merge.py`

| Argument | Description |
|---|---|
| `input_glob` | Single JSON file or quoted glob pattern |
| `-o`, `--out-dir` | Destination folder (default: `merged/`) |
| `--fields` | Explicit `norm_*` fields to merge instead of auto-detecting them |
| `--langs` | Language codes to keep during merge |
| `--norm` | Normalize text before scoring |
| `--strategy` | Merge strategy: `centroid` or `vote` |
| `--csv` | Emit per-segment CSV alongside the merged JSON |
| `--plot` | Save corpus-level CER bar chart (`.png`) |

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
- Or force CPU-only execution:
  ```bash
  CUDA_VISIBLE_DEVICES= python pipeline_service.py --lang es
  ```

**Docker commands require `sudo`**

```bash
sudo usermod -aG docker $USER
# Log out and back in (or run: newgrp docker)
```

**Model download fails inside Docker**

Pre-download on the host and mount:

```bash
docker run --rm -v $(pwd)/utils/models:/app/utils/models \
  fsp-pipeline python scripts/download_models.py --lang all
```

**`lid.176.bin` cannot be opened**

The `lid.176.bin` file must exist on the host inside `utils/models/fasttext/`.
When `utils/models/` is mounted into the container, it overrides the image copy.

```bash
wget -q "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" \
  -O utils/models/fasttext/lid.176.bin
```

**Permission denied on `utils/models/`, `inputs/`, or `merged/`**

If you ran Docker without `--user $(id -u):$(id -g)`, files may be owned by root.
Fix retroactively:

```bash
sudo chown -R $(whoami) utils/models/ inputs/ merged/
```

To prevent this, always include `--user $(id -u):$(id -g)` in `docker run` commands.

**Singularity build fails with "no space left on device"**

Apptainer uses `/tmp` for intermediate files. If `/tmp` is small,
`deployment/build/build_singularity.sh` redirects to a project-local temp directory.
If it still fails, free disk space or set `APPTAINER_TMPDIR` to a location with
at least 10 GB free.

---

## Notes

- Forced alignment always runs on CPU regardless of `--device`.
- Batch mode continues processing even if an individual input fails.
- Ensure >= 24 GB effective memory (RAM + swap) before processing long recordings.
- The Docker image does **not** contain models. They are mounted from the host at runtime.
