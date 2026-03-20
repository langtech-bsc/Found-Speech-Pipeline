#!/usr/bin/env bash
# ===========================================================================
# run_singularity.sh
# ===========================================================================
# Run the FSP pipeline inside a Singularity/Apptainer container.
#
# Usage:
#   ./run_singularity.sh --input-id PKuuatqwz00 --lang es
#   ./run_singularity.sh --input-id-file /path/to/ids.txt --lang es   # batch from file
#   ./run_singularity.sh --lang ca              # batch mode (all pairs in ingestion/)
#
# The script automatically binds:
#   ingestion/      -> /app/ingestion      (input WAV+TSV)
#   inputs/         -> /app/inputs         (intermediate files)
#   merged/         -> /app/merged         (final output)
#   ${LID_MODEL_PATH:-utils/models/lid.176.bin} -> /app/utils/models/lid.176.bin
#   ${NEMO_MODEL_DIR:-utils/models/nemo} -> /app/utils/models/nemo
#   ${HF_MODEL_DIR:-utils/models/huggingface} -> /app/utils/models/huggingface
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF="${SCRIPT_DIR}/fsp-pipeline.sif"
DEFAULT_MODELS_ROOT="/gpfs/projects/bsc88/speech/ASR/models"

# Detect Singularity or Apptainer
if command -v apptainer &>/dev/null; then
    RUNNER=apptainer
elif command -v singularity &>/dev/null; then
    RUNNER=singularity
else
    echo "Neither 'singularity' nor 'apptainer' found."
    exit 1
fi

# Ensure the .sif exists
if [[ ! -f "${SIF}" ]]; then
    echo "Singularity image not found: ${SIF}"
    echo "   Run ./build_singularity.sh first."
    exit 1
fi

INGESTION_DIR="${FSP_INGESTION_DIR:-${SCRIPT_DIR}/ingestion}"
INPUTS_DIR="${FSP_INPUTS_DIR:-${SCRIPT_DIR}/inputs}"
MERGED_DIR="${FSP_MERGED_DIR:-${SCRIPT_DIR}/merged}"

MODELS_ROOT_HOST="${MODELS_ROOT:-${DEFAULT_MODELS_ROOT}}"
LID_MODEL_PATH_HOST="${LID_MODEL_PATH:-${MODELS_ROOT_HOST}/fasttext/lid.176.bin}"
NEMO_MODEL_DIR_HOST="${NEMO_MODEL_DIR:-${MODELS_ROOT_HOST}}"
HF_MODEL_DIR_HOST="${HF_MODEL_DIR:-${MODELS_ROOT_HOST}}"

# Create output directories if they don't exist
mkdir -p "${INPUTS_DIR}" "${MERGED_DIR}"

# Fail fast on bad paths instead of trying sudo/chown fixes.
[[ -d "${INGESTION_DIR}" ]] || { echo "Ingestion directory not found: ${INGESTION_DIR}"; exit 1; }
[[ -w "${INPUTS_DIR}" ]] || { echo "Inputs directory is not writable: ${INPUTS_DIR}"; exit 1; }
[[ -w "${MERGED_DIR}" ]] || { echo "Merged directory is not writable: ${MERGED_DIR}"; exit 1; }
[[ -f "${LID_MODEL_PATH_HOST}" ]] || { echo "Language-ID model not found: ${LID_MODEL_PATH_HOST}"; exit 1; }
[[ -d "${NEMO_MODEL_DIR_HOST}" ]] || { echo "NeMo model directory not found: ${NEMO_MODEL_DIR_HOST}"; exit 1; }
[[ -d "${HF_MODEL_DIR_HOST}" ]] || { echo "HF model directory not found: ${HF_MODEL_DIR_HOST}"; exit 1; }

# GPU support if available
GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FLAG="--nv"
fi

echo "==========================================================="
echo "Running FSP pipeline via ${RUNNER}"
echo "Image: ${SIF}"
echo "LID model: ${LID_MODEL_PATH_HOST}"
echo "NeMo dir: ${NEMO_MODEL_DIR_HOST}"
echo "HF dir: ${HF_MODEL_DIR_HOST}"
echo "Args: ${*:-<batch mode>}"
echo "==========================================================="

exec "${RUNNER}" exec ${GPU_FLAG} \
    --bind "${INGESTION_DIR}:/app/ingestion" \
    --bind "${INPUTS_DIR}:/app/inputs" \
    --bind "${MERGED_DIR}:/app/merged" \
    --bind "${SCRIPT_DIR}/NeMo:/app/NeMo:ro" \
    --bind "${SCRIPT_DIR}/fsp:/app/fsp:ro" \
    --bind "${SCRIPT_DIR}/pipeline_service.py:/app/pipeline_service.py:ro" \
    --bind "${LID_MODEL_PATH_HOST}:/app/utils/models/lid.176.bin:ro" \
    --bind "${NEMO_MODEL_DIR_HOST}:/app/utils/models/nemo:ro" \
    --bind "${HF_MODEL_DIR_HOST}:/app/utils/models/huggingface:ro" \
    --env MODELS_ROOT=/app/utils/models \
    --env MODEL_DIR=/app/utils/models \
    --env LID_MODEL_PATH=/app/utils/models/lid.176.bin \
    --env NEMO_MODEL_DIR=/app/utils/models/nemo \
    --env HF_MODEL_DIR=/app/utils/models/huggingface \
    "${SIF}" \
    python /app/pipeline_service.py "$@"
