#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_singularity_image.sh
# Run the FSP pipeline inside a Singularity/Apptainer container on HPC,
# using only the code baked into the .sif image.
#
# Examples:
#   ./run_singularity_image.sh --input-id PKuuatqwz00 --lang es
#   ./run_singularity_image.sh --lang ca
#
# Optional environment overrides:
#   export FSP_INGESTION_DIR=/path/to/ingestion
#   export FSP_INPUTS_DIR=/path/to/inputs
#   export FSP_MERGED_DIR=/path/to/merged
#   export MODELS_ROOT=/gpfs/projects/bsc88/speech/ASR/models
#   export LID_MODEL_PATH=/path/to/lid.176.bin
#   export FSP_CACHE_DIR=/scratch/$USER/fsp_cache
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/fsp-pipeline.sif"
DEFAULT_MODELS_ROOT="/gpfs/projects/bsc88/speech/ASR/models"

if command -v apptainer >/dev/null 2>&1; then
    RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNNER=singularity
else
    echo "ERROR: neither 'apptainer' nor 'singularity' was found in PATH." >&2
    exit 1
fi

if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: container image not found: ${SIF}" >&2
    echo "Build it first or point SIF to the correct location." >&2
    exit 1
fi

INGESTION_DIR="${FSP_INGESTION_DIR:-${SCRIPT_DIR}/ingestion}"
INPUTS_DIR="${FSP_INPUTS_DIR:-${SCRIPT_DIR}/inputs}"
MERGED_DIR="${FSP_MERGED_DIR:-${SCRIPT_DIR}/merged}"

MODELS_ROOT_HOST="${MODELS_ROOT:-${DEFAULT_MODELS_ROOT}}"
MODEL_ROOT_HOST="${MODEL_DIR:-${MODELS_ROOT_HOST}}"
LID_MODEL_PATH_HOST="${LID_MODEL_PATH:-${MODEL_ROOT_HOST}/fasttext/lid.176.bin}"

DEFAULT_CACHE_ROOT="${FSP_CACHE_DIR:-${SCRATCH:-${HOME}}/fsp_cache}"
TMPDIR_HOST="${TMPDIR:-${DEFAULT_CACHE_ROOT}/tmp}"
HF_CACHE_DIR="${HF_HOME:-${TMPDIR_HOST}/huggingface}"
TORCH_CACHE_DIR="${TORCH_HOME:-${TMPDIR_HOST}/torch}"
XDG_CACHE_HOME_DIR="${XDG_CACHE_HOME:-${TMPDIR_HOST}/xdg-cache}"
MPLCONFIGDIR_HOST="${MPLCONFIGDIR:-${TMPDIR_HOST}/matplotlib}"

mkdir -p \
    "${INPUTS_DIR}" \
    "${MERGED_DIR}" \
    "${HF_CACHE_DIR}" \
    "${TORCH_CACHE_DIR}" \
    "${XDG_CACHE_HOME_DIR}" \
    "${MPLCONFIGDIR_HOST}" \
    "${TMPDIR_HOST}"

check_readable_file() {
    local p="$1"
    local label="$2"
    if [[ ! -f "$p" ]]; then
        echo "ERROR: ${label} file not found: $p" >&2
        exit 1
    fi
    if [[ ! -r "$p" ]]; then
        echo "ERROR: ${label} file is not readable: $p" >&2
        exit 1
    fi
}

check_readable_dir() {
    local p="$1"
    local label="$2"
    if [[ ! -d "$p" ]]; then
        echo "ERROR: ${label} directory not found: $p" >&2
        exit 1
    fi
    if [[ ! -r "$p" || ! -x "$p" ]]; then
        echo "ERROR: ${label} directory is not accessible: $p" >&2
        exit 1
    fi
}

check_writable_dir() {
    local p="$1"
    local label="$2"
    if [[ ! -d "$p" ]]; then
        echo "ERROR: ${label} directory not found: $p" >&2
        exit 1
    fi
    if [[ ! -w "$p" ]]; then
        echo "ERROR: ${label} directory is not writable: $p" >&2
        echo "Use a path under \$HOME, \$SCRATCH, or another directory you own." >&2
        exit 1
    fi
}

check_readable_dir "${INGESTION_DIR}" "ingestion"
check_writable_dir "${INPUTS_DIR}" "inputs"
check_writable_dir "${MERGED_DIR}" "merged"
check_readable_file "${LID_MODEL_PATH_HOST}" "LID model"
check_readable_dir "${MODEL_ROOT_HOST}" "shared model root"
check_writable_dir "${TMPDIR_HOST}" "TMPDIR"

GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FLAG="--nv"
fi

echo "==========================================================="
echo "Running FSP pipeline via ${RUNNER} (image-only code)"
echo "Image:        ${SIF}"
echo "Ingestion:    ${INGESTION_DIR}"
echo "Inputs:       ${INPUTS_DIR}"
echo "Merged:       ${MERGED_DIR}"
echo "LID model:    ${LID_MODEL_PATH_HOST}"
echo "Model root:   ${MODEL_ROOT_HOST}"
echo "HF cache:     ${HF_CACHE_DIR}"
echo "Torch cache:  ${TORCH_CACHE_DIR}"
echo "TMPDIR:       ${TMPDIR_HOST}"
echo "Args:         ${*:-<batch mode>}"
echo "==========================================================="

exec "${RUNNER}" exec ${GPU_FLAG} --cleanenv \
    --bind "${INGESTION_DIR}:/app/ingestion" \
    --bind "${INPUTS_DIR}:/app/inputs" \
    --bind "${MERGED_DIR}:/app/merged" \
    --bind "${MODEL_ROOT_HOST}:/app/model_root:ro" \
    --bind "${TMPDIR_HOST}:/tmp" \
    --env MODELS_ROOT=/app/model_root \
    --env MODEL_DIR=/app/model_root \
    --env LID_MODEL_PATH=/app/model_root/fasttext/lid.176.bin \
    --env NEMO_MODEL_DIR=/app/model_root \
    --env HF_MODEL_DIR=/app/model_root \
    --env HF_HOME=/tmp/huggingface \
    --env HF_HUB_CACHE=/tmp/huggingface/hub \
    --env TRANSFORMERS_CACHE=/tmp/huggingface/hub \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TORCH_HOME=/tmp/torch \
    --env XDG_CACHE_HOME=/tmp/xdg-cache \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --env TMPDIR=/tmp \
    "${SIF}" \
    python /app/pipeline_service.py "$@"
