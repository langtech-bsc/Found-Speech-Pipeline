#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_singularity_image.sh
# Run the FSP pipeline inside a Singularity/Apptainer container on HPC,
# either using only the code baked into the .sif image or overlaying the
# current checkout for quick local testing.
#
# Examples:
#   ./deployment/run/run_singularity_image.sh --input-id PKuuatqwz00 --lang es
#   ./deployment/run/run_singularity_image.sh --code-source repo --input-id PKuuatqwz00 --lang es
#   ./deployment/run/run_singularity_image.sh --lang ca
#
# Optional environment overrides:
#   export FSP_INGESTION_DIR=/path/to/ingestion
#   export FSP_INPUTS_DIR=/path/to/inputs
#   export FSP_MERGED_DIR=/path/to/merged
#   export MODELS_ROOT=/path/to/models
#   export LID_MODEL_PATH=/path/to/lid.176.bin
#   export FSP_CACHE_DIR=/path/to/cache
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIF="${REPO_ROOT}/fsp-pipeline.sif"
DEFAULT_MODELS_ROOT="${REPO_ROOT}/utils/models"
CODE_SOURCE="image"
PIPELINE_ARGS=()

print_help() {
    cat <<'EOF'
Usage:
  ./deployment/run/run_singularity_image.sh [--code-source image|repo] [pipeline args]

Modes:
  --code-source image   Run the code baked into the .sif image (default)
  --code-source repo    Overlay the current checkout into the container so
                        local repo changes can be tested without rebuilding

Examples:
  ./deployment/run/run_singularity_image.sh --input-id my_recording --lang es
  ./deployment/run/run_singularity_image.sh --code-source repo --input-id my_recording --lang es
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --code-source)
            CODE_SOURCE="${2:-}"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            PIPELINE_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${CODE_SOURCE}" != "image" && "${CODE_SOURCE}" != "repo" ]]; then
    echo "ERROR: --code-source must be 'image' or 'repo'." >&2
    exit 1
fi

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

INGESTION_DIR="${FSP_INGESTION_DIR:-${REPO_ROOT}/ingestion}"
INPUTS_DIR="${FSP_INPUTS_DIR:-${REPO_ROOT}/inputs}"
MERGED_DIR="${FSP_MERGED_DIR:-${REPO_ROOT}/merged}"

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

CODE_BINDS=()
if [[ "${CODE_SOURCE}" == "repo" ]]; then
    CODE_BINDS=(
        --bind "${REPO_ROOT}/fsp:/app/fsp:ro"
        --bind "${REPO_ROOT}/NeMo:/app/NeMo:ro"
        --bind "${REPO_ROOT}/steps:/app/steps:ro"
        --bind "${REPO_ROOT}/scripts:/app/scripts:ro"
        --bind "${REPO_ROOT}/pipeline_service.py:/app/pipeline_service.py:ro"
    )
fi

echo "==========================================================="
echo "Running FSP pipeline via ${RUNNER}"
echo "Image:        ${SIF}"
echo "Code source:  ${CODE_SOURCE}"
echo "Ingestion:    ${INGESTION_DIR}"
echo "Inputs:       ${INPUTS_DIR}"
echo "Merged:       ${MERGED_DIR}"
echo "LID model:    ${LID_MODEL_PATH_HOST}"
echo "Model root:   ${MODEL_ROOT_HOST}"
echo "HF cache:     ${HF_CACHE_DIR}"
echo "Torch cache:  ${TORCH_CACHE_DIR}"
echo "TMPDIR:       ${TMPDIR_HOST}"
echo "Args:         ${PIPELINE_ARGS[*]:-<batch mode>}"
echo "==========================================================="

exec "${RUNNER}" exec ${GPU_FLAG} --cleanenv \
    --bind "${INGESTION_DIR}:/app/ingestion" \
    --bind "${INPUTS_DIR}:/app/inputs" \
    --bind "${MERGED_DIR}:/app/merged" \
    --bind "${MODEL_ROOT_HOST}:/app/model_root:ro" \
    --bind "${TMPDIR_HOST}:/tmp" \
    "${CODE_BINDS[@]}" \
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
    python /app/pipeline_service.py "${PIPELINE_ARGS[@]}"
