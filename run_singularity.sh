#!/usr/bin/env bash
# ===========================================================================
# run_singularity.sh
# ===========================================================================
# Run the FSP pipeline inside a Singularity/Apptainer container.
#
# Usage:
#   ./run_singularity.sh --input-id PKuuatqwz00 --lang es
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

LID_MODEL_PATH_HOST="${LID_MODEL_PATH:-${SCRIPT_DIR}/utils/models/lid.176.bin}"
NEMO_MODEL_DIR_HOST="${NEMO_MODEL_DIR:-${SCRIPT_DIR}/utils/models/nemo}"
HF_MODEL_DIR_HOST="${HF_MODEL_DIR:-${SCRIPT_DIR}/utils/models/huggingface}"

# Create output directories if they don't exist
mkdir -p "${SCRIPT_DIR}/inputs" "${SCRIPT_DIR}/merged"

# Pre-flight: fix root-owned files from previous Docker runs
# Singularity runs as the current user, so root-owned files will cause PermissionErrors
for dir in inputs merged; do
    target="${SCRIPT_DIR}/${dir}"
    if [[ -d "${target}" ]] && find "${target}" -maxdepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
        echo "Fixing permissions on ${dir}/ (root-owned files from Docker)..."
        sudo chown -R "$(whoami)" "${target}"
    fi
done

if [[ -e "${LID_MODEL_PATH_HOST}" && ! -w "${LID_MODEL_PATH_HOST}" ]]; then
    echo "Fixing permissions on language-ID model ${LID_MODEL_PATH_HOST}..."
    sudo chown "$(whoami)" "${LID_MODEL_PATH_HOST}"
fi

if [[ -d "${NEMO_MODEL_DIR_HOST}" ]] && find "${NEMO_MODEL_DIR_HOST}" -maxdepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
    echo "Fixing permissions on NeMo model directory ${NEMO_MODEL_DIR_HOST}..."
    sudo chown -R "$(whoami)" "${NEMO_MODEL_DIR_HOST}"
fi

if [[ -d "${HF_MODEL_DIR_HOST}" ]] && find "${HF_MODEL_DIR_HOST}" -maxdepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
    echo "Fixing permissions on HuggingFace model directory ${HF_MODEL_DIR_HOST}..."
    sudo chown -R "$(whoami)" "${HF_MODEL_DIR_HOST}"
fi

# Run the pipeline
echo "==========================================================="
echo "Running FSP pipeline via ${RUNNER}"
echo "Image: ${SIF}"
echo "LID model: ${LID_MODEL_PATH_HOST}"
echo "NeMo dir: ${NEMO_MODEL_DIR_HOST}"
echo "HF dir: ${HF_MODEL_DIR_HOST}"
echo "Args: ${*:-<batch mode>}"
echo "==========================================================="

${RUNNER} exec \
    --bind "${SCRIPT_DIR}/ingestion:/app/ingestion" \
    --bind "${SCRIPT_DIR}/inputs:/app/inputs" \
    --bind "${SCRIPT_DIR}/merged:/app/merged" \
    --bind "${LID_MODEL_PATH_HOST}:/app/utils/models/lid.176.bin" \
    --bind "${NEMO_MODEL_DIR_HOST}:/app/utils/models/nemo" \
    --bind "${HF_MODEL_DIR_HOST}:/app/utils/models/huggingface" \
    "${SIF}" \
    python /app/pipeline_service.py "$@"
