#!/usr/bin/env bash
# ===========================================================================
# run_singularity.sh
# ===========================================================================
# Run the FSP pipeline inside a Singularity/Apptainer container.
#
# Usage:
#   ./run_singularity.sh --input-id PKuuatqwz00 --lang es --device [cpu/gpu/auto]
#   ./run_singularity.sh --lang ca              # batch mode (all pairs in ingestion/)
#
# The script automatically binds:
#   ingestion/      --> /app/ingestion      (input WAV+TSV)
#   inputs/         --> /app/inputs         (intermediate files)
#   merged/         --> /app/merged         (final output)
#   utils/models/   --> /app/utils/models   (pre-downloaded models)
# ===========================================================================

set -euo pipefail

module purge
module load EB/apps EB/install CUDA/12.4.0
module load singularity

echo "Loaded modules:"
module list

nvidia-smi

# Source environment variable file
source ./.env

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

# Create output directories if they don't exist
mkdir -p "${SCRIPT_DIR}/inputs" "${SCRIPT_DIR}/merged"


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
    --nv \
    --bind "${SCRIPT_DIR}/ingestion:/app/ingestion" \
    --bind "${SCRIPT_DIR}/inputs:/app/inputs" \
    --bind "${SCRIPT_DIR}/merged:/app/merged" \
    --bind "${LID_MODEL_PATH_HOST}:/app/utils/models/fasttext/lid.176.bin" \
    --bind "${NEMO_MODEL_DIR_HOST}:/app/utils/models/nemo" \
    --bind "${HF_MODEL_DIR_HOST}:/app/utils/models/huggingface" \
    --env MODEL_DIR=/app/utils/models \
    --env LID_MODEL_PATH=/app/utils/models/fasttext/lid.176.bin \
    --env NEMO_MODEL_DIR=/app/utils/models/nemo \
    --env HF_MODEL_DIR=/app/utils/models/huggingface \
    "${SIF}" \
    python /app/pipeline_service.py "$@"
