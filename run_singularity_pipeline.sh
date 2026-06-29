#!/bin/bash
#SBATCH --job-name=fsp-singularity
##SBATCH --account=<your-account>
##SBATCH --qos=<your-qos>
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

INPUT_ID="${1:-dtTCbYIHLps}"
LANG="${2:-eu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
DEFAULT_MODELS_ROOT="${PROJECT_DIR}/utils/models"

mkdir -p "${PROJECT_DIR}/slurm_output"
mkdir -p "${PROJECT_DIR}/inputs" "${PROJECT_DIR}/merged"

cd "${PROJECT_DIR}"

# Load container runtime if your cluster uses modules
module purge || true
module load singularity || true

export MODELS_ROOT="${MODELS_ROOT:-${DEFAULT_MODELS_ROOT}}"
export MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}}"
export LID_MODEL_PATH="${LID_MODEL_PATH:-${MODELS_ROOT}/fasttext/lid.176.bin}"
export NEMO_MODEL_DIR="${NEMO_MODEL_DIR:-${MODELS_ROOT}}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-${MODELS_ROOT}}"
export FSP_INPUTS_DIR="${FSP_INPUTS_DIR:-${PROJECT_DIR}/inputs}"
export FSP_MERGED_DIR="${FSP_MERGED_DIR:-${PROJECT_DIR}/merged}"

echo "==========================================================="
echo "FSP Pipeline"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Project:    ${PROJECT_DIR}"
echo "Input ID:   ${INPUT_ID}"
echo "Language:   ${LANG}"
echo "Models:     ${MODELS_ROOT}"
echo "Inputs out: ${FSP_INPUTS_DIR}"
echo "Merged out: ${FSP_MERGED_DIR}"
echo "GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "none detected"
echo "==========================================================="

srun ./run_singularity.sh --input-id "${INPUT_ID}" --lang "${LANG}"

echo "Job finished."
echo "Outputs are in: ${FSP_MERGED_DIR}"
