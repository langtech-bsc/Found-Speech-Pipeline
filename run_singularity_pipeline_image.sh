#!/bin/bash
#SBATCH --job-name=fsp-singularity-image
#SBATCH --account=bsc88
#SBATCH --qos=acc_debug
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

PROJECT_DIR="/gpfs/home/bsc/bsc700374/Found-Speech-Pipeline"
DEFAULT_MODELS_ROOT="/gpfs/projects/bsc88/speech/ASR/models"
JOB_ROOT="${SCRATCH:-/tmp}/${USER}/fsp/${SLURM_JOB_ID}"
LOCAL_INPUTS_DIR="${PROJECT_DIR}/inputs"
LOCAL_MERGED_DIR="${PROJECT_DIR}/merged"
LOCAL_CACHE_DIR="${SCRATCH:-/tmp}/${USER}/fsp_cache"
LOCAL_TMPDIR="${SCRATCH:-/tmp}/${USER}/fsp_tmp"

mkdir -p "${PROJECT_DIR}/slurm_output"
mkdir -p "${JOB_ROOT}"/{inputs,merged,cache,tmp}
mkdir -p "${LOCAL_INPUTS_DIR}" "${LOCAL_MERGED_DIR}" "${LOCAL_CACHE_DIR}" "${LOCAL_TMPDIR}"

cd "${PROJECT_DIR}"

# Load container runtime if your cluster uses modules
module purge || true
module load apptainer || module load singularity || true

# Single shared model root on BSC, while still allowing overrides.
export MODELS_ROOT="${MODELS_ROOT:-${DEFAULT_MODELS_ROOT}}"
export MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}}"
export LID_MODEL_PATH="${LID_MODEL_PATH:-${MODELS_ROOT}/fasttext/lid.176.bin}"
export NEMO_MODEL_DIR="${NEMO_MODEL_DIR:-${MODELS_ROOT}}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-${MODELS_ROOT}}"

# Default to repo-local outputs so results persist after the Slurm job exits.
export FSP_INPUTS_DIR="${FSP_INPUTS_DIR:-${LOCAL_INPUTS_DIR}}"
export FSP_MERGED_DIR="${FSP_MERGED_DIR:-${LOCAL_MERGED_DIR}}"
export FSP_CACHE_DIR="${FSP_CACHE_DIR:-${LOCAL_CACHE_DIR}}"
export TMPDIR="${TMPDIR:-${LOCAL_TMPDIR}}"

echo "==========================================================="
echo "FSP Pipeline (image-only)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Project:    ${PROJECT_DIR}"
echo "Input ID:   ${INPUT_ID}"
echo "Language:   ${LANG}"
echo "Scratch:    ${JOB_ROOT}"
echo "Models:     ${MODELS_ROOT}"
echo "Inputs out: ${FSP_INPUTS_DIR}"
echo "Merged out: ${FSP_MERGED_DIR}"
echo "GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "none detected"
echo "==========================================================="

srun ./run_singularity_image.sh --input-id "${INPUT_ID}" --lang "${LANG}"

echo "Job finished."
echo "Outputs are in: ${FSP_MERGED_DIR}"
