#!/bin/bash
#SBATCH --job-name=fsp-pipeline
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# GitHub-safe Slurm wrapper for the main pipeline.
# Add site-specific directives such as --account/--qos locally if your cluster requires them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_output

# Allow callers to point at a specific interpreter/environment.
if [[ -n "${PYTHON:-}" ]]; then
  export PATH="$(dirname "${PYTHON}"):${PATH}"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

# Model locations can be supplied by the caller or inherited from the environment.
export MODELS_ROOT="${MODELS_ROOT:-}"
export MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}}"
export LID_MODEL_PATH="${LID_MODEL_PATH:-${MODELS_ROOT:+${MODELS_ROOT}/fasttext/lid.176.bin}}"
export NEMO_MODEL_DIR="${NEMO_MODEL_DIR:-${MODELS_ROOT}}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-${MODELS_ROOT}}"

INPUT_ID="${1:-dtTCbYIHLps}"
LANG="${2:-eu}"

echo "════════════════════════════════════════════════════"
echo "  FSP Pipeline — ${INPUT_ID} [${LANG}]"
echo "  Node: $(hostname)"
echo "  Python: ${PYTHON}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo "  Models: ${MODELS_ROOT:-<unset>}"
echo "  LID: ${LID_MODEL_PATH:-<unset>}"
echo "════════════════════════════════════════════════════"

"${PYTHON}" pipeline_service.py --input-id "$INPUT_ID" --lang "$LANG"
