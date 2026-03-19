#!/bin/bash
#SBATCH --job-name=fsp-gl-phi-load
#SBATCH --account=bsc88
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="/gpfs/home/bsc/bsc700374/Found-Speech-Pipeline"
MODELS_ROOT_DEFAULT="/gpfs/projects/bsc88/speech/ASR/models"
IMAGE_PATH_DEFAULT="${REPO_DIR}/gl-extra-asr.sif"

cd "${REPO_DIR}"
mkdir -p slurm_output

module load singularity

export MODELS_ROOT="${MODELS_ROOT:-$MODELS_ROOT_DEFAULT}"
export IMAGE_PATH="${IMAGE_PATH:-$IMAGE_PATH_DEFAULT}"

echo "════════════════════════════════════════════════════"
echo "  GL Phi Load Test"
echo "  Node:    $(hostname)"
echo "  Models:  ${MODELS_ROOT}"
echo "  Image:   ${IMAGE_PATH}"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo "════════════════════════════════════════════════════"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Missing image: ${IMAGE_PATH}" >&2
    exit 1
fi

if [[ ! -d "${MODELS_ROOT}/phi-4-multimodal-instruct-gl-v1.0" ]]; then
    echo "Missing model directory: ${MODELS_ROOT}/phi-4-multimodal-instruct-gl-v1.0" >&2
    exit 1
fi

if command -v apptainer &>/dev/null; then
    RUNNER=apptainer
elif command -v singularity &>/dev/null; then
    RUNNER=singularity
else
    echo "Neither 'apptainer' nor 'singularity' is available." >&2
    exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"${RUNNER}" exec --nv \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind "${MODELS_ROOT}:${MODELS_ROOT}" \
    "${IMAGE_PATH}" \
    python - <<'PY'
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
import tempfile

model_path = "/gpfs/projects/bsc88/speech/ASR/models/phi-4-multimodal-instruct-gl-v1.0"
repo_path = Path(model_path)
safe_name = "".join(ch if ch.isalnum() else "_" for ch in repo_path.name).strip("_")
alias_root = Path(tempfile.gettempdir()) / "fsp_hf_model_aliases"
alias_root.mkdir(parents=True, exist_ok=True)
alias_path = alias_root / safe_name
if not alias_path.exists():
    alias_path.symlink_to(repo_path.resolve(), target_is_directory=True)

print("loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(
    alias_path,
    trust_remote_code=True,
    local_files_only=True,
)
print("processor: OK", flush=True)

print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    alias_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True,
)
model = model.to("cuda")
model.eval()
print("model: OK", flush=True)
print("model_class:", type(model).__name__, flush=True)
print("cuda_device:", torch.cuda.get_device_name(0), flush=True)
PY
