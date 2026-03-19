#!/bin/bash
#SBATCH --job-name=fsp-gl-phi-seg
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

INPUT_ID="${1:-JynlbvTgWzM}"
REPO_DIR="/gpfs/home/bsc/bsc700374/Found-Speech-Pipeline"
MODELS_ROOT_DEFAULT="/gpfs/projects/bsc88/speech/ASR/models"
IMAGE_PATH_DEFAULT="${REPO_DIR}/gl-extra-asr.sif"

cd "${REPO_DIR}"
mkdir -p slurm_output

module load singularity

export MODELS_ROOT="${MODELS_ROOT:-$MODELS_ROOT_DEFAULT}"
export IMAGE_PATH="${IMAGE_PATH:-$IMAGE_PATH_DEFAULT}"

JSON_PATH="${REPO_DIR}/inputs/output_segment/final_output_${INPUT_ID}.json"

echo "════════════════════════════════════════════════════"
echo "  GL Phi Segment Test"
echo "  Input ID: ${INPUT_ID}"
echo "  JSON:     ${JSON_PATH}"
echo "  Node:     $(hostname)"
echo "  Models:   ${MODELS_ROOT}"
echo "  Image:    ${IMAGE_PATH}"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo "════════════════════════════════════════════════════"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Missing image: ${IMAGE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${JSON_PATH}" ]]; then
    echo "Missing JSON: ${JSON_PATH}" >&2
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
import json
import tempfile
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

input_id = "JynlbvTgWzM"
json_path = Path(f"/gpfs/home/bsc/bsc700374/Found-Speech-Pipeline/inputs/output_segment/final_output_{input_id}.json")
model_path = Path("/gpfs/projects/bsc88/speech/ASR/models/phi-4-multimodal-instruct-gl-v1.0")

data = json.loads(json_path.read_text(encoding="utf-8"))
segment_path = None
for block in data.values():
    for seg in block.get("results", []):
        if seg.get("language") == "gl":
            segment_path = Path(seg["segment_path"])
            break
    if segment_path:
        break

if segment_path is None:
    raise SystemExit("No Galician segment found in JSON")

safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_path.name).strip("_")
alias_root = Path(tempfile.gettempdir()) / "fsp_hf_model_aliases"
alias_root.mkdir(parents=True, exist_ok=True)
alias_path = alias_root / safe_name
if not alias_path.exists():
    alias_path.symlink_to(model_path.resolve(), target_is_directory=True)

print(f"segment_path: {segment_path}", flush=True)

processor = AutoProcessor.from_pretrained(
    alias_path,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    alias_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True,
)
model = model.to("cuda")
model.eval()

audio, sample_rate = librosa.load(str(segment_path), sr=16000, mono=True)
user_msg = {
    "role": "user",
    "content": "<|audio_1|>\nTranscribe the audio clip into Galician text.",
}
prompt = processor.tokenizer.apply_chat_template(
    [user_msg],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(
    text=prompt,
    audios=[(audio, sample_rate)],
    return_tensors="pt",
)
inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_logits_to_keep=1,
    )

new_tokens = output_ids[:, prompt_len:] if prompt_len else output_ids
text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
print("phi_output:", text, flush=True)
PY
