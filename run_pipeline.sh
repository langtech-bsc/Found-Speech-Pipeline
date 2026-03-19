#!/bin/bash
#SBATCH --job-name=fsp-pipeline
#SBATCH --account=bsc88
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Main pipeline env: keep the NeMo-compatible environment here.
export PYTHON=/home/bsc/bsc700374/.conda/envs/fsp-bsc/bin/python
export PATH="/home/bsc/bsc700374/.conda/envs/fsp-bsc/bin:$PATH"

INPUT_ID="${1:-dtTCbYIHLps}"
LANG="${2:-eu}"

cd /gpfs/home/bsc/bsc700374/Found-Speech-Pipeline
mkdir -p slurm_output

echo "════════════════════════════════════════════════════"
echo "  FSP Pipeline — ${INPUT_ID} [${LANG}]"
echo "  Node: $(hostname)"
echo "  Python: $(which python)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo "════════════════════════════════════════════════════"

python pipeline_service.py --input-id "$INPUT_ID" --lang "$LANG"
