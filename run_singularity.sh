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
#   ingestion/      → /app/ingestion      (input WAV+TSV)
#   inputs/         → /app/inputs         (intermediate files)
#   merged/         → /app/merged         (final output)
#   utils/models/   → /app/utils/models   (pre-downloaded models)
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
    echo "❌ Neither 'singularity' nor 'apptainer' found."
    exit 1
fi

# Ensure the .sif exists
if [[ ! -f "${SIF}" ]]; then
    echo "❌ Singularity image not found: ${SIF}"
    echo "   Run ./build_singularity.sh first."
    exit 1
fi

# Create output directories if they don't exist
mkdir -p "${SCRIPT_DIR}/inputs" "${SCRIPT_DIR}/merged"

# Pre-flight: fix root-owned files from previous Docker runs
# Singularity runs as the current user, so root-owned files will cause PermissionErrors
for dir in inputs merged utils/models; do
    target="${SCRIPT_DIR}/${dir}"
    if [[ -d "${target}" ]] && find "${target}" -maxdepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
        echo "⚠  Fixing permissions on ${dir}/ (root-owned files from Docker)..."
        sudo chown -R "$(whoami)" "${target}"
    fi
done

# Run the pipeline
echo "═══════════════════════════════════════════════════════════"
echo "  Running FSP pipeline via ${RUNNER}"
echo "  Image: ${SIF}"
echo "  Args:  ${*:-<batch mode>}"
echo "═══════════════════════════════════════════════════════════"

${RUNNER} exec \
    --bind "${SCRIPT_DIR}/ingestion:/app/ingestion" \
    --bind "${SCRIPT_DIR}/inputs:/app/inputs" \
    --bind "${SCRIPT_DIR}/merged:/app/merged" \
    --bind "${SCRIPT_DIR}/utils/models:/app/utils/models" \
    "${SIF}" \
    python /app/pipeline_service.py "$@"
