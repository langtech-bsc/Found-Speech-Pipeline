#!/usr/bin/env bash
# ===========================================================================
# build_singularity.sh
# ===========================================================================
# Build a Singularity/Apptainer .sif image from the Docker image.
#
# Usage:
#   ./deployment/build/build_docker.sh                           # build Docker image only
#   ./deployment/build/build_singularity.sh                      # uses defaults
#   ./deployment/build/build_singularity.sh --docker-tag my-tag   # custom Docker tag
#   ./deployment/build/build_singularity.sh --sif my-output.sif   # custom .sif name
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_TAG="${DOCKER_TAG:-fsp-pipeline}"
SIF_NAME="${SIF_NAME:-fsp-pipeline.sif}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker-tag) DOCKER_TAG="$2"; shift 2 ;;
        --sif)        SIF_NAME="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--docker-tag TAG] [--sif OUTPUT.sif]"
            echo ""
            echo "Builds a Singularity/Apptainer .sif from the Docker image."
            echo ""
            echo "  --docker-tag TAG    Docker image tag (default: fsp-pipeline)"
            echo "  --sif OUTPUT.sif    Output .sif filename (default: fsp-pipeline.sif)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect Singularity or Apptainer
if command -v apptainer &>/dev/null; then
    BUILDER=apptainer
elif command -v singularity &>/dev/null; then
    BUILDER=singularity
else
    echo "❌ Neither 'singularity' nor 'apptainer' found."
    echo ""
    echo "Install Apptainer (recommended):"
    echo "  sudo apt-get update && sudo apt-get install -y apptainer"
    echo ""
    echo "Or install Singularity:"
    echo "  https://docs.sylabs.io/guides/latest/admin-guide/installation.html"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Building Singularity image"
echo "  Docker tag : ${DOCKER_TAG}"
echo "  Output     : ${SIF_NAME}"
echo "  Builder    : ${BUILDER}"
echo "═══════════════════════════════════════════════════════════"

# Step 1: Ensure the Docker image exists
if ! docker image inspect "${DOCKER_TAG}" &>/dev/null; then
    echo ""
    echo "► Docker image '${DOCKER_TAG}' not found. Building it first..."
    "${REPO_ROOT}/deployment/build/build_docker.sh" --docker-tag "${DOCKER_TAG}"
fi

# Step 2: Convert Docker image → .sif
# Use a temp dir on the same filesystem as the output (avoids /tmp running out of space)
TMPDIR_BUILD="${REPO_ROOT}/.apptainer_tmp"
mkdir -p "${TMPDIR_BUILD}"
export APPTAINER_TMPDIR="${TMPDIR_BUILD}"
export SINGULARITY_TMPDIR="${TMPDIR_BUILD}"

echo ""
echo "► Converting Docker image → ${SIF_NAME} ..."
echo "  (This may take several minutes depending on image size)"
echo "  Temp dir: ${TMPDIR_BUILD}"
echo ""

"${BUILDER}" build "${REPO_ROOT}/${SIF_NAME}" "docker-daemon://${DOCKER_TAG}:latest"

# Clean up temp dir
rm -rf "${TMPDIR_BUILD}"

echo ""
echo "✅ Singularity image created: ${SIF_NAME}"
echo "   Size: $(du -h "${REPO_ROOT}/${SIF_NAME}" | cut -f1)"
echo ""
echo "Run it with:"
echo "  ./deployment/run/run_singularity_image.sh --input-id <ID> --lang es"
