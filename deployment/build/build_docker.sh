#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# build_docker.sh
# ===========================================================================
# Build the Docker image using a staged context colocated with the Dockerfile.
#
# Usage:
#   ./deployment/build/build_docker.sh
#   ./deployment/build/build_docker.sh --docker-tag my-tag
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/deployment/containers/docker"
DOCKERFILE_PATH="${DOCKER_DIR}/Dockerfile"
IGNOREFILE_PATH="${DOCKER_DIR}/.dockerignore"
CONTEXT_DIR="${DOCKER_DIR}/.context"
DOCKER_TAG="${DOCKER_TAG:-fsp-pipeline}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker-tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--docker-tag TAG]"
            echo ""
            echo "Stages a minimal Docker build context under deployment/containers/docker/"
            echo "and builds the image from the colocated Dockerfile and .dockerignore."
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Dockerfile not found: ${DOCKERFILE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${IGNOREFILE_PATH}" ]]; then
    echo ".dockerignore not found: ${IGNOREFILE_PATH}" >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Building Docker image"
echo "  Docker tag : ${DOCKER_TAG}"
echo "  Dockerfile : ${DOCKERFILE_PATH}"
echo "  Ignorefile : ${IGNOREFILE_PATH}"
echo "  Context    : ${CONTEXT_DIR}"
echo "═══════════════════════════════════════════════════════════"

rm -rf "${CONTEXT_DIR}"
mkdir -p "${CONTEXT_DIR}"

# Stage only the files needed by the image so Dockerfile and .dockerignore can
# live together without depending on the repository root as build context.
cp "${DOCKERFILE_PATH}" "${CONTEXT_DIR}/Dockerfile"
cp "${IGNOREFILE_PATH}" "${CONTEXT_DIR}/.dockerignore"
cp "${REPO_ROOT}/requirements.txt" "${CONTEXT_DIR}/requirements.txt"
cp "${REPO_ROOT}/pipeline_service.py" "${CONTEXT_DIR}/pipeline_service.py"
cp -a "${REPO_ROOT}/fsp" "${CONTEXT_DIR}/fsp"
cp -a "${REPO_ROOT}/scripts" "${CONTEXT_DIR}/scripts"
cp -a "${REPO_ROOT}/steps" "${CONTEXT_DIR}/steps"
cp -a "${REPO_ROOT}/NeMo" "${CONTEXT_DIR}/NeMo"

docker build -t "${DOCKER_TAG}" -f "${CONTEXT_DIR}/Dockerfile" "${CONTEXT_DIR}"

echo
echo "Built Docker image: ${DOCKER_TAG}"
