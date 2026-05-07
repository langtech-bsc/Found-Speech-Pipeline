#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEF_FILE="${DEF_FILE:-${REPO_ROOT}/containers/gl-extra-asr.def}"
SIF_NAME="${SIF_NAME:-${REPO_ROOT}/fsp-gl-extra-asr.sif}"
TMPDIR_BUILD="${TMPDIR_BUILD:-/tmp/apptainer_tmp_gl_extra_asr}"
CACHE_DIR_BUILD="${CACHE_DIR_BUILD:-/tmp/apptainer_cache_gl_extra_asr}"

if command -v apptainer &>/dev/null; then
    BUILDER=apptainer
elif command -v singularity &>/dev/null; then
    BUILDER=singularity
else
    echo "Neither 'apptainer' nor 'singularity' is available." >&2
    exit 1
fi

if [[ ! -f "${DEF_FILE}" ]]; then
    echo "Definition file not found: ${DEF_FILE}" >&2
    exit 1
fi

mkdir -p "${TMPDIR_BUILD}"
mkdir -p "${CACHE_DIR_BUILD}"
export APPTAINER_TMPDIR="${TMPDIR_BUILD}"
export SINGULARITY_TMPDIR="${TMPDIR_BUILD}"
export APPTAINER_CACHEDIR="${CACHE_DIR_BUILD}"
export SINGULARITY_CACHEDIR="${CACHE_DIR_BUILD}"

echo "═══════════════════════════════════════════════════════════"
echo "  Building GL extra ASR image"
echo "  Definition : ${DEF_FILE}"
echo "  Output     : ${SIF_NAME}"
echo "  Builder    : ${BUILDER}"
echo "  Temp dir   : ${TMPDIR_BUILD}"
echo "  Cache dir  : ${CACHE_DIR_BUILD}"
echo "═══════════════════════════════════════════════════════════"

"${BUILDER}" build "${SIF_NAME}" "${DEF_FILE}"

echo
echo "Created: ${SIF_NAME}"
du -h "${SIF_NAME}"
