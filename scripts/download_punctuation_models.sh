#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODELS_ROOT="${REPO_ROOT}/utils/models"
MODELS_ROOT="${1:-${MODELS_ROOT:-$DEFAULT_MODELS_ROOT}}"

REPOS=(
  "UMUTeam/catalan_capitalization_punctuation_restoration:catalan_capitalization_punctuation_restoration"
  "UMUTeam/spanish_capitalization_punctuation_restoration:spanish_capitalization_punctuation_restoration"
  "UMUTeam/galician_capitalization_punctuation_restoration:galician_capitalization_punctuation_restoration"
  "HiTZ/cap-punct-eu:cap-punct-eu"
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

is_lfs_pointer() {
  local file="$1"
  [[ -f "$file" ]] && head -n 1 "$file" | grep -q "https://git-lfs.github.com/spec/v1"
}

validate_model_dir() {
  local dir="$1"
  local weights_file=""

  if [[ ! -f "$dir/config.json" ]]; then
    echo "Missing config.json in $dir" >&2
    return 1
  fi

  for candidate in \
    "$dir/model.safetensors" \
    "$dir/pytorch_model.bin" \
    "$dir/tf_model.h5" \
    "$dir/flax_model.msgpack"; do
    if [[ -f "$candidate" ]]; then
      weights_file="$candidate"
      break
    fi
  done

  if [[ -z "$weights_file" ]]; then
    echo "No supported weights file found in $dir" >&2
    return 1
  fi

  if is_lfs_pointer "$weights_file"; then
    echo "Weights file is still a Git LFS pointer in $dir" >&2
    return 1
  fi
}

clone_or_update_repo() {
  local repo_id="$1"
  local target_dir="$2"
  local repo_url="https://huggingface.co/${repo_id}"
  local parent_dir

  parent_dir="$(dirname "$target_dir")"
  mkdir -p "$parent_dir"

  if [[ -d "$target_dir/.git" ]]; then
    echo "Updating ${repo_id} -> ${target_dir}"
    git -C "$target_dir" fetch --all --tags
    git -C "$target_dir" pull --ff-only
  else
    echo "Cloning ${repo_id} -> ${target_dir}"
    rm -rf "$target_dir"
    git clone "$repo_url" "$target_dir"
  fi

  git -C "$target_dir" lfs install --local
  git -C "$target_dir" lfs pull
  validate_model_dir "$target_dir"
}

main() {
  require_cmd git
  require_cmd head
  require_cmd grep

  if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required but not available. Install it first." >&2
    exit 1
  fi

  mkdir -p "$MODELS_ROOT"
  echo "Downloading punctuation models into: $MODELS_ROOT"

  local entry repo_id dir_name target_dir
  for entry in "${REPOS[@]}"; do
    repo_id="${entry%%:*}"
    dir_name="${entry##*:}"
    target_dir="${MODELS_ROOT}/${dir_name}"
    clone_or_update_repo "$repo_id" "$target_dir"
  done

  echo
  echo "Downloaded punctuation models:"
  for entry in "${REPOS[@]}"; do
    dir_name="${entry##*:}"
    echo "  - ${MODELS_ROOT}/${dir_name}"
  done
}

main "$@"
