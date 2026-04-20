#!/bin/bash

# Split Ingestion Files into Buckets and Run Found Speech Pipeline
# ----------------------------------------------------------------
# This script splits ingestion audio files into N buckets and then runs the Found Speech Pipeline.
#
# USAGE:
#   ./create_splits_run_pipeline.sh [-n NUM_BUCKETS] [-o OUTPUT_DIR] [-l LANG] [-i INPUT_DIR]
#
# REQUIRED ARGUMENTS:
#   -n <num_buckets>   Number of buckets to split files into
#   -o <output_dir>    Directory where split files will be written
#   -l <lang>          Language code for downstream processing
#
# OPTIONAL ARGUMENTS (with defaults):
#   -i <input_dir>     Input directory containing .wav audio files to process (default: ingestion/)
#
# EXAMPLE:
#   ./create_splits_run_pipeline.sh -n 5 -o input_ids -l es
#   ./create_splits_run_pipeline.sh -n 10 -o input_ids -l ca -i custom/path
#
# DEPENDENCIES:
#   - Singularity image
#   - Python script: scripts/split.py
#
# NOTES:
#   - Output directory will be created if it doesn't exist
#   - Each generated file is printed to stdout for verification

set -euo pipefail

num_buckets=""
output_dir=""
lang=""
input_dir="ingestion/"

while getopts "n:o:l:i:" opt; do
    case $opt in
        n) num_buckets="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        l) lang="$OPTARG" ;;
        i) input_dir="$OPTARG" ;;
        \?) echo "Invalid option -$OPTARG"; exit 1 ;;
    esac
done

echo "num_buckets: $num_buckets"
echo "output_dir: $output_dir"
echo "lang: $lang"
echo "input_dir: $input_dir"

source ./.env

module load singularity

if [ ! -d "${input_dir}" ]; then
    echo "Cannot find ingest directory '$input_dir'"
    exit 1
fi

if [ ! -d "${output_dir}" ]; then
    mkdir -p "${output_dir}"
    echo "Created output directory '${output_dir}'"
else
    echo "Using existing output directory '${output_dir}'"
fi

singularity exec --no-home "${SIF}" python scripts/split.py "${input_dir}" -n "${num_buckets}" -o "${output_dir}"

echo ""

for file in "${output_dir}"/*; do
    if [ -f "$file" ]; then
        echo "Running pipeline with input id file: $file";
        sbatch ./run_singularity.sh --input-id-file "${file}" --lang "${lang}"
    fi
done
