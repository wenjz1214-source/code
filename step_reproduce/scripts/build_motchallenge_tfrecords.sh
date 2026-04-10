#!/bin/bash
# Convert MOTChallenge-STEP data to TFRecords.
# Usage: conda activate step_reproduce && bash scripts/build_motchallenge_tfrecords.sh

set -euo pipefail

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

MOTSTEP_ROOT="<MOTCHALLENGE_STEP_ROOT>"
OUTPUT_DIR="${MOTSTEP_ROOT}/tfrecords"

mkdir -p "${OUTPUT_DIR}"

for split in train test; do
  dir="${MOTSTEP_ROOT}/images/${split}"
  if [ ! -d "${dir}" ]; then
    echo "ERROR: Missing ${dir}"
    echo "Run scripts/download_motchallenge_step.sh first."
    exit 1
  fi
  echo "Found ${split}: $(ls "${dir}" | wc -l) sequences"
done

echo "Building single-frame TFRecords ..."
<STEP_PYTHON> deeplab2/data/build_step_data.py \
  --step_root="${MOTSTEP_ROOT}" \
  --output_dir="${OUTPUT_DIR}"

echo "Building two-frame TFRecords (for Motion-DeepLab) ..."
<STEP_PYTHON> deeplab2/data/build_step_data.py \
  --step_root="${MOTSTEP_ROOT}" \
  --output_dir="${OUTPUT_DIR}/two_frames" \
  --use_two_frames

echo "TFRecords created at ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
