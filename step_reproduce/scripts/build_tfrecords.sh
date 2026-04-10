#!/bin/bash
# Convert KITTI-STEP data to TFRecords
# Prerequisites: KITTI images must be downloaded and organized first
# Usage: conda activate step_reproduce && source scripts/setup_env.sh && bash scripts/build_tfrecords.sh

set -e

source "$(dirname "$0")/setup_env.sh"

cd "${PROJECT_ROOT}"

KITTI_STEP_ROOT="<KITTI_STEP_ROOT>"
OUTPUT_DIR="${KITTI_STEP_ROOT}/tfrecords"

mkdir -p "${OUTPUT_DIR}"

# Verify data structure
for split in train val test; do
    dir="${KITTI_STEP_ROOT}/images/${split}"
    if [ ! -d "${dir}" ]; then
        echo "ERROR: Missing ${dir}"
        echo "Run scripts/download_kitti_images.sh first."
        exit 1
    fi
    echo "Found ${split}: $(ls ${dir} | wc -l) sequences"
done

echo "Building single-frame TFRecords..."
python deeplab2/data/build_step_data.py \
    --step_root="${KITTI_STEP_ROOT}" \
    --output_dir="${OUTPUT_DIR}"

echo "Building two-frame TFRecords (for Motion-DeepLab)..."
python deeplab2/data/build_step_data.py \
    --step_root="${KITTI_STEP_ROOT}" \
    --output_dir="${OUTPUT_DIR}/two_frames" \
    --use_two_frames

echo "TFRecords created at ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
