#!/bin/bash
# Evaluate B1 IoU Association on KITTI-STEP using raw panoptic predictions.
# Usage: conda activate step_reproduce && bash scripts/eval_b1_iou_assoc.sh [pred_root] [output_root]

set -euo pipefail

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

KITTI_STEP_ROOT="/share_data/wenjingzhong/kitti_step"
GT_DIR="${KITTI_STEP_ROOT}/panoptic_maps/val"
PRED_ROOT="${1:-${KITTI_STEP_ROOT}/model_output/panoptic_deeplab_kitti_step_a16_safe/vis_ckpt30000/raw_panoptic}"
OUTPUT_ROOT="${2:-${KITTI_STEP_ROOT}/model_output/b1_iou_assoc_kitti_step}"

mkdir -p "${OUTPUT_ROOT}"

echo "============================================"
echo "Evaluating B1 IoU Association"
echo "  GT dir:     ${GT_DIR}"
echo "  Pred dir:   ${PRED_ROOT}"
echo "  Output dir: ${OUTPUT_ROOT}"
echo "============================================"

/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python deeplab2/tracker/iou_tracker.py \
  --gt="${GT_DIR}" \
  --pred="${PRED_ROOT}" \
  --output="${OUTPUT_ROOT}" \
  --dataset="kitti_step" \
  --input_channels=3
