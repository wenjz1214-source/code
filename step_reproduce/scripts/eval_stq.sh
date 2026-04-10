#!/bin/bash
# Evaluate STQ (Segmentation and Tracking Quality) metric
# This uses the numpy implementation from DeepLab2
# Usage: conda activate step_reproduce && bash scripts/eval_stq.sh [predictions_dir]
set -e

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

PREDICTIONS_DIR=${1}
KITTI_STEP_ROOT="/share_data/wenjingzhong/kitti_step"
GT_DIR="${KITTI_STEP_ROOT}/panoptic_maps/val"

if [ -z "${PREDICTIONS_DIR}" ]; then
    echo "Usage: $0 <predictions_dir>"
    echo "  predictions_dir: directory containing predicted panoptic maps"
    exit 1
fi

echo "============================================"
echo "Evaluating STQ metric"
echo "  GT dir:          ${GT_DIR}"
echo "  Predictions dir: ${PREDICTIONS_DIR}"
echo "============================================"

python -c "
from deeplab2.evaluation.numpy import segmentation_and_tracking_quality as stq_lib
import numpy as np
import os
from PIL import Image
import glob

gt_dir = '${GT_DIR}'
pred_dir = '${PREDICTIONS_DIR}'

n_classes = 19
thing_list = [11, 13]
ignore_label = 255
max_instances_per_category = 1000
offset = 256 * 256 * 256

stq_obj = stq_lib.STQuality(n_classes, thing_list, ignore_label, max_instances_per_category, offset)

sequences = sorted(os.listdir(gt_dir))
for seq in sequences:
    gt_seq_dir = os.path.join(gt_dir, seq)
    pred_seq_dir = os.path.join(pred_dir, seq)
    if not os.path.isdir(pred_seq_dir):
        print(f'Warning: missing predictions for sequence {seq}')
        continue

    frames = sorted(glob.glob(os.path.join(gt_seq_dir, '*.png')))
    for gt_path in frames:
        fname = os.path.basename(gt_path)
        pred_path = os.path.join(pred_seq_dir, fname)
        if not os.path.exists(pred_path):
            continue

        gt_img = np.array(Image.open(gt_path))
        pred_img = np.array(Image.open(pred_path))

        gt_panoptic = gt_img[:, :, 0].astype(np.int32) * 256 * 256 + \
                      gt_img[:, :, 1].astype(np.int32) * 256 + \
                      gt_img[:, :, 2].astype(np.int32)
        pred_panoptic = pred_img[:, :, 0].astype(np.int32) * 256 * 256 + \
                        pred_img[:, :, 1].astype(np.int32) * 256 + \
                        pred_img[:, :, 2].astype(np.int32)

        stq_obj.update_state(gt_panoptic.flatten(), pred_panoptic.flatten(), seq)

result = stq_obj.result()
print(f\"STQ:  {result['STQ'] * 100:.2f}%\")
print(f\"AQ:   {result['AQ'] * 100:.2f}%\")
print(f\"IoU:  {result['IoU'] * 100:.2f}%\")
print()
print('Paper reference (Motion-DeepLab): STQ=57.7%, with PQ=42.08, mIoU=63.15')
"
