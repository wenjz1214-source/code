#!/bin/bash
# Evaluate a trained model on KITTI-STEP validation set
# Usage: conda activate step_reproduce && bash scripts/eval_model.sh [panoptic|motion]
set -e

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

MODEL_TYPE=${1:-"panoptic"}
KITTI_STEP_ROOT="<KITTI_STEP_ROOT>"

if [ "${MODEL_TYPE}" = "panoptic" ]; then
    CONFIG_FILE="deeplab2/configs/kitti/panoptic_deeplab/resnet50_os32.textproto"
    EXPERIMENT_NAME="panoptic_deeplab_kitti_step"
    export TRAIN_SET="${KITTI_STEP_ROOT}/tfrecords/train@10.tfrecord"
    export VAL_SET="${KITTI_STEP_ROOT}/tfrecords/val@10.tfrecord"
    export INIT_CHECKPOINT=""
elif [ "${MODEL_TYPE}" = "motion" ]; then
    CONFIG_FILE="deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto"
    EXPERIMENT_NAME="motion_deeplab_kitti_step"
    export TRAIN_SET="${KITTI_STEP_ROOT}/tfrecords/two_frames/train@10.tfrecord"
    export VAL_SET="${KITTI_STEP_ROOT}/tfrecords/two_frames/val@10.tfrecord"
    export INIT_CHECKPOINT=""
else
    echo "Usage: $0 [panoptic|motion]"
    exit 1
fi

export EXPERIMENT_NAME
MODEL_DIR="${KITTI_STEP_ROOT}/model_output/${EXPERIMENT_NAME}"

echo "============================================"
echo "Evaluating ${MODEL_TYPE} model on KITTI-STEP"
echo "  Config:    ${CONFIG_FILE}"
echo "  Model dir: ${MODEL_DIR}"
echo "============================================"

python trainer/train.py \
    --config_file="${CONFIG_FILE}" \
    --mode=eval \
    --model_dir="${MODEL_DIR}" \
    --num_gpus=1
