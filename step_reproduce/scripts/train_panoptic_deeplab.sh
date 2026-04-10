#!/bin/bash
# Train Panoptic-DeepLab (single-frame baseline) on KITTI-STEP.
# Usage: conda activate step_reproduce && bash scripts/train_panoptic_deeplab.sh [NUM_GPUS]
set -e

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

NUM_GPUS=${1:-1}
KITTI_STEP_ROOT="/share_data/wenjingzhong/kitti_step"

export EXPERIMENT_NAME="panoptic_deeplab_kitti_step"
export INIT_CHECKPOINT="${KITTI_STEP_ROOT}/checkpoints/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine/ckpt-60000"
export TRAIN_SET="${KITTI_STEP_ROOT}/tfrecords/train@10.tfrecord"
export VAL_SET="${KITTI_STEP_ROOT}/tfrecords/val@10.tfrecord"

CONFIG_FILE="deeplab2/configs/kitti/panoptic_deeplab/resnet50_os32.textproto"
MODEL_DIR="${PANOPTIC_DEEPLAB_MODEL_DIR:-${KITTI_STEP_ROOT}/model_output/panoptic_deeplab_kitti_step_a16_safe}"

mkdir -p "${MODEL_DIR}"

echo "============================================"
echo "Training Panoptic-DeepLab on KITTI-STEP"
echo "  Config:     ${CONFIG_FILE}"
echo "  Checkpoint: ${INIT_CHECKPOINT}"
echo "  Model dir:  ${MODEL_DIR}"
echo "  GPUs:       ${NUM_GPUS}"
echo "============================================"

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
PYTHON_BIN="${STEP_PYTHON_BIN:-/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python}"
"${PYTHON_BIN}" -u deeplab2/trainer/train.py \
    --config_file="${CONFIG_FILE}" \
    --mode=train \
    --model_dir="${MODEL_DIR}" \
    --num_gpus="${NUM_GPUS}"
