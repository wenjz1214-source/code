#!/bin/bash
# Train Motion-DeepLab (two-frame baseline) on KITTI-STEP
# Usage: conda activate step_reproduce && bash scripts/train_motion_deeplab.sh [NUM_GPUS]
#
# GPU count: pass NUM_GPUS as $1, or omit to use all GPUs reported by nvidia-smi (min 1).
# After a bad H100 run, use a fresh model dir so training does not resume poisoned ckpts:
#   MOTION_DEEPLAB_MODEL_DIR=/path/to/model_output/motion_deeplab_kitti_step_ampere bash ...
set -e

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

KITTI_STEP_ROOT="<KITTI_STEP_ROOT>"

if [ -n "${1:-}" ]; then
  NUM_GPUS="$1"
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  NUM_GPUS=$((NUM_GPUS + 0))
  if [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=1
  fi
fi

# Paths are set in deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto
CONFIG_FILE="deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto"
# Use a fresh default path so we do not accidentally resume a bad H100 checkpoint.
MODEL_DIR="${MOTION_DEEPLAB_MODEL_DIR:-${KITTI_STEP_ROOT}/model_output/motion_deeplab_kitti_step_a16_safe}"

mkdir -p "${MODEL_DIR}"

echo "============================================"
echo "Training Motion-DeepLab on KITTI-STEP"
echo "  Config:     ${CONFIG_FILE}"
echo "  Model dir:  ${MODEL_DIR}"
echo "  GPUs:       ${NUM_GPUS}  (MirroredStrategy: global batch from textproto is split per GPU)"
echo "  NOTE: TF 2.6 + Hopper (H100) can PTX-JIT to NaN (loss may log as 0). Prefer Ampere+."
echo "  If NCCL errors on multi-GPU, try: export NCCL_P2P_DISABLE=1"
echo "============================================"

# 重定向到文件/pipe 时避免 stdout/stderr 全缓冲，否则长时间看不到新日志
export PYTHONUNBUFFERED=1
# Let TF grow allocations gradually; this reduces allocator pressure on 16 GB A16s.
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
python -u deeplab2/trainer/train.py \
    --config_file="${CONFIG_FILE}" \
    --mode=train \
    --model_dir="${MODEL_DIR}" \
    --num_gpus="${NUM_GPUS}"
