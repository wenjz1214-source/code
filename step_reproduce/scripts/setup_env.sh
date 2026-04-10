#!/bin/bash
# Setup environment variables for DeepLab2 STEP reproduction
# Usage: source scripts/setup_env.sh

PROJECT_ROOT="<STEP_ROOT>"
CONDA_ENV_LIB="<STEP_CONDA_LIB>"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${PROJECT_ROOT}/cocoapi/PythonAPI"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH}"

export KITTI_STEP_ROOT="<KITTI_STEP_ROOT>"
export TFRECORD_DIR="${KITTI_STEP_ROOT}/tfrecords"
export CHECKPOINT_DIR="${KITTI_STEP_ROOT}/checkpoints"
export MODEL_DIR="${KITTI_STEP_ROOT}/model_output"

echo "Environment configured:"
echo "  PYTHONPATH includes: deeplab2, orbit, cocoapi"
echo "  KITTI_STEP_ROOT: ${KITTI_STEP_ROOT}"
echo "  TFRECORD_DIR:    ${TFRECORD_DIR}"
echo "  CHECKPOINT_DIR:  ${CHECKPOINT_DIR}"
echo "  MODEL_DIR:       ${MODEL_DIR}"
