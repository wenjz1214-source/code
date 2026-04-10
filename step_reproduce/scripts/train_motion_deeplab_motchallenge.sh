#!/bin/bash
# Train Motion-DeepLab (B4) on MOTChallenge-STEP.
# Usage: conda activate step_reproduce && bash scripts/train_motion_deeplab_motchallenge.sh [NUM_GPUS]

set -euo pipefail

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

NUM_GPUS=${1:-1}
MOTSTEP_ROOT="/share_data/wenjingzhong/motchallenge_step"

CONFIG_FILE="deeplab2/configs/motchallenge/motion_deeplab/resnet50_os32.textproto"
MODEL_DIR="${MOT_MOTION_DEEPLAB_MODEL_DIR:-${MOTSTEP_ROOT}/model_output/motion_deeplab_motchallenge_step_a16_fixinit}"

mkdir -p "${MODEL_DIR}"

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python - <<'PY'
from pathlib import Path
from google.protobuf import text_format
from deeplab2 import config_pb2

config_path = Path('deeplab2/configs/motchallenge/motion_deeplab/resnet50_os32.textproto')
text = config_path.read_text()
text = text.replace('${EXPERIMENT_NAME}', 'motion_deeplab_motchallenge_step')
text = text.replace('${INIT_CHECKPOINT}', '/share_data/wenjingzhong/motchallenge_step/checkpoints/motion_deeplab_motchallenge_first_and_last/pretrained-1')
text = text.replace('${TRAIN_SET}', '/share_data/wenjingzhong/motchallenge_step/tfrecords/two_frames/train*.tfrecord')
text = text.replace('${VAL_SET}', '/share_data/wenjingzhong/motchallenge_step/tfrecords/two_frames/val*.tfrecord')
tmp = Path('/tmp/motchallenge_motion_resnet50_os32_local.textproto')
cfg = config_pb2.ExperimentOptions()
text_format.Parse(text, cfg)
cfg.trainer_options.solver_options.use_sync_batchnorm = False
cfg.trainer_options.solver_options.batchnorm_epsilon = 0.001
cfg.trainer_options.solver_options.use_gradient_clipping = True
cfg.trainer_options.solver_options.clip_gradient_norm = 1.0
cfg.train_dataset_options.batch_size = 1
cfg.train_dataset_options.crop_size[:] = [545, 961]
cfg.eval_dataset_options.crop_size[:] = [545, 961]
cfg.trainer_options.steps_per_loop = 20
cfg.trainer_options.save_summaries_steps = 20
cfg.trainer_options.save_checkpoints_steps = 200
cfg.trainer_options.solver_options.training_number_of_steps = 2000
tmp.write_text(text_format.MessageToString(cfg))
print(tmp)
PY

TMP_CONFIG=/tmp/motchallenge_motion_resnet50_os32_local.textproto

echo "============================================"
echo "Training Motion-DeepLab on MOTChallenge-STEP"
echo "  Config:     ${TMP_CONFIG}"
echo "  Model dir:  ${MODEL_DIR}"
echo "  GPUs:       ${NUM_GPUS}"
echo "============================================"

/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python -u deeplab2/trainer/train.py \
  --config_file="${TMP_CONFIG}" \
  --mode=train \
  --model_dir="${MODEL_DIR}" \
  --num_gpus="${NUM_GPUS}"
