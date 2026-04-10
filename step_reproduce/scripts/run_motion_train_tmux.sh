#!/bin/bash
# Start Motion-DeepLab training inside tmux with timestamped logs (tee).
#
# Usage:
#   bash scripts/run_motion_train_tmux.sh
#
# Optional env:
#   MOTION_DEEPLAB_MODEL_DIR — checkpoint root (see train_motion_deeplab.sh)
#   MOTION_TRAIN_NUM_GPUS    — e.g. 4; unset = use all visible GPUs
#   MOTION_TMUX_SESSION      — tmux session name (default: step_motion_ampere)
#   MOTION_NCCL_P2P_DISABLE=1 — bake NCCL_P2P_DISABLE=1 into runner (多卡 PCIe 挂死时可试)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KITTI_STEP_ROOT="<KITTI_STEP_ROOT>"
LOG_ROOT="${KITTI_STEP_ROOT}/model_output/train_logs"
SESSION="${MOTION_TMUX_SESSION:-step_motion_ampere}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/motion_deeplab_ampere_${TS}.log"
LATEST_LINK="${LOG_ROOT}/motion_deeplab_ampere_latest.log"
CONDA_SH="<MINICONDA_ROOT>/etc/profile.d/conda.sh"
DEFAULT_MODEL_DIR="${KITTI_STEP_ROOT}/model_output/motion_deeplab_kitti_step_a16_safe"
MODEL_DIR="${MOTION_DEEPLAB_MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
NUM_GPUS="${MOTION_TRAIN_NUM_GPUS:-}"

mkdir -p "${LOG_ROOT}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LINK}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux 会话已存在: ${SESSION}"
  echo "  接入: tmux attach -t ${SESSION}"
  echo "  或结束: tmux kill-session -t ${SESSION}"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "未找到 tmux，请先安装: conda install -n step_reproduce -c conda-forge tmux"
  exit 1
fi

{
  echo "======== motion_deeplab train log ========"
  echo "started_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host: $(hostname)"
  echo "session: ${SESSION}"
  echo "log_file: ${LOG_FILE}"
  echo "project: ${PROJECT_ROOT}"
  echo "MOTION_DEEPLAB_MODEL_DIR: ${MODEL_DIR}"
  echo "MOTION_TRAIN_NUM_GPUS: ${NUM_GPUS:-<all visible>}"
  echo "MOTION_NCCL_P2P_DISABLE: ${MOTION_NCCL_P2P_DISABLE:-0}"
  nvidia-smi -L 2>/dev/null || true
  echo "=========================================="
} | tee "${LOG_FILE}"

RUNNER="${LOG_ROOT}/_runner_motion_${TS}.sh"
if [ -n "${NUM_GPUS}" ]; then
  TRAIN_CMD="bash scripts/train_motion_deeplab.sh ${NUM_GPUS}"
else
  TRAIN_CMD="bash scripts/train_motion_deeplab.sh"
fi

NCCL_SNIPPET="# NCCL: default (P2P on)"
if [ "${MOTION_NCCL_P2P_DISABLE:-0}" = 1 ]; then
  NCCL_SNIPPET='export NCCL_P2P_DISABLE=1
echo "NCCL_P2P_DISABLE=1 (launch flag MOTION_NCCL_P2P_DISABLE=1)"'
fi

cat > "${RUNNER}" <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee -a "${LOG_FILE}") 2>&1
export PYTHONUNBUFFERED=1
${NCCL_SNIPPET}
source "${CONDA_SH}"
conda activate step_reproduce
cd "${PROJECT_ROOT}"
source scripts/setup_env.sh
export MPLCONFIGDIR="<MPLCONFIGDIR>"
mkdir -p "\${MPLCONFIGDIR}"
export MOTION_DEEPLAB_MODEL_DIR="${MODEL_DIR}"
echo "train_cmd: ${TRAIN_CMD}"
${TRAIN_CMD}
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION}" "${RUNNER}"

echo "已在 tmux 中启动 Motion-DeepLab 训练。"
echo "  会话: ${SESSION}"
echo "  日志: ${LOG_FILE}"
echo "  最新: ${LATEST_LINK}"
echo "  runner: ${RUNNER}"
echo "  接入: tmux attach -t ${SESSION}"
echo "  跟踪: tail -f ${LATEST_LINK}"
