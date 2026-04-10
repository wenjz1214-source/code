#!/bin/bash
# 一键检查毕设环境：Python、CUDA、MSDA 扩展、数据、权重是否就绪

set -euo pipefail
P="${GRAD_PROJECT_ROOT:-<PROJECT_ROOT>}"
V="${P}/venv_tf"
export PATH="${V}/bin:${PATH}"
export MPLCONFIGDIR="${P}/logs/.matplotlib"

echo "========== 毕设环境检查 =========="
echo "项目: ${P}"
echo ""

echo "[1] Python"
"${V}/bin/python" -V || { echo "FAIL: venv 不存在"; exit 1; }

echo ""
echo "[2] PyTorch / CUDA"
"${V}/bin/python" - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "[3] MultiScaleDeformableAttention（必须先 import torch）"
if "${V}/bin/python" -c "import torch; import MultiScaleDeformableAttention; print('  OK')"; then
  :
else
  echo "  FAIL — 请在 code/trackformer/src/trackformer/models/ops 下执行:"
  echo "    python setup.py build install"
  exit 1
fi

echo ""
echo "[4] MOT17 数据"
if [ -d "${P}/data/MOT17/train" ]; then
  echo "  OK: ${P}/data/MOT17/train"
else
  echo "  FAIL: 缺少 MOT17"
fi

echo ""
echo "[5] 预训练权重（跑 track.py 必需）"
CKPT="${P}/models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"
CFG="${P}/models/mot17_crowdhuman_deformable_multi_frame/config.yaml"
if [ -f "${CKPT}" ] && [ -f "${CFG}" ]; then
  echo "  OK: ${CKPT}"
  ls -lh "${CKPT}"
else
  echo "  FAIL: 缺少 checkpoint 或 config.yaml"
  echo "  运行: bash ${P}/scripts/download_models.sh"
  echo "  或按脚本内「手动步骤」上传 zip 后解压到 ${P}/models/"
fi

echo ""
echo "========== 检查结束 =========="
