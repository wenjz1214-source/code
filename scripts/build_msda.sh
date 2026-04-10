#!/bin/bash
# 编译安装 MultiScaleDeformableAttention（Deformable DETR CUDA 算子）
set -euo pipefail
P="${GRAD_PROJECT_ROOT:-/share_data/wenjingzhong/graduation_project}"
V="${P}/venv_tf"
export PATH="${V}/bin:${PATH}"
cd "${P}/code/trackformer/src/trackformer/models/ops"
echo ">>> 在 $(pwd) 编译 MSDA ..."
"${V}/bin/python" setup.py build install
echo ">>> 验证（须先 import torch）:"
"${V}/bin/python" -c "import torch; import MultiScaleDeformableAttention; print('OK')"
