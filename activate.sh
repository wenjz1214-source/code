#!/bin/bash
# 一键激活毕设环境
# 使用方法: source /share_data/wenjingzhong/graduation_project/activate.sh

export CONDARC="/share_data/wenjingzhong/graduation_project/.condarc"
export PROJECT_ROOT="/share_data/wenjingzhong/graduation_project"

source ~/.bashrc 2>/dev/null || true
conda activate "${PROJECT_ROOT}/conda_envs/trackformer_grad"

cd "${PROJECT_ROOT}"

echo "毕设环境已激活!"
echo "  项目目录: ${PROJECT_ROOT}"
echo "  Python:   $(python --version 2>&1)"
echo "  Conda 环境: ${CONDA_PREFIX}"
echo ""
echo "常用命令:"
echo "  bash scripts/setup_env.sh           # 首次安装环境"
echo "  bash scripts/download_data_models.sh # 下载数据和模型"
echo "  bash scripts/run_pipeline.sh track_reid --visualize  # 运行跟踪+ReID"
echo "  python scripts/pipeline.py --mode full               # 完整 pipeline"
