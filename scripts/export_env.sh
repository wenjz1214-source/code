#!/bin/bash
# 导出可复现的环境配置文件

PROJECT_ROOT="/share_data/wenjingzhong/graduation_project"
ENV_PATH="${PROJECT_ROOT}/conda_envs/trackformer_grad"
export CONDARC="${PROJECT_ROOT}/.condarc"

source ~/.bashrc 2>/dev/null || true
conda activate "${ENV_PATH}"

echo "导出环境配置..."

# 1. Conda environment.yml
conda env export -p "${ENV_PATH}" > "${PROJECT_ROOT}/environment.yml"
echo "  -> environment.yml"

# 2. pip requirements
python -m pip freeze > "${PROJECT_ROOT}/pip_requirements.txt"
echo "  -> pip_requirements.txt"

# 3. Conda 精确规格 (可跨平台复现)
conda list -p "${ENV_PATH}" --explicit > "${PROJECT_ROOT}/conda_explicit_spec.txt"
echo "  -> conda_explicit_spec.txt"

echo ""
echo "环境配置已导出到 ${PROJECT_ROOT}/"
echo ""
echo "复现环境方法:"
echo "  方式1 (推荐): bash scripts/setup_env.sh"
echo "  方式2: conda env create -f environment.yml -p /path/to/new_env"
echo "  方式3: conda create --file conda_explicit_spec.txt -p /path/to/new_env"
