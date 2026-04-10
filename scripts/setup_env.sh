#!/bin/bash
set -e

PROJECT_ROOT="<PROJECT_ROOT>"
ENV_PATH="${PROJECT_ROOT}/conda_envs/trackformer_grad"
export CONDARC="${PROJECT_ROOT}/.condarc"

echo "=========================================="
echo " TrackFormer 毕设环境安装脚本"
echo "=========================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "Conda 环境: ${ENV_PATH}"
echo ""

source ~/.bashrc 2>/dev/null || true

# Step 1: 创建 conda 环境
echo "[1/6] 创建 conda 环境 (Python 3.10)..."
if [ -d "${ENV_PATH}" ]; then
    echo "  -> 已存在，先删除旧环境..."
    rm -rf "${ENV_PATH}"
fi
conda create -p "${ENV_PATH}" python=3.10 pip -y
echo "  -> conda 环境创建完成"

# Step 2: 激活环境
echo "[2/6] 激活环境..."
conda activate "${ENV_PATH}"
echo "  -> Python: $(python --version)"
echo "  -> pip: $(python -m pip --version)"

# Step 3: 安装 PyTorch (使用官方 wheel 源，非清华 PyPI)
echo "[3/6] 安装 PyTorch 2.4.0 + CUDA 12.4..."
python -m pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124
echo "  -> PyTorch 安装完成"

# Step 4: 安装其他依赖 (使用清华 PyPI 源加速)
echo "[4/6] 安装项目依赖..."
TSINGHUA="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
python -m pip install -i ${TSINGHUA} \
    numpy==1.24.4 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    "opencv-python>=4.8,<4.10" \
    Pillow==10.2.0 \
    Cython==3.0.8 \
    pycocotools \
    motmetrics==1.4.0 \
    pandas==2.1.4 \
    sacred==0.8.5 \
    PyYAML==6.0.1 \
    tqdm==4.66.1 \
    scikit-image==0.22.0 \
    lap==0.4.0 \
    pyaml \
    submitit \
    seaborn \
    visdom \
    gdown
echo "  -> 依赖安装完成"

# Step 5: 安装 pycocotools (修复版)
echo "[5/6] 安装修复版 pycocotools..."
python -m pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
echo "  -> pycocotools 安装完成"

# Step 6: 安装 TrackFormer 本体
echo "[6/6] 安装 TrackFormer..."
cd "${PROJECT_ROOT}/code/trackformer"
python -m pip install -e .
echo "  -> TrackFormer 安装完成"

# 验证
echo ""
echo "=========================================="
echo " 验证安装"
echo "=========================================="
python -c "
import torch, torchvision, numpy, scipy, matplotlib, cv2, PIL
import motmetrics, sacred, yaml, tqdm
print('PyTorch:     ', torch.__version__)
print('TorchVision: ', torchvision.__version__)
print('CUDA 可用:   ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:   ', torch.version.cuda)
    print('GPU 设备:    ', torch.cuda.get_device_name(0))
print('NumPy:       ', numpy.__version__)
print('OpenCV:      ', cv2.__version__)
print()
print('所有依赖安装验证通过!')
"

echo ""
echo "=========================================="
echo " 安装完成!"
echo "=========================================="
echo ""
echo "激活环境:  export CONDARC=${PROJECT_ROOT}/.condarc && conda activate ${ENV_PATH}"
echo "项目目录:  cd ${PROJECT_ROOT}"
echo ""
