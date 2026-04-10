#!/bin/bash
set -e

PROJECT_ROOT="<PROJECT_ROOT>"
ENV_PATH="${PROJECT_ROOT}/conda_envs/trackformer_grad"
LOG="${PROJECT_ROOT}/logs/run_all.log"
export CONDARC="${PROJECT_ROOT}/.condarc"

mkdir -p "${PROJECT_ROOT}/logs"

exec > >(tee -a "${LOG}") 2>&1

echo "=========================================="
echo " [$(date)] 开始完整流程"
echo "=========================================="

# ============================================
# STEP 1: 创建 Conda 环境
# ============================================
echo ""
echo "[STEP 1/5] 创建 Conda 环境..."
source ~/.bashrc 2>/dev/null || true

if [ ! -f "${ENV_PATH}/bin/python" ]; then
    rm -rf "${ENV_PATH}"
    conda create -p "${ENV_PATH}" python=3.10 pip -y
    echo "  -> 环境创建完成"
else
    echo "  -> 环境已存在，跳过"
fi

conda activate "${ENV_PATH}"
echo "  Python: $(python --version)"
echo "  pip: $(python -m pip --version)"

# ============================================
# STEP 2: 安装依赖
# ============================================
echo ""
echo "[STEP 2/5] 安装依赖..."

python -c "import torch; print('torch', torch.__version__)" 2>/dev/null && TORCH_OK=1 || TORCH_OK=0

if [ "$TORCH_OK" = "0" ]; then
    echo "  安装 PyTorch 2.4.0 + CUDA 12.4..."
    python -m pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124
    echo "  -> PyTorch 安装完成"
else
    echo "  -> PyTorch 已安装，跳过"
fi

TSINGHUA="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

echo "  安装其他依赖 (清华源)..."
python -m pip install -i ${TSINGHUA} \
    "numpy>=1.24,<1.27" \
    "scipy>=1.11,<1.13" \
    "matplotlib>=3.7,<3.9" \
    "opencv-python>=4.8,<4.11" \
    "Pillow>=10.0,<10.4" \
    "Cython>=3.0" \
    pycocotools \
    "motmetrics>=1.2,<1.5" \
    "pandas>=2.0,<2.2" \
    "sacred>=0.8" \
    "PyYAML>=6.0" \
    "tqdm>=4.66" \
    "scikit-image>=0.21" \
    "lap>=0.4" \
    pyaml \
    submitit \
    seaborn \
    gdown \
    2>&1 | tail -5
echo "  -> 依赖安装完成"

echo "  安装 TrackFormer..."
cd "${PROJECT_ROOT}/code/trackformer"
python -m pip install -e . 2>&1 | tail -3
echo "  -> TrackFormer 安装完成"

# 创建软链接
ln -sfn "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/code/trackformer/data"
ln -sfn "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/code/trackformer/models"

echo ""
echo "  验证安装:"
python -c "
import torch, torchvision, numpy, scipy, cv2, PIL
import motmetrics, sacred, yaml, tqdm
print('  PyTorch:    ', torch.__version__)
print('  TorchVision:', torchvision.__version__)
print('  CUDA:       ', torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else '')
print('  NumPy:      ', numpy.__version__)
print('  OpenCV:     ', cv2.__version__)
print('  所有依赖 OK!')
"

# ============================================
# STEP 3: 下载数据集
# ============================================
echo ""
echo "[STEP 3/5] 下载数据集..."
cd "${PROJECT_ROOT}/data"

if [ ! -d "MOT17" ]; then
    echo "  下载 MOT17..."
    wget -q --show-progress https://motchallenge.net/data/MOT17.zip -O MOT17.zip
    unzip -q -o MOT17.zip
    rm MOT17.zip
    echo "  -> MOT17 下载完成"
else
    echo "  -> MOT17 已存在"
fi

echo "  生成 COCO 格式标注..."
cd "${PROJECT_ROOT}/code/trackformer"
python src/generate_coco_from_mot.py 2>&1 | tail -5
echo "  -> 标注生成完成"

# ============================================
# STEP 4: 下载预训练模型
# ============================================
echo ""
echo "[STEP 4/5] 下载预训练模型..."
cd "${PROJECT_ROOT}/models"

if [ ! -d "mot17_crowdhuman_deformable_multi_frame" ]; then
    echo "  下载预训练模型..."
    wget -q --show-progress https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip -O trackformer_models_v1.zip
    unzip -q -o trackformer_models_v1.zip
    rm trackformer_models_v1.zip
    echo "  -> 模型下载完成"
else
    echo "  -> 模型已存在"
fi

echo "  模型文件:"
ls -la "${PROJECT_ROOT}/models/"

# ============================================
# STEP 5: 运行实验
# ============================================
echo ""
echo "[STEP 5/5] 运行实验..."
cd "${PROJECT_ROOT}/code/trackformer"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="${PROJECT_ROOT}/outputs/experiment_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

echo ""
echo "===== 实验 A: MOT17 跟踪 + ReID ====="
echo "  数据集: MOT17-TRAIN"
echo "  模型: mot17_crowdhuman_deformable_multi_frame"
echo "  开始时间: $(date)"

python src/track.py with \
    reid \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/mot17_train_reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/mot17_train_reid.log"

echo ""
echo "  结束时间: $(date)"
echo ""

echo "===== 实验 B: MOT17 跟踪 (无 ReID) ====="
python src/track.py with \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/mot17_train_no_reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/mot17_train_no_reid.log"

echo ""
echo "=========================================="
echo " [$(date)] 全部完成!"
echo " 实验输出: ${EXP_DIR}"
echo "=========================================="

# 生成实验报告
python -c "
import os, re, json
exp_dir = '${EXP_DIR}'
report_lines = []
report_lines.append('# 实验报告')
report_lines.append('')
report_lines.append(f'生成时间: $(date)')
report_lines.append(f'实验目录: {exp_dir}')
report_lines.append('')

for exp_name in ['mot17_train_reid', 'mot17_train_no_reid']:
    log_file = os.path.join(exp_dir, f'{exp_name}.log')
    if os.path.exists(log_file):
        report_lines.append(f'## {exp_name}')
        report_lines.append('')
        report_lines.append('\`\`\`')
        with open(log_file) as f:
            content = f.read()
            # Extract the OVERALL line and evaluation metrics
            for line in content.split('\n'):
                if any(k in line for k in ['MOTA', 'IDF1', 'OVERALL', 'RUNTIME', 'NUM TRACKS', 'ReIDs', 'EVAL']):
                    report_lines.append(line.strip())
        report_lines.append('\`\`\`')
        report_lines.append('')

report = '\n'.join(report_lines)
report_path = os.path.join(exp_dir, 'experiment_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f'实验报告已生成: {report_path}')
" 2>&1

echo ""
echo "===== 日志文件位置 ====="
echo "完整日志: ${LOG}"
echo "实验日志: ${EXP_DIR}/"
ls -la "${EXP_DIR}/"
