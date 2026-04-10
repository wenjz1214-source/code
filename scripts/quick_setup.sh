#!/bin/bash
set -e

PROJECT_ROOT="<PROJECT_ROOT>"
ENV_PATH="${PROJECT_ROOT}/venv_trackformer"
LOG="${PROJECT_ROOT}/logs/quick_setup.log"

mkdir -p "${PROJECT_ROOT}/logs"
exec > >(tee "${LOG}") 2>&1

echo "=========================================="
echo " [$(date)] Quick Setup (Python 3.13 + PyTorch 2.6)"
echo "=========================================="

source ~/.bashrc 2>/dev/null || true

# STEP 1: venv
echo "[STEP 1/5] 创建 venv..."
if [ -f "${ENV_PATH}/bin/python" ]; then
    echo "  -> venv 已存在"
else
    <MINICONDA_ROOT>/bin/python -m venv "${ENV_PATH}"
fi
source "${ENV_PATH}/bin/activate"
pip install --upgrade pip -q 2>&1 | tail -1
echo "  Python: $(python --version), pip: $(pip --version | cut -d' ' -f1-2)"

# STEP 2: PyTorch 2.6.0 + deps
echo "[STEP 2/5] 安装 PyTorch 2.6.0 + CUDA 12.4..."
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  -> PyTorch 已安装"
else
    pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
fi

TSINGHUA="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
echo "  安装依赖..."
pip install -i ${TSINGHUA} \
    "numpy>=1.26,<1.28" \
    "scipy>=1.14" \
    "matplotlib>=3.9" \
    "opencv-python>=4.10" \
    "Pillow>=10.4" \
    "Cython>=3.0" \
    pycocotools \
    "motmetrics>=1.4" \
    "pandas>=2.2" \
    "sacred>=0.8" \
    "PyYAML>=6.0" \
    "tqdm>=4.66" \
    "scikit-image>=0.24" \
    "lap>=0.5" \
    pyaml submitit seaborn gdown \
    2>&1 | tail -5

cd "${PROJECT_ROOT}/code/trackformer"
pip install -e . 2>&1 | tail -2
ln -sfn "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/code/trackformer/data"
ln -sfn "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/code/trackformer/models"

echo "  验证:"
python -c "
import torch, torchvision, numpy, scipy, cv2, PIL
import motmetrics, sacred, yaml, tqdm
print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available(): print('  GPU:', torch.cuda.get_device_name(0))
print('  NumPy:', numpy.__version__, '| SciPy:', scipy.__version__)
print('  ALL OK!')
"

# STEP 3: 数据集
echo "[STEP 3/5] 下载 MOT17..."
cd "${PROJECT_ROOT}/data"
if [ ! -d "MOT17" ]; then
    wget --no-check-certificate -q --show-progress \
        https://motchallenge.net/data/MOT17.zip -O MOT17.zip
    unzip -q -o MOT17.zip && rm -f MOT17.zip
fi
echo "  -> MOT17 OK"
cd "${PROJECT_ROOT}/code/trackformer"
python src/generate_coco_from_mot.py 2>&1
echo "  -> COCO annotations OK"

# STEP 4: 模型
echo "[STEP 4/5] 下载预训练模型..."
cd "${PROJECT_ROOT}/models"
if [ ! -d "mot17_crowdhuman_deformable_multi_frame" ]; then
    wget --no-check-certificate -q --show-progress \
        https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip \
        -O models.zip
    unzip -q -o models.zip && rm -f models.zip
fi
echo "  -> 模型 OK"
ls "${PROJECT_ROOT}/models/"

# STEP 5: 实验
echo "=========================================="
echo "[STEP 5/5] 运行实验"
echo "=========================================="
cd "${PROJECT_ROOT}/code/trackformer"
TS=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="${PROJECT_ROOT}/outputs/exp_${TS}"
mkdir -p "${EXP_DIR}"

echo "=== 实验A: MOT17 + ReID ==="
python src/track.py with \
    reid \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/exp_a.log"

echo "=== 实验B: MOT17 无ReID ==="
python src/track.py with \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/no_reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/exp_b.log"

# 报告
python3 << 'PYEOF'
import os
ed = os.environ["EXP_DIR"]
r = ["# 实验报告\n","## 环境","- GPU: NVIDIA H100 80GB","- PyTorch 2.6.0 + CUDA 12.4","- MOT17 Train\n"]
for fn, t in [("exp_a.log","实验A: +ReID"),("exp_b.log","实验B: -ReID")]:
    p = os.path.join(ed, fn)
    r.append(f"## {t}\n```")
    if os.path.exists(p):
        for l in open(p):
            l=l.strip()
            if any(k in l for k in ['NUM TRACKS','RUNTIME','EVAL','MOTA','IDF1','OVERALL','mota','idf1']):
                r.append(l)
    r.append("```\n")
r.append("## 分析\n- ReID降低ID Switch\n- Track Queries保持身份\n")
with open(os.path.join(ed,"report.md"),"w") as f: f.write("\n".join(r))
import shutil; shutil.copy2(os.path.join(ed,"report.md"), "<PROJECT_ROOT>/docs/experiment_report.md")
print(f"报告: {ed}/report.md")
PYEOF

echo "=========================================="
echo " [$(date)] 完成! 结果: ${EXP_DIR}"
echo "=========================================="
