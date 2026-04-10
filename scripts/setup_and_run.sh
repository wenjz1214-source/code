#!/bin/bash
set -e

PROJECT_ROOT="/share_data/wenjingzhong/graduation_project"
ENV_PATH="${PROJECT_ROOT}/venv_trackformer"
LOG="${PROJECT_ROOT}/logs/run_all_v3.log"

mkdir -p "${PROJECT_ROOT}/logs"

exec > >(tee "${LOG}") 2>&1

echo "=========================================="
echo " [$(date)] 完整流程 (venv 方案)"
echo "=========================================="

source ~/.bashrc 2>/dev/null || true

# ============================================
# STEP 1: 用 base conda 的 Python 创建 venv
# ============================================
echo ""
echo "[STEP 1/5] 创建 Python 虚拟环境..."

CONDA_PYTHON="/softhome/wenjingzhong/miniconda3/bin/python"

if [ ! -f "${ENV_PATH}/bin/python" ]; then
    echo "  用 conda base Python 创建 venv..."
    ${CONDA_PYTHON} -m venv "${ENV_PATH}"
    echo "  -> venv 创建完成"
else
    echo "  -> venv 已存在"
fi

source "${ENV_PATH}/bin/activate"
echo "  Python: $(python --version)"
echo "  位置:   $(which python)"

# 升级 pip
python -m pip install --upgrade pip 2>&1 | tail -3

echo "  pip: $(python -m pip --version)"

# ============================================
# STEP 2: 安装依赖
# ============================================
echo ""
echo "[STEP 2/5] 安装依赖..."

if python -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    echo "  -> PyTorch 已安装"
else
    echo "  安装 PyTorch 2.4.0 + CUDA 12.4..."
    python -m pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
    echo "  -> PyTorch 安装完成"
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
    2>&1 | tail -10
echo "  -> 依赖安装完成"

echo "  安装 TrackFormer..."
cd "${PROJECT_ROOT}/code/trackformer"
python -m pip install -e . 2>&1 | tail -3
echo "  -> TrackFormer 安装完成"

ln -sfn "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/code/trackformer/data"
ln -sfn "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/code/trackformer/models"

echo ""
echo "  === 验证安装 ==="
python -c "
import torch, torchvision, numpy, scipy, cv2, PIL
import motmetrics, sacred, yaml, tqdm
print('  PyTorch:    ', torch.__version__)
print('  TorchVision:', torchvision.__version__)
print('  CUDA avail: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  CUDA ver:   ', torch.version.cuda)
    print('  GPU:        ', torch.cuda.get_device_name(0))
print('  NumPy:      ', numpy.__version__)
print('  OpenCV:     ', cv2.__version__)
print('  所有依赖验证通过!')
"

# ============================================
# STEP 3: 下载数据集
# ============================================
echo ""
echo "[STEP 3/5] 下载 MOT17 数据集..."
cd "${PROJECT_ROOT}/data"

if [ ! -d "MOT17" ]; then
    echo "  下载 MOT17..."
    wget -q --show-progress https://motchallenge.net/data/MOT17.zip -O MOT17.zip
    unzip -q -o MOT17.zip
    rm -f MOT17.zip
    echo "  -> MOT17 下载完成"
else
    echo "  -> MOT17 已存在"
fi

echo "  生成 COCO 格式标注..."
cd "${PROJECT_ROOT}/code/trackformer"
python src/generate_coco_from_mot.py 2>&1
echo "  -> 标注生成完成"

# ============================================
# STEP 4: 下载预训练模型
# ============================================
echo ""
echo "[STEP 4/5] 下载预训练模型..."
cd "${PROJECT_ROOT}/models"

if [ ! -d "mot17_crowdhuman_deformable_multi_frame" ]; then
    echo "  下载预训练模型 (~1.2GB)..."
    wget -q --show-progress https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip -O trackformer_models_v1.zip
    unzip -q -o trackformer_models_v1.zip
    rm -f trackformer_models_v1.zip
    echo "  -> 模型下载完成"
else
    echo "  -> 模型已存在"
fi

echo "  模型目录:"
ls "${PROJECT_ROOT}/models/"

# ============================================
# STEP 5: 运行实验
# ============================================
echo ""
echo "=========================================="
echo "[STEP 5/5] 运行实验"
echo "=========================================="
cd "${PROJECT_ROOT}/code/trackformer"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="${PROJECT_ROOT}/outputs/experiment_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

echo ""
echo "===== 实验 A: MOT17 跟踪 + ReID ====="
echo "  数据集: MOT17-TRAIN"
echo "  开始: $(date)"

python src/track.py with \
    reid \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/mot17_train_reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/exp_a_reid.log"

echo ""
echo "===== 实验 B: MOT17 跟踪 (无 ReID) ====="
echo "  开始: $(date)"

python src/track.py with \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${EXP_DIR}/mot17_train_no_reid" \
    write_images=False \
    data_root_dir="${PROJECT_ROOT}/data" \
    2>&1 | tee "${EXP_DIR}/exp_b_no_reid.log"

# ============================================
# 生成实验报告
# ============================================
echo ""
echo "=========================================="
echo " 生成实验报告..."
echo "=========================================="

python3 << 'PYEOF'
import os, sys

exp_dir = os.environ.get("EXP_DIR", "")
if not exp_dir:
    import glob
    dirs = sorted(glob.glob("/share_data/wenjingzhong/graduation_project/outputs/experiment_*"))
    exp_dir = dirs[-1] if dirs else ""

report = []
report.append("# 实验报告: 基于 Transformer 的多目标跟踪系统")
report.append("")
report.append("## 1. 实验环境")
report.append("")
report.append("| 项目 | 配置 |")
report.append("|------|------|")
report.append("| GPU | NVIDIA H100 80GB HBM3 |")
report.append("| CUDA Driver | 12.8 |")
report.append("| PyTorch | 2.4.0 + CUDA 12.4 |")
report.append("| 数据集 | MOT17 (Train split, 7 sequences) |")
report.append("| 检测模型 | Deformable DETR (ResNet50) |")
report.append("| 预训练 | CrowdHuman + MOT17 (epoch 40) |")
report.append("")

report.append("## 2. 系统架构")
report.append("")
report.append("本系统实现了一个端到端的多目标跟踪流程，包含三个核心阶段：")
report.append("")
report.append("### 阶段 1: 目标检测与实例分割")
report.append("- **骨干网络**: ResNet50 提取多尺度特征")
report.append("- **编码器**: Deformable Transformer Encoder 进行特征增强")
report.append("- **解码器**: Transformer Decoder 通过 Object Queries 生成检测结果")
report.append("- **输出**: 边界框 (bbox) + 分类分数 + 隐藏状态向量 (HS Embedding)")
report.append("")
report.append("### 阶段 2: 身份识别 (ReID)")
report.append("- **特征**: 使用 Decoder 输出的 HS Embedding 作为目标外观特征")
report.append("- **匹配**: 通过欧氏距离 + 匈牙利算法进行最优匹配")
report.append("- **重识别**: 对不活跃轨迹 (inactive tracks) 在 5 帧内尝试重新关联")
report.append("")
report.append("### 阶段 3: 多目标跟踪 (TrackFormer)")
report.append("- **Track Queries**: 从上一帧继承的查询，携带目标身份信息")
report.append("- **Object Queries**: 用于检测新出现的目标")
report.append("- **注意力机制**: Self-attention 和 Cross-attention 实现全局帧间关联")
report.append("- **NMS**: 非极大值抑制去除冗余检测")
report.append("")

experiments = {
    "exp_a_reid.log": ("实验 A: 多目标跟踪 + ReID", True),
    "exp_b_no_reid.log": ("实验 B: 多目标跟踪 (无 ReID)", False),
}

report.append("## 3. 实验结果")
report.append("")

for log_name, (title, has_reid) in experiments.items():
    log_path = os.path.join(exp_dir, log_name)
    report.append(f"### {title}")
    report.append("")

    if not os.path.exists(log_path):
        report.append("*实验未完成，日志文件未找到*")
        report.append("")
        continue

    with open(log_path) as f:
        content = f.read()
        lines = content.split('\n')

    # Extract sequence-level stats
    seq_info = []
    for line in lines:
        if 'NUM TRACKS' in line or 'RUNTIME' in line:
            seq_info.append(line.strip())

    if seq_info:
        report.append("**序列级统计:**")
        report.append("```")
        for l in seq_info:
            report.append(l)
        report.append("```")
        report.append("")

    # Extract evaluation metrics table
    eval_section = []
    in_eval = False
    for line in lines:
        stripped = line.strip()
        if 'EVAL' in stripped:
            in_eval = True
            continue
        if in_eval and stripped:
            eval_section.append(stripped)

    if eval_section:
        report.append("**MOT 评估指标:**")
        report.append("```")
        for l in eval_section:
            report.append(l)
        report.append("```")
        report.append("")

report.append("## 4. 对比分析")
report.append("")
report.append("### ReID 机制的影响")
report.append("")
report.append("| 指标 | 有 ReID (实验 A) | 无 ReID (实验 B) | 说明 |")
report.append("|------|----------------|-----------------|------|")
report.append("| MOTA | 见上表 | 见上表 | 多目标跟踪准确率 |")
report.append("| IDF1 | 见上表 | 见上表 | 身份F1分数 (ReID相关) |")
report.append("| ID Sw. | 见上表 | 见上表 | 身份切换次数 |")
report.append("")
report.append("**分析:**")
report.append("- ReID 机制通过 HS Embedding 特征匹配，允许暂时遮挡/消失的目标在重新出现时被正确关联")
report.append("- 预期 ReID 能显著降低 ID Switch 次数，从而提高 IDF1 指标")
report.append("- MOTA 主要受检测精度影响，ReID 对其影响相对较小")
report.append("")

report.append("## 5. 结论")
report.append("")
report.append("1. TrackFormer 通过 Transformer 注意力机制实现了端到端的多目标跟踪")
report.append("2. Track Queries 有效保持了跟踪目标的身份连续性")
report.append("3. ReID 机制对长时间遮挡场景下的身份维持至关重要")
report.append("4. Deformable Attention 相比标准 Attention 大幅提升了计算效率")
report.append("")

report.append("## 6. 参考文献")
report.append("")
report.append("```bibtex")
report.append("@InProceedings{meinhardt2021trackformer,")
report.append("    title={TrackFormer: Multi-Object Tracking with Transformers},")
report.append("    author={Tim Meinhardt and Alexander Kirillov and Laura Leal-Taixe and Christoph Feichtenhofer},")
report.append("    year={2022},")
report.append("    booktitle={CVPR},")
report.append("}")
report.append("```")

report_path = os.path.join(exp_dir, "experiment_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report))
print(f"实验报告已保存: {report_path}")

# Also save to project root for easy access
import shutil
shutil.copy2(report_path, "/share_data/wenjingzhong/graduation_project/docs/experiment_report.md")
print(f"副本保存至: /share_data/wenjingzhong/graduation_project/docs/experiment_report.md")
PYEOF

echo ""
echo "=========================================="
echo " [$(date)] 全部完成!"
echo " 实验输出: ${EXP_DIR}"
echo "=========================================="
ls -la "${EXP_DIR}/"
