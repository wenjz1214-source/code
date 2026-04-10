#!/bin/bash
set -e

PROJECT_ROOT="/share_data/wenjingzhong/graduation_project"
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/models"

echo "=========================================="
echo " 数据集 & 预训练模型下载脚本"
echo "=========================================="

# Step 1: 下载 MOT17 数据集
echo "[1/4] 下载 MOT17 数据集..."
cd "${DATA_DIR}"
if [ ! -d "MOT17" ]; then
    wget -c https://motchallenge.net/data/MOT17.zip -O MOT17.zip
    unzip -o MOT17.zip
    rm MOT17.zip
    echo "  -> MOT17 下载完成"
else
    echo "  -> MOT17 已存在，跳过"
fi

# Step 2: 下载 MOTS20 数据集 (用于分割任务)
echo "[2/4] 下载 MOTS20 数据集..."
if [ ! -d "MOTS" ]; then
    wget -c https://motchallenge.net/data/MOTS.zip -O MOTS.zip
    unzip -o MOTS.zip
    rm MOTS.zip
    echo "  -> MOTS20 下载完成"
else
    echo "  -> MOTS20 已存在，跳过"
fi

# Step 3: 生成 COCO 格式标注
echo "[3/4] 生成 COCO 格式标注..."
export CONDARC="${PROJECT_ROOT}/.condarc"
source ~/.bashrc 2>/dev/null || true
conda activate "${PROJECT_ROOT}/conda_envs/trackformer_grad"

cd "${PROJECT_ROOT}/code/trackformer"

# 为 TrackFormer 创建数据软链接
ln -sfn "${DATA_DIR}" "${PROJECT_ROOT}/code/trackformer/data"
ln -sfn "${MODEL_DIR}" "${PROJECT_ROOT}/code/trackformer/models"

python src/generate_coco_from_mot.py
python src/generate_coco_from_mot.py --mots
echo "  -> COCO 标注生成完成"

# Step 4: 下载预训练模型
echo "[4/4] 下载预训练模型..."
cd "${MODEL_DIR}"
if [ ! -f "trackformer_models_v1.zip" ] && [ ! -d "mot17_crowdhuman_deformable_multi_frame" ]; then
    wget -c https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
    unzip -o trackformer_models_v1.zip
    rm trackformer_models_v1.zip
    echo "  -> 预训练模型下载完成"
else
    echo "  -> 预训练模型已存在，跳过"
fi

echo ""
echo "=========================================="
echo " 数据和模型下载完成!"
echo "=========================================="
echo "数据目录:  ${DATA_DIR}"
echo "模型目录:  ${MODEL_DIR}"
echo ""
ls -la "${DATA_DIR}"
echo ""
ls -la "${MODEL_DIR}"
