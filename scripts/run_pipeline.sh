#!/bin/bash
set -e

PROJECT_ROOT="<PROJECT_ROOT>"
CODE_DIR="${PROJECT_ROOT}/code/trackformer"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
export CONDARC="${PROJECT_ROOT}/.condarc"

source ~/.bashrc 2>/dev/null || true
conda activate "${PROJECT_ROOT}/conda_envs/trackformer_grad"

cd "${CODE_DIR}"

usage() {
    echo "用法: $0 <模式> [选项]"
    echo ""
    echo "模式:"
    echo "  detect        仅运行目标检测 (Deformable DETR)"
    echo "  segment       运行实例分割 (MOTS20, 含 mask)"
    echo "  track         运行多目标跟踪 (MOT17, 不含 mask)"
    echo "  track_reid    运行多目标跟踪 + ReID"
    echo "  full          完整流程: 分割 + 识别 + 跟踪"
    echo "  demo          处理自定义视频"
    echo ""
    echo "选项:"
    echo "  --dataset     数据集名称 (默认: MOT17-ALL-ALL)"
    echo "  --checkpoint  模型权重路径"
    echo "  --output      输出目录"
    echo "  --visualize   生成可视化结果"
    echo ""
    echo "示例:"
    echo "  $0 track_reid"
    echo "  $0 segment --visualize"
    echo "  $0 demo --input /path/to/video.mp4"
    exit 1
}

MODE=${1:-""}
shift 2>/dev/null || true

DATASET="MOT17-ALL-ALL"
CHECKPOINT=""
VIS="False"
INPUT_VIDEO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --output) OUTPUT_DIR="$2"; shift 2;;
        --visualize) VIS="pretty"; shift;;
        --input) INPUT_VIDEO="$2"; shift 2;;
        *) echo "未知选项: $1"; usage;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_OUTPUT="${OUTPUT_DIR}/${MODE}_${TIMESTAMP}"
mkdir -p "${RUN_OUTPUT}"

case ${MODE} in
    detect)
        echo "====== 阶段 1: 目标检测 (Deformable DETR) ======"
        CKPT=${CHECKPOINT:-"models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"}
        python src/track.py with \
            dataset_name="${DATASET}" \
            obj_detect_checkpoint_file="${CKPT}" \
            output_dir="${RUN_OUTPUT}" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data"
        ;;

    segment)
        echo "====== 阶段 1: 实例分割 (MOTS20 + Mask Head) ======"
        CKPT=${CHECKPOINT:-"models/mots20_train_masks/checkpoint.pth"}
        python src/track.py with \
            dataset_name=MOTS20-ALL \
            obj_detect_checkpoint_file="${CKPT}" \
            output_dir="${RUN_OUTPUT}" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data"
        ;;

    track)
        echo "====== 多目标跟踪 (MOT17) ======"
        CKPT=${CHECKPOINT:-"models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"}
        python src/track.py with \
            dataset_name="${DATASET}" \
            obj_detect_checkpoint_file="${CKPT}" \
            output_dir="${RUN_OUTPUT}" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data"
        ;;

    track_reid)
        echo "====== 多目标跟踪 + ReID 识别 ======"
        CKPT=${CHECKPOINT:-"models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"}
        python src/track.py with \
            reid \
            dataset_name="${DATASET}" \
            obj_detect_checkpoint_file="${CKPT}" \
            output_dir="${RUN_OUTPUT}" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data"
        ;;

    full)
        echo "====== 完整流程: 分割 → 识别 → 跟踪 ======"
        echo ""
        echo "--- 步骤 1/3: 实例分割 (MOTS20) ---"
        CKPT_SEG=${CHECKPOINT:-"models/mots20_train_masks/checkpoint.pth"}
        python src/track.py with \
            dataset_name=MOTS20-ALL \
            obj_detect_checkpoint_file="${CKPT_SEG}" \
            output_dir="${RUN_OUTPUT}/step1_segmentation" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data" || echo "分割步骤需要 MOTS20 模型权重"

        echo ""
        echo "--- 步骤 2/3: ReID 识别 + 跟踪 (MOT17) ---"
        CKPT_TRACK="models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"
        python src/track.py with \
            reid \
            dataset_name="${DATASET}" \
            obj_detect_checkpoint_file="${CKPT_TRACK}" \
            output_dir="${RUN_OUTPUT}/step2_tracking_reid" \
            write_images="${VIS}" \
            data_root_dir="${PROJECT_ROOT}/data"

        echo ""
        echo "--- 步骤 3/3: 结果汇总 ---"
        python -c "
import os, json
output_dir = '${RUN_OUTPUT}'
print('='*50)
print('完整流程运行完成')
print('='*50)
print(f'分割结果: {output_dir}/step1_segmentation/')
print(f'跟踪结果: {output_dir}/step2_tracking_reid/')
for d in ['step1_segmentation', 'step2_tracking_reid']:
    p = os.path.join(output_dir, d)
    if os.path.exists(p):
        files = os.listdir(p)
        print(f'  {d}: {len(files)} 个输出文件')
"
        ;;

    demo)
        if [ -z "${INPUT_VIDEO}" ]; then
            echo "错误: demo 模式需要 --input 参数指定视频路径"
            exit 1
        fi

        DEMO_DIR="${RUN_OUTPUT}/demo_frames"
        mkdir -p "${DEMO_DIR}"

        echo "====== Demo: 处理自定义视频 ======"
        echo "[1/3] 提取视频帧..."
        ffmpeg -i "${INPUT_VIDEO}" -vf fps=30 "${DEMO_DIR}/%06d.png"

        echo "[2/3] 运行跟踪..."
        CKPT=${CHECKPOINT:-"models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"}
        python src/track.py with \
            dataset_name=DEMO \
            data_root_dir="${DEMO_DIR}" \
            obj_detect_checkpoint_file="${CKPT}" \
            output_dir="${RUN_OUTPUT}" \
            write_images=pretty

        echo "[3/3] 生成输出视频..."
        ffmpeg -f image2 -framerate 15 \
            -i "${RUN_OUTPUT}/DEMO/${DEMO_DIR}/%06d.jpg" \
            -vcodec libx264 -y \
            "${RUN_OUTPUT}/result.mp4" \
            -vf scale=640:-1

        echo "输出视频: ${RUN_OUTPUT}/result.mp4"
        ;;

    *)
        usage
        ;;
esac

echo ""
echo "=========================================="
echo "运行完成! 结果保存在: ${RUN_OUTPUT}"
echo "=========================================="
