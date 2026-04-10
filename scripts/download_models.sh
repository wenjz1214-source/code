#!/bin/bash
# 下载 TrackFormer 官方预训练 zip（需能访问 vision.in.tum.de）
# 若集群屏蔽外网或 TUM 不可达：在本机/VPN 下载后放到 MODELS_DIR，或设置 TRACKFORMER_ZIP 指向已下载文件。

set -euo pipefail

P="${GRAD_PROJECT_ROOT:-/share_data/wenjingzhong/graduation_project}"
MODELS_DIR="${P}/models"
URL="https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip"
ZIP_NAME="trackformer_models_v1.zip"
ZIP_PATH="${TRACKFORMER_ZIP:-${MODELS_DIR}/${ZIP_NAME}}"
MIN_BYTES=$((900 * 1024 * 1024)) # ~0.9GB，完整包约 1.2GB

mkdir -p "${MODELS_DIR}"
LOG="${P}/logs/download_models.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== download_models $(date) ==="
echo "目标目录: ${MODELS_DIR}"
echo "ZIP 路径: ${ZIP_PATH}"

if [ -d "${MODELS_DIR}/mot17_crowdhuman_deformable_multi_frame" ] \
   && [ -f "${MODELS_DIR}/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth" ]; then
  echo "已存在 mot17_crowdhuman_deformable_multi_frame，跳过下载。"
  exit 0
fi

if [ -f "${ZIP_PATH}" ]; then
  SZ=$(stat -c%s "${ZIP_PATH}" 2>/dev/null || echo 0)
  echo "发现已有 ZIP，大小 ${SZ} bytes"
  if [ "${SZ}" -ge "${MIN_BYTES}" ]; then
    echo "解压到 ${MODELS_DIR} ..."
    unzip -o -q "${ZIP_PATH}" -d "${MODELS_DIR}"
    echo "完成。"
    exit 0
  fi
  echo "ZIP 过小或损坏，删除后重新下载..."
  rm -f "${ZIP_PATH}"
fi

echo "尝试从 TUM 下载（若失败请看下方「手动步骤」）..."
echo "URL: ${URL}"

if command -v curl >/dev/null 2>&1; then
  if curl -fL --connect-timeout 30 --retry 2 --retry-delay 5 --max-time 7200 \
      -o "${ZIP_PATH}.part" "${URL}"; then
    mv -f "${ZIP_PATH}.part" "${ZIP_PATH}"
  else
    rm -f "${ZIP_PATH}.part"
    echo ""
    echo "========== 自动下载失败 =========="
    echo "常见原因：1) 集群无法访问 vision.in.tum.de  2) 防火墙"
    echo ""
    echo "【手动步骤】在能访问外网的电脑执行："
    echo "  wget '${URL}' -O ${ZIP_NAME}"
    echo "然后上传到服务器："
    echo "  scp ${ZIP_NAME} <user>@<host>:${MODELS_DIR}/"
    echo "再在本机执行："
    echo "  export TRACKFORMER_ZIP=${MODELS_DIR}/${ZIP_NAME}"
    echo "  bash ${P}/scripts/download_models.sh"
    echo "=================================="
    exit 1
  fi
else
  echo "未找到 curl，请安装 curl 或使用 wget。"
  exit 1
fi

SZ=$(stat -c%s "${ZIP_PATH}")
echo "下载完成，大小 ${SZ} bytes"
if [ "${SZ}" -lt "${MIN_BYTES}" ]; then
  echo "错误: 文件过小，可能不是正确的 zip。"
  exit 1
fi

echo "解压..."
unzip -o -q "${ZIP_PATH}" -d "${MODELS_DIR}"
echo "解压完成。模型目录："
ls -la "${MODELS_DIR}/mot17_crowdhuman_deformable_multi_frame/" | head -20 || true
