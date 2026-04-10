#!/bin/bash
# Download and prepare MOTChallenge-STEP data.
#
# Usage:
#   conda activate step_reproduce
#   bash scripts/download_motchallenge_step.sh

set -euo pipefail

source "$(dirname "$0")/setup_env.sh"
cd "${PROJECT_ROOT}"

MOTSTEP_ROOT="<MOTCHALLENGE_STEP_ROOT>"
RAW_DIR="${MOTSTEP_ROOT}/raw"
IMG_DIR="${MOTSTEP_ROOT}/images"

MOTS_URL="https://motchallenge.net/data/MOTS.zip"
GT_URL="https://motchallenge.net/data/motchallenge-step.tar.gz"

mkdir -p "${RAW_DIR}" "${IMG_DIR}"

echo "============================================"
echo "Preparing MOTChallenge-STEP"
echo "  Root: ${MOTSTEP_ROOT}"
echo "  Raw:  ${RAW_DIR}"
echo "============================================"

cd "${RAW_DIR}"

if [ ! -f MOTS.zip ]; then
  echo "Downloading MOTS.zip ..."
  wget -c "${MOTS_URL}" -O MOTS.zip
else
  echo "Found existing MOTS.zip"
fi

if [ ! -f motchallenge-step.tar.gz ]; then
  echo "Downloading motchallenge-step.tar.gz ..."
  wget -c "${GT_URL}" -O motchallenge-step.tar.gz
else
  echo "Found existing motchallenge-step.tar.gz"
fi

if [ ! -d MOTS ]; then
  echo "Unzipping MOTS.zip ..."
  unzip -o MOTS.zip
fi

mkdir -p "${IMG_DIR}/train/0002" "${IMG_DIR}/train/0009"
mkdir -p "${IMG_DIR}/test/0001" "${IMG_DIR}/test/0007"

echo "Syncing image folders ..."
cp -a MOTS/train/MOTS20-02/img1/. "${IMG_DIR}/train/0002/"
cp -a MOTS/train/MOTS20-09/img1/. "${IMG_DIR}/train/0009/"
cp -a MOTS/test/MOTS20-01/img1/. "${IMG_DIR}/test/0001/"
cp -a MOTS/test/MOTS20-07/img1/. "${IMG_DIR}/test/0007/"

cd "${MOTSTEP_ROOT}"
if [ ! -d panoptic_maps ]; then
  echo "Extracting motchallenge-step.tar.gz ..."
  tar -xvf "${RAW_DIR}/motchallenge-step.tar.gz"
fi

echo "Done. Expected layout:"
echo "  ${MOTSTEP_ROOT}/images/train/{0002,0009}"
echo "  ${MOTSTEP_ROOT}/images/test/{0001,0007}"
echo "  ${MOTSTEP_ROOT}/panoptic_maps/..."
