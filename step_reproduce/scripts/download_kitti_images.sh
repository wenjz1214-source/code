#!/bin/bash
# Download and organize KITTI-STEP images
# KITTI tracking images (~15GB) from official site
#
# If automatic download is too slow, manually download from:
#   https://www.cvlibs.net/datasets/kitti/eval_tracking.php
#   -> "Download left color images of tracking data set (15 GB)"
#   Place the zip at: /share_data/wenjingzhong/kitti_step/images/data_tracking_image_2.zip
#   Then re-run this script.

set -e

KITTI_STEP_ROOT="/share_data/wenjingzhong/kitti_step"
IMAGE_DIR="${KITTI_STEP_ROOT}/images"

mkdir -p "${IMAGE_DIR}"
cd "${IMAGE_DIR}"

ZIP_FILE="data_tracking_image_2.zip"

if [ ! -f "${ZIP_FILE}" ]; then
    echo "============================================"
    echo "Downloading KITTI tracking images (~15GB)..."
    echo "This may take a while depending on network."
    echo "============================================"
    wget -c "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip" \
         -O "${ZIP_FILE}" || {
        echo "ERROR: Download failed or too slow."
        echo "Please download manually and place at: ${IMAGE_DIR}/${ZIP_FILE}"
        exit 1
    }
fi

if [ -d "train" ] && [ -d "val" ] && [ -d "test" ]; then
    echo "Images already organized. Skipping."
    exit 0
fi

echo "Extracting images..."
unzip -q "${ZIP_FILE}"

echo "Organizing directory structure..."
mv testing/image_02/ test/
rm -rf testing/

mkdir -p val
for seq in 0002 0006 0007 0008 0010 0013 0014 0016 0018; do
    mv "training/image_02/${seq}" "val/"
done

mv training/image_02/ train/
rm -rf training

echo "Train sequences: $(ls train/ | wc -l)"
echo "Val sequences:   $(ls val/ | wc -l)"
echo "Test sequences:  $(ls test/ | wc -l)"
echo "Done!"
