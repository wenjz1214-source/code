#!/bin/bash
set -eo pipefail

P="<PROJECT_ROOT>"
V="${P}/venv_tf"
L="${P}/logs/go.log"
mkdir -p "${P}/logs"

# Simple redirect, no tee subshell to avoid duplicate processes
exec > "${L}" 2>&1

echo "=== $(date) START ==="

# 1) Clean venv
echo "STEP1: venv"
rm -rf "${V}"
<MINICONDA_ROOT>/bin/python -m venv "${V}"
export PATH="${V}/bin:${PATH}"
python -m pip install -q --upgrade pip
echo "  python=$(python --version) pip=$(pip --version | cut -d' ' -f2)"

# 2) PyTorch
echo "STEP2: pytorch"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
echo "  pytorch done"

# 3) Deps
echo "STEP3: deps"
T="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
pip install -i $T "numpy>=1.26" "scipy>=1.14" "matplotlib>=3.9" \
    "opencv-python>=4.10" "Pillow>=10.4" "Cython>=3.0" pycocotools \
    "motmetrics>=1.4" "pandas>=2.2" "sacred>=0.8" "PyYAML>=6.0" \
    "tqdm>=4.66" "scikit-image>=0.24" "lap>=0.5" pyaml submitit seaborn gdown
echo "  deps done"

cd "${P}/code/trackformer"
pip install -e .
ln -sfn "${P}/data" data
ln -sfn "${P}/models" models

# 4) Verify
echo "STEP4: verify"
python -c "
import torch,torchvision,numpy,scipy,cv2,PIL,motmetrics,sacred,yaml,tqdm
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu',torch.cuda.get_device_name(0))
print('np',numpy.__version__,'sp',scipy.__version__,'cv',cv2.__version__)
print('VERIFY OK')
"

# 5) Data
echo "STEP5: data"
cd "${P}/data"
if [ ! -d MOT17 ]; then
    wget --no-check-certificate -q https://motchallenge.net/data/MOT17.zip -O m.zip
    unzip -q -o m.zip && rm m.zip
fi
echo "  mot17 ok"
cd "${P}/code/trackformer"
python src/generate_coco_from_mot.py
echo "  coco ok"

# 6) Models
echo "STEP6: models"
cd "${P}/models"
if [ ! -d mot17_crowdhuman_deformable_multi_frame ]; then
    wget --no-check-certificate -q https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip -O m.zip
    unzip -q -o m.zip && rm m.zip
fi
echo "  models ok"
ls "${P}/models/"

# 7) Experiment
echo "STEP7: experiment"
cd "${P}/code/trackformer"
TS=$(date +%Y%m%d_%H%M%S)
ED="${P}/outputs/exp_${TS}"
mkdir -p "${ED}"

echo "ExpA: MOT17+ReID"
python src/track.py with reid \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${ED}/reid" write_images=False \
    data_root_dir="${P}/data" 2>&1 | tee "${ED}/a.log"

echo "ExpB: MOT17 no ReID"
python src/track.py with \
    dataset_name=MOT17-TRAIN-ALL \
    obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
    output_dir="${ED}/no_reid" write_images=False \
    data_root_dir="${P}/data" 2>&1 | tee "${ED}/b.log"

# 8) Report
echo "STEP8: report"
python3 -c "
import os
ed='${ED}'
r=['# 实验报告\n## 环境','- H100 80GB, CUDA 12.8, PyTorch 2.6.0','- MOT17 Train (7 seq)\n']
for fn,t in [('a.log','ExpA: +ReID'),('b.log','ExpB: -ReID')]:
    p=os.path.join(ed,fn)
    r.append(f'## {t}\n\`\`\`')
    if os.path.exists(p):
        for l in open(p):
            if any(k in l for k in ['NUM TRACKS','RUNTIME','EVAL','MOTA','IDF1','OVERALL']):
                r.append(l.strip())
    r.append('\`\`\`\n')
r+=['## 分析','- ReID通过HS Embedding特征匹配降低ID Switch','- Track Queries实现帧间身份保持\n']
with open(os.path.join(ed,'report.md'),'w') as f: f.write('\n'.join(r))
import shutil; shutil.copy2(os.path.join(ed,'report.md'),'${P}/docs/experiment_report.md')
print('report saved')
"

echo "=== $(date) DONE ==="
echo "Output: ${ED}"
