#!/bin/bash
# MOT17 Train 全流程：有 ReID / 无 ReID 两组实验，并生成 report.md
# 日志同时写入文件并在终端打印（之前 exec 只写文件导致看不到报错）

set -euo pipefail

P="${GRAD_PROJECT_ROOT:-<PROJECT_ROOT>}"
V="${P}/venv_tf"
LOG="${P}/logs/run_exp.log"
LASTERR="${P}/logs/run_exp.last_error.txt"

mkdir -p "${P}/logs" "${P}/outputs"

# 终端 + 日志双输出
echo ">>> 主日志: ${LOG}"
echo ">>> 若失败，末尾错误摘要: ${LASTERR}"
exec > >(tee -a "${LOG}") 2>&1

trap 'EC=$?; LINE=$LINENO; CMD=$BASH_COMMAND; \
  { echo ""; echo "========== 脚本失败 =========="; \
    echo "退出码: $EC  行号: $LINE"; echo "命令: $CMD"; \
    echo "--- 日志最后 120 行 ---"; tail -120 "${LOG}"; } | tee "${LASTERR}"; \
  exit "$EC"' ERR

export PATH="${V}/bin:${PATH}"
export MPLCONFIGDIR="${P}/logs/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo ""
echo "=== $(date) run_exp 开始 ==="

# 数据软链：代码内路径为 code/trackformer/data/MOT17
rm -f "${P}/code/trackformer/data/data"
ln -sfn "${P}/data/MOT17" "${P}/code/trackformer/data/MOT17"
ln -sfn "${P}/models" "${P}/code/trackformer/models" 2>/dev/null || true

echo ""
echo "STEP0: 检查环境与权重"
if ! "${P}/scripts/check_env.sh"; then
  echo "环境检查未通过，请先修复后再运行本脚本。"
  exit 1
fi

CKPT="${P}/models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth"
if [ ! -f "${CKPT}" ]; then
  echo ""
  echo "未找到权重，尝试 download_models.sh ..."
  bash "${P}/scripts/download_models.sh" || {
    echo "下载失败。请阅读 ${P}/logs/download_models.log 与脚本内说明。"
    exit 1
  }
fi

if [ ! -f "${CKPT}" ]; then
  echo "仍无 checkpoint: ${CKPT}"
  exit 1
fi

cd "${P}/code/trackformer"
TS=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="${P}/outputs/exp_${TS}"
mkdir -p "${EXP_DIR}"

echo ""
echo "=== ExpA: MOT17-TRAIN + ReID ==="
echo "开始: $(date)"
python src/track.py with reid \
  dataset_name=MOT17-TRAIN-ALL \
  obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
  output_dir="${EXP_DIR}/reid" \
  write_images=False \
  data_root_dir="${P}/data" \
  2>&1 | tee "${EXP_DIR}/a.log"
echo "结束: $(date)"

echo ""
echo "=== ExpB: MOT17-TRAIN 无 ReID ==="
echo "开始: $(date)"
python src/track.py with \
  dataset_name=MOT17-TRAIN-ALL \
  obj_detect_checkpoint_file=models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth \
  output_dir="${EXP_DIR}/no_reid" \
  write_images=False \
  data_root_dir="${P}/data" \
  2>&1 | tee "${EXP_DIR}/b.log"
echo "结束: $(date)"

echo ""
echo "STEP: 从日志提取指标写入 experiment_report.md"
python3 << PYEOF
import os, re, glob

ed = os.environ.get("EXP_DIR", "")
p = "${P}"

def extract_metrics(path):
    if not os.path.isfile(path):
        return []
    lines = open(path, errors="ignore").read().splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if any(k in s for k in (
            "MOTA", "IDF1", "IDF", "IDSW", "ID Sw", "MT", "ML", "FP", "FN",
            "OVERALL", "NUM TRACKS", "RUNTIME", "ReIDs", "EVAL", "mota", "idf1",
        )):
            out.append(s)
    return out

r = ["# 实验报告（本机运行结果）\n", f"实验目录: `{ed}`\n"]
for fn, title in [("a.log", "实验A: +ReID"), ("b.log", "实验B: 无ReID")]:
    r.append(f"## {title}\n")
    mp = os.path.join(ed, fn)
    rows = extract_metrics(mp)
    if rows:
        r.append("```")
        r.extend(rows[-80:])
        r.append("```\n")
    else:
        r.append("*未从日志解析到指标行，请直接打开日志查看。*\n")

out_md = os.path.join(ed, "report.md")
os.makedirs(os.path.join(p, "docs"), exist_ok=True)
with open(out_md, "w") as f:
    f.write("\n".join(r))
import shutil
shutil.copy2(out_md, os.path.join(p, "docs", "experiment_report.md"))
print("已写入:", out_md)
print("已复制:", os.path.join(p, "docs", "experiment_report.md"))
PYEOF

echo ""
echo "=== $(date) run_exp 全部完成 ==="
echo "结果目录: ${EXP_DIR}"
