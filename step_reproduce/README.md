# STEP: Segmenting and Tracking Every Pixel - 论文复现

论文: [STEP: Segmenting and Tracking Every Pixel](https://arxiv.org/abs/2102.11859) (NeurIPS 2021)

基于 [DeepLab2](https://github.com/google-research/deeplab2) 官方实现

## 环境

- Conda 环境: `step_reproduce` (Python 3.9 + TF 2.6 + CUDA 11.2)
- GPU: **建议在 A100 / L40 / V100 等 Ampere 或更早架构上训练**（与 TF 2.6 预编译目标一致）

### 重要：为什么在 H100 上 loss 会变成 0？

这不是「步数太少」或「学习率写错」导致的。

1. `deeplab2/trainer/trainer.py` 里对 loss 做了 `tf.where(is_nan, 0.0, loss)`，所以 **真实情况是 loss / 梯度已经变成 NaN**，日志里才显示成全 0。
2. TensorFlow 2.6 发布时还没有 Hopper（H100，compute capability 9.0）。在 H100 上只能 **PTX 即时编译** CUDA kernel；实测会出现 **Conv 等算子反向传播得到 NaN 梯度**（可用仓库内 `test_grad.py`、`debug_data.py` 复现）。
3. 多卡时的 `NCCL: invalid usage` 也来自 **旧 NCCL + Hopper** 组合，与 DeepLab2 配置无关。

**可行做法**：换到带 A100 等机器用同一 conda 环境训练；或整体升级到 TensorFlow 2.12+（含 sm_90）并自行解决与 DeepLab2 / Orbit 的 API 差异（工作量较大）。

## 目录结构

```
/share_data/wenjingzhong/
├── graduation_project/step_reproduce/   # 代码
│   ├── deeplab2/                        # DeepLab2 库
│   ├── models/                          # TF models (含 Orbit)
│   ├── cocoapi/                         # pycocotools
│   └── scripts/                         # 运行脚本
│       ├── setup_env.sh                 # 环境变量设置
│       ├── download_kitti_images.sh     # 下载 KITTI 图像
│       ├── build_tfrecords.sh           # 生成 TFRecord
│       ├── train_panoptic_deeplab.sh    # 训练单帧基线
│       ├── train_motion_deeplab.sh      # 训练双帧基线
│       ├── eval_model.sh               # 评估模型
│       └── eval_stq.sh                 # STQ 指标评估
└── kitti_step/                          # 数据
    ├── images/                          # KITTI 图像 (需下载)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── panoptic_maps/                   # STEP 标注 (已下载)
    │   ├── train/
    │   └── val/
    ├── tfrecords/                       # TFRecord (待生成)
    └── checkpoints/                     # 预训练模型 (已下载)
```

## 使用步骤

### 1. 下载 KITTI 图像 (唯一需要手动操作的步骤)

从 KITTI 官网下载 tracking 图像:
- 访问 https://www.cvlibs.net/datasets/kitti/eval_tracking.php
- 注册/登录后下载 "left color images of tracking data set" (~15GB)
- 保存到 `/share_data/wenjingzhong/kitti_step/images/data_tracking_image_2.zip`
- 运行: `bash scripts/download_kitti_images.sh`

### 2. 生成 TFRecord

```bash
conda activate step_reproduce
bash scripts/build_tfrecords.sh
```

### 3. 训练模型

```bash
conda activate step_reproduce

# 单帧基线 (Panoptic-DeepLab)
bash scripts/train_panoptic_deeplab.sh [NUM_GPUS]

# 双帧基线 (Motion-DeepLab) - 论文核心方法
# 不传 NUM_GPUS 时默认使用 nvidia-smi 可见的全部 GPU（MirroredStrategy，config 里 batch_size 为全局 batch，会按卡数均分）。
# 若在 H100 上训出过 NaN，勿在同一 model_dir 续训；可另设目录，例如：
#   MOTION_DEEPLAB_MODEL_DIR=/share_data/wenjingzhong/kitti_step/model_output/motion_deeplab_kitti_step_ampere bash scripts/train_motion_deeplab.sh
bash scripts/train_motion_deeplab.sh [NUM_GPUS]

# 后台 + tmux + 按时间戳落盘日志（推荐长时间跑）
bash scripts/run_motion_train_tmux.sh
# 日志目录: /share_data/wenjingzhong/kitti_step/model_output/train_logs/（软链 motion_deeplab_ampere_latest.log）
```

### 4. 评估

```bash
conda activate step_reproduce
bash scripts/eval_model.sh panoptic   # 或 motion
bash scripts/eval_stq.sh <predictions_dir>
```

## 论文期望结果

| Model | PQ | AP Mask | mIoU | STQ |
|-------|-----|---------|------|-----|
| Panoptic-DeepLab (单帧) | 48.31 | 42.22 | 71.16 | - |
| Motion-DeepLab (双帧) | 42.08 | 37.52 | 63.15 | 57.7 |

## 已完成

- [x] Conda 环境 (Python 3.9 + TF 2.6 + CUDA 11.2)
- [x] DeepLab2 + Orbit + cocoapi 克隆与编译
- [x] Protobuf 编译
- [x] 自定义 ops 编译 (CPU)
- [x] TF GPU 检测验证 (8x H100)
- [x] KITTI-STEP 标注下载 (panoptic_maps)
- [x] 预训练 checkpoint 下载

## 待完成

- [ ] KITTI 图像下载 (~15GB, 需从官网获取)
- [ ] TFRecord 生成
- [ ] 模型训练
- [ ] 评估与结果对比
