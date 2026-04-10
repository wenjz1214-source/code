# Graduation Project Workspace

本仓库是一个毕业设计工作区，不是单一上游项目。当前实际包含两条主线：

- `STEP / DeepLab2`：论文 **STEP: Segmenting and Tracking Every Pixel** 复现，当前是主线。
- `TrackFormer`：早期 MOT 跟踪实验保留代码，当前不是主结果来源。

如果你的目标是复现实验、查看结果、继续训练或评估，请优先看 `step_reproduce/`。

## 1. 当前状态

### 1.1 主复现对象

当前主复现对象是论文：

- `STEP: Segmenting and Tracking Every Pixel`

代码基础来自：

- `step_reproduce/deeplab2/`：Google Research DeepLab2
- `step_reproduce/models/`：TensorFlow Models / Orbit
- `step_reproduce/cocoapi/`：COCO API

### 1.2 当前已经完成的主要结果

#### KITTI-STEP

`B1: IoU Association`

- `STQ = 0.5681`
- `AQ = 0.4724`
- `IoU / SQ = 0.6831`

`B4: Motion-DeepLab`

- `STQ = 0.5698`
- `AQ = 0.5093`
- `IoU / SQ = 0.6376`
- `PQ = 0.4354`
- `AP_Mask = 0.3990`
- `mIoU = 0.6383`

#### MOTChallenge-STEP

`B4: Motion-DeepLab (fixinit 版本)`

目前已经修到可正常输出非零指标。

最新全量评估结果：

- `STQ = 0.2509`
- `AQ = 0.1028`
- `IoU / SQ = 0.6123`
- `PQ = 0.5237`
- `AP_Mask = 0.0607`

说明：

- 这条线最初因为 `BatchNorm + batch_size=1 + eval moving stats` 导致评估时输出塌缩为全背景。
- 目前代码里已经加入针对 `motchallenge_step` 的评估修复，结果不再是全 0。

### 1.3 和论文结果的对应关系

论文主指标是：

- `STQ`：整体视频分割跟踪质量
- `AQ`：关联质量
- `SQ / IoU`：分割质量

论文中的关键参考值：

#### KITTI-STEP

- `B1`: `STQ 0.58 / AQ 0.47 / SQ 0.71`
- `B4`: `STQ 0.58 / AQ 0.51 / SQ 0.67`  
  DeepLab2 文档给出的 `Motion-DeepLab` 参考值还包括：
  `PQ 42.08 / AP_Mask 37.52 / mIoU 63.15 / STQ 57.7`

#### MOTChallenge-STEP

- `B4`: `STQ 0.35 / AQ 0.19 / SQ 0.62`

## 2. 目录结构

根目录下重要目录如下：

```text
graduation_project/
├── README.md
├── docs/                           # 论文写作、中期报告、计划文档
├── scripts/                        # TrackFormer 相关辅助脚本
├── code/
│   └── trackformer/                # TrackFormer 源码
├── step_reproduce/                 # STEP 复现主目录
│   ├── README.md
│   ├── deeplab2/                   # DeepLab2 主代码
│   ├── models/                     # TensorFlow Models / Orbit
│   ├── cocoapi/                    # pycocotools
│   └── scripts/                    # STEP 训练/评估脚本
├── data/                           # 早期 TrackFormer 数据目录
├── logs/
├── outputs/
├── conda_envs/
└── conda_pkgs/
```

## 3. 实验环境

当前这套实验实际使用的环境：

- OS：`Ubuntu 22.04.5 LTS`
- GPU：`NVIDIA A16 x4`
- Driver：`580.126.16`
- Python：`3.9.23`
- TensorFlow：`2.6.0`
- CUDA（TF build）：`11.2`
- cuDNN（TF build）：`8.1`

推荐直接使用固定 Python：

```bash
/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python
```

### 3.1 为什么不用 H100

在本工作区中，`TensorFlow 2.6 + H100` 会出现数值不稳定问题，训练中容易出现：

- `NaN`
- loss 记录为 `0`
- 多卡 `NCCL` 问题

因此当前主结果统一在 `A16` 上完成。

## 4. 数据与默认路径

### 4.1 KITTI-STEP

- 图像：`/share_data/wenjingzhong/kitti_step/images`
- 标注：`/share_data/wenjingzhong/kitti_step/panoptic_maps`
- TFRecord：`/share_data/wenjingzhong/kitti_step/tfrecords`
- 预训练与初始化 checkpoint：`/share_data/wenjingzhong/kitti_step/checkpoints`
- 训练输出：`/share_data/wenjingzhong/kitti_step/model_output`

### 4.2 MOTChallenge-STEP

- 图像：`/share_data/wenjingzhong/motchallenge_step/images`
- 标注：`/share_data/wenjingzhong/motchallenge_step/panoptic_maps`
- TFRecord：`/share_data/wenjingzhong/motchallenge_step/tfrecords`
- 初始化 checkpoint：`/share_data/wenjingzhong/motchallenge_step/checkpoints`
- 训练输出：`/share_data/wenjingzhong/motchallenge_step/model_output`

## 5. 进入环境

### 5.1 推荐方式

```bash
cd /share_data/wenjingzhong/graduation_project/step_reproduce
source scripts/setup_env.sh
```

然后直接调用固定 Python：

```bash
/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python
```

### 5.2 可选方式

如果你本机能正常使用 Conda，也可以：

```bash
conda activate step_reproduce
```

但当前仓库更推荐上一种方式，因为脚本已经绑定了：

- `PYTHONPATH`
- `LD_LIBRARY_PATH`
- `deeplab2 / models / cocoapi`

## 6. 常用命令

以下命令默认在：

```bash
cd /share_data/wenjingzhong/graduation_project/step_reproduce
source scripts/setup_env.sh
```

之后执行。

### 6.1 KITTI-STEP：生成 TFRecord

```bash
bash scripts/build_tfrecords.sh
```

### 6.2 KITTI-STEP：训练 B1 前置模型（Panoptic-DeepLab）

```bash
bash scripts/train_panoptic_deeplab.sh 1
```

默认输出目录：

- `/share_data/wenjingzhong/kitti_step/model_output/panoptic_deeplab_kitti_step_a16_safe`

### 6.3 KITTI-STEP：训练 B4（Motion-DeepLab）

```bash
bash scripts/train_motion_deeplab.sh 1
```

默认输出目录：

- `/share_data/wenjingzhong/kitti_step/model_output/motion_deeplab_kitti_step_a16_safe`

### 6.4 KITTI-STEP：评估 B4

```bash
bash scripts/eval_model.sh motion
```

或直接：

```bash
/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python -u deeplab2/trainer/train.py \
  --config_file=deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto \
  --mode=eval \
  --model_dir=/share_data/wenjingzhong/kitti_step/model_output/motion_deeplab_kitti_step_a16_safe \
  --num_gpus=1
```

### 6.5 KITTI-STEP：评估 B1

先保证 B1 的单帧预测已经生成，然后运行：

```bash
bash scripts/eval_b1_iou_assoc.sh
```

默认脚本会读取：

- 预测：`/share_data/wenjingzhong/kitti_step/model_output/panoptic_deeplab_kitti_step_a16_safe/vis_ckpt30000/raw_panoptic`
- GT：`/share_data/wenjingzhong/kitti_step/panoptic_maps/val`

输出目录：

- `/share_data/wenjingzhong/kitti_step/model_output/b1_iou_assoc_kitti_step`

### 6.6 MOTChallenge-STEP：下载与准备数据

```bash
bash scripts/download_motchallenge_step.sh
bash scripts/build_motchallenge_tfrecords.sh
```

### 6.7 MOTChallenge-STEP：训练 B4

```bash
bash scripts/train_motion_deeplab_motchallenge.sh 1
```

这个脚本会自动生成本地配置：

- `/tmp/motchallenge_motion_resnet50_os32_local.textproto`

默认输出目录：

- `/share_data/wenjingzhong/motchallenge_step/model_output/motion_deeplab_motchallenge_step_a16_fixinit`

### 6.8 MOTChallenge-STEP：评估 B4

```bash
export CUDA_VISIBLE_DEVICES=0

/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python -u deeplab2/trainer/train.py \
  --config_file=/tmp/motchallenge_motion_resnet50_os32_local.textproto \
  --mode=eval \
  --model_dir=/share_data/wenjingzhong/motchallenge_step/model_output/motion_deeplab_motchallenge_step_a16_fixinit \
  --num_gpus=1
```

如果想快速 smoke test：

```bash
export CUDA_VISIBLE_DEVICES=0

/share_data/wenjingzhong/conda_envs/step_reproduce/bin/python -u deeplab2/trainer/train.py \
  --config_file=/tmp/motchallenge_motion_resnet50_os32_eval10.textproto \
  --mode=eval \
  --model_dir=/tmp/mot_fixinit_evalfast \
  --num_gpus=1
```

## 7. 可视化结果位置

### 7.1 MOTChallenge-STEP B4

当前最完整的可视化结果在：

- `/share_data/wenjingzhong/motchallenge_step/model_output/motion_deeplab_motchallenge_step_a16_fixinit/motion_deeplab_motchallenge_step/vis`

包含：

- `*_image.png`
- `*_semantic_prediction.png`
- `*_panoptic_prediction.png`
- `*_instance_prediction.png`
- `*_center_prediction.png`
- `*_offset_prediction_rgb.png`

### 7.2 KITTI-STEP B1

最终跟踪结果：

- `/share_data/wenjingzhong/kitti_step/model_output/b1_iou_assoc_kitti_step`

### 7.3 KITTI-STEP B1 前置单帧原始输出

- `/share_data/wenjingzhong/kitti_step/model_output/panoptic_deeplab_kitti_step_a16_safe/vis_ckpt30000/raw_panoptic`
- `/share_data/wenjingzhong/kitti_step/model_output/panoptic_deeplab_kitti_step_a16_safe/vis_ckpt30000/raw_semantic`

## 8. 关键改动与已知问题

### 8.1 当前仓库相对上游的关键本地修复

#### KITTI-STEP

- `Motion-DeepLab` 默认改为普通 `BatchNorm`
- `batchnorm_epsilon = 0.001`
- 打开梯度裁剪
- 默认模型目录切到新的 `a16_safe` 路径，避免续训坏 checkpoint

#### MOTChallenge-STEP

- 构造了新的初始化 checkpoint：
  - `first layer + semantic last layer`
- 训练步数改为 `2000`
- `crop_size` 改为 `545 x 961`
- `batch_size` 改为 `1`
- 针对 `motchallenge_step + 非 SyncBN`，在 `MotionDeepLab` 的 eval 路径中改为使用 batch statistics
- 修复了 tracking 后处理中“空 center 输入”的 shape 问题

### 8.2 为什么 MOTChallenge 的第一次评估是全 0

最初那版 `MOTChallenge B4` 的主要问题不是标签错误，而是：

- `batch_size=1`
- 普通 `BatchNorm`
- eval 时使用 moving statistics

结果导致：

- `semantic prediction` 几乎塌成全背景
- `center heatmap` 全 0
- `panoptic prediction` 没有实例

修复后，这个问题已经不再是“全 0”。

### 8.3 为什么 MOTChallenge 的 STQ 仍低于论文

当前结果说明：

- `IoU / SQ` 已经接近论文
- 差距主要在 `AQ`

也就是说：

- 分割质量已经基本到位
- 跨帧关联质量仍偏弱

## 9. 当前结果总表

### 9.1 KITTI-STEP

`B1: IoU Association`

- `STQ = 0.5681`
- `AQ = 0.4724`
- `IoU / SQ = 0.6831`

论文：

- `STQ = 0.58`
- `AQ = 0.47`
- `SQ = 0.71`

`B4: Motion-DeepLab`

- `STQ = 0.5698`
- `AQ = 0.5093`
- `IoU / SQ = 0.6376`
- `PQ = 0.4354`
- `AP_Mask = 0.3990`
- `mIoU = 0.6383`

论文：

- `STQ = 0.577`
- `AQ = 0.51`
- `SQ = 0.67`

DeepLab2 文档参考值：

- `PQ = 42.08`
- `AP_Mask = 37.52`
- `mIoU = 63.15`
- `STQ = 57.7`

### 9.2 MOTChallenge-STEP

`B4: Motion-DeepLab (fixinit, latest full eval)`

- `STQ = 0.2509`
- `AQ = 0.1028`
- `IoU / SQ = 0.6123`
- `PQ = 0.5237`
- `AP_Mask = 0.0607`

论文：

- `STQ = 0.35`
- `AQ = 0.19`
- `SQ = 0.62`

`B4: Motion-DeepLab (fixinit, 10-step smoke eval)`

- `STQ = 0.3968`
- `AQ = 0.2715`
- `IoU / SQ = 0.5799`
- `PQ = 0.5210`
- `AP_Mask = 0.0675`

说明：

- smoke eval 只用于快速确认修复有效，不能替代全量结论。

## 10. 与论文写作相关的文档

根目录下可直接参考：

- `docs/experiment_report.md`
- `docs/STEP主结果与扩展实验规划.md`
- `docs/毕设任务与进度.md`

## 11. 说明

### 11.1 这个仓库不是干净上游镜像

当前工作区是“实验工作区”，不是一个只包含单项目源码的干净镜像。  
也就是说：

- 根目录里混合了 `TrackFormer` 和 `STEP`
- 配置里有本机绝对路径
- 部分训练脚本是为了当前服务器环境定制的

### 11.2 如果迁移到别的机器

需要重点检查：

- `/share_data/wenjingzhong/...` 绝对路径
- `step_reproduce/scripts/setup_env.sh`
- 数据根目录
- checkpoint 根目录
- GPU 类型是否还是适合 `TF 2.6`

### 11.3 如果上传到 GitHub

建议只上传：

- 代码
- 配置
- 脚本
- 文档

不要直接上传：

- 数据集
- `model_output`
- `checkpoints`
- `*.tfrecord`
- 大型训练产物

权重更适合放网盘或对象存储，并在 README 中附下载链接。
