# STEP 主结果与扩展实验规划

## 1. 最终目标与阶段策略

本项目的最终要求是尽可能复现整篇 `STEP: Segmenting and Tracking Every Pixel` 论文，而不是只做单一模型。

最终目标定义为：

- 数据集：`KITTI-STEP` 与 `MOTChallenge-STEP`
- baseline：`B1 IoU Assoc.`、`B2 SORT`、`B3 Mask Propagation`、`B4 Motion-DeepLab`
- 主指标：`STQ`
- 同时报告：`AQ`、`SQ`、`VPQ`、`PTQ`、`sPTQ`、`IDS`、`sIDS`、`sMOTSA`、`MOTSP`

但执行上按阶段推进：

1. 先保住 `KITTI-STEP + B4 Motion-DeepLab` 训练和评估闭环。
2. 再补 `KITTI-STEP` 上的 `B1/B2/B3`。
3. 最后补 `MOTChallenge-STEP` 上对应实验。

这样做的目的不是缩小最终范围，而是优先保证先有一条稳定、可交付、可答辩的主结果链路。

## 2. 当前主结果目标

当前阶段的主结果只要求先回答三个问题：

1. `Motion-DeepLab` 是否在本地环境稳定训练。
2. 是否能在 `KITTI-STEP` 上得到有效的 `STQ / AQ / SQ / PQ / AP Mask / mIoU`。
3. 结果与原论文 `KITTI-STEP B4` 的差距是多少，差距来自哪里。

原论文参考值：

- `PQ = 42.08`
- `AP Mask = 37.52`
- `mIoU = 63.15`
- `STQ = 57.7`

## 3. 分阶段实验规划

分阶段实验既服务当前主结果，也逐步铺到整篇论文复现。

### 扩展实验 A：训练稳定性对比

- 配置 1：`SyncBatchNorm + 多卡`
- 配置 2：`Standard BatchNorm + 单卡/保守配置`
- 目标：说明本地 A16 环境下，哪种配置更稳定，是否出现 `NaN`。

### 扩展实验 B：不同训练阶段结果对比

- checkpoint：例如 `step 500 / 1000 / 2000 / best`
- 指标：`STQ / AQ / SQ / PQ / AP Mask / mIoU`
- 目标：说明模型是否真的在收敛，而不是只“能跑”。

### 扩展实验 C：定性可视化

- 输出内容：`raw_panoptic`、彩色可视化结果、若干序列截图
- 目标：展示像素级分割和时序跟踪是否合理。

### 扩展实验 D：KITTI-STEP 上的 B1 / B2 / B3 baseline

- `B1 IoU Assoc.`
- `B2 SORT`
- `B3 Mask Propagation`
- 目标：把论文表 2 的 baseline 尽量补齐，形成与你当前 `B4` 主结果可对照的一组本地复现实验。

### 扩展实验 E：MOTChallenge-STEP 补充实验

- 先做 `B4 Motion-DeepLab`
- 后续再视时间补 `B1/B2/B3`
- 目标：逐步向论文表 3 靠拢。

## 4. 当前实际进度

截至 `2026-03-27`，已确认：

- `KITTI-STEP` 图像、标注、TFRecord、预训练 checkpoint 已经就位。
- `H100 + TF 2.6` 的 `NaN` 问题已确认是环境/硬件兼容性问题，不再继续作为主训练环境。
- `A16 + TF 2.6` 上最小梯度测试正常。
- 已将 `Motion-DeepLab` 默认训练配置切到更保守路径：
  - `use_sync_batchnorm: false`
  - `batchnorm_epsilon: 0.001`
  - fresh model dir，避免续训旧坏 checkpoint
  - 开启 `TF_FORCE_GPU_ALLOW_GROWTH`

当前正在进行的主训练：

- tmux 会话：`step_motion_a16_safe`
- 日志：`/share_data/wenjingzhong/kitti_step/model_output/train_logs/motion_deeplab_ampere_latest.log`
- 模型目录：`/share_data/wenjingzhong/kitti_step/model_output/motion_deeplab_kitti_step_a16_safe`

最新已观察到的训练状态：

- `step 100`：loss 全部有限
- `step 200`：loss 全部有限
- `step 300`：loss 全部有限
- `step 400`：loss 全部有限
- 尚未出现 `NaN`

当前 baseline 支持情况：

- `B4 Motion-DeepLab`：训练链路已跑通，正在持续训练。
- `B1 IoU Assoc.`：仓库内已有 `iou_tracker.py`，存在实现基础。
- `B2 SORT`：仓库内未见现成可直接运行的 STEP 版训练/评估脚本，后续需补。
- `B3 Mask Propagation`：仓库内未见现成 RAFT + STEP 的整合脚本，后续需补。

## 5. 最近必须完成的事项

### 第一优先级

- 盯到训练至少稳定通过 `500` 和 `1000` step。
- 保存并整理第一版有效 loss 曲线与 checkpoint。

### 第二优先级

- 用当前 checkpoint 跑 `eval_model.sh motion`
- 跑 `eval_stq.sh <predictions_dir>`
- 拿到第一版真实 `STQ/AQ/IoU` 和 `PQ/AP Mask/mIoU`

### 第三优先级

- 打开评估可视化输出。
- 选取 2 到 3 个序列做截图和 qualitative analysis。

### 第四优先级

- 梳理并补齐 `KITTI-STEP` 上的 `B1/B2/B3` 实验链路。
- 明确每个 baseline 的输入、依赖、评估脚本和输出格式。

### 第五优先级

- 准备 `MOTChallenge-STEP` 数据与配置。
- 至少补做 `B4 Motion-DeepLab` 一次训练/评估实验。

## 6. 当前暂缓项

- `TrackFormer/MOT17` 与 `STEP` 混合叙事
- 与 STEP 主线无关的额外工程工作

`B1/B2/B3` 和 `MOTChallenge-STEP` 不再被视为放弃项，而是被明确放到后续阶段中补齐。

## 7. 进度查看命令

```bash
tmux attach -t step_motion_a16_safe
tail -f /share_data/wenjingzhong/kitti_step/model_output/train_logs/motion_deeplab_ampere_latest.log
```

## 8. 最终交付物

本轮优先交付以下内容形成第一闭环：

1. `KITTI-STEP + B4 Motion-DeepLab` 训练稳定日志
2. 一版真实评估表：`STQ / AQ / SQ / PQ / AP Mask / mIoU`
3. 一组可视化结果图
4. 一段与论文结果的差距分析

在此基础上，继续补：

5. `KITTI-STEP` 上的 `B1/B2/B3`
6. `MOTChallenge-STEP` 上的补充实验

这样既满足“整篇论文复现”的最终要求，也符合当前工程推进顺序。
