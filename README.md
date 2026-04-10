# 基于 Transformer 的多目标跟踪系统

> 毕业设计项目 — 分割 + 识别 + 跟踪 完整流程

## 项目概述

本项目实现了一个完整的多目标跟踪 (Multi-Object Tracking, MOT) 系统，基于 **TrackFormer** 架构。系统由三个核心阶段组成：

| 阶段 | 功能 | 技术方案 |
|------|------|----------|
| **阶段 1** | 目标检测与实例分割 | Deformable DETR + FPN + Mask Head |
| **阶段 2** | 特征提取与身份识别 | HS Embeddings + ReID (重识别) |
| **阶段 3** | 多目标跟踪 | Track Queries + Attention-based Tracking |

### 系统架构

```
输入视频帧序列
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 1: 目标检测与实例分割          │
│  ┌─────────┐  ┌──────────────────┐  │
│  │ ResNet50 │→│ Deformable       │  │
│  │ Backbone │  │ Transformer      │  │
│  └─────────┘  │ (Encoder-Decoder)│  │
│               └────────┬─────────┘  │
│                        │            │
│         ┌──────────────┼──────────┐ │
│         ▼              ▼          ▼ │
│    检测框 BBox    分割 Mask   特征向量│
└─────────┬──────────────┬──────────┬─┘
          │              │          │
          ▼              ▼          ▼
┌─────────────────────────────────────┐
│  阶段 2: 身份识别 (ReID)             │
│  - HS Embedding 特征匹配             │
│  - 匈牙利算法/贪心匹配               │
│  - 不活跃轨迹重识别                   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  阶段 3: 多目标跟踪                  │
│  - Track Queries (身份保持)          │
│  - Object Queries (新目标初始化)      │
│  - Self/Cross Attention (全局关联)    │
│  - NMS + Score Filtering             │
└─────────────────┬───────────────────┘
                  │
                  ▼
          跟踪结果输出
    (轨迹 ID + BBox + Mask + Score)
```

## 运行实验与性能指标（重要）

1. **先看进度说明**：[docs/毕设任务与进度.md](docs/毕设任务与进度.md) — 毕设要完成什么、当前做到哪一步、为何暂时没有本机 MOTA 数字。  
2. **环境自检**：`bash scripts/check_env.sh`（含 PyTorch、MSDA 扩展、MOT17、权重是否存在）。  
3. **权重下载**：`bash scripts/download_models.sh`（需能访问 `vision.in.tum.de`；集群不通时请按脚本内说明用本机下载后 `scp` 上传）。  
4. **跑全流程 + 生成报告**：`bash scripts/run_exp.sh`  
   - 日志：`logs/run_exp.log`（**终端也会同步打印**）  
   - 若失败：`logs/run_exp.last_error.txt`（含最后 120 行日志）  
5. **MSDA 扩展编译**（换机器或重装环境后）：`bash scripts/build_msda.sh`  

指标在 `track.py` 运行结束时的 **EVAL / OVERALL / MOTA / IDF1** 等行；`run_exp.sh` 会尝试写入 `docs/experiment_report.md`。

## 目录结构

```
graduation_project/
├── README.md                    # 本文件
├── activate.sh                  # 一键激活环境
├── .condarc                     # Conda 配置 (环境存储在 /share_data)
│
├── code/
│   └── trackformer/             # TrackFormer 源码 (git clone)
│       ├── src/
│       │   ├── track.py         # 跟踪主入口
│       │   ├── train.py         # 训练主入口
│       │   └── trackformer/     # 核心模块
│       │       ├── models/
│       │       │   ├── deformable_detr.py      # Deformable DETR 检测器
│       │       │   ├── detr_segmentation.py    # 分割 Head
│       │       │   ├── detr_tracking.py        # 跟踪扩展
│       │       │   └── tracker.py              # Tracker + ReID 核心
│       │       └── datasets/
│       │           └── tracking/               # MOT/MOTS 数据加载
│       └── cfgs/                # 配置文件
│
├── data/                        # 数据集
│   ├── MOT17/                   # MOT17 数据集
│   └── MOTS/                    # MOTS20 数据集 (含分割标注)
│
├── models/                      # 预训练模型权重
│   ├── mot17_crowdhuman_deformable_multi_frame/
│   └── mots20_train_masks/
│
├── outputs/                     # 运行输出 (跟踪结果、可视化)
├── logs/                        # 训练日志
├── docs/                        # 额外文档
│
├── scripts/
│   ├── setup_env.sh             # 环境安装脚本
│   ├── download_data_models.sh  # 数据/模型下载脚本
│   ├── run_pipeline.sh          # Shell 运行脚本
│   ├── pipeline.py              # Python Pipeline 入口
│   └── export_env.sh            # 环境导出脚本
│
├── conda_envs/
│   └── trackformer_grad/        # Conda 环境 (不占家目录配额)
└── conda_pkgs/                  # Conda 包缓存
```

## 快速开始

### 1. 安装环境

```bash
# 首次安装 (约 10-15 分钟)
bash /share_data/wenjingzhong/graduation_project/scripts/setup_env.sh
```

### 2. 下载数据和模型

```bash
bash /share_data/wenjingzhong/graduation_project/scripts/download_data_models.sh
```

### 3. 激活环境

```bash
source /share_data/wenjingzhong/graduation_project/activate.sh
```

### 4. 运行

```bash
# 方式一: Shell 脚本
bash scripts/run_pipeline.sh track_reid              # 跟踪 + ReID
bash scripts/run_pipeline.sh segment --visualize      # 分割 + 可视化
bash scripts/run_pipeline.sh full --visualize          # 完整流程

# 方式二: Python 脚本
python scripts/pipeline.py --mode track_reid --dataset MOT17-TRAIN-ALL
python scripts/pipeline.py --mode full --visualize

# 方式三: 直接调用 TrackFormer
cd code/trackformer
python src/track.py with reid
```

## 运行模式

| 模式 | 说明 | 对应毕设章节 |
|------|------|-------------|
| `detect` | 仅目标检测 | 第三章 目标检测模块 |
| `segment` | 实例分割 (MOTS20) | 第三章 分割模块 |
| `track` | 多目标跟踪 | 第四章 跟踪模块 |
| `track_reid` | 跟踪 + ReID 识别 | 第四章 识别+跟踪 |
| `full` | 完整流程 | 第五章 系统集成 |
| `demo` | 自定义视频 | 演示 |

## 评估指标

系统使用标准 MOT 评估指标:

- **MOTA** (Multiple Object Tracking Accuracy): 综合准确率
- **IDF1** (ID F1 Score): 身份识别准确率
- **MT** (Mostly Tracked): 大部分帧被跟踪的目标数
- **ML** (Mostly Lost): 大部分帧丢失的目标数
- **FP** (False Positives): 误检数
- **FN** (False Negatives): 漏检数
- **ID Sw.** (ID Switches): 身份切换次数

## 训练 (可选)

如需在自定义数据集上微调:

```bash
source activate.sh
cd code/trackformer

# MOT17 微调
python src/train.py with \
    mot17_crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=../../models/my_custom_model
```

## 环境导出

```bash
bash scripts/export_env.sh
# 生成 environment.yml 和 pip_requirements.txt
```

## 关键技术说明

### Deformable DETR (检测+分割)
- 多尺度特征: 使用 FPN 从 backbone 提取多层级特征
- 可变形注意力: 替代标准注意力，仅关注稀疏采样点，计算效率更高
- Mask Head: 在检测基础上添加分割分支，输出像素级 mask

### Track Queries (跟踪)
- Object Queries: Transformer Decoder 的标准查询，用于检测新目标
- Track Queries: 从上一帧继承的查询，携带目标身份信息
- 通过 self-attention 和 encoder-decoder attention 实现帧间关联

### ReID (重识别)
- HS Embeddings: Decoder 输出的隐藏状态向量作为目标的外观特征
- 匹配策略: 使用欧氏距离 + 匈牙利算法进行最优匹配
- Inactive Patience: 允许短暂消失的目标在重新出现时被重新识别

## 参考文献

```
@InProceedings{meinhardt2021trackformer,
    title={TrackFormer: Multi-Object Tracking with Transformers},
    author={Tim Meinhardt and Alexander Kirillov and Laura Leal-Taixe and Christoph Feichtenhofer},
    year={2022},
    booktitle={CVPR},
}
```

## 硬件要求

- GPU: NVIDIA H100 80GB (当前服务器配置)
- CUDA: 12.8
- 磁盘: 数据+模型约需 10-20 GB
