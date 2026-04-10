"""
毕业设计: 基于 Transformer 的多目标跟踪系统 — 完整 Pipeline

流程概述:
    阶段 1 — 目标检测与实例分割 (Deformable DETR + Mask Head)
    阶段 2 — 特征提取与身份识别 (ReID via HS Embeddings)
    阶段 3 — 多目标跟踪        (TrackFormer Attention-based Tracking)

使用方法:
    python pipeline.py --mode full --data_root /path/to/data --visualize
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKFORMER_ROOT = PROJECT_ROOT / "code" / "trackformer"
sys.path.insert(0, str(TRACKFORMER_ROOT / "src"))

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import evaluate_mot_accums, get_mot_accum


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载检测/分割模型和后处理器"""
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    args = nested_dict_to_namespace(yaml.unsafe_load(open(config_path)))
    img_transform = args.img_transform

    model, _, postprocessors = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        k.replace("detr.", ""): v
        for k, v in checkpoint["model"].items()
        if "track_encoding" not in k
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if hasattr(model, "tracking"):
        model.tracking()

    epoch = checkpoint.get("epoch", "unknown")
    print(f"  模型加载完成 [epoch: {epoch}]")

    return model, postprocessors, img_transform


def run_detection(model, postprocessors, dataset, device="cuda"):
    """阶段1: 目标检测 — 输出每帧的检测框和置信度"""
    print("\n" + "=" * 60)
    print("  阶段 1: 目标检测 (Deformable DETR)")
    print("=" * 60)

    all_detections = {}
    for seq in dataset:
        seq_name = str(seq)
        print(f"\n  处理序列: {seq_name}")
        seq_dets = {}

        loader = torch.utils.data.DataLoader(seq)
        for frame_id, frame_data in enumerate(loader):
            img = frame_data["img"].to(device)
            orig_size = frame_data["orig_size"].to(device)

            with torch.no_grad():
                outputs, _, features, _, _ = model(img)

            results = postprocessors["bbox"](outputs, orig_size)[0]
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()

            keep = (scores > 0.5) & (labels == 0)
            seq_dets[frame_id] = {
                "boxes": boxes[keep].tolist(),
                "scores": scores[keep].tolist(),
                "n_objects": int(keep.sum()),
            }

            if frame_id % 50 == 0:
                print(f"    帧 {frame_id}: 检测到 {keep.sum()} 个目标")

        all_detections[seq_name] = seq_dets
    return all_detections


def run_tracking_with_reid(
    model, postprocessors, img_transform, dataset_name, data_root,
    tracker_cfg, device="cuda"
):
    """阶段2+3: ReID 识别 + 注意力跟踪 (TrackFormer 核心)"""
    print("\n" + "=" * 60)
    print("  阶段 2: 特征提取与 ReID 身份识别")
    print("  阶段 3: 基于注意力的多目标跟踪")
    print("=" * 60)

    tracker = Tracker(model, postprocessors, tracker_cfg, False)
    dataset = TrackDatasetFactory(dataset_name, root_dir=data_root, img_transform=img_transform)

    all_results = {}
    mot_accums = []
    total_time = 0
    total_frames = 0

    for seq in dataset:
        tracker.reset()
        seq_name = str(seq)
        print(f"\n  跟踪序列: {seq_name}")

        loader = torch.utils.data.DataLoader(seq)
        total_frames += len(loader)

        start = time.time()
        for frame_id, frame_data in enumerate(loader):
            with torch.no_grad():
                tracker.step(frame_data)
        elapsed = time.time() - start
        total_time += elapsed

        results = tracker.get_results()
        all_results[seq_name] = results

        print(f"    轨迹数: {len(results)}")
        print(f"    ReID 次数: {tracker.num_reids}")
        print(f"    运行时间: {elapsed:.2f}s ({len(loader)/elapsed:.1f} FPS)")

        if not seq.no_gt:
            mot_accum = get_mot_accum(results, loader)
            mot_accums.append(mot_accum)

    if mot_accums:
        print("\n" + "=" * 60)
        print("  评估结果")
        print("=" * 60)
        summary, str_summary = evaluate_mot_accums(
            mot_accums,
            [str(s) for s in dataset if not s.no_gt],
        )
        print(str_summary)

    if total_time > 0:
        print(f"\n  总帧数: {total_frames}, 总时间: {total_time:.2f}s, "
              f"平均: {total_frames/total_time:.1f} FPS")

    return all_results, mot_accums


def visualize_results(results, seq_loader, output_dir, seq_name):
    """可视化跟踪结果"""
    from trackformer.util.track_utils import plot_sequence
    vis_dir = os.path.join(output_dir, seq_name)
    os.makedirs(vis_dir, exist_ok=True)
    plot_sequence(results, seq_loader, vis_dir, "pretty", False)
    print(f"  可视化结果保存至: {vis_dir}")


def save_results(results, output_dir, filename="tracking_results.json"):
    """保存跟踪结果为 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    serializable = {}
    for seq_name, seq_results in results.items():
        serializable[seq_name] = {}
        for track_id, track_data in seq_results.items():
            serializable[seq_name][str(track_id)] = {}
            for frame_id, frame_data in track_data.items():
                serializable[seq_name][str(track_id)][str(frame_id)] = {
                    "bbox": frame_data["bbox"].tolist()
                    if hasattr(frame_data["bbox"], "tolist")
                    else frame_data["bbox"],
                    "score": float(frame_data["score"]),
                }

    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  结果保存至: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="基于 Transformer 的多目标跟踪系统 Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["detect", "track", "track_reid", "segment", "full"],
        default="track_reid",
        help="运行模式",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="模型权重路径 (默认使用预训练权重)",
    )
    parser.add_argument(
        "--dataset",
        default="MOT17-TRAIN-ALL",
        help="数据集名称 (参见 factory.py)",
    )
    parser.add_argument(
        "--data_root",
        default=str(PROJECT_ROOT / "data"),
        help="数据根目录",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs"),
        help="输出目录",
    )
    parser.add_argument("--visualize", action="store_true", help="生成可视化")
    parser.add_argument("--device", default="cuda", help="计算设备")
    args = parser.parse_args()

    print("=" * 60)
    print("  基于 Transformer 的多目标跟踪系统")
    print("  (Deformable DETR + ReID + TrackFormer)")
    print("=" * 60)
    print(f"  模式:     {args.mode}")
    print(f"  数据集:   {args.dataset}")
    print(f"  数据目录: {args.data_root}")
    print(f"  设备:     {args.device}")

    default_checkpoints = {
        "detect": "models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth",
        "track": "models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth",
        "track_reid": "models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth",
        "segment": "models/mots20_train_masks/checkpoint.pth",
        "full": "models/mot17_crowdhuman_deformable_multi_frame/checkpoint_epoch_40.pth",
    }

    ckpt = args.checkpoint or os.path.join(
        str(TRACKFORMER_ROOT), default_checkpoints[args.mode]
    )
    print(f"  权重:     {ckpt}")

    print("\n加载模型...")
    model, postprocessors, img_transform = load_model(ckpt, args.device)

    tracker_cfg = {
        "detection_obj_score_thresh": 0.4,
        "track_obj_score_thresh": 0.4,
        "detection_nms_thresh": 0.9,
        "track_nms_thresh": 0.9,
        "steps_termination": 1,
        "prev_frame_dist": 1,
        "inactive_patience": 5 if "reid" in args.mode or args.mode == "full" else -1,
        "reid_sim_threshold": 0.0,
        "reid_sim_only": False,
        "reid_score_thresh": 0.4,
        "reid_greedy_matching": False,
        "public_detections": False,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output = os.path.join(args.output, f"{args.mode}_{timestamp}")
    os.makedirs(run_output, exist_ok=True)

    if args.mode == "detect":
        dataset = TrackDatasetFactory(
            args.dataset, root_dir=args.data_root, img_transform=img_transform
        )
        detections = run_detection(model, postprocessors, dataset, args.device)
        with open(os.path.join(run_output, "detections.json"), "w") as f:
            json.dump(detections, f, indent=2)
        print(f"\n检测结果保存至: {run_output}/detections.json")

    elif args.mode in ("track", "track_reid", "segment", "full"):
        results, accums = run_tracking_with_reid(
            model, postprocessors, img_transform,
            args.dataset, args.data_root,
            tracker_cfg, args.device,
        )
        save_results(results, run_output)

    print("\n" + "=" * 60)
    print(f"  Pipeline 运行完成!")
    print(f"  输出目录: {run_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
