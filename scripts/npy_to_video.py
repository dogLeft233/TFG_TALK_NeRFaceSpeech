"""将 .npy 数据集文件转换为 MP4 视频的简单脚本。

核心功能：
- 读取 binarizer 生成的 `.npy`（允许 pickle 的字典）。
- 从 train/val 样本中提取图像帧序列。
- 将图像序列合成为 MP4 视频。

用法示例：
    python scripts/npy_to_video.py \
        --dataset data/geneface_datasets/data/binary/videos/May/trainval_dataset.npy \
        --output outputs/may_dataset.mp4 \
        --split val \
        --fps 25

说明：
- 默认使用 `gt_img_fname`，如果没有则使用 `ori_img_fname`。
- 图像按 `idx` 排序。
- 支持指定帧率、图像字段等参数。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 .npy 数据集转换为 MP4 视频")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="binarizer 生成的 .npy 字典文件（allow_pickle=True）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出视频路径（.mp4）",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="选择使用 train_samples 或 val_samples",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="输出视频帧率（默认25）",
    )
    parser.add_argument(
        "--image-field",
        type=str,
        default="auto",
        choices=["auto", "gt_img_fname", "ori_img_fname", "head_img_fname"],
        help="使用的图像字段（auto=自动选择：优先gt_img_fname，其次ori_img_fname）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大帧数（默认全部）",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="调整视频尺寸（例如：--resize 512 512）",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Dict[str, Any]:
    """加载 .npy 数据集文件"""
    data = np.load(path, allow_pickle=True).tolist()
    if not isinstance(data, dict):
        raise ValueError("数据格式异常：顶层应为 dict")
    return data


def select_samples(ds: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """选择并排序样本"""
    key = f"{split}_samples"
    if key not in ds:
        raise KeyError(f"数据集中不存在键: {key}")
    samples = ds[key]
    # 按 idx 排序
    samples = sorted(samples, key=lambda x: x.get("idx", 0))
    return samples


def get_image_paths(
    samples: List[Dict[str, Any]], 
    image_field: str = "auto"
) -> List[Path]:
    """从样本中提取图像路径列表"""
    image_paths = []
    
    for sample in samples:
        img_path = None
        
        if image_field == "auto":
            # 自动选择：优先 gt_img_fname，其次 ori_img_fname
            img_path = sample.get("gt_img_fname") or sample.get("ori_img_fname")
        else:
            img_path = sample.get(image_field)
        
        if img_path:
            img_path = Path(img_path)
            if img_path.exists():
                image_paths.append(img_path)
            else:
                print(f"[警告] 图像不存在，跳过: {img_path}")
        else:
            print(f"[警告] 样本 {sample.get('idx', 'unknown')} 缺少图像字段")
    
    return image_paths


def images_to_video(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 25,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> None:
    """将图像序列合成为视频"""
    if len(image_paths) == 0:
        raise ValueError("未找到任何图像文件")
    
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    
    print(f"[处理] 共 {len(image_paths)} 帧图像")
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"无法读取第一帧: {image_paths[0]}")
    
    h, w = first_frame.shape[:2]
    
    # 如果指定了 resize，使用新尺寸
    if resize is not None:
        w, h = resize
        print(f"[调整] 视频尺寸: {w}x{h}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    if not out.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")
    
    try:
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[警告] 跳过无法读取的帧: {img_path}")
                continue
            
            # 调整尺寸（如果需要）
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            
            out.write(frame)
            
            # 进度显示
            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                print(f"  已处理 {i + 1}/{len(image_paths)} 帧 ({100*(i+1)/len(image_paths):.1f}%)")
    finally:
        out.release()
    
    print(f"[完成] 视频已保存: {output_path}")
    print(f"      帧数: {len(image_paths)}, 帧率: {fps} fps, 尺寸: {w}x{h}")


def main() -> int:
    args = parse_args()
    
    if not args.dataset.exists():
        raise FileNotFoundError(f"数据集文件不存在: {args.dataset}")
    
    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print(f"[加载] 数据集: {args.dataset}")
    ds = load_dataset(args.dataset)
    
    # 选择样本
    samples = select_samples(ds, args.split)
    print(f"[信息] 共 {len(samples)} 条样本")
    
    # 提取图像路径
    print(f"[提取] 使用图像字段: {args.image_field}")
    image_paths = get_image_paths(samples, args.image_field)
    
    if len(image_paths) == 0:
        raise ValueError("未找到任何有效的图像文件")
    
    print(f"[找到] 共 {len(image_paths)} 个有效图像文件")
    
    # 转换为视频
    print("\n" + "="*60)
    print("[开始] 合成视频...")
    print("="*60)
    images_to_video(
        image_paths=image_paths,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        resize=tuple(args.resize) if args.resize else None,
    )
    
    print("\n" + "="*60)
    print("[完成] 视频转换完成！")
    print("="*60)
    print(f"输出文件: {args.output}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

