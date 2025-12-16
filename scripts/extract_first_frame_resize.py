"""从 mp4 提取第一帧，先放大到 1024x1024（双三次），再缩小到 224x224 保存。

用法示例：
    python scripts/extract_first_frame_resize.py \
        --video data/geneface_datasets/data/raw/videos/May.mp4 \
        --output output/May_firstframe_224.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 mp4 第一帧并做 1024→224 的缩放")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="输入 mp4 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出图像路径（例如 output/first_frame_224.png）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"video 不存在: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"无法读取第一帧: {args.video}")

    # 先放大到 1024x1024（双三次插值）
    img_up = cv2.resize(frame, (1024, 1024),interpolation=cv2.INTER_CUBIC)
    # 再缩小到 224x224（面积插值适合缩小）
    img_down = cv2.resize(img_up, (224, 224))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img_down):
        raise RuntimeError(f"保存失败: {args.output}")

    print(f"[完成] 第一帧已处理并保存到: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


