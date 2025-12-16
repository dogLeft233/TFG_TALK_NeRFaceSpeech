"""批量对图像先放大到 1024x1024（双三次），再缩小到 224x224 后保存。

处理流程：
- 从输入目录读取所有图像（默认 *.png, *.jpg, *.jpeg）
- 对每张图执行：
    img_up   = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    img_down = cv2.resize(img_up, (224, 224), interpolation=cv2.INTER_AREA)
- 将 img_down 保存到输出目录，文件名不变

用法示例：
    python scripts/resize_up_1024_down_224.py \
        --input-dir NeRFFaceSpeech_Code/out_test_real \
        --output-dir NeRFFaceSpeech_Code/out_test_real_224
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="先放大到 1024x1024，再缩小到 224x224 的图像批处理脚本")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="输入图像目录（包含 png/jpg/jpeg）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出图像目录",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir 不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input-dir 不是目录: {input_dir}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(input_dir.glob(ext)))

    if not files:
        raise FileNotFoundError(f"在目录中未找到任何图像文件: {input_dir}")
    return files


def process_image(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"[警告] 无法读取图像，跳过: {src}")
        return

    # 先放大到 1024x1024（双三次插值）
    img_up = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    # 再缩小到 224x224（面积插值适合缩小）
    img_down = cv2.resize(img_up, (224, 224), interpolation=cv2.INTER_AREA)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), img_down):
        print(f"[警告] 保存失败: {dst}")
    else:
        print(f"[保存] {dst}")


def main() -> int:
    args = parse_args()

    images = list_images(args.input_dir)
    print(f"[信息] 在目录 {args.input_dir} 中找到 {len(images)} 张图像")

    for i, src in enumerate(images):
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        print(f"[处理] ({i + 1}/{len(images)}) {src.name}")
        process_image(src, dst)

    print("[完成] 所有图像处理完成")
    print(f"输出目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


