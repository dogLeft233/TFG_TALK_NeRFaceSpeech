"""批量运行 NeRFFaceSpeech 推理的简单脚本。

核心功能：
- 读取 binarizer 生成的 `.npy`（允许 pickle 的字典）。
- 选取 train/val 样本的 `head_img_fname` 作为输入人脸。
- 调用现有 CLI `main_NeRFFaceSpeech_audio_driven_from_image.py` 逐样本生成结果。

用法示例：
    python scripts/batch_infer_nerffacespeech.py \
        --dataset data/geneface_datasets/data/binary/videos/May/trainval_dataset.npy \
        --network /path/to/your_network.pkl \
        --outdir outputs/may_batch \
        --split val \
        --audio data/geneface_datasets/data/raw/videos/May/audio.wav \
        --limit 10

说明：
- `--audio` 会作为 `--test_data` 传给原脚本（当前管线依赖 SadTalker 的 mel 提取逻辑）。
- 为节约时间，脚本检测到目标子目录下已有 `output_NeRFFaceSpeech.mp4` 会跳过。
- 每个样本输出目录形如：`{outdir}/sample_{idx:05d}_{sample_idx}`。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 NeRFFaceSpeech 推理结果")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="binarizer 生成的 .npy 字典文件（allow_pickle=True）",
    )
    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="训练好的生成器 pkl（传给 --network）",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="批量输出根目录",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="选择使用 train_samples 或 val_samples",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="音频文件路径，作为 --test_data 传入（当前管线使用音频路径生成 mel）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的样本数（默认全量）",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True).tolist()
    if not isinstance(data, dict):
        raise ValueError("数据格式异常：顶层应为 dict")
    return data


def select_samples(ds: Dict[str, Any], split: str, limit: int | None) -> List[Dict[str, Any]]:
    key = f"{split}_samples"
    if key not in ds:
        raise KeyError(f"数据集中不存在键: {key}")
    samples = ds[key]
    if limit is not None:
        samples = samples[:limit]
    return samples


def run_one(sample: Dict[str, Any], network: Path, audio: Path, outdir: Path, sample_idx: int) -> None:
    head_img = sample.get("head_img_fname") or sample.get("ori_img_fname")
    if not head_img:
        raise ValueError(f"样本 {sample_idx} 缺少 head_img_fname/ori_img_fname")

    sample_out = outdir / f"sample_{sample_idx:05d}_{sample.get('idx', sample_idx)}"
    sample_out.mkdir(parents=True, exist_ok=True)

    final_mp4 = sample_out / "output_NeRFFaceSpeech.mp4"
    if final_mp4.exists():
        print(f"[跳过] 已存在: {final_mp4}")
        return

    cmd = [
        "python",
        "NeRFFaceSpeech_Code/StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py",
        "--network",
        str(network),
        "--outdir",
        str(sample_out),
        "--test_img",
        str(head_img),
        "--test_data",
        str(audio),
    ]

    print(f"[运行] 样本 {sample_idx} -> {sample_out}")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset 不存在: {args.dataset}")
    if not args.network.exists():
        raise FileNotFoundError(f"network 不存在: {args.network}")
    if not args.audio.exists():
        raise FileNotFoundError(f"audio 不存在: {args.audio}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    samples = select_samples(ds, args.split, args.limit)
    print(f"共 {len(samples)} 条样本，将逐条运行推理...")

    for i, sample in enumerate(samples):
        run_one(sample, args.network, args.audio, args.outdir, i)

    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

