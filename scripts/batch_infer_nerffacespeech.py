"""批量运行 NeRFFaceSpeech 推理的简单脚本。

核心功能：
- 读取 binarizer 生成的 `.npy`（允许 pickle 的字典）。
- 从数据集中提取完整音频和关键图像（第一帧）。
- 调用现有 CLI `main_NeRFFaceSpeech_audio_driven_from_image.py` 生成推理结果。
- 从样本序列重建真值视频。
- 保存推理结果和对应的真值视频。

用法示例：
    python scripts/batch_infer_nerffacespeech.py \
        --dataset data/geneface_datasets/data/binary/videos/May/trainval_dataset.npy \
        --network /path/to/your_network.pkl \
        --outdir outputs/may_batch \
        --split val

说明：
- 脚本会自动从数据集目录推断音频文件路径（查找 .wav 或 .mp3）。
- 选择第一帧作为关键图像。
- 从所有样本的 `gt_img_fname` 重建真值视频。
- 为节约时间，脚本检测到目标目录下已有 `output_NeRFFaceSpeech.mp4` 会跳过。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import shutil

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
        default=None,
        help="音频文件路径（可选，如果不提供则从数据集目录自动推断）",
    )
    parser.add_argument(
        "--keyframe-idx",
        type=int,
        default=0,
        help="关键图像索引（默认0，即第一帧）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="重建真值视频的帧率（默认25）",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True).tolist()
    if not isinstance(data, dict):
        raise ValueError("数据格式异常：顶层应为 dict")
    return data


def select_samples(ds: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    key = f"{split}_samples"
    if key not in ds:
        raise KeyError(f"数据集中不存在键: {key}")
    samples = ds[key]
    # 按 idx 排序
    samples = sorted(samples, key=lambda x: x.get("idx", 0))
    return samples


def find_audio_file(dataset_path: Path) -> Optional[Path]:
    """从数据集目录推断音频文件路径"""
    # 尝试在数据集目录的父目录查找（通常结构：.../videos/VideoName/）
    dataset_dir = dataset_path.parent
    parent_dir = dataset_dir.parent
    
    # 常见音频文件名
    audio_names = ["audio.wav", "audio.mp3", "audio.m4a"]
    
    # 先检查数据集目录
    for name in audio_names:
        audio_path = dataset_dir / name
        if audio_path.exists():
            return audio_path
    
    # 检查父目录（raw/videos/VideoName/）
    for name in audio_names:
        audio_path = parent_dir / name
        if audio_path.exists():
            return audio_path
    
    # 查找任何 .wav 或 .mp3 文件
    for ext in ["*.wav", "*.mp3", "*.m4a"]:
        matches = list(dataset_dir.glob(ext))
        if matches:
            return matches[0]
        matches = list(parent_dir.glob(ext))
        if matches:
            return matches[0]
    
    return None


def build_gt_video(
    samples: List[Dict[str, Any]], 
    output_path: Path, 
    fps: int = 25
) -> None:
    """从样本序列重建真值视频"""
    if output_path.exists():
        print(f"[跳过] 真值视频已存在: {output_path}")
        return
    
    # 收集所有 gt_img_fname，按 idx 排序
    gt_frames = []
    for sample in samples:
        gt_img = sample.get("gt_img_fname")
        if gt_img and Path(gt_img).exists():
            gt_frames.append(gt_img)
        else:
            # 如果没有 gt_img_fname，尝试使用 ori_img_fname
            ori_img = sample.get("ori_img_fname")
            if ori_img and Path(ori_img).exists():
                gt_frames.append(ori_img)
    
    if len(gt_frames) == 0:
        raise ValueError("未找到任何真值图像帧")
    
    print(f"[重建] 从 {len(gt_frames)} 帧重建真值视频: {output_path}")
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(gt_frames[0])
    if first_frame is None:
        raise ValueError(f"无法读取第一帧: {gt_frames[0]}")
    
    h, w = first_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    try:
        for i, frame_path in enumerate(gt_frames):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[警告] 跳过无法读取的帧: {frame_path}")
                continue
            
            # 调整尺寸（如果需要）
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            
            out.write(frame)
            
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(gt_frames)} 帧")
    finally:
        out.release()
    
    print(f"[完成] 真值视频已保存: {output_path}")


def run_inference(
    key_img: Path,
    audio: Path,
    network: Path,
    outdir: Path,
    video_name: str = "video"
) -> Path:
    """运行单次推理，返回生成的视频路径"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    final_mp4 = outdir / "output_NeRFFaceSpeech.mp4"
    if final_mp4.exists():
        print(f"[跳过] 推理结果已存在: {final_mp4}")
        return final_mp4
    
    if not key_img.exists():
        raise FileNotFoundError(f"关键图像不存在: {key_img}")
    if not audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio}")
    
    print(f"[推理] 关键图像: {key_img}")
    print(f"[推理] 音频文件: {audio}")
    print(f"[推理] 输出目录: {outdir}")
    
    cmd = [
        "python",
        "NeRFFaceSpeech_Code/StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py",
        "--network",
        str(network),
        "--outdir",
        str(outdir),
        "--test_img",
        str(key_img),
        "--test_data",
        str(audio),
    ]
    
    subprocess.run(cmd, check=True)
    
    if not final_mp4.exists():
        raise RuntimeError(f"推理完成但未找到输出视频: {final_mp4}")
    
    print(f"[完成] 推理结果已保存: {final_mp4}")
    return final_mp4


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset 不存在: {args.dataset}")
    if not args.network.exists():
        raise FileNotFoundError(f"network 不存在: {args.network}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print(f"[加载] 数据集: {args.dataset}")
    ds = load_dataset(args.dataset)
    samples = select_samples(ds, args.split)
    print(f"[信息] 共 {len(samples)} 条样本")

    # 查找音频文件
    if args.audio is None:
        print("[查找] 自动推断音频文件路径...")
        audio_path = find_audio_file(args.dataset)
        if audio_path is None:
            raise FileNotFoundError(
                f"未找到音频文件。请手动指定 --audio 参数，"
                f"或确保数据集目录下存在 audio.wav/audio.mp3"
            )
        print(f"[找到] 音频文件: {audio_path}")
    else:
        audio_path = args.audio
        if not audio_path.exists():
            raise FileNotFoundError(f"指定的音频文件不存在: {audio_path}")

    # 选择关键图像（第一帧或指定索引）
    if args.keyframe_idx >= len(samples):
        raise ValueError(f"关键图像索引 {args.keyframe_idx} 超出范围（共 {len(samples)} 帧）")
    
    key_sample = samples[args.keyframe_idx]
    key_img = key_sample.get("head_img_fname") or key_sample.get("ori_img_fname")
    if not key_img:
        raise ValueError(f"样本 {args.keyframe_idx} 缺少 head_img_fname/ori_img_fname")
    
    key_img_path = Path(key_img)
    if not key_img_path.exists():
        raise FileNotFoundError(f"关键图像不存在: {key_img_path}")
    
    print(f"[关键] 使用第 {args.keyframe_idx} 帧作为关键图像: {key_img_path}")

    # 运行推理
    print("\n" + "="*60)
    print("[开始] 运行模型推理...")
    print("="*60)
    pred_video = run_inference(
        key_img=key_img_path,
        audio=audio_path,
        network=args.network,
        outdir=args.outdir / "prediction",
        video_name="prediction"
    )

    # 重建真值视频
    print("\n" + "="*60)
    print("[开始] 重建真值视频...")
    print("="*60)
    gt_video = args.outdir / "gt_video.mp4"
    build_gt_video(samples, gt_video, fps=args.fps)

    # 保存关键图像副本（保持原始格式）
    key_img_ext = key_img_path.suffix or ".jpg"
    key_img_copy = args.outdir / f"keyframe{key_img_ext}"
    if not key_img_copy.exists():
        shutil.copy2(key_img_path, key_img_copy)
        print(f"[保存] 关键图像副本: {key_img_copy}")

    # 保存音频文件副本
    audio_copy = args.outdir / audio_path.name
    if not audio_copy.exists():
        shutil.copy2(audio_path, audio_copy)
        print(f"[保存] 音频文件副本: {audio_copy}")

    print("\n" + "="*60)
    print("[完成] 所有任务已完成！")
    print("="*60)
    print(f"推理结果: {pred_video}")
    print(f"真值视频: {gt_video}")
    print(f"关键图像: {key_img_copy}")
    print(f"音频文件: {audio_copy}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

