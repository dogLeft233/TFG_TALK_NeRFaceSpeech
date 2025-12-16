"""对 raw 视频目录中的每个 mp4 文件：

1. 提取完整音频为 wav。
2. 随机抽取一帧图像作为关键帧。
3. 调用 `main_NeRFFaceSpeech_audio_driven_from_image.py` 进行模型推理。
4. 将推理结果保存在独立子目录中，方便后续评测。

典型目录结构：
    data/geneface_datasets/data/raw/videos/
        Macron.mp4
        May.mp4
        ...

用法示例：
    python scripts/video_batch_infer_from_raw.py \
        --video-dir data/geneface_datasets/data/raw/videos \
        --network /path/to/your_network.pkl \
        --outdir outputs/raw_video_infer

说明：
- 每个 mp4 会在 `--outdir` 下生成一个同名子目录，例如 `outputs/raw_video_infer/May/`。
- 子目录内会包含：
    - `audio.wav`          : 从 mp4 提取的音频
    - `keyframe.png`       : 随机抽取的一帧图像
    - `output_NeRFFaceSpeech.mp4` : 模型推理生成的视频
- 如果某个视频对应的输出目录下已经存在 `output_NeRFFaceSpeech.mp4`，则会跳过该视频。
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import cv2

# 项目根目录 = 当前脚本所在目录的上级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NeRFFaceSpeech 代码根目录
NERF_CODE_DIR = PROJECT_ROOT / "NeRFFaceSpeech_Code"
# NeRFFaceSpeech 专用环境的 python 可执行文件
NERF_ENV_PYTHON = PROJECT_ROOT / "environment" / "nerffacespeech" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 raw mp4 批量生成 NeRFFaceSpeech 推理结果")
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="包含多个 mp4 文件的目录（例如 data/geneface_datasets/data/raw/videos）",
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
        help="批量输出根目录，每个视频一个子目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子，用于选择关键帧（默认0）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定，则即使已有 output_NeRFFaceSpeech.mp4 也会重新生成",
    )
    return parser.parse_args()


def list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"video-dir 不存在: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video-dir 不是目录: {video_dir}")

    videos: list[Path] = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        videos.extend(video_dir.glob(ext))
    videos = sorted(videos)
    if not videos:
        raise FileNotFoundError(f"在目录中未找到任何视频文件: {video_dir}")
    return videos


def extract_audio(video_path: Path, out_wav: Path) -> None:
    """使用 ffmpeg 从视频中提取音频到 wav 文件。

    为了与大部分语音模型兼容，这里统一转为 16kHz 单声道 PCM。
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # -y 覆盖输出；-vn 去掉视频；-ar 16000 采样率；-ac 1 单声道；pcm_s16le 无压缩 PCM
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_wav),
    ]

    print(f"[音频] {video_path.name} -> {out_wav.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg 提取音频失败: {video_path}") from exc


def sample_random_frame(video_path: Path, out_image: Path, rng: random.Random) -> None:
    """从视频中随机抽取一帧并保存为图像。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数异常: {video_path}")

    # 随机选一个帧索引 [0, frame_count-1]
    idx = rng.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"无法读取随机帧 idx={idx} 来自视频: {video_path}")

    out_image.parent.mkdir(parents=True, exist_ok=True)
    # 保存为 PNG，避免压缩损失
    if not cv2.imwrite(str(out_image), frame):
        raise RuntimeError(f"保存关键帧失败: {out_image}")

    print(f"[关键帧] {video_path.name} -> {out_image.name} (idx={idx}, 总帧数={frame_count})")


def run_inference(network: Path, outdir: Path, keyframe: Path, audio_wav: Path) -> Path:
    """调用现有 CLI 进行推理，返回生成的视频路径。"""
    outdir.mkdir(parents=True, exist_ok=True)
    pred_mp4 = outdir / "output_NeRFFaceSpeech.mp4"

    # 如果存在 nerffacespeech 环境，则优先使用该环境的 python
    python_exe = NERF_ENV_PYTHON if NERF_ENV_PYTHON.exists() else "python"

    # 在 NeRFFaceSpeech 代码根目录下运行，从而让脚本中的相对路径（如 pretrained_networks/seg.pth）生效
    cmd = [
        str(python_exe),
        "StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py",
        "--network",
        str(network),
        "--outdir",
        str(outdir),
        "--test_img",
        str(keyframe),
        "--test_data",
        str(audio_wav),
    ]

    print(f"[推理] 输出目录: {outdir}")
    subprocess.run(cmd, check=True, cwd=str(NERF_CODE_DIR))

    if not pred_mp4.exists():
        raise RuntimeError(f"推理完成但未找到输出视频: {pred_mp4}")

    print(f"[完成] 推理结果: {pred_mp4}")
    return pred_mp4


def process_one_video(
    video_path: Path,
    network: Path,
    out_root: Path,
    rng: random.Random,
    overwrite: bool = False,
) -> None:
    name = video_path.stem  # e.g. May, Macron
    video_outdir = out_root / name
    video_outdir.mkdir(parents=True, exist_ok=True)

    pred_mp4 = video_outdir / "output_NeRFFaceSpeech.mp4"
    if pred_mp4.exists() and not overwrite:
        print(f"[跳过] 已存在推理结果: {pred_mp4}")
        return

    # 1) 提取音频
    audio_wav = video_outdir / "audio.wav"
    extract_audio(video_path, audio_wav)

    # 2) 抽取随机关键帧
    keyframe = video_outdir / "keyframe.png"
    sample_random_frame(video_path, keyframe, rng)

    # 3) 运行推理
    run_inference(network=network, outdir=video_outdir, keyframe=keyframe, audio_wav=audio_wav)


def main() -> int:
    args = parse_args()

    if not args.network.exists():
        raise FileNotFoundError(f"network 不存在: {args.network}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.video_dir)
    print(f"[信息] 在目录 {args.video_dir} 中找到 {len(videos)} 个视频文件")

    rng = random.Random(args.seed)

    for i, v in enumerate(videos):
        print("\n" + "=" * 60)
        print(f"[处理] ({i + 1}/{len(videos)}) {v.name}")
        print("=" * 60)
        try:
            process_one_video(
                video_path=v,
                network=args.network,
                out_root=args.outdir,
                rng=rng,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 处理视频 {v} 时出错: {exc}")

    print("\n" + "=" * 60)
    print("[完成] 全部视频处理结束")
    print("=" * 60)
    print(f"输出目录: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


