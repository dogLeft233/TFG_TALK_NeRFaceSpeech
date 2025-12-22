#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型预下载脚本（使用系统默认缓存目录）

模型分组：
- fan  : face-alignment (FAN)
- core : torchvision / HuggingFace / Whisper
- all  : 下载全部

示例：
  python download_models.py --group fan
  python download_models.py --group core
  python download_models.py --group all
"""

import sys
import os
import argparse
import traceback

print("=" * 60)
print("模型预下载脚本（使用系统默认缓存目录）")
print("=" * 60)
print()

# ========================
# 1️⃣ Torch / torchvision
# ========================
def download_torchvision_models():
    print("[core] 下载 torchvision 模型...")
    from torchvision import models

    models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

    # 显式下载 resnet18 权重，确保在指定 checkpoints 目录中存在
    _download_resnet18_checkpoint()

    print("✓ torchvision 模型下载完成\n")

# ========================
# 2️⃣ 额外权重直连下载工具
# ========================
def _download_2dfan4_checkpoint():
    """显式下载 2DFAN4 checkpoint 到 torch hub 默认目录."""
    url = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar"

    try:
        import torch
        from torch.hub import _get_torch_home, download_url_to_file
    except Exception:
        print("⚠️ 无法导入 torch 或 torch.hub，跳过 2DFAN4 模型直连下载")
        return

    torch_home = _get_torch_home()
    checkpoints_dir = os.path.join(torch_home, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    dst = os.path.join(checkpoints_dir, "2DFAN4-11f355bf06.pth.tar")
    if os.path.exists(dst):
        print(f"[fan] 2DFAN4 checkpoint 已存在：{dst}")
        return

    print(f"[fan] 直接下载 2DFAN4 checkpoint 到: {dst}")
    download_url_to_file(url, dst, progress=True)
    print("[fan] 2DFAN4 checkpoint 下载完成")


def _download_resnet18_checkpoint():
    """显式下载 resnet18 checkpoint 到 torch hub 默认目录."""
    url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"

    try:
        import torch
        from torch.hub import _get_torch_home, download_url_to_file
    except Exception:
        print("⚠️ 无法导入 torch 或 torch.hub，跳过 resnet18 模型直连下载")
        return

    torch_home = _get_torch_home()
    checkpoints_dir = os.path.join(torch_home, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    dst = os.path.join(checkpoints_dir, "resnet18-5c106cde.pth")
    if os.path.exists(dst):
        print(f"[core] resnet18 checkpoint 已存在：{dst}")
        return

    print(f"[core] 直接下载 resnet18 checkpoint 到: {dst}")
    download_url_to_file(url, dst, progress=True)
    print("[core] resnet18 checkpoint 下载完成")


# ========================
# 3️⃣ Face Alignment (FAN)
# ========================
def download_face_alignment_models():
    print("[2/4] 下载 face-alignment / FAN 模型...")
    import face_alignment

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu"
    )
    del fa

    # 显式下载 2DFAN4 checkpoint，避免运行时再次联网
    _download_2dfan4_checkpoint()

    print("✓ face-alignment 模型下载完成\n")

# ========================
# 4️⃣ HuggingFace chatterbox
# ========================
def download_huggingface_models():
    print("[core] 下载 HuggingFace 模型: ResembleAI/chatterbox...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        local_dir_use_symlinks=False
    )

    print("✓ HuggingFace chatterbox 下载完成\n")

# ========================
# 5️⃣ Whisper
# ========================
def download_whisper_models():
    print("[core] 下载 Whisper 模型 (base)...")
    import whisper

    whisper.load_model("base")

    print("✓ Whisper base 下载完成\n")

# ========================
# 主入口
# ========================
def main():
    parser = argparse.ArgumentParser(
        description="模型预下载脚本（支持按组下载）"
    )
    parser.add_argument(
        "--group",
        choices=["fan", "core", "all"],
        default="all",
        help="选择要下载的模型组"
    )

    args = parser.parse_args()

    try:
        if args.group in ("core", "all"):
            download_torchvision_models()
            download_huggingface_models()
            download_whisper_models()

        if args.group in ("fan", "all"):
            download_face_alignment_models()

    except Exception:
        print("\n❌ 模型下载过程中发生错误：")
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("🎉 模型下载完成！")
    print(f"下载组: {args.group}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()