"""
配置管理模块
使用相对路径，基于项目根目录
"""
import os
from pathlib import Path

# 获取项目根目录（fastapi_server 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ==================== 路径配置 ====================

# NeRFFaceSpeech 代码目录
NERF_CODE_DIR = PROJECT_ROOT / "NeRFFaceSpeech_Code"

# 输出目录
OUTPUT_VIDEO_DIR = NERF_CODE_DIR / "outputs" / "video"
OUTPUT_AUDIO_DIR = NERF_CODE_DIR / "outputs" / "audio"

# 模型目录
MODEL_DIR = NERF_CODE_DIR / "pretrained_networks"

# ==================== Conda 环境配置 ====================

# LLM Talk 环境（在 environment 文件夹中）
LLM_CONDA_ENV = PROJECT_ROOT / "environment" / "llm_talk"
LLM_CONDA_PYTHON = LLM_CONDA_ENV / "bin" / "python"

# NeRF 环境（在 environment 文件夹中）
NERF_CONDA_ENV = PROJECT_ROOT / "environment" / "nerffacespeech"
NERF_CONDA_PYTHON = NERF_CONDA_ENV / "bin" / "python"

# ==================== 脚本路径配置 ====================

# LLM Talk 脚本
LLM_TALK_SCRIPT = PROJECT_ROOT / "llm_talk" / "talk.py"

# NeRF 脚本
NERF_SCRIPT = NERF_CODE_DIR / "StyleNeRF" / "main_NeRFFaceSpeech_audio_driven_w_given_poses.py"

# ==================== 资源路径配置 ====================

# 角色音频提示文件
CHARACTER_AUDIO_PROMPTS = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "绫波丽.wav",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" /"Aerith.mp3",
}

# 角色测试图片
CHARACTER_TEST_IMAGES = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "ayanami.png",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" / "Aerith.jpg",
}

# ==================== 辅助函数 ====================

def ensure_dirs():
    """确保必要的目录存在"""
    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_character_audio_prompt(character: str) -> Path:
    """获取角色的音频提示文件路径"""
    prompt = CHARACTER_AUDIO_PROMPTS.get(character)
    if prompt is None:
        raise ValueError(f"未知角色: {character}")
    return prompt

def get_character_test_image(character: str) -> Path:
    """获取角色的测试图片路径"""
    img = CHARACTER_TEST_IMAGES.get(character)
    if img is None:
        raise ValueError(f"未知角色: {character}")
    return img

# 启动时确保目录存在
ensure_dirs()
