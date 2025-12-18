import subprocess
import os
from pathlib import Path

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    NERF_SCRIPT,
    NERF_CODE_DIR as NERF_WORKDIR,
    MODEL_DIR,
    get_character_test_image
)

def generate_video(
    audio_path: str,
    character: str,
    output_path: str,
    model_name: str
) -> bool:
    """生成视频文件"""
    try:
        test_img = get_character_test_image(character)
    except ValueError as e:
        print(f"错误: {e}")
        return False

    # ---------- 模型路径安全拼接 ----------
    network_path = MODEL_DIR / model_name
    if not model_name.endswith(".pkl") or not network_path.exists():
        print(f"非法或不存在的模型：{network_path}")
        return False

    env = os.environ.copy()
    # 与可运行版本一致：使用 PATH 和 PYTHONPATH
    env["PATH"] = f"{NERF_CONDA_ENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(NERF_WORKDIR)

    cmd = [
        str(NERF_CONDA_PYTHON),
        str(NERF_SCRIPT),
        f"--outdir={output_path}",
        "--trunc=0.7",
        f"--network={network_path}",
        f"--test_data={audio_path}",
        f"--test_img={test_img}",
        "--motion_guide_img_folder=frames"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(NERF_WORKDIR), env=env)
        return True
    except Exception as e:
        print("NeRF 视频生成错误：", e)
        return False
