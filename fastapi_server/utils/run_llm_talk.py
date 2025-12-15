import subprocess
import os
from pathlib import Path

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT,
    LLM_CONDA_PYTHON,
    LLM_TALK_SCRIPT as TALK_SCRIPT,
    get_character_audio_prompt,
)

def generate_audio(text: str, output_path: str, character: str) -> bool:
    """生成音频文件"""
    try:
        audio_prompt = get_character_audio_prompt(character)
    except ValueError as e:
        print(f"错误: {e}")
        return False

    env = os.environ.copy()
    # 确保可以以包形式调用 llm_talk（支持相对导入）
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        str(LLM_CONDA_PYTHON),
        "-m",
        "llm_talk.talk",
        f"--input_text={text}",
        f"--audio_prompt_path={audio_prompt}",
        f"--output_path={output_path}",
    ]

    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))
        return True
    except Exception as e:
        print("LLM 语音生成错误：", e)
        return False
