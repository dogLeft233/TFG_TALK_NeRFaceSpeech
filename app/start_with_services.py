import subprocess
import sys
import time
import signal
import atexit
import os
from pathlib import Path

from shared.config import LLM_CONDA_PYTHON, LLM_CONDA_ENV


"""
一键启动脚本：先启动 TTS 服务和 ASR 服务，再启动后端。

环境约定：
- TTS / ASR 服务使用 `environment/llm_talk` 对应的 Conda 环境（通过 `LLM_CONDA_PYTHON` / `LLM_CONDA_ENV` 自动解析）
- 后端使用当前运行本脚本的 Python（通常是 `environment/api` 对应环境）

使用方法（在仓库根目录执行）：

    cd app
    python start_with_services.py

默认端口：
  - TTS: 8001  -> services.tts_service:app
  - ASR: 8002  -> services.asr_service:app
  - 后端: 8000 -> backend.main:app
"""

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"

TTS_HOST = "0.0.0.0"
TTS_PORT = 8001

ASR_HOST = "0.0.0.0"
ASR_PORT = 8002

BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000

processes = []


def cleanup(*_):
    """退出时清理所有子进程"""
    for p in processes:
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    # 再等一会儿，强制杀掉未退出的进程
    time.sleep(1)
    for p in processes:
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def start_tts():
    """启动 TTS 服务（services.tts_service:app，使用 llm_talk 环境）"""

    # 优先使用 llm_talk 环境中的 Python
    python_path = LLM_CONDA_PYTHON if LLM_CONDA_PYTHON and LLM_CONDA_PYTHON.exists() else Path(sys.executable)

    cmd = [
        str(python_path),
        "-m",
        "uvicorn",
        "services.tts_service:app",
        "--host",
        TTS_HOST,
        "--port",
        str(TTS_PORT),
    ]
    print(f"[start] 启动 TTS 服务 (Python: {python_path}): {' '.join(cmd)}")

    # 确保使用 llm_talk 环境的 PATH（如果存在）
    env = os.environ.copy()
    if LLM_CONDA_ENV and LLM_CONDA_ENV.exists():
        llm_bin = LLM_CONDA_ENV / "bin"
        if llm_bin.exists():
            env["PATH"] = f"{llm_bin}{os.pathsep}{env.get('PATH','')}"

    proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env)
    processes.append(proc)
    # 等待几秒，让服务起来
    time.sleep(3)
    return proc


def start_asr():
    """启动 ASR 服务（services.asr_service:app，使用 llm_talk 环境）"""

    python_path = LLM_CONDA_PYTHON if LLM_CONDA_PYTHON and LLM_CONDA_PYTHON.exists() else Path(sys.executable)

    cmd = [
        str(python_path),
        "-m",
        "uvicorn",
        "services.asr_service:app",
        "--host",
        ASR_HOST,
        "--port",
        str(ASR_PORT),
    ]
    print(f"[start] 启动 ASR 服务 (Python: {python_path}): {' '.join(cmd)}")

    env = os.environ.copy()
    if LLM_CONDA_ENV and LLM_CONDA_ENV.exists():
        llm_bin = LLM_CONDA_ENV / "bin"
        if llm_bin.exists():
            env["PATH"] = f"{llm_bin}{os.pathsep}{env.get('PATH','')}"

    proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env)
    processes.append(proc)
    # 等待几秒，让服务起来
    time.sleep(3)
    return proc


def start_backend():
    """启动后端（backend.main:app）"""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
    ]
    print(f"[start] 启动后端: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(APP_DIR))
    processes.append(proc)
    return proc


def main():
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 60)
    print(" 启动 TTS 服务 + ASR 服务 + 后端")
    print("=" * 60)

    # 确保在 app 目录下运行
    print(f"[start] 工作目录: {APP_DIR}")

    # 1. 启动 TTS
    tts_proc = start_tts()
    print(f"[start] TTS 服务 PID: {tts_proc.pid}")

    # 2. 启动 ASR
    asr_proc = start_asr()
    print(f"[start] ASR 服务 PID: {asr_proc.pid}")

    # 3. 启动后端
    backend_proc = start_backend()
    print(f"[start] 后端服务 PID: {backend_proc.pid}")

    print("\n所有服务已启动：")
    print(f"  TTS:   http://{TTS_HOST}:{TTS_PORT}")
    print(f"  ASR:   http://{ASR_HOST}:{ASR_PORT}")
    print(f"  后端:  http://{BACKEND_HOST}:{BACKEND_PORT}")
    print("\n按 Ctrl+C 停止所有服务")

    # 等待后端退出（Ctrl+C 会触发 cleanup）
    backend_proc.wait()


if __name__ == "__main__":
    main()


