#!/usr/bin/env python3
"""
一键启动脚本 - NeRFFaceSpeech 开发者模式
运行方式: python start.py
功能: 自动设置环境变量，启动后端和前端服务器，打开浏览器
"""
import subprocess
import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent.resolve()

# 导入配置以获取API环境的Python路径
sys.path.insert(0, str(CURRENT_DIR))
API_CONDA_PYTHON = None
try:
    from config import API_CONDA_PYTHON
except (ImportError, AttributeError):
    # 如果导入失败或属性不存在，尝试从项目路径构造
    PROJECT_ROOT = CURRENT_DIR.parent
    api_env_python = PROJECT_ROOT / "environment" / "api" / "bin" / "python"
    if api_env_python.exists():
        API_CONDA_PYTHON = api_env_python
    else:
        API_CONDA_PYTHON = Path(sys.executable)

def set_environment_variables():
    """设置必要的环境变量"""
    print("=" * 60)
    print("设置环境变量...")
    print("=" * 60)
    
    env_vars = {
        'PIP_INDEX_URL': 'https://pypi.tuna.tsinghua.edu.cn/simple',
        'TORCH_HOME': '/root/autodl-tmp/weights',
        'HF_ENDPOINT': 'https://hf-mirror.com',
        'HF_HOME': '/root/autodl-tmp/Hugging_Face'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    
    print("环境变量设置完成\n")

def check_environment(env_python: Path, env_name: str) -> tuple[bool, str]:
    """
    检查环境是否可用
    
    Returns:
        (是否可用, 错误信息)
    """
    if not env_python.exists():
        return False, f"Python可执行文件不存在: {env_python}"
    
    # 检查uvicorn是否可用
    try:
        result = subprocess.run(
            [str(env_python), "-m", "uvicorn", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, f"uvicorn不可用: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "检查uvicorn时超时"
    except Exception as e:
        return False, f"检查环境时出错: {str(e)}"
    
    return True, ""

def start_backend_server():
    """启动后端服务器（FastAPI）"""
    print("=" * 60)
    print("启动后端服务器...")
    print("=" * 60)
    
    # 优先使用API环境的Python
    api_python = None
    env_error = None
    
    if API_CONDA_PYTHON:
        print(f"检查API环境: {API_CONDA_PYTHON}")
        is_available, error_msg = check_environment(API_CONDA_PYTHON, "API")
        if is_available:
            api_python = API_CONDA_PYTHON
            print(f"✅ API环境可用: {api_python}")
        else:
            env_error = error_msg
            print(f"❌ API环境检查失败: {error_msg}")
            print(f"   环境路径: {API_CONDA_PYTHON}")
            if API_CONDA_PYTHON.parent.parent.exists():
                print(f"   环境目录存在: {API_CONDA_PYTHON.parent.parent}")
            else:
                print(f"   环境目录不存在: {API_CONDA_PYTHON.parent.parent}")
    
    # 如果API环境不可用，尝试使用当前Python
    if api_python is None:
        current_python = Path(sys.executable)
        print(f"\n检查当前Python环境: {current_python}")
        is_available, error_msg = check_environment(current_python, "当前")
        if is_available:
            api_python = current_python
            print(f"✅ 当前Python环境可用: {api_python}")
        else:
            print(f"❌ 当前Python环境检查失败: {error_msg}")
            print("\n" + "=" * 60)
            print("❌ 错误: 无法启动后端服务器")
            print("=" * 60)
            if env_error:
                print(f"\nAPI环境错误: {env_error}")
            print(f"当前Python错误: {error_msg}")
            print("\n解决方案:")
            print("1. 确保API环境存在: PROJECT_ROOT/environment/api")
            print("2. 或在当前环境安装uvicorn: pip install uvicorn fastapi")
            print("=" * 60 + "\n")
            return None
    
    backend_cmd = [
        str(api_python), "-m", "uvicorn",
        "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # 设置工作目录
    env = os.environ.copy()
    
    # 在后台启动后端服务器
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(CURRENT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 实时输出后端日志
    def log_backend_output():
        for line in backend_process.stdout:
            print(f"[后端] {line.rstrip()}")
    
    log_thread = threading.Thread(target=log_backend_output, daemon=True)
    log_thread.start()
    
    # 等待后端启动
    print("等待后端服务器启动...")
    time.sleep(3)
    
    # 检查后端是否启动成功
    try:
        import requests
        response = requests.get("http://localhost:8000/docs", timeout=2)
        if response.status_code == 200:
            print("✅ 后端服务器启动成功！")
            print("   后端地址: http://localhost:8000/")
            print("   API文档: http://localhost:8000/docs\n")
        else:
            print("⚠️  后端服务器可能未正常启动，请检查日志\n")
    except Exception as e:
        print(f"⚠️  后端服务器检查失败: {e}")
        print("   服务器可能仍在启动中，请稍候...\n")
    
    return backend_process

def start_frontend_server():
    """启动前端服务器"""
    print("=" * 60)
    print("启动前端服务器...")
    print("=" * 60)
    
    frontend_cmd = [sys.executable, "simple_web.py"]
    
    # 设置工作目录
    env = os.environ.copy()
    
    # 在后台启动前端服务器
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=str(CURRENT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 实时输出前端日志
    def log_frontend_output():
        for line in frontend_process.stdout:
            print(f"[前端] {line.rstrip()}")
    
    log_thread = threading.Thread(target=log_frontend_output, daemon=True)
    log_thread.start()
    
    # 等待前端启动
    print("等待前端服务器启动...")
    time.sleep(2)
    
    print("✅ 前端服务器启动成功！")
    print("   前端地址: http://localhost:7860/\n")
    
    return frontend_process

def open_browser():
    """打开浏览器显示选择页面"""
    url = "http://localhost:7860/start.html"
    print("=" * 60)
    print("打开浏览器...")
    print(f"访问地址: {url}")
    print("=" * 60)
    
    # 等待服务器完全启动
    time.sleep(1)
    
    try:
        webbrowser.open(url)
        print(f"✅ 浏览器已打开: {url}\n")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print(f"   请手动访问: {url}\n")

def enable_network_acceleration():
    """开启学术加速"""
    print("=" * 60)
    print("开启学术加速...")
    print("=" * 60)
    
    network_turbo_file = Path("/etc/network_turbo")
    if not network_turbo_file.exists():
        print("⚠️  学术加速配置文件不存在: /etc/network_turbo")
        print("   （不影响使用，但网络可能较慢）\n")
        return
    
    try:
        # 使用bash执行source命令
        # 注意：source命令在子进程中执行，环境变量不会影响当前进程
        # 如果需要环境变量生效，需要读取文件内容并手动设置
        result = subprocess.run(
            ["bash", "-c", "source /etc/network_turbo && echo '学术加速已开启'"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ 学术加速已开启")
            if result.stdout.strip():
                print(f"   {result.stdout.strip()}\n")
            else:
                print("\n")
        else:
            print("⚠️  学术加速开启失败（但不影响使用）")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}\n")
            else:
                print("\n")
    except subprocess.TimeoutExpired:
        print("⚠️  开启学术加速超时（但不影响使用）\n")
    except Exception as e:
        print(f"⚠️  开启学术加速时出错: {e}（但不影响使用）\n")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NeRFFaceSpeech 开发者模式启动")
    print("=" * 60 + "\n")
    
    try:
        # 0. 开启学术加速
        enable_network_acceleration()
        
        # 1. 设置环境变量
        set_environment_variables()
        
        # 2. 启动后端服务器
        backend_process = start_backend_server()
        if backend_process is None:
            # 后端启动失败，退出
            print("启动失败，程序退出")
            return
        
        # 3. 启动前端服务器
        frontend_process = start_frontend_server()
        
        # 4. 打开浏览器
        open_browser()
        
        # 5. 保持运行
        print("=" * 60)
        print("服务器运行中...")
        print("按 Ctrl+C 停止所有服务器")
        print("=" * 60 + "\n")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                if backend_process.poll() is not None:
                    print("\n❌ 后端服务器意外停止")
                    break
                if frontend_process.poll() is not None:
                    print("\n❌ 前端服务器意外停止")
                    break
        except KeyboardInterrupt:
            print("\n\n正在停止服务器...")
            
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理进程
        try:
            if 'backend_process' in locals():
                backend_process.terminate()
                backend_process.wait(timeout=5)
        except:
            pass
        
        try:
            if 'frontend_process' in locals():
                frontend_process.terminate()
                frontend_process.wait(timeout=5)
        except:
            pass
        
        print("✅ 所有服务器已停止")
        print("感谢使用 NeRFFaceSpeech！")

if __name__ == "__main__":
    main()

