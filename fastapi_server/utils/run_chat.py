"""
聊天对话工具模块
使用 subprocess 方式调用 llm_talk（与可运行版本一致）
"""
import subprocess
import os
import json
from pathlib import Path

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT,
    LLM_CONDA_PYTHON,
    NERF_CODE_DIR,
    get_character_audio_prompt,
)

# API 桥接脚本路径
API_BRIDGE_SCRIPT = Path(__file__).parent / "llm_talk_api_bridge.py"

# 使用项目根目录作为工作目录和 PYTHONPATH
LLM_WORKDIR = PROJECT_ROOT  # 使用项目根目录


def _call_llm_api_bridge(mode: str, user_input: str, character: str = "ayanami", 
                         enable_audio: bool = True, audio_prompt_path: str = None) -> dict:
    """
    通过 subprocess 调用 llm_talk API 桥接脚本
    
    Args:
        mode: "talk" 或 "llm_only"
        user_input: 用户输入
        character: 角色名称
        enable_audio: 是否生成音频
        audio_prompt_path: 音频提示路径
    
    Returns:
        dict: API 响应结果
    """
    if not LLM_CONDA_PYTHON.exists():
        return {
            "success": False,
            "error": f"LLM conda 环境不存在: {LLM_CONDA_PYTHON}"
        }
    
    if not API_BRIDGE_SCRIPT.exists():
        return {
            "success": False,
            "error": f"API 桥接脚本不存在: {API_BRIDGE_SCRIPT}"
        }
    
    # 构建命令
    cmd = [
        str(LLM_CONDA_PYTHON),
        str(API_BRIDGE_SCRIPT),
        "--mode", mode,
        "--user_input", user_input,
        "--character", character,
        "--enable_audio", str(enable_audio),
    ]
    
    if audio_prompt_path:
        cmd.extend(["--audio_prompt_path", audio_prompt_path])
    
    # 设置环境变量（与可运行版本一致）
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LLM_WORKDIR)
    
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            cwd=str(LLM_WORKDIR),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # 解析 JSON 输出
        try:
            output = result.stdout.strip()
            # 查找 JSON 部分（可能有一些日志输出）
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    "success": False,
                    "error": f"无法解析输出为 JSON: {output[:200]}"
                }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON 解析失败: {str(e)}, 输出: {result.stdout[:200]}"
            }
            
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else e.stdout if hasattr(e, 'stdout') else str(e)
        return {
            "success": False,
            "error": f"subprocess 调用失败: {error_output[:500]}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"调用 llm_talk API 时发生错误: {str(e)}"
        }


def chat_with_llm(user_input: str, character: str = "ayanami", enable_audio: bool = True) -> dict:
    """
    与LLM进行对话，返回文本回答和音频
    
    Args:
        user_input: 用户输入的文本
        character: 角色名称（用于TTS音色选择）
        enable_audio: 是否生成音频
    
    Returns:
        dict: 包含 success, data, error 的响应
    """
    try:
        # 获取角色对应的音频提示路径（用于TTS音色）
        try:
            audio_prompt_path = str(get_character_audio_prompt(character))
        except ValueError:
            # 如果配置中没有，使用默认路径
            audio_prompt_path = None
        
        # 通过 subprocess 调用 API
        result = _call_llm_api_bridge(
            mode="talk",
            user_input=user_input,
            character=character,
            enable_audio=enable_audio,
            audio_prompt_path=audio_prompt_path
        )
        
        if result.get('success'):
            # 提取需要的数据
            data = result.get('data', {})
            response = {
                "success": True,
                "data": {
                    "user_input": data.get('user_input', user_input),
                    "llm_answer": data.get('llm_answer', ''),
                    "audio_base64": None,
                    "audio_duration": 0
                }
            }
            
            # 如果启用了音频且生成成功，添加音频数据
            if enable_audio and data.get('combined_audio') and data['combined_audio'].get('success'):
                combined_audio = data['combined_audio']
                response["data"]["audio_base64"] = combined_audio.get('combined_base64_data')
                response["data"]["audio_duration"] = combined_audio.get('total_duration', 0)
            
            return response
        else:
            error_info = result.get('error', {})
            error_message = error_info.get('message', '未知错误') if isinstance(error_info, dict) else str(error_info)
            return {
                "success": False,
                "error": error_message
            }
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"对话处理失败: {error_detail}")
        return {
            "success": False,
            "error": f"对话处理失败: {str(e)}"
        }


def get_llm_only(user_input: str) -> dict:
    """
    仅获取LLM回答，不生成音频（用于快速响应）
    
    Args:
        user_input: 用户输入的文本
    
    Returns:
        dict: 包含 success, data, error 的响应
    """
    try:
        result = _call_llm_api_bridge(
            mode="llm_only",
            user_input=user_input
        )
        return result
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"LLM调用失败: {error_detail}")
        return {
            "success": False,
            "error": f"LLM调用失败: {str(e)}"
        }
