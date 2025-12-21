"""
ASR 服务客户端
通过 HTTP 调用独立的 ASR 服务，避免每次重新加载模型
"""
import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 导入配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import ASR_SERVICE_URL

logger = logging.getLogger(__name__)

# 请求超时设置（秒）
REQUEST_TIMEOUT = 300  # 5分钟，ASR识别可能需要较长时间

def check_asr_service_health() -> bool:
    """检查 ASR 服务是否可用"""
    try:
        response = requests.get(
            f"{ASR_SERVICE_URL}/health",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except Exception as e:
        logger.warning(f"ASR 服务健康检查失败: {e}")
        return False

def transcribe_audio_via_service(
    audio_input: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务进行语音识别
    
    Args:
        audio_input: 音频输入，可以是文件路径或Base64字符串
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    try:
        # 判断输入类型
        if os.path.exists(audio_input):
            # 文件路径
            request_data = {
                "audio_path": audio_input,
                "model_name": model_name,
                "language": language,
                "task": task
            }
            
            logger.info(f"[ASR Client] 调用 ASR 服务识别音频文件")
            logger.info(f"[ASR Client] 文件路径: {audio_input}")
            logger.info(f"[ASR Client] 模型: {model_name}")
            
            # 调用 ASR 服务的文件识别接口
            response = requests.post(
                f"{ASR_SERVICE_URL}/api/asr/transcribe_file",
                json=request_data,
                timeout=REQUEST_TIMEOUT
            )
        else:
            # Base64字符串
            request_data = {
                "audio_base64": audio_input,
                "model_name": model_name,
                "language": language,
                "task": task
            }
            
            logger.info(f"[ASR Client] 调用 ASR 服务识别Base64音频")
            logger.info(f"[ASR Client] Base64长度: {len(audio_input)}")
            logger.info(f"[ASR Client] 模型: {model_name}")
            
            # 调用 ASR 服务的Base64识别接口
            response = requests.post(
                f"{ASR_SERVICE_URL}/api/asr/transcribe",
                json=request_data,
                timeout=REQUEST_TIMEOUT
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                text = result.get("data", {}).get("text", "")
                detected_language = result.get("data", {}).get("language", "unknown")
                logger.info(f"[ASR Client] ✅ 识别成功")
                logger.info(f"[ASR Client] 识别文本: {text[:100]}...")
                logger.info(f"[ASR Client] 检测语言: {detected_language}")
                return True, text
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                logger.error(f"[ASR Client] ❌ 识别失败: {error_msg}")
                return False, None
        else:
            error_detail = response.text
            logger.error(f"[ASR Client] ❌ ASR 服务返回错误: {response.status_code}")
            logger.error(f"[ASR Client] 错误详情: {error_detail}")
            return False, None
            
    except requests.exceptions.Timeout:
        logger.error(f"[ASR Client] ❌ 请求超时（>{REQUEST_TIMEOUT}秒）")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error(f"[ASR Client] ❌ 无法连接到 ASR 服务: {ASR_SERVICE_URL}")
        logger.error(f"[ASR Client] 请确保 ASR 服务正在运行")
        return False, None
    except Exception as e:
        logger.error(f"[ASR Client] ❌ 调用 ASR 服务时发生错误: {e}")
        import traceback
        logger.error(f"[ASR Client] 错误详情: {traceback.format_exc()}")
        return False, None

def transcribe_audio_file_via_service(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务识别音频文件
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    return transcribe_audio_via_service(
        audio_input=audio_path,
        model_name=model_name,
        language=language,
        task=task
    )

def transcribe_base64_audio_via_service(
    audio_base64: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务识别Base64音频
    
    Args:
        audio_base64: Base64编码的音频数据
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    return transcribe_audio_via_service(
        audio_input=audio_base64,
        model_name=model_name,
        language=language,
        task=task
    )

