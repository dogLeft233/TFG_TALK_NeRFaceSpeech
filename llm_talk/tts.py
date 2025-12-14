import torch
import io
import base64
import logging
from typing import Optional, Dict, Any
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import soundfile as sf
import numpy as np
import webrtcvad
import re
import math

# 配置日志
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TTS_MODEL = None  # 延迟加载


CLIP_DB = 1000 #句尾静音裁剪阈值

#---------------------------------------------------------------------

class TTSError(Exception):
    """自定义TTS异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)
        
#---------------------------------------------------------------------

def load_tts_model():
    """延迟加载TTS模型"""
    global TTS_MODEL
    if TTS_MODEL is None:
        try:
            logger.info(f"正在加载TTS模型到设备: {DEVICE}")
            TTS_MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            logger.info("TTS模型加载成功")
        except Exception as e:
            error_msg = f"TTS模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise TTSError(error_msg, "MODEL_LOAD_ERROR", e)
    return TTS_MODEL

def unload_tts_model():
    """
    从内存中释放TTS模型
    
    Returns:
        bool: 是否成功释放
    """
    global TTS_MODEL
    try:
        if TTS_MODEL is not None:
            logger.info("正在释放TTS模型...")
            
            # 如果模型有清理方法，调用它
            if hasattr(TTS_MODEL, 'cleanup'):
                TTS_MODEL.cleanup()
            elif hasattr(TTS_MODEL, 'close'):
                TTS_MODEL.close()
            
            # 删除模型引用
            del TTS_MODEL
            TTS_MODEL = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 如果使用CUDA，清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("TTS模型已成功释放")
            return True
        else:
            logger.info("TTS模型未加载，无需释放")
            return True
            
    except Exception as e:
        error_msg = f"TTS模型释放失败: {str(e)}"
        logger.error(error_msg)
        # 即使释放失败，也尝试重置全局变量
        TTS_MODEL = None
        return False

def reload_tts_model():
    """
    重新加载TTS模型（先释放再加载）
    
    Returns:
        bool: 是否成功重新加载
    """
    try:
        logger.info("开始重新加载TTS模型...")
        
        # 先释放现有模型
        unload_success = unload_tts_model()
        if not unload_success:
            logger.warning("模型释放失败，但继续尝试重新加载")
        
        # 重新加载模型
        load_tts_model()
        
        logger.info("TTS模型重新加载成功")
        return True
        
    except Exception as e:
        error_msg = f"TTS模型重新加载失败: {str(e)}"
        logger.error(error_msg)
        raise TTSError(error_msg, "MODEL_RELOAD_ERROR", e)

def get_model_status():
    """
    获取TTS模型状态
    
    Returns:
        dict: 模型状态信息
    """
    global TTS_MODEL
    
    status = {
        'loaded': TTS_MODEL is not None,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'memory_info': {}
    }
    
    if TTS_MODEL is not None:
        status['model_type'] = type(TTS_MODEL).__name__
        
        # 获取内存使用情况
        if torch.cuda.is_available():
            status['memory_info'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
    
    return status

#---------------------------------------------------------------------

def wav_float_to_int16(wav: np.ndarray) -> np.ndarray:
    # 将 float 波形 (-1..1) 转为 int16
    wav_i16 = (wav * 32767.0).astype(np.int16)
    return wav_i16

def vad_trim(wav: np.ndarray, sample_rate: int, aggressiveness: int = 2,
             frame_ms: int = 30, padding_ms: int = 300) -> np.ndarray:
    """
    使用 WebRTC VAD 裁剪尾部非语音（并保留前段）
    aggressiveness: 0-3，越高越严格（更少噪音误判为语音）
    frame_ms: 10/20/30 常用
    padding_ms: 保留尾部的额外毫秒（避免切断尾音）
    """
    if wav.size == 0:
        return wav
    # ensure int16 PCM
    if np.issubdtype(wav.dtype, np.floating):
        wav_i16 = wav_float_to_int16(wav)
    else:
        wav_i16 = wav.astype(np.int16)

    vad = webrtcvad.Vad(aggressiveness)
    frame_bytes = int(sample_rate * (frame_ms / 1000.0))  # samples per frame
    byte_width = 2  # int16 -> 2 bytes
    step = frame_bytes
    n_frames = math.ceil(len(wav_i16) / step)

    is_speech = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * step
        end = min((i + 1) * step, len(wav_i16))
        frame = wav_i16[start:end]
        if len(frame) < frame_bytes:
            # pad
            frame = np.pad(frame, (0, frame_bytes - len(frame)), constant_values=0)
        raw_bytes = frame.tobytes()
        try:
            is_speech[i] = vad.is_speech(raw_bytes, sample_rate)
        except Exception:
            is_speech[i] = False

    non_speech_idx = np.where(is_speech)[0]
    if non_speech_idx.size == 0:
        # 无检测到语音，返回原波形
        return wav
    last_speech_frame = non_speech_idx[-1]
    # 计算要保留到的样本
    keep_samples = min(len(wav_i16), int((last_speech_frame + 1) * step + (padding_ms / 1000.0) * sample_rate))
    wav_trimmed = wav[:keep_samples]  # 原 wav 是 float 或 int16，都可以直接剪切（保持原 dtype）
    # 小淡出避免突变
    fade_len = int(0.03 * sample_rate)
    if fade_len*2 < len(wav_trimmed):
        window = np.linspace(1.0, 0.0, fade_len)
        wav_trimmed[-fade_len:] = wav_trimmed[-fade_len:] * window
    return wav_trimmed

#---------------------------------------------------------------------

def convert_text_to_wav_chatterbox(text: str,
                                 language_id: str = 'zh',
                                 audio_prompt_path: Optional[str] = None,
                                 exaggeration: float = 0.5,
                                 cfg_weight: float = 0.3,
                                 temperature: float = 0.7,
                                 repetition_penalty: float = 1.5,
                                 min_p: float = 0.01,
                                 top_p: float = 0.9
                                 ) -> Dict[str, Any]:
    """
    将文本转换为WAV音频数据，返回内存中的音频数据
    
    Args:
        text: 要转换的文本
        language_id: 语言ID (默认'zh'中文)
        audio_prompt_path: 音频提示文件路径
        exaggeration: 夸张程度
        cfg_weight: CFG权重
        temperature: 温度参数（建议0.6-0.8，较低的值可以减少随机性和重复）
        repetition_penalty: 重复惩罚（默认2.0，建议1.5-2.5，值越高越能防止重复token导致生成过早停止）
        min_p: 最小概率
        top_p: 顶部概率（建议0.85-0.95）
        sample_rate: 采样率
    
    Returns:
        Dict[str, Any]: 包含音频数据的字典
        {
            'success': bool,
            'data': {
                'wav_data': bytes,  # WAV文件的二进制数据
                'base64_data': str,  # Base64编码的音频数据
                'sample_rate': int,  # 采样率
                'duration': float,  # 音频时长（秒）
                'text': str,  # 原始文本
                'audio_info': dict  # 音频信息
            },
            'error': None or dict
        }
    """
    try:
        # 输入验证
        if not text or not isinstance(text, str):
            raise TTSError("文本不能为空且必须是字符串", "INVALID_INPUT")
        
        if len(text.strip()) == 0:
            raise TTSError("文本内容不能为空", "EMPTY_TEXT")
        
        if len(text) > 1000:  # 限制文本长度
            raise TTSError("文本长度超过限制（1000字符）", "TEXT_TOO_LONG")
    
        # text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # # # text = text.translate(str.maketrans('，。、？！', ',.,?!'))
        # text = re.sub(r'[，。、？！]', ' ', text)
        
        logger.info(f"开始TTS转换，文本长度: {len(text)}")
        logger.debug(f"exaggeration:{exaggeration}")
        logger.debug(f"cfg_weight:{cfg_weight}")
        logger.debug(f"temperature:{temperature}")
        logger.debug(f"repetition_penalty:{repetition_penalty}")
        logger.debug(f"min_p:{min_p}")
        logger.debug(f"top_p:{top_p}")
        logger.debug(f"text:{text}")
        
        # 加载模型
        model = load_tts_model()
        
        # 生成音频
        # 注意：如果遇到token重复导致生成过早停止的问题，可以尝试：
        # 1. 增加repetition_penalty（当前默认2.0）
        # 2. 降低temperature（减少随机性）
        # 3. 调整top_p参数
        logger.info("开始生成音频...")
        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p
        )
        
        sample_rate = model.sr
        
        # 验证生成的音频
        if wav is None:
            raise TTSError("TTS模型返回空音频", "EMPTY_AUDIO")
        
        # 转换为numpy数组
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = np.array(wav)
        
        # 确保音频数据是1D数组
        if wav_np.ndim > 1:
            wav_np = wav_np.flatten()
            
        # 如果是 float，确保范围合理（-1..1）
        if np.issubdtype(wav_np.dtype, np.floating):
            max_abs = np.max(np.abs(wav_np)) if wav_np.size else 0.0
            if max_abs > 1.0:
                wav_np = wav_np / max_abs

        #trim
        try:
            wav_np = vad_trim(wav_np, sample_rate, aggressiveness=2)
        except Exception:
            # 使用简单能量阈值裁剪
            energy = wav_np**2
            thresh = 1e-6  # 根据你的数据调整
            non_silent = np.where(energy > thresh)[0]
            if non_silent.size:
                wav_np = wav_np[:non_silent[-1]+1]
            else:
                wav_np = wav_np[:0]
        
        # 计算音频时长
        duration = len(wav_np) / sample_rate
        
        # 创建内存中的WAV文件
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wav_np, sample_rate, format='WAV')
        wav_data = wav_buffer.getvalue()
        wav_buffer.close()
        
        # 转换为Base64编码
        base64_data = base64.b64encode(wav_data).decode('utf-8')
        
        # 音频信息
        audio_info = {
            'sample_rate': sample_rate,
            'duration': duration,
            'channels': 1,
            'samples': len(wav_np),
            'format': 'WAV',
            'bit_depth': 16
        }
        
        logger.info(f"TTS转换成功，音频时长: {duration:.2f}秒")
        
        return {
            'success': True,
            'data': {
                'wav_data': wav_data,
                'base64_data': base64_data,
                'sample_rate': sample_rate,
                'duration': duration,
                'text': text,
                'audio_info': audio_info
            },
            'error': None
        }
        
    except TTSError:
        # 重新抛出TTS错误
        raise
    except Exception as e:
        # 捕获其他异常并转换为TTS错误
        error_msg = f"TTS转换时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise TTSError(error_msg, "TTS_CONVERSION_ERROR", e)

#---------------------------------------------------------------------
    
def get_tts_response_api(text: str, **kwargs) -> Dict[str, Any]:
    """
    为前端提供的TTS API接口，返回标准化的响应格式
    
    Args:
        text: 要转换的文本
        **kwargs: 其他TTS参数
    
    Returns:
        Dict[str, Any]: 标准化的API响应
    """
    try:
        result = convert_text_to_wav_chatterbox(text, **kwargs)
        return result
    except TTSError as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'TTSError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f"未知错误: {str(e)}",
                'type': 'Exception'
            }
        }

def save_wav_to_file(wav_data: bytes, file_path: str) -> bool:
    """
    将WAV数据保存到文件
    
    Args:
        wav_data: WAV二进制数据
        file_path: 保存路径
    
    Returns:
        bool: 是否保存成功
    """
    try:
        with open(file_path, 'wb') as f:
            f.write(wav_data)
        logger.info(f"WAV文件已保存到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存WAV文件失败: {str(e)}")
        return False

def manage_tts_model(action: str) -> Dict[str, Any]:
    """
    管理TTS模型的API接口
    
    Args:
        action: 操作类型 ('load', 'unload', 'reload', 'status')
    
    Returns:
        Dict[str, Any]: 操作结果
    """
    try:
        if action == 'load':
            load_tts_model()
            return {
                'success': True,
                'message': 'TTS模型加载成功',
                'action': 'load'
            }
            
        elif action == 'unload':
            success = unload_tts_model()
            return {
                'success': success,
                'message': 'TTS模型释放成功' if success else 'TTS模型释放失败',
                'action': 'unload'
            }
            
        elif action == 'reload':
            reload_tts_model()
            return {
                'success': True,
                'message': 'TTS模型重新加载成功',
                'action': 'reload'
            }
            
        elif action == 'status':
            status = get_model_status()
            return {
                'success': True,
                'data': status,
                'action': 'status'
            }
            
        else:
            return {
                'success': False,
                'error': {
                    'code': 'INVALID_ACTION',
                    'message': f'无效的操作: {action}。支持的操作: load, unload, reload, status',
                    'type': 'ValueError'
                }
            }
            
    except TTSError as e:
        return {
            'success': False,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'TTSError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f'未知错误: {str(e)}',
                'type': 'Exception'
            }
        }
    
#---------------------------------------------------------------------

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试用例
    test_text = "寿限无寿限无ウンコ投げ机一昨日の新ちゃんのパンツ新八の人生バルムンク=フェザリオンアイザック=シュナイダー三分の一の纯情な感情の残った三分の二はさかむけが気になる感情裏切りは仆の名前をしっているようでしらないのを仆はしっている留守スルメめだかかずのここえだめめだかこのめだかはさっきと违う奴だから池乃めだかの方だからラー油ゆうていみやおうきむこうぺぺぺぺぺぺぺぺぺぺぺぺ（おあとがよろしいようでこれにておしまい）ビチグソ丸。"
    
    language_id= 'ja'
    
    print("=== TTS测试开始 ===")
    try:
        # 测试1: 模型状态检查
        print("\n--- 测试1: 模型状态检查 ---")
        status_result = manage_tts_model('status')
        if status_result['success']:
            print(f"✅ 模型状态: {status_result['data']}")
        else:
            print(f"❌ 状态检查失败: {status_result['error']['message']}")
        
        # 测试2: TTS转换
        print("\n--- 测试2: TTS转换 ---")
        result = get_tts_response_api(test_text, language_id=language_id, audio_prompt_path="../assets/Aerith.mp3")
        
        if result['success']:
            print(f"✅ TTS转换成功")
            print(f"📝 原始文本: {result['data']['text']}")
            print(f"🎵 音频时长: {result['data']['duration']:.2f}秒")
            print(f"📊 采样率: {result['data']['sample_rate']}Hz")
            print(f"📁 音频信息: {result['data']['audio_info']}")
            print(f"💾 WAV数据大小: {len(result['data']['wav_data'])} bytes")
            print(f"🔤 Base64数据长度: {len(result['data']['base64_data'])} 字符")

            save_wav_to_file(result['data']['wav_data'], 'test_output.wav')
            print("💾 音频已保存到 test_output.wav")
            
        else:
            print(f"❌ TTS转换失败: {result['error']['message']}")
        
        # 测试3: 模型释放
        print("\n--- 测试3: 模型释放 ---")
        unload_result = manage_tts_model('unload')
        if unload_result['success']:
            print(f"✅ {unload_result['message']}")
        else:
            print(f"❌ 模型释放失败: {unload_result['error']['message']}")
        
        # 测试4: 模型重新加载
        print("\n--- 测试4: 模型重新加载 ---")
        reload_result = manage_tts_model('reload')
        if reload_result['success']:
            print(f"✅ {reload_result['message']}")
        else:
            print(f"❌ 模型重新加载失败: {reload_result['error']['message']}")
        
        # 测试5: 重新检查状态
        print("\n--- 测试5: 重新检查状态 ---")
        final_status = manage_tts_model('status')
        if final_status['success']:
            print(f"✅ 最终模型状态: {final_status['data']}")
        else:
            print(f"❌ 状态检查失败: {final_status['error']['message']}")
            
    except Exception as e:
        print(f"💥 测试过程中发生异常: {str(e)}")
    
    print("\n=== TTS测试结束 ===")